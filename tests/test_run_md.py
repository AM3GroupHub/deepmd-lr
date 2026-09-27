"""examples/07-multi-gpu-md/run_md.py, the production MD driver, end to end.

Short runs of the float64 fixture model through the script's `main()`, all on
one GPU; "two partitions" means that GPU twice. The fixture has random weights,
so these tests pin the plumbing rather than the physics:
- every ensemble runs and writes its files, NPH and full-cell NPT included;
- a restart reproduces an uninterrupted run;
- two partitions match one;
- unwrapped input is folded before it reaches the evaluator;
- --relax-cell relaxes the cell before the MD;
- bad input exits with code 1 and a message, and instability with code 2.
"""

import importlib.util
import json
import pathlib

import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from .test_joint_model import build, frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the MD kernels need a CUDA device"
)

_SPEC = importlib.util.spec_from_file_location(
    "run_md",
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "07-multi-gpu-md" / "run_md.py",
)
run_md = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_md)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """A loadable checkpoint of the fixture model, and a starting structure."""
    root = tmp_path_factory.mktemp("run_md")
    model = build()
    state = {f"model.Default.{k}": v for k, v in model.state_dict().items()}
    state["_extra_state"] = {"model_params": json.loads(model.model_def_script)}
    ckpt = root / "model.ckpt.pt"
    torch.save({"model": state}, ckpt)
    coord, atype, cell = frame(n_mol=16, box=14.0)
    type_map = list(model.get_type_map())
    atoms = Atoms([type_map[t] for t in atype[0].tolist()],
                  positions=coord[0].cpu().numpy(),
                  cell=cell[0].cpu().numpy(), pbc=True)
    xyz = root / "start.extxyz"
    ase.io.write(xyz, atoms)
    return ckpt, xyz


def run(inputs, out, *extra, structure=None):
    ckpt, xyz = inputs
    return run_md.main([
        "--model", str(ckpt), "--structure", str(structure or xyz),
        "--out", str(out), "--dt", "0.25", "--seed", "3",
        "--max-temperature", "0", *extra,
    ])


ENSEMBLES = {
    "nve": ["--ensemble", "nve"],
    "langevin": ["--ensemble", "nvt", "--thermostat", "langevin"],
    "nhc": ["--ensemble", "nvt", "--thermostat", "nhc"],
    "rescale": ["--ensemble", "nvt", "--thermostat", "rescale"],
    "npt": ["--ensemble", "npt"],
    "npt_full": ["--ensemble", "npt", "--barostat", "full"],
    "nph": ["--ensemble", "nph"],
}


@pytest.mark.parametrize("ensemble", list(ENSEMBLES))
def test_each_ensemble_runs_and_writes_its_files(inputs, tmp_path, ensemble):
    out = tmp_path / ensemble
    rc = run(inputs, out, *ENSEMBLES[ensemble], "--steps", "10",
             "--log-every", "5", "--traj-every", "5", "--checkpoint-every", "5")
    assert rc == 0
    log = np.loadtxt(out / "md.log")
    assert log[:, 0].tolist() == [0, 5, 10]
    assert np.isfinite(log).all() or np.isnan(log[0, -1])   # ms/step at step 0
    frames = ase.io.read(out / "traj.extxyz", ":")
    assert [f.info["step"] for f in frames] == [0, 5, 10]
    assert np.abs(ase.io.read(out / "final.extxyz").get_velocities()).max() > 0
    with np.load(out / "checkpoint.npz") as z:
        assert json.loads(str(z["meta"]))["nsteps"] == 10
    segment = json.loads((out / "run.json").read_text())["segments"][-1]
    assert segment["status"] == "finished"


@pytest.mark.parametrize("ensemble", ["langevin", "npt", "nph"])
def test_restart_reproduces_an_uninterrupted_run(inputs, tmp_path, ensemble):
    """Langevin replays its noise from the step number; NPT and NPH restore the
    cell and its velocity. Either way, 6 + 6 steps must equal 12."""
    common = [*ENSEMBLES[ensemble], "--log-every", "3", "--traj-every", "3",
              "--checkpoint-every", "6"]
    assert run(inputs, tmp_path / "a", *common, "--steps", "12") == 0
    assert run(inputs, tmp_path / "b", *common, "--steps", "6") == 0
    assert run(inputs, tmp_path / "b", *common, "--steps", "12", "--restart") == 0
    a = ase.io.read(tmp_path / "a" / "final.extxyz")
    b = ase.io.read(tmp_path / "b" / "final.extxyz")
    np.testing.assert_allclose(b.positions, a.positions, atol=1e-7)
    np.testing.assert_allclose(b.cell.array, a.cell.array, atol=1e-9)
    la = np.loadtxt(tmp_path / "a" / "md.log")
    lb = np.loadtxt(tmp_path / "b" / "md.log")
    assert la[:, 0].tolist() == lb[:, 0].tolist()      # no duplicated rows
    np.testing.assert_allclose(lb[:, 1:-1], la[:, 1:-1], atol=1e-5)


def test_logging_does_not_change_an_npt_trajectory(inputs, tmp_path):
    """A log row reads the pressure. That used to overwrite the pressure the
    next barostat half-step consumed, so logging every step changed the run."""
    common = [*ENSEMBLES["npt"], "--steps", "40", "--traj-every", "0",
              "--checkpoint-every", "40"]
    assert run(inputs, tmp_path / "quiet", *common, "--log-every", "40") == 0
    assert run(inputs, tmp_path / "loud", *common, "--log-every", "1") == 0
    with np.load(tmp_path / "quiet" / "checkpoint.npz") as q, \
            np.load(tmp_path / "loud" / "checkpoint.npz") as lo:
        np.testing.assert_allclose(lo["positions"], q["positions"], atol=1e-9)
        np.testing.assert_allclose(lo["cell"], q["cell"], atol=1e-11)
        np.testing.assert_allclose(lo["cell_v"], q["cell_v"], atol=1e-14)


def _step0(out):
    first = ase.io.read(out / "traj.extxyz", 0)
    return first.get_potential_energy(), first.get_forces()


def test_two_partitions_match_one(inputs, tmp_path):
    common = ["--ensemble", "nve", "--steps", "1", "--traj-every", "1",
              "--traj-forces"]
    assert run(inputs, tmp_path / "one", *common, "--devices", "cuda:0") == 0
    assert run(inputs, tmp_path / "two", *common,
               "--devices", "cuda:0,cuda:0") == 0
    e1, f1 = _step0(tmp_path / "one")
    e2, f2 = _step0(tmp_path / "two")
    assert e2 == pytest.approx(e1, abs=1e-8)
    np.testing.assert_allclose(f2, f1, atol=1e-8)


def test_unwrapped_input_is_folded(inputs, tmp_path):
    """Atoms several cell lengths outside the box are what an integrator
    produces after a few ps of diffusion. The fold makes them harmless."""
    _, xyz = inputs
    atoms = ase.io.read(xyz)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(atoms), 12, replace=False)
    atoms.positions[idx] += rng.integers(-3, 4, size=(12, 3)) @ atoms.cell.array
    shifted = tmp_path / "shifted.extxyz"
    ase.io.write(shifted, atoms)
    common = ["--ensemble", "nve", "--steps", "1", "--traj-every", "1",
              "--traj-forces", "--traj-wrapped"]
    assert run(inputs, tmp_path / "ref", *common) == 0
    assert run(inputs, tmp_path / "shifted", *common, structure=shifted) == 0
    e1, f1 = _step0(tmp_path / "ref")
    e2, f2 = _step0(tmp_path / "shifted")
    assert e2 == pytest.approx(e1, abs=1e-8)
    np.testing.assert_allclose(f2, f1, atol=1e-8)


def test_relax_cell_changes_the_cell_and_keeps_its_shape(inputs, tmp_path, capsys):
    out = tmp_path / "cell"
    assert run(inputs, out, "--steps", "2", "--relax-steps", "20",
               "--relax-cell", "isotropic", "--no-relax-checks") == 0
    assert "cell (isotropic):" in capsys.readouterr().out
    start, final = ase.io.read(inputs[1]), ase.io.read(out / "final.extxyz")
    ratio = np.diag(final.cell.array) / np.diag(start.cell.array)
    assert np.ptp(ratio) < 1e-12 and abs(ratio[0] - 1.0) > 1e-4
    assert run(inputs, tmp_path / "c2", "--relax-cell", "full") == 1
    assert "--relax-cell relaxes during --relax-steps" in capsys.readouterr().err


def test_bad_input_exits_1_with_a_message(inputs, tmp_path, capsys):
    out = tmp_path / "run"
    assert run(inputs, out, "--steps", "2") == 0
    assert run(inputs, out, "--steps", "4") == 1            # would overwrite
    assert "already holds a run" in capsys.readouterr().err
    (out / "blowup.extxyz").write_text("stale")
    assert run(inputs, out, "--steps", "3", "--overwrite") == 0
    assert not (out / "blowup.extxyz").exists()             # a clean slate
    with np.load(out / "checkpoint.npz") as z:
        assert json.loads(str(z["meta"]))["nsteps"] == 3
    assert run(inputs, out, "--steps", "4", "--restart",
               "--temperature", "350") == 1              # physics changed
    assert "temperature: 300.0 -> 350.0" in capsys.readouterr().err
    assert run(inputs, tmp_path / "e", "--ensemble", "npt",
               "--efield", "0.1", "0", "0") == 1
    assert "--efield needs --ensemble nvt" in capsys.readouterr().err
    assert run(inputs, tmp_path / "b", "--barostat", "full") == 1
    assert "--barostat applies to --ensemble npt or nph" in capsys.readouterr().err
    assert run(inputs, tmp_path / "b2", "--ensemble", "npt", "--steps", "2") == 0
    assert run(inputs, tmp_path / "b2", "--ensemble", "npt", "--barostat", "full",
               "--steps", "4", "--restart") == 1       # the cell mode changed
    assert "barostat: 'isotropic' -> 'full'" in capsys.readouterr().err

    _, xyz = inputs
    atoms = ase.io.read(xyz)
    atoms.symbols[0] = "Na"
    foreign = tmp_path / "foreign.extxyz"
    ase.io.write(foreign, atoms)
    assert run(inputs, tmp_path / "f", structure=foreign) == 1
    assert "['Na'] are not in the model's type_map" in capsys.readouterr().err


def test_instability_exits_2_and_keeps_the_configuration(inputs, tmp_path):
    ckpt, xyz = inputs
    out = tmp_path / "hot"
    rc = run_md.main(["--model", str(ckpt), "--structure", str(xyz),
                      "--out", str(out), "--steps", "5", "--log-every", "1",
                      "--max-temperature", "1"])      # 1 K: trips at once
    assert rc == 2
    assert len(ase.io.read(out / "blowup.extxyz")) == len(ase.io.read(xyz))
    status = json.loads((out / "run.json").read_text())["segments"][-1]["status"]
    assert status.startswith("unstable")
