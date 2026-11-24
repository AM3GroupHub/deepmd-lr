# SPDX-License-Identifier: MIT
"""The long-range model on DPA4, DeePMD-kit's message-passing descriptor.

DPA4 gives the model the same graph interface as DPA4C, so the long-range
term, the charge head and the charge-response force need nothing new on one
device. What does differ is locality: an atom's output reaches several
neighbour shells, and the paths that assume one shell -- the Born tensor's
pair formula, partitions with a halo of one cutoff -- must refuse rather than
return wrong numbers.
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.model.model import BaseModel
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.model.lr_model import receptive_hops, require_one_hop
from deepmd_lr.neighbors import build_neighbor_graph_fast

from .test_joint_model import DT, FITTING, LONG_RANGE, build, frame

RCUT = 4.0
DESCRIPTOR = {
    "type": "dpa4",
    "rcut": RCUT,
    "sel": 40,
    "channels": 16,
    "lmax": 2,
    "n_blocks": 2,
    "n_radial": 8,
    "precision": "float64",
    "use_amp": False,
    "seed": 1,
}


def build_dpa4(long_range: bool = True, fitting: str = "ener", **descriptor):
    """A small DPA4 model whose weights have all been moved off their
    initial values: DPA4 starts with a readout that returns zero, and a zero
    energy would pass every force test."""
    cfg = {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": {**copy.deepcopy(DESCRIPTOR), **descriptor},
        "fitting_net": {**copy.deepcopy(FITTING), "type": fitting},
    }
    if long_range:
        cfg["long_range"] = copy.deepcopy(LONG_RANGE)
    model = get_model(cfg)
    generator = torch.Generator(device="cpu").manual_seed(11)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator, dtype=DT)
            parameter.add_(0.05 * noise.to(parameter))
    return model.eval()


def evaluate(model, coord, atype, cell):
    return model(coord, atype, box=cell)


def derivative(f, h: float = 5e-4) -> float:
    """Fourth-order central difference of ``f`` at zero. The random weights
    make a rough surface: the second-order stencil at a step small enough to
    resolve it would sit on the round-off of the energy."""
    return (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)


class TestBuild:
    def test_the_model_is_dp_lr_on_a_dpa4_descriptor(self):
        model = build_dpa4()
        assert type(model).__name__ == "LongRangeEnergyModel"
        assert type(model.atomic_model.descriptor).__name__ == "DescrptDPA4"
        assert model.has_long_range

    def test_the_fitting_net_deepmd_kit_pairs_with_dpa4_is_accepted(self):
        model = build_dpa4(fitting="dpa4_ener")
        coord, atype, cell = frame(n_mol=6, box=10.0)
        assert torch.isfinite(evaluate(model, coord, atype, cell)["energy"]).all()

    def test_another_fitting_type_is_still_refused(self):
        with pytest.raises(ValueError, match="energy model"):
            build_dpa4(fitting="dipole")

    def test_it_round_trips_through_serialization(self):
        model = build_dpa4()
        coord, atype, cell = frame(n_mol=6, box=10.0)
        again = BaseModel.deserialize(model.serialize()).to(DEVICE).eval()
        assert type(again) is type(model)
        a, b = evaluate(model, coord, atype, cell), evaluate(again, coord, atype, cell)
        for key in ("energy", "force", "charge"):
            assert torch.allclose(a[key], b[key], atol=1e-10), key


class TestLocality:
    """How far an atom's descriptor output reaches, which is what the halo
    of a partition and the Born tensor will have to cover."""

    def test_dpa4c_is_one_hop(self):
        assert receptive_hops(build()) == 1

    @pytest.mark.parametrize(
        ("n_blocks", "use_env_seed"), [(1, True), (2, True), (2, False), (3, True)]
    )
    def test_the_reach_is_the_blocks_and_the_seed(self, n_blocks, use_env_seed):
        """A chain with one neighbour per hop: moving the atom ``k`` links
        away changes the first atom's features up to ``k = hops`` and not at
        all beyond."""
        model = build_dpa4(
            long_range=False, n_blocks=n_blocks, use_env_seed=use_env_seed
        )
        hops = receptive_hops(model)
        assert hops == n_blocks + int(use_env_seed)
        n = hops + 3
        base = torch.zeros((1, n, 3), dtype=DT, device=DEVICE)
        base[0, :, 0] = 0.9 * RCUT * torch.arange(n, dtype=DT, device=DEVICE)
        atype = torch.zeros((1, n), dtype=torch.int64, device=DEVICE)

        def features(coord):
            graph = build_neighbor_graph_fast(coord, atype, None, RCUT)
            out = model.atomic_model.descriptor.call_graph(graph, atype.reshape(-1))
            return out[0][0].detach()

        reference = features(base)
        changed = []
        for k in range(1, n):
            moved = base.clone()
            moved[0, k, 1] += 0.2
            # Exactly zero beyond the reach; the signal that does arrive
            # fades by orders of magnitude per hop, so no threshold.
            changed.append(bool((features(moved) != reference).any()))
        assert changed == [k <= hops for k in range(1, n)]


class TestSerial:
    """One device, the whole system: exact for any depth of message passing."""

    @pytest.mark.parametrize("long_range", [False, True])
    def test_forces_and_virial_are_the_energy_gradient(self, long_range):
        model = build_dpa4(long_range=long_range)
        coord, atype, cell = frame(n_mol=8, box=10.0)
        out = evaluate(model, coord, atype, cell)
        force = out["force"][0]
        assert float(force.abs().max()) > 1e-3  # a model that does something
        # The short-range force is the gradient to 1e-10 eV/A. The reciprocal
        # sum's force is not quite the gradient of its mesh energy: with these
        # weights fixed the gap moves between 6e-11 and 2e-7 eV/A with the
        # mesh, the spline order and alpha, so it is the sum's, not DPA4's.
        tolerance = 1e-6 if long_range else 1e-8
        for atom, axis in [(0, 0), (7, 1), (13, 2), (20, 0)]:

            def displaced(step, atom=atom, axis=axis):
                moved = coord.clone()
                moved[0, atom, axis] += step
                return float(evaluate(model, moved, atype, cell)["energy"].sum())

            assert -derivative(displaced) == pytest.approx(
                float(force[atom, axis]), abs=tolerance
            )

        # The virial against an isotropic strain: -dE/d ln V.
        def strained(step):
            scale = 1 + step
            energy = evaluate(model, coord * scale, atype, cell * scale)["energy"]
            return float(energy.sum())

        trace = out["virial"].reshape(3, 3).diagonal().sum()
        assert -derivative(strained) == pytest.approx(float(trace), abs=1e-6)

    def test_the_charges_carry_the_total_and_move_with_the_geometry(self):
        model = build_dpa4()
        coord, atype, cell = frame(n_mol=8, box=10.0)
        charge = evaluate(model, coord, atype, cell)["charge"]
        assert float(charge.sum()) == pytest.approx(0.0, abs=1e-10)
        assert float(charge.abs().max()) > 1e-4

    def test_a_training_step_reaches_every_part(self):
        """Energy and force loss: the force term needs the second derivative
        through the message passing and the charge response."""
        model = build_dpa4().train()
        coord, atype, cell = frame(n_mol=6, box=10.0)
        out = model(coord, atype, box=cell)
        (out["energy"].sum() + out["force"].square().sum()).backward()
        for name in ("descriptor", "fitting_net"):
            module = getattr(model.atomic_model, name)
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            assert any(float(g.abs().max()) > 0 for g in grads), name
        grads = [p.grad for p in model.charge_fitting.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)

    def test_the_frame_condition_reaches_the_charges(self):
        model = build_dpa4(add_chg_spin_ebd=True, default_chg_spin=[0.0, 1.0])
        coord, atype, cell = frame(n_mol=6, box=10.0)
        neutral = model(coord, atype, box=cell)["charge"]
        condition = torch.tensor([[1.0, 2.0]], dtype=DT, device=DEVICE)
        cation = model(coord, atype, box=cell, charge_spin=condition)["charge"]
        assert float(neutral.sum()) == pytest.approx(0.0, abs=1e-10)
        assert float(cation.sum()) == pytest.approx(1.0, abs=1e-10)


class TestInterfaces:
    """What sits on top of the model reaches DPA4 the way it reaches DPA4C."""

    def test_the_ase_calculator_is_the_model(self):
        from ase import Atoms

        from deepmd_lr.ase import LongRangeCalculator

        model = build_dpa4()
        coord, atype, cell = frame(n_mol=6, box=10.0)
        symbols = ["OH"[t] for t in atype[0].tolist()]
        atoms = Atoms(
            symbols, coord[0].cpu().numpy(), cell=cell[0].cpu().numpy(), pbc=True
        )
        atoms.calc = LongRangeCalculator(model)
        direct = evaluate(model, coord, atype, cell)
        assert atoms.get_potential_energy() == pytest.approx(
            float(direct["energy"]), abs=1e-9
        )
        assert np.allclose(
            atoms.get_forces(), direct["force"][0].detach().cpu().numpy(), atol=1e-9
        )

    def test_the_charge_filter_keeps_the_total_and_the_gradient(self):
        model = build_dpa4()
        coord, atype, cell = frame(n_mol=8, box=10.0)
        plain = evaluate(model, coord, atype, cell)["charge"]
        threshold = float(plain.abs().median())
        model.charge_filter = (threshold, 0.5 * threshold)
        out = evaluate(model, coord, atype, cell)
        assert int((out["charge"] == 0).sum()) > 0
        assert float(out["charge"].sum()) == pytest.approx(0.0, abs=1e-10)

        def displaced(step):
            moved = coord.clone()
            moved[0, 3, 1] += step
            return float(evaluate(model, moved, atype, cell)["energy"].sum())

        assert -derivative(displaced) == pytest.approx(
            float(out["force"][0, 3, 1]), abs=1e-6
        )

    def test_a_same_as_link_shares_the_dpa4_backbone(self):
        from .test_same_as import build_group, pseudo_head_config

        cfg = pseudo_head_config()
        cfg["shared_dict"]["des"] = copy.deepcopy(DESCRIPTOR)
        models = build_group(cfg)
        assert type(models["bulk"].atomic_model.descriptor).__name__ == "DescrptDPA4"
        assert models["slab"].atomic_model is models["bulk"].atomic_model
        assert models["slab"].charge_fitting is models["bulk"].charge_fitting


def test_cli_training_checkpoint_and_restart(tmp_path):
    """``dp train`` with the long-range loss on a DPA4 model, the fitting net
    DeePMD-kit pairs with it at its automatic width, and a restart."""
    from deepmd_lr.ase import load_model

    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    (data / "set.000").mkdir(parents=True)
    coord, atype, cell = frame(n_mol=2, box=8.0)
    n_frame, n_atom = 4, atype.shape[1]
    (data / "type.raw").write_text("\n".join(map(str, atype[0].tolist())) + "\n")
    (data / "type_map.raw").write_text("O\nH\n")
    coordinates = coord[0].cpu().numpy()[None] + 0.05 * rng.normal(
        size=(n_frame, n_atom, 3)
    )
    np.save(data / "set.000/coord.npy", coordinates.reshape(n_frame, -1))
    np.save(data / "set.000/box.npy", np.tile((np.eye(3) * 8.0).reshape(1, 9), (4, 1)))
    np.save(data / "set.000/energy.npy", rng.normal(size=n_frame))
    np.save(data / "set.000/force.npy", rng.normal(size=(n_frame, n_atom * 3)))
    descriptor = {**DESCRIPTOR, "precision": "float32"}
    del descriptor["use_amp"]  # DeePMD-kit's default
    cfg = {
        "model": {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": descriptor,
            "fitting_net": {"type": "dpa4_ener", "neuron": [0], "seed": 1},
            "long_range": {
                "charge_net": {"neuron": [16, 16], "seed": 2},
                "reciprocal": {"alpha": 0.4, "mesh_dimensions": [12, 12, 12]},
            },
        },
        "learning_rate": {"start_lr": 1e-4, "stop_lr": 1e-5, "decay_steps": 1},
        "loss": {
            "type": "ener_lr",
            "start_pref_e": 1.0,
            "limit_pref_e": 1.0,
            "start_pref_f": 1.0,
            "limit_pref_f": 1.0,
            "start_pref_qn": 100.0,
            "limit_pref_qn": 100.0,
        },
        "training": {
            "training_data": {"systems": [str(data)], "batch_size": 1},
            "numb_steps": 2,
            "disp_freq": 1,
            "save_freq": 1,
            "seed": 1,
        },
    }
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps(cfg))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    command = [sys.executable, "-m", "deepmd", "--pt-expt", "train", str(inp)]
    proc = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    checkpoint = tmp_path / "model.ckpt.pt"
    model = load_model(str(checkpoint), device="cpu")
    assert type(model).__name__ == "LongRangeEnergyModel"
    assert type(model.atomic_model.descriptor).__name__ == "DescrptDPA4"
    out = model(coord.cpu(), atype.cpu(), box=torch.eye(3, dtype=DT).unsqueeze(0) * 8.0)
    assert torch.isfinite(out["force"]).all() and "charge" in out
    cfg["training"]["numb_steps"] = 3
    inp.write_text(json.dumps(cfg))
    proc = subprocess.run(
        [*command, "--restart", str(checkpoint)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


class TestRefusals:
    """What assumes one neighbour shell says so instead of running."""

    def test_the_helper_names_the_descriptor_and_its_reach(self):
        require_one_hop(build(), "anything")
        with pytest.raises(NotImplementedError, match="DescrptDPA4.*3 neighbour"):
            require_one_hop(build_dpa4(), "The Born tensor")

    def test_born_charges_from_the_model(self):
        model = build_dpa4()
        model.compute_born = True
        coord, atype, cell = frame(n_mol=6, box=10.0)
        with pytest.raises(NotImplementedError, match="Born tensor"):
            evaluate(model, coord, atype, cell)

    def test_compression(self):
        with pytest.raises(NotImplementedError, match="no compressed form"):
            build_dpa4().enable_compression()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine runs on GPU")
class TestEngine:
    @pytest.fixture
    def system(self):
        return frame(n_mol=16, box=14.0)

    @staticmethod
    def host(system):
        coord, atype, cell = system
        return (t[0].cpu().numpy() for t in (coord, atype, cell))

    @pytest.mark.parametrize("long_range", [False, True])
    def test_one_device_is_the_serial_model(self, system, long_range):
        from deepmd_lr.nvalchemi import DistributedEvaluator

        model = build_dpa4(long_range=long_range)
        coord, atype, cell = self.host(system)
        serial = evaluate(model, *system)
        evaluator = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        energy, force = evaluator(coord)
        assert energy == pytest.approx(float(serial["energy"]), abs=1e-8)
        assert (
            np.abs(
                force.cpu().numpy() - serial["force"][0].detach().cpu().numpy()
            ).max()
            < 1e-8
        )

    def test_a_halo_partition_is_refused(self, system):
        from deepmd_lr.nvalchemi import evaluate_partitioned

        coord, atype, cell = self.host(system)
        with pytest.raises(NotImplementedError, match="halo of ghost atoms"):
            evaluate_partitioned(
                build_dpa4(), coord, atype, cell, n_partitions=2, devices=[DEVICE] * 2
            )

    def test_several_devices_and_born_charges_are_refused_at_construction(self, system):
        from deepmd_lr.nvalchemi import DistributedEvaluator

        model = build_dpa4()
        _coord, atype, cell = self.host(system)
        with pytest.raises(NotImplementedError, match="several devices"):
            DistributedEvaluator(
                model, atype, cell, devices=[DEVICE] * 2, memory_headroom=None
            )
        with pytest.raises(NotImplementedError, match="Born tensor"):
            DistributedEvaluator(
                model,
                atype,
                cell,
                devices=[DEVICE],
                memory_headroom=None,
                compute_born=True,
            )
