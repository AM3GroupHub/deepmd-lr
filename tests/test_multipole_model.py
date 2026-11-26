# SPDX-License-Identifier: MIT
"""A model whose atoms carry dipoles and quadrupoles beside their charges.

What is pinned:
- **The moments are tensors**: rotating the frame rotates every dipole as a
  vector and every quadrupole as a rank-2 tensor, and leaves the energy alone.
  This is the test that reads DPA4's harmonic convention.
- **They are in the energy**: the long-range energy is the multipole term
  evaluated on the charges and moments the model reports.
- **Forces and virial** are the gradient of that energy.
- **The plumbing**: training reaches the head, serialization restores it, and
  what cannot work is refused when the model is built.
"""

import copy
import json

import numpy as np
import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.model.model import BaseModel
from deepmd.pt_expt.utils.env import DEVICE
from deepmd.utils.argcheck import normalize

from deepmd_lr.lr.multipole import MultipoleLongRangeEnergy

from .test_curriculum import (
    DPA4,
    REPO,
    changed,
    dp_train,
    run_python,
    write_dataset,
)
from .test_dpa4 import DESCRIPTOR, derivative, evaluate, roughened
from .test_joint_model import DT, FITTING, frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)

HEAD = {
    "neuron": [16, 16],
    "activation_function": "silu",
    "precision": "float64",
    "seed": 5,
}
RECIPROCAL = {"method": "ewald", "alpha": 0.4, "k_cutoff": 5.0}


def config(lmax: int = 2, **descriptor) -> dict:
    return {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": {**copy.deepcopy(DESCRIPTOR), **descriptor},
        "fitting_net": copy.deepcopy(FITTING),
        "long_range": {
            "charge_net": {**HEAD, "seed": 7},
            "multipole_net": {**HEAD, "lmax": lmax},
            "reciprocal": copy.deepcopy(RECIPROCAL),
        },
    }


def build(lmax: int = 2, **descriptor):
    return roughened(get_model(config(lmax, **descriptor)))


def rotation(seed: int = 4) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return torch.tensor(q, dtype=DT, device=DEVICE)


class TestMoments:
    @pytest.mark.parametrize("lmax", [1, 2])
    def test_the_model_reports_them(self, lmax):
        model = build(lmax)
        coord, atype, cell = frame(n_mol=6, box=10.0)
        out = evaluate(model, coord, atype, cell)
        n = atype.shape[1]
        assert out["dipole"].shape == (1, n, 3)
        assert float(out["dipole"].abs().max()) > 1e-5
        assert ("quadrupole" in out) == (lmax == 2)
        if lmax == 2:
            q = out["quadrupole"]
            assert q.shape == (1, n, 3, 3)
            assert float(q.abs().max()) > 1e-5
            assert torch.allclose(q, q.transpose(-1, -2), atol=1e-12)
            assert float(q.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()) < 1e-10

    def test_they_rotate_with_the_frame(self):
        model = build(2)
        coord, atype, cell = frame(n_mol=6, box=10.0)
        turn = rotation()
        here = evaluate(model, coord, atype, cell)
        there = evaluate(model, coord @ turn.T, atype, cell @ turn.T)
        assert float(there["energy"]) == pytest.approx(float(here["energy"]), abs=1e-8)
        torch.testing.assert_close(there["charge"], here["charge"], atol=1e-9, rtol=0)
        torch.testing.assert_close(
            there["dipole"], here["dipole"] @ turn.T, atol=1e-9, rtol=0
        )
        torch.testing.assert_close(
            there["quadrupole"],
            turn @ here["quadrupole"] @ turn.T,
            atol=1e-9,
            rtol=0,
        )
        torch.testing.assert_close(
            there["force"], here["force"] @ turn.T, atol=1e-7, rtol=0
        )

    def test_they_are_what_the_long_range_energy_sums(self):
        model = build(2)
        coord, atype, cell = frame(n_mol=6, box=10.0)
        out = evaluate(model, coord, atype, cell)
        assert isinstance(model.long_range_energy, MultipoleLongRangeEnergy)
        direct = model.long_range_energy(
            coord,
            out["charge"],
            cell,
            dipole=out["dipole"],
            quadrupole=out["quadrupole"],
        ).sum()
        assert float(out["energy_long_range"]) == pytest.approx(float(direct), abs=1e-9)
        charges_only = model.long_range_energy(coord, out["charge"], cell).sum()
        assert abs(float(direct - charges_only)) > 1e-6

    def test_one_frame_does_not_see_the_other(self):
        model = build(1)
        a, b = frame(n_mol=6, box=10.0, seed=1), frame(n_mol=6, box=10.0, seed=2)
        both = evaluate(
            model, torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]),
            torch.cat([a[2], b[2]]),
        )  # fmt: skip
        for i, one in enumerate((a, b)):
            alone = evaluate(model, *one)
            for key in ("energy", "force", "dipole"):
                torch.testing.assert_close(
                    both[key][i : i + 1], alone[key], atol=1e-8, rtol=0
                )


class TestDerivatives:
    @pytest.mark.parametrize("lmax", [1, 2])
    def test_forces_and_virial_are_the_energy_gradient(self, lmax):
        model = build(lmax)
        coord, atype, cell = frame(n_mol=8, box=10.0)
        out = evaluate(model, coord, atype, cell)
        force = out["force"][0]
        assert float(force.abs().max()) > 1e-3
        for atom, axis in [(0, 0), (7, 1), (13, 2), (20, 0)]:

            def displaced(step, atom=atom, axis=axis):
                moved = coord.clone()
                moved[0, atom, axis] += step
                return float(evaluate(model, moved, atype, cell)["energy"].sum())

            assert -derivative(displaced) == pytest.approx(
                float(force[atom, axis]), abs=1e-7
            )

        def strained(step):
            scale = 1 + step
            energy = evaluate(model, coord * scale, atype, cell * scale)["energy"]
            return float(energy.sum())

        trace = out["virial"].reshape(3, 3).diagonal().sum()
        assert -derivative(strained) == pytest.approx(float(trace), abs=1e-6)

        # A shear, which a virial with an antisymmetric part would get wrong:
        # the moments turn with the frame, so the true one is symmetric.
        virial = out["virial"].reshape(3, 3)
        torch.testing.assert_close(virial, virial.T, atol=1e-9, rtol=0)

        def sheared(step):
            strain = torch.eye(3, dtype=DT, device=DEVICE)
            strain[0, 1] = strain[1, 0] = step
            energy = evaluate(model, coord @ strain, atype, cell @ strain)["energy"]
            return float(energy.sum())

        assert -derivative(sheared) == pytest.approx(
            float(virial[0, 1] + virial[1, 0]), abs=1e-6
        )

    def test_a_training_step_reaches_the_head(self):
        model = build(2).train()
        coord, atype, cell = frame(n_mol=6, box=10.0)
        out = model(coord, atype, box=cell)
        (out["energy"].sum() + out["force"].square().sum()).backward()
        for head in (model.multipole_fitting, model.charge_fitting):
            grads = [p.grad for p in head.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert any(float(g.abs().max()) > 0 for g in grads)
        descriptor = model.atomic_model.descriptor
        grads = [p.grad for p in descriptor.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


class TestBornTensor:
    """``Z = dP/dr`` with ``P = sum q r + sum mu``: the dipoles add their own
    derivative to the charges' part, and the quadrupoles nothing."""

    @staticmethod
    def born(model, coord, atype, cell) -> torch.Tensor:
        model.compute_born = True
        try:
            return evaluate(model, coord, atype, cell)["born_charge"][0]
        finally:
            model.compute_born = False

    def test_the_dipoles_add_their_derivative(self):
        model = build(2)
        coord, atype, cell = frame(n_mol=4, box=10.0)
        full = self.born(model, coord, atype, cell)
        # The charges come from the invariant features alone: without the
        # head they, and their part of Z, are what they were.
        head, term = model.multipole_fitting, model.long_range_energy
        model.multipole_fitting = None
        model.long_range_energy = type(term).__mro__[1](**term.serialize())
        try:
            charges_only = self.born(model, coord, atype, cell)
        finally:
            model.multipole_fitting, model.long_range_energy = head, term
        added = full - charges_only
        assert float(added.abs().max()) > 1e-5
        for atom, axis in [(0, 0), (4, 1), (9, 2)]:

            def moment(step, atom=atom, axis=axis):
                moved = coord.clone()
                moved[0, atom, axis] += step
                return evaluate(model, moved, atype, cell)["dipole"][0].sum(0)

            h = 1e-4
            column = (
                8 * (moment(h) - moment(-h)) - (moment(2 * h) - moment(-2 * h))
            ) / (12 * h)
            torch.testing.assert_close(added[atom, :, axis], column, atol=1e-7, rtol=0)

    def test_the_sum_rule_is_still_the_total_charge(self):
        model = build(2)
        coord, atype, cell = frame(n_mol=4, box=10.0)
        total = self.born(model, coord, atype, cell).sum(0)
        assert float(total.abs().max()) < 1e-8  # a neutral frame

    def test_it_rotates_as_a_tensor(self):
        model = build(2)
        coord, atype, cell = frame(n_mol=4, box=10.0)
        turn = rotation()
        here = self.born(model, coord, atype, cell)
        there = self.born(model, coord @ turn.T, atype, cell @ turn.T)
        torch.testing.assert_close(there, turn @ here @ turn.T, atol=1e-8, rtol=0)

    def test_a_loss_on_it_reaches_the_head(self):
        model = build(1).train()
        model.compute_born = True
        coord, atype, cell = frame(n_mol=4, box=10.0)
        out = model(coord, atype, box=cell)
        out["born_charge"].square().sum().backward()
        grads = [p.grad for p in model.multipole_fitting.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().max()) > 0 for g in grads)


class TestEngine:
    """Partitions and the MD evaluator: the moments cross a partition beside
    the charges, and come back as ``dE/dmu`` and ``dE/dQ``."""

    @staticmethod
    def host(system):
        coord, atype, cell = system
        return (t[0].cpu().numpy() for t in (coord, atype, cell))

    @pytest.mark.parametrize("lmax", [1, 2])
    def test_partitions_are_the_serial_model(self, lmax):
        from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy

        model = build(lmax, n_blocks=1)
        system = frame(n_mol=16, box=14.0)
        coord, atype, cell = self.host(system)
        serial = evaluate(model, *system)
        keys = ["force", "virial", "charge", "dipole"] + ["quadrupole"] * (lmax == 2)
        for n in (1, 2, 4):
            for merged in (False, True):
                out = to_numpy(
                    evaluate_partitioned(
                        model, coord, atype, cell, n_partitions=n,
                        devices=[DEVICE] * n, compute_virial=True,
                        merge_backward=merged,
                    )
                )  # fmt: skip
                assert out["energy"] == pytest.approx(float(serial["energy"]), abs=1e-8)
                for key in keys:
                    reference = (
                        serial[key].detach().cpu().numpy().reshape(out[key].shape)
                    )
                    assert np.abs(out[key] - reference).max() < 1e-8, (n, merged, key)

    def test_born_tensors_agree_on_every_route(self):
        from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy

        model = build(2, n_blocks=1)
        system = frame(n_mol=16, box=14.0)
        coord, atype, cell = self.host(system)
        serial = TestBornTensor.born(model, *system).cpu().numpy()
        for n in (1, 2, 4):
            out = to_numpy(
                evaluate_partitioned(
                    model, coord, atype, cell, n_partitions=n,
                    devices=[DEVICE] * n, compute_born=True,
                )
            )  # fmt: skip
            assert np.abs(out["born_charge"] - serial).max() < 1e-8, n

    def test_the_evaluator_and_the_calculator_report_the_moments(self):
        from ase import Atoms

        from deepmd_lr.ase import LongRangeCalculator
        from deepmd_lr.nvalchemi import DistributedEvaluator

        model = build(2)
        system = frame(n_mol=6, box=10.0)
        coord, atype, cell = self.host(system)
        serial = evaluate(model, *system)
        evaluator = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        energy, force = evaluator(coord)
        assert energy == pytest.approx(float(serial["energy"]), abs=1e-8)
        assert (force - serial["force"][0].detach()).abs().max() < 1e-8
        dipole = torch.as_tensor(evaluator.results["dipole"], device=DEVICE)
        assert torch.allclose(dipole, serial["dipole"][0], atol=1e-10)
        atoms = Atoms(
            symbols=["O" if t == 0 else "H" for t in atype],
            positions=coord, cell=cell, pbc=True,
        )  # fmt: skip
        atoms.calc = LongRangeCalculator(model)
        assert atoms.get_potential_energy() == pytest.approx(
            float(serial["energy"]), abs=1e-8
        )
        for key, name in (("dipole", "dipoles"), ("quadrupole", "quadrupoles")):
            assert np.allclose(
                atoms.calc.results[name], serial[key][0].detach().cpu().numpy(),
                atol=1e-10,
            )  # fmt: skip


class TestMesh:
    def test_a_mesh_model_is_the_explicit_one(self):
        """The same weights on PME and on the explicit sum."""
        exact = build(2)
        cfg = config(2)
        cfg["long_range"]["reciprocal"] = {
            "method": "pme", "alpha": 0.4, "mesh_dimensions": [40, 40, 40],
        }  # fmt: skip
        mesh = roughened(get_model(cfg))
        coord, atype, cell = frame(n_mol=6, box=10.0)
        a, b = evaluate(exact, coord, atype, cell), evaluate(mesh, coord, atype, cell)
        assert float(a["energy"]) == pytest.approx(float(b["energy"]), abs=1e-5)
        torch.testing.assert_close(a["force"], b["force"], atol=1e-4, rtol=0)
        torch.testing.assert_close(a["dipole"], b["dipole"], atol=1e-12, rtol=0)

    def test_the_evaluator_takes_a_finer_mesh(self):
        cfg = config(1)
        cfg["long_range"]["reciprocal"] = {"method": "pme", "alpha": 0.4}
        model = get_model(cfg)
        term = model.long_range_energy_for(np.eye(3) * 12.4, (True, True, True))
        assert tuple(term.mesh_dimensions) == (20, 20, 20)


class TestPlumbing:
    def test_it_round_trips_through_serialization(self):
        model = build(2)
        coord, atype, cell = frame(n_mol=6, box=10.0)
        again = BaseModel.deserialize(model.serialize()).to(DEVICE).eval()
        assert again.has_multipoles
        assert isinstance(again.long_range_energy, MultipoleLongRangeEnergy)
        a, b = evaluate(model, coord, atype, cell), evaluate(again, coord, atype, cell)
        for key in ("energy", "force", "charge", "dipole", "quadrupole"):
            assert torch.allclose(a[key], b[key], atol=1e-10), key

    def test_without_the_head_nothing_is_reported(self):
        cfg = config()
        del cfg["long_range"]["multipole_net"]
        model = roughened(get_model(cfg))
        assert not model.has_multipoles
        out = evaluate(model, *frame(n_mol=6, box=10.0))
        assert "dipole" not in out and "quadrupole" not in out

    def test_the_schema_takes_the_block(self):
        cfg = {
            "model": config(1),
            "learning_rate": {"start_lr": 1e-3, "stop_lr": 1e-5},
            "loss": {"type": "ener"},
            "training": {
                "training_data": {"systems": ["x"], "batch_size": 1},
                "numb_steps": 1,
            },
        }
        block = normalize(cfg)["model"]["long_range"]["multipole_net"]
        assert block["lmax"] == 1 and block["neuron"] == [16, 16]


class TestRefusals:
    def test_a_descriptor_without_equivariant_features(self):
        cfg = config()
        cfg["descriptor"] = {"type": "dpa4c", "rcut": 4.0, "channels": 8, "lmax": 2}
        with pytest.raises(NotImplementedError, match="dpa4"):
            get_model(cfg)

    def test_a_moment_beyond_the_quadrupole(self):
        with pytest.raises(ValueError, match="lmax"):
            get_model(config(3))

    def test_a_schedule_that_ends_below_the_moment(self):
        cfg = config(2)
        cfg["descriptor"]["lmax"] = 1
        with pytest.raises(ValueError, match="degree 1"):
            get_model(cfg)

    def test_a_cluster(self):
        model = build(1)
        coord, atype, _ = frame(n_mol=4, box=10.0)
        with pytest.raises(NotImplementedError, match="isolated"):
            model(coord, atype, box=None)


def test_the_curriculum_through_the_cli(tmp_path):
    """SR, then the charge and multipole heads alone on it, then everything:
    `dp train`, the graft and a restart on a model with a multipole head."""
    data = write_dataset(tmp_path)
    short_range = {"type": "standard", "type_map": ["O", "H"], **copy.deepcopy(DPA4)}
    long_range = {
        **copy.deepcopy(short_range),
        "type": "dp_lr",
        "long_range": {
            "charge_net": {"neuron": [16, 16], "seed": 2},
            "multipole_net": {"neuron": [16, 16], "seed": 3, "lmax": 2},
            "reciprocal": {"method": "ewald", "alpha": 0.4, "k_cutoff": 4.0},
        },
    }

    def weights(path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)["model"]

    def stage(name: str, model: dict, *flags: str) -> dict:
        folder = tmp_path / name
        folder.mkdir()
        cfg = {
            "model": model,
            "learning_rate": {"start_lr": 1e-3, "stop_lr": 1e-4, "decay_steps": 1},
            "loss": {"type": "ener", "start_pref_e": 1.0, "limit_pref_e": 1.0,
                     "start_pref_f": 1.0, "limit_pref_f": 1.0},
            "training": {
                "training_data": {"systems": [str(data)], "batch_size": 1},
                "numb_steps": 3, "disp_freq": 1, "save_freq": 3, "seed": 1,
            },
        }  # fmt: skip
        (folder / "input.json").write_text(json.dumps(cfg))
        if name == "lr":
            run_python(
                str(REPO / "scripts" / "graft_short_range.py"),
                "--short-range", str(tmp_path / "sr" / "model.ckpt.pt"),
                "--input", "input.json", "--out", "init.ckpt.pt",
                cwd=folder,
            )  # fmt: skip
        dp_train("input.json", *flags, cwd=folder)
        return weights(folder / "model.ckpt.pt")

    stage("sr", short_range)
    head_only = copy.deepcopy(long_range)
    head_only["long_range"]["train_charge_head_only"] = True
    lr = stage("lr", head_only, "--init-model", "init.ckpt")
    moved = changed(weights(tmp_path / "lr" / "init.ckpt.pt"), lr)
    assert any("multipole_fitting." in key for key in moved)
    assert any("charge_fitting." in key for key in moved)
    assert all("_fitting." in key and ".fitting_net." not in key for key in moved)
    ft = stage("ft", long_range, "--init-model", str(tmp_path / "lr" / "model.ckpt"))
    moved = changed(lr, ft)
    assert any("multipole_fitting." in key for key in moved)
    assert any(".descriptor." in key for key in moved)
