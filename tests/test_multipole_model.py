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

import numpy as np
import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.model.model import BaseModel
from deepmd.pt_expt.utils.env import DEVICE
from deepmd.utils.argcheck import normalize

from deepmd_lr.lr.multipole import MultipoleLongRangeEnergy

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

    def test_a_mesh(self):
        cfg = config()
        cfg["long_range"]["reciprocal"] = {"method": "pme", "alpha": 0.4}
        with pytest.raises(ValueError, match="ewald"):
            get_model(cfg)
