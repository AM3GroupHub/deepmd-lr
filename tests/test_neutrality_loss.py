"""The `ener_lr` loss: EnergyLoss plus a penalty on the neutrality projection's shift.

What is pinned:
- **The shift the model reports** is exactly what the projection added to
  every atom: charge = raw + shift, with shift = (Q - sum raw) / N, masks and
  a total charge included.
- **The loss** equals upstream's `ener` loss when both prefactors are zero,
  and otherwise adds pref * mean(shift^2) (or mean |shift| under mae), with
  the prefactor on DeePMD-kit's learning-rate schedule.
- **The gradient** of the term reaches the charge head and the shared
  descriptor, and a few steps on it shrink the shift.
- **The plumbing**: the schema accepts the type, the trainer's loss factory
  builds it, it serializes, and a model with no charges is refused.
"""

import copy

import numpy as np
import pytest
import torch
from deepmd.dpmodel.loss.ener import EnergyLoss
from deepmd.pt_expt.train import training
from deepmd.utils.argcheck import normalize

from deepmd_lr.fitting.charge import projection_shift
from deepmd_lr.loss import LOSS_TYPE, LongRangeEnergyLoss
from deepmd_lr.model import lr_model

from .test_joint_model import DEVICE, build, frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)

ENER = {"start_pref_e": 0.02, "limit_pref_e": 1.0, "start_pref_f": 1000.0,
        "limit_pref_f": 1.0}
START_LR = 1e-3
#: A negligible energy term: upstream's loss is a plain 0 when every one of its
#: prefactors is zero, which its own `sqrt` then refuses.
PENALTY_ONLY = {"start_pref_e": 1e-12, "limit_pref_e": 1e-12, "start_pref_f": 0.0,
                "limit_pref_f": 0.0}


def predict(model, charge_spin=None, n_frames=2):
    """A forward over `n_frames` frames, with the raw charges the projection saw."""
    coords, atypes, cells = zip(*(frame(seed=s) for s in range(n_frames)), strict=True)
    coord, atype, cell = torch.cat(coords), torch.cat(atypes), torch.cat(cells)
    seen = {}
    original = lr_model.project_to_total_charge

    def spy(raw, total, mask=None):
        seen["raw"], seen["mask"] = raw, mask
        return original(raw, total, mask=mask)

    lr_model.project_to_total_charge = spy
    try:
        out = model(coord, atype, box=cell, charge_spin=charge_spin)
    finally:
        lr_model.project_to_total_charge = original
    return out, seen["raw"], seen["mask"], coord.shape[0]


def labels(out, nf):
    return {
        "energy": out["energy"].detach() + 0.1,
        "find_energy": 1.0,
        "force": out["force"].detach() + 0.01,
        "find_force": 1.0,
    }


class TestShift:
    def test_it_is_what_the_projection_added(self):
        model = build()
        out, raw, mask, nf = predict(model)
        shift = out["charge_projection_shift"]
        assert shift.shape == (nf,)
        n = mask.sum(-1).to(raw.dtype)
        expected = -(raw * mask).sum(-1) / n
        torch.testing.assert_close(shift, expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(out["charge"], raw + shift[:, None])

    def test_a_total_charge_enters_it(self):
        model = build()
        cs = torch.tensor([[1.0, 2.0]], dtype=torch.float64, device=DEVICE)
        out, raw, mask, nf = predict(model, charge_spin=cs)
        n = mask.sum(-1).to(raw.dtype)
        expected = (1.0 - (raw * mask).sum(-1)) / n
        torch.testing.assert_close(
            out["charge_projection_shift"], expected, rtol=0, atol=1e-12
        )

    def test_masked_atoms_take_no_part(self):
        raw = torch.tensor([[0.5, 0.2, 9.0]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        charge = torch.tensor([[0.15, -0.15, 0.0]], dtype=torch.float64)
        assert projection_shift(charge, raw, mask).item() == pytest.approx(-0.35)

    def test_a_short_range_model_reports_none(self):
        out, *_ = predict_sr()
        assert "charge_projection_shift" not in out


def predict_sr():
    model = build(long_range=False)
    coord, atype, cell = frame()
    return model(coord, atype, box=cell), None


class TestLoss:
    def test_zero_prefactors_are_the_ener_loss(self):
        model = build()
        out, _, _, nf = predict(model)
        lab = labels(out, nf)
        ours = LongRangeEnergyLoss(starter_learning_rate=START_LR, **ENER)
        theirs = EnergyLoss(starter_learning_rate=START_LR, **ENER)
        a, ma = ours(START_LR / 2, 12, out, lab)
        b, mb = theirs(START_LR / 2, 12, out, lab)
        assert float(a) == float(b)
        assert "rmse_qn" not in ma and ma.keys() == mb.keys()

    @pytest.mark.parametrize("loss_func", ["mse", "mae"])
    def test_the_term_and_its_schedule(self, loss_func):
        model = build()
        out, _, _, nf = predict(model)
        lab = labels(out, nf)
        start, limit, lr = 3.0, 0.5, START_LR / 4
        ours = LongRangeEnergyLoss(starter_learning_rate=START_LR, loss_func=loss_func,
                                   start_pref_qn=start, limit_pref_qn=limit, **ENER)
        base = EnergyLoss(starter_learning_rate=START_LR, loss_func=loss_func, **ENER)
        a, more = ours(lr, 12, out, lab)
        b, _ = base(lr, 12, out, lab)
        shift = out["charge_projection_shift"]
        pref = limit + (start - limit) * lr / START_LR
        term = (shift.abs().mean() if loss_func == "mae" else (shift**2).mean())
        torch.testing.assert_close(a, b + pref * term)
        torch.testing.assert_close(more["rmse_qn"], (shift**2).mean().sqrt())
        torch.testing.assert_close(more["rmse"], a.sqrt())
        assert ("mae_qn" in more) == (loss_func == "mae")

    def test_its_gradient_reaches_the_head_and_the_backbone(self):
        model = build()
        model.train()
        out, _, _, nf = predict(model)
        loss = LongRangeEnergyLoss(starter_learning_rate=START_LR, **PENALTY_ONLY,
                                   start_pref_qn=1.0, limit_pref_qn=1.0)
        total, _ = loss(START_LR, 12, out, labels(out, nf))
        total.backward()

        def grad_norm(module):
            return sum(float(p.grad.abs().sum()) for p in module.parameters()
                       if p.grad is not None)

        assert grad_norm(model.charge_fitting) > 0
        assert grad_norm(model.atomic_model.descriptor) > 0

    def test_a_few_steps_shrink_the_shift(self):
        """What the term is for: the raw charges come to satisfy the constraint
        themselves, so the projection has less and less to add."""
        torch.manual_seed(0)
        model = build()
        model.train()
        loss = LongRangeEnergyLoss(starter_learning_rate=START_LR, **PENALTY_ONLY,
                                   start_pref_qn=1.0, limit_pref_qn=1.0)
        opt = torch.optim.Adam(model.charge_fitting.parameters(), lr=1e-2)
        history = []
        for _ in range(30):
            out, _, _, nf = predict(model)
            total, more = loss(START_LR, 12, out, labels(out, nf))
            history.append(float(more["rmse_qn"]))
            opt.zero_grad()
            total.backward()
            opt.step()
        assert history[-1] < 0.25 * history[0]      # 0.279 -> 0.046 e measured

    def test_a_model_without_charges_is_refused(self):
        out, _ = predict_sr()
        loss = LongRangeEnergyLoss(starter_learning_rate=START_LR, start_pref_qn=1.0,
                                   limit_pref_qn=1.0, **ENER)
        with pytest.raises(ValueError, match="long_range"):
            loss(START_LR, 12, out, labels(out, 1))

    def test_negative_prefactors_are_refused(self):
        with pytest.raises(ValueError, match=">= 0"):
            LongRangeEnergyLoss(starter_learning_rate=START_LR, start_pref_qn=-1.0)

    def test_it_round_trips(self):
        loss = LongRangeEnergyLoss(starter_learning_rate=START_LR, start_pref_qn=2.0,
                                   limit_pref_qn=0.5, loss_func="mae", **ENER)
        again = LongRangeEnergyLoss.deserialize(loss.serialize())
        assert again.serialize() == loss.serialize()
        assert (again.start_pref_qn, again.limit_pref_qn) == (2.0, 0.5)


class TestPlumbing:
    CONFIG = {
        "model": {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": {"type": "dpa4c", "rcut": 6.0, "channels": 16, "lmax": 2,
                           "basis_type": "bessel", "n_radial": 8, "radial_modes": 2,
                           "seed": 1},
            "fitting_net": {"neuron": [16], "seed": 1},
            "long_range": {"charge_net": {"neuron": [8]},
                           "reciprocal": {"alpha": 0.4, "mesh_dimensions": [8, 8, 8]}},
        },
        "learning_rate": {"type": "exp", "start_lr": 1e-3, "stop_lr": 1e-5},
        "loss": {"type": LOSS_TYPE, "start_pref_qn": 10.0, "limit_pref_qn": 1.0},
        "training": {"training_data": {"systems": ["./data"], "batch_size": 1},
                     "numb_steps": 1, "seed": 1},
    }

    def test_the_schema_accepts_it_with_the_ener_arguments(self):
        cfg = normalize(copy.deepcopy(self.CONFIG))
        loss = cfg["loss"]
        assert loss["type"] == LOSS_TYPE
        assert (loss["start_pref_qn"], loss["limit_pref_qn"]) == (10.0, 1.0)
        assert "start_pref_f" in loss            # the ener arguments came along

    def test_the_schema_rejects_an_unknown_key(self):
        cfg = copy.deepcopy(self.CONFIG)
        cfg["loss"]["pref_qn"] = 1.0
        with pytest.raises(Exception, match="pref_qn"):
            normalize(cfg)

    def test_the_trainers_factory_builds_it(self):
        params = normalize(copy.deepcopy(self.CONFIG))["loss"]
        loss = training.get_loss(dict(params), 1e-3, 2, None)
        assert isinstance(loss, LongRangeEnergyLoss)
        assert loss.has_qn and loss.starter_learning_rate == 1e-3
        other = training.get_loss({"type": "ener"}, 1e-3, 2, None)
        assert type(other) is EnergyLoss

    def test_the_hook_is_idempotent(self):
        from deepmd_lr import loss as loss_module

        hooked = training.get_loss
        loss_module.install()
        assert training.get_loss is hooked


def test_the_shift_matches_a_measured_offset():
    """The quantity behind the C/Si finding, on the fixture: shift = -mean raw
    for a neutral frame, whatever constant the raw charges carry."""
    model = build()
    out, raw, mask, nf = predict(model)
    np.testing.assert_allclose(
        out["charge_projection_shift"].detach().cpu().numpy(),
        -raw.detach().mean(-1).cpu().numpy(),
        atol=1e-12,
    )
