# SPDX-License-Identifier: MIT
"""The Born term of the ``ener_lr`` loss: labelled Born effective charges
supervising the charge head.

What is pinned:
- **The value**: ``pref_bec * mean((s Z - Z_label)^2)`` (absolute under mae)
  on top of the ``ener`` loss, on the learning-rate schedule, and nothing
  from a system without the label.
- **The gradient** reaches the charge head through the model's own Born
  tensor, and steps on it alone bring the tensors of a student model to a
  teacher's, on DPA4C and on DPA4.
- **The plumbing**: the label is declared, the trainer's loss factory
  switches the model's Born tensors on, a model without charges is refused,
  and ``dp train`` runs it on a dataset where one system has the label and
  one has not, also in the charge-head-only stage.
"""

import json

import numpy as np
import pytest
import torch
from deepmd.pt_expt.train import training
from deepmd.utils.argcheck import normalize

from deepmd_lr.loss import BORN_KEY, LOSS_TYPE, LongRangeEnergyLoss

from .test_curriculum import dp_train, long_range, write_dataset
from .test_dpa4 import build_dpa4
from .test_joint_model import build, frame
from .test_neutrality_loss import PENALTY_ONLY, START_LR

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)

ENER = {
    "start_pref_e": 1.0,
    "limit_pref_e": 1.0,
    "start_pref_f": 10.0,
    "limit_pref_f": 1.0,
}


def labelled(model, find: float = 1.0, seed: int = 3):
    """A forward with Born tensors, and labels around them."""
    coord, atype, cell = frame()
    model.compute_born = True
    out = model(coord, atype, box=cell)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(out[BORN_KEY].shape, generator=generator, dtype=torch.float64)
    nf, nloc = atype.shape
    label = {
        "energy": out["energy"].detach() + 0.1,
        "find_energy": 1.0,
        "force": out["force"].detach() + 0.01,
        "find_force": 1.0,
        BORN_KEY: (out[BORN_KEY].detach() + 0.1 * noise.to(coord)).reshape(nf, -1),
        f"find_{BORN_KEY}": find,
    }
    return out, label, nloc


class TestValue:
    @pytest.mark.parametrize("loss_func", ["mse", "mae"])
    def test_term_added_to_the_energy_loss_on_schedule(self, loss_func):
        out, label, nloc = labelled(build())
        scale, start, limit, rate = 1.7, 4.0, 0.5, 0.25 * START_LR
        base = LongRangeEnergyLoss(
            starter_learning_rate=START_LR, loss_func=loss_func, **ENER
        )
        full = LongRangeEnergyLoss(
            starter_learning_rate=START_LR,
            loss_func=loss_func,
            start_pref_bec=start,
            limit_pref_bec=limit,
            bec_scale=scale,
            **ENER,
        )
        reference, _ = base.call(rate, nloc, out, label)
        value, more = full.call(rate, nloc, out, label)
        diff = scale * out[BORN_KEY] - label[BORN_KEY].reshape(out[BORN_KEY].shape)
        term = diff.abs().mean() if loss_func == "mae" else (diff**2).mean()
        pref = limit + (start - limit) * 0.25
        torch.testing.assert_close(value, reference + pref * term)
        torch.testing.assert_close(more["rmse_bec"], (diff**2).mean().sqrt())
        if loss_func == "mae":
            torch.testing.assert_close(more["mae_bec"], diff.abs().mean())

    def test_a_system_without_the_label_contributes_nothing(self):
        out, label, nloc = labelled(build(), find=0.0)
        base = LongRangeEnergyLoss(starter_learning_rate=START_LR, **ENER)
        full = LongRangeEnergyLoss(
            starter_learning_rate=START_LR, start_pref_bec=5.0, **ENER
        )
        reference, _ = base.call(START_LR, nloc, out, label)
        value, more = full.call(START_LR, nloc, out, label)
        torch.testing.assert_close(value, reference)
        assert torch.isnan(torch.as_tensor(more["rmse_bec"]))

    def test_label_declared_only_with_the_term(self):
        plain = LongRangeEnergyLoss(starter_learning_rate=START_LR, **ENER)
        assert BORN_KEY not in [item.key for item in plain.label_requirement]
        full = LongRangeEnergyLoss(
            starter_learning_rate=START_LR, start_pref_bec=1.0, **ENER
        )
        (item,) = [i for i in full.label_requirement if i.key == BORN_KEY]
        assert item.ndof == 9 and item.atomic and not item.must

    def test_settings_checked_and_serialized(self):
        with pytest.raises(ValueError, match="pref_bec"):
            LongRangeEnergyLoss(starter_learning_rate=START_LR, start_pref_bec=-1.0)
        with pytest.raises(ValueError, match="bec_scale"):
            LongRangeEnergyLoss(starter_learning_rate=START_LR, bec_scale=0.0)
        loss = LongRangeEnergyLoss(
            starter_learning_rate=START_LR,
            start_pref_bec=2.0,
            limit_pref_bec=1.0,
            bec_scale=1.75,
            **ENER,
        )
        again = LongRangeEnergyLoss.deserialize(loss.serialize())
        assert (again.start_pref_bec, again.limit_pref_bec, again.bec_scale) == (
            2.0,
            1.0,
            1.75,
        )


class TestFactory:
    PARAMS = {"type": LOSS_TYPE, "start_pref_bec": 1.0, "limit_pref_bec": 1.0}

    def test_schema_accepts_the_keys(self):
        cfg = {
            "model": long_range("dpa4c", charge_head_only=False),
            "learning_rate": {"start_lr": 1e-3, "stop_lr": 1e-5},
            "loss": {**self.PARAMS, "bec_scale": 1.75},
            "training": {
                "training_data": {"systems": ["x"], "batch_size": 1},
                "numb_steps": 1,
            },
        }
        assert normalize(cfg)["loss"]["bec_scale"] == 1.75

    def test_factory_switches_the_born_tensors_on(self):
        model = build()
        assert not model.compute_born
        loss = training.get_loss(dict(self.PARAMS), START_LR, 2, model)
        assert loss.has_bec and model.compute_born
        # Without the term the model is left as it was.
        other = build()
        training.get_loss({"type": LOSS_TYPE}, START_LR, 2, other)
        assert not other.compute_born

    def test_model_without_charges_refused(self):
        with pytest.raises(ValueError, match="long_range"):
            training.get_loss(dict(self.PARAMS), START_LR, 2, build(long_range=False))


def distinct(model, seed: int):
    """``model`` with its charge head moved off its initial weights."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.charge_fitting.parameters():
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float64
            )
            parameter.add_(0.3 * noise.to(parameter))
    return model


@pytest.mark.parametrize("descriptor", ["dpa4c", "dpa4"])
def test_steps_on_the_term_bring_a_student_to_its_teacher(descriptor):
    """Only the charge head differs between the two models and only the Born
    term has weight, so what falls is the supervision at work."""
    make = build if descriptor == "dpa4c" else build_dpa4
    coord, atype, cell = frame()
    teacher = distinct(make(), seed=1)
    teacher.compute_born = True
    target = teacher(coord, atype, box=cell)
    student = distinct(make(), seed=2).train()
    student.compute_born = True
    student.atomic_model.requires_grad_(False)
    loss = LongRangeEnergyLoss(
        starter_learning_rate=START_LR,
        **PENALTY_ONLY,
        start_pref_bec=1.0,
        limit_pref_bec=1.0,
    )
    nf, nloc = atype.shape
    label = {
        "energy": target["energy"].detach(),
        "find_energy": 1.0,
        BORN_KEY: target[BORN_KEY].detach().reshape(nf, -1),
        f"find_{BORN_KEY}": 1.0,
    }
    parameters = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=3e-3)
    history = []
    for _ in range(60):
        optimizer.zero_grad()
        out = student(coord, atype, box=cell)
        value, more = loss.call(START_LR, nloc, out, label)
        value.backward()
        assert all(p.grad is not None for p in parameters)
        optimizer.step()
        history.append(float(more["rmse_bec"]))
    assert history[0] > 1e-3
    assert history[-1] < 0.3 * history[0]


@pytest.mark.parametrize("charge_head_only", [False, True])
def test_cli_training_with_and_without_the_label(tmp_path, charge_head_only):
    """One system with ``born_charge.npy`` and one without, through
    ``dp train``; the log carries the Born error."""
    bare = write_dataset(tmp_path)
    (tmp_path / "with").mkdir()
    labelled_system = write_dataset(tmp_path / "with")
    n_frame = np.load(labelled_system / "set.000/coord.npy").shape[0]
    n_atom = len((labelled_system / "type.raw").read_text().split())
    rng = np.random.default_rng(1)
    np.save(
        labelled_system / f"set.000/{BORN_KEY}.npy",
        0.3 * rng.normal(size=(n_frame, n_atom * 9)),
    )
    cfg = {
        "model": long_range("dpa4c", charge_head_only),
        "learning_rate": {"start_lr": 1e-3, "stop_lr": 1e-4, "decay_steps": 1},
        "loss": {
            "type": LOSS_TYPE,
            "start_pref_e": 1.0,
            "limit_pref_e": 1.0,
            "start_pref_f": 1.0,
            "limit_pref_f": 1.0,
            "start_pref_bec": 10.0,
            "limit_pref_bec": 10.0,
            "bec_scale": 1.5,
        },
        "training": {
            "training_data": {
                "systems": [str(bare), str(labelled_system)],
                "batch_size": 1,
            },
            "validation_data": {
                "systems": [str(labelled_system)],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": 6,
            "disp_freq": 1,
            "save_freq": 6,
            "seed": 1,
        },
    }
    (tmp_path / "input.json").write_text(json.dumps(cfg))
    dp_train("input.json", cwd=tmp_path)
    header, *rows = [
        line.split()
        for line in (tmp_path / "lcurve.out").read_text().splitlines()
        if not line.startswith("# If")
    ]
    column = header.index("rmse_bec_val") - 1  # the header starts with "#"
    values = np.array([float(row[column]) for row in rows])
    assert np.isfinite(values).all() and (values > 0).all()
