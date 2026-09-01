"""Regression tests for staged checkpoint grafting."""

import pytest
import torch

from scripts.graft_short_range import (
    DEFAULT_CHARGE_SCALE,
    _scale_charge_head_output,
)

from .test_joint_model import build, frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)


def test_charge_output_scale_rejects_exact_zero():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        _scale_charge_head_output({}, 0.0)


def test_scaled_charge_head_starts_small_but_remains_trainable():
    model = build()
    state = model.state_dict()
    changed = _scale_charge_head_output(state, DEFAULT_CHARGE_SCALE)
    assert changed > 0
    model.load_state_dict(state)
    model.train()

    coord, atype, cell = frame()
    out = model(coord, atype, box=cell)
    assert 0.0 < float(out["charge"].abs().max().detach()) < 0.1

    out["energy_long_range"].sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.charge_fitting.parameters()
        if parameter.requires_grad
    ]
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in gradients
    )
