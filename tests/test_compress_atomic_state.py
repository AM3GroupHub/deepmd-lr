"""Failed compression preserves the model and permits a clean retry."""

import os

import pytest
import torch

from deepmd_lr.model import lr_model

from .test_compress import build32


@pytest.fixture(autouse=True)
def available_operator(monkeypatch):
    # Table construction uses PyTorch; CUDA dispatch is outside these tests.
    monkeypatch.setattr(lr_model, "_kernel_available", lambda: True)


@pytest.mark.parametrize("level", [None, "0", "1"])
@pytest.mark.parametrize("condition", [[1, 2, 3], [float("nan"), 1], [0.5, 1]])
def test_invalid_condition_leaves_state_unchanged(monkeypatch, level, condition):
    if level is None:
        monkeypatch.delenv("DP_CUDA_INFER", raising=False)
    else:
        monkeypatch.setenv("DP_CUDA_INFER", level)
    model = build32(chg_ebd=True)
    descriptor = model.atomic_model.descriptor
    before = descriptor.state_dict()
    with pytest.raises(ValueError):
        model.enable_compression(charge_spin=condition)
    assert not descriptor.compress
    assert getattr(model, "_compressed_charge_spin", None) is None
    assert os.environ.get("DP_CUDA_INFER") == level
    assert descriptor.state_dict().keys() == before.keys()
    for key, value in before.items():
        torch.testing.assert_close(descriptor.state_dict()[key], value)
    model.enable_compression(charge_spin=[0, 1])
    assert descriptor.compress
    assert model._compressed_charge_spin == (0, 1)


@pytest.mark.parametrize("method", ["enable_compression", "apply_charge_state"])
def test_partial_failure_restores_buffers_and_allows_retry(monkeypatch, method):
    monkeypatch.delenv("DP_CUDA_INFER", raising=False)
    model = build32(chg_ebd=True)
    descriptor = model.atomic_model.descriptor
    original = getattr(type(descriptor), method)
    buffers = descriptor._buffers.copy()
    had_scalars = hasattr(descriptor, "_compression_scalars")
    scalars = getattr(descriptor, "_compression_scalars", None)

    def fail_after_mutation(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected post-mutation failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(descriptor), method, fail_after_mutation)
        with pytest.raises(RuntimeError, match="injected"):
            model.enable_compression(charge_spin=[0, 1])
    assert model.atomic_model.descriptor is descriptor
    assert not descriptor.compress
    assert descriptor._buffers.keys() == buffers.keys()
    assert all(descriptor._buffers[key] is value for key, value in buffers.items())
    assert hasattr(descriptor, "_compression_scalars") == had_scalars
    assert getattr(descriptor, "_compression_scalars", None) == scalars
    assert "DP_CUDA_INFER" not in os.environ
    model.enable_compression(charge_spin=[0, 1])
    assert descriptor.compress
    with pytest.raises(ValueError, match="already"):
        model.enable_compression(charge_spin=[0, 1])
    assert descriptor.compress
    assert model._compressed_charge_spin == (0, 1)
