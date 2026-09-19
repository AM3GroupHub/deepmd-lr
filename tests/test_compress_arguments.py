"""Compression switches and numeric strides have distinct contracts."""

import os

import pytest

from deepmd_lr.compress import compress_model
from deepmd_lr.model import lr_model

from .test_compress import build32


@pytest.mark.parametrize(
    "value", [0, 0.0, -0.01, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_numeric_stride_is_rejected(value):
    with pytest.raises(ValueError, match="positive and finite"):
        compress_model(None, value)


@pytest.mark.parametrize("value", [None, "0.01", [], {}])
def test_non_numeric_stride_is_rejected(value):
    with pytest.raises(TypeError, match="bool or"):
        compress_model(None, value)


def test_false_true_and_positive_stride(monkeypatch):
    model = build32()
    calls = []
    monkeypatch.setattr(
        model, "enable_compression", lambda **kwargs: calls.append(kwargs)
    )
    assert compress_model(model, False) is model
    assert calls == []
    compress_model(model, True)
    compress_model(model, 0.002)
    assert calls == [
        {"charge_spin": None},
        {"table_stride_1": 0.002, "charge_spin": None},
    ]


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_direct_model_api_rejects_bad_stride_before_mutation(monkeypatch, value):
    monkeypatch.setenv("DP_CUDA_INFER", "0")
    model = build32()
    with pytest.raises(ValueError, match="positive and finite"):
        model.enable_compression(table_stride_1=value)
    assert not model.atomic_model.descriptor.compress
    assert os.environ["DP_CUDA_INFER"] == "0"


@pytest.mark.parametrize("long_range", [False, True])
def test_level_two_is_rejected_for_both_model_types(monkeypatch, long_range):
    monkeypatch.setattr(lr_model, "_kernel_available", lambda: True)
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    model = build32(long_range=long_range)
    with pytest.raises(RuntimeError, match="DP_CUDA_INFER=2"):
        model.enable_compression()
    assert not model.atomic_model.descriptor.compress
