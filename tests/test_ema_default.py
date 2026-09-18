# SPDX-License-Identifier: LGPL-3.0-or-later
"""EMA is required for ordinary and long-range training."""

import copy
import subprocess
import sys

import pytest

from deepmd_lr.ema_default import (
    EMA_DEFAULTS,
    apply_ema_default,
    has_long_range_model,
)

SINGLE = {
    "model": {
        "type": "dp_lr",
        "type_map": ["H", "O"],
        "descriptor": {},
        "fitting_net": {},
        "long_range": {},
    },
    "training": {"numb_steps": 1000},
}
MULTI = {
    "model": {
        "shared_dict": {"type_map_all": ["H", "O"]},
        "model_dict": {
            "bulk": {"type": "dp_lr", "type_map": "type_map_all"},
            "slab": {"same_as": "bulk", "long_range": {"reciprocal": {"alpha": 0.4}}},
        },
    },
    "training": {"numb_steps": 1000},
}
PLAIN = {
    "model": {"type": "standard", "type_map": ["H", "O"]},
    "training": {"numb_steps": 1000},
}


def test_detects_dp_lr_in_single_and_multi_branch_configs():
    assert has_long_range_model(SINGLE["model"])
    assert has_long_range_model(MULTI["model"])
    assert not has_long_range_model(PLAIN["model"])
    assert not has_long_range_model(None)


def test_sets_defaults_for_ordinary_and_lr_models():
    single = apply_ema_default(copy.deepcopy(SINGLE))
    assert single["training"]["enable_ema"] is True
    assert single["training"]["ema_decay"] == EMA_DEFAULTS["ema_decay"] == 0.999
    multi = apply_ema_default(copy.deepcopy(MULTI))
    assert multi["training"]["enable_ema"] is True
    plain = apply_ema_default(copy.deepcopy(PLAIN))
    assert plain["training"]["enable_ema"] is True
    assert plain["training"]["ema_decay"] == 0.999


def test_explicit_disable_is_overridden_and_custom_decay_is_preserved():
    off = copy.deepcopy(SINGLE)
    off["training"]["enable_ema"] = False
    assert apply_ema_default(off)["training"] == {
        "numb_steps": 1000,
        "enable_ema": True,
        "ema_decay": 0.999,
    }
    on = copy.deepcopy(SINGLE)
    on["training"].update(enable_ema=True, ema_decay=0.99)
    assert apply_ema_default(on)["training"]["ema_decay"] == 0.99


def test_zero_stage_two_or_more_is_rejected():
    cfg = copy.deepcopy(SINGLE)
    cfg["training"]["zero_stage"] = 2
    with pytest.raises(ValueError, match="zero_stage must be below 2"):
        apply_ema_default(cfg)


def test_missing_training_section_is_not_invented():
    cfg = {"model": copy.deepcopy(SINGLE["model"])}
    assert apply_ema_default(cfg) == cfg and "training" not in cfg


def test_hook_is_installed_once_by_the_entry_point():
    # Fresh interpreter: importing the backend fires deepmd_lr.register -> install().
    code = """
# entry point -> deepmd_lr.register -> ema_default.install()
import deepmd.pt_expt  # noqa: F401
from deepmd.pt_expt.train import training
from deepmd_lr import ema_default
first = training.Trainer.__init__
assert getattr(first, "_deepmd_lr_ema_default", False), "not installed"
ema_default.install()
assert training.Trainer.__init__ is first, "not idempotent"
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_cli_normalization_and_saved_config(tmp_path):
    """Use the actual pt_expt CLI normalization and output writer, without training."""
    import json

    from deepmd.pt_expt.entrypoints.main import PTExptTrainEntrypoint

    from .test_plugin_registration import BASE_MODEL_CONFIG

    entrypoint = PTExptTrainEntrypoint()
    for model_type in ("dp_lr", "standard"):
        for override in (
            {},
            {"enable_ema": False},
            {"ema_decay": 0.98},
            {"enable_ema": True, "ema_decay": 0.97},
            {"zero_stage": 2},
        ):
            for multi in (False, True):
                model = {"type": model_type, **copy.deepcopy(BASE_MODEL_CONFIG)}
                data = {"training_data": {"systems": ["./unused"], "batch_size": 1}}
                raw = {
                    "model": {"model_dict": {"head": model}} if multi else model,
                    "learning_rate": {"type": "exp", "start_lr": 1e-3, "stop_lr": 1e-5},
                    "loss_dict" if multi else "loss": {"head": {"type": "ener"}}
                    if multi
                    else {"type": "ener"},
                    "training": {
                        "numb_steps": 1,
                        **override,
                        **({"data_dict": {"head": data}} if multi else data),
                    },
                }
                original = copy.deepcopy(raw)
                if override.get("zero_stage", 0) >= 2:
                    with pytest.raises(ValueError, match="zero_stage must be below 2"):
                        entrypoint.normalize_config(raw, multi_task=multi)
                    assert raw == original
                    continue
                normalized = entrypoint.normalize_config(raw, multi_task=multi)
                assert raw == original
                assert normalized["training"]["enable_ema"] is True
                assert normalized["training"]["ema_decay"] == override.get(
                    "ema_decay", 0.999
                )
                output = tmp_path / "out.json"
                entrypoint.dump_config(normalized, str(output))
                assert (
                    json.loads(output.read_text())["training"] == normalized["training"]
                )
                again = entrypoint.normalize_config(normalized, multi_task=multi)
                assert again["training"] == normalized["training"]


def test_normalization_hook_is_idempotent():
    from deepmd.dpmodel.train.entrypoint import AbstractTrainEntrypoint

    from deepmd_lr.ema_default import install

    hook = AbstractTrainEntrypoint.normalize_config
    assert getattr(hook, "_deepmd_lr_ema_default", False)
    install()
    assert AbstractTrainEntrypoint.normalize_config is hook
