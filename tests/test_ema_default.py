# SPDX-License-Identifier: LGPL-3.0-or-later
"""EMA is on by default for dp_lr training, and only then."""

import copy
import subprocess
import sys

from deepmd_lr.ema_default import (
    EMA_DEFAULTS,
    apply_ema_default,
    apply_to_normalized,
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


def test_sets_defaults_only_for_dp_lr():
    single = apply_ema_default(copy.deepcopy(SINGLE))
    assert single["training"]["enable_ema"] is True
    assert single["training"]["ema_decay"] == EMA_DEFAULTS["ema_decay"] == 0.999
    multi = apply_ema_default(copy.deepcopy(MULTI))
    assert multi["training"]["enable_ema"] is True
    plain = apply_ema_default(copy.deepcopy(PLAIN))
    assert (
        "enable_ema" not in plain["training"] and "ema_decay" not in plain["training"]
    )


def test_explicit_user_choice_wins_either_way():
    off = copy.deepcopy(SINGLE)
    off["training"]["enable_ema"] = False
    assert apply_ema_default(off)["training"] == {
        "numb_steps": 1000,
        "enable_ema": False,
    }
    on = copy.deepcopy(SINGLE)
    on["training"].update(enable_ema=True, ema_decay=0.99)
    assert apply_ema_default(on)["training"]["ema_decay"] == 0.99


def test_zero_stage_two_or_more_is_left_alone():
    cfg = copy.deepcopy(SINGLE)
    cfg["training"]["zero_stage"] = 2
    assert "enable_ema" not in apply_ema_default(cfg)["training"]


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


def normalized(config):
    """What argcheck does to every config: fill in the declared defaults."""
    out = copy.deepcopy(config)
    out["training"].setdefault("enable_ema", False)
    out["training"].setdefault("ema_decay", 0.999)
    return out


def test_normalization_default_is_not_mistaken_for_a_user_choice():
    # `dp train` normalizes before building the trainer, so by then enable_ema is
    # already present and False. Deciding from the raw config is the whole point.
    silent = apply_to_normalized(SINGLE, normalized(SINGLE))
    assert silent["training"]["enable_ema"] is True
    assert silent["training"]["ema_decay"] == EMA_DEFAULTS["ema_decay"]

    off = copy.deepcopy(SINGLE)
    off["training"]["enable_ema"] = False
    assert apply_to_normalized(off, normalized(off))["training"]["enable_ema"] is False

    on = copy.deepcopy(SINGLE)
    on["training"].update(enable_ema=True, ema_decay=0.99)
    assert apply_to_normalized(on, normalized(on))["training"]["ema_decay"] == 0.99

    # a non-dp_lr config is returned exactly as the normalizer produced it
    assert apply_to_normalized(PLAIN, normalized(PLAIN)) == normalized(PLAIN)


def test_entrypoint_normalize_config_is_wrapped_by_the_entry_point():
    code = """
import deepmd.pt_expt  # noqa: F401
from deepmd.dpmodel.train.entrypoint import AbstractTrainEntrypoint
hook = AbstractTrainEntrypoint.normalize_config
assert getattr(hook, "_deepmd_lr_ema_default", False)
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
