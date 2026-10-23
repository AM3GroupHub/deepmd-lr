# SPDX-License-Identifier: MIT
"""Enforce EMA for training when the deepmd-lr plugin is loaded.

Ordinary and long-range models always use ``enable_ema=True``, even if the
input explicitly disables it. ``ema_decay`` defaults to 0.999; a user-specified
decay is preserved. ``zero_stage >= 2`` is rejected because DeePMD-kit does not
support it with EMA.

The CLI hook applies defaults to a copy of the raw configuration before
normalization inserts upstream defaults. Consequently validation, the saved
normalized configuration (normally ``out.json``), and training agree. The
original input file is not rewritten. A Trainer hook also supports direct
Python callers that supply an unnormalized configuration.

EMA deployment snapshots and restartable live checkpoints remain distinct;
this module does not select which weights downstream evaluation uses.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "EMA_DEFAULTS",
    "apply_ema_default",
    "install",
]

#: Required enable flag and default decay.
EMA_DEFAULTS: dict[str, Any] = {"enable_ema": True, "ema_decay": 0.999}

# DeePMD-kit: "training.enable_ema currently only supports training.zero_stage < 2".
_ZERO_STAGE_LIMIT = 2


def apply_ema_default(config: dict[str, Any]) -> dict[str, Any]:
    """Fill in EMA defaults for a training config, in place.

    Returns the same mapping for convenience. A missing ``training`` section is
    left alone: that is DeePMD-kit's problem to report, not ours to invent.
    """
    training = config.get("training")
    if not isinstance(training, dict):
        return config
    if int(training.get("zero_stage", 0) or 0) >= _ZERO_STAGE_LIMIT:
        raise ValueError(
            "deepmd-lr requires EMA for all training; "
            "training.zero_stage must be below 2 for EMA support"
        )
    training["enable_ema"] = EMA_DEFAULTS["enable_ema"]
    training.setdefault("ema_decay", EMA_DEFAULTS["ema_decay"])
    return config


def install() -> None:
    """Wrap the entrypoint's normalization and the trainer. Idempotent.

    ``normalize_config`` is the seam that matters: it is the last place the raw,
    user-written config is visible before argcheck fills in every default.
    """
    from deepmd.dpmodel.train import entrypoint as _entrypoint
    from deepmd.pt_expt.train import training as _training

    current_normalize = _entrypoint.AbstractTrainEntrypoint.normalize_config
    if not getattr(current_normalize, "_deepmd_lr_ema_default", False):

        def normalize_config(self: Any, config: Any, **kwargs: Any) -> Any:
            if isinstance(config, dict):
                config = apply_ema_default(copy.deepcopy(config))
            return current_normalize(self, config, **kwargs)

        normalize_config._deepmd_lr_ema_default = True  # type: ignore[attr-defined]
        normalize_config.__doc__ = current_normalize.__doc__
        _entrypoint.AbstractTrainEntrypoint.normalize_config = normalize_config

    current_init = _training.Trainer.__init__
    if getattr(current_init, "_deepmd_lr_ema_default", False):
        return

    def __init__(self: Any, config: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(config, dict):
            apply_ema_default(config)
        current_init(self, config, *args, **kwargs)

    __init__._deepmd_lr_ema_default = True  # type: ignore[attr-defined]
    __init__.__doc__ = current_init.__doc__
    _training.Trainer.__init__ = __init__
