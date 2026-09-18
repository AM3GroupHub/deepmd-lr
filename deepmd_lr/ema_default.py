# SPDX-License-Identifier: LGPL-3.0-or-later
"""EMA on by default for ``dp_lr`` training.

Why
---
The long-range model trains two heads through one descriptor, and the force
loss is weighted a thousand times the energy loss at the start of DeePMD-kit's
default schedule (``start_pref_f 1000 -> limit_pref_f 1``). As the weight moves
towards energy late in training, the force-shaped parameters are free to drift
on a small, noisy gradient. An exponential moving average of the parameters --
``ema = decay * ema + (1 - decay) * param`` after every optimizer step -- is the
cheapest defence: it costs one extra copy of the parameters and no compute
worth measuring, and the averaged weights are what a production model should
carry. DeePMD-kit 3.2.0 implements it natively (``training.enable_ema``,
``training.ema_decay``, periodic ``*_ema.ckpt-<step>.pt`` snapshots), but it is
off unless asked for. For ``dp_lr`` configurations this module asks for it.

What it does
------------
:func:`apply_ema_default` sets ``training.enable_ema = True`` and
``training.ema_decay = 0.999`` **only when**

* the model is ``dp_lr`` (or any ``model_dict`` branch is), and
* the user did not write ``enable_ema`` themselves (an explicit ``false`` is
  respected -- this is a default, not a policy), and
* ``training.zero_stage`` is below 2, the only regime DeePMD-kit's EMA supports.

Where it is applied matters, and it is the reason this module hooks the
entrypoint rather than the trainer alone. ``dp train`` runs
``AbstractTrainEntrypoint.normalize_config`` before it builds the trainer, and
DeePMD-kit's argument normalization *fills in every declared default* -- so by
the time the trainer sees the config, ``training.enable_ema`` is already
present and ``False``, and "the user did not write it" can no longer be told
from "the user wrote false". :func:`install` therefore wraps
``normalize_config``: it reads the raw config on the way in, remembers whether
the key was written, and applies the default to the normalized config on the
way out. ``Trainer.__init__`` stays wrapped as well, for configs built in
Python and passed straight to the trainer. Both wrappers are pass-throughs when
no ``dp_lr`` model is present, so a plain DeePMD-kit run is byte-unchanged.

What to know
------------
* The EMA snapshot (``model_ema.ckpt.pt``) holds averaged parameters and the
  live persistent buffers (``out_bias``, ``out_std``, descriptor statistics),
  with neither optimizer nor EMA state. It is a deployment snapshot; use it for
  MD and evaluation, and use the live ``model.ckpt.pt`` for ``--restart``.
* ``--init-model`` starts a new run and a new average. A staged schedule
  therefore re-accumulates the EMA inside every stage; ``--restart`` continues
  it.
* The memory window is about ``1 / (1 - decay)`` steps: 0.999 averages the last
  ~1000 steps. Keep that below a few percent of the run (or of each stage). For
  runs shorter than ~2000 steps set ``"enable_ema": false`` -- after 200 steps
  the snapshot is still 82 % initial weights.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EMA_DEFAULTS",
    "apply_ema_default",
    "apply_to_normalized",
    "has_long_range_model",
    "install",
]

#: The defaults this module writes when the user wrote nothing.
EMA_DEFAULTS: dict[str, Any] = {"enable_ema": True, "ema_decay": 0.999}

_MODEL_TYPE = "dp_lr"
# DeePMD-kit: "training.enable_ema currently only supports training.zero_stage < 2".
_ZERO_STAGE_LIMIT = 2


def has_long_range_model(model_config: Any) -> bool:
    """True when the model, or any ``model_dict`` branch, is ``dp_lr``."""
    if not isinstance(model_config, dict):
        return False
    if model_config.get("type") == _MODEL_TYPE:
        return True
    branches = model_config.get("model_dict")
    if not isinstance(branches, dict):
        return False
    return any(
        isinstance(branch, dict) and branch.get("type") == _MODEL_TYPE
        for branch in branches.values()
    )


def apply_ema_default(config: dict[str, Any]) -> dict[str, Any]:
    """Fill in EMA defaults for a ``dp_lr`` training config, in place.

    Returns the same mapping for convenience. A missing ``training`` section is
    left alone: that is DeePMD-kit's problem to report, not ours to invent.
    """
    if not has_long_range_model(config.get("model")):
        return config
    training = config.get("training")
    if not isinstance(training, dict):
        return config
    if "enable_ema" in training:
        # An explicit choice, either way, wins. Do not touch ema_decay either:
        # a user who wrote enable_ema wrote what they meant.
        return config
    if int(training.get("zero_stage", 0) or 0) >= _ZERO_STAGE_LIMIT:
        return config
    training["enable_ema"] = EMA_DEFAULTS["enable_ema"]
    training.setdefault("ema_decay", EMA_DEFAULTS["ema_decay"])
    return config


def apply_to_normalized(raw: Any, normalized: Any) -> Any:
    """Apply the default to a normalized config, deciding from the raw one.

    Argcheck normalization inserts ``enable_ema: false`` into every config, so the
    normalized config alone cannot say whether the user chose it. ``raw`` is the
    config as written; only when it is silent *and* the model is ``dp_lr`` is the
    inserted default replaced. Any other config is returned untouched.
    """
    if not isinstance(normalized, dict) or not isinstance(
        normalized.get("training"), dict
    ):
        return normalized
    if not has_long_range_model(normalized.get("model")):
        return normalized
    written = (
        isinstance(raw, dict)
        and isinstance(raw.get("training"), dict)
        and "enable_ema" in raw["training"]
    )
    if written:
        return normalized
    normalized["training"].pop("enable_ema", None)
    apply_ema_default(normalized)
    return normalized


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
            return apply_to_normalized(
                config, current_normalize(self, config, **kwargs)
            )

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
