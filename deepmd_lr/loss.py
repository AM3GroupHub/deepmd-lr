# SPDX-License-Identifier: MIT
"""The ``ener_lr`` loss: DeePMD-kit's energy loss plus a charge-neutrality term.

Why the term exists
-------------------
The charge head's raw charges are projected onto the frame's total charge
by adding one constant to every atom (``project_to_total_charge``). Any
constant in the raw charges is therefore invisible to the energy, and a
training frame of a single material leaves that material's raw offset free.
A trained C/Si model came out with raw charges of +0.84963 e in bulk diamond
and +0.84664 e in bulk silicon: harmless in each bulk, but at an interface
the projection subtracts their common mean, leaving every C atom slightly
positive and every Si atom slightly negative. That separation grows with
the thickness of the materials, not with the interface area, and the
long-range forces it drives broke MD on the longer cells.

The penalty pins the free constant instead of leaving it to chance. With
``delta_f = (Q_f - sum_i raw_i) / N_f``, the shift the projection applied
to every atom of frame ``f`` (the model's ``charge_projection_shift``), the
term is

    pref_qn * mean_f delta_f^2          (loss_func "mse")
    pref_qn * mean_f |delta_f|          (loss_func "mae")

so the raw charges are driven to satisfy the constraint by themselves and
the projection becomes a small correction. Frames of each material then fix
that material's offset near zero, and offsets can no longer differ between
materials by more than the term allows. Nothing in the architecture changes,
and inference is untouched: the projection still enforces the total charge
exactly.

``pref_qn`` follows DeePMD-kit's schedule, ``limit + (start - limit) *
lr / start_lr``. The term is dimensionless per atom and needs no label.
"""

from __future__ import annotations

from typing import Any

from deepmd.dpmodel.loss.ener import EnergyLoss

__all__ = ["LOSS_TYPE", "LongRangeEnergyLoss", "install"]

#: The loss ``type`` that selects this class.
LOSS_TYPE = "ener_lr"

#: The model output the term reads, ``(nf,)``.
SHIFT_KEY = "charge_projection_shift"


class LongRangeEnergyLoss(EnergyLoss):
    """:class:`EnergyLoss` plus ``pref_qn`` times the squared (or absolute)
    shift the neutrality projection applied; see the module docstring.

    Parameters
    ----------
    start_pref_qn, limit_pref_qn
        The prefactor of the charge-neutrality term at the start and at the
        end of training. Both zero leaves exactly the ``ener`` loss.
    **kwargs
        Everything :class:`EnergyLoss` takes.
    """

    def __init__(
        self,
        *args: Any,
        start_pref_qn: float = 0.0,
        limit_pref_qn: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if start_pref_qn < 0 or limit_pref_qn < 0:
            raise ValueError("start_pref_qn and limit_pref_qn must be >= 0")
        self.start_pref_qn = float(start_pref_qn)
        self.limit_pref_qn = float(limit_pref_qn)
        self.has_qn = self.start_pref_qn != 0.0 or self.limit_pref_qn != 0.0

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mae: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        loss, more_loss = super().call(
            learning_rate, natoms, model_dict, label_dict, mae=mae
        )
        if not self.has_qn:
            return loss, more_loss
        if SHIFT_KEY not in model_dict:
            raise ValueError(
                f"loss '{LOSS_TYPE}' with a charge-neutrality prefactor needs a "
                "'dp_lr' model with a `long_range` block: this model reports "
                f"no `{SHIFT_KEY}`, so it has no charges to constrain"
            )
        import array_api_compat

        shift = model_dict[SHIFT_KEY]
        xp = array_api_compat.array_namespace(shift)
        shift = xp.astype(xp.reshape(shift, (-1,)), loss.dtype)
        lr_ratio = learning_rate / self.starter_learning_rate
        pref = self.limit_pref_qn + (self.start_pref_qn - self.limit_pref_qn) * lr_ratio
        l2 = xp.mean(shift * shift)
        l1 = xp.mean(xp.abs(shift))
        loss = loss + pref * (l1 if self.loss_func == "mae" else l2)
        more_loss["rmse_qn"] = xp.sqrt(l2)
        if mae or self.loss_func == "mae":
            more_loss["mae_qn"] = l1
        # Upstream's aggregate is sqrt of the whole objective; keep it so.
        more_loss["rmse"] = xp.sqrt(loss)
        self.l2_l = loss
        self.l2_more = more_loss
        return loss, more_loss

    def serialize(self) -> dict:
        data = super().serialize()
        data["@class"] = "LongRangeEnergyLoss"
        data["start_pref_qn"] = self.start_pref_qn
        data["limit_pref_qn"] = self.limit_pref_qn
        return data

    @classmethod
    def deserialize(cls, data: dict) -> LongRangeEnergyLoss:
        data = data.copy()
        prefs = {
            "start_pref_qn": data.pop("start_pref_qn", 0.0),
            "limit_pref_qn": data.pop("limit_pref_qn", 0.0),
        }
        data["@class"] = "EnergyLoss"
        base = EnergyLoss.deserialize(data).serialize()
        base.pop("@class"), base.pop("@version")
        return cls(**base, **prefs)


def install() -> None:
    """Let ``pt_expt``'s loss factory build ``ener_lr``. Idempotent.

    Upstream's ``get_loss`` is a fixed if/elif over its own types, read from
    the training module at call time by the trainer and by ``dp change-bias``,
    so wrapping the module attribute reaches both.
    """
    from deepmd.pt_expt.train import training as _training

    current = _training.get_loss
    if getattr(current, "_deepmd_lr_patched", False):
        return

    def get_loss(
        loss_params: dict[str, Any], start_lr: float, _ntypes: int, _model: Any
    ) -> Any:
        if loss_params.get("type") == LOSS_TYPE:
            params = dict(loss_params)
            params.pop("type")
            params["starter_learning_rate"] = start_lr
            return LongRangeEnergyLoss(**params)
        return current(loss_params, start_lr, _ntypes, _model)

    get_loss._deepmd_lr_patched = True  # type: ignore[attr-defined]
    get_loss.__wrapped__ = current  # type: ignore[attr-defined]
    get_loss.__doc__ = current.__doc__
    _training.get_loss = get_loss
