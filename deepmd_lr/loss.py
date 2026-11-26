# SPDX-License-Identifier: MIT
"""The ``ener_lr`` loss: DeePMD-kit's energy loss plus a charge-neutrality
term and, optionally, a term on labelled Born effective charges.

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

The Born term
-------------
Energies and forces see the charges only through an energy quadratic in
them, which fixes neither their sign nor much of how they respond to a
displacement. ``pref_bec`` adds, on frames that carry a ``born_charge``
label (``born_charge.npy``, nine numbers per atom, ``[polarization,
displacement]``, in e),

    pref_bec * mean (s Z_model - Z_label)^2        (loss_func "mse")
    pref_bec * mean |s Z_model - Z_label|          (loss_func "mae")

over every atom and component, where ``Z_model`` is the model's own Born
tensor (:mod:`deepmd_lr.born`) and ``s = bec_scale`` the factor between
model and reference charges, ``sqrt(eps_inf)``; it is a setting, not fitted,
so the term fixes the sign and the size of the charges as well as their
response. Frames of a system without the label contribute nothing. The
tensor is differentiated once more for the gradient, as the forces are.
"""

from __future__ import annotations

from typing import Any

from deepmd.dpmodel.loss.ener import EnergyLoss
from deepmd.utils.data import DataRequirementItem

__all__ = ["LOSS_TYPE", "LongRangeEnergyLoss", "install"]

#: The loss ``type`` that selects this class.
LOSS_TYPE = "ener_lr"

#: The model output the term reads, ``(nf,)``.
SHIFT_KEY = "charge_projection_shift"

#: The Born tensors: the model's output and the label of the same name,
#: ``(nf, nloc, 3, 3)`` and ``(nf, nloc * 9)``.
BORN_KEY = "born_charge"


class LongRangeEnergyLoss(EnergyLoss):
    """:class:`EnergyLoss` plus ``pref_qn`` times the squared (or absolute)
    shift the neutrality projection applied; see the module docstring.

    Parameters
    ----------
    start_pref_qn, limit_pref_qn
        The prefactor of the charge-neutrality term at the start and at the
        end of training. Both zero leaves exactly the ``ener`` loss.
    start_pref_bec, limit_pref_bec
        The prefactor of the Born term at the start and at the end of
        training. Both zero leaves it out.
    bec_scale
        The factor from the model's Born tensors to the labels',
        ``sqrt(eps_inf)``.
    **kwargs
        Everything :class:`EnergyLoss` takes.
    """

    def __init__(
        self,
        *args: Any,
        start_pref_qn: float = 0.0,
        limit_pref_qn: float = 0.0,
        start_pref_bec: float = 0.0,
        limit_pref_bec: float = 0.0,
        bec_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if start_pref_qn < 0 or limit_pref_qn < 0:
            raise ValueError("start_pref_qn and limit_pref_qn must be >= 0")
        if start_pref_bec < 0 or limit_pref_bec < 0:
            raise ValueError("start_pref_bec and limit_pref_bec must be >= 0")
        if not bec_scale > 0:
            raise ValueError(f"bec_scale must be positive, got {bec_scale!r}")
        self.start_pref_qn = float(start_pref_qn)
        self.limit_pref_qn = float(limit_pref_qn)
        self.has_qn = self.start_pref_qn != 0.0 or self.limit_pref_qn != 0.0
        self.start_pref_bec = float(start_pref_bec)
        self.limit_pref_bec = float(limit_pref_bec)
        self.bec_scale = float(bec_scale)
        self.has_bec = self.start_pref_bec != 0.0 or self.limit_pref_bec != 0.0

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        requirement = list(super().label_requirement)
        if self.has_bec:
            requirement.append(
                DataRequirementItem(
                    BORN_KEY, ndof=9, atomic=True, must=False, high_prec=False
                )
            )
        return requirement

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
        if not (self.has_qn or self.has_bec):
            return loss, more_loss
        import array_api_compat

        lr_ratio = learning_rate / self.starter_learning_rate

        def scheduled(start: float, limit: float) -> float:
            """A prefactor on the learning-rate schedule, as upstream's."""
            return limit + (start - limit) * lr_ratio

        absolute = self.loss_func == "mae"
        if self.has_qn:
            if SHIFT_KEY not in model_dict:
                raise ValueError(
                    f"loss '{LOSS_TYPE}' with a charge-neutrality prefactor needs "
                    "a 'dp_lr' model with a `long_range` block: this model "
                    f"reports no `{SHIFT_KEY}`, so it has no charges to constrain"
                )
            shift = model_dict[SHIFT_KEY]
            xp = array_api_compat.array_namespace(shift)
            shift = xp.astype(xp.reshape(shift, (-1,)), loss.dtype)
            pref = scheduled(self.start_pref_qn, self.limit_pref_qn)
            l2, l1 = xp.mean(shift * shift), xp.mean(xp.abs(shift))
            loss = loss + pref * (l1 if absolute else l2)
            more_loss["rmse_qn"] = xp.sqrt(l2)
            if mae or absolute:
                more_loss["mae_qn"] = l1
        if self.has_bec:
            if BORN_KEY not in model_dict:
                raise ValueError(
                    f"loss '{LOSS_TYPE}' with a Born prefactor needs a 'dp_lr' "
                    "model with a `long_range` block that computes its Born "
                    f"tensors: this model reports no `{BORN_KEY}`"
                )
            born = model_dict[BORN_KEY]
            xp = array_api_compat.array_namespace(born)
            found = label_dict[f"find_{BORN_KEY}"]
            label = xp.reshape(label_dict[BORN_KEY], born.shape)
            diff = xp.astype(self.bec_scale * born - label, loss.dtype)
            pref = scheduled(self.start_pref_bec, self.limit_pref_bec)
            l2, l1 = xp.mean(diff * diff), xp.mean(xp.abs(diff))
            loss = loss + found * pref * (l1 if absolute else l2)
            more_loss["rmse_bec"] = self.display_if_exist(xp.sqrt(l2), found)
            if mae or absolute:
                more_loss["mae_bec"] = self.display_if_exist(l1, found)
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
        data["start_pref_bec"] = self.start_pref_bec
        data["limit_pref_bec"] = self.limit_pref_bec
        data["bec_scale"] = self.bec_scale
        return data

    @classmethod
    def deserialize(cls, data: dict) -> LongRangeEnergyLoss:
        data = data.copy()
        prefs = {
            "start_pref_qn": data.pop("start_pref_qn", 0.0),
            "limit_pref_qn": data.pop("limit_pref_qn", 0.0),
            "start_pref_bec": data.pop("start_pref_bec", 0.0),
            "limit_pref_bec": data.pop("limit_pref_bec", 0.0),
            "bec_scale": data.pop("bec_scale", 1.0),
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
            loss = LongRangeEnergyLoss(**params)
            if loss.has_bec:
                # The model computes its Born tensors only when asked to.
                if getattr(_model, "charge_fitting", None) is None:
                    raise ValueError(
                        f"loss '{LOSS_TYPE}' with a Born prefactor needs a "
                        "'dp_lr' model with a `long_range` block"
                    )
                _model.compute_born = True
            return loss
        return current(loss_params, start_lr, _ntypes, _model)

    get_loss._deepmd_lr_patched = True  # type: ignore[attr-defined]
    get_loss.__wrapped__ = current  # type: ignore[attr-defined]
    get_loss.__doc__ = current.__doc__
    _training.get_loss = get_loss
