# SPDX-License-Identifier: MIT
"""Environment-dependent partial charges for the long-range term.

DPA4C takes the *total* charge of a frame as an input condition (see
``deepmd.utils.charge_state``); it emits no per-atom charges. PME needs
per-atom charges, so this module adds a second head on the shared descriptor
that predicts them.

Where the total-charge constraint is applied, and why not here
--------------------------------------------------------------
Physically the charges must satisfy :math:`\\sum_i q_i = Q_\\text{total}`.
That constraint is a per-frame reduction, and a fitting net cannot express it:
on the graph route ``call_graph`` receives the flat node axis ``(N,)`` with no
frame structure at all -- it reuses the dense forward by treating each node as
its own single-atom frame -- so it cannot tell which atoms share a frame, and
``charge_spin`` is never passed to a fitting in the first place.

So the head here predicts *raw* charges, and :func:`project_to_total_charge`
is applied one level up, at the model, where both the frame structure and
``charge_spin`` are available. Keeping the projection a separate pure function
also makes it testable on its own, independently of any network.
"""

import math
from typing import (
    Any,
)

import array_api_compat
import torch
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)

#: Name of the per-atom variable this head produces, and the key it is
#: registered and serialized under.
CHARGE_VAR_NAME = "charge"


def resolve_charge_spin(
    model: Any,
    charge_spin: Any,
    nf: int,
    ref: torch.Tensor,
) -> torch.Tensor | None:
    """Resolve explicit/default frame conditions on the caller's device."""
    if charge_spin is None:
        charge_spin = model.get_default_chg_spin()
    if charge_spin is None:
        return None
    condition = torch.as_tensor(charge_spin, dtype=ref.dtype, device=ref.device)
    if condition.numel() not in (2, 2 * nf):
        raise ValueError(
            "charge_spin must hold one [charge, multiplicity] pair per frame"
        )
    condition = condition.reshape(-1, 2).expand(nf, 2)
    # A compressed DPA4C reads the charge state baked into its tables and
    # ignores the runtime condition; a request that differs would be honoured
    # by the projection and silently dropped by the descriptor.
    folded = getattr(model, "_compressed_charge_spin", None)
    if folded is not None and not torch.equal(
        condition, condition.new_tensor(folded).expand_as(condition)
    ):
        raise ValueError(
            f"charge_spin {condition[0].tolist()} differs from the state "
            f"{list(folded)} folded into the compressed tables, which the "
            "kernel reads instead of the runtime condition. Compress with "
            "charge_spin set to this run's state, or run uncompressed."
        )
    return condition


def projection_shift(
    charge: torch.Tensor, raw_charge: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """The uniform shift the total-charge projection applied, per frame.

    :func:`project_to_total_charge` adds the same amount to every real atom
    of a frame, ``(Q - sum_j raw_j) / N``; this returns it, shape ``(nf,)``,
    measured from the two charge sets so it holds for any mask. It is the
    per-atom gap between the raw and the projected charges, and the quantity
    the ``ener_lr`` loss penalises (:mod:`deepmd_lr.loss` says why).
    """
    diff = charge - raw_charge
    if mask is None:
        return diff.mean(dim=-1)
    weight = mask.to(diff.dtype)
    n_real = weight.sum(dim=-1)
    return (diff * weight).sum(dim=-1) / torch.clamp(n_real, min=1.0)


def check_charge_filter(charge_filter: Any) -> tuple[float, float] | None:
    """Validate ``charge_filter = (c_t, delta)``: finite, ``0 < delta <= c_t``.

    ``None`` and ``False`` both return None (no filter)."""
    if charge_filter is None or charge_filter is False:
        return None
    try:
        c_t, delta = (float(x) for x in charge_filter)
    except (TypeError, ValueError):
        raise ValueError("charge_filter must be a pair (c_t, delta)") from None
    if not (math.isfinite(c_t) and math.isfinite(delta) and 0.0 < delta <= c_t):
        raise ValueError(
            f"charge_filter needs finite 0 < delta <= c_t, got c_t={c_t}, delta={delta}"
        )
    return c_t, delta


def charge_filter_weight(
    charge: torch.Tensor, c_t: float, delta: float
) -> torch.Tensor:
    """The filter's weight: 0 for ``|q| <= c_t - delta``, 1 for ``|q| >= c_t + delta``,
    and ``35 u^4 - 84 u^5 + 70 u^6 - 20 u^7`` between, with
    ``u = (|q| - c_t + delta) / (2 delta)`` -- C^3 at both ends."""
    u = torch.clamp((charge.abs() - (c_t - delta)) / (2.0 * delta), 0.0, 1.0)
    u2 = u * u
    return u2 * u2 * (35.0 - 84.0 * u + 70.0 * u2 - 20.0 * u2 * u)


def filter_charges(
    charge: torch.Tensor,
    total_charge: Any,
    charge_filter: tuple[float, float],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Switch small charges off, smoothly, and restore the total on the rest.

    Used in training and inference when configured on the model, or as a
    simulation override. Below a threshold an atom's charge is exactly zero,
    so a bulk whose atoms each carry a little charge carries none.

    With ``x = |q_i|`` and ``u = (x - c_t + delta) / (2 delta)`` clipped to
    [0, 1], the weight is

    .. math:: y_i = 35 u^4 - 84 u^5 + 70 u^6 - 20 u^7,

    exactly 0 for ``|q| <= c_t - delta``, exactly 1 for ``|q| >= c_t + delta``,
    and C^3 in between, so forces and the Hessian stay continuous. The frame's
    total charge is then restored on the atoms that keep charge, in proportion
    to their weight,

    .. math:: q'_i = y_i q_i + y_i (Q - \sum_j y_j q_j) / \sum_j y_j ,

    so an atom switched off stays exactly at zero. A frame with no atom left
    above the threshold spreads ``Q`` uniformly, as the plain projection does.

    ``charge`` is ``(nf, nloc)`` and already satisfies the total; ``mask``
    marks the real atoms. Differentiable, so autograd carries the filter into
    the charge-response force.
    """
    dtype = charge.dtype
    weight = torch.ones_like(charge) if mask is None else mask.to(dtype)
    y = charge_filter_weight(charge, *charge_filter) * weight
    kept = y * charge
    if not hasattr(total_charge, "shape"):
        total = torch.full(
            charge.shape[:-1], float(total_charge), dtype=dtype, device=charge.device
        )
    else:
        total = torch.as_tensor(total_charge, dtype=dtype, device=charge.device)
        total = total.reshape(-1).expand(charge.shape[0])
    deficit = total - kept.sum(dim=-1)
    y_sum = y.sum(dim=-1)
    n_real = torch.clamp(weight.sum(dim=-1), min=1.0)
    any_kept = y_sum > 0
    share = torch.where(
        any_kept[:, None],
        y / torch.where(any_kept, y_sum, torch.ones_like(y_sum))[:, None],
        weight / n_real[:, None],
    )
    return kept + share * deficit[:, None]


def charge_mask(model: Any, atype: torch.Tensor) -> torch.Tensor:
    """Atoms eligible to carry charge, including fitting-type exclusions."""
    mask = model.atomic_model.make_atom_mask(atype).bool()
    for atom_type in model.charge_fitting.exclude_types:
        mask = mask & (atype != atom_type)
    return mask


def project_to_total_charge(
    raw_charge: Array,
    total_charge: Array | float = 0.0,
    mask: Array | None = None,
) -> Array:
    r"""Shift raw per-atom charges so each frame carries its total charge.

    .. math::
        q_i = \tilde q_i + \frac{Q_\text{total} - \sum_j \tilde q_j}{N}

    The deficit is spread uniformly over the real atoms of the frame. This is
    the cheapest constraint that is exact, differentiable everywhere, and
    permutation-equivariant -- an important combination, because the charges
    are trained through the PME energy and any non-differentiable projection
    (clamping, rounding, a hard renormalization by division) would either
    block gradients or bias them.

    Note the constraint is enforced *exactly*, not as a penalty term: a
    penalty would let a frame drift off-neutral during training, and a
    non-neutral cell makes the Ewald reciprocal sum conditionally convergent,
    which is exactly the pathology the term is meant to avoid.

    Parameters
    ----------
    raw_charge
        Unconstrained per-atom charges, shape ``(nf, nloc)``.
    total_charge
        Target total charge per frame: a scalar applied to every frame, or an
        array of shape ``(nf,)``. Defaults to 0.0 (neutral).
    mask
        Optional ``(nf, nloc)`` mask of real atoms; falsy entries are treated
        as padding, take no share of the deficit, and are returned as zero.
        ``None`` means every atom is real.

    Returns
    -------
    Array
        Constrained charges, shape ``(nf, nloc)``. Summing over ``nloc``
        reproduces ``total_charge`` for every frame.
    """
    xp = array_api_compat.array_namespace(raw_charge)
    dtype = raw_charge.dtype

    if mask is None:
        weight = xp.ones_like(raw_charge)
    else:
        weight = xp.astype(mask, dtype)

    # Padding atoms must not contribute to the sum they are being corrected by.
    masked = raw_charge * weight
    current = xp.sum(masked, axis=-1)  # (nf,)
    n_real = xp.sum(weight, axis=-1)  # (nf,)

    target = total_charge
    if not hasattr(target, "shape"):
        target = xp.full_like(current, target)
    else:
        target = xp.astype(xp.reshape(target, (-1,)), dtype)

    # A frame with no real atoms has nothing to correct; guard the divide so it
    # yields 0 rather than a NaN that would poison the whole batch's gradients.
    safe_n = xp.where(n_real > 0, n_real, xp.ones_like(n_real))
    shift = xp.where(n_real > 0, (target - current) / safe_n, xp.zeros_like(n_real))

    return (raw_charge + shift[..., None]) * weight


class ChargeFittingNetDP(InvarFitting):
    # Compose rather than replace the base documentation, matching how
    # EnergyFittingNet extends InvarFitting.
    __doc__ = r"""Fit raw per-atom partial charges.

        :math:`\tilde q_i = F_\theta(\mathcal D_i)`.

        The output is deliberately unconstrained; the total-charge constraint is
        applied by :func:`project_to_total_charge` at the model level, for the
        reasons given in this module's docstring.

        """ + (InvarFitting.__doc__ or "")

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        # Mutable defaults mirror deepmd's own fitting signatures verbatim
        # (InvarFitting, EnergyFittingNet); matching them keeps this head
        # drop-in wherever a built-in fitting is accepted.
        neuron: list[int] = [128, 128, 128],  # noqa: B006
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        rcond: float | None = None,
        trainable: list[bool] | None = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        exclude_types: list[int] = [],  # noqa: B006
        type_map: list[str] | None = None,
        seed: int | list[int] | None = None,
        default_fparam: list | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            var_name=CHARGE_VAR_NAME,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            rcond=rcond,
            trainable=trainable,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            seed=seed,
            default_fparam=default_fparam,
            **kwargs,
        )

    def serialize(self) -> dict:
        """Serialize, tagging the wire format with this fitting's own type."""
        return {**super().serialize(), "type": CHARGE_VAR_NAME}

    @classmethod
    def deserialize(cls, data: dict) -> "ChargeFittingNetDP":
        """Drop the fields this subclass fixes, then defer to the base.

        ``var_name`` and ``dim_out`` are constants of this class, not
        constructor arguments, so they must be removed before the base
        deserializer forwards the dict as keyword arguments -- otherwise they
        collide with the values ``__init__`` passes up.

        Everything else is left alone deliberately: ``@class``, ``type``,
        ``@variables`` and ``nets`` are popped by ``GeneralFitting``, and
        ``@version`` is checked by ``InvarFitting``. Popping ``type`` here
        would starve that base pop, which has no default and raises KeyError.
        """
        data = data.copy()
        data.pop("var_name", None)
        data.pop("dim_out", None)
        return super().deserialize(data)


@BaseFitting.register(CHARGE_VAR_NAME)
@torch_module
class ChargeFittingNet(ChargeFittingNetDP):
    """PyTorch (pt_expt) binding of :class:`ChargeFittingNetDP`."""
