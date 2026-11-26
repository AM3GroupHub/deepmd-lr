# SPDX-License-Identifier: MIT
"""The ``dp_lr`` model type: a DPA4C energy model with a long-range term.

The term hangs off the top-level model because it needs absolute coordinates
and the cell, which DeePMD-kit's atomic-model contract does not provide:
``forward_common_atomic`` sees extended coordinates and a neighbour list, and
on the graph route geometry enters ONLY through ``edge_vec``. The cell is
available one level up, at ``call_common(coord, atype, box, ...)``.

Nor is its energy added to the atomic output. There, force and virial come
from ``edge_force_virial``, the derivative with respect to ``edge_vec``; the
reciprocal-space energy does not depend on ``edge_vec``, so it would shift the
energy with exactly zero force and virial, and the model would quietly stop
being conservative. The long-range branch takes its own derivatives instead:
with respect to the coordinates (force) and to a symmetric strain applied to
coordinates and cell together (virial), in DeePMD-kit's convention read off
``edge_force_virial``: ``force = -dE/dr`` and ``virial = -dE/de`` for
``r -> (I + e) r``.

The charge head reads the descriptor features the energy head has already
computed (``_capturing_backbone``), so a frame runs the descriptor once. A
second pass, ``_predict_charges``, runs only when the energy path did not go
through the graph lower and so left nothing to capture.
"""

import contextlib
import copy
import functools
import json
import math
import os
from collections.abc import (
    Callable,
    Iterator,
)
from numbers import Real
from typing import (
    Any,
    NamedTuple,
)

import numpy as np
import torch
from ase.data import (
    covalent_radii,
)
from deepmd.dpmodel.model.edge_transform_output import (
    node_ownership_mask,
)
from deepmd.dpmodel.model.model_factory import (
    get_model_components,
)
from deepmd.dpmodel.utils.neighbor_graph.derivatives import (
    edge_force_virial,
)
from deepmd.kernels.cuda.edge_force_virial import (
    frame_scalar_sum,
)
from deepmd.kernels.utils import (
    cuda_infer_level,
)
from deepmd.pt_expt.descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.fitting import (
    BaseFitting,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)

from deepmd_lr.born import (
    born_charges,
    screening_factor,
)
from deepmd_lr.compress import (
    FOLDED_ATTR,
    folded_charge_spin,
)
from deepmd_lr.fitting.charge import (
    charge_mask,
    check_charge_filter,
    filter_charges,
    project_to_total_charge,
    projection_shift,
    resolve_charge_spin,
)
from deepmd_lr.fitting.multipole import (
    cartesian_features,
    contract,
    equivariant_channels,
)
from deepmd_lr.lr.multipole import (
    build_term,
)
from deepmd_lr.lr.term import (
    LongRangeTerm,
    isolated_term_for,
    system_reciprocal,
)
from deepmd_lr.neighbors import (
    build_neighbor_graph_fast,
)
from deepmd_lr.zbl import (
    ZBL,
    ZBL_KEY,
    Contacts,
    add_envelope_gradient,
    contact_output,
    pair_key,
)

#: The ``model.type`` that selects this class, and the tag written into
#: checkpoints so a serialized model round-trips back to this class rather
#: than to a plain ``EnergyModel``.
MODEL_TYPE = "dp_lr"

#: Wire tag of the atomic model this class wraps. ``DPAtomicModel.serialize``
#: emits ``type: "standard"`` and its ``deserialize`` pops that key without a
#: default, so the round-trip has to put it back (see :meth:`deserialize`).
_ATOMIC_MODEL_TYPE = "standard"
#: Wire key for the atomic model's own type when it is not that one.
_WRAPPED_TYPE_KEY = "atomic_model_type"

#: Key of the optional long-range block in the model config.
LONG_RANGE_KEY = "long_range"

#: Key of the long-range sum's settings inside that block, whatever `method`
#: evaluates it.
RECIPROCAL_KEY = "reciprocal"

#: The old name of that key, still accepted in configs and checkpoints.
_LEGACY_RECIPROCAL_KEY = "pme"

#: Raised wherever compression is asked for, or found, without the compiled
#: kernel. The wording carries the consequence, because the alternative is a
#: model that says it is compressed and runs at the old speed.
_NO_KERNEL = (
    "compression needs the compiled operator "
    "`torch.ops.deepmd.dpa4c_graph_compress`, which is not loaded. The PyPI "
    "deepmd-kit wheel does not ship it: build deepmd-kit 3.2.0 from source "
    "with DP_VARIANT=cuda and run in that environment. Without the kernel the "
    "tables are built and then silently ignored -- `DescrptDPA4C.call_graph` "
    "falls back to the uncompressed math, so nothing gets faster."
)

#: The smallest envelope a short-contact window may leave any atom of the
#: training data with (:meth:`LongRangeEnergyModel.check_zbl_against`). A
#: contact that only grazes a window's upper edge leaves more than this.
_ZBL_ON_DATA = 0.99

#: Raised when a Hessian and ZBL meet, from whichever side comes second.
_NO_HESSIAN = (
    "the Hessian is not supported with ZBL: DeePMD-kit's Hessian wrapper "
    "evaluates the atomic model directly, past the graph lower where the "
    "term is added, so the second derivatives would silently leave it out"
)


def _weighted_ownership(mask: Any, envelope: torch.Tensor) -> torch.Tensor:
    """DeePMD-kit's per-node ownership mask times the envelope, float64;
    refused for anything that is not that mask.

    The short-contact envelope reaches the fused operators on the per-node
    boolean mask DeePMD-kit hands them (``DescrptDPA4C.fused_energy_force_graph``
    and ``_prepare_graph_nodes``, which this module wraps). Should a later
    DeePMD-kit pass something else in that place, the envelope would weigh
    the wrong tensor without an error anywhere.
    """
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise RuntimeError(
            "the short-contact envelope (`zbl`) is handed to DeePMD-kit's "
            "fused energy/force operators on their per-node ownership mask, "
            "and this DeePMD-kit passed something else in its place "
            f"({type(mask).__name__}"
            + (f", {mask.dtype}" if isinstance(mask, torch.Tensor) else "")
            + "): the interface has changed. Evaluate at DP_CUDA_INFER=1, "
            "whose autograd lower does not depend on it."
        )
    return mask.to(torch.float64) * envelope


@contextlib.contextmanager
def _hooked(obj: Any, name: str, replacement: Any) -> Iterator[None]:
    """Replace the method ``obj.name`` by ``replacement`` for one pass.

    Undone by deleting, unless the object carried its own attribute:
    assigning the original back would leave a bound method in the instance
    dict, shadowing the class attribute and holding the instance.
    """
    had_own = name in obj.__dict__
    original = getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if had_own:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


#: The energy fitting nets a ``dp_lr`` model accepts: DeePMD-kit's standard
#: one, and the one it pairs with the DPA4 descriptor. Both are invariant
#: per-atom energy heads.
_ENERGY_FITTINGS = ("ener", "dpa4_ener", "sezm_ener")


def _learned(atomic_model: Any) -> Any:
    """``atomic_model`` itself, or the learned child (the one with the
    descriptor) of DeePMD-kit's bridged composition
    ``[learned, inner_potential]``."""
    children = getattr(atomic_model, "models", None)
    if children is None:
        return atomic_model
    return next(child for child in children if hasattr(child, "descriptor"))


def backbone(model: Any) -> Any:
    """The learned atomic model of ``model``: the one with the descriptor,
    also under DeePMD-kit's ZBL bridging (``bridging_method``)."""
    return _learned(model.atomic_model)


def message_blocks(model: Any) -> int | None:
    """The number of message-passing blocks of a DPA4 descriptor; 0 for a
    descriptor without message passing, None for one this module cannot
    tell."""
    descriptor = backbone(model).descriptor
    if not descriptor.has_message_passing():
        return 0
    if type(descriptor).__name__ == "DescrptDPA4":
        return len(descriptor.l_schedule)
    return None


def receptive_hops(model: Any) -> int | None:
    """How many neighbour shells an atom's descriptor output reaches, in
    units of the cutoff; None for a descriptor this module cannot tell.

    DPA4C scans the edges into an atom once: one hop, and everything an atom
    carries (energy, raw charge) depends on its own neighbours only. DPA4
    passes messages: each interaction block reads its neighbours' states, and
    the environment seed has already folded one shell into the state the
    first block reads. (DeePMD-kit's bridging adds a margin to this, not a
    shell: :func:`gate_margin`.)
    """
    blocks = message_blocks(model)
    if not blocks:  # no message passing (one hop), or unknown
        return 1 if blocks == 0 else None
    return blocks + int(bool(backbone(model).descriptor.use_env_seed))


def gate_margin(model: Any) -> float:
    """How far beyond ``receptive_hops`` cutoffs an atom's output reaches, in
    angstrom: ``bridging_r_outer`` for a DPA4 with DeePMD-kit's bridging,
    else 0.

    With bridging a neighbour's message is gated by that neighbour's own
    short contacts, the pairs closer than ``bridging_r_outer``. The outermost
    atom an output reads can thus have a contact partner just beyond the last
    cutoff, and that partner's position matters.
    """
    descriptor = backbone(model).descriptor
    if getattr(descriptor, "bridging_switch", None) is None:
        return 0.0
    return float(descriptor.inner_clamp_r_outer)


def require_one_hop(model: Any, what: str) -> None:
    """Refuse ``what`` on a message-passing descriptor.

    Partitions with a halo of one cutoff assume that an atom's output depends
    on the edges into that atom alone. On a descriptor that passes messages
    they would return wrong numbers without an error anywhere.
    """
    hops = receptive_hops(model)
    if hops == 1:
        return
    name = type(backbone(model).descriptor).__name__
    reach = "an unknown number of" if hops is None else str(hops)
    raise NotImplementedError(
        f"{what} is not available on {name}: it passes messages, so an "
        f"atom's energy and charge reach {reach} neighbour shells, and this "
        "path assumes one. Energies, forces, virials and charges on one "
        "device are exact for any depth."
    )


#: Key in ``long_range``: the radii ``[r_lo, r_hi]`` in angstrom across which
#: an atom's charge is switched off at a short contact.
CHARGE_SHIELD_KEY = "charge_shield"

#: Key in ``long_range``: the head giving every atom a dipole (and a
#: quadrupole) beside its charge.
MULTIPOLE_KEY = "multipole_net"

#: Key in ``long_range``: true freezes the short-range model, so that only the
#: charge head trains.
CHARGE_HEAD_ONLY_KEY = "train_charge_head_only"


def _bridging_radii(data: dict) -> tuple[float, float] | None:
    """``(r_inner, r_outer)`` of an active ``bridging_method``, else None."""
    if str(data.get("bridging_method", "none")).lower() in ("none", ""):
        return None
    return (
        float(data.get("bridging_r_inner", 0.5)),
        float(data.get("bridging_r_outer", 0.8)),
    )


def _bridged_atomic_model(data: dict) -> Any:
    """DeePMD-kit's bridged composition ``[learned, inner_potential]`` for a
    ``dp_lr`` config with ``bridging_method``, built by DeePMD-kit itself.

    Its own expansion and linear-model builder do the work, on the config a
    ``standard`` model would have: the ZBL pair sum, the inner clamp in the
    descriptor, the composition's statistics and exclusions are all theirs.
    """
    from deepmd.pt_expt.model.get_model import get_linear_model
    from deepmd.utils.bridging import expand_bridging_method

    ours = (LONG_RANGE_KEY, ZBL_KEY, "type")
    plain = {key: value for key, value in data.items() if key not in ours}
    return get_linear_model(
        expand_bridging_method({**plain, "type": "standard"})
    ).atomic_model


def _kernel_available() -> bool:
    """Whether ``deepmd::dpa4c_graph_compress`` is registered in this process."""
    from deepmd.kernels.cuda.dpa4c.graph_compress import (
        op_available,
    )

    return bool(op_available())


def _strip_comments(block: dict) -> dict:
    """Drop DeePMD-kit's ``_comment`` annotation keys from a config block.

    ``dp train`` normalizes them away, but a raw config read straight from
    JSON (``scripts/graft_short_range.py``, a hand-built ``get_model`` call)
    still carries them, and the long-range term refuses unknown keys.
    """
    return {k: v for k, v in block.items() if not k.startswith("_comment")}


def _symmetric_strain(dtype, device, batch: tuple[int, ...] = ()):
    """A strain leaf ``eps`` and the transform ``I + sym(eps)`` it defines.

    ``eps`` is the leaf a virial differentiates with respect to; only its
    symmetric part acts, so a rotation cannot masquerade as stress. ``batch``
    prepends frame axes: ``()`` for one ``(3, 3)`` strain, ``(nf,)`` for one per
    frame.
    """
    eps = torch.zeros((*batch, 3, 3), dtype=dtype, device=device, requires_grad=True)
    identity = torch.eye(3, dtype=dtype, device=device).expand(*batch, 3, 3)
    return eps, identity + 0.5 * (eps + eps.transpose(-1, -2))


def _run_backbone(model, graph, atype_flat, charge_spin):
    """Run ``model``'s descriptor on ``graph``: ``(features, rot_mat)``.

    The charge head must see the SAME frame conditioning as the energy head,
    or the two descriptor passes disagree about what system they are
    describing. Gated on the descriptor's own capability, exactly as
    ``DPAtomicModel.forward_atomic_graph`` gates it -- only the DPA4 family
    accepts the keyword, and ``dp_lr`` allows any descriptor.
    """
    descriptor = backbone(model).descriptor
    kwargs = (
        {"charge_spin": charge_spin}
        if backbone(model).supports_charge_spin
        else {}
    )
    return descriptor.call_graph(
        graph,
        atype_flat,
        type_embedding=descriptor.graph_type_embedding_table(),
        **kwargs,
    )


def _is_compressed(model) -> bool:
    """Whether ``model``'s descriptor is running from compressed tables."""
    return bool(getattr(backbone(model).descriptor, "compress", False))


class _Charges(NamedTuple):
    """The charges of one evaluation, ``(nf, nloc)`` each, and the shift."""

    #: On the frame's total charge: what the long-range term sums.
    projected: torch.Tensor
    #: The charge head's output.
    raw: torch.Tensor
    #: What the Born tensor differentiates: the raw charges, or with a
    #: short-contact envelope ``w (raw + shift)`` at a fixed shift.
    local: torch.Tensor
    #: ``(nf,)``, what the projection added to every atom that carries weight.
    shift: torch.Tensor
    #: The short-contact envelope the projection was weighted with, or None.
    #: In inference a leaf, a plain factor whose own derivative comes from
    #: the list of contacts; otherwise a function of the edge vectors.
    envelope: torch.Tensor | None = None
    #: Atomic dipoles ``(nf, nloc, 3)`` and quadrupoles ``(nf, nloc, 3, 3)``
    #: of a model with a multipole head, already weighted by the mask and the
    #: envelope; None without one (and the quadrupole below ``lmax`` 2).
    dipole: torch.Tensor | None = None
    quadrupole: torch.Tensor | None = None


class _Strained(NamedTuple):
    """The leaves one long-range evaluation differentiates against, and the
    geometry they define."""

    #: ``(nf, nloc, 3)`` coordinate leaf: the force is ``-dE/d(leaf)``.
    leaf: torch.Tensor
    #: ``(nf, 3, 3)`` strain leaf: the virial is ``-dE/d(eps)``.
    eps: torch.Tensor
    #: ``I + sym(eps)``, per frame.
    strain: torch.Tensor
    #: The leaf coordinates and the cell (None: no cell) under ``strain``.
    coord: torch.Tensor
    box: torch.Tensor | None
    #: The unstrained cell, ``(nf, 3, 3)``, detached.
    base_box: torch.Tensor | None


def _strained_geometry(
    coord: torch.Tensor, box: torch.Tensor | None, nf: int, nloc: int
) -> _Strained:
    """Fresh leaves for coordinates and a per-frame symmetric strain.

    The trainer passes coordinates as ``(nf, nloc * 3)`` and the cell as
    ``(nf, 9)``, eager callers ``(nf, nloc, 3)`` and ``(nf, 3, 3)``; both are
    normalized. Detaching the geometry is safe for training: the energy still
    depends on the parameters through the charge head, so ``create_graph``
    keeps the force differentiable in them. The strain is symmetrized so a
    rotation cannot leak into the virial. With no cell only the coordinates
    are strained, which gives the pair virial ``sum_ij r_ij (x) f_ij`` --
    what DeePMD-kit's own energy model returns for a ``box=None`` frame, so
    the total stays consistent with the short-range part.
    """
    base_box = None if box is None else box.detach().reshape(nf, 3, 3)
    eps, strain = _symmetric_strain(coord.dtype, coord.device, batch=(nf,))
    leaf = coord.detach().reshape(nf, nloc, 3).clone().requires_grad_(True)
    # Row vectors: r' = r (I + eps)^T, and the cell rows transform alike.
    return _Strained(
        leaf=leaf,
        eps=eps,
        strain=strain,
        coord=torch.bmm(leaf, strain.transpose(-1, -2)),
        box=(
            None if base_box is None else torch.bmm(base_box, strain.transpose(-1, -2))
        ),
        base_box=base_box,
    )


@contextlib.contextmanager
def _restored_on_failure(descriptor: Any) -> Iterator[None]:
    """Put back the buffers and attributes upstream's ``enable_compression``
    registers or replaces if the body raises, keeping the original objects
    so shared references survive a failed compression."""
    buffers = descriptor._buffers.copy()
    nonpersistent = descriptor._non_persistent_buffers_set.copy()
    was_compressed = descriptor.compress
    had_scalars = hasattr(descriptor, "_compression_scalars")
    scalars = getattr(descriptor, "_compression_scalars", None)
    had_folded = FOLDED_ATTR in descriptor.__dict__
    folded = descriptor.__dict__.get(FOLDED_ATTR)
    try:
        yield
    except BaseException:
        descriptor._buffers.clear()
        descriptor._buffers.update(buffers)
        descriptor._non_persistent_buffers_set.clear()
        descriptor._non_persistent_buffers_set.update(nonpersistent)
        descriptor.compress = was_compressed
        if had_scalars:
            descriptor._compression_scalars = scalars
        elif hasattr(descriptor, "_compression_scalars"):
            del descriptor._compression_scalars
        if had_folded:
            setattr(descriptor, FOLDED_ATTR, folded)
        else:
            descriptor.__dict__.pop(FOLDED_ATTR, None)
        raise


@BaseModel.register(MODEL_TYPE)
class LongRangeEnergyModel(EnergyModel):
    """Energy model that carries an explicit long-range electrostatic term.

    Reached from an input file through ``model.type: "dp_lr"``. The generic
    dispatch in ``deepmd.dpmodel.model.model_factory.get_model`` ends with
    ``base_model.get_class_by_type(model_type).get_model(data)``, so a
    plugin-registered type needs no change to DeePMD-kit itself.
    """

    def __init__(
        self,
        *args: Any,
        charge_fitting: Any = None,
        long_range_energy: LongRangeTerm | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Registered as submodules so they move with `.to(device)`, appear in
        # `state_dict()`, and are picked up by the optimizer.
        self.charge_fitting = charge_fitting
        self.long_range_energy = long_range_energy
        #: The head giving every atom a dipole (and a quadrupole) beside its
        #: charge (``long_range.multipole_net``), or None.
        self.multipole_fitting = None
        #: Also return Born effective charge tensors from `forward`. Off by
        #: default because it costs an extra backward pass per call, and only
        #: a field simulation needs them. Set it on the model, not in the
        #: config -- a field is a property of the simulation, not the model.
        self.compute_born = False
        #: High-frequency dielectric constant of the medium. Reported charges
        #: and Born tensors are multiplied by sqrt(eps_inf), the factor that
        #: turns the model's Coulomb charges into screened ones; the energy
        #: and forces are untouched. 1.0 is vacuum. See `deepmd_lr.born`.
        self.eps_inf = 1.0
        #: ``(c_t, delta)`` from ``long_range.charge_filter``, or None: part of
        #: the model, applied in training and simulation, and saved.
        self.trained_charge_filter: tuple[float, float] | None = None
        #: A simulation-time override of it: ``(c_t, delta)``, ``False`` for
        #: off, or None to use the model's own. Never saved; refused in
        #: training, where only the model's own filter applies.
        self.charge_filter: Any = None
        #: True on the non-base members of a `same_as` group, whose
        #: `atomic_model` is the base's very object. A plain bool, not a
        #: module: see `deepmd_lr.same_as`.
        self._same_as_link = False
        #: How many `_capturing_backbone` contexts are open. The graph lower
        #: is allowed only inside one -- that is where the long-range term
        #: gets added -- and captures nest, so this is a count, not a flag.
        self._capture_depth = 0
        #: The short-contact term (``zbl`` in the config), or None: ZBL
        #: repulsion inside a window, and the envelope that switches the
        #: fitted energy and the raw charge of an atom in contact off there.
        #: It holds no weights; see :meth:`set_zbl`.
        self.zbl: ZBL | None = None
        #: The envelope that switches charges off at a short contact when
        #: DeePMD-kit's own bridging handles the energy there, or None; see
        #: :meth:`set_charge_shield`.
        self.charge_shield: ZBL | None = None
        #: The contacts of the graph lower now running, for the capture it
        #: runs inside to take along; None outside one and on the torch path.
        self._contacts: Contacts | None = None

    @property
    def effective_charge_filter(self) -> tuple[float, float] | None:
        """The filter applied now: the simulation-time override if one is set
        (``False`` switching the model's own off), else the model's own."""
        if self.charge_filter is None:
            return self.trained_charge_filter
        return check_charge_filter(self.charge_filter)

    @property
    def _compressed_charge_spin(self) -> tuple[float, float] | None:
        """The (charge, multiplicity) folded into the compressed tables, or None.

        The compressed kernel ignores the runtime condition, so a forward with
        a different one is refused (`resolve_charge_spin`). Read from the
        descriptor, which records it whichever method folded the tables --
        this class's `enable_compression`, upstream's, or a later
        `apply_charge_state` (see `deepmd_lr.compress`).
        """
        return folded_charge_spin(backbone(self).descriptor)

    @_compressed_charge_spin.setter
    def _compressed_charge_spin(self, value: tuple[float, float] | None) -> None:
        setattr(backbone(self).descriptor, FOLDED_ATTR, value)

    def change_out_bias(
        self, merged: Any, bias_adjust_mode: str = "change-by-statistic"
    ) -> None:
        """Refit the per-type energy offsets, once per `same_as` group.

        Every member of such a group shares one ``atomic_model``, so
        ``out_bias`` is one tensor -- but DeePMD-kit loops over branch keys and
        would apply the same delta to it once per branch. ``"change-by-statistic"``
        *adds*, so with two branches the offsets would land at twice the
        correction. The base branch owns the update; the links stand down.

        The delta is the same for every member whatever its long-range
        method, because the regression subtracts the **atomic** model's
        prediction, which is short-range only.
        """
        if self._same_as_link:
            return
        super().change_out_bias(merged, bias_adjust_mode=bias_adjust_mode)

    @property
    def has_long_range(self) -> bool:
        """Whether a long-range term is configured. Without one this model is
        exactly the standard energy model."""
        return (
            self.charge_fitting is not None and self.long_range_energy is not None
        )

    @property
    def has_multipoles(self) -> bool:
        """Whether the atoms carry dipoles (and quadrupoles) in the
        long-range term."""
        return self.multipole_fitting is not None

    def long_range_energy_for(
        self, cell: Any, pbc: Any, settings: dict[str, Any] | None = None
    ) -> torch.nn.Module | None:
        """A long-range term built for one system; this model's own is left
        as it is.

        The system's ``pbc`` picks the 3D, the slab or the isolated sum, and
        its cell the PME mesh (:func:`~deepmd_lr.lr.term.system_reciprocal`).
        The term holds no weights, so every system can have one of its own:
        that is how evaluators and calculators share a model without
        disturbing each other. A model without a long-range block gives None.

        The settings adapted are ``settings``, checked as
        :meth:`set_long_range` checks them, or by default the model's as they
        were before any :meth:`adapt_long_range`. Adapting its current ones
        instead would leave one large cell's mesh behind for every smaller
        one after it: the rule never lowers a mesh.
        """
        if not self.has_long_range:
            if settings is not None:
                self._checked_long_range(settings)  # raises: no block
            return None
        if settings is not None:
            base = self._checked_long_range(settings)
        elif getattr(self, "_reciprocal_base", None) is not None:
            base = self._reciprocal_base
        else:
            base = self.long_range_energy.serialize()
        return build_term(system_reciprocal(base, cell, pbc), self.has_multipoles)

    def adapt_long_range(self, cell: Any, pbc: Any) -> bool:
        """Install the term :meth:`long_range_energy_for` builds for this
        system; returns whether it changed.

        The ASE calculator does this before each frame, because it runs the
        model's own forward. The GPU evaluator builds a term of its own and
        never changes the model. An explicit :meth:`set_long_range` is a new
        starting point for both.
        """
        term = self.long_range_energy_for(cell, pbc)
        if term is None or term.serialize() == self.long_range_energy.serialize():
            return False
        if getattr(self, "_reciprocal_base", None) is None:
            self._reciprocal_base = self.long_range_energy.serialize()
        self._install_long_range(term.serialize())
        return True

    def set_long_range(self, settings: dict[str, Any]) -> None:
        """Swap the long-range term for one built from ``settings``.

        The term contributes no ``state_dict`` entry, so how the sum is taken
        is not a property of the checkpoint: a model trained on slabs can be
        asked for bulk or isolated energies by rebinding this attribute.

        ``settings`` **replaces** the reciprocal block rather than merging into
        it: the isolated term refuses any periodic setting actually set, so
        folding a ``pme`` block into ``method: "isolated"`` would raise.
        ``alpha`` is inherited and may not change -- the descriptor learned
        the ``erfc(alpha r)/r`` remainder of *that* split.

        The new settings are also what :meth:`long_range_energy_for` and
        :meth:`adapt_long_range` adapt from.
        """
        self._install_long_range(settings)
        self._reciprocal_base = None

    def _checked_long_range(self, settings: dict[str, Any]) -> dict[str, Any]:
        """``settings`` with this model's alpha, which they may not change."""
        if not self.has_long_range:
            raise ValueError(
                "this model has no `long_range` block, so there is no "
                "long-range term to set. Train with `type: dp_lr` and a "
                "`long_range` section to get one."
            )
        current = self.long_range_energy.serialize()
        settings = {"alpha": current["alpha"], **settings}
        if float(settings["alpha"]) != float(current["alpha"]):
            raise ValueError(
                f"alpha is fixed at {current['alpha']} by training and cannot "
                f"be changed here, got {settings['alpha']}. It sets where "
                "Coulomb is split, and the network learned the short-ranged "
                "remainder of *that* split; moving it silently invalidates "
                "every weight in the model."
            )
        return settings

    def _install_long_range(self, settings: dict[str, Any]) -> None:
        """The swap itself, leaving :meth:`adapt_long_range`'s base alone."""
        settings = self._checked_long_range(settings)
        self.long_range_energy = build_term(settings, self.has_multipoles)
        # Keep the recorded config in step: a model re-serialised after a swap
        # describes the term it has, not the one it was trained with.
        script = getattr(self, "model_def_script", None)
        if not script:
            return
        recorded = json.loads(script)
        block = recorded.get(LONG_RANGE_KEY)
        if block is None:
            return
        block.pop(_LEGACY_RECIPROCAL_KEY, None)
        block[RECIPROCAL_KEY] = self.long_range_energy.serialize()
        self.model_def_script = json.dumps(recorded)

    @classmethod
    def get_model(cls, model_params: dict) -> "LongRangeEnergyModel":
        """Build the model from a normalized ``model`` config section.

        Mirrors ``deepmd.dpmodel.model.model_factory.get_standard_model``
        rather than calling it, because that function looks the model class up
        by *fitting* type and would hand back a plain ``EnergyModel``. The
        descriptor and fitting construction itself is reused verbatim through
        :func:`get_model_components`, which owns the ``ntypes`` / ``type_map``
        / ``dim_descrpt`` injection that is easy to get subtly wrong.
        """
        data = copy.deepcopy(model_params)
        data.pop("type", None)
        if data.get(ZBL_KEY) is not None and data.get("use_srtab") is not None:
            raise ValueError(
                "`zbl` and `use_srtab` both put a pair potential under the "
                "network; choose one"
            )

        fitting_type = str((data.get("fitting_net") or {}).get("type", "ener"))
        if fitting_type not in _ENERGY_FITTINGS:
            raise ValueError(
                f"The '{MODEL_TYPE}' model is an energy model and requires "
                f"fitting_net.type {' or '.join(map(repr, _ENERGY_FITTINGS))}, "
                f"got '{fitting_type}'. A long-range electrostatic term is only "
                "defined as an energy contribution."
            )
        bridging = _bridging_radii(data)
        if bridging is None:
            descriptor, fitting, _ = get_model_components(
                data,
                descriptor_base=BaseDescriptor,
                fitting_base=BaseFitting,
                backend_name="pt_expt",
            )
            base = {
                "descriptor": descriptor,
                "fitting": fitting,
                "type_map": data["type_map"],
                "atom_exclude_types": data.get("atom_exclude_types", []),
                "pair_exclude_types": data.get("pair_exclude_types", []),
            }
        else:
            if data.get(ZBL_KEY) is not None:
                raise ValueError(
                    "`zbl` and `bridging_method` both handle short contacts; "
                    "`zbl` is for DPA4C, `bridging_method` for DPA4"
                )
            base = {"atomic_model_": _bridged_atomic_model(data)}
            descriptor = _learned(base["atomic_model_"]).descriptor

        charge_fitting, long_range_energy, trained_filter = cls._build_long_range(
            data, descriptor
        )
        model = cls(
            **base, charge_fitting=charge_fitting, long_range_energy=long_range_energy
        )
        model.trained_charge_filter = trained_filter
        model.multipole_fitting = cls._build_multipole(data, descriptor)
        block = data.get(LONG_RANGE_KEY) or {}
        if charge_fitting is not None:
            # The key's own value, or with bridging its radii: a normalized
            # config writes the key as null.
            shield = block.get(CHARGE_SHIELD_KEY)
            model.set_charge_shield(bridging if shield is None else shield)
        if block.get(CHARGE_HEAD_ONLY_KEY):
            # Every weight of the short-range model, whatever its descriptor
            # and fitting net make of their own `trainable` key (DeePMD-kit's
            # `dpa4_ener` net trains with it false).
            model.atomic_model.requires_grad_(False)
        # Before the config is recorded, which already holds the block.
        model.set_zbl(data.get(ZBL_KEY))
        model.model_def_script = json.dumps(model_params)
        return model

    def set_zbl(self, settings: dict[str, Any] | None) -> None:
        """Give the model the short-contact term of ``settings``, the
        config's ``zbl`` block; None removes it.

        ``{}`` is a complete block: every pair of elements then has the
        default window, placed below a bond by the covalent radii. Inside a
        window ZBL repulsion comes in and the network goes out: the fitted
        energy of every atom is multiplied by the envelope, the product over
        its neighbours of the window's step, and its charge is weighted by it
        (:mod:`deepmd_lr.zbl`). The term holds no weights and adds no
        ``state_dict`` entry, so a trained model can be given one: with every
        window's upper edge at or below the shortest contact of its data,
        nothing it was fitted to changes. Training checks that against its
        own data (:meth:`compute_or_load_stat`); for a trained model it is
        the caller's to check (:meth:`check_zbl_against`). The
        recorded config follows, so a model saved afterwards describes the
        term it has.
        """
        if settings is None:
            self.zbl = None
        else:
            if getattr(self, "_hessian_enabled", False):
                raise NotImplementedError(_NO_HESSIAN)
            descriptor = backbone(self).descriptor
            if type(descriptor).__name__ != "DescrptDPA4C":
                raise ValueError("zbl supports the dpa4c descriptor only")
            if (
                self.atomic_model.atom_exclude_types
                or self.atomic_model.pair_exclude_types
            ):
                raise ValueError("zbl with atom or pair exclusions is not supported")
            if self.charge_shield is not None:
                raise ValueError("a charge shield already weighs the charges")
            zbl = ZBL(
                self.get_type_map(), settings, device=next(self.parameters()).device
            )
            rcut = float(descriptor.get_rcut())
            if zbl.max_outer > rcut:
                raise ValueError(
                    f"a zbl window ends at {zbl.max_outer:g} A for the largest "
                    f"pair of this type_map, beyond descriptor.rcut {rcut:g} A: "
                    "a pair entering the neighbour list would arrive inside "
                    "the switch. Lower `window`, or set it to null and give "
                    "`pairs` for the elements in use."
                )
            # A module made after `.eval()` would still be in training mode,
            # where the envelope takes the slower torch arithmetic.
            self.zbl = zbl.train(self.training)
        script = getattr(self, "model_def_script", None)
        if not script:
            return
        recorded = json.loads(script)
        if self.zbl is None:
            recorded.pop(ZBL_KEY, None)
        else:
            recorded[ZBL_KEY] = self.zbl.serialize()
        self.model_def_script = json.dumps(recorded)

    @classmethod
    def _build_long_range(
        cls, data: dict, descriptor: Any
    ) -> tuple[Any, LongRangeTerm | None, tuple[float, float] | None]:
        """The charge head, the long-range term and the trained charge filter
        from ``long_range``; all None without the block."""
        block = data.get(LONG_RANGE_KEY)
        if not block:
            return None, None, None
        block = _strip_comments(block)

        charge_net = _strip_comments(copy.deepcopy(block.get("charge_net", {})))
        charge_net.pop("type", None)
        charge_net["ntypes"] = descriptor.get_ntypes()
        charge_net["dim_descrpt"] = descriptor.get_dim_out()
        charge_net["mixed_types"] = descriptor.mixed_types()
        charge_net["type_map"] = copy.deepcopy(data["type_map"])
        charge_fitting = BaseFitting.get_class_by_type("charge")(**charge_net)

        settings = block.get(RECIPROCAL_KEY, block.get(_LEGACY_RECIPROCAL_KEY, {}))
        long_range_energy = build_term(
            _strip_comments(copy.deepcopy(settings)),
            block.get(MULTIPOLE_KEY) is not None,
        )
        trained_filter = check_charge_filter(block.get("charge_filter"))
        return charge_fitting, long_range_energy, trained_filter

    @classmethod
    def _build_multipole(cls, data: dict, descriptor: Any) -> Any:
        """The multipole head of ``long_range.multipole_net``, or None."""
        settings = (data.get(LONG_RANGE_KEY) or {}).get(MULTIPOLE_KEY)
        if settings is None:
            return None
        settings = _strip_comments(copy.deepcopy(settings))
        settings.pop("type", None)
        settings["channels"] = equivariant_channels(
            descriptor, int(settings.get("lmax", 1))
        )
        settings["ntypes"] = descriptor.get_ntypes()
        settings["dim_descrpt"] = descriptor.get_dim_out()
        settings["mixed_types"] = descriptor.mixed_types()
        settings["type_map"] = copy.deepcopy(data["type_map"])
        return BaseFitting.get_class_by_type("multipole")(**settings)

    def _predict_charges(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor | None,
        strain: torch.Tensor | None = None,
        *,
        return_raw: bool = False,
    ) -> "torch.Tensor | _Charges":
        """Per-atom charges from a second backbone pass, projected onto the
        frame's total charge (a per-frame reduction the head, seeing a flat
        node axis, cannot express).

        ``strain`` is a per-frame ``(nf, 3, 3)`` matrix, ``I + sym(eps)``,
        applied to the edge vectors AFTER the graph is built.
        """
        descriptor = backbone(self).descriptor
        self._refuse_compressed_fallback()
        graph = build_neighbor_graph_fast(coord, atype, box, descriptor.get_rcut())
        if strain is not None:
            # The virial needs d(charge)/d(strain). Through a strained cell
            # that derivative would run through the box, which of DeePMD-kit's
            # neighbour builders only the dense O(N^2) one carries (`nv`
            # returns 6.7e-13 for it, silently; the dense build cost 63 GB and
            # 3 s per training step at 6520 atoms). An edge vector
            # ``coord[j] + S @ box - coord[i]`` transforms as a whole by
            # ``(I + eps)^T``, so straining the finished edges is identical and
            # keeps the O(N) builder; the partitioned path relies on the same.
            # Frames occupy contiguous node blocks of ``nloc`` and an edge
            # never crosses frames, so its destination picks the strain;
            # dummy edges are zero and stay zero.
            nloc = coord.shape[1]
            frame_of_edge = graph.edge_index[1] // nloc
            graph.edge_vec = torch.einsum(
                "eab,eb->ea", strain[frame_of_edge], graph.edge_vec
            )
        gg, rot_mat = _run_backbone(self, graph, atype.reshape(-1), charge_spin)
        return self._charges_from_features(
            gg,
            rot_mat,
            atype,
            charge_spin,
            mask,
            (graph.edge_index, graph.edge_vec, graph.edge_mask),
            return_raw=return_raw,
        )

    def _refuse_compressed_fallback(self) -> None:
        """Refuse the second backbone pass on a compressed model.

        At ``DP_CUDA_INFER=1`` the compressed branch lives inside
        ``call_graph``, so the energy pass is always captured. Reaching the
        fallback means the capture was bypassed, and it would run the slow
        math on a graph with no CSR views while the model claims otherwise.
        """
        if _is_compressed(self):
            raise RuntimeError(
                "the fallback charge pass was reached on a compressed model: "
                "the shared-backbone capture was not taken, and compressed "
                "evaluation supports only the captured route. The usual cause "
                "is DP_CUDA_INFER changed after compression; it must be exactly "
                "1 (2 selects a fused kernel that bypasses "
                "`descriptor.call_graph`)."
            )

    @contextlib.contextmanager
    def _capturing_backbone(self) -> Any:
        """Yield a dict that fills with the descriptor's output as it runs.

        The tensors the charge head needs are local to DeePMD-kit's
        ``forward_common_lower_graph``, which makes its own ``edge_vec`` leaf
        and hands it to the descriptor, so ``descriptor.call_graph`` is
        wrapped for the duration of one forward and restored afterwards,
        exception or not. Yields an empty dict when there is no long-range
        term to feed.

        While a capture is open the graph lower is permitted
        (``_capture_depth``): whoever holds this context adds the long-range
        term afterwards -- ``forward`` here, or the partitioned driver. Any
        other caller of the lower would silently report the short-range
        energy alone.
        """
        captured: dict = {}
        if not self.has_long_range:
            yield captured
            return

        descriptor = backbone(self).descriptor
        unhooked = descriptor.call_graph

        def capture(graph: Any, atype_flat: Any, *args: Any, **kwargs: Any) -> Any:
            out = unhooked(graph, atype_flat, *args, **kwargs)
            # Only the FIRST call is the energy head's. A later one is the
            # fallback re-running the backbone, and must not overwrite this.
            if "features" not in captured:
                captured.update(
                    features=out[0],
                    rot_mat=out[1],
                    edge_vec=graph.edge_vec,
                    edge_index=graph.edge_index,
                    edge_mask=graph.edge_mask,
                    n_node=graph.n_node,
                    # The short contacts the lower found on this graph, in
                    # inference: the charges weigh in the same envelope.
                    contacts=self._contacts,
                )
            return out

        def capture_state(*args: Any, **kwargs: Any) -> Any:
            out = unhooked_state(*args, **kwargs)
            # The node state with every degree, which `call_graph` drops.
            captured.setdefault("equivariant", out[1])
            return out

        self._capture_depth += 1
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(_hooked(descriptor, "call_graph", capture))
                if self.has_multipoles:
                    unhooked_state = descriptor._run_graph
                    stack.enter_context(
                        _hooked(descriptor, "_run_graph", capture_state)
                    )
                yield captured
        finally:
            self._capture_depth -= 1

    @contextlib.contextmanager
    def _enveloping_energy(
        self, atype: torch.Tensor, factor: torch.Tensor | None = None
    ) -> Iterator[dict]:
        """Weigh the fitted atomic energies by the short-contact envelope for
        one pass through the graph lower; nothing without a ``zbl`` term.

        The term stays outside the learned model: DeePMD-kit evaluates the
        network by whichever route it is configured for, and the envelope
        goes in as a weight per atom at the one place each route has for it.

        * **The autograd lower** (training; inference at ``DP_CUDA_INFER``
          0 or 1, and at 2 wherever the fused operators do not apply). The
          fitting net's output is local to ``forward_common_atomic_graph``,
          between the descriptor and the per-type energy offsets, so the
          atomic model's ``forward_atomic_graph`` is wrapped and its energy
          multiplied.
        * **The fused energy/force operators** (a compressed model at
          ``DP_CUDA_INFER=2``). They seed their backward with the per-node
          ownership mask and multiply the atomic energies by it, so the
          envelope rides in on that mask through the descriptor's
          ``fused_energy_force_graph``, and the operators return the weighted
          energies and their forces as values, unchanged themselves.

        Both hooks are set for the pass and removed afterwards, exception or
        not, as :meth:`_capturing_backbone` wraps the descriptor; DeePMD-kit
        calls the one its level selects. Either way the per-type offsets stay
        whole: an atom in contact keeps its offset and loses what the network
        added.

        ``factor`` is the envelope as a plain value per node (an evaluation:
        :meth:`ZBL.contacts`). The backward then gives ``w d fitted``, and
        the yielded dict carries
        ``enveloped``, the weighted fitted energies ``(N, 1)``, for the
        caller to add ``fitted d w`` from. Without it (training) the envelope
        is computed from the graph's own ``edge_vec``, DeePMD-kit's autograd
        leaf, and the energy backward yields the force and virial of the
        product.

        ``atype`` is the lower's own, where a padded node still has type -1.
        """
        state: dict = {}
        if self.zbl is None:
            yield state
            return
        zbl = self.zbl
        atomic_model = self.atomic_model
        descriptor = atomic_model.descriptor
        unhooked = atomic_model.forward_atomic_graph
        unfused = getattr(descriptor, "fused_energy_force_graph", None)

        def enveloped(graph: Any, atype_clamped: Any, *args: Any, **kwargs: Any) -> Any:
            ret = dict(unhooked(graph, atype_clamped, *args, **kwargs))
            energy = ret["energy"]
            weight = factor
            if weight is None:
                weight = zbl.envelope(
                    atype, graph.edge_index, graph.edge_vec, graph.edge_mask
                )
            ret["energy"] = energy * weight.to(energy.dtype).reshape(
                -1, *([1] * (energy.ndim - 1))
            )
            if factor is not None:
                state["enveloped"] = ret["energy"]
            return ret

        def fused(
            fitting: Any,
            graph: Any,
            atype_clamped: torch.Tensor,
            ownership: torch.Tensor,
            atom_bias: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if factor is None:
                return None  # no plain weight to hand over: the autograd lower
            out = unfused(
                fitting,
                graph,
                atype_clamped,
                _weighted_ownership(ownership, factor),
                atom_bias,
                *args,
                **kwargs,
            )
            if out is None:
                return None  # not served: DeePMD-kit takes the autograd lower
            energy, atom_energy, *rest = out
            state["enveloped"], kept = self._offsets_kept(
                atom_energy, atype_clamped, ownership, factor
            )
            energy = energy + frame_scalar_sum(kept, graph.n_node).to(energy.dtype)
            return (energy, atom_energy + kept.to(atom_energy.dtype), *rest)

        with contextlib.ExitStack() as hooks:
            hooks.enter_context(
                _hooked(atomic_model, "forward_atomic_graph", enveloped)
            )
            if unfused is not None:
                hooks.enter_context(
                    _hooked(descriptor, "fused_energy_force_graph", fused)
                )
            yield state

    def _offsets_kept(
        self,
        atom_energy: torch.Tensor,
        atype_clamped: torch.Tensor,
        ownership: torch.Tensor,
        envelope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """What a fused operator's atomic energies hold once it was handed
        the envelope on its ownership mask, ``(N, 1)`` float64 each: the
        weighted fitted energies ``w f``, and the part of each atom's offset
        to give back.

        The operators multiply the per-type offsets (DeePMD-kit's output
        statistics for the energy, added after the fitting net) along with
        the network; an atom in contact keeps its offset whole.
        """
        owned = ownership.to(torch.float64)
        weight = owned * envelope
        offset = self.atomic_model.out_bias[0, :, 0].to(torch.float64)[atype_clamped]
        enveloped = atom_energy - (weight * offset).unsqueeze(-1)
        return enveloped, ((owned - weight) * offset).unsqueeze(-1)

    @staticmethod
    def _usable(captured: dict) -> dict | None:
        """The capture, if it can actually carry the charge response.

        An ``edge_vec`` without ``requires_grad`` means the energy path did
        not route through the graph lower, so there is nothing to
        differentiate and the backbone has to be run again.
        """
        if captured and captured["edge_vec"].requires_grad:
            return captured
        return None

    def raw_charges(
        self, features: torch.Tensor, rot_mat: torch.Tensor | None, atype_flat: Any
    ) -> torch.Tensor:
        """The charge head's output for every node, ``(N,)``, from an
        already-computed descriptor output: raw, before the projection onto
        the total charge."""
        return self.charge_fitting.call_graph(features, atype_flat, gr=rot_mat)[
            "charge"
        ].reshape(-1)

    def charge_envelope(
        self,
        atype_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """The short-contact envelope of every node, ``(N,)``, on the graph
        the charges were computed on; None for a model without ``zbl``.

        It enters the charges as the projection's weight, ``q_i = w_i (raw_i
        + shift)``: an atom in contact carries no charge at all, whatever
        offset the head's raw output has, and the others share the deficit.
        Being a function of the edges into each node only, it is as local as
        the raw charges, so the charge-response force and the Born tensor
        differentiate it per node and per partition the same way.
        """
        envelope = self.zbl if self.zbl is not None else self.charge_shield
        if envelope is None:
            return None
        return envelope.envelope(atype_flat, edge_index, edge_vec, edge_mask)

    @property
    def has_charge_envelope(self) -> bool:
        """Whether the charges are weighted by a short-contact envelope: a
        ``zbl`` block's, or the charge shield's (never both)."""
        return self.zbl is not None or self.charge_shield is not None

    def set_charge_shield(self, radii: Any) -> None:
        """Switch an atom's charge off at a short contact, between the radii
        ``[r_lo, r_hi]`` in angstrom; None removes the shield.

        For a model whose short contacts DeePMD-kit's ZBL bridging handles
        (DPA4): the energy needs nothing from here, and the charges of two
        atoms pushed together should not be the network's. The weight is the
        envelope of :mod:`deepmd_lr.zbl`, the product over an atom's
        neighbours of a C2 step from 0 at ``r_lo`` to 1 at ``r_hi``, and it
        enters the total-charge projection as with a ``zbl`` block. By
        default the radii are the bridging's own; ``long_range.charge_shield``
        overrides them. No weights, no ``state_dict`` entry.
        """
        if radii is None:
            self.charge_shield = None
            return
        if self.zbl is not None:
            raise ValueError("a `zbl` block already weighs the charges")
        try:
            r_lo, r_hi = (float(r) for r in radii)
        except (TypeError, ValueError):
            raise ValueError(
                f"charge_shield must be [r_lo, r_hi] in angstrom, got {radii!r}"
            ) from None
        if not (math.isfinite(r_lo) and math.isfinite(r_hi) and 0.0 <= r_lo < r_hi):
            raise ValueError(
                f"charge_shield needs 0 <= r_lo < r_hi, got [{r_lo}, {r_hi}]"
            )
        rcut = float(self.get_rcut())
        if r_hi > rcut:
            raise ValueError(
                f"charge_shield ends at {r_hi:g} A, beyond the cutoff {rcut:g} A"
            )
        shield = ZBL.from_radii(
            self.get_type_map(), r_lo, r_hi, device=next(self.parameters()).device
        )
        self.charge_shield = shield.train(self.training)

    def _charges_from_features(
        self,
        features: torch.Tensor,
        rot_mat: torch.Tensor | None,
        atype: torch.Tensor,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor | None,
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        contacts: Contacts | None = None,
        return_raw: bool = False,
        equivariant: torch.Tensor | None = None,
    ) -> "torch.Tensor | _Charges":
        """Projected charges ``(nf, nloc)`` from an already-computed
        descriptor output on the graph ``edges = (edge_index, edge_vec,
        edge_mask)``; with ``return_raw`` a :class:`_Charges` with the raw
        and the local ones too. Without ``charge_spin`` every frame is
        neutral.

        ``contacts`` are the short contacts the graph lower found on this
        graph in inference: the envelope is then a plain factor, returned as
        a leaf so the caller can ask for the energy's derivative with respect
        to it and add its share from the list.
        """
        nf, nloc = atype.shape[0], atype.shape[1]
        atype_flat = atype.reshape(-1)
        raw = self.raw_charges(features, rot_mat, atype_flat).reshape(nf, nloc)
        total = charge_spin[:, 0].to(raw.dtype) if charge_spin is not None else 0.0
        if contacts is not None:
            envelope = contacts.envelope.to(raw.dtype).reshape(nf, nloc)
            envelope = envelope.clone().requires_grad_(True)
        else:
            envelope = self.charge_envelope(atype_flat, *edges)
            if envelope is not None:
                envelope = envelope.to(raw.dtype).reshape(nf, nloc)
        weight = mask
        if envelope is not None:
            weight = envelope if mask is None else envelope * mask.to(raw.dtype)
        projected = project_to_total_charge(raw, total, mask=weight)
        if not return_raw:
            return projected
        shift = projection_shift(projected, raw, weight)
        # The Born tensor differentiates the LOCAL charges, not the projected
        # ones. The projection's shift is a global mean, which couples every
        # atom to every other one and would put unbounded distant terms into
        # a quantity that must be local -- see `deepmd_lr/born.py` -- so it is
        # held fixed: the raw charges themselves without an envelope, and
        # ``w (raw + shift)`` with one.
        local = raw
        if envelope is not None:
            local = envelope * (raw + shift.detach()[:, None])
        dipole, quadrupole = self._moments(features, atype, equivariant, weight)
        return _Charges(projected, raw, local, shift, envelope, dipole, quadrupole)

    def _moments(
        self,
        features: torch.Tensor,
        atype: torch.Tensor,
        equivariant: torch.Tensor | None,
        weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Atomic dipoles ``(nf, nloc, 3)`` and quadrupoles ``(nf, nloc, 3,
        3)`` from the descriptor's ``equivariant`` node state, weighted as the
        charges are (mask and short-contact envelope, ``(nf, nloc)``); both
        None without a multipole head."""
        head = self.multipole_fitting
        if head is None:
            return None, None
        if equivariant is None:
            raise NotImplementedError(
                "the multipole head needs the descriptor's equivariant node "
                "state, which only the shared backbone capture carries; this "
                "evaluation took a path without it"
            )
        dipole, quadrupole = self.raw_moments(features, equivariant, atype.reshape(-1))
        nf = atype.shape[0]
        dipole = dipole.reshape(nf, -1, 3)
        if weight is not None:
            dipole = dipole * weight.to(dipole.dtype)[..., None]
        if quadrupole is not None:
            quadrupole = quadrupole.reshape(nf, -1, 3, 3)
            if weight is not None:
                quadrupole = quadrupole * weight.to(quadrupole.dtype)[..., None, None]
        return dipole, quadrupole

    def raw_moments(
        self,
        features: torch.Tensor,
        equivariant: torch.Tensor,
        atype_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The multipole head's dipoles ``(N, 3)`` and quadrupoles ``(N, 3,
        3)`` (None below ``lmax`` 2) on the flat node axis, before the mask
        and the envelope weigh them."""
        head = self.multipole_fitting
        coefficient = head.call_graph(features, atype_flat)[head.var_name]
        vectors, tensors = cartesian_features(
            backbone(self).descriptor, equivariant, head.lmax
        )
        return contract(coefficient, vectors, tensors)

    def _term_energy(
        self,
        term: LongRangeTerm,
        geometry: "_Strained",
        charge: torch.Tensor,
        mask: torch.Tensor,
        charges: _Charges,
    ) -> torch.Tensor:
        """The long-range energy per atom, with the dipoles and quadrupoles
        of a model that has them."""
        if not self.has_multipoles:
            return term(geometry.coord, charge, geometry.box, mask=mask)
        return term(
            geometry.coord,
            charge,
            geometry.box,
            mask=mask,
            dipole=charges.dipole,
            quadrupole=charges.quadrupole,
        )

    def _long_range(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        charge_spin: torch.Tensor | None,
        do_atomic_virial: bool,
        shared: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Long-range energy with its own force and virial.

        The derivatives are taken here rather than inherited because the
        energy is not a function of ``edge_vec``; see the module docstring.
        ``shared`` is the captured backbone (:meth:`_capturing_backbone`), or
        None to run it again (:meth:`_direct_long_range`).
        """
        if do_atomic_virial:
            # The k-space energy is collective: there is no per-atom virial
            # to attribute, so refuse rather than invent one.
            raise NotImplementedError(
                "do_atomic_virial is not supported with a long-range term: "
                "the reciprocal-space energy is collective and admits no "
                "meaningful per-atom virial decomposition."
            )
        nf, nloc = atype.shape[0], atype.shape[1]
        geometry = _strained_geometry(coord, box, nf, nloc)
        # The frame picks the sum: with no cell, the isolated one at this
        # model's alpha, whatever method the periodic frames use.
        term = (
            self.long_range_energy
            if geometry.base_box is not None
            else isolated_term_for(self.long_range_energy)
        )
        mask = charge_mask(self, atype)
        filtered = self._charge_filter_for(charge_spin, mask)
        args = (term, geometry, atype, charge_spin, mask, filtered)
        if shared is not None:
            result = self._shared_long_range(shared, *args)
        else:
            result = self._direct_long_range(*args)
        atom_energy = result["atom_energy"]
        out = {
            "atom_energy": atom_energy,
            "energy": atom_energy.sum(dim=1),
            "force": result["force"],
            # (nf, 9), DeePMD-kit's `energy_derv_c_redu` layout.
            "virial": result["virial"],
            # Reported as screened charges, sqrt(eps_inf) q; the energy used
            # the model's own charges, which it was trained with.
            "charge": result["charge"] * screening_factor(self.eps_inf),
            "charge_projection_shift": result["shift"],
        }
        if result["born"] is not None:
            out["born_charge"] = result["born"]
        # Screened like the charges: sqrt(eps_inf) times the model's own.
        for key in ("dipole", "quadrupole"):
            if result.get(key) is not None:
                out[key] = result[key] * screening_factor(self.eps_inf)
        return out

    def _charge_filter_for(
        self,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> Callable[[_Charges], torch.Tensor]:
        """Filter the projected charges, retaining their envelope for the
        fallback; return them unchanged without a filter."""
        if self.training and self.charge_filter is not None:
            raise RuntimeError(
                "a simulation-time charge_filter override is set; clear it "
                "(model.charge_filter = None) before training. A filter meant "
                "for training belongs in the config, long_range.charge_filter"
            )
        charge_filter = self.effective_charge_filter
        if charge_filter is None:
            return lambda charges: charges.projected
        if self.compute_born:
            raise NotImplementedError(
                "charge_filter cannot be combined with Born charges yet: "
                "the Born tensor is built from the unfiltered charges"
            )
        total = charge_spin[:, 0] if charge_spin is not None else 0.0
        return lambda charges: filter_charges(
            charges.projected,
            total,
            charge_filter,
            mask=mask,
            fallback_weight=(
                None
                if charges.envelope is None or charge_spin is None
                else charges.envelope * mask
            ),
        )

    def _shared_long_range(
        self,
        shared: dict,
        term: LongRangeTerm,
        geometry: _Strained,
        atype: torch.Tensor,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor,
        filtered: Callable[[_Charges], torch.Tensor],
    ) -> dict[str, torch.Tensor | None]:
        """The long-range energy and its derivatives on the captured backbone.

        DeePMD-kit detaches ``edge_vec`` into its own leaf before the backbone
        runs, so the captured features depend on that leaf alone, not on
        ``geometry``'s. The charge response is therefore taken with respect to
        ``edge_vec`` and converted by DeePMD-kit's ``edge_force_virial``,
        exactly as the short-range energy's is; ``geometry``'s leaves carry
        the sum's explicit geometry dependence. The energy head's backward
        sets ``retain_graph=True``, which keeps the captured graph traversable
        here.
        """
        nf, nloc = atype.shape[0], atype.shape[1]
        create_graph = self.training
        contacts = shared.get("contacts")
        charges = self._charges_from_features(
            shared["features"],
            shared["rot_mat"],
            atype,
            charge_spin,
            mask,
            (shared["edge_index"], shared["edge_vec"], shared["edge_mask"]),
            contacts=contacts,
            return_raw=True,
            equivariant=shared.get("equivariant"),
        )
        charge = filtered(charges)
        atom_energy = self._term_energy(term, geometry, charge, mask, charges)
        edge_vec = shared["edge_vec"]
        leaves = [geometry.leaf, geometry.eps, edge_vec]
        if contacts is not None:
            leaves.append(charges.envelope)
        grad_coord, grad_eps, g_e, *through_envelope = torch.autograd.grad(
            atom_energy.sum(),
            leaves,
            create_graph=create_graph,
            retain_graph=True,
        )
        if contacts is not None:
            # The charges took the envelope as a plain factor: its share of
            # the response comes from the list of contacts.
            g_e = add_envelope_gradient(g_e, contacts, through_envelope[0])
        response_force, _atom_virial, response_virial = edge_force_virial(
            g_e,
            edge_vec,
            shared["edge_index"],
            shared["edge_mask"],
            shared["n_node"],
        )
        born = None
        if self.compute_born:
            # A second, cheaper VJP over the tape the gradient above left
            # alive; one pass covers every edge (`deepmd_lr.born`).
            if contacts is not None and self.has_multipoles:
                raise NotImplementedError(
                    "Born tensors of atomic dipoles with a `zbl` block"
                )
            finish = None
            if contacts is not None:
                # d local / d w = raw + shift, per node.
                through = (charges.raw + charges.shift.detach()[:, None]).detach()
                finish = functools.partial(
                    add_envelope_gradient, contacts=contacts, coefficient=through
                )
            born = born_charges(
                charge,
                charges.local.reshape(-1),
                edge_vec,
                shared["edge_index"],
                shared["edge_mask"],
                create_graph=create_graph,
                eps_inf=self.eps_inf,
                finish_gradient=finish,
                dipole=(
                    None if charges.dipole is None else charges.dipole.reshape(-1, 3)
                ),
            )
        virial = (-grad_eps).reshape(nf, 3, 3) + response_virial.reshape(nf, 3, 3)
        if self.has_multipoles:
            # With moments that are tensors, neither half is invariant under
            # a rotation on its own: turning the atoms at fixed dipoles
            # changes the energy, and so does turning the dipoles. The strain
            # is symmetric and drops the first half's antisymmetric part, the
            # edge sum keeps the second's; they cancel in the true virial,
            # which is the symmetric part of what is here.
            virial = 0.5 * (virial + virial.transpose(-1, -2))
        return {
            "atom_energy": atom_energy,
            "charge": charge,
            "shift": charges.shift,
            "force": -grad_coord + response_force.reshape(nf, nloc, 3),
            "virial": virial.reshape(nf, 9),
            "born": born,
            "dipole": charges.dipole,
            "quadrupole": charges.quadrupole,
        }

    def _direct_long_range(
        self,
        term: LongRangeTerm,
        geometry: _Strained,
        atype: torch.Tensor,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor,
        filtered: Callable[[_Charges], torch.Tensor],
    ) -> dict[str, torch.Tensor | None]:
        """The long-range energy and its derivatives from a second backbone
        pass on a graph strained with ``geometry``, for when the energy path
        did not go through the graph lower and nothing was captured."""
        self._refuse_compressed_fallback()
        if self.has_multipoles:
            raise NotImplementedError(
                "a model with a multipole head needs the shared backbone "
                "capture, but this forward took the fallback path, so the "
                "descriptor's equivariant node state was not captured"
            )
        if self.compute_born:
            # No captured `edge_vec`, so no cheap route to Z. Notably taken
            # under DP_CUDA_INFER >= 2, whose fused kernel bypasses
            # `descriptor.call_graph`.
            raise NotImplementedError(
                "Born effective charges need the shared backbone capture, "
                "but this forward took the fallback path, so no edge graph "
                "was captured. The usual cause is DP_CUDA_INFER >= 2 "
                "selecting a fused energy/force kernel; unset it (or set "
                "it below 2) for field simulations."
            )
        charges = self._predict_charges(
            geometry.leaf,
            atype,
            geometry.base_box,
            charge_spin,
            mask,
            strain=geometry.strain,
            return_raw=True,
        )
        charge = filtered(charges)
        atom_energy = term(geometry.coord, charge, geometry.box, mask=mask)
        grad_coord, grad_eps = torch.autograd.grad(
            atom_energy.sum(),
            [geometry.leaf, geometry.eps],
            create_graph=self.training,
        )
        return {
            "atom_energy": atom_energy,
            "charge": charge,
            "shift": charges.shift,
            "force": -grad_coord,
            "virial": (-grad_eps).reshape(atype.shape[0], 9),
            "born": None,
        }

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Short-range DPA4C energy plus the long-range electrostatic term.

        The backbone runs once, inside ``super().forward()``, and the charge
        head reads its output (a second descriptor pass was 18% of the forward
        at 6520 atoms).
        """
        if self.has_long_range:
            charge_spin = resolve_charge_spin(self, charge_spin, atype.shape[0], coord)
        elif self._compressed_charge_spin is not None:
            # Check only, so the short-range model passes its input on exactly
            # as the standard one does: the kernel reads the folded state, and
            # a different request would be silently dropped.
            resolve_charge_spin(self, charge_spin, atype.shape[0], coord)
        if (
            self.has_long_range
            and box is not None
            and not self.long_range_energy.needs_cell
        ):
            raise ValueError(
                "this model's long-range term is method 'isolated', which sums "
                "no periodic images, but the frame has a cell. A model trained "
                "with method 'isolated' runs on systems with no periodic axis."
            )
        with self._capturing_backbone() as captured:
            short_range = super().forward(
                coord,
                atype,
                box=box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                charge_spin=charge_spin,
                **kwargs,
            )
        if not self.has_long_range:
            return short_range
        long_range = self._long_range(
            coord,
            atype,
            box,
            charge_spin,
            do_atomic_virial,
            shared=self._usable(captured),
        )
        return self._combine(short_range, long_range)

    @staticmethod
    def _combine(
        short_range: dict[str, torch.Tensor], long_range: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """The short-range output with the long-range term added, plus the
        diagnostics: charges, the neutrality projection's per-frame shift
        ``(nf,)`` (which the ``ener_lr`` loss can penalise), Born charges if
        computed, and both energies separately."""
        out = dict(short_range)
        for key in ("energy", "atom_energy", "force", "virial"):
            if key in out:
                out[key] = short_range[key] + long_range[key]
        out["charge"] = long_range["charge"]
        out["charge_projection_shift"] = long_range["charge_projection_shift"]
        for key in ("born_charge", "dipole", "quadrupole"):
            if key in long_range:
                out[key] = long_range[key]
        out["energy_long_range"] = long_range["energy"]
        out["energy_short_range"] = short_range["energy"]
        return out

    def _refuse_export(self, what: str) -> None:
        """Refuse an interface that carries no cell, so cannot see the term.

        The exported artifact is traced from ``forward_lower_graph*``, which
        takes ``edge_vec`` and no cell. Left alone, ``dp freeze`` succeeds and
        the artifact silently drops the long-range energy: -93.74 eV against
        the eager model's -55.12 on one benchmark frame, the whole 38.6 eV
        long-range contribution gone without a word.
        """
        raise NotImplementedError(
            f"{what} is not supported for a 'dp_lr' model with a long_range "
            "block. The exported graph lower receives edge vectors and no "
            "cell, so it cannot evaluate the reciprocal-space term, and the "
            "artifact would silently contain only the short-range energy. "
            "For MD use deepmd_lr.ase.LongRangeCalculator, which drives the "
            "model through its full forward. Freezing the short-range model "
            "alone (omit `long_range`) still works."
        )

    def forward_lower_graph_exportable(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse to export a model whose long-range term cannot be traced."""
        if self.has_long_range:
            self._refuse_export("Graph-lower export (`dp freeze`)")
        return super().forward_lower_graph_exportable(*args, **kwargs)

    def forward_lower_graph_exportable_with_comm(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Refuse the multi-rank export twin for the same reason."""
        if self.has_long_range:
            self._refuse_export("Graph-lower export (`dp freeze`)")
        return super().forward_lower_graph_exportable_with_comm(*args, **kwargs)

    def forward_lower(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse the dense lower path, which also carries no cell."""
        if self.zbl is not None:
            raise NotImplementedError(
                "ZBL is evaluated on the graph lower; the dense "
                "(neighbour-list) lower would drop it"
            )
        if self.has_long_range:
            self._refuse_export("The lower (neighbour-list) interface")
        return super().forward_lower(*args, **kwargs)

    def forward_lower_canonical_graph(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor,
        source: torch.Tensor,
        edge_vec: torch.Tensor,
        destination_row_ptr: torch.Tensor,
        source_row_ptr: torch.Tensor,
        source_order: torch.Tensor,
        *,
        do_atomic_virial: bool,
        spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """The compact canonical lower of a compressed model, which a
        compressed export traces and LAMMPS runs; refused with a long-range
        term, whose reciprocal part no lower carries.

        A ``zbl`` term goes around DeePMD-kit's operators here as it does on
        the graph lower. The canonical energy/force operator seeds its
        backward with the per-node ownership and multiplies the atomic
        energies by it, so the envelope is handed over on that mask (through
        the atomic model's ``_prepare_graph_nodes``, wrapped for the call),
        and the ZBL energy and the envelope's own force and virial are added
        from the list of contacts found on the same compact edge stream
        (:meth:`ZBL.canonical_contacts`): torch ops, which a trace records,
        one comparison per edge and the rest on the few edges that pass it.
        """
        if self.has_long_range:
            self._refuse_export("Canonical graph-lower export (compressed `dp freeze`)")
        lower = functools.partial(
            super().forward_lower_canonical_graph,
            atype,
            n_node,
            n_local,
            source,
            edge_vec,
            destination_row_ptr,
            source_row_ptr,
            source_order,
            do_atomic_virial=do_atomic_virial,
            spin=spin,
        )
        if self.zbl is None:
            return lower()
        n_cap = atype.shape[0]
        owned = node_ownership_mask(n_node, n_local, n_cap).to(torch.bool)
        contacts = self.zbl.canonical_contacts(
            atype, source, edge_vec, destination_row_ptr, owned
        )
        atomic_model = self.atomic_model
        unhooked = atomic_model._prepare_graph_nodes
        seen: dict = {}

        def weighted(*args: Any, **kwargs: Any) -> Any:
            atype_clamped, mask = unhooked(*args, **kwargs)
            seen.update(atype=atype_clamped, mask=mask)
            return atype_clamped, _weighted_ownership(mask, contacts.envelope)

        with _hooked(atomic_model, "_prepare_graph_nodes", weighted):
            result = lower()
        # As on the graph lower: the operator weighed the offsets with the
        # network, an atom keeps its own, and the list of contacts adds ZBL
        # and the envelope's own force and virial.
        enveloped, kept = self._offsets_kept(
            result["atom_energy"], seen["atype"], seen["mask"], contacts.envelope
        )
        extra = contact_output(contacts, enveloped, n_node, n_local, do_atomic_virial)
        added = extra["energy"] + kept
        nf = n_node.shape[0]
        result["atom_energy"] = result["atom_energy"] + added
        result["energy"] = result["energy"] + frame_scalar_sum(added, n_node)
        result["force"] = result["force"] + extra["energy_derv_r"].reshape(n_cap, 3)
        result["virial"] = result["virial"] + extra["energy_derv_c_redu"].reshape(
            nf, 3, 3
        )
        if do_atomic_virial:
            result["atom_virial"] = result["atom_virial"] + extra[
                "energy_derv_c"
            ].reshape(n_cap, 3, 3)
        result["mask"] = seen["mask"].to(torch.int32)
        return result

    def forward_common_lower_graph(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        destination_order: torch.Tensor | None = None,
        destination_row_ptr: torch.Tensor | None = None,
        source_order: torch.Tensor | None = None,
        source_row_ptr: torch.Tensor | None = None,
        destination_sorted: bool = False,
        do_atomic_virial: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """The graph lower, permitted only where the long-range term is added.

        ``DeepPot``/``dp test`` call this on a checkpoint and compiled
        training traces it, and it returns the short-range energy alone -- so
        outside a ``_capturing_backbone`` context it is refused for the same
        reason ``dp freeze`` is. ``forward`` and the partitioned driver hold
        that context and add the term themselves.

        Every compressed evaluation, serial or partitioned, passes through
        here, so the compressed kernel's preconditions are checked on every
        step (:meth:`_check_compressed_lower`).

        The short-contact term is added here, around the learned model:
        every route to an energy on a graph -- ``forward``, the partitioned
        driver, DeepPot, a graph export of the short-range model -- passes
        through this lower. The model is evaluated with the envelope as a
        weight on its atomic energies (:meth:`_enveloping_energy`), by the
        autograd lower or by DeePMD-kit's fused operators, whichever its
        ``DP_CUDA_INFER`` level selects, and the ZBL energy and the
        envelope's own force and virial are added to what it returns.
        """
        if self.has_long_range and self._capture_depth == 0:
            self._refuse_export(
                "The graph-lower interface (DeepPot, `dp test`, compiled training)"
            )
        if _is_compressed(self):
            self._check_compressed_lower(destination_order, destination_row_ptr)
        # An evaluation takes the envelope as a plain weight per atom: one
        # pass over the edges finds the short contacts (a kernel on a CUDA
        # device, torch ops in a trace or on a CPU), and all the term adds is
        # summed from that list. Training differentiates the envelope through
        # the model's own backward, as torch ops over every edge.
        zbl = self.zbl
        contacts = None
        if zbl is not None and not self.training:
            owned = None
            if n_local is not None and n_local is not n_node:
                owned = node_ownership_mask(n_node, n_local, atype.shape[0])
            contacts = zbl.contacts(atype, edge_index, edge_vec, edge_mask, owned)
        self._contacts = contacts
        factor = None if contacts is None else contacts.envelope
        try:
            with self._enveloping_energy(atype, factor) as state:
                result = super().forward_common_lower_graph(
                    atype,
                    n_node,
                    n_local,
                    edge_index,
                    edge_vec,
                    edge_mask,
                    destination_order,
                    destination_row_ptr,
                    source_order,
                    source_row_ptr,
                    destination_sorted=destination_sorted,
                    do_atomic_virial=do_atomic_virial,
                    **kwargs,
                )
        finally:
            self._contacts = None
        if contacts is not None:
            extra = contact_output(
                contacts, state["enveloped"], n_node, n_local, do_atomic_virial
            )
        elif zbl is not None:
            extra = zbl.graph_output(
                atype,
                n_node,
                n_local,
                edge_index,
                edge_vec,
                edge_mask,
                destination_order=destination_order,
                destination_row_ptr=destination_row_ptr,
                source_order=source_order,
                source_row_ptr=source_row_ptr,
                do_atomic_virial=do_atomic_virial,
            )
        else:
            extra = {}
        for key, value in extra.items():
            result[key] = result[key] + value
        return result

    def _check_compressed_lower(
        self,
        destination_order: torch.Tensor | None,
        destination_row_ptr: torch.Tensor | None,
    ) -> None:
        """Refuse a compressed graph-lower call the kernel would not serve.

        Without the operator, the right ``DP_CUDA_INFER`` or the CSR views,
        ``DescrptDPA4C.call_graph`` falls back to the uncompressed math
        without saying so. A long-range model needs level 1 (at 2 a fused
        kernel bypasses the backbone the charge head shares); a short-range
        one may use 2, DeePMD-kit's faster fused energy/force pipeline, with
        or without a ``zbl`` term. The level is checked before the CSR views:
        at level 0 upstream builds none, so the missing views are the symptom
        and the level the cause.

        Training mode is the same fall-through, and is refused here rather
        than at compression: DeePMD-kit's ``dp compress`` compresses the
        model it has just deserialized, which is in training mode, and only
        serializes it. What must not happen is a forward in that state.
        """
        if self.training:
            raise RuntimeError(
                "a compressed model was called in training mode, where the "
                "descriptor skips its compressed kernel and runs the "
                "uncompressed math without saying so. Call `.eval()` first: "
                "a compressed DPA4C is an inference snapshot."
            )
        if not _kernel_available():
            raise RuntimeError(_NO_KERNEL)
        level = cuda_infer_level()
        allowed = (1,) if self.has_long_range else (1, 2)
        if level not in allowed:
            raise RuntimeError(
                f"DP_CUDA_INFER={level} on a compressed model: at 0 the "
                "tables are built and never used"
                + (
                    ", at 2 a fused kernel bypasses the shared backbone the "
                    "charge head needs. Set DP_CUDA_INFER=1."
                    if self.has_long_range
                    else ". Set DP_CUDA_INFER=1 or 2 (2: the fused energy/force path)."
                )
            )
        if destination_order is None or destination_row_ptr is None:
            raise RuntimeError(
                "a compressed DPA4C needs the destination CSR views on "
                "every graph and this one carries none, so the descriptor "
                "would silently fall back to the uncompressed math. Graphs "
                "built by DeePMD-kit's own forward get them from "
                "`get_geo_compress()`; a hand-built graph needs "
                "`build_neighbor_graph_fast(..., with_csr=True)`."
            )

    def enable_compression(
        self,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
        *,
        charge_spin: Any = None,
    ) -> None:
        """Compress the DPA4C backbone for inference; the rest is untouched.

        Upstream's signature, kept so any caller of the DeePMD-kit method
        meets the same checks. For DPA4C only ``table_stride_1`` matters: the
        radial table spans ``[0, rcut]`` analytically, so the extrapolation,
        second stride and overflow check are all discarded upstream.

        What it commits you to (each refused rather than warned):

        * The compiled kernel must be loaded (see :mod:`deepmd_lr.compress`).
        * An inference snapshot. The model may be in either mode when it is
          compressed (``dp compress`` compresses the one it has just
          deserialized, in training mode, and serializes it), but it runs in
          evaluation mode only: a forward in training mode is refused
          (:meth:`_check_compressed_lower`), and DeePMD-kit refuses
          ``.train()`` on a compressed descriptor.
        * ``DP_CUDA_INFER`` becomes ``1`` if unset -- the one process-global
          side effect, because DeePMD-kit re-reads the variable on every call
          and offers no API for it. Level 2 is refused for a long-range model:
          its fused energy/force kernel bypasses ``descriptor.call_graph``,
          and with it the shared backbone, the charge response and the Born
          charges.
        * The frame condition is **folded into the tables**. With
          ``add_chg_spin_ebd`` the kernel reads the charge state baked in at
          this call, not the one passed at forward time; pass the run's
          ``charge_spin`` here, and a forward with a different one is refused.

        Nothing is persisted: the tables are a deterministic function of the
        weights, the stride and the charge state, and rebuild at load in well
        under a second.
        """
        level = self._check_compression_request(table_stride_1)
        state = self._charge_state_to_fold(charge_spin)
        descriptor = backbone(self).descriptor
        with _restored_on_failure(descriptor):
            super().enable_compression(
                table_extrapolate, table_stride_1, table_stride_2, check_frequency
            )
            if state is not None:
                descriptor.apply_charge_state(state)
        # Only successful compression changes the process-wide inference level.
        if level == 0:
            os.environ["DP_CUDA_INFER"] = "1"

    def _check_compression_request(self, table_stride_1: Any) -> int:
        """Refuse a compression this model could not run; the current
        ``DP_CUDA_INFER`` level."""
        if isinstance(table_stride_1, bool) or not isinstance(table_stride_1, Real):
            raise TypeError("table_stride_1 must be a positive finite number")
        if not math.isfinite(table_stride_1) or table_stride_1 <= 0:
            raise ValueError("table_stride_1 must be positive and finite")
        descriptor = backbone(self).descriptor
        if not hasattr(descriptor, "compress"):
            raise NotImplementedError(
                f"{type(descriptor).__name__} has no compressed form: "
                "compression is DeePMD-kit's tabulated kernel for DPA4C. Run "
                "this model uncompressed."
            )
        if not _kernel_available():
            raise RuntimeError(_NO_KERNEL)
        level = cuda_infer_level()
        if level >= 2 and self.has_long_range:
            raise RuntimeError(
                "DP_CUDA_INFER=2 selects the fused energy/force kernel, which "
                "bypasses `descriptor.call_graph`: the charge head could no "
                "longer share the backbone and Born charges would be refused. "
                "Set DP_CUDA_INFER=1, the level compression needs. (A model "
                "without `long_range` may use 2.)"
            )
        return level

    def _charge_state_to_fold(self, charge_spin: Any) -> list | None:
        """The validated ``[charge, multiplicity]`` to fold into the tables,
        or None for a descriptor without the frame-condition embedding."""
        if not backbone(self).add_chg_spin_ebd:
            return None
        from deepmd.utils.charge_state import validate_charge_state

        condition = torch.as_tensor(
            self.get_default_chg_spin() if charge_spin is None else charge_spin,
            dtype=torch.float64,
        ).reshape(-1)
        if condition.numel() != 2:
            raise ValueError("charge_spin must hold one [charge, multiplicity] pair")
        return list(validate_charge_state(condition.tolist()))

    def compute_or_load_stat(
        self, sampled_func: Any, *args: Any, **kwargs: Any
    ) -> None:
        """Compute statistics, tolerating a non-periodic training set.

        DeePMD-kit's ``make_stat_input`` returns an all-**zero** ``(nframes,
        9)`` box for a ``nopbc`` system rather than ``None``
        (``utils/model_stat.py``). Consumers test ``box is None``, so a
        singular cell reaches ``normalize_coord`` and training dies in
        ``linalg.inv`` before the first step. The training loop reads through
        ``DeepmdDataSetForLoader``, which does return ``None``; only the
        statistics pass sees the zeros, so they are normalized here. A zero box
        is never a legitimate cell, so this cannot mask a real problem.
        """

        checked: list = []

        def sanitized() -> Any:
            sampled = sampled_func()
            for frame in sampled:
                box = frame.get("box") if hasattr(frame, "get") else None
                if box is not None and not np.asarray(box).any():
                    frame["box"] = None
            if not checked:
                # Once: upstream may ask for the samples several times.
                checked.append(True)
                self.check_zbl_against(sampled)
            return sampled

        return super().compute_or_load_stat(sanitized, *args, **kwargs)

    def check_zbl_against(self, sampled: Any) -> None:
        """Refuse a short-contact window that reaches into ``sampled`` data.

        ``sampled`` is DeePMD-kit's statistics format, a list of systems each
        with ``coord`` ``(nframes, natoms * 3)``, ``atype`` ``(nframes,
        natoms)`` and ``box`` ``(nframes, 9)`` or None. Training calls this on
        its own samples; for a trained model given a term afterwards it is
        the caller's to run, on the data the model was fitted to
        (``examples/10-zbl/scan.py --data`` does).

        Inside a window the network is switched off, so a window that covers
        fitted geometries has the model trained with part of itself turned
        down on its own data, against several eV of ZBL: a fit that goes
        wrong without saying why. The default window sits below a bond in
        units of the covalent radii, and bonds differ (O-H in water stops at
        0.90 radius sums, Nb-O in a perovskite at 0.75), so the frames
        sampled for the statistics are checked and the pairs that need a
        window of their own are named, with one that fits.
        """
        if self.zbl is None:
            return
        zbl = self.zbl
        device = next(self.parameters()).device
        smallest, reached = 1.0, {}
        for system in sampled:
            nf = np.asarray(system["coord"]).shape[0]
            coord = torch.as_tensor(
                np.asarray(system["coord"]), dtype=torch.float64, device=device
            ).reshape(nf, -1, 3)
            atype = torch.as_tensor(
                np.asarray(system["atype"]), dtype=torch.int64, device=device
            ).reshape(nf, -1)
            box = system.get("box")
            if box is not None:
                box = torch.as_tensor(
                    np.asarray(box), dtype=torch.float64, device=device
                ).reshape(nf, 3, 3)
            graph = build_neighbor_graph_fast(coord, atype, box, zbl.max_outer)
            factor, pairs = zbl.inside(
                atype.reshape(-1), graph.edge_index, graph.edge_vec, graph.edge_mask
            )
            smallest = min(smallest, factor)
            for key, x in pairs.items():
                reached[key] = min(reached.get(key, 1.0), x)
        if smallest >= _ZBL_ON_DATA:
            return
        type_map = list(self.get_type_map())
        found, fits = [], {}
        for (i, j), x in sorted(reached.items(), key=lambda item: item[1]):
            total = float(covalent_radii[int(zbl.numbers[i])]) + float(
                covalent_radii[int(zbl.numbers[j])]
            )
            x_lo, x_hi = float(zbl.r_lo[i, j]) / total, float(zbl.r_hi[i, j]) / total
            contact = math.floor(100 * x * x_hi) / 100
            name = pair_key(type_map[i], type_map[j])
            found.append(f"{name} {x * x_hi:.2f} (window {x_lo:.2f}-{x_hi:.2f})")
            fits[name] = [round(max(contact - (x_hi - x_lo), 0.0), 2), contact]
        raise ValueError(
            "a zbl window reaches into the training data: the network would be "
            f"switched down to {smallest:.2f} of its output on a fitted "
            "geometry. Shortest contacts inside a window, in covalent-radius "
            f"sums: {', '.join(found)}. Give these pairs a window below their "
            f'data: "pairs": {json.dumps(fits)}'
        )

    def enable_hessian(self) -> None:
        """Refuse the Hessian with ZBL: DeePMD-kit's Hessian wrapper calls
        the atomic model directly, past the lower where the term is added."""
        if self.zbl is not None:
            raise NotImplementedError(_NO_HESSIAN)
        super().enable_hessian()

    def forward_ragged(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse the flat-node-axis path, which never reaches our forward.

        ``ModelWrapper`` routes a batch here whenever the dataset supplies
        ``n_node`` (``pt_expt/train/wrapper.py``), and the inherited
        implementation skips the long-range term, so training would silently
        optimize a short-range model. Only the LMDB dataset turns ragged
        batching on (``use_ragged_batches``); npy and HDF5 take ``forward``.
        Lift this once ``_long_range`` understands a flat node axis.
        """
        if self.has_long_range:
            raise NotImplementedError(
                "Ragged (flat node axis) batches are not supported for a "
                "'dp_lr' model with a long_range block. The long-range term "
                "reads a rectangular (nf, nloc) layout, and the inherited "
                "ragged path would return the short-range energy alone "
                "without raising. Use a dataset that yields rectangular "
                "batches (npy or HDF5 rather than LMDB), or drop the "
                "`long_range` block."
            )
        return super().forward_ragged(*args, **kwargs)

    def serialize(self) -> dict[str, Any]:
        """Serialize, stamping our own type so the round-trip returns this class.

        ``make_model`` delegates serialization to the atomic model, which tags
        itself ``standard``; left alone, model-level deserialization would
        route that to ``EnergyModel``.
        """
        data = super().serialize()
        if data.get("type") != _ATOMIC_MODEL_TYPE:
            # A bridged composition: remember what to rebuild it as.
            data[_WRAPPED_TYPE_KEY] = data["type"]
        data["type"] = MODEL_TYPE
        if self.has_long_range:
            data[LONG_RANGE_KEY] = {
                "charge_net": self.charge_fitting.serialize(),
                RECIPROCAL_KEY: self.long_range_energy.serialize(),
            }
            if self.has_multipoles:
                data[LONG_RANGE_KEY][MULTIPOLE_KEY] = (
                    self.multipole_fitting.serialize()
                )
            if self.trained_charge_filter is not None:
                data[LONG_RANGE_KEY]["charge_filter"] = list(self.trained_charge_filter)
            if self.charge_shield is not None:
                data[LONG_RANGE_KEY][CHARGE_SHIELD_KEY] = list(self.charge_shield.radii)
        if self.zbl is not None:
            data[ZBL_KEY] = self.zbl.serialize()
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LongRangeEnergyModel":
        """Restore the atomic model's own wire tag, then defer to the base."""
        data = data.copy()
        data["type"] = data.pop(_WRAPPED_TYPE_KEY, _ATOMIC_MODEL_TYPE)
        block = data.pop(LONG_RANGE_KEY, None)
        zbl = data.pop(ZBL_KEY, None)
        if data["type"] == _ATOMIC_MODEL_TYPE:
            model = super().deserialize(data)
        else:
            from deepmd.pt_expt.model.dp_linear_model import LinearEnergyModel

            model = cls(atomic_model_=LinearEnergyModel.deserialize(data).atomic_model)
        model.set_zbl(zbl)
        if block is not None:
            model.charge_fitting = BaseFitting.deserialize(block["charge_net"])
            if block.get(MULTIPOLE_KEY) is not None:
                model.multipole_fitting = BaseFitting.deserialize(block[MULTIPOLE_KEY])
            settings = block.get(RECIPROCAL_KEY, block.get(_LEGACY_RECIPROCAL_KEY))
            model.long_range_energy = build_term(settings, model.has_multipoles)
            model.trained_charge_filter = check_charge_filter(
                block.get("charge_filter")
            )
            model.set_charge_shield(block.get(CHARGE_SHIELD_KEY))
        return model
