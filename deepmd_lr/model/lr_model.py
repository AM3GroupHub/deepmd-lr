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
from deepmd.dpmodel.model.model_factory import (
    get_model_components,
)
from deepmd.dpmodel.utils.neighbor_graph.derivatives import (
    edge_force_virial,
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
from deepmd_lr.lr.term import (
    LongRangeTerm,
    build_long_range_energy,
    isolated_term_for,
    system_reciprocal,
)
from deepmd_lr.neighbors import (
    build_neighbor_graph_fast,
)

#: The ``model.type`` that selects this class, and the tag written into
#: checkpoints so a serialized model round-trips back to this class rather
#: than to a plain ``EnergyModel``.
MODEL_TYPE = "dp_lr"

#: Wire tag of the atomic model this class wraps. ``DPAtomicModel.serialize``
#: emits ``type: "standard"`` and its ``deserialize`` pops that key without a
#: default, so the round-trip has to put it back (see :meth:`deserialize`).
_ATOMIC_MODEL_TYPE = "standard"

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
    descriptor = model.atomic_model.descriptor
    kwargs = (
        {"charge_spin": charge_spin}
        if model.atomic_model.supports_charge_spin
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
    return bool(getattr(model.atomic_model.descriptor, "compress", False))


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
        return folded_charge_spin(self.atomic_model.descriptor)

    @_compressed_charge_spin.setter
    def _compressed_charge_spin(self, value: tuple[float, float] | None) -> None:
        setattr(self.atomic_model.descriptor, FOLDED_ATTR, value)

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
        return build_long_range_energy(system_reciprocal(base, cell, pbc))

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
        self.long_range_energy = build_long_range_energy(settings)
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

        descriptor, fitting, fitting_type = get_model_components(
            data,
            descriptor_base=BaseDescriptor,
            fitting_base=BaseFitting,
            backend_name="pt_expt",
        )
        if fitting_type != "ener":
            raise ValueError(
                f"The '{MODEL_TYPE}' model is an energy model and requires "
                f"fitting_net.type 'ener', got '{fitting_type}'. A long-range "
                "electrostatic term is only defined as an energy contribution."
            )

        charge_fitting, long_range_energy, trained_filter = cls._build_long_range(
            data, descriptor
        )
        model = cls(
            descriptor=descriptor,
            fitting=fitting,
            type_map=data["type_map"],
            atom_exclude_types=data.get("atom_exclude_types", []),
            pair_exclude_types=data.get("pair_exclude_types", []),
            charge_fitting=charge_fitting,
            long_range_energy=long_range_energy,
        )
        model.trained_charge_filter = trained_filter
        model.model_def_script = json.dumps(model_params)
        return model

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
        long_range_energy = build_long_range_energy(
            _strip_comments(copy.deepcopy(settings))
        )
        trained_filter = check_charge_filter(block.get("charge_filter"))
        return charge_fitting, long_range_energy, trained_filter

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Per-atom charges from a second backbone pass, projected onto the
        frame's total charge (a per-frame reduction the head, seeing a flat
        node axis, cannot express).

        ``strain`` is a per-frame ``(nf, 3, 3)`` matrix, ``I + sym(eps)``,
        applied to the edge vectors AFTER the graph is built.
        """
        descriptor = self.atomic_model.descriptor
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
            gg, rot_mat, atype, charge_spin, mask, return_raw=return_raw
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

        descriptor = self.atomic_model.descriptor
        unhooked = descriptor.call_graph

        def capture(graph: Any, atype_flat: Any, *args: Any, **kwargs: Any) -> Any:
            out = unhooked(graph, atype_flat, *args, **kwargs)
            # Only the FIRST call is the energy head's. A later one is the
            # fallback re-running the backbone, and must not overwrite this.
            if not captured:
                captured.update(
                    features=out[0],
                    rot_mat=out[1],
                    edge_vec=graph.edge_vec,
                    edge_index=graph.edge_index,
                    edge_mask=graph.edge_mask,
                    n_node=graph.n_node,
                )
            return out

        # Undo by deleting unless the descriptor carried its own `call_graph`:
        # assigning `unhooked` back would leave a bound method in the instance
        # dict, shadowing the class attribute and holding the instance.
        had_own = "call_graph" in descriptor.__dict__
        descriptor.call_graph = capture
        self._capture_depth += 1
        try:
            yield captured
        finally:
            self._capture_depth -= 1
            if had_own:
                descriptor.call_graph = unhooked
            else:
                del descriptor.call_graph

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

    def _charges_from_features(
        self,
        features: torch.Tensor,
        rot_mat: torch.Tensor | None,
        atype: torch.Tensor,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor | None,
        *,
        return_raw: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Projected charges ``(nf, nloc)`` from an already-computed
        descriptor output, and with ``return_raw`` the raw ones too. Without
        ``charge_spin`` every frame is neutral."""
        nf, nloc = atype.shape[0], atype.shape[1]
        atype_flat = atype.reshape(-1)
        raw = self.charge_fitting.call_graph(features, atype_flat, gr=rot_mat)[
            "charge"
        ]
        raw = raw.reshape(nf, nloc)
        total = (
            charge_spin[:, 0].to(raw.dtype)
            if charge_spin is not None
            else torch.zeros(nf, dtype=raw.dtype, device=raw.device)
        )
        projected = project_to_total_charge(raw, total, mask=mask)
        # The Born tensor differentiates the RAW charges, not these. The
        # projection subtracts a global mean, which couples every atom to every
        # other one and would put unbounded distant terms into a quantity that
        # must be local -- see `deepmd_lr/born.py`.
        return (projected, raw) if return_raw else projected

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
        filtered = self._charge_filter_for(charge_spin, nf, coord, mask)
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
            "charge_projection_shift": projection_shift(
                result["projected"], result["raw"], mask
            ),
        }
        if result["born"] is not None:
            out["born_charge"] = result["born"]
        return out

    def _charge_filter_for(
        self,
        charge_spin: torch.Tensor | None,
        nf: int,
        coord: torch.Tensor,
        mask: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """The charge filter this evaluation applies, as ``projected ->
        charge`` (the identity without one)."""
        if self.training and self.charge_filter is not None:
            raise RuntimeError(
                "a simulation-time charge_filter override is set; clear it "
                "(model.charge_filter = None) before training. A filter meant "
                "for training belongs in the config, long_range.charge_filter"
            )
        charge_filter = self.effective_charge_filter
        if charge_filter is None:
            return lambda projected: projected
        if self.compute_born:
            raise NotImplementedError(
                "charge_filter cannot be combined with Born charges yet: "
                "the Born tensor is built from the unfiltered charges"
            )
        total = (
            charge_spin[:, 0].to(coord.dtype)
            if charge_spin is not None
            else torch.zeros(nf, dtype=coord.dtype, device=coord.device)
        )
        return lambda projected: filter_charges(
            projected, total, charge_filter, mask=mask
        )

    def _shared_long_range(
        self,
        shared: dict,
        term: LongRangeTerm,
        geometry: _Strained,
        atype: torch.Tensor,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor,
        filtered: Callable[[torch.Tensor], torch.Tensor],
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
        projected, raw = self._charges_from_features(
            shared["features"],
            shared["rot_mat"],
            atype,
            charge_spin,
            mask,
            return_raw=True,
        )
        charge = filtered(projected)
        atom_energy = term(geometry.coord, charge, geometry.box, mask=mask)
        edge_vec = shared["edge_vec"]
        grad_coord, grad_eps, g_e = torch.autograd.grad(
            atom_energy.sum(),
            [geometry.leaf, geometry.eps, edge_vec],
            create_graph=create_graph,
            retain_graph=True,
        )
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
            # alive; one pass covers every edge because the descriptor is
            # one-hop.
            born = born_charges(
                charge,
                raw.reshape(-1),
                edge_vec,
                shared["edge_index"],
                shared["edge_mask"],
                create_graph=create_graph,
                eps_inf=self.eps_inf,
            )
        return {
            "atom_energy": atom_energy,
            "charge": charge,
            "projected": projected,
            "raw": raw,
            "force": -grad_coord + response_force.reshape(nf, nloc, 3),
            "virial": (-grad_eps).reshape(nf, 9) + response_virial.reshape(nf, 9),
            "born": born,
        }

    def _direct_long_range(
        self,
        term: LongRangeTerm,
        geometry: _Strained,
        atype: torch.Tensor,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor,
        filtered: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, torch.Tensor | None]:
        """The long-range energy and its derivatives from a second backbone
        pass on a graph strained with ``geometry``, for when the energy path
        did not go through the graph lower and nothing was captured."""
        self._refuse_compressed_fallback()
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
        projected, raw = self._predict_charges(
            geometry.leaf,
            atype,
            geometry.base_box,
            charge_spin,
            mask,
            strain=geometry.strain,
            return_raw=True,
        )
        charge = filtered(projected)
        atom_energy = term(geometry.coord, charge, geometry.box, mask=mask)
        grad_coord, grad_eps = torch.autograd.grad(
            atom_energy.sum(),
            [geometry.leaf, geometry.eps],
            create_graph=self.training,
        )
        return {
            "atom_energy": atom_energy,
            "charge": charge,
            "projected": projected,
            "raw": raw,
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
        if "born_charge" in long_range:
            out["born_charge"] = long_range["born_charge"]
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
        if self.has_long_range:
            self._refuse_export("The lower (neighbour-list) interface")
        return super().forward_lower(*args, **kwargs)

    def forward_lower_canonical_graph(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse the compact canonical lower, traced for a compressed export."""
        if self.has_long_range:
            self._refuse_export("Canonical graph-lower export (compressed `dp freeze`)")
        return super().forward_lower_canonical_graph(*args, **kwargs)

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
        *args: Any,
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
        """
        if self.has_long_range and self._capture_depth == 0:
            self._refuse_export(
                "The graph-lower interface (DeepPot, `dp test`, compiled training)"
            )
        if _is_compressed(self):
            self._check_compressed_lower(destination_order, destination_row_ptr)
        return super().forward_common_lower_graph(
            atype,
            n_node,
            n_local,
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            *args,
            **kwargs,
        )

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
        one may use 2, DeePMD-kit's faster fused energy/force pipeline. The
        level is checked before the CSR views: at level 0 upstream builds
        none, so the missing views are the symptom and the level the cause.
        """
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
        * An evaluation snapshot: a compressed DPA4C cannot re-enter training.
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
        descriptor = self.atomic_model.descriptor
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
        if not _kernel_available():
            raise RuntimeError(_NO_KERNEL)
        if self.training:
            raise RuntimeError(
                "compress an evaluation snapshot: call `.eval()` first. A "
                "compressed DPA4C cannot re-enter training mode."
            )
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
        if not self.atomic_model.add_chg_spin_ebd:
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

        def sanitized() -> Any:
            sampled = sampled_func()
            for frame in sampled:
                box = frame.get("box") if hasattr(frame, "get") else None
                if box is not None and not np.asarray(box).any():
                    frame["box"] = None
            return sampled

        return super().compute_or_load_stat(sanitized, *args, **kwargs)

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
        data["type"] = MODEL_TYPE
        if self.has_long_range:
            data[LONG_RANGE_KEY] = {
                "charge_net": self.charge_fitting.serialize(),
                RECIPROCAL_KEY: self.long_range_energy.serialize(),
            }
            if self.trained_charge_filter is not None:
                data[LONG_RANGE_KEY]["charge_filter"] = list(self.trained_charge_filter)
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LongRangeEnergyModel":
        """Restore the atomic model's own wire tag, then defer to the base."""
        data = data.copy()
        data["type"] = _ATOMIC_MODEL_TYPE
        block = data.pop(LONG_RANGE_KEY, None)
        model = super().deserialize(data)
        if block is not None:
            model.charge_fitting = BaseFitting.deserialize(block["charge_net"])
            settings = block.get(RECIPROCAL_KEY, block.get(_LEGACY_RECIPROCAL_KEY))
            model.long_range_energy = build_long_range_energy(settings)
            model.trained_charge_filter = check_charge_filter(
                block.get("charge_filter")
            )
        return model
