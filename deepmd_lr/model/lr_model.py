# SPDX-License-Identifier: LGPL-3.0-or-later
"""The ``dp_lr`` model type: a DPA4C energy model with a long-range term.

Why this is a *model* type and not an *atomic model* type
--------------------------------------------------------
A long-range electrostatic term needs absolute coordinates and the cell.
DeePMD-kit's atomic-model contract gives neither: ``forward_common_atomic``
receives extended coordinates plus a neighbour list, and the graph route is
narrower still -- ``NeighborGraph`` states that geometry enters the model
ONLY through ``edge_vec``. The cell is available exactly one level up, at
``call_common(coord, atype, box, ...)``. So the long-range term hangs off the
top-level model, which is what this class is.

Why the long-range energy is added at the model level, not to the atomic energy
------------------------------------------------------------------------------
It is tempting to add the long-range per-atom energy into the atomic output
and let the existing machinery produce force and virial. That would be
silently wrong. On the graph route, force and virial come from
``edge_force_virial``, which differentiates the energy with respect to
``edge_vec`` and scatters the result along edges. The reciprocal-space energy does not
depend on ``edge_vec`` at all -- it depends on absolute coordinates and the
cell -- so it would contribute exactly **zero** force and virial while still
shifting the energy. Energies would move, forces would not, and the model
would quietly stop being conservative.

So the long-range branch carries its own derivatives: the energy is
differentiated with respect to the coordinates (force) and with respect to a
symmetric strain applied to coordinates and cell together (virial), then
added to the short-range results. The strain convention is DeePMD-kit's own,
read off ``edge_force_virial``: ``force = -dE/dr`` and ``virial = -dE/de``
for ``r -> (I + e) r``.

Cost
----
The charge head reads the descriptor features the energy head has already
computed: ``_capturing_backbone`` records them during the energy pass, so a
frame runs the descriptor once. A second pass, ``_predict_charges``, runs only
when the energy path did not route through the graph lower and so left nothing
to capture.
"""

import contextlib
import copy
import json
import math
import os
from numbers import Real
from typing import (
    Any,
)

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

#: Key of the reciprocal-sum settings inside that block. Named after what it
#: configures -- the reciprocal-space sum -- rather than after one way of
#: evaluating it, since `method` selects between `pme` and `ewald`.
RECIPROCAL_KEY = "reciprocal"

#: The name this block had before the rename. Still accepted, in configs and
#: in serialized checkpoints, so existing inputs and models keep working.
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

    ``dp train`` normalizes the input first and those keys never reach a
    constructor, but a raw config read straight from JSON -- which is what
    ``scripts/graft_short_range.py`` and any hand-built call to ``get_model``
    do -- still carries them. Since this model rejects unknown parameters
    rather than absorbing them, they have to be removed explicitly or an
    annotated config fails where the same config works through the CLI.
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
        #: ``(c_t, delta)`` or None. Simulation only: charges below about c_t
        #: are switched off after the projection (`filter_charges`). Set on the
        #: model by the ASE calculator, never saved, refused in training.
        self.charge_filter: tuple[float, float] | None = None
        #: True on the non-base members of a `same_as` group, whose
        #: `atomic_model` is the base's very object. A plain bool, not a
        #: module: see `deepmd_lr.same_as`.
        self._same_as_link = False
        #: How many `_capturing_backbone` contexts are open. The graph lower
        #: is allowed only inside one -- that is where the long-range term
        #: gets added -- and captures nest, so this is a count, not a flag.
        self._capture_depth = 0

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

        The delta itself is identical for every member regardless of which
        reciprocal method they use, because the regression subtracts the
        **atomic** model's prediction -- short-range only, since the long-range
        term lives one level above it. That is what makes fitting the offsets
        over the group's pooled frames well defined.
        """
        if self._same_as_link:
            return
        super().change_out_bias(merged, bias_adjust_mode=bias_adjust_mode)

    @property
    def has_long_range(self) -> bool:
        """Whether a long-range term is configured.

        Without a ``long_range`` block this model is exactly the standard
        energy model, which is what makes "Phase 4 reduces to Phase 1" a
        testable property rather than an aspiration.
        """
        return (
            self.charge_fitting is not None and self.long_range_energy is not None
        )

    def long_range_energy_for(
        self, cell: Any, pbc: Any, settings: dict[str, Any] | None = None
    ) -> torch.nn.Module | None:
        """A long-range term built for one system; this model's own is left
        as it is.

        The system's ``pbc`` picks the 3D sum or the slab one, and its cell
        the PME mesh (:func:`~deepmd_lr.lr.term.system_reciprocal`). The term
        holds no weights, so every system can have one of its own: that is
        how evaluators and calculators share a model without disturbing each
        other. The ``isolated`` term comes back unchanged, and a model without
        a long-range block gives None.

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

        Neither :class:`~deepmd_lr.lr.term.LongRangeEnergy` nor
        :class:`~deepmd_lr.lr.term.IsolatedLongRangeEnergy` contributes a
        single ``state_dict`` entry -- every weight lives in the descriptor,
        the fitting net and the charge head. So how the reciprocal sum is
        evaluated is not a property of the *checkpoint* at all, and a model
        trained on slabs can be asked for bulk energies, or for isolated ones,
        by rebinding this one attribute. That is what makes the mixed-boundary
        training of `same_as` useful at inference time.

        ``settings`` **replaces** the reciprocal block rather than merging into
        it. Merging cannot work: ``IsolatedLongRangeEnergy`` accepts every
        periodic parameter and then refuses any that was actually set, so
        folding a ``pme`` block into ``method: "isolated"`` raises. Replacement
        is the one shape that is correct for all three methods, and it leaves
        no merge semantics to get subtly wrong.

        ``alpha`` is the exception, and is inherited rather than replaced. It
        is where the Ewald split falls, and the descriptor was trained to
        supply exactly the ``erfc(alpha r)/r`` remainder left at *that* split
        -- so it defines the model as surely as the weights do. Passing a
        different one is refused rather than silently honoured.

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
        # Keep the recorded config in step, so a model that is re-serialised
        # after a swap describes the term it actually has rather than the one
        # it was trained with.
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

        charge_fitting, long_range_energy = cls._build_long_range(data, descriptor)

        model = cls(
            descriptor=descriptor,
            fitting=fitting,
            type_map=data["type_map"],
            atom_exclude_types=data.get("atom_exclude_types", []),
            pair_exclude_types=data.get("pair_exclude_types", []),
            charge_fitting=charge_fitting,
            long_range_energy=long_range_energy,
        )
        model.model_def_script = json.dumps(model_params)
        return model

    @classmethod
    def _build_long_range(
        cls, data: dict, descriptor: Any
    ) -> tuple[Any, LongRangeTerm | None]:
        """Build the charge head and the reciprocal term from ``long_range``.

        Returns ``(None, None)`` when the block is absent, which leaves this
        model behaviourally identical to the standard energy model.
        """
        block = data.get(LONG_RANGE_KEY)
        if not block:
            return None, None
        block = _strip_comments(block)

        charge_net = _strip_comments(copy.deepcopy(block.get("charge_net", {})))
        charge_net.pop("type", None)
        charge_net["ntypes"] = descriptor.get_ntypes()
        charge_net["dim_descrpt"] = descriptor.get_dim_out()
        charge_net["mixed_types"] = descriptor.mixed_types()
        charge_net["type_map"] = copy.deepcopy(data["type_map"])
        charge_fitting = BaseFitting.get_class_by_type("charge")(**charge_net)

        # `reciprocal` is the current name; `pme` is the old one, kept working
        # because it appears in every config written before the rename. The
        # block configures the reciprocal sum however it is evaluated, so
        # naming it `pme` was wrong the moment `method: "ewald"` existed.
        settings = block.get(RECIPROCAL_KEY, block.get(_LEGACY_RECIPROCAL_KEY, {}))
        long_range_energy = build_long_range_energy(
            _strip_comments(copy.deepcopy(settings))
        )
        return charge_fitting, long_range_energy


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
        """Per-atom charges satisfying the frame's total-charge constraint.

        Runs the shared descriptor and the charge head on the graph route,
        then applies the projection here rather than inside the head -- the
        head sees a flat node axis with no frame structure and so cannot
        express a per-frame reduction (see ``deepmd_lr.fitting.charge``).

        ``strain`` is a per-frame ``(nf, 3, 3)`` matrix, ``I + sym(eps)``,
        applied to the edge vectors AFTER the graph is built -- see below for
        why it is not applied to the coordinates and cell before.
        """
        descriptor = self.atomic_model.descriptor
        if _is_compressed(self):
            # At DP_CUDA_INFER=1 the compressed branch lives inside
            # `call_graph`, so the energy pass is always captured and this
            # second pass never runs. Reaching it means the capture was
            # bypassed -- refuse rather than quietly run the slow math on a
            # graph that carries no CSR views.
            raise RuntimeError(
                "the fallback charge pass was reached on a compressed model; "
                "the shared-backbone capture should have been taken. "
                "Compressed evaluation supports only the captured route -- "
                "check DP_CUDA_INFER (it must be exactly 1)."
            )
        graph = build_neighbor_graph_fast(coord, atype, box, descriptor.get_rcut())
        if strain is not None:
            # The virial needs d(charge)/d(strain). Building the graph from
            # strained coordinates and a strained cell would route that
            # derivative through the box -- and of DeePMD-kit's four
            # neighbour builders only the dense O(N^2) one carries
            # d(edge_vec)/d(box); `nv` returns 6.7e-13 for it, silently. At
            # 6520 atoms the dense build cost 63 GB and 3 s per training step.
            #
            # But the strain is affine, and an edge vector is
            # ``coord[j] + S @ box - coord[i]``: every term transforms by the
            # same ``(I + eps)^T``, so straining the finished edge vectors is
            # identical to straining the inputs, and the derivative now flows
            # through ``edge_vec`` alone. That is what lets the O(N) builder
            # be used here. The partitioned path relies on the same fact.
            #
            # Edges are flat over all frames, frames occupy contiguous node
            # blocks of ``nloc``, and both ends of an edge are in the same
            # frame -- so the destination node picks the frame's strain.
            # Dummy (masked) edges have zero vectors and stay zero.
            nloc = coord.shape[1]
            frame_of_edge = graph.edge_index[1] // nloc
            graph.edge_vec = torch.einsum(
                "eab,eb->ea", strain[frame_of_edge], graph.edge_vec
            )
        gg, rot_mat = _run_backbone(self, graph, atype.reshape(-1), charge_spin)
        # The same head and projection a captured backbone goes through. Column
        # 0 of charge_spin is the frame's total charge; absent it, the frame is
        # assumed neutral, the common case in training data.
        return self._charges_from_features(
            gg, rot_mat, atype, charge_spin, mask, return_raw=return_raw
        )

    @contextlib.contextmanager
    def _capturing_backbone(self) -> Any:
        """Yield a dict that fills with the descriptor's output as it runs.

        The charge head reads the same features as the energy head, so the
        backbone only needs to run once. Getting at them means observing
        DeePMD-kit's own forward: the tensors we need are local to
        ``forward_common_lower_graph``, which creates its own ``edge_vec``
        leaf and hands it to the descriptor. Overriding that method does not
        help -- it would still be DeePMD-kit's leaf the descriptor sees -- so
        ``descriptor.call_graph`` is wrapped for the duration of one forward
        and restored afterwards, exception or not.

        Yields an empty dict when there is no long-range term to feed.

        While a capture is open the graph lower is permitted
        (``_capture_depth``): whoever holds this context is the one adding the
        long-range term afterwards -- ``forward`` here, or the partitioned
        driver. Any other caller of the lower is one that would silently
        report the short-range energy alone.
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

        # Whether the descriptor already carried its own ``call_graph``
        # decides how to undo this. Assigning ``unhooked`` back would leave a
        # bound method sitting in the instance dict, permanently shadowing the
        # class attribute and holding a reference to the instance; deleting
        # restores the descriptor to the state it was actually in.
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
        """Charges from an already-computed descriptor output.

        The charge head is a per-atom MLP over the same features the energy
        head reads, so once the backbone has run there is nothing to repeat.
        """
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

        Coordinates and cell are strained together by a symmetric ``eps`` so
        one differentiation yields the virial in DeePMD-kit's convention.
        ``eps`` is symmetrized explicitly: an unsymmetrized strain would let
        an infinitesimal rotation leak into the derivative, and the virial of
        a rotation-invariant energy must be symmetric.
        """
        nf, nloc = atype.shape[0], atype.shape[1]
        dtype, device = coord.dtype, coord.device

        # The trainer passes coordinates flattened to (nf, nloc * 3) and the
        # cell to (nf, 9), while eager callers pass (nf, nloc, 3) and
        # (nf, 3, 3). Normalize both before any matrix product; `nloc` comes
        # from `atype`, which is unambiguous either way.
        #
        # Differentiate against fresh leaves. Detaching the geometry is safe
        # for training: the long-range energy still depends on the network
        # parameters through the charge head, so `create_graph` keeps the
        # force differentiable with respect to them.
        base_coord = coord.detach().reshape(nf, nloc, 3)
        base_box = None if box is None else box.detach().reshape(nf, 3, 3)
        eps, strain = _symmetric_strain(dtype, device, batch=(nf,))

        leaf_coord = base_coord.clone().requires_grad_(True)
        # Row vectors: r' = r (I + eps)^T, and the cell rows transform alike.
        strained_coord = torch.bmm(leaf_coord, strain.transpose(-1, -2))
        # With no cell there is nothing to strain but the coordinates, and
        # that is not a degenerate case: straining positions alone gives the
        # pair virial `sum_ij r_ij (x) f_ij`, which is exactly what
        # DeePMD-kit's own energy model already returns for a `box=None`
        # frame. Skipping the virial here would leave our total disagreeing
        # with the short-range part it is added to.
        strained_box = (
            None if base_box is None else torch.bmm(base_box, strain.transpose(-1, -2))
        )

        mask = charge_mask(self, atype)
        create_graph = self.training
        born = None
        charge_filter = check_charge_filter(self.charge_filter)
        if charge_filter is not None:
            if self.training:
                raise RuntimeError(
                    "charge_filter is a simulation-only setting; clear it "
                    "(model.charge_filter = None) before training"
                )
            if self.compute_born:
                raise NotImplementedError(
                    "charge_filter cannot be combined with Born charges yet: "
                    "the Born tensor is built from the unfiltered charges"
                )
            total = (
                charge_spin[:, 0].to(coord.dtype)
                if charge_spin is not None
                else torch.zeros(nf, dtype=coord.dtype, device=device)
            )

        def filtered(projected: torch.Tensor) -> torch.Tensor:
            if charge_filter is None:
                return projected
            return filter_charges(projected, total, charge_filter, mask=mask)

        if shared is not None:
            # SHARED BACKBONE. The energy head already ran the descriptor; its
            # output is reused here rather than recomputed, which is the same
            # 144-dim feature per atom feeding a second per-atom MLP.
            #
            # DeePMD-kit detaches `edge_vec` into its own leaf before the
            # backbone runs, so those features carry no dependence on our
            # coordinate leaf or strain -- only on that leaf. The charge
            # response is therefore taken with respect to `edge_vec` and
            # converted by DeePMD-kit's own `edge_force_virial`, exactly as it
            # converts the short-range energy's edge gradient. Its backward
            # sets `retain_graph=True`, which is what leaves the captured
            # graph traversable here.
            projected, raw_charge = self._charges_from_features(
                shared["features"],
                shared["rot_mat"],
                atype,
                charge_spin,
                mask,
                return_raw=True,
            )
            charge = filtered(projected)
            atom_energy = self.long_range_energy(
                strained_coord, charge, strained_box, mask=mask
            )
            energy = atom_energy.sum()
            edge_vec = shared["edge_vec"]
            grad_coord, grad_eps, g_e = torch.autograd.grad(
                energy,
                [leaf_coord, eps, edge_vec],
                create_graph=create_graph,
                retain_graph=True,
            )
            # `leaf_coord`/`eps` now carry ONLY the reciprocal sum's explicit
            # geometry dependence; the charge response arrives through the
            # edges. Summing the two is the same total as before.
            response_force, _atom_virial, response_virial = edge_force_virial(
                g_e,
                edge_vec,
                shared["edge_index"],
                shared["edge_mask"],
                shared["n_node"],
            )
            force = -grad_coord + response_force.reshape(nf, nloc, 3)
            virial = (-grad_eps).reshape(nf, 9) + response_virial.reshape(nf, 9)
            if self.compute_born:
                # A second, cheaper VJP over the same tape the energy gradient
                # above left alive with `retain_graph=True`. One pass covers
                # every edge because the descriptor is one-hop.
                born = born_charges(
                    charge,
                    raw_charge.reshape(-1),
                    edge_vec,
                    shared["edge_index"],
                    shared["edge_mask"],
                    create_graph=create_graph,
                    eps_inf=self.eps_inf,
                )
        else:
            # Fallback: run the backbone again on our own strained graph. Used
            # when the energy path did not route through the graph lower, so
            # nothing was captured.
            if _is_compressed(self):
                # Compressed inference is defined only on the captured route;
                # the fallback would rebuild a CSR-less graph and run the
                # uncompressed math while the model still claims otherwise.
                raise RuntimeError(
                    "a compressed model took the fallback path: the shared "
                    "backbone was not captured. The usual cause is "
                    "DP_CUDA_INFER changed after compression; it must be "
                    "exactly 1 (2 selects a fused kernel that bypasses "
                    "`descriptor.call_graph`)."
                )
            if self.compute_born:
                # The fallback rebuilds its own graph, so there is no captured
                # `edge_vec` to differentiate and no cheap route to Z. This
                # path is taken when the energy never went through the graph
                # lower -- notably under DP_CUDA_INFER >= 2, where a fused
                # kernel bypasses `descriptor.call_graph` entirely. Refuse
                # rather than return a tensor that is quietly absent or wrong.
                raise NotImplementedError(
                    "Born effective charges need the shared backbone capture, "
                    "but this forward took the fallback path, so no edge graph "
                    "was captured. The usual cause is DP_CUDA_INFER >= 2 "
                    "selecting a fused energy/force kernel; unset it (or set "
                    "it below 2) for field simulations."
                )
            projected, raw_charge = self._predict_charges(
                leaf_coord,
                atype,
                base_box,
                charge_spin,
                mask,
                strain=strain,
                return_raw=True,
            )
            charge = filtered(projected)
            atom_energy = self.long_range_energy(
                strained_coord, charge, strained_box, mask=mask
            )
            energy = atom_energy.sum()
            grad_coord, grad_eps = torch.autograd.grad(
                energy,
                [leaf_coord, eps],
                create_graph=create_graph,
            )
            force = -grad_coord
            virial = (-grad_eps).reshape(nf, 9)

        out = {
            "atom_energy": atom_energy,
            "energy": atom_energy.sum(dim=1),
            "force": force,
            # DeePMD-kit reports the virial flattened to (nf, 9), matching
            # `energy_derv_c_redu`; match that rather than the natural (nf, 3, 3).
            "virial": virial,
            # Reported as screened charges, q~ = sqrt(eps_inf) q. The energy
            # above was evaluated with the model's own `charge`, which is what
            # it was trained with; only the diagnostic output is rescaled.
            "charge": charge * screening_factor(self.eps_inf),
            "charge_projection_shift": projection_shift(projected, raw_charge, mask),
        }
        if born is not None:
            out["born_charge"] = born
        if do_atomic_virial:
            # A per-atom decomposition of a mesh-based reciprocal sum is not
            # well defined -- the k-space energy is inherently collective --
            # so refuse rather than invent an attribution.
            raise NotImplementedError(
                "do_atomic_virial is not supported with a long-range term: "
                "the reciprocal-space energy is collective and admits no "
                "meaningful per-atom virial decomposition."
            )
        return out

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
        """Short-range DPA4C energy plus the long-range electrostatic term."""
        # The backbone runs once, inside `super().forward()`, and the charge
        # head reads its output -- the second descriptor pass this replaces
        # was 18% of the forward at 6520 atoms.
        if self.has_long_range:
            charge_spin = resolve_charge_spin(self, charge_spin, atype.shape[0], coord)
        elif self._compressed_charge_spin is not None:
            # Check only, so the short-range model passes its input on exactly
            # as the standard one does: the kernel reads the folded state, and
            # a different request would be silently dropped.
            resolve_charge_spin(self, charge_spin, atype.shape[0], coord)
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
        if box is None and self.long_range_energy.needs_cell:
            raise ValueError(
                "The 'dp_lr' model has a long-range term configured with "
                f"method '{self.long_range_energy.method}', which requires a "
                "periodic cell, but box is None. For an isolated system set "
                "`long_range.reciprocal.method` to 'isolated', which sums the "
                "same erf-damped interaction directly with no cell."
            )

        long_range = self._long_range(
            coord,
            atype,
            box,
            charge_spin,
            do_atomic_virial,
            shared=self._usable(captured),
        )

        out = dict(short_range)
        out["energy"] = short_range["energy"] + long_range["energy"]
        out["atom_energy"] = short_range["atom_energy"] + long_range["atom_energy"]
        if "force" in out:
            out["force"] = short_range["force"] + long_range["force"]
        if "virial" in out:
            out["virial"] = short_range["virial"] + long_range["virial"]
        # Diagnostics: the pieces a user needs to see whether the long-range
        # term is doing anything, without re-running the model.
        out["charge"] = long_range["charge"]
        # What the neutrality projection added to every atom of each frame,
        # (nf,); the `ener_lr` loss can penalise it (`deepmd_lr.loss`).
        out["charge_projection_shift"] = long_range["charge_projection_shift"]
        if "born_charge" in long_range:
            out["born_charge"] = long_range["born_charge"]
        out["energy_long_range"] = long_range["energy"]
        out["energy_short_range"] = short_range["energy"]
        return out

    def _refuse_export(self, what: str) -> None:
        """Block the graph-lower export path when a long-range term is set.

        The exported artifact is traced from ``forward_lower_graph*``, which
        takes ``edge_vec`` and no cell -- so it cannot see the long-range
        term at all. Left alone, ``dp freeze`` succeeds and produces a model
        that silently returns only the short-range energy: measured on the
        Phase 5 benchmark, the frozen artifact came back at -93.74 eV against
        the eager model's -55.12, having dropped the entire 38.6 eV
        long-range contribution without a word.

        A loud failure is the only safe behaviour. Use the ASE calculator in
        ``deepmd_lr.ase`` for MD, which drives the eager model through
        ``forward`` and therefore has the cell.
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

        This is the method ``DeepPot``/``dp test`` call on a checkpoint and
        the one compiled training traces, and it returns the short-range
        energy alone -- so outside a ``_capturing_backbone`` context it is
        refused for the same reason ``dp freeze`` is. ``forward`` and the
        partitioned driver both hold that context and add the term themselves.

        It is also the one choke point every compressed evaluation passes
        through, serial or partitioned, so the preconditions of the
        compressed kernel are enforced here on every step: without the CSR
        views, the operator, or ``DP_CUDA_INFER=1``, ``DescrptDPA4C.call_graph``
        would fall back to the uncompressed math without saying so.
        """
        if self.has_long_range and self._capture_depth == 0:
            self._refuse_export(
                "The graph-lower interface (DeepPot, `dp test`, compiled training)"
            )
        if _is_compressed(self):
            if not _kernel_available():
                raise RuntimeError(_NO_KERNEL)
            # Level before CSR: at level 0 upstream builds no CSR at all, so
            # the missing views are the symptom and the level is the cause.
            level = cuda_infer_level()
            if level != 1:
                raise RuntimeError(
                    f"DP_CUDA_INFER={level} on a compressed model: at 0 the "
                    "tables are built and never used, at 2 a fused kernel "
                    "bypasses the shared backbone. Set DP_CUDA_INFER=1."
                )
            if destination_order is None or destination_row_ptr is None:
                raise RuntimeError(
                    "a compressed DPA4C needs the destination CSR views on "
                    "every graph and this one carries none, so the descriptor "
                    "would silently fall back to the uncompressed math. Graphs "
                    "built by DeePMD-kit's own forward get them from "
                    "`get_geo_compress()`; a hand-built graph needs "
                    "`attach_edge_csr`."
                )
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

        What it commits you to, and why each is refused rather than warned:

        * The compiled kernel must be loaded (see :mod:`deepmd_lr.compress`).
        * An evaluation snapshot: a compressed DPA4C cannot re-enter training.
        * ``DP_CUDA_INFER`` becomes ``1`` if unset -- the one process-global
          side effect here, because DeePMD-kit re-reads the variable on every
          call and offers no API for it; a second switch to remember would be
          redundant with asking for compression. Level 2 is refused for every
          model: its fused energy/force kernel bypasses
          ``descriptor.call_graph``, and with it the shared backbone, the
          charge response and the Born charges.
        * The frame condition is **folded into the tables**. With
          ``add_chg_spin_ebd`` the kernel reads the charge state baked in at
          this call, not the one passed at forward time; pass the run's
          ``charge_spin`` here, and a forward with a different one is refused.

        Nothing is persisted. The tables are a deterministic function of the
        weights, the stride and the charge state, and build in well under a
        second, so they are rebuilt at load rather than saved.
        """
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
        if level >= 2:
            raise RuntimeError(
                "DP_CUDA_INFER=2 selects the fused energy/force kernel, which "
                "bypasses `descriptor.call_graph`: the charge head could no "
                "longer share the backbone and Born charges would be refused. "
                "Set DP_CUDA_INFER=1, the level compression needs."
            )
        descriptor = self.atomic_model.descriptor
        state = None
        if self.atomic_model.add_chg_spin_ebd:
            from deepmd.utils.charge_state import validate_charge_state

            condition = torch.as_tensor(
                self.get_default_chg_spin() if charge_spin is None else charge_spin,
                dtype=torch.float64,
            ).reshape(-1)
            if condition.numel() != 2:
                raise ValueError(
                    "charge_spin must hold one [charge, multiplicity] pair"
                )
            state = list(validate_charge_state(condition.tolist()))

        # Upstream registers/replaces buffers and sets these attributes. Preserve
        # the original objects so failed construction also preserves shared refs.
        buffers = descriptor._buffers.copy()
        nonpersistent = descriptor._non_persistent_buffers_set.copy()
        was_compressed = descriptor.compress
        had_scalars = hasattr(descriptor, "_compression_scalars")
        scalars = getattr(descriptor, "_compression_scalars", None)
        had_folded = FOLDED_ATTR in descriptor.__dict__
        folded = descriptor.__dict__.get(FOLDED_ATTR)
        try:
            super().enable_compression(
                table_extrapolate, table_stride_1, table_stride_2, check_frequency
            )
            if state is not None:
                descriptor.apply_charge_state(state)
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
        # Only successful compression changes the process-wide inference level.
        if level == 0:
            os.environ["DP_CUDA_INFER"] = "1"

    def compute_or_load_stat(
        self, sampled_func: Any, *args: Any, **kwargs: Any
    ) -> None:
        """Compute statistics, tolerating a non-periodic training set.

        DeePMD-kit's ``make_stat_input`` returns an all-**zero** box for a
        system marked ``nopbc`` rather than ``None``
        (``utils/model_stat.py``; measured as a ``(nframes, 9)`` array of
        zeros). Every consumer downstream tests ``box is None`` --
        ``DescrptDPA4C.compute_input_stats`` does, explicitly -- so the test
        never fires, a singular cell reaches ``normalize_coord``, and training
        dies in ``linalg.inv`` before the first step:

            linalg.inv: The diagonal element 1 is zero ... input matrix is
            singular

        The training loop itself is unaffected: it reads through
        ``DeepmdDataSetForLoader``, which does return ``None``. Only the
        statistics pass sees the zeros. So normalise them here, where the
        sample enters this model, rather than patching DeePMD-kit.

        A zero box is never a legitimate cell -- the term itself refuses one
        as degenerate -- so treating it as "no cell" cannot mask a real
        problem.
        """
        import numpy as np

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

        ``ModelWrapper`` routes a batch to ``forward_ragged`` whenever the
        dataset supplies ``n_node`` (``pt_expt/train/wrapper.py:255-257``), and
        the inherited implementation goes straight to the base energy model --
        so the long-range term is skipped and training silently optimizes a
        pure short-range model.

        Nothing triggers this today: ``pt_expt`` only turns ragged batching on
        for datasets exposing ``use_ragged_batches``
        (``pt_expt/train/training.py:2433-2443``), which is the LMDB dataset
        alone, so npy and HDF5 runs always take the rectangular ``forward``.
        But variable-size molecules are exactly what ragged batching is for,
        and a wrong answer with no error is the worst outcome available. Fail
        instead, and lift this once ``_long_range`` understands a flat node
        axis with per-frame offsets.
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
            # Accept the old key so checkpoints written before the rename
            # still load.
            settings = block.get(RECIPROCAL_KEY, block.get(_LEGACY_RECIPROCAL_KEY))
            model.long_range_energy = build_long_range_energy(settings)
        return model
