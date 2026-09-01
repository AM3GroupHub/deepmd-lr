# SPDX-License-Identifier: LGPL-3.0-or-later
"""The ``dp_lr`` model type: a DPA4C energy model with room for a PME term.

Why this is a *model* type and not an *atomic model* type
--------------------------------------------------------
A long-range electrostatic term needs absolute coordinates and the cell.
DeePMD-kit's atomic-model contract gives neither: ``forward_common_atomic``
receives extended coordinates plus a neighbour list, and the graph route is
narrower still -- ``NeighborGraph`` states that geometry enters the model
ONLY through ``edge_vec``. The cell is available exactly one level up, at
``call_common(coord, atype, box, ...)``. So the PME term has to hang off the
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
The charge head needs the descriptor, and the short-range path consumes and
discards its own descriptor output, so a frame currently runs the descriptor
twice. Correctness first: sharing one pass means giving the atomic model a
second, non-reducible ``charge`` output, and ``init_out_stat`` sizes its bias
and std buffers from *every* output key -- which would entangle charges with
the out-stat and data-statistics machinery that has no charge labels. The
carry-through itself is known to work (``fit_output_to_model_output_graph``
does ``model_ret = dict(fit_ret.items())`` and merely skips reduction for
non-reducible keys), so the optimization is available later if profiling says
it matters.
"""

import copy
import json
from typing import (
    Any,
)

import torch
from deepmd.dpmodel.model.model_factory import (
    get_model_components,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
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

from deepmd_lr.fitting.charge import (
    project_to_total_charge,
)
from deepmd_lr.pme.term import (
    LongRangeEnergy,
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


@BaseModel.register(MODEL_TYPE)
class LongRangeEnergyModel(EnergyModel):
    """Energy model that will carry an explicit long-range electrostatic term.

    Reached from an input file through ``model.type: "dp_lr"``. The generic
    dispatch in ``deepmd.dpmodel.model.model_factory.get_model`` ends with
    ``base_model.get_class_by_type(model_type).get_model(data)``, so a
    plugin-registered type needs no change to DeePMD-kit itself.
    """

    def __init__(
        self,
        *args: Any,
        charge_fitting: Any = None,
        pme: LongRangeEnergy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Registered as submodules so they move with `.to(device)`, appear in
        # `state_dict()`, and are picked up by the optimizer.
        self.charge_fitting = charge_fitting
        self.pme = pme

    @property
    def has_long_range(self) -> bool:
        """Whether a long-range term is configured.

        Without a ``long_range`` block this model is exactly the standard
        energy model, which is what makes "Phase 4 reduces to Phase 1" a
        testable property rather than an aspiration.
        """
        return self.charge_fitting is not None and self.pme is not None

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
        model_params_old = copy.deepcopy(model_params)
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

        charge_fitting, pme = cls._build_long_range(data, descriptor)

        model = cls(
            descriptor=descriptor,
            fitting=fitting,
            type_map=data["type_map"],
            atom_exclude_types=data.get("atom_exclude_types", []),
            pair_exclude_types=data.get("pair_exclude_types", []),
            charge_fitting=charge_fitting,
            pme=pme,
        )
        model.model_def_script = json.dumps(model_params_old)
        return model

    @classmethod
    def _build_long_range(
        cls, data: dict, descriptor: Any
    ) -> tuple[Any, LongRangeEnergy | None]:
        """Build the charge head and PME term from the ``long_range`` block.

        Returns ``(None, None)`` when the block is absent, which leaves this
        model behaviourally identical to the standard energy model.
        """
        block = data.get(LONG_RANGE_KEY)
        if not block:
            return None, None

        charge_net = copy.deepcopy(block.get("charge_net", {}))
        charge_net.pop("type", None)
        charge_net["ntypes"] = descriptor.get_ntypes()
        charge_net["dim_descrpt"] = descriptor.get_dim_out()
        charge_net["mixed_types"] = descriptor.mixed_types()
        charge_net["type_map"] = copy.deepcopy(data["type_map"])
        charge_fitting = BaseFitting.get_class_by_type("charge")(**charge_net)

        pme = LongRangeEnergy(**copy.deepcopy(block.get("pme", {})))
        return charge_fitting, pme


    def _predict_charges(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        charge_spin: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-atom charges satisfying the frame's total-charge constraint.

        Runs the shared descriptor and the charge head on the graph route,
        then applies the projection here rather than inside the head -- the
        head sees a flat node axis with no frame structure and so cannot
        express a per-frame reduction (see ``deepmd_lr.fitting.charge``).
        """
        descriptor = self.atomic_model.descriptor
        graph = build_neighbor_graph(coord, atype, box, descriptor.get_rcut())
        atype_flat = atype.reshape(-1)
        # The charge head must see the SAME frame conditioning as the energy
        # head, or the two descriptor passes disagree about what system they
        # are describing. Gated on the descriptor's own capability, exactly as
        # `DPAtomicModel.forward_atomic_graph` gates it -- only the DPA4
        # family accepts the keyword, and `dp_lr` allows any descriptor.
        charge_spin_kwargs = (
            {"charge_spin": charge_spin}
            if self.atomic_model.supports_charge_spin
            else {}
        )
        gg, rot_mat = descriptor.call_graph(
            graph,
            atype_flat,
            type_embedding=descriptor.graph_type_embedding_table(),
            **charge_spin_kwargs,
        )
        raw = self.charge_fitting.call_graph(gg, atype_flat, gr=rot_mat)["charge"]
        raw = raw.reshape(coord.shape[0], coord.shape[1])

        # Column 0 of charge_spin is the frame's total charge; absent it, the
        # frame is assumed neutral, which is the common case in training data.
        total = (
            charge_spin[:, 0].to(raw.dtype)
            if charge_spin is not None
            else torch.zeros(raw.shape[0], dtype=raw.dtype, device=raw.device)
        )
        return project_to_total_charge(raw, total, mask=mask)

    def _long_range(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        charge_spin: torch.Tensor | None,
        do_atomic_virial: bool,
    ) -> dict[str, torch.Tensor]:
        """Long-range energy with its own force and virial.

        The derivatives are taken here rather than inherited because the PME
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
        base_box = box.detach().reshape(nf, 3, 3)
        eps = torch.zeros(
            (nf, 3, 3), dtype=dtype, device=device, requires_grad=True
        )
        sym = 0.5 * (eps + eps.transpose(-1, -2))
        strain = torch.eye(3, dtype=dtype, device=device).expand(nf, 3, 3) + sym

        leaf_coord = base_coord.clone().requires_grad_(True)
        # Row vectors: r' = r (I + eps)^T, and the cell rows transform alike.
        strained_coord = torch.bmm(leaf_coord, strain.transpose(-1, -2))
        strained_box = torch.bmm(base_box, strain.transpose(-1, -2))

        mask = self.atomic_model.make_atom_mask(atype)
        charge = self._predict_charges(
            strained_coord, atype, strained_box, charge_spin, mask
        )
        atom_energy = self.pme(strained_coord, charge, strained_box, mask=mask)
        energy = atom_energy.sum()

        create_graph = self.training
        grad_coord, grad_eps = torch.autograd.grad(
            energy,
            [leaf_coord, eps],
            create_graph=create_graph,
        )

        out = {
            "atom_energy": atom_energy,
            "energy": atom_energy.sum(dim=1),
            "force": -grad_coord,
            # DeePMD-kit reports the virial flattened to (nf, 9), matching
            # `energy_derv_c_redu`; match that rather than the natural (nf, 3, 3).
            "virial": (-grad_eps).reshape(nf, 9),
            "charge": charge,
        }
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
        if box is None:
            raise ValueError(
                "The 'dp_lr' model has a long-range term configured, which "
                "requires a periodic cell, but box is None."
            )

        long_range = self._long_range(
            coord, atype, box, charge_spin, do_atomic_virial
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
                "pme": self.pme.serialize(),
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
            model.pme = LongRangeEnergy.deserialize(block["pme"])
        return model
