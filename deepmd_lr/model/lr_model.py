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

Phase 1 scope
-------------
This class is deliberately a pure pass-through: it builds and evaluates a
standard DPA4C energy model and adds nothing. That isolates "does the plugin
mechanism work" from "is the physics right", so a numerical difference in a
later phase can only come from the physics. Phase 4 adds the charge head and
the PME energy by overriding ``forward``; see PLAN.md.
"""

import copy
import json
from typing import (
    Any,
)

from deepmd.dpmodel.model.model_factory import (
    get_model_components,
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

#: The ``model.type`` that selects this class, and the tag written into
#: checkpoints so a serialized model round-trips back to this class rather
#: than to a plain ``EnergyModel``.
MODEL_TYPE = "dp_lr"

#: Wire tag of the atomic model this class wraps. ``DPAtomicModel.serialize``
#: emits ``type: "standard"`` and its ``deserialize`` pops that key without a
#: default, so the round-trip has to put it back (see :meth:`deserialize`).
_ATOMIC_MODEL_TYPE = "standard"


@BaseModel.register(MODEL_TYPE)
class LongRangeEnergyModel(EnergyModel):
    """Energy model that will carry an explicit long-range electrostatic term.

    Reached from an input file through ``model.type: "dp_lr"``. The generic
    dispatch in ``deepmd.dpmodel.model.model_factory.get_model`` ends with
    ``base_model.get_class_by_type(model_type).get_model(data)``, so a
    plugin-registered type needs no change to DeePMD-kit itself.
    """

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

        model = cls(
            descriptor=descriptor,
            fitting=fitting,
            type_map=data["type_map"],
            atom_exclude_types=data.get("atom_exclude_types", []),
            pair_exclude_types=data.get("pair_exclude_types", []),
        )
        model.model_def_script = json.dumps(model_params_old)
        return model

    def serialize(self) -> dict[str, Any]:
        """Serialize, stamping our own type so the round-trip returns this class.

        ``make_model`` delegates serialization to the atomic model, which tags
        itself ``standard``; left alone, model-level deserialization would
        route that to ``EnergyModel``.
        """
        data = super().serialize()
        data["type"] = MODEL_TYPE
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LongRangeEnergyModel":
        """Restore the atomic model's own wire tag, then defer to the base."""
        data = data.copy()
        data["type"] = _ATOMIC_MODEL_TYPE
        return super().deserialize(data)
