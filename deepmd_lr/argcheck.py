# SPDX-License-Identifier: LGPL-3.0-or-later
"""Input-schema arguments for the ``dp_lr`` model type.

DeePMD-kit validates and normalizes the input JSON against a ``dargs``
schema before building anything, so a model type that is not in that schema
is rejected at ``dp train`` time no matter how it is registered elsewhere.
``deepmd.utils.argcheck`` exposes its ``ArgsPlugin`` registries at module
level for exactly this, so the schema extends from a plugin without
patching DeePMD-kit.
"""

from dargs import (
    Argument,
)
from deepmd.utils.argcheck import (
    descrpt_variant_type_args,
    fitting_variant_type_args,
    model_args_plugin,
)

from deepmd_lr.model.lr_model import (
    MODEL_TYPE,
)


@model_args_plugin.register(MODEL_TYPE)
def dp_lr_model_args() -> Argument:
    """Schema for ``model.type: "dp_lr"``.

    Mirrors the built-in ``standard`` model -- a descriptor plus a fitting
    net -- and reuses the same variant helpers, so every descriptor and
    fitting DeePMD-kit knows about stays available here and new ones appear
    automatically. The long-range block (charge head and PME parameters)
    is added in Phase 4; keeping it out now means a Phase 1 config is
    exactly a standard config with a different type, which is what makes
    the baseline comparison meaningful.
    """
    return Argument(
        MODEL_TYPE,
        dict,
        [
            Argument(
                "descriptor",
                dict,
                [],
                [descrpt_variant_type_args()],
                doc="The descriptor of atomic environment.",
            ),
            Argument(
                "fitting_net",
                dict,
                [],
                [fitting_variant_type_args()],
                doc="The fitting of physical properties. Must be an energy fitting.",
            ),
        ],
        doc=(
            "Energy model with an explicit long-range electrostatic term "
            "(PME, via nvalchemi-toolkit-ops). Currently equivalent to the "
            "standard model; the long-range term is added in a later phase."
        ),
    )
