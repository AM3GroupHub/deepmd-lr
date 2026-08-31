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
    fitting_args_plugin,
    fitting_variant_type_args,
    model_args_plugin,
)

from deepmd_lr.fitting.charge import (
    CHARGE_VAR_NAME,
)
from deepmd_lr.model.lr_model import (
    LONG_RANGE_KEY,
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
            _long_range_args(),
        ],
        doc=(
            "Energy model with an explicit long-range electrostatic term "
            "(PME, via nvalchemi-toolkit-ops). Without a `long_range` block "
            "it is exactly the standard energy model."
        ),
    )


@fitting_args_plugin.register(CHARGE_VAR_NAME)
def charge_fitting_args() -> list[Argument]:
    """Schema for ``fitting_net.type: "charge"``.

    The same knobs as the energy fitting net, minus the ones that are
    energy-specific (``atom_ener``, ``tot_ener_zero``) or fixed by this head
    (``var_name``, ``dim_out``). Returning a list rather than an ``Argument``
    lets ``ArgsPlugin`` name the block from the registration key, which is how
    the built-in fitting nets are declared.
    """
    return [
        Argument(
            "neuron",
            list[int],
            optional=True,
            default=[128, 128, 128],
            doc="The number of neurons in each hidden layer of the charge head.",
        ),
        Argument(
            "activation_function",
            str,
            optional=True,
            default="tanh",
            doc="The activation function in the charge head.",
        ),
        Argument(
            "resnet_dt",
            bool,
            optional=True,
            default=True,
            doc='Whether to use a "Timestep" in the skip connection.',
        ),
        Argument(
            "precision",
            str,
            optional=True,
            default="default",
            doc="The precision of the charge head parameters.",
        ),
        Argument(
            "numb_fparam",
            int,
            optional=True,
            default=0,
            doc="The dimension of the frame parameter.",
        ),
        Argument(
            "numb_aparam",
            int,
            optional=True,
            default=0,
            doc="The dimension of the atomic parameter.",
        ),
        Argument(
            "trainable",
            [bool, list[bool]],
            optional=True,
            default=True,
            doc="Whether the parameters in the charge head are trainable.",
        ),
        Argument(
            "seed",
            [int, list[int], None],
            optional=True,
            doc="Random seed for parameter initialization of the charge head.",
        ),
        Argument(
            "exclude_types",
            list[int],
            optional=True,
            default=[],
            doc="Atoms of the listed types are given zero charge.",
        ),
    ]


def _long_range_args() -> Argument:
    """Schema for the optional ``long_range`` block.

    Optional by design: omitting it leaves ``dp_lr`` behaviourally identical
    to the standard energy model, which is what makes the "adding the
    long-range term is the only thing that changed" comparison meaningful.
    """
    charge_net = Argument(
        "charge_net",
        dict,
        charge_fitting_args(),
        doc="The head predicting per-atom charges from the shared descriptor.",
    )
    pme = Argument(
        "pme",
        dict,
        [
            Argument(
                "alpha",
                float,
                optional=True,
                default=0.4,
                doc="Ewald splitting parameter in inverse angstrom. Unlike a "
                "full Ewald sum this is a real modelling choice: it sets how "
                "much of the Coulomb interaction is left to the short-range "
                "network (erfc(alpha r) / r) and how much this term supplies. "
                "It must be large enough that the network's share has decayed "
                "inside the descriptor cutoff -- alpha 0.4 leaves erfc(2.4) = "
                "7e-4 at a 6 angstrom cutoff -- and must match between "
                "training and any later MD.",
            ),
            Argument(
                "mesh_spacing",
                [float, None],
                optional=True,
                default=None,
                doc="Reciprocal-mesh spacing in angstrom; the mesh is "
                "ceil(L / spacing), a step function of the cell, so prefer "
                "`mesh_dimensions` if the cell can change. Defaults to 1.0 "
                "when no mesh is given.",
            ),
            Argument(
                "mesh_dimensions",
                [list[int], None],
                optional=True,
                default=None,
                doc="Fixed reciprocal-mesh dimensions [nx, ny, nz], "
                "overriding `mesh_spacing`. Prefer this under NPT: a "
                "spacing-derived mesh jumps a whole grid point when a cell "
                "length crosses a multiple of the spacing, making the energy "
                "discontinuous there.",
            ),
            Argument(
                "spline_order",
                int,
                optional=True,
                default=6,
                doc="B-spline interpolation order. Must be between 3 and 6: "
                "at 7 and above the underlying library silently drops the "
                "reciprocal-space sum.",
            ),
        ],
        optional=True,
        default={},
        doc="Reciprocal-space (long-range) electrostatics, evaluated by "
        "particle-mesh Ewald. Only the long-range half of the Ewald "
        "decomposition is computed here; the screened real-space half is "
        "already inside the descriptor cutoff and is left to the network.",
    )
    return Argument(
        LONG_RANGE_KEY,
        dict,
        [charge_net, pme],
        optional=True,
        default=None,
        doc="The long-range electrostatic term. Omit for a pure short-range "
        "model identical to the standard energy model.",
    )
