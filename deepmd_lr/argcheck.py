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
    loss_args_plugin,
    loss_ener,
    model_args_plugin,
)

from deepmd_lr.fitting.charge import (
    CHARGE_VAR_NAME,
)
from deepmd_lr.loss import (
    LOSS_TYPE,
)
from deepmd_lr.model.lr_model import (
    _LEGACY_RECIPROCAL_KEY,
    LONG_RANGE_KEY,
    MODEL_TYPE,
    RECIPROCAL_KEY,
)
from deepmd_lr.same_as import (
    CHARGE_SAME_AS_KEY,
    SAME_AS_KEY,
)


@model_args_plugin.register(MODEL_TYPE)
def dp_lr_model_args() -> Argument:
    """Schema for ``model.type: "dp_lr"``.

    Mirrors the built-in ``standard`` model -- a descriptor plus a fitting
    net -- and reuses the same variant helpers, so every descriptor and
    fitting DeePMD-kit knows about stays available here and new ones appear
    automatically.

    ``model_branch_alias`` and ``info`` are carried over from ``standard``
    because a ``dp_lr`` branch inside a multi-task ``model_dict`` is
    validated against *this* schema, not that one. Without them, writing
    either key in a branch fails normalization with "undefined key ... is
    not allowed in strict mode" -- the model is unusable in exactly the
    place those keys exist for.
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
            Argument(
                "model_branch_alias",
                list[str],
                optional=True,
                default=[],
                doc=(
                    "List of aliases for this model branch. Multiple aliases "
                    "can be defined, and any alias can reference this branch "
                    "throughout the model usage. Used only in multi-task "
                    "models."
                ),
            ),
            Argument(
                "info",
                dict,
                optional=True,
                default={},
                doc=(
                    "Dictionary of metadata for this model or model branch. "
                    "Store arbitrary key-value pairs with model- or "
                    "branch-specific information. Used in both single- and "
                    "multi-task models."
                ),
            ),
            Argument(
                SAME_AS_KEY,
                str,
                optional=True,
                default=None,
                doc=(
                    "Name of another branch in `model_dict` that this branch "
                    "is a copy of. The two share every trainable parameter -- "
                    "descriptor, fitting net, charge head and per-type energy "
                    "offsets are the same objects -- and differ only in "
                    "`long_range.reciprocal`. This is how one set of weights "
                    "is trained on a dataset mixing bulk and slab systems: "
                    "the long-range term holds no weights, so only the sum "
                    "over periodic images has to differ. Such a branch may "
                    "set only `long_range`, `model_branch_alias` and `info`; "
                    "everything else is inherited."
                ),
            ),
            Argument(
                CHARGE_SAME_AS_KEY,
                str,
                optional=True,
                default=None,
                doc=(
                    "Name of another base branch in `model_dict` whose charge "
                    "head this branch uses. This branch keeps its own energy "
                    "fitting net and energy zero (a different level of theory), "
                    "but the per-atom charges -- and with them the long-range "
                    "term, which holds no weights -- are one shared model. Both "
                    "branches must share the descriptor (one `shared_dict` "
                    "entry), and write an identical `charge_net` block and the "
                    "same alpha; `same_as` links of either branch follow it."
                ),
            ),
            Argument(
                "zbl",
                dict,
                [
                    Argument(
                        "r_inner",
                        [int, float],
                        doc="Full ZBL below this distance (Angstrom).",
                    ),
                    Argument(
                        "r_outer",
                        [int, float],
                        doc=(
                            "Zero ZBL above this distance (Angstrom); "
                            "no larger than descriptor.rcut."
                        ),
                    ),
                ],
                optional=True,
                doc=(
                    "Fixed additive ZBL prior with a quintic switch; enabled only "
                    "when provided. Not use_srtab or descriptor clamping."
                ),
            ),
            _long_range_args(),
        ],
        doc=(
            "Energy model with an explicit long-range electrostatic term "
            "(PME or Ewald via nvalchemi-toolkit-ops, or the isolated term for "
            "systems without a cell). Without a `long_range` block it is "
            "exactly the standard energy model."
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
    reciprocal = Argument(
        RECIPROCAL_KEY,
        dict,
        [
            Argument(
                "method",
                str,
                optional=True,
                default="pme",
                doc="How to evaluate the long-range sum. Under periodic "
                "boundary conditions: 'pme' (mesh + FFT, O(N log N)) or "
                "'ewald' (explicit k-vector sum, exact but much slower -- "
                "114 ms vs 3 ms at 2048 atoms); both compute the same "
                "quantity, so use 'ewald' to check a 'pme' setting, not for "
                "production. For a system with no cell: 'isolated', which "
                "sums q_i q_j erf(alpha r) / r directly over every pair. That "
                "is the same quantity again -- the periodic result converges "
                "onto it as the cell grows -- so a shared descriptor sees the "
                "same erfc(alpha r) / r remainder either way. 'isolated' "
                "takes only `alpha`; every other setting here describes a sum "
                "over periodic images and is refused.",
            ),
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
                doc="method='pme' only. Reciprocal-mesh spacing in angstrom; "
                "the mesh is ceil(L / spacing), a step function of the cell, "
                "so prefer `mesh_dimensions` if the cell can change. Defaults "
                "to 1.0 when no mesh is given.",
            ),
            Argument(
                "mesh_dimensions",
                [list[int], None],
                optional=True,
                default=None,
                doc="method='pme' only. Fixed reciprocal-mesh dimensions "
                "[nx, ny, nz], overriding `mesh_spacing`. Prefer this under "
                "NPT: a spacing-derived mesh jumps a whole grid point when a "
                "cell length crosses a multiple of the spacing, making the "
                "energy discontinuous there.",
            ),
            Argument(
                "spline_order",
                int,
                optional=True,
                default=6,
                doc="method='pme' only. B-spline interpolation order, between "
                "3 and 6: at 7 and above the underlying library silently "
                "drops the reciprocal-space sum.",
            ),
            Argument(
                "k_cutoff",
                [float, None],
                optional=True,
                default=None,
                doc="method='ewald' only. Largest |k| included, in inverse "
                "angstrom. null derives it from `alpha` and `accuracy` as "
                "2 alpha sqrt(-ln(accuracy)).",
            ),
            Argument(
                "accuracy",
                float,
                optional=True,
                default=1e-6,
                doc="method='ewald' only. Target accuracy used to derive "
                "`k_cutoff` when it is not given.",
            ),
            Argument(
                "slab_correction",
                bool,
                optional=True,
                default=False,
                doc="Add the Yeh-Berkowitz correction for a 2D-periodic slab. "
                "Ewald assumes 3D periodicity, so a slab in a padded box "
                "still interacts with its images through the vacuum; for a "
                "slab with a net dipole that error is large and adding more "
                "vacuum barely helps. Requires `pbc` with one axis false. "
                "Leave off for a genuinely 3D-periodic system.",
            ),
            Argument(
                "pbc",
                [list[bool], None],
                optional=True,
                default=None,
                doc="Periodicity per axis, e.g. [true, true, false] for a slab "
                "stacked along z. Defaults to fully periodic. Must have "
                "exactly one false entry when `slab_correction` is on, and "
                "none when it is off.",
            ),
        ],
        alias=[_LEGACY_RECIPROCAL_KEY],
        optional=True,
        default={},
        doc="Reciprocal-space (long-range) electrostatics. Only the "
        "long-range half of the Ewald decomposition is computed here; the "
        "screened real-space half is already inside the descriptor cutoff "
        "and is left to the network. Parameters belonging to the other "
        "`method` are rejected rather than ignored.",
    )
    charge_filter = Argument(
        "charge_filter",
        [list[float], None],
        optional=True,
        default=None,
        doc="[c_t, delta] in e, or null. After the neutrality projection, charges "
        "with |q| <= c_t - delta are set exactly to zero and those above c_t + "
        "delta kept, with a C^3 polynomial between; the total is restored on the "
        "atoms that keep charge. Part of the model: applied in training and in "
        "every simulation, and saved with it, so a bulk whose atoms carry a "
        "little charge each carries none. Born charges and fields are refused "
        "with it. See deepmd_lr.fitting.charge.filter_charges.",
    )
    return Argument(
        LONG_RANGE_KEY,
        dict,
        [charge_net, reciprocal, charge_filter],
        optional=True,
        default=None,
        doc="The long-range electrostatic term. Omit for a pure short-range "
        "model identical to the standard energy model.",
    )


@loss_args_plugin.register(LOSS_TYPE)
def ener_lr_loss_args() -> list[Argument]:
    """The ``ener`` loss's arguments plus the charge-neutrality prefactors."""
    doc = (
        "Prefactor of the charge-neutrality term, the mean over frames of the "
        "squared shift the total-charge projection applied to every atom "
        "(absolute shift under `loss_func: mae`). It drives the raw charges "
        "to satisfy the constraint by themselves, so that a constant the "
        "projection would remove cannot differ between materials and charge "
        "an interface's bulks. Needs a `dp_lr` model with a `long_range` "
        "block. Scheduled as the other prefactors: limit + (start - limit) * "
        "lr / start_lr."
    )
    return [
        *loss_ener(),
        Argument("start_pref_qn", [float, int], optional=True, default=0.0,
                 doc=f"{doc} This is the value at the start of training."),
        Argument("limit_pref_qn", [float, int], optional=True, default=0.0,
                 doc=f"{doc} This is the value at the end of training."),
    ]
