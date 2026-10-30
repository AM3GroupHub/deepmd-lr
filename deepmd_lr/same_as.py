# SPDX-License-Identifier: MIT
"""``same_as``: several branches that are one model with different boundaries.

The problem
-----------
A branch's long-range settings (``slab_correction``, ``pbc``) are fixed at
construction, so one branch cannot train on both bulk and slab frames. But
**the long-range term holds no trainable weights** -- neither
:class:`~deepmd_lr.lr.term.LongRangeEnergy` nor
:class:`~deepmd_lr.lr.term.IsolatedLongRangeEnergy` contributes a single
``state_dict`` entry -- so one set of weights serves every boundary
condition; only the sum over periodic images differs.

So a branch can declare itself a copy of another, writing only what differs::

    "model_dict": {
      "bulk": { "type_map": "tm", "descriptor": {...}, "fitting_net": {...},
                "long_range": {"charge_net": {...},
                               "reciprocal": {"method": "pme", "alpha": 0.4}} },
      "slab": { "same_as": "bulk",
                "long_range": {"reciprocal": {"method": "pme", "alpha": 0.4,
                                              "slab_correction": true,
                                              "pbc": [true, true, false]}} }
    }

The invariant
-------------
    A ``same_as`` branch is the base branch's config **verbatim** except
    ``long_range.reciprocal``, and at runtime it *is* the base model with a
    different ``long_range_energy`` bolted on. There is exactly **one**
    ``atomic_model`` and **one** ``charge_fitting`` per group.

Everything here follows from that. Note in particular what it makes free:
because there is literally one ``atomic_model`` object, the per-type energy
offsets (``bias_atom_e``, ``out_bias``, ``out_std``) are one tensor, so
"identical offsets across the group" is not something to arrange. What *is*
left to arrange is that the single fit of those offsets sees the **union** of
the group's frames, and that the post-training delta is applied once.

Why not per-frame dispatch
--------------------------
The physics would allow choosing the boundary treatment per frame: ``pbc``
reaches only ``compute_slab_correction``, which already takes per-system
``pbc``. The data layer does not: DeePMD-kit has no per-frame boundary
channel (``nopbc`` is a per-*system* file read once at load), and adding one
means patching the batch splitter. Branches keep the plugin out of the data
path.

One caveat: the Yeh-Berkowitz correction is ``2 pi M_z^2 / V``,
which is identically zero for a **non-polar** slab. If your slabs carry no
dipole across the vacuum, one ordinary 3D branch already serves both and none
of this is needed. It earns its keep for polar slabs -- a dipolar interface,
an electrode under electrolyte -- where the correction is large and decays
only as ``1/L_z``.

How it is installed
-------------------
Four upstream symbols are wrapped, none edited, all from :mod:`deepmd_lr.register`
which already runs on every ``deepmd.pt_expt`` import. Each wrapper is a
straight pass-through when no ``same_as`` branch is present, so an ordinary
run is byte-unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "CHARGE_SAME_AS_KEY",
    "SAME_AS_KEY",
    "charge_groups",
    "expand_same_as",
    "groups",
    "install",
    "pool",
]

#: The branch key that names the branch this one copies.
SAME_AS_KEY = "same_as"
#: ``charge_same_as``: a *base* branch (its own energy head and energy zero) that
#: uses another base branch's charge head -- one shared charge network under two
#: levels of theory. The long-range term holds no weights, so sharing the charge
#: head is all it takes for both theories to see one electrostatics.
CHARGE_SAME_AS_KEY = "charge_same_as"

#: What a ``same_as`` branch may write for itself. Everything else is
#: inherited, and writing it would be discarded by the aliasing below -- so it
#: is refused rather than silently ignored.
_BRANCH_ALLOWED = frozenset({SAME_AS_KEY, "long_range", "model_branch_alias", "info"})

#: What such a branch may write inside ``long_range``. `charge_net` is
#: excluded deliberately: it is a parameter-bearing module and must be the
#: same object across the group.
_LONG_RANGE_ALLOWED = frozenset({"reciprocal", "pme"})


def _real_keys(block: dict[str, Any]) -> set[str]:
    """Keys that configure something, ignoring ``_comment`` annotations."""
    return {key for key in block if not key.startswith("_")}


def charge_groups(model_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Map each charge-owning base branch to every branch that must use its charge head.

    The list holds the owner first, then each ``charge_same_as`` branch, then
    the ``same_as`` links of all of those (a link is its base, so it follows its
    base's charge head). Empty for an ordinary config.

    Every ``charge_same_as`` is validated on the way, which is what lets
    :func:`expand_same_as` refuse a bad one as the config is read.
    """
    out: dict[str, list[str]] = {}
    links = groups(model_dict)
    for key, branch in model_dict.items():
        if not isinstance(branch, dict) or branch.get(SAME_AS_KEY):
            # A `same_as` link is its base (after expansion it even carries the
            # base's `charge_same_as`); it is listed through its base below.
            continue
        owner = branch.get(CHARGE_SAME_AS_KEY)
        if not owner:
            continue
        _validated_charge_owner(model_dict, key, owner, branch)
        members = out.setdefault(owner, [owner] + links.get(owner, [owner])[1:])
        members.append(key)
        members.extend(links.get(key, [key])[1:])
    return out


def _normalized_charge_net(block: dict[str, Any]) -> dict[str, Any]:
    """``block`` as the model is built from it: defaults in, comments out.

    Compared raw, a default written out on one branch and left out on the
    other would count as a difference, though the two build the same head.
    """
    from dargs import Argument

    from deepmd_lr.argcheck import charge_fitting_args

    schema = Argument("charge_net", dict, charge_fitting_args())
    return schema.normalize_value(copy.deepcopy(block), trim_pattern="_*")


def _validated_charge_owner(
    model_dict: dict[str, Any], key: str, owner: str, branch: dict[str, Any]
) -> None:
    """Refuse every way a shared charge head could be configured wrong.

    From the config alone; that the two branches share a descriptor can only
    be read off the built modules (:func:`_require_shared_descriptor`).
    """
    if owner == key:
        raise ValueError(
            f"branch {key!r}: `{CHARGE_SAME_AS_KEY}` cannot point at itself"
        )
    if owner not in model_dict or not isinstance(model_dict[owner], dict):
        raise ValueError(
            f"branch {key!r}: `{CHARGE_SAME_AS_KEY}` names {owner!r}, which is not "
            f"a branch. Available: {sorted(model_dict)}"
        )
    base = model_dict[owner]
    if base.get(SAME_AS_KEY) or base.get(CHARGE_SAME_AS_KEY):
        raise ValueError(
            f"branch {key!r}: {owner!r} must be the branch that OWNS the charge "
            f"head -- not a `{SAME_AS_KEY}` link and not itself a "
            f"`{CHARGE_SAME_AS_KEY}` branch (no chains)."
        )
    for name, block in ((owner, base), (key, branch)):
        has_head = (block.get("long_range") or {}).get("charge_net")
        if block.get("type") != "dp_lr" or not has_head:
            raise ValueError(
                f"branch {name!r}: `{CHARGE_SAME_AS_KEY}` needs a `dp_lr` branch "
                "with a `long_range.charge_net` block on both sides."
            )
    own = _normalized_charge_net(branch["long_range"]["charge_net"])
    theirs = _normalized_charge_net(base["long_range"]["charge_net"])
    if own != theirs:
        differing = sorted(
            k for k in set(own) | set(theirs) if own.get(k) != theirs.get(k)
        )
        raise ValueError(
            f"branch {key!r}: its `charge_net` block differs from {owner!r}'s "
            f"in {differing}. One charge head has one configuration; write it "
            "identically on both branches."
        )
    if _alpha(base["long_range"]) != _alpha(branch["long_range"]):
        raise ValueError(
            f"branch {key!r}: alpha differs from {owner!r}. One charge head "
            "serves one Coulomb split."
        )


def groups(model_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Map each base branch to its pseudo-head group, base first.

    Branches with no ``same_as`` pointing at them do not appear at all, which
    is what makes every hook below a no-op for an ordinary config.
    """
    found: dict[str, list[str]] = {}
    for key, branch in model_dict.items():
        if not isinstance(branch, dict):
            continue
        base = branch.get(SAME_AS_KEY)
        if base:
            found.setdefault(base, [base]).append(key)
    return found


def _alpha(block: dict[str, Any]) -> float:
    """The Ewald splitting parameter a ``long_range`` block asks for."""
    from deepmd_lr.lr.term import DEFAULT_ALPHA

    settings = block.get("reciprocal", block.get("pme", {})) or {}
    return float(settings.get("alpha", DEFAULT_ALPHA))


def expand_same_as(model_config: dict[str, Any]) -> dict[str, Any]:
    """Expand every ``same_as`` branch into a full config, in place.

    Runs *after* DeePMD-kit's own ``preprocess_shared_params``, so the base it
    copies has already had its ``shared_dict`` string references resolved. The
    pseudo branch therefore never appears in ``shared_links`` -- which keeps
    ``merge_env_stat`` from folding one branch's descriptor statistics into
    the base twice, and keeps the ``shared_level > 0`` compile scan seeing only
    genuine links. Nothing is lost: the aliasing in :func:`_alias_groups`
    subsumes whatever sharing would have been recorded.
    """
    model_dict = model_config.get("model_dict")
    if not isinstance(model_dict, dict):
        return model_config

    pending = {
        key: branch
        for key, branch in model_dict.items()
        if isinstance(branch, dict) and branch.get(SAME_AS_KEY)
    }
    for key, branch in pending.items():
        base_key = branch[SAME_AS_KEY]
        base = _validated_base(model_dict, key, base_key, branch)

        merged = copy.deepcopy(base)
        # Branch-local metadata is never inherited. `model_branch_alias` must
        # not be, because DeePMD-kit asserts aliases are unique across
        # branches and `ase.load_model` resolves a head through them; `info`
        # must not be, because an inherited "theory: RPBE" on the slab head
        # would simply be a lie.
        merged.pop("model_branch_alias", None)
        merged.pop("info", None)
        merged[SAME_AS_KEY] = base_key

        long_range = copy.deepcopy(base.get("long_range") or {})
        # Drop BOTH spellings before applying the branch's own, or an
        # inherited `reciprocal` would survive alongside a branch-written
        # legacy `pme` and the alias resolution would see two.
        long_range.pop("reciprocal", None)
        long_range.pop("pme", None)
        long_range.update(branch.get("long_range") or {})
        merged["long_range"] = long_range

        for carried in ("model_branch_alias", "info"):
            if carried in branch:
                merged[carried] = copy.deepcopy(branch[carried])

        model_dict[key] = merged
    # Refuse a bad `charge_same_as` now, from the config alone. The aliasing
    # would find it too, but only in `share_params`: after every branch is
    # built and the training data has been sampled for statistics.
    charge_groups(model_dict)
    return model_config


def _validated_base(
    model_dict: dict[str, Any],
    key: str,
    base_key: str,
    branch: dict[str, Any],
) -> dict[str, Any]:
    """Resolve ``base_key`` and refuse every way the group could diverge.

    These checks are the whole safety story. A key that slipped through would
    build a *different* model whose weights the aliasing then throws away --
    training would run, converge, and be quietly meaningless.
    """
    base = _resolved_base(model_dict, key, base_key)
    _check_link_keys(key, base_key, branch)
    own_long_range = branch.get("long_range") or {}
    base_alpha, branch_alpha = _alpha(base["long_range"]), _alpha(own_long_range)
    if own_long_range and base_alpha != branch_alpha:
        raise ValueError(
            f"branch {key!r}: alpha is {branch_alpha} but {base_key!r} uses "
            f"{base_alpha}. alpha is where Coulomb is split, and the one "
            "shared descriptor can only learn the `erfc(alpha r)/r` remainder "
            "of a single split."
        )
    if (base.get("fitting_net") or {}).get("dim_case_embd", 0):
        raise ValueError(
            f"branch {key!r}: `{SAME_AS_KEY}` is incompatible with "
            "`fitting_net.dim_case_embd > 0`. A case embedding gives each "
            "branch a different index in one `case_embd` buffer, which the "
            "shared fitting net collapses to a single value -- the branches "
            "would silently share an index. Drop the case embedding: these "
            "branches are one model, which is what a case embedding exists to "
            "deny."
        )
    return base


def _resolved_base(
    model_dict: dict[str, Any], key: str, base_key: str
) -> dict[str, Any]:
    """The base config ``key`` points at, refused unless it can be one."""
    if base_key == key:
        raise ValueError(f"branch {key!r}: `{SAME_AS_KEY}` cannot point at itself")
    if base_key not in model_dict:
        raise ValueError(
            f"branch {key!r}: `{SAME_AS_KEY}` names {base_key!r}, which is not "
            f"a branch. Available: {sorted(model_dict)}"
        )
    base = model_dict[base_key]
    if not isinstance(base, dict):
        raise ValueError(f"branch {key!r}: {base_key!r} is not a model config")
    if base.get(SAME_AS_KEY):
        raise ValueError(
            f"branch {key!r}: {base_key!r} is itself a `{SAME_AS_KEY}` branch. "
            "Point every member of a group at the one branch that defines it "
            "-- a chain would make it ambiguous which model is the original."
        )
    if base.get("type") != "dp_lr":
        raise ValueError(
            f"branch {key!r}: {base_key!r} has type {base.get('type')!r}, but "
            f"`{SAME_AS_KEY}` exists to vary the long-range term and only "
            "`dp_lr` has one."
        )
    if not base.get("long_range"):
        raise ValueError(
            f"branch {key!r}: {base_key!r} has no `long_range` block, so there "
            "is nothing for this branch to differ in."
        )
    return base


def _check_link_keys(key: str, base_key: str, branch: dict[str, Any]) -> None:
    """Refuse every key a link may not write, at the top and in ``long_range``."""
    extra = _real_keys(branch) - _BRANCH_ALLOWED
    if CHARGE_SAME_AS_KEY in extra:
        raise ValueError(
            f"branch {key!r}: a `{SAME_AS_KEY}` link uses its base's charge "
            f"head, so `{CHARGE_SAME_AS_KEY}` belongs on {base_key!r}; the "
            "link follows it there."
        )
    if extra:
        raise ValueError(
            f"branch {key!r} declares `{SAME_AS_KEY}: {base_key}` but also sets "
            f"{sorted(extra)}. A pseudo-head IS the base model -- those keys "
            "would be built and then discarded when the modules are aliased. "
            f"Set them on {base_key!r}, or drop `{SAME_AS_KEY}` and write the "
            "branch out in full."
        )
    extra = _real_keys(branch.get("long_range") or {}) - _LONG_RANGE_ALLOWED
    if extra:
        detail = (
            " `charge_net` holds trainable weights and must be the same module "
            "object across the group, so it is inherited, never overridden."
            if "charge_net" in extra
            else ""
        )
        raise ValueError(
            f"branch {key!r}: only {sorted(_LONG_RANGE_ALLOWED)} may be set "
            f"under `long_range` in a `{SAME_AS_KEY}` branch, got "
            f"{sorted(extra)}.{detail}"
        )


# --------------------------------------------------------------------------
# The four hooks.
#
# Order matters and is not free to change:
#
#   preprocess_config        upstream preprocess_shared_params, then expansion
#   normalize_config         the `dp_lr` schema accepts `same_as`
#   Trainer.__init__
#     _POOLED filled from training_data
#     get_model per branch                two independent models
#     compute_or_load_stat                pooled samples  <- hook 3
#     share_params x3                     upstream merge, then OUR ALIASING
#     DDP wrap, optimizer                 see one deduped parameter set
#   train loop                            one gradient, one update per step
#   change_model_out_bias_by_task         base applies the delta once
# --------------------------------------------------------------------------

#: ``id(DeepmdDataSystem)`` -> the data systems of its group, base first. The
#: values hold strong references, so an id here cannot be recycled onto a
#: different object.
_POOLED: dict[int, list[Any]] = {}


def pool(data: Any, nbatches: int, sample: Any, **kwargs: Any) -> list:
    """Sample a data system, or its whole group when it belongs to one.

    Concatenating the per-system sample lists is indistinguishable from having
    listed every system under one branch: ``compute_output_stats`` indexes that
    list per system and feeds all of them into one global regression. So this
    is the exact union least-squares, with per-frame weighting for free --
    unlike averaging the per-branch biases, which is a different estimator and
    plainly wrong for the delta fit, a regression on residuals.

    A system that is in no group falls straight through, which is what makes
    an ordinary run byte-unchanged.
    """
    group = _POOLED.get(id(data))
    if group is None:
        return sample(data, nbatches, **kwargs)
    return [s for system in group for s in sample(system, nbatches, **kwargs)]


def _require_shared_descriptor(wrapper: Any, owner: str, key: str) -> None:
    """Refuse a ``charge_same_as`` pair whose descriptors are two models.

    The charge head reads the descriptor's features, so one head is one
    electrostatics only if both branches feed it the same descriptor. Nothing
    in the branch config says whether they do -- an inline descriptor with the
    same settings builds a second, independently trained copy -- so this asks
    the built modules, after DeePMD-kit's own sharing has run. Two copies agree
    at initialisation only because their seeds match, then drift apart as each
    theory trains its own; with the type maps in different orders, one head
    reads each branch's type indices as the owner's, and the same oxygen got
    opposite-sign charges under the two theories.
    """
    ours = list(wrapper.model[key].atomic_model.descriptor.parameters())
    theirs = list(wrapper.model[owner].atomic_model.descriptor.parameters())
    if len(ours) != len(theirs) or any(
        p.data_ptr() != q.data_ptr() for p, q in zip(ours, theirs, strict=True)
    ):
        raise ValueError(
            f"branch {key!r}: `{CHARGE_SAME_AS_KEY}: {owner}` needs the two "
            "branches to share their descriptor, since the charge head reads "
            "its features. Point both `descriptor` entries at one "
            "`shared_dict` key."
        )


def _alias_groups(wrapper: Any) -> None:
    """Make every member of a group the same model but for its long-range term.

    Two rebindings, not a per-submodule loop, and both choices are forced:

    * ``BaseAtomicModel._store_out_stat`` ends by *binding a fresh array over*
      ``out_bias`` rather than copying into it -- DeePMD-kit says as much where
      it lists them as per-task buffers. So aliasing the buffers individually
      is undone by the first ``change_out_bias``. Aliasing the object that
      owns them is the only durable form.
    * ``InvarFitting.share_params`` deliberately excludes ``bias_atom_e``, so
      reusing it would leave each branch its own energy zero -- the opposite
      of one model.
    """
    model_dict = (getattr(wrapper, "model_params", None) or {}).get("model_dict")
    if not isinstance(model_dict, dict):
        return
    for base_key, members in groups(model_dict).items():
        base = wrapper.model[base_key]
        for key in members[1:]:
            link = wrapper.model[key]
            link.atomic_model = base.atomic_model
            link.charge_fitting = base.charge_fitting
            # A plain bool: assigning a Module here would register a submodule
            # and recurse into the state dict.
            link._same_as_link = True
    # `charge_same_as`: one charge head under several theories. Done AFTER the
    # `same_as` aliasing so that a link that was just pointed at its base's
    # charge head is pointed again at the owner's.
    for owner, members in charge_groups(model_dict).items():
        head = wrapper.model[owner].charge_fitting
        for key in members[1:]:
            _require_shared_descriptor(wrapper, owner, key)
            wrapper.model[key].charge_fitting = head


def install() -> None:
    """Wrap the four upstream symbols. Idempotent."""
    from deepmd.pt_expt.train import training as _training
    from deepmd.pt_expt.train import wrapper as _wrapper
    from deepmd.pt_expt.utils import multi_task as _multi_task

    if getattr(_multi_task.preprocess_shared_params, "_deepmd_lr_patched", False):
        return
    _wrap_preprocess(_multi_task)
    _wrap_share_params(_wrapper.ModelWrapper)
    _wrap_stat_input(_training)
    _wrap_trainer_init(_training.Trainer)


def _wrap_preprocess(module: Any) -> None:
    """Expand ``same_as`` right after upstream resolves ``shared_dict``."""
    original = module.preprocess_shared_params

    def preprocess_shared_params(model_config: dict[str, Any]) -> tuple:
        model_config, shared_links = original(model_config)
        expand_same_as(model_config)
        return model_config, shared_links

    preprocess_shared_params._deepmd_lr_patched = True
    preprocess_shared_params.__doc__ = original.__doc__
    module.preprocess_shared_params = preprocess_shared_params


def _wrap_share_params(wrapper_cls: Any) -> None:
    """Alias the groups after upstream's own parameter sharing."""
    original = wrapper_cls.share_params

    def share_params(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _alias_groups(self)

    share_params.__doc__ = original.__doc__
    wrapper_cls.share_params = share_params


def _wrap_stat_input(training: Any) -> None:
    """Pool a group's samples for the statistics.

    ``make_stat_input`` is imported into training.py's namespace and called
    through the module global, so this is the name that takes effect. The
    sampler closures are built and consumed inside ``Trainer.__init__`` with
    no seam between them, which is why the pooling has to happen here rather
    than at a call site.
    """
    original = training.make_stat_input

    def make_stat_input(data: Any, nbatches: int, **kwargs: Any) -> list:
        return pool(data, nbatches, original, **kwargs)

    make_stat_input.__doc__ = original.__doc__
    training.make_stat_input = make_stat_input


def _wrap_trainer_init(trainer_cls: Any) -> None:
    """Fill :data:`_POOLED` from the training data before the trainer runs."""
    original = trainer_cls.__init__

    def __init__(
        self: Any, config: Any, training_data: Any, *args: Any, **kwargs: Any
    ) -> None:
        _POOLED.clear()
        model_dict = (config.get("model") or {}).get("model_dict")
        if isinstance(model_dict, dict) and isinstance(training_data, dict):
            for members in groups(model_dict).values():
                # Base first: `compute_output_stats` filters the keys it was
                # asked for against `sampled[0].keys()`.
                systems = [
                    training_data[key] for key in members if key in training_data
                ]
                for system in systems:
                    _POOLED[id(system)] = systems
        original(self, config, training_data, *args, **kwargs)

    __init__.__doc__ = original.__doc__
    trainer_cls.__init__ = __init__
