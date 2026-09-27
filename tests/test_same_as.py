"""``same_as``: pseudo-heads that are one model with different boundaries.

A dataset mixing bulk and slab systems needs one set of weights evaluated
under two boundary treatments. Because the long-range term holds no trainable
parameters, that is possible -- and ``same_as`` is how it is declared.

What could go wrong here is silent, which is why most of these tests are about
identity rather than numbers. If a branch quietly built its own descriptor, or
kept its own energy zero, or applied the post-training bias delta twice,
training would still run and converge; the model would just not be the model
anyone asked for.
"""

import copy
import json

import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.train.wrapper import ModelWrapper
from deepmd.pt_expt.utils.multi_task import preprocess_shared_params
from deepmd.utils.argcheck import normalize

from deepmd_lr.same_as import expand_same_as, groups

from .test_joint_model import DESCRIPTOR, FITTING, LONG_RANGE, build, frame
from .test_multi_task import lr_branch

SLAB_RECIPROCAL = {
    "alpha": 0.4,
    "mesh_dimensions": [15, 15, 15],
    "spline_order": 6,
    "slab_correction": True,
    "pbc": [True, True, False],
}


def pseudo_head_config(slab: dict | None = None, extra: dict | None = None) -> dict:
    """A bulk branch and a `same_as` slab branch, before normalization.

    Note the `shared_dict` type_map: DeePMD-kit asserts a multi-task model has
    exactly one *shared* type map reference, so even a config whose branches
    are copies of one another has to name it there.
    """
    branch = {
        "same_as": "bulk",
        "long_range": {"reciprocal": dict(slab or SLAB_RECIPROCAL)},
    }
    branch.update(extra or {})
    return {
        "shared_dict": {"tm": ["O", "H"], "des": copy.deepcopy(DESCRIPTOR)},
        "model_dict": {"bulk": lr_branch(), "slab": branch},
    }


def normalized(model: dict):
    """Run the real pipeline order: preprocess (which expands), then normalize."""
    cfg = {
        "model": model,
        "learning_rate": {"start_lr": 1e-3},
        "loss_dict": {k: {"type": "ener"} for k in model["model_dict"]},
        "training": {
            "data_dict": {
                k: {"training_data": {"systems": ["./d"]}} for k in model["model_dict"]
            },
            "numb_steps": 1,
        },
    }
    cfg["model"], links = preprocess_shared_params(cfg["model"])
    return normalize(cfg, multi_task=True), links


def build_group(model: dict | None = None):
    """Build both branches and alias them, exactly as the trainer does."""
    cfg, links = normalized(model or pseudo_head_config())
    models = {
        key: get_model(copy.deepcopy(params))
        for key, params in cfg["model"]["model_dict"].items()
    }
    prob = {key: 1.0 / len(models) for key in models}
    # `model_params` is not optional here: it is where the wrapper reads the
    # groups from. The trainer always passes it; a test that forgets would get
    # unaliased branches and no error at all.
    ModelWrapper(models, model_params=cfg["model"]).share_params(
        links, prob, resume=True
    )
    return models


class TestExpansion:
    def test_the_link_is_the_base_but_for_its_reciprocal(self):
        cfg, _ = normalized(pseudo_head_config())
        bulk = copy.deepcopy(cfg["model"]["model_dict"]["bulk"])
        slab = copy.deepcopy(cfg["model"]["model_dict"]["slab"])
        assert slab["same_as"] == "bulk"
        assert slab["long_range"]["reciprocal"]["slab_correction"] is True
        assert bulk["long_range"]["reciprocal"]["slab_correction"] is False
        # Everything else -- descriptor, fitting net, charge head, type map --
        # is the base's, verbatim.
        for block in (bulk, slab):
            block.pop("same_as")
            block["long_range"].pop("reciprocal")
        assert bulk == slab

    def test_groups_reports_the_base_first(self):
        assert groups(pseudo_head_config()["model_dict"]) == {"bulk": ["bulk", "slab"]}

    def test_an_ordinary_config_is_untouched(self):
        """Every hook must be a pass-through when nobody asked for one."""
        plain = {"model_dict": {"a": lr_branch(), "b": lr_branch()}}
        before = copy.deepcopy(plain)
        assert expand_same_as(plain) == before
        assert groups(plain["model_dict"]) == {}

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (
                {"fitting_net": copy.deepcopy(FITTING)},
                "would be built and then discarded",
            ),
            ({"descriptor": "des"}, "would be built and then discarded"),
            ({"same_as": "nope"}, "not a branch"),
            ({"same_as": "slab"}, "cannot point at itself"),
        ],
    )
    def test_a_divergent_branch_is_refused(self, extra, message):
        with pytest.raises(ValueError, match=message):
            normalized(pseudo_head_config(extra=extra))

    def test_overriding_the_charge_head_is_refused(self):
        """It holds weights, so it must be the same object -- not a copy."""
        model = pseudo_head_config()
        model["model_dict"]["slab"]["long_range"]["charge_net"] = copy.deepcopy(
            LONG_RANGE["charge_net"]
        )
        with pytest.raises(ValueError, match="charge_net"):
            normalized(model)

    def test_a_different_alpha_is_refused(self):
        with pytest.raises(ValueError, match="alpha is"):
            normalized(pseudo_head_config(slab={**SLAB_RECIPROCAL, "alpha": 0.5}))

    def test_a_chain_is_refused(self):
        model = pseudo_head_config()
        model["model_dict"]["third"] = {"same_as": "slab", "long_range": {}}
        with pytest.raises(ValueError, match="is itself a `same_as` branch"):
            normalized(model)

    def test_a_case_embedding_is_refused(self):
        """Two branches, two case indices, one shared buffer."""
        model = pseudo_head_config()
        model["model_dict"]["bulk"]["fitting_net"] = {
            **copy.deepcopy(FITTING),
            "dim_case_embd": 2,
        }
        with pytest.raises(ValueError, match="dim_case_embd"):
            normalized(model)


class TestAliasing:
    def test_the_branches_are_one_model(self):
        models = build_group()
        assert models["slab"].atomic_model is models["bulk"].atomic_model
        assert models["slab"].charge_fitting is models["bulk"].charge_fitting

    def test_only_the_long_range_term_differs(self):
        models = build_group()
        assert models["slab"].long_range_energy is not models["bulk"].long_range_energy
        assert models["slab"].long_range_energy.slab_correction is True
        assert models["bulk"].long_range_energy.slab_correction is False
        assert models["slab"].long_range_energy.alpha == (
            models["bulk"].long_range_energy.alpha
        )

    def test_every_weight_is_shared_including_the_charge_head(self):
        """The inverse of `test_multi_task.TestSharing`, which pins the
        opposite for a genuine multi-head. Both must hold at once."""
        models = build_group()
        sa = models["bulk"].state_dict()
        sb = models["slab"].state_dict()
        # `_min_nbor_dist` is the one deliberate exception: it is a statistic of
        # each branch's own DATA, not a weight, and bulk and slab frames may
        # genuinely have different closest pairs.
        common = (set(sa) & set(sb)) - {"_min_nbor_dist"}
        assert common
        untied = {k for k in common if sa[k].data_ptr() != sb[k].data_ptr()}
        assert not untied, f"not shared: {sorted(untied)[:5]}"
        assert any(k.startswith("charge_fitting.") for k in common)
        assert any("out_bias" in k for k in common)

    def test_the_optimizer_sees_one_parameter_set(self):
        """Aliased modules appear once in `parameters()`, so shared weights get
        one gradient and one update per step however the branch was drawn."""
        models = build_group()
        wrapper = ModelWrapper(models, model_params={})
        one = len(list(models["bulk"].parameters()))
        assert len(list(wrapper.parameters())) == one

    def test_without_model_params_nothing_is_aliased(self):
        """The failure mode worth naming: the wrapper reads the groups from
        `model_params`, so omitting it silently yields two separate models."""
        cfg, links = normalized(pseudo_head_config())
        models = {
            key: get_model(copy.deepcopy(params))
            for key, params in cfg["model"]["model_dict"].items()
        }
        ModelWrapper(models).share_params(links, {k: 0.5 for k in models}, resume=True)
        assert models["slab"].atomic_model is not models["bulk"].atomic_model


class TestOffsets:
    def test_a_shared_bias_survives_a_refit(self):
        """The test a per-buffer alias fails. `_store_out_stat` BINDS a fresh
        array over `out_bias` rather than copying into it, so aliasing the
        tensors instead of the object that owns them would come undone the
        first time the bias is refitted."""
        models = build_group()
        bulk = models["bulk"].atomic_model
        bulk.out_bias = bulk.out_bias + 1.25
        assert torch.equal(models["slab"].atomic_model.out_bias, bulk.out_bias)

    def test_the_delta_is_applied_once_per_group(self):
        """`change-by-statistic` ADDS. DeePMD-kit loops over branch keys, and
        with one shared `atomic_model` that would double the correction."""
        models = build_group()
        assert models["bulk"]._same_as_link is False
        assert models["slab"]._same_as_link is True

        calls = []
        models["bulk"].atomic_model.change_out_bias = lambda *a, **k: calls.append(1)
        for model in models.values():
            model.change_out_bias({}, bias_adjust_mode="change-by-statistic")
        assert len(calls) == 1


class TestPhysics:
    def test_the_branches_differ_by_exactly_the_slab_correction(self):
        """One weight set, two boundary treatments: whatever the charges are,
        the two energies must differ by the Yeh-Berkowitz term alone."""
        from deepmd_lr.lr.term import (
            ELECTROSTATIC_CONVERSION,
            compute_slab_correction,
        )

        models = build_group()
        coord, atype, cell = frame()
        bulk = models["bulk"](coord, atype, box=cell)
        slab = models["slab"](coord, atype, box=cell)
        # Same weights, so the short-range half is identical.
        assert torch.allclose(
            bulk["energy_short_range"], slab["energy_short_range"], atol=1e-10
        )
        difference = float(slab["energy"] - bulk["energy"])
        charges = bulk["charge"]
        pbc = torch.tensor(
            [[True, True, False]], dtype=torch.bool, device=coord.device
        )
        batch_idx = torch.zeros(
            coord.shape[1], dtype=torch.int32, device=coord.device
        )
        expected = ELECTROSTATIC_CONVERSION * float(
            compute_slab_correction(
                coord.reshape(-1, 3),
                charges.reshape(-1),
                cell,
                pbc,
                batch_idx=batch_idx,
                energy_reduction="atom",
            ).sum()
        )
        assert difference == pytest.approx(expected, abs=1e-8)


class TestPooledStatistics:
    """The offsets must be fit over the group's frames pooled, not per branch.

    With one shared ``atomic_model`` the offsets are already one tensor, so
    nothing has to be copied between branches -- but the single fit has to see
    every frame. Concatenating the per-system sample lists is exactly
    equivalent to having listed all the systems under one branch, because
    ``compute_output_stats`` indexes that list per system and feeds all of them
    into one regression.
    """

    def test_a_group_samples_the_union(self):
        from deepmd_lr.same_as import _POOLED, pool

        systems = ["bulk-data", "slab-data"]

        def sample(data, nbatches, **kwargs):
            return [{"system": data, "nbatches": nbatches}]

        _POOLED.clear()
        _POOLED.update({id(s): systems for s in systems})
        try:
            for system in systems:
                drawn = pool(system, 10, sample)
                # Whichever branch asks, it gets the union -- so the one fit of
                # the shared offsets sees every frame in the group.
                assert [d["system"] for d in drawn] == systems
        finally:
            _POOLED.clear()

    def test_an_ungrouped_system_passes_straight_through(self):
        """An ordinary run must be byte-unchanged, not merely equivalent."""
        from deepmd_lr.same_as import pool

        sentinel = object()
        assert pool("lonely", 10, lambda *a, **k: sentinel) is sentinel


class TestSetLongRange:
    """The inference half: the boundary treatment is not a property of the
    checkpoint, because the long-range term holds no weights."""

    def test_swapping_changes_the_term_and_no_weight(self):
        model = build()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.set_long_range({**SLAB_RECIPROCAL})
        assert model.long_range_energy.slab_correction is True
        after = model.state_dict()
        assert set(before) == set(after)
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_swapping_to_isolated_drops_the_cell_requirement(self):
        model = build()
        assert model.long_range_energy.needs_cell is True
        model.set_long_range({"method": "isolated"})
        assert model.long_range_energy.needs_cell is False
        # Replacement, not a merge: the periodic knobs of the old block would
        # have been refused by `IsolatedLongRangeEnergy`.
        assert model.long_range_energy.alpha == LONG_RANGE["reciprocal"]["alpha"]

    def test_the_recorded_config_keeps_up(self):
        import json

        model = build()
        model.set_long_range({"method": "isolated"})
        recorded = json.loads(model.model_def_script)
        assert recorded["long_range"]["reciprocal"]["method"] == "isolated"

    def test_a_different_alpha_is_refused(self):
        model = build()
        with pytest.raises(ValueError, match="alpha is fixed"):
            model.set_long_range({"method": "isolated", "alpha": 0.5})

    def test_a_short_range_model_says_so(self):
        with pytest.raises(ValueError, match="no `long_range` block"):
            build(long_range=False).set_long_range({"method": "isolated"})

    def test_load_model_applies_it(self, tmp_path):
        """The whole load path, through a checkpoint on disk: the weights must
        still match strictly even though the term was replaced."""
        import torch as _torch

        from deepmd_lr.ase import load_model

        model = build()
        cfg = json.loads(model.model_def_script)
        state = {f"model.Default.{k}": v for k, v in model.state_dict().items()}
        state["_extra_state"] = {"model_params": cfg}
        path = tmp_path / "model.ckpt.pt"
        _torch.save({"model": state}, path)

        plain = load_model(str(path), device="cpu")
        assert plain.long_range_energy.slab_correction is False
        # `load_model` re-checks strictly, so reaching here at all proves the
        # swapped term costs no state_dict key.
        swapped = load_model(str(path), device="cpu", long_range=SLAB_RECIPROCAL)
        assert swapped.long_range_energy.slab_correction is True
        assert swapped.long_range_energy.pbc == (True, True, False)
