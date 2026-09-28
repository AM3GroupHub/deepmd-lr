"""``charge_same_as``: one charge network under two levels of theory.

Two base branches (say r2SCAN and a MACE teacher) each keep their own energy
fitting net and energy zero, but predict charges -- and so the long-range
term, which holds no weights -- through ONE shared charge head. ``same_as``
links of either branch follow it: the four heads ttt_dft / ttf_dft /
ttt_mace / ttf_mace are one backbone, one charge head, two energy heads,
two boundary treatments.

As with ``same_as``, what could go wrong is silent: a branch that quietly
built its own charge head would train and converge on 117 slab frames and
drift everywhere else. So these tests are about object identity, gradient
routing and round-trips, not numbers.
"""

import copy

import pytest
import torch
from dargs import Argument
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.train.wrapper import ModelWrapper
from deepmd.pt_expt.utils.multi_task import preprocess_shared_params
from deepmd.utils.argcheck import normalize

from deepmd_lr.argcheck import charge_fitting_args
from deepmd_lr.same_as import charge_groups

from .test_joint_model import DESCRIPTOR, frame
from .test_multi_task import lr_branch
from .test_same_as import SLAB_RECIPROCAL

HEADS = ("ttt_dft", "ttf_dft", "ttt_mace", "ttf_mace")


def two_theory_config(extra_mace: dict | None = None) -> dict:
    """Two theories x two boundaries: dft owns the charge head, mace borrows it."""
    mace = lr_branch()
    mace["charge_same_as"] = "ttt_dft"
    mace.update(extra_mace or {})
    return {
        "shared_dict": {"tm": ["O", "H"], "des": copy.deepcopy(DESCRIPTOR)},
        "model_dict": {
            "ttt_dft": lr_branch(),
            "ttf_dft": {
                "same_as": "ttt_dft",
                "long_range": {"reciprocal": dict(SLAB_RECIPROCAL)},
            },
            "ttt_mace": mace,
            "ttf_mace": {
                "same_as": "ttt_mace",
                "long_range": {"reciprocal": dict(SLAB_RECIPROCAL)},
            },
        },
    }


def normalized(model: dict):
    """Real pipeline order: preprocess (expands same_as), then normalize."""
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


def build(model: dict | None = None):
    """Build the four branches and alias them exactly as the trainer does."""
    cfg, links = normalized(model or two_theory_config())
    models = {
        key: get_model(copy.deepcopy(params))
        for key, params in cfg["model"]["model_dict"].items()
    }
    prob = {key: 1.0 / len(models) for key in models}
    wrapper = ModelWrapper(models, model_params=cfg["model"])
    wrapper.share_params(links, prob, resume=True)
    return models, wrapper, cfg


class TestGrouping:
    def test_the_owner_comes_first_and_links_follow(self):
        cfg, _ = normalized(two_theory_config())
        assert charge_groups(cfg["model"]["model_dict"]) == {
            "ttt_dft": ["ttt_dft", "ttf_dft", "ttt_mace", "ttf_mace"]
        }

    def test_an_ordinary_config_has_no_groups(self):
        assert charge_groups({"a": lr_branch(), "b": lr_branch()}) == {}

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            ({"charge_same_as": "ttt_mace"}, "cannot point at itself"),
            ({"charge_same_as": "nope"}, "not a branch"),
            ({"charge_same_as": "ttf_dft"}, "OWNS the charge head"),
        ],
    )
    def test_a_bad_owner_is_refused(self, extra, message):
        with pytest.raises(ValueError, match=message):
            build(two_theory_config(extra_mace=extra))

    def test_a_different_charge_net_is_refused(self):
        mace = {"long_range": copy.deepcopy(lr_branch()["long_range"])}
        mace["long_range"]["charge_net"]["neuron"] = [16, 16]
        with pytest.raises(ValueError, match="differs from"):
            build(two_theory_config(extra_mace=mace))

    @pytest.mark.parametrize(
        "type_map", [None, ["H", "O"]], ids=["same_type_map", "reversed_type_map"]
    )
    def test_an_unshared_descriptor_is_refused(self, type_map):
        """An inline descriptor with the same settings is a second model: the
        charges agreed only through equal seeds, then drifted by 3e-3 e as it
        trained, and with the type map reversed the same oxygen got opposite
        signs under the two theories. Only the built modules can tell."""
        extra = {"descriptor": copy.deepcopy(DESCRIPTOR)}
        if type_map is not None:
            extra["type_map"] = type_map
        with pytest.raises(ValueError, match="share their descriptor"):
            build(two_theory_config(extra_mace=extra))

    def test_a_different_alpha_is_refused(self):
        """The whole borrowing theory on another split; its slab link matches
        it, so `same_as` has nothing to object to and this check is what
        fires."""
        model = two_theory_config()
        model["model_dict"]["ttt_mace"]["long_range"]["reciprocal"]["alpha"] = 0.3
        model["model_dict"]["ttf_mace"]["long_range"]["reciprocal"]["alpha"] = 0.3
        with pytest.raises(ValueError, match="alpha differs"):
            build(model)

    def test_a_default_written_out_matches_one_left_out(self):
        """The blocks are compared as the heads are built from them."""
        schema = Argument("charge_net", dict, charge_fitting_args())
        defaults = schema.normalize_value({}, trim_pattern="_*")
        mace = {"long_range": copy.deepcopy(lr_branch()["long_range"])}
        mace["long_range"]["charge_net"]["resnet_dt"] = defaults["resnet_dt"]
        models, _, _ = build(two_theory_config(extra_mace=mace))
        assert models["ttt_mace"].charge_fitting is models["ttt_dft"].charge_fitting

    def test_a_bad_owner_fails_as_the_config_is_read(self):
        """Before any model is built or any data sampled for statistics."""
        model = two_theory_config(extra_mace={"charge_same_as": "nope"})
        with pytest.raises(ValueError, match="not a branch"):
            preprocess_shared_params(model)

    def test_charge_same_as_on_a_link_points_to_its_base(self):
        model = two_theory_config()
        model["model_dict"]["ttf_mace"]["charge_same_as"] = "ttt_dft"
        with pytest.raises(ValueError, match="belongs on 'ttt_mace'"):
            build(model)

    def test_a_chain_is_refused(self):
        model = two_theory_config()
        model["model_dict"]["third"] = lr_branch()
        model["model_dict"]["third"]["charge_same_as"] = "ttt_mace"
        with pytest.raises(ValueError, match="no chains"):
            build(model)


class TestIdentity:
    def test_one_charge_head_for_four_branches(self):
        models, _, _ = build()
        head = models["ttt_dft"].charge_fitting
        for key in HEADS[1:]:
            assert models[key].charge_fitting is head, key

    def test_the_theories_keep_their_own_energy_heads_and_zeros(self):
        models, _, _ = build()
        dft, mace = models["ttt_dft"].atomic_model, models["ttt_mace"].atomic_model
        assert dft is not mace
        assert dft.fitting_net is not mace.fitting_net
        assert dft.out_bias.data_ptr() != mace.out_bias.data_ptr()
        # the boundary links are their theory's model
        assert models["ttf_dft"].atomic_model is dft
        assert models["ttf_mace"].atomic_model is mace

    def test_the_backbone_is_shared_the_usual_way(self):
        models, _, _ = build()
        da, db = (
            models["ttt_dft"].atomic_model.descriptor,
            models["ttt_mace"].atomic_model.descriptor,
        )
        assert da.type_embedding is db.type_embedding
        assert da.readout is db.readout

    def test_the_optimizer_sees_one_charge_head(self):
        """Unique parameters: all of one theory, plus the other theory's
        energy head."""
        models, wrapper, _ = build()
        n_dft = len(list(models["ttt_dft"].parameters()))
        n_mace_fitting = len(
            list(models["ttt_mace"].atomic_model.fitting_net.parameters())
        )
        assert len(list(wrapper.parameters())) == n_dft + n_mace_fitting


class TestGradients:
    def _grads(self, model):
        coord, atype, cell = frame()
        model.zero_grad()
        out = model(coord, atype, box=cell)
        (out["energy"].sum() + out["force"].square().sum()).backward()

    def _has_grad(self, module):
        return any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )

    def _clear(self, models):
        for m in models.values():
            for p in m.parameters():
                p.grad = None

    def test_a_dft_batch_moves_the_shared_charges_and_the_dft_head_only(self):
        models, _, _ = build()
        self._clear(models)
        self._grads(models["ttt_dft"])
        assert self._has_grad(models["ttt_dft"].charge_fitting)
        assert self._has_grad(models["ttt_dft"].atomic_model.fitting_net)
        assert not self._has_grad(models["ttt_mace"].atomic_model.fitting_net)

    def test_a_mace_batch_moves_the_shared_charges_and_the_mace_head_only(self):
        models, _, _ = build()
        self._clear(models)
        self._grads(models["ttf_mace"])
        assert self._has_grad(models["ttt_dft"].charge_fitting)  # the same object
        assert self._has_grad(models["ttt_mace"].atomic_model.fitting_net)
        assert not self._has_grad(models["ttt_dft"].atomic_model.fitting_net)


class TestPhysics:
    def test_both_theories_predict_the_same_charges(self):
        models, _, _ = build()
        coord, atype, cell = frame()
        q_dft = models["ttt_dft"](coord, atype, box=cell)["charge"]
        q_mace = models["ttt_mace"](coord, atype, box=cell)["charge"]
        assert torch.allclose(q_dft, q_mace, atol=1e-12)

    def test_the_long_range_energy_is_one_and_the_short_range_differs(self):
        models, _, _ = build()
        coord, atype, cell = frame()
        a, b = (
            models["ttt_dft"](coord, atype, box=cell),
            models["ttt_mace"](coord, atype, box=cell),
        )
        assert torch.allclose(
            a["energy_long_range"], b["energy_long_range"], atol=1e-10
        )
        # Two energy heads seeded alike give the same short-range energy;
        # perturb one to show they are independent modules.
        with torch.no_grad():
            for p in models["ttt_mace"].atomic_model.fitting_net.parameters():
                p.add_(0.01)
        b2 = models["ttt_mace"](coord, atype, box=cell)
        a2 = models["ttt_dft"](coord, atype, box=cell)
        assert not torch.allclose(a2["energy_short_range"], b2["energy_short_range"])
        assert torch.allclose(
            a2["energy_short_range"], a["energy_short_range"], atol=1e-10
        )

    def test_boundary_links_differ_by_the_slab_correction_only(self):
        models, _, _ = build()
        coord, atype, cell = frame()
        for theory in ("dft", "mace"):
            bulk = models["ttt_" + theory](coord, atype, box=cell)
            slab = models["ttf_" + theory](coord, atype, box=cell)
            assert torch.allclose(
                bulk["energy_short_range"], slab["energy_short_range"], atol=1e-10
            )
            assert torch.allclose(bulk["charge"], slab["charge"], atol=1e-12)


class TestRoundTrip:
    def test_state_dict_reload_keeps_the_identities_and_values(self):
        models, wrapper, _ = build()
        with torch.no_grad():
            for p in models["ttt_dft"].charge_fitting.parameters():
                p.add_(0.5)
        state = copy.deepcopy(wrapper.state_dict())
        models2, wrapper2, _ = build()
        wrapper2.load_state_dict(state)
        head = models2["ttt_dft"].charge_fitting
        assert all(models2[k].charge_fitting is head for k in HEADS)
        for p, q in zip(
            models["ttt_dft"].charge_fitting.parameters(),
            head.parameters(),
            strict=True,
        ):
            assert torch.equal(p, q)
        assert (
            models2["ttt_dft"].atomic_model.fitting_net
            is not models2["ttt_mace"].atomic_model.fitting_net
        )
