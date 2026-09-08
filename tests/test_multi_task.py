"""Multi-task (multi-head) training with a shared DPA4C backbone.

DeePMD-kit already supports multi-task in the ``pt_expt`` backend, and our
``dp_lr`` type drops into it unchanged -- so most of what matters here is
*checking that it stays true*, not new machinery.

Two things could break it silently. Sharing works by aliasing module objects
(``pt_expt/descriptor/se_e2_a.py:46-52``), so if branches ended up holding the
same descriptor *wrapper*, the ``call_graph`` patch that ``_capturing_backbone``
installs would leak from one head into another and the wrong features would
feed the wrong charge net. And a branch is validated against ``dp_lr``'s own
schema rather than ``standard``'s, so any multi-task key we fail to declare
becomes a hard normalization error at the one place it is meant to be used.
"""

import copy

import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.train.wrapper import ModelWrapper
from deepmd.pt_expt.utils.multi_task import preprocess_shared_params
from deepmd.utils.argcheck import normalize

from .test_joint_model import DESCRIPTOR, FITTING, LONG_RANGE, build, frame


def multi_task_config(branches: dict, shared_descriptor: bool = True) -> dict:
    """A normalized two-plus-branch config, in the order the trainer uses it.

    ``preprocess_shared_params`` runs *before* ``normalize`` (see
    ``pt_expt/entrypoints/main.py:321-330``): it substitutes the shared_dict
    references, and only then does the schema see real dicts. Normalizing
    first would reject every reference as a type error -- for ``standard``
    just as much as for ``dp_lr`` -- so the order is the point of this helper.
    """
    model = {
        "shared_dict": {
            "tm": ["O", "H"],
            "des": copy.deepcopy(DESCRIPTOR),
        },
        "model_dict": copy.deepcopy(branches),
    }
    if not shared_descriptor:
        for branch in model["model_dict"].values():
            branch["descriptor"] = copy.deepcopy(DESCRIPTOR)
        model["shared_dict"].pop("des")
    cfg = {
        "model": model,
        "learning_rate": {"start_lr": 1e-3},
        "loss_dict": {k: {"type": "ener"} for k in branches},
        "training": {
            "data_dict": {k: {"training_data": {"systems": ["./d"]}} for k in branches},
            "numb_steps": 1,
        },
    }
    cfg["model"], links = preprocess_shared_params(cfg["model"])
    return normalize(cfg, multi_task=True), links


def lr_branch(long_range: bool = True) -> dict:
    branch = {
        "type": "dp_lr",
        "type_map": "tm",
        "descriptor": "des",
        "fitting_net": copy.deepcopy(FITTING),
    }
    if long_range:
        branch["long_range"] = copy.deepcopy(LONG_RANGE)
    return branch


def build_shared(branches: dict):
    """Build every branch and tie the shared parameters, as the trainer does."""
    cfg, links = multi_task_config(branches)
    models = {
        key: get_model(copy.deepcopy(params))
        for key, params in cfg["model"]["model_dict"].items()
    }
    prob = {key: 1.0 / len(models) for key in models}
    ModelWrapper(models).share_params(links, prob, resume=True)
    return models


class TestSchema:
    def test_dp_lr_branch_normalizes(self):
        """A dp_lr branch is accepted inside model_dict, references and all."""
        cfg, links = multi_task_config({"a": lr_branch(), "b": lr_branch()})
        assert set(cfg["model"]["model_dict"]) == {"a", "b"}
        # Both branches point at the one shared descriptor.
        assert list(links) == ["des"]
        assert [link["model_key"] for link in links["des"]["links"]] == ["a", "b"]
        assert all(link["shared_level"] == 0 for link in links["des"]["links"])

    def test_model_branch_alias_is_accepted(self):
        """The multi-task alias key must be declared on our type, not just on
        `standard` -- a branch is validated against its own schema."""
        branch = lr_branch()
        branch["model_branch_alias"] = ["nickname"]
        cfg, _ = multi_task_config({"a": branch, "b": lr_branch()})
        assert cfg["model"]["model_dict"]["a"]["model_branch_alias"] == ["nickname"]

    def test_info_is_accepted(self):
        branch = lr_branch()
        branch["info"] = {"theory": "PBE-D3"}
        cfg, _ = multi_task_config({"a": branch, "b": lr_branch()})
        assert cfg["model"]["model_dict"]["a"]["info"] == {"theory": "PBE-D3"}

    def test_heterogeneous_branches_normalize(self):
        """One branch with a long-range term, one without -- the layout the
        molecule head uses before the isolated term exists."""
        cfg, _ = multi_task_config({"a": lr_branch(True), "b": lr_branch(False)})
        assert cfg["model"]["model_dict"]["a"]["long_range"] is not None
        assert not cfg["model"]["model_dict"]["b"].get("long_range")


class TestSharing:
    def test_branches_build_with_their_own_long_range(self):
        models = build_shared({"a": lr_branch(), "b": lr_branch()})
        assert set(models) == {"a", "b"}
        assert all(m.has_long_range for m in models.values())

    def test_descriptor_is_shared_but_heads_are_not(self):
        """Sharing must reach the backbone and stop there.

        A shared charge head would force one set of charges on datasets at
        different levels of theory; a shared energy head would defeat the
        point of multi-task entirely.
        """
        models = build_shared({"a": lr_branch(), "b": lr_branch()})
        da = models["a"].atomic_model.descriptor
        db = models["b"].atomic_model.descriptor

        # Same parameters ...
        assert da.type_embedding is db.type_embedding
        assert da.readout is db.readout
        # ... reached through per-branch wrapper objects, which is what keeps
        # the call_graph patch below from crossing heads.
        assert da is not db
        # ... and nothing above the backbone is tied.
        assert models["a"].charge_fitting is not models["b"].charge_fitting
        assert models["a"].long_range_energy is not models["b"].long_range_energy

    def test_shared_storage_is_real(self):
        """Aliasing, not copying: the tensors must be the same memory."""
        models = build_shared({"a": lr_branch(), "b": lr_branch()})
        sa, sb = models["a"].state_dict(), models["b"].state_dict()
        common = set(sa) & set(sb)
        tied = {k for k in common if sa[k].data_ptr() == sb[k].data_ptr()}
        assert tied, "no parameter is actually shared"
        # The charge head is per-branch, so it must NOT be among them.
        assert not any(k.startswith("charge_fitting.") for k in tied)

    def test_unshared_descriptor_stays_separate(self):
        """Control: without the shared_dict reference nothing is tied."""
        cfg, links = multi_task_config(
            {"a": lr_branch(), "b": lr_branch()}, shared_descriptor=False
        )
        assert not links
        models = {
            key: get_model(copy.deepcopy(params))
            for key, params in cfg["model"]["model_dict"].items()
        }
        da = models["a"].atomic_model.descriptor
        db = models["b"].atomic_model.descriptor
        assert da.type_embedding is not db.type_embedding


class TestBackboneCaptureIsolation:
    """`_capturing_backbone` patches `descriptor.call_graph` in place.

    Under sharing that is only safe because each branch keeps its own
    descriptor wrapper. If that ever changes, one head's charge net would read
    another head's features -- a wrong answer with no error anywhere.
    """

    def test_patch_does_not_reach_the_other_branch(self):
        models = build_shared({"a": lr_branch(), "b": lr_branch()})
        da = models["a"].atomic_model.descriptor
        db = models["b"].atomic_model.descriptor
        with models["a"]._capturing_backbone():
            assert "call_graph" in da.__dict__
            assert "call_graph" not in db.__dict__

    def test_patch_is_removed_afterwards(self):
        """Restoring by assignment would leave a bound method shadowing the
        class attribute forever; the descriptor must come back unchanged."""
        model = build()
        descriptor = model.atomic_model.descriptor
        assert "call_graph" not in descriptor.__dict__
        with model._capturing_backbone():
            assert "call_graph" in descriptor.__dict__
        assert "call_graph" not in descriptor.__dict__

    def test_nested_capture_restores_the_outer_patch(self):
        model = build()
        descriptor = model.atomic_model.descriptor
        with model._capturing_backbone():
            outer = descriptor.__dict__["call_graph"]
            with model._capturing_backbone():
                assert descriptor.__dict__["call_graph"] is not outer
            assert descriptor.__dict__["call_graph"] is outer
        assert "call_graph" not in descriptor.__dict__

    def test_no_patch_left_when_a_forward_raises(self):
        model = build()
        descriptor = model.atomic_model.descriptor
        with pytest.raises(RuntimeError):
            with model._capturing_backbone():
                raise RuntimeError("boom")
        assert "call_graph" not in descriptor.__dict__


class TestRaggedIsRefused:
    """The ragged path bypasses our forward, so it must fail rather than
    return the short-range energy alone."""

    def test_forward_ragged_raises_with_long_range(self):
        model = build(long_range=True)
        coord, atype, cell = frame(n_mol=2)
        n_node = torch.tensor([coord.shape[1]], dtype=torch.int64, device=coord.device)
        with pytest.raises(NotImplementedError, match="[Rr]agged"):
            model.forward_ragged(
                coord.reshape(-1, 3), atype.reshape(-1), n_node, box=cell
            )

    def test_forward_ragged_is_allowed_without_long_range(self):
        """A short-range dp_lr is an ordinary energy model and must not be
        penalized by the guard."""
        model = build(long_range=False)
        coord, atype, cell = frame(n_mol=2)
        n_node = torch.tensor([coord.shape[1]], dtype=torch.int64, device=coord.device)
        out = model.forward_ragged(
            coord.reshape(-1, 3), atype.reshape(-1), n_node, box=cell
        )
        assert torch.isfinite(out["energy"]).all()


class TestThreeBranchInterfaceBulkMolecule:
    """The configuration this feature exists for.

    A 2D-periodic slab, a 3D-periodic bulk and an isolated molecule, sharing
    one DPA4C backbone. Each branch needs its own boundary treatment -- slab
    correction on, off, and no cell at all -- which is precisely why they
    cannot be one head no matter how the energies are labelled.
    """

    @staticmethod
    def config():
        descriptor = copy.deepcopy(DESCRIPTOR)
        # Charge conditioning has to be on for every branch or the shared
        # descriptor refuses to link; the default then keeps the neutral
        # branches from having to carry a charge_spin of their own.
        descriptor["add_chg_spin_ebd"] = True
        descriptor["default_chg_spin"] = [0, 1]

        def branch(reciprocal=None):
            b = {
                "type": "dp_lr",
                "type_map": "tm",
                "descriptor": "des",
                "fitting_net": copy.deepcopy(FITTING),
            }
            if reciprocal is not None:
                b["long_range"] = {
                    "charge_net": copy.deepcopy(LONG_RANGE["charge_net"]),
                    "reciprocal": reciprocal,
                }
            return b

        model = {
            "shared_dict": {"tm": ["O", "H"], "des": descriptor},
            "model_dict": {
                "interface": branch(
                    {
                        "alpha": 0.4,
                        "mesh_dimensions": [15, 15, 30],
                        "slab_correction": True,
                        "pbc": [True, True, False],
                    }
                ),
                "bulk": branch(
                    {
                        "alpha": 0.4,
                        "mesh_dimensions": [15, 15, 15],
                        "slab_correction": False,
                        "pbc": [True, True, True],
                    }
                ),
                # Same alpha as the periodic branches -- that is what makes
                # the remainder left to the shared backbone the same shape in
                # every branch.
                "molecule": branch({"method": "isolated", "alpha": 0.4}),
            },
        }
        keys = list(model["model_dict"])
        cfg = {
            "model": model,
            "learning_rate": {"start_lr": 1e-3},
            "loss_dict": {
                "interface": {"type": "ener"},
                "bulk": {"type": "ener"},
                # No cell, so no virial label to fit.
                "molecule": {"type": "ener", "start_pref_v": 0, "limit_pref_v": 0},
            },
            "training": {
                "model_prob": {"interface": 0.4, "bulk": 0.4, "molecule": 0.2},
                "data_dict": {k: {"training_data": {"systems": ["./d"]}} for k in keys},
                "numb_steps": 1,
            },
        }
        cfg["model"], links = preprocess_shared_params(cfg["model"])
        return normalize(cfg, multi_task=True), links

    def build(self):
        cfg, links = self.config()
        models = {
            key: get_model(copy.deepcopy(params))
            for key, params in cfg["model"]["model_dict"].items()
        }
        ModelWrapper(models).share_params(
            links, {"interface": 0.4, "bulk": 0.4, "molecule": 0.2}, resume=True
        )
        return models

    def test_each_branch_gets_its_own_boundary_treatment(self):
        models = self.build()
        assert models["interface"].long_range_energy.slab_correction is True
        assert models["bulk"].long_range_energy.slab_correction is False
        assert models["molecule"].long_range_energy.needs_cell is False
        assert models["interface"].long_range_energy.needs_cell is True

    def test_every_branch_splits_coulomb_at_the_same_alpha(self):
        """The shared backbone learns the erfc(alpha r)/r remainder. If the
        branches split at different alphas it would be asked for two different
        functions at once."""
        models = self.build()
        alphas = {m.long_range_energy.alpha for m in models.values()}
        assert alphas == {0.4}

    def test_all_three_share_one_backbone(self):
        models = self.build()
        embeddings = [m.atomic_model.descriptor.type_embedding for m in models.values()]
        assert embeddings[0] is embeddings[1] is embeddings[2]

    def test_every_branch_runs_on_its_own_kind_of_frame(self):
        """Periodic branches get a cell, the molecule branch gets none -- and
        the molecule branch takes a per-frame total charge."""
        models = self.build()
        coord, atype, cell = frame(n_mol=3)
        anion = torch.tensor([[-1.0, 1.0]], dtype=coord.dtype, device=coord.device)

        for name, box, kwargs in (
            ("interface", cell, {}),
            ("bulk", cell, {}),
            ("molecule", None, {"charge_spin": anion}),
        ):
            out = models[name](coord, atype, box=box, **kwargs)
            assert torch.isfinite(out["energy"]).all(), name
            assert torch.isfinite(out["force"]).all(), name
            assert torch.isfinite(out["energy_long_range"]).all(), name
            # The pieces must still add up in every boundary condition.
            assert float(out["energy"]) == pytest.approx(
                float(out["energy_short_range"]) + float(out["energy_long_range"]),
                rel=1e-10,
            ), name

    def test_a_molecule_step_updates_the_shared_backbone(self):
        """The point of the whole arrangement: gradients from one dataset
        reach the descriptor the other branches use."""
        models = self.build()
        coord, atype, _cell = frame(n_mol=3)
        anion = torch.tensor([[-1.0, 1.0]], dtype=coord.dtype, device=coord.device)

        weight = next(
            iter(
                models["interface"].atomic_model.descriptor.type_embedding.parameters()
            )
        )
        before = weight.clone()
        models["molecule"](coord, atype, box=None, charge_spin=anion)[
            "energy"
        ].sum().backward()
        assert weight.grad is not None
        assert float(weight.grad.abs().sum()) > 0.0

        torch.optim.SGD([weight], lr=1e-2).step()
        assert not torch.equal(before, weight)

    def test_a_charged_molecule_keeps_its_total_charge(self):
        models = self.build()
        coord, atype, _cell = frame(n_mol=3)
        for total in (-1.0, 0.0, 2.0):
            cs = torch.tensor([[total, 1.0]], dtype=coord.dtype, device=coord.device)
            out = models["molecule"](coord, atype, box=None, charge_spin=cs)
            assert float(out["charge"].sum()) == pytest.approx(total, abs=1e-10)

    def test_periodic_branches_conserve_total_charge(self):
        models = self.build()
        coord, atype, cell = frame(n_mol=3)
        for name in ("interface", "bulk"):
            out = models[name](coord, atype, box=cell)
            assert out["charge"].sum().abs() < 1e-10, name
