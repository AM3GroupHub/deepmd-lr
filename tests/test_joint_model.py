"""Phase 4: the joint short-range + long-range model.

The central risk this file addresses: the long-range energy is NOT a function
of `edge_vec`, so it cannot inherit DeePMD-kit's edge-based force and virial
machinery. If the long-range branch's own derivatives were wrong -- or absent
-- the energy would shift while the forces did not, and the model would
quietly stop being conservative. Finite differences against the total energy
are therefore the load-bearing tests here, not a nice-to-have.
"""

import copy

import numpy as np
import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.model.model import BaseModel
from deepmd.pt_expt.utils.env import DEVICE

DT = torch.float64

DESCRIPTOR = {
    "type": "dpa4c",
    "rcut": 6.0,
    "channels": 16,
    "lmax": 2,
    "basis_type": "bessel",
    "n_radial": 8,
    "radial_modes": 2,
    "precision": "float64",
    "seed": 42,
}
FITTING = {
    "neuron": [32, 32],
    "activation_function": "silu",
    "precision": "float64",
    "seed": 42,
}
LONG_RANGE = {
    "charge_net": {
        "neuron": [32, 32],
        "activation_function": "silu",
        "precision": "float64",
        "seed": 7,
    },
    # mesh_dimensions rather than mesh_spacing: the mesh is ceil(L / spacing),
    # so with L=10 and spacing=1.0 a strain of 1e-5 flips it from 10 to 11 and
    # the energy is discontinuous exactly where the virial finite difference
    # samples it. Pinning the mesh keeps the energy smooth in the cell.
    # No rcut: only the reciprocal half is computed, so there is no
    # real-space cutoff to set.
    "reciprocal": {"alpha": 0.4, "mesh_dimensions": [15, 15, 15], "spline_order": 6},
}


def build(long_range: bool = True):
    cfg = {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": copy.deepcopy(DESCRIPTOR),
        "fitting_net": copy.deepcopy(FITTING),
    }
    if long_range:
        cfg["long_range"] = copy.deepcopy(LONG_RANGE)
    model = get_model(cfg)
    model.eval()
    return model


def frame(n_mol: int = 4, box: float = 10.0, seed: int = 0):
    """A periodic water-like box with well-separated molecules."""
    rng = np.random.default_rng(seed)
    coords = []
    for _ in range(n_mol):
        o = rng.uniform(1.0, box - 1.0, size=3)
        coords.append(o)
        coords.append(o + np.array([0.96, 0.0, 0.0]))
        coords.append(o + np.array([-0.24, 0.93, 0.0]))
    coord = torch.tensor(np.array(coords), dtype=DT, device=DEVICE).unsqueeze(0)
    atype = torch.tensor([0, 1, 1] * n_mol, dtype=torch.int64, device=DEVICE).unsqueeze(
        0
    )
    cell = (torch.eye(3, dtype=DT, device=DEVICE) * box).unsqueeze(0)
    return coord, atype, cell


class TestComposition:
    def test_long_range_is_optional(self):
        """Without the block, dp_lr is exactly the Phase 1 model."""
        assert build(long_range=False).has_long_range is False
        assert build(long_range=True).has_long_range is True

    def test_energy_decomposes(self):
        """The reported pieces must actually add up to the total."""
        model = build()
        coord, atype, cell = frame()
        out = model(coord, atype, box=cell)
        assert torch.allclose(
            out["energy"],
            out["energy_short_range"] + out["energy_long_range"],
            atol=1e-12,
        )

    def test_short_range_part_matches_the_pure_model(self):
        """Adding the long-range term must not perturb the short-range one."""
        joint = build(long_range=True)
        pure = build(long_range=False)
        pure.load_state_dict(
            {
                k: v
                for k, v in joint.state_dict().items()
                if not k.startswith(("charge_fitting.", "pme."))
            },
            strict=False,
        )
        coord, atype, cell = frame()
        a = joint(coord, atype, box=cell)["energy_short_range"]
        b = pure(coord, atype, box=cell)["energy"]
        assert torch.allclose(a, b, atol=1e-10)

    def test_long_range_term_is_not_negligible(self):
        """Guard against a term that is wired in but effectively zero, which
        would make the conservation tests below pass vacuously."""
        model = build()
        coord, atype, cell = frame()
        out = model(coord, atype, box=cell)
        assert abs(float(out["energy_long_range"])) > 1e-3

    def test_charges_satisfy_the_constraint(self):
        model = build()
        coord, atype, cell = frame()
        charge = model(coord, atype, box=cell)["charge"]
        assert abs(float(charge.sum())) < 1e-10

    def test_charge_state_conditions_the_charge_head_descriptor(self):
        """A required charge/spin condition must reach both descriptor passes."""
        cfg = {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": copy.deepcopy(DESCRIPTOR),
            "fitting_net": copy.deepcopy(FITTING),
            "long_range": copy.deepcopy(LONG_RANGE),
        }
        cfg["descriptor"]["add_chg_spin_ebd"] = True
        model = get_model(cfg).eval()
        coord, atype, cell = frame()
        neutral = torch.tensor([[0.0, 1.0]], dtype=DT, device=DEVICE)
        different_spin = torch.tensor([[0.0, 3.0]], dtype=DT, device=DEVICE)
        q_neutral = model(coord, atype, box=cell, charge_spin=neutral)["charge"]
        q_spin = model(coord, atype, box=cell, charge_spin=different_spin)["charge"]
        assert torch.allclose(q_neutral.sum(-1), neutral[:, 0], atol=1e-10)
        assert torch.allclose(q_spin.sum(-1), different_spin[:, 0], atol=1e-10)


class TestCallingConventions:
    """The trainer and eager callers pass different shapes for the same data.

    DeePMD-kit's trainer flattens coordinates to (nf, nloc * 3) and the cell
    to (nf, 9), while eager callers use (nf, nloc, 3) and (nf, 3, 3). The
    unit tests originally only exercised the eager shapes, so the long-range
    branch raised `batch1 must be a 3D tensor` on the very first training
    step; these pin both conventions.
    """

    def test_accepts_flattened_inputs(self):
        model = build()
        coord, atype, cell = frame()
        nf, nloc = atype.shape

        rect = model(coord, atype, box=cell)
        flat = model(coord.reshape(nf, nloc * 3), atype, box=cell.reshape(nf, 9))
        for key in ("energy", "energy_long_range", "charge"):
            assert torch.allclose(rect[key], flat[key], atol=1e-10), key

    def test_flattened_inputs_give_the_same_force(self):
        model = build()
        coord, atype, cell = frame()
        nf, nloc = atype.shape
        rect = model(coord, atype, box=cell)["force"]
        flat = model(coord.reshape(nf, nloc * 3), atype, box=cell.reshape(nf, 9))[
            "force"
        ]
        assert rect.shape == flat.shape == (nf, nloc, 3)
        assert torch.allclose(rect, flat, atol=1e-10)


class TestConservation:
    """Forces and virial must be exact derivatives of the reported energy."""

    def test_force_matches_finite_difference(self):
        """The load-bearing test: the long-range branch computes its own
        force because it cannot inherit the edge-based one. A missing or
        mis-signed long-range force shows up here and essentially nowhere
        else in normal use."""
        model = build()
        coord, atype, cell = frame()

        analytic = model(coord, atype, box=cell)["force"]

        h = 1e-5
        numeric = torch.zeros_like(analytic)
        for atom in range(coord.shape[1]):
            for axis in range(3):
                plus, minus = coord.clone(), coord.clone()
                plus[0, atom, axis] += h
                minus[0, atom, axis] -= h
                e_plus = float(model(plus, atype, box=cell)["energy"])
                e_minus = float(model(minus, atype, box=cell)["energy"])
                numeric[0, atom, axis] = -(e_plus - e_minus) / (2 * h)

        assert torch.allclose(analytic, numeric, atol=2e-6, rtol=2e-4), (
            f"max |diff| = {float((analytic - numeric).abs().max()):.3e}"
        )

    def test_virial_matches_finite_difference(self):
        """Strain the coordinates and cell together and compare to -dE/de."""
        model = build()
        coord, atype, cell = frame()
        analytic = model(coord, atype, box=cell)["virial"].reshape(3, 3)

        h = 1e-5
        numeric = torch.zeros((3, 3), dtype=DT, device=DEVICE)
        eye = torch.eye(3, dtype=DT, device=DEVICE)
        for i in range(3):
            for j in range(3):
                delta = torch.zeros((3, 3), dtype=DT, device=DEVICE)
                delta[i, j] = h
                delta = 0.5 * (delta + delta.T)  # symmetric strain
                energies = []
                for sign in (1.0, -1.0):
                    strain = eye + sign * delta
                    e = model(coord @ strain.T, atype, box=cell @ strain.T)["energy"]
                    energies.append(float(e))
                numeric[i, j] = -(energies[0] - energies[1]) / (2 * h)

        assert torch.allclose(analytic, numeric, atol=2e-5, rtol=2e-4), (
            f"max |diff| = {float((analytic - numeric).abs().max()):.3e}"
        )

    def test_virial_is_right_above_the_neighbour_crossover(self):
        """The finite-difference virial check again, on a large frame.

        Only the dense builder carries a cell derivative -- `nv` returns
        6.7e-13 where dense gives 1.9e3 -- so a virial that leaned on it would
        break exactly where the cell list runs. Periodic frames use the cell
        list at every size now, so this is the large-frame end of the same
        guarantee the small-frame check above pins.
        """
        model = build()
        coord, atype, cell = frame(n_mol=192, box=32.0)  # 576 atoms
        analytic = model(coord, atype, box=cell)["virial"].reshape(3, 3)

        h = 1e-5
        eye = torch.eye(3, dtype=DT, device=DEVICE)
        numeric = torch.zeros((3, 3), dtype=DT, device=DEVICE)
        for i in range(3):
            for j in range(3):
                delta = torch.zeros((3, 3), dtype=DT, device=DEVICE)
                delta[i, j] = h
                delta = 0.5 * (delta + delta.T)
                energies = []
                for sign in (1.0, -1.0):
                    strain = eye + sign * delta
                    energies.append(
                        float(
                            model(
                                coord @ strain.T, atype, box=cell @ strain.T
                            )["energy"]
                        )
                    )
                numeric[i, j] = -(energies[0] - energies[1]) / (2 * h)

        assert torch.allclose(analytic, numeric, atol=2e-4, rtol=2e-3), (
            f"max |diff| = {float((analytic - numeric).abs().max()):.3e} -- a "
            "cell-blind neighbour builder reached the charge graph"
        )

    def test_shared_backbone_matches_recomputing_it(self):
        """The charge head reads the energy head's features; same answer.

        DeePMD-kit detaches `edge_vec` into its own leaf before the backbone
        runs, so the shared features carry no dependence on our coordinate
        leaf or strain. The charge response therefore comes from
        `dE/d(edge_vec)` via `edge_force_virial` instead. This asserts the two
        routes agree -- if they ever diverge, the reformulation is wrong, and
        it would be wrong silently.
        """
        model = build()
        coord, atype, cell = frame(n_mol=64, box=24.0)
        shared = model(coord, atype, box=cell)

        original = type(model)._long_range

        def without_sharing(self, *args, **kwargs):
            kwargs["shared"] = None
            return original(self, *args, **kwargs)

        type(model)._long_range = without_sharing
        try:
            recomputed = model(coord, atype, box=cell)
        finally:
            type(model)._long_range = original

        for key, tol in (("force", 2e-4), ("virial", 5e-3), ("charge", 1e-5)):
            a = shared[key].detach()
            b = recomputed[key].detach()
            worst = float((a - b).abs().max())
            assert worst < tol, f"{key}: max|diff| = {worst:.3e}"
        assert abs(float(shared["energy"]) - float(recomputed["energy"])) < 1e-3

    def test_backbone_runs_once_per_forward(self):
        """The point of the sharing: one descriptor pass, not two."""
        model = build()
        coord, atype, cell = frame(n_mol=32, box=18.0)
        descriptor = model.atomic_model.descriptor
        original = descriptor.call_graph
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        descriptor.call_graph = counted
        try:
            model(coord, atype, box=cell)
        finally:
            descriptor.call_graph = original
        assert len(calls) == 1, f"backbone ran {len(calls)} times, expected 1"

    def test_virial_is_symmetric(self):
        """A rotation-invariant energy has a symmetric virial."""
        model = build()
        coord, atype, cell = frame()
        virial = model(coord, atype, box=cell)["virial"].reshape(3, 3)
        assert torch.allclose(virial, virial.T, atol=1e-8)


class TestTraining:
    def test_gradients_reach_the_charge_head(self):
        """Energy labels alone must be able to train the charge head."""
        model = build()
        model.train()
        coord, atype, cell = frame()
        model(coord, atype, box=cell)["energy"].sum().backward()
        grads = [p.grad for p in model.charge_fitting.parameters() if p.requires_grad]
        assert grads and any(
            g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
            for g in grads
        )

    def test_force_loss_parameter_gradient_matches_finite_difference(self):
        """The deepest link in the chain, and the only one checked by value.

        Training minimises a force loss, so it needs ``dL/dtheta`` where
        ``L`` already contains a derivative -- a second-order path through
        the long-range branch, enabled by ``create_graph=self.training``. The
        neighbouring tests assert that gradient is merely nonzero and finite,
        which a sign error or a dropped term would satisfy. This one compares
        it to a central difference in the parameter itself.
        """
        model = build()
        model.train()
        coord, atype, cell = frame()

        def force_loss() -> torch.Tensor:
            return (model(coord, atype, box=cell)["force"] ** 2).sum()

        model.zero_grad(set_to_none=True)
        force_loss().backward()
        # By name rather than by attribute walk: `_module_networks` is a
        # ModuleDict with string keys, and the widths depend on the config.
        weight = dict(model.charge_fitting.named_parameters())[
            "nets._module_networks.0.layers.0.w"
        ]
        analytic = weight.grad.detach().clone()
        assert analytic.abs().sum() > 0, "no gradient reached the charge head"

        # A few entries rather than every one: each costs two forwards, and a
        # systematic error shows up in any of them.
        rows, cols = weight.shape
        picks = [(0, 0), (rows // 3, cols // 2), (rows - 1, cols - 1)]
        h = 1e-6
        for row, col in picks:
            with torch.no_grad():
                weight[row, col] += h
            plus = float(force_loss().detach())
            with torch.no_grad():
                weight[row, col] -= 2 * h
            minus = float(force_loss().detach())
            with torch.no_grad():
                weight[row, col] += h
            numeric = (plus - minus) / (2 * h)
            got = float(analytic[row, col])
            scale = max(abs(numeric), abs(got), 1e-8)
            assert abs(got - numeric) / scale < 1e-4, (
                f"w[{row},{col}]: analytic {got:.6e} vs numeric {numeric:.6e}"
            )


class TestGuards:
    def test_accepts_the_legacy_pme_block_name(self):
        """`long_range.pme` was renamed to `long_range.reciprocal`.

        The old name was wrong as soon as `method: "ewald"` existed -- a block
        called `pme` configuring an explicit k-vector sum. It stays accepted
        because it appears in every config written before the rename.
        """
        cfg = {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": copy.deepcopy(DESCRIPTOR),
            "fitting_net": copy.deepcopy(FITTING),
            "long_range": copy.deepcopy(LONG_RANGE),
        }
        legacy = copy.deepcopy(cfg)
        legacy["long_range"]["pme"] = legacy["long_range"].pop("reciprocal")

        new, old = get_model(cfg), get_model(legacy)
        assert new.has_long_range and old.has_long_range
        assert (
            new.long_range_energy.serialize()
            == old.long_range_energy.serialize()
        )

    def test_deserializes_a_checkpoint_written_before_the_rename(self):
        """Old checkpoints carry the long-range block under `pme`."""
        model = build(long_range=True)
        data = model.serialize()
        assert "reciprocal" in data["long_range"]
        data["long_range"]["pme"] = data["long_range"].pop("reciprocal")
        restored = type(model).deserialize(data)
        assert restored.has_long_range
        assert (
            restored.long_range_energy.serialize()
            == model.long_range_energy.serialize()
        )

    def test_accepts_comment_keys_in_a_raw_config(self):
        """`_comment` is DeePMD-kit's annotation convention.

        `dp train` strips these during normalization, so they never reach a
        constructor through the CLI. A raw config read from JSON -- what the
        graft script and any direct `get_model` call use -- still carries
        them, and this model rejects unknown parameters rather than absorbing
        them. Without explicit stripping an annotated config fails exactly
        where the same config works through the CLI.
        """
        cfg = {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": copy.deepcopy(DESCRIPTOR),
            "fitting_net": copy.deepcopy(FITTING),
            "long_range": copy.deepcopy(LONG_RANGE),
        }
        cfg["long_range"]["_comment"] = "explaining the block"
        cfg["long_range"]["reciprocal"]["_comment_alpha"] = "why this alpha"
        cfg["long_range"]["charge_net"]["_comment"] = "why this width"
        model = get_model(cfg)
        assert model.has_long_range

    def test_rejects_unknown_long_range_parameters(self):
        """A stale key must fail loudly rather than run different physics."""
        cfg = {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": copy.deepcopy(DESCRIPTOR),
            "fitting_net": copy.deepcopy(FITTING),
            "long_range": copy.deepcopy(LONG_RANGE),
        }
        cfg["long_range"]["reciprocal"]["rcut"] = 4.0
        with pytest.raises(TypeError, match="rcut"):
            get_model(cfg)

    def test_rejects_missing_box(self):
        model = build()
        coord, atype, _ = frame()
        with pytest.raises(ValueError, match="requires a periodic cell"):
            model(coord, atype, box=None)

    def test_rejects_atomic_virial(self):
        """A per-atom decomposition of a collective k-space sum is undefined."""
        model = build()
        coord, atype, cell = frame()
        with pytest.raises(NotImplementedError, match="per-atom virial"):
            model(coord, atype, box=cell, do_atomic_virial=True)

    def test_rejects_bad_spline_order_from_config(self):
        cfg = {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": copy.deepcopy(DESCRIPTOR),
            "fitting_net": copy.deepcopy(FITTING),
            "long_range": copy.deepcopy(LONG_RANGE),
        }
        cfg["long_range"]["reciprocal"]["spline_order"] = 8
        with pytest.raises(ValueError, match="silently dropped"):
            get_model(cfg)


class TestSerialization:
    def test_round_trip_preserves_the_long_range_term(self):
        model = build()
        data = model.serialize()
        assert data["type"] == "dp_lr"
        assert "long_range" in data

        restored = BaseModel.deserialize(data)
        restored.eval()
        assert restored.has_long_range

        coord, atype, cell = frame()
        before = model(coord, atype, box=cell)
        after = restored.to(DEVICE)(coord, atype, box=cell)
        for key in ("energy", "energy_long_range", "charge"):
            assert torch.allclose(before[key], after[key], atol=1e-10), key

    def test_round_trip_without_long_range(self):
        model = build(long_range=False)
        data = model.serialize()
        assert "long_range" not in data
        assert BaseModel.deserialize(data).has_long_range is False
