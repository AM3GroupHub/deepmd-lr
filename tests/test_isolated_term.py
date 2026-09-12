"""The long-range term for systems with no cell.

The load-bearing claim is not that this term is *reasonable* for an isolated
molecule -- lots of things would be -- but that it is the **same quantity** the
periodic term computes. Ewald splits Coulomb at alpha into a screened part the
descriptor learns and a smooth part we supply; if the isolated branch supplied
a differently-shaped remainder, a shared backbone trained on both would be
asked to do two incompatible jobs at once.

So the central test here is the large-cell limit: grow the box and the periodic
`recip - self` must converge onto the direct erf sum, as 1/L^3. That single
check would catch a wrong alpha, a missing factor of a half, a self term added
or forgotten, or a wrong unit conversion -- every way this could be subtly
wrong while still looking plausible.
"""

import copy

import numpy as np
import pytest
import torch
from deepmd.dpmodel.modifier.dipole_charge import ELECTROSTATIC_CONVERSION
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.lr.term import (
    IsolatedLongRangeEnergy,
    LongRangeEnergy,
    build_long_range_energy,
)

from .test_joint_model import DESCRIPTOR, FITTING, LONG_RANGE

DT = torch.float64
ALPHA = 0.4


def cluster(n: int = 8, spread: float = 6.0, seed: int = 0, charged: float = 0.0):
    """A neutral (or deliberately charged) point-charge cluster."""
    rng = np.random.default_rng(seed)
    coord = torch.tensor(
        rng.uniform(0.0, spread, size=(1, n, 3)), dtype=DT, device=DEVICE
    )
    q = rng.normal(size=(1, n))
    q = q - q.mean() + charged / n
    return coord, torch.tensor(q, dtype=DT, device=DEVICE)


def pairwise_erf(coord, charge, alpha=ALPHA):
    """The textbook sum, written out independently of the implementation."""
    c = coord[0].detach().cpu().numpy()
    q = charge[0].detach().cpu().numpy()
    from scipy.special import erf  # noqa: PLC0415

    total = 0.0
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            r = float(np.linalg.norm(c[i] - c[j]))
            total += q[i] * q[j] * erf(alpha * r) / r
    return total * ELECTROSTATIC_CONVERSION


class TestMatchesThePeriodicTerm:
    """The reason this term is defined the way it is."""

    def test_converges_to_the_periodic_result_as_the_cell_grows(self):
        coord, q = cluster()
        isolated = IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)
        reference = isolated(coord, q).sum()

        periodic = LongRangeEnergy(alpha=ALPHA, method="ewald", accuracy=1e-10).to(
            DEVICE
        )
        diffs = []
        for length in (80.0, 160.0, 320.0):
            box = (torch.eye(3, dtype=DT, device=DEVICE) * length).unsqueeze(0)
            shifted = coord + length / 2 - 3.0
            diffs.append(abs(float(periodic(shifted, q, box).sum() - reference)))

        # Leading error is the dipole-image interaction, which falls as 1/L^3.
        # Anything shallower means the two terms are not the same quantity.
        assert diffs[0] > diffs[1] > diffs[2]
        for coarse, fine in zip(diffs, diffs[1:], strict=False):
            assert coarse / fine == pytest.approx(8.0, rel=0.15)
        assert diffs[-1] < 2e-4

    def test_matches_a_hand_written_pair_sum(self):
        pytest.importorskip("scipy")
        coord, q = cluster(n=6, seed=3)
        got = float(IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)(coord, q).sum())
        assert got == pytest.approx(pairwise_erf(coord, q), rel=1e-12)

    def test_no_self_energy_is_left_in(self):
        """A single charge cannot interact with anything, including itself."""
        coord = torch.zeros((1, 1, 3), dtype=DT, device=DEVICE)
        q = torch.tensor([[1.5]], dtype=DT, device=DEVICE)
        assert float(IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)(coord, q)) == 0.0


class TestPhysics:
    def test_invariant_under_translation_and_rotation(self):
        coord, q = cluster()
        term = IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)
        base = term(coord, q).sum()

        shifted = term(
            coord + torch.tensor([3.1, -2.0, 7.7], dtype=DT, device=DEVICE), q
        )
        assert float(shifted.sum()) == pytest.approx(float(base), rel=1e-12)

        theta = 0.7
        rot = torch.tensor(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=DT,
            device=DEVICE,
        )
        turned = term(coord @ rot.T, q).sum()
        assert float(turned) == pytest.approx(float(base), rel=1e-12)

    def test_permutation_invariant(self):
        coord, q = cluster()
        term = IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)
        order = torch.randperm(coord.shape[1], device=DEVICE)
        assert float(term(coord[:, order], q[:, order]).sum()) == pytest.approx(
            float(term(coord, q).sum()), rel=1e-12
        )

    def test_charged_clusters_are_finite_and_correct(self):
        """No neutralising background is needed without periodic images, so a
        net charge is not a special case here the way it is in a cell."""
        pytest.importorskip("scipy")
        for net in (-2.0, -1.0, 1.0, 3.0):
            coord, q = cluster(charged=net, seed=11)
            assert float(q.sum()) == pytest.approx(net, abs=1e-12)
            got = float(IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)(coord, q).sum())
            assert np.isfinite(got)
            assert got == pytest.approx(pairwise_erf(coord, q), rel=1e-12)

    def test_masked_atoms_are_exactly_absent(self):
        coord, q = cluster(n=8)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], device=DEVICE)
        term = IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)
        masked = term(coord, q, mask=mask).sum()
        trimmed = term(coord[:, :5], q[:, :5]).sum()
        assert float(masked) == pytest.approx(float(trimmed), abs=1e-14)

    def test_batching_matches_frame_by_frame(self):
        c0, q0 = cluster(seed=1)
        c1, q1 = cluster(seed=2)
        term = IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)
        batched = term(torch.cat([c0, c1]), torch.cat([q0, q1]))
        assert float(batched[0].sum()) == pytest.approx(float(term(c0, q0).sum()))
        assert float(batched[1].sum()) == pytest.approx(float(term(c1, q1).sum()))


class TestDifferentiability:
    def test_force_matches_finite_difference(self):
        coord, q = cluster(n=6, seed=5)
        term = IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)
        leaf = coord.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(term(leaf, q).sum(), leaf)

        eps = 1e-6
        for i in (0, 3, 5):
            for k in range(3):
                plus, minus = coord.clone(), coord.clone()
                plus[0, i, k] += eps
                minus[0, i, k] -= eps
                fd = float(term(plus, q).sum() - term(minus, q).sum()) / (2 * eps)
                assert float(grad[0, i, k]) == pytest.approx(fd, abs=1e-8)

    def test_survives_double_backward(self):
        """Force-loss training differentiates the force again, so the term has
        to be twice differentiable -- the requirement that rules out
        nvalchemiops' coulomb kernel."""
        coord, q = cluster(n=5, seed=6)
        term = IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)
        leaf = coord.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(term(leaf, q).sum(), leaf, create_graph=True)
        (second,) = torch.autograd.grad(grad.pow(2).sum(), leaf)
        assert torch.isfinite(second).all()
        assert float(second.abs().sum()) > 0.0

    def test_charge_gradient_is_available(self):
        coord, q = cluster()
        leaf = q.clone().requires_grad_(True)
        term = IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)
        (grad,) = torch.autograd.grad(term(coord, leaf).sum(), leaf)
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0

    def test_coincident_atoms_do_not_produce_nan_gradients(self):
        """sqrt(0) has an infinite derivative, so the diagonal has to be
        removed before the square root, not after."""
        coord = torch.zeros((1, 3, 3), dtype=DT, device=DEVICE)
        coord[0, 1, 0] = 1.5
        coord[0, 2, 1] = 2.5
        q = torch.tensor([[1.0, -0.5, -0.5]], dtype=DT, device=DEVICE)
        leaf = coord.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(
            IsolatedLongRangeEnergy(alpha=ALPHA).to(DEVICE)(leaf, q).sum(), leaf
        )
        assert torch.isfinite(grad).all()


class TestConfiguration:
    def test_factory_dispatches_on_method(self):
        assert isinstance(
            build_long_range_energy({"method": "isolated", "alpha": 0.4}),
            IsolatedLongRangeEnergy,
        )
        for method in ("pme", "ewald"):
            assert isinstance(
                build_long_range_energy({"method": method, "alpha": 0.4}),
                LongRangeEnergy,
            )

    def test_factory_defaults_to_pme(self):
        assert isinstance(build_long_range_energy({"alpha": 0.4}), LongRangeEnergy)

    def test_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="method must be one of"):
            build_long_range_energy({"method": "nonsense"})

    def test_schema_defaults_are_not_mistaken_for_settings(self):
        """`normalize` fills the whole reciprocal block in before the model is
        built, so an isolated config arrives carrying periodic keys nobody
        wrote. Refusing those would make the method unusable from a config
        file -- which is exactly what happened the first time."""
        from deepmd_lr.lr.term import DEFAULT_SPLINE_ORDER  # noqa: PLC0415

        term = IsolatedLongRangeEnergy(
            method="isolated",
            alpha=0.4,
            mesh_spacing=None,
            mesh_dimensions=None,
            spline_order=DEFAULT_SPLINE_ORDER,
            k_cutoff=None,
            accuracy=1e-6,
            slab_correction=False,
            pbc=None,
        )
        assert term.alpha == 0.4

    def test_periodic_only_settings_are_refused(self):
        """Silently ignoring a mesh setting would be indistinguishable from
        applying it."""
        for bad in (
            {"slab_correction": True},
            {"pbc": [True, True, False]},
            {"mesh_dimensions": [8, 8, 8]},
            {"k_cutoff": 1.0},
        ):
            with pytest.raises(ValueError, match="do not apply to"):
                IsolatedLongRangeEnergy(alpha=0.4, **bad)

    def test_rejects_nonpositive_alpha(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            IsolatedLongRangeEnergy(alpha=0.0)

    def test_serialization_round_trip(self):
        term = IsolatedLongRangeEnergy(alpha=0.37)
        restored = build_long_range_energy(term.serialize())
        assert isinstance(restored, IsolatedLongRangeEnergy)
        assert restored.alpha == 0.37

    def test_needs_cell_distinguishes_the_two(self):
        assert IsolatedLongRangeEnergy(alpha=0.4).needs_cell is False
        assert LongRangeEnergy(alpha=0.4).needs_cell is True


def isolated_model():
    cfg = {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": copy.deepcopy(DESCRIPTOR),
        "fitting_net": copy.deepcopy(FITTING),
        "long_range": {
            "charge_net": copy.deepcopy(LONG_RANGE["charge_net"]),
            "reciprocal": {"method": "isolated", "alpha": ALPHA},
        },
    }
    model = get_model(cfg)
    model.eval()
    return model


def molecule(seed: int = 0):
    """One isolated cluster of water-like molecules -- no cell."""
    rng = np.random.default_rng(seed)
    coords = []
    for _ in range(3):
        o = rng.uniform(0.0, 5.0, size=3)
        coords += [o, o + np.array([0.96, 0.0, 0.0]), o + np.array([-0.24, 0.93, 0.0])]
    coord = torch.tensor(np.array(coords), dtype=DT, device=DEVICE).unsqueeze(0)
    atype = torch.tensor([0, 1, 1] * 3, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    return coord, atype


class TestInTheModel:
    def test_runs_without_a_cell(self):
        coord, atype = molecule()
        out = isolated_model()(coord, atype, box=None)
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["force"]).all()
        assert "energy_long_range" in out
        assert float(out["energy_long_range"].abs().sum()) > 0.0

    def test_energy_still_decomposes(self):
        coord, atype = molecule()
        out = isolated_model()(coord, atype, box=None)
        total = float(out["energy"])
        parts = float(out["energy_short_range"]) + float(out["energy_long_range"])
        assert total == pytest.approx(parts, rel=1e-10)

    def test_forces_are_conservative(self):
        """The charge-response force is what makes this non-trivial: charges
        move when atoms do, so -dE/dr is not just the fixed-charge force."""
        model = isolated_model()
        coord, atype = molecule(seed=2)
        out = model(coord, atype, box=None)
        force = out["force"].detach()

        eps = 1e-5
        for i in (0, 4, 8):
            for k in range(3):
                plus, minus = coord.clone(), coord.clone()
                plus[0, i, k] += eps
                minus[0, i, k] -= eps
                e_plus = float(model(plus, atype, box=None)["energy"])
                e_minus = float(model(minus, atype, box=None)["energy"])
                fd = -(e_plus - e_minus) / (2 * eps)
                assert float(force[0, i, k]) == pytest.approx(fd, abs=2e-6)

    def test_charges_respect_a_requested_total(self):
        model = isolated_model()
        coord, atype = molecule()
        for total in (0.0, -1.0, 2.0):
            cs = torch.tensor([[total, 1.0]], dtype=DT, device=DEVICE)
            out = model(coord, atype, box=None, charge_spin=cs)
            assert float(out["charge"].sum()) == pytest.approx(total, abs=1e-10)

    def test_a_periodic_term_still_refuses_a_missing_cell(self):
        """Relaxing the guard must not relax it for the term that does need
        a cell."""
        from .test_joint_model import build  # noqa: PLC0415

        coord, atype = molecule()
        with pytest.raises(ValueError, match="requires a periodic cell"):
            build(long_range=True)(coord, atype, box=None)

    def test_serialization_round_trip_through_the_model(self):
        model = isolated_model()
        from deepmd.pt_expt.model.model import BaseModel  # noqa: PLC0415

        restored = BaseModel.deserialize(model.serialize())
        assert isinstance(restored.long_range_energy, IsolatedLongRangeEnergy)
        coord, atype = molecule()
        got = float(restored.to(DEVICE)(coord, atype, box=None)["energy"])
        assert got == pytest.approx(float(model(coord, atype, box=None)["energy"]))


class TestNonPeriodicStatistics:
    """Training statistics must survive a `nopbc` dataset.

    DeePMD-kit's `make_stat_input` hands back an all-ZERO box for a system
    marked `nopbc`, not `None`. Everything downstream tests `box is None` --
    `DescrptDPA4C.compute_input_stats` does so explicitly -- so the test never
    fires and a singular cell reaches `linalg.inv`, killing training before
    step 1. The training loop itself is fine; it reads through a different
    path that does return `None`. Only the statistics pass sees the zeros.
    """

    def test_zero_box_is_treated_as_no_cell(self):
        model = isolated_model()
        coord, atype = molecule()
        nframes, natoms = 4, coord.shape[1]
        rng = np.random.default_rng(0)
        seen = {}

        def sampled():
            return [
                {
                    "coord": rng.uniform(0, 5, (nframes, natoms * 3)),
                    "atype": np.tile(atype[0].cpu().numpy(), (nframes, 1)),
                    # the shape and dtype make_stat_input actually produces
                    "box": np.zeros((nframes, 9)),
                    "natoms": np.full((nframes, 2 + 2), natoms),
                }
            ]

        def capture(_self, func, *args, **kwargs):
            # patched onto the parent class, so it is called bound -- the
            # sampled_func arrives as the SECOND argument
            seen["box"] = func()[0]["box"]

        original = type(model).__mro__[1].compute_or_load_stat
        try:
            type(model).__mro__[1].compute_or_load_stat = capture
            model.compute_or_load_stat(sampled)
        finally:
            type(model).__mro__[1].compute_or_load_stat = original
        assert seen["box"] is None, (
            "an all-zero box must reach the descriptor as None, or "
            "compute_input_stats inverts a singular cell"
        )

    def test_a_real_box_is_left_alone(self):
        """The sanitising must not eat a legitimate cell."""
        from .test_joint_model import build  # noqa: PLC0415

        model = build(long_range=True)
        cell = np.tile(np.eye(3).reshape(-1) * 12.0, (3, 1))
        seen = {}

        def sampled():
            return [{"box": cell.copy()}]

        def capture(_self, func, *args, **kwargs):
            # patched onto the parent class, so it is called bound -- the
            # sampled_func arrives as the SECOND argument
            seen["box"] = func()[0]["box"]

        original = type(model).__mro__[1].compute_or_load_stat
        try:
            type(model).__mro__[1].compute_or_load_stat = capture
            model.compute_or_load_stat(sampled)
        finally:
            type(model).__mro__[1].compute_or_load_stat = original
        assert seen["box"] is not None
        assert np.allclose(seen["box"], cell)
