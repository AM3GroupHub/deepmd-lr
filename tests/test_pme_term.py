"""Phase 3: the long-range (reciprocal-space) electrostatic term.

`tests/test_pme_physics.py` validates the nvalchemi engine itself. This file
validates *our wrapper*: that it computes the reciprocal half of the Ewald
decomposition in DeePMD-kit's own convention and units, that the boundary
conditions Ewald is invalid under are rejected rather than silently computed,
and that gradients reach both the coordinates and the charges.

The oracle is DeePMD-kit's `ewald_reciprocal_energy`, the function its DPLR
path uses. Matching it is the whole specification: same half of the split,
same units, same treatment of the self term (DeePMD-kit keeps it in;
nvalchemi subtracts it, so the wrapper adds it back).
"""

import itertools

import numpy as np
import pytest
import torch
from deepmd.dpmodel.modifier.dipole_charge import (
    ELECTROSTATIC_CONVERSION as KE,
)
from deepmd.dpmodel.modifier.dipole_charge import (
    compute_ewald_grids,
    ewald_reciprocal_energy,
)
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.pme import MAX_SPLINE_ORDER, LongRangeEnergy

MADELUNG_NACL = 1.7475645946331821
DT = torch.float64
#: Fine enough that the mesh is never the limiting error in these comparisons.
FINE_MESH = (64, 64, 64)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the PME kernels run on GPU"
)


def nacl(reps: int = 3, a: float = 2.0):
    """Rocksalt lattice with nearest-neighbour distance ``a / 2``."""
    pos, q = [], []
    for s in itertools.product(range(reps), repeat=3):
        for i, j, k in itertools.product(range(2), repeat=3):
            pos.append([(s[0] + i / 2) * a, (s[1] + j / 2) * a, (s[2] + k / 2) * a])
            q.append(1.0 if (i + j + k) % 2 == 0 else -1.0)
    coord = torch.tensor(pos, dtype=DT, device=DEVICE).unsqueeze(0)
    charge = torch.tensor(q, dtype=DT, device=DEVICE).unsqueeze(0)
    cell = (torch.eye(3, dtype=DT, device=DEVICE) * (a * reps)).unsqueeze(0)
    return coord, charge, cell


def random_neutral(n: int = 16, box: float = 12.0, seed: int = 0):
    """A neutral cell of random charges -- the general case, not a lattice."""
    rng = np.random.default_rng(seed)
    coord = torch.tensor(
        rng.uniform(0.0, box, size=(1, n, 3)), dtype=DT, device=DEVICE
    )
    charge = torch.tensor(rng.normal(size=(1, n)), dtype=DT, device=DEVICE)
    charge = charge - charge.mean()
    cell = (torch.eye(3, dtype=DT, device=DEVICE) * box).unsqueeze(0)
    return coord, charge, cell


def deepmd_reference(coord, charge, cell, alpha):
    """DeePMD-kit's own DPLR reciprocal energy, in eV, for one frame."""
    grids = compute_ewald_grids(cell.detach().cpu().numpy(), 0.4)
    return float(ewald_reciprocal_energy(coord, charge, cell, grids, alpha)[0, 0])


class TestMatchesDeepmdConvention:
    """The specification: reproduce DeePMD-kit's DPLR reciprocal energy.

    Pins three things at once -- that only the reciprocal half is computed,
    that the self term is handled DeePMD-kit's way, and that the eV
    conversion is right.
    """

    @pytest.mark.parametrize("alpha", [0.3, 0.4, 0.6])
    def test_matches_dplr_reciprocal_energy(self, alpha):
        coord, charge, cell = random_neutral()
        term = LongRangeEnergy(alpha=alpha, mesh_dimensions=FINE_MESH)
        ours = float(term(coord, charge, cell).sum())
        reference = deepmd_reference(coord, charge, cell, alpha)
        assert abs(ours - reference) < 1e-5 * abs(reference)

    def test_matches_for_an_ionic_lattice(self):
        """A crystal, where the structure factor is sharp rather than broad.

        alpha is 1.0 here, chosen between two constraints. A rocksalt lattice
        orders charge on a 2 A period, so its structure factor lives entirely
        at short wavelength; at alpha = 0.5 the Gaussian exp(-k^2/4 alpha^2)
        suppresses the reciprocal term to ~1e-9 eV and comparing two zeros
        would prove nothing. But the oracle degrades in the other direction:
        `compute_ewald_grids` uses a fixed 0.4 A k-grid, which truncates as
        alpha grows, so by alpha = 2.0 the *reference* is the inaccurate one
        (they differ by 0.6%, and by 10% at alpha = 3.0). alpha = 1.0 is
        where the term carries real weight and the oracle is still converged.
        """
        coord, charge, cell = nacl()
        term = LongRangeEnergy(alpha=1.0, mesh_dimensions=FINE_MESH)
        ours = float(term(coord, charge, cell).sum())
        reference = deepmd_reference(coord, charge, cell, 1.0)
        assert abs(reference) > 1.0, "reference is ~0; the test would be vacuous"
        assert abs(ours - reference) < 1e-5 * abs(reference)

    def test_reciprocal_term_is_small_for_short_wavelength_order(self):
        """The physical corollary, worth stating explicitly.

        The long-range term only carries weight where the charge density has
        long-wavelength structure. A tightly alternating lattice at a
        realistic alpha has almost none, so almost all of its electrostatics
        is short-ranged and the network handles it. This is why alpha is a
        modelling choice and not a free parameter.
        """
        coord, charge, cell = nacl()
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=FINE_MESH)
        assert abs(float(term(coord, charge, cell).sum())) < 1e-6

    def test_excludes_the_real_space_part(self):
        """The reciprocal sum alone must NOT equal the full Coulomb energy.

        A guard against silently reintroducing the real-space term, which
        would double-count what the short-range network already covers.
        """
        a = 2.0
        coord, charge, cell = nacl(reps=3, a=a)
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=FINE_MESH)
        ours = float(term(coord, charge, cell).sum())
        full_coulomb = -MADELUNG_NACL * (coord.shape[1] / 2) * KE / (a / 2)
        assert abs(ours - full_coulomb) > 0.1 * abs(full_coulomb)

    def test_alpha_changes_the_split(self):
        """alpha is a modelling choice here, not an invariance.

        In a full Ewald sum the total is alpha-independent. For the
        reciprocal half alone it must move, because alpha is precisely what
        decides how much work is handed to the short-range network.
        """
        coord, charge, cell = random_neutral()
        a = float(LongRangeEnergy(alpha=0.3, mesh_dimensions=FINE_MESH)(
            coord, charge, cell).sum())
        b = float(LongRangeEnergy(alpha=0.6, mesh_dimensions=FINE_MESH)(
            coord, charge, cell).sum())
        assert abs(a - b) > 0.1 * abs(a)

    def test_needs_no_neighbor_list(self):
        """Dropping the real-space half removes the neighbour list entirely,
        so no cutoff can exceed the cell and no capacity can overflow. This
        cell is far smaller than any sensible real-space cutoff."""
        coord, charge, cell = nacl(reps=2, a=2.0)  # L = 4 A
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        assert torch.isfinite(term(coord, charge, cell)).all()


class TestBoundaryConditions:
    """Ewald is only valid for a periodic, non-degenerate cell."""

    def test_rejects_missing_box(self):
        coord, charge, _ = random_neutral()
        with pytest.raises(ValueError, match="periodic cell"):
            LongRangeEnergy()(coord, charge, None)

    def test_rejects_degenerate_cell(self):
        coord, charge, cell = random_neutral()
        flat = cell.clone()
        flat[0, 2, 2] = 0.0
        with pytest.raises(ValueError, match="non-degenerate cell"):
            LongRangeEnergy()(coord, charge, flat)

    def test_rejects_bad_shapes(self):
        coord, charge, cell = random_neutral()
        with pytest.raises(ValueError, match=r"shape \(nf, nloc\)"):
            LongRangeEnergy()(coord, charge[:, :1], cell)


class TestParameterValidation:
    """Guard the silent-failure mode found in Phase 0."""

    @pytest.mark.parametrize("order", [7, 8, 10])
    def test_rejects_unsupported_spline_order(self, order):
        with pytest.raises(ValueError, match="silently dropped"):
            LongRangeEnergy(spline_order=order)

    def test_rejects_too_low_spline_order(self):
        with pytest.raises(ValueError, match="at least 3"):
            LongRangeEnergy(spline_order=2)

    def test_default_order_is_within_supported_range(self):
        assert LongRangeEnergy().spline_order <= MAX_SPLINE_ORDER

    @pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": -1.0},
                                        {"mesh_spacing": 0.0}])
    def test_rejects_nonsense_parameters(self, kwargs):
        with pytest.raises(ValueError):
            LongRangeEnergy(**kwargs)

    def test_rejects_bad_mesh_dimensions(self):
        with pytest.raises(ValueError, match="three integers"):
            LongRangeEnergy(mesh_dimensions=(8, 8))
        with pytest.raises(ValueError, match="three integers"):
            LongRangeEnergy(mesh_dimensions=(8, 8, 1))


class TestMasking:
    def test_padding_atoms_do_not_contribute(self):
        """A masked atom must be exactly absent, not merely small."""
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        coord, charge, cell = nacl(reps=2, a=2.0)

        ghost = torch.tensor(
            [[[1.1, 0.7, 0.3], [3.3, 2.9, 1.7]]], dtype=DT, device=DEVICE
        )
        padded_coord = torch.cat([coord, ghost], dim=1)
        padded_charge = torch.cat(
            [charge, torch.tensor([[5.0, -3.0]], dtype=DT, device=DEVICE)], dim=1
        )
        mask = torch.ones(padded_coord.shape[:2], dtype=torch.bool, device=DEVICE)
        mask[:, -2:] = False

        plain = term(coord, charge, cell).sum()
        masked = term(padded_coord, padded_charge, cell, mask=mask).sum()
        assert torch.allclose(plain, masked, atol=1e-9)

    def test_padding_atoms_receive_zero_energy(self):
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        coord, charge, cell = nacl(reps=2, a=2.0)
        mask = torch.ones(coord.shape[:2], dtype=torch.bool, device=DEVICE)
        mask[:, -1] = False
        energy = term(coord, charge, cell, mask=mask)
        assert torch.allclose(energy[0, -1], torch.zeros(1, dtype=DT, device=DEVICE))


class TestBatching:
    def test_batch_matches_individual_frames(self):
        """Frames in a batch must not leak into each other."""
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        c1, q1, cell1 = nacl(reps=2, a=2.0)
        c2, q2, cell2 = nacl(reps=2, a=2.0)
        c2 = c2 + 0.37

        batched = term(
            torch.cat([c1, c2]), torch.cat([q1, q2]), torch.cat([cell1, cell2])
        )
        separate = torch.cat([term(c1, q1, cell1), term(c2, q2, cell2)])
        assert torch.allclose(batched, separate, atol=1e-9)

    def test_identical_frames_give_identical_energies(self):
        """Cheap, sharp check that per-frame bookkeeping is right.

        This caught a real bug when the term still built a real-space
        neighbour list: nvalchemi's batched `naive` builder silently
        under-reported neighbours for every frame after the first. The
        reciprocal-only term has no neighbour list, so that failure mode is
        gone by construction -- but the invariant is cheap to keep.
        """
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        coord, charge, cell = nacl(reps=2, a=2.0)
        n = 3
        energy = term(
            coord.repeat(n, 1, 1), charge.repeat(n, 1), cell.repeat(n, 1, 1)
        )
        totals = energy.sum(dim=(1, 2))
        assert torch.allclose(totals, totals[0].expand(n), atol=1e-9)

    def test_per_atom_energies_have_the_right_shape(self):
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        coord, charge, cell = nacl(reps=2, a=2.0)
        energy = term(coord, charge, cell)
        assert energy.shape == (1, coord.shape[1], 1)
        assert torch.isfinite(energy).all()


class TestGradients:
    """Charges are trained through this term, so both paths must carry grad."""

    def test_gradcheck_positions_and_charges(self):
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(16, 16, 16))
        coord, charge, cell = nacl(reps=2, a=2.0)

        c = coord.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x: term(x, charge, cell).sum(), (c,), eps=1e-6, atol=1e-5
        )
        q = charge.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x: term(coord, x, cell).sum(), (q,), eps=1e-6, atol=1e-5
        )

    def test_double_backward_reaches_charges(self):
        """The joint training path: energy -> force -> loss -> d/dq."""
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(16, 16, 16))
        coord, charge, cell = nacl(reps=2, a=2.0)
        c = coord.clone().requires_grad_(True)
        q = charge.clone().requires_grad_(True)

        energy = term(c, q, cell).sum()
        force = -torch.autograd.grad(energy, c, create_graph=True)[0]
        grad_q = torch.autograd.grad((force**2).sum(), q)[0]
        assert torch.isfinite(grad_q).all()
        assert float(grad_q.abs().sum()) > 0.0

    def test_zero_charges_give_zero_energy(self):
        """The joint model reduces to the short-range one when charges vanish."""
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(16, 16, 16))
        coord, charge, cell = nacl(reps=2, a=2.0)
        energy = term(coord, torch.zeros_like(charge), cell)
        assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-12)


class TestChargeSignDegeneracy:
    """The global sign of the charges is unobservable, by construction.

    The energy is bilinear in charge, so every product q_i q_j is unchanged
    by q -> -q. For a neutral cell the two assignments are exactly
    degenerate, and no amount of energy or force data can distinguish them.
    A charge head trained only on energies and forces therefore recovers
    magnitudes and relative signs, and picks the overall sign arbitrarily --
    in the Phase 5 benchmark it settled on the inverted branch. Anyone
    comparing learned charges against a population analysis has to break the
    symmetry themselves.
    """

    def test_energy_is_invariant_under_global_charge_inversion(self):
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        coord, charge, cell = nacl(reps=3, a=2.0)
        assert torch.allclose(
            term(coord, charge, cell), term(coord, -charge, cell), atol=1e-12
        )

    def test_forces_are_invariant_under_global_charge_inversion(self):
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        coord, charge, cell = nacl(reps=3, a=2.0)

        def force(q):
            c = coord.clone().requires_grad_(True)
            return -torch.autograd.grad(term(c, q, cell).sum(), c)[0]

        assert torch.allclose(force(charge), force(-charge), atol=1e-10)


class TestMeshChoice:
    """A spacing-derived mesh is a step function of the cell.

    Smoothness is measured as the stability of a central difference under
    uniform scaling across two step sizes. Comparing energies directly would
    prove nothing: scaling the system genuinely changes its energy. A smooth
    energy gives the same derivative at both step sizes; a discontinuous one
    gives a derivative that explodes as the step shrinks onto the jump.
    """

    @staticmethod
    def _scaling_derivative(term, coord, charge, cell, eps):
        hi = float(term(coord * (1 + eps), charge, cell * (1 + eps)).sum())
        lo = float(term(coord * (1 - eps), charge, cell * (1 - eps)).sum())
        return (hi - lo) / (2 * eps)

    def test_spacing_derived_mesh_jumps_at_a_boundary(self):
        """Document the hazard rather than leave it to be rediscovered.

        It first appeared as a 2% virial error that looked like a bad strain
        derivative. Harmless at fixed cell (NVE/NVT); a real problem under
        NPT or any strain derivative.

        Uses random charges rather than a lattice: a lattice's reciprocal
        energy is near zero at a realistic alpha, and a derivative test on a
        quantity that is already zero measures nothing.
        """
        coord, charge, cell = random_neutral(n=32, box=12.0)
        term = LongRangeEnergy(alpha=0.5, mesh_spacing=1.0)  # 12.0 / 1.0 == 12
        coarse = self._scaling_derivative(term, coord, charge, cell, 1e-4)
        fine = self._scaling_derivative(term, coord, charge, cell, 1e-6)
        assert abs(coarse) > 1e-3, "derivative is ~0; the test would be vacuous"
        assert abs(fine - coarse) > 0.1 * abs(coarse)

    def test_fixed_mesh_is_smooth_across_that_boundary(self):
        coord, charge, cell = random_neutral(n=32, box=12.0)
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(12, 12, 12))
        coarse = self._scaling_derivative(term, coord, charge, cell, 1e-4)
        fine = self._scaling_derivative(term, coord, charge, cell, 1e-6)
        assert abs(coarse) > 1e-3, "derivative is ~0; the test would be vacuous"
        assert abs(fine - coarse) < 1e-3 * abs(coarse)


class TestSerialization:
    def test_round_trip(self):
        term = LongRangeEnergy(
            alpha=0.35, mesh_dimensions=(20, 20, 20), spline_order=5
        )
        restored = LongRangeEnergy.deserialize(term.serialize())
        assert restored.serialize() == term.serialize()
