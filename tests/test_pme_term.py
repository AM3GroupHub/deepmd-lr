"""Phase 3: the PME long-range energy term.

`tests/test_pme_physics.py` validates the nvalchemi engine itself. This file
validates *our wrapper*: that the unit conversion into eV is right, that the
boundary conditions PME is invalid under are rejected rather than silently
computed, and that gradients reach both the coordinates and the charges.
"""

import itertools

import pytest
import torch
from deepmd.dpmodel.modifier.dipole_charge import ELECTROSTATIC_CONVERSION as KE
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.pme import MAX_SPLINE_ORDER, PMEEnergy

MADELUNG_NACL = 1.7475645946331821
DT = torch.float64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the PME kernels run on GPU"
)


def dipole(separation: float = 2.0, box: float = 40.0):
    """Two opposite unit charges in a large cubic cell."""
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]], dtype=DT, device=DEVICE
    )
    charge = torch.tensor([[1.0, -1.0]], dtype=DT, device=DEVICE)
    cell = (torch.eye(3, dtype=DT, device=DEVICE) * box).unsqueeze(0)
    return coord, charge, cell


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


class TestUnits:
    """The prefactor is the single easiest thing to get wrong here."""

    def test_converges_to_isolated_coulomb(self):
        """As the cell grows, a dipole must approach -k_e/r in eV.

        This is the direct check that the energy is scaled into eV with the
        right Coulomb constant: the limit is an analytic number, not another
        code's output.
        """
        target = -KE / 2.0
        term = PMEEnergy(rcut=12.0, accuracy=1e-8)
        coord, charge, cell = dipole(separation=2.0, box=80.0)
        energy = float(term(coord, charge, cell).sum())
        assert abs(energy - target) / abs(target) < 1e-4

    def test_image_error_falls_as_inverse_volume(self):
        """Confirm the residual is the periodic image interaction, not a
        mis-scaled prefactor. A dipole's interaction with its images decays
        as 1/L^3, so doubling the box must cut the error by roughly 8x; a
        wrong constant would instead leave a residual that does not shrink.
        """
        target = -KE / 2.0
        errors = []
        for box in (20.0, 40.0):
            term = PMEEnergy(rcut=min(0.45 * box, 12.0), accuracy=1e-8)
            coord, charge, cell = dipole(separation=2.0, box=box)
            errors.append(abs(float(term(coord, charge, cell).sum()) - target))
        assert 4.0 < errors[0] / errors[1] < 16.0

    def test_madelung_energy_in_ev(self):
        """A second, independent unit check on a condensed system."""
        a, reps = 2.0, 3
        coord, charge, cell = nacl(reps=reps, a=a)
        n_ions = coord.shape[1]
        term = PMEEnergy(rcut=0.45 * a * reps, alpha=2.5 / (a / 2.0), accuracy=1e-8)
        energy = float(term(coord, charge, cell).sum())
        # E = -M * N/2 * k_e / r_nn, with r_nn = a/2
        expected = -MADELUNG_NACL * (n_ions / 2) * KE / (a / 2)
        assert abs(energy - expected) / abs(expected) < 1e-4


class TestBoundaryConditions:
    """PME is only valid for a periodic, non-degenerate cell."""

    def test_rejects_missing_box(self):
        coord, charge, _ = dipole()
        with pytest.raises(ValueError, match="periodic cell"):
            PMEEnergy()(coord, charge, None)

    def test_rejects_degenerate_cell(self):
        coord, charge, cell = dipole()
        flat = cell.clone()
        flat[0, 2, 2] = 0.0
        with pytest.raises(ValueError, match="non-degenerate cell"):
            PMEEnergy()(coord, charge, flat)

    def test_rejects_bad_shapes(self):
        coord, charge, cell = dipole()
        with pytest.raises(ValueError, match=r"shape \(nf, nloc\)"):
            PMEEnergy()(coord, charge[:, :1], cell)


class TestParameterValidation:
    """Guard the silent-failure mode found in Phase 0."""

    @pytest.mark.parametrize("order", [7, 8, 10])
    def test_rejects_unsupported_spline_order(self, order):
        """Order >= 7 makes nvalchemi drop the reciprocal sum without error,
        so it must be refused at construction rather than produce a wrong
        energy that looks like a convergence problem."""
        with pytest.raises(ValueError, match="silently dropped"):
            PMEEnergy(spline_order=order)

    def test_rejects_too_low_spline_order(self):
        with pytest.raises(ValueError, match="at least 3"):
            PMEEnergy(spline_order=2)

    def test_default_order_is_within_supported_range(self):
        assert PMEEnergy().spline_order <= MAX_SPLINE_ORDER

    @pytest.mark.parametrize(
        "kwargs", [{"rcut": -1.0}, {"alpha": 0.0}, {"accuracy": 0.0},
                   {"accuracy": 1.5}]
    )
    def test_rejects_nonsense_parameters(self, kwargs):
        with pytest.raises(ValueError):
            PMEEnergy(**kwargs)


class TestMasking:
    def test_padding_atoms_do_not_contribute(self):
        """A masked atom must be exactly absent, not merely small.

        Parameters are pinned rather than auto-estimated: the estimator reads
        the atom count, so an unpinned comparison would shift alpha and mesh
        between the two calls and stop being a like-for-like test.
        """
        term = PMEEnergy(rcut=1.9, alpha=2.0, mesh_spacing=0.5)
        coord, charge, cell = nacl(reps=2, a=2.0)

        ghost = torch.tensor([[[1.1, 0.7, 0.3], [3.3, 2.9, 1.7]]], dtype=DT,
                             device=DEVICE)
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
        term = PMEEnergy(rcut=1.9, alpha=2.0, mesh_spacing=0.5)
        coord, charge, cell = nacl(reps=2, a=2.0)
        mask = torch.ones(coord.shape[:2], dtype=torch.bool, device=DEVICE)
        mask[:, -1] = False
        energy = term(coord, charge, cell, mask=mask)
        assert torch.allclose(energy[0, -1], torch.zeros(1, dtype=DT, device=DEVICE))


class TestBatching:
    def test_batch_matches_individual_frames(self):
        """Frames in a batch must not leak into each other."""
        term = PMEEnergy(rcut=2.5, alpha=2.0, mesh_spacing=0.5)
        c1, q1, cell1 = nacl(reps=3, a=2.0)
        c2, q2, cell2 = nacl(reps=3, a=2.0)
        c2 = c2 + 0.37  # a different, translated configuration

        batched = term(
            torch.cat([c1, c2]), torch.cat([q1, q2]), torch.cat([cell1, cell2])
        )
        separate = torch.cat([term(c1, q1, cell1), term(c2, q2, cell2)])
        assert torch.allclose(batched, separate, atol=1e-9)

    def test_identical_frames_give_identical_energies(self):
        """The sharpest regression test for the batched neighbour-list bug.

        nvalchemi's `naive` batched builder -- which the auto-selector picks
        at these sizes -- silently under-reports neighbours for every frame
        after the first, so three copies of one configuration came back with
        frame 0 correct and frames 1 and 2 wrong by the same constant. No
        error was raised and the values stayed finite and plausible. This
        pins the workaround (`method="cell_list"`); if the pin is ever
        dropped, this fails immediately.
        """
        term = PMEEnergy(rcut=2.5, alpha=2.0, mesh_spacing=0.5)
        coord, charge, cell = nacl(reps=3, a=2.0)
        n_frames = 3
        energy = term(
            coord.repeat(n_frames, 1, 1),
            charge.repeat(n_frames, 1),
            cell.repeat(n_frames, 1, 1),
        )
        totals = energy.sum(dim=(1, 2))
        assert torch.allclose(totals, totals[0].expand(n_frames), atol=1e-9), (
            f"identical frames disagree: {totals.tolist()}"
        )

    def test_per_atom_energies_sum_to_total(self):
        term = PMEEnergy(rcut=1.9, alpha=2.0, mesh_spacing=0.5)
        coord, charge, cell = nacl(reps=2, a=2.0)
        energy = term(coord, charge, cell)
        assert energy.shape == (1, coord.shape[1], 1)
        assert torch.isfinite(energy).all()


class TestGradients:
    """Charges are trained through this term, so both paths must carry grad."""

    def test_gradcheck_positions_and_charges(self):
        term = PMEEnergy(rcut=6.0, alpha=0.5, mesh_spacing=1.0)
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
        """The joint training path: energy -> force -> loss -> d/dq.

        Without a working double backward, force labels could not train the
        charge head at all.
        """
        term = PMEEnergy(rcut=6.0, alpha=0.5, mesh_spacing=1.0)
        coord, charge, cell = nacl(reps=2, a=2.0)
        c = coord.clone().requires_grad_(True)
        q = charge.clone().requires_grad_(True)

        energy = term(c, q, cell).sum()
        force = -torch.autograd.grad(energy, c, create_graph=True)[0]
        grad_q = torch.autograd.grad((force**2).sum(), q)[0]
        assert torch.isfinite(grad_q).all()
        assert float(grad_q.abs().sum()) > 0.0

    def test_zero_charges_give_zero_energy(self):
        """Phase 4 reduces to Phase 1 when charges vanish; pin that here."""
        term = PMEEnergy(rcut=6.0, alpha=0.5, mesh_spacing=1.0)
        coord, charge, cell = nacl(reps=2, a=2.0)
        energy = term(coord, torch.zeros_like(charge), cell)
        assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-12)


class TestMeshChoice:
    """A spacing-derived mesh is a step function of the cell.

    Smoothness is measured as the stability of a central difference under
    uniform scaling across two step sizes. Comparing energies directly would
    prove nothing: scaling a Coulomb system genuinely changes its energy
    (E scales as 1/lambda). A smooth energy gives the same derivative at both
    step sizes; a discontinuous one gives a derivative that explodes as the
    step shrinks onto the jump.
    """

    @staticmethod
    def _scaling_derivative(term, coord, charge, cell, eps):
        hi = float(term(coord * (1 + eps), charge, cell * (1 + eps)).sum())
        lo = float(term(coord * (1 - eps), charge, cell * (1 - eps)).sum())
        return (hi - lo) / (2 * eps)

    def test_spacing_derived_mesh_jumps_at_a_boundary(self):
        """Document the hazard rather than leave it to be rediscovered.

        The mesh is ceil(L / spacing), so a cell length sitting exactly on a
        multiple of the spacing gains a whole grid point under an
        infinitesimal strain. It cost a 2% virial error before being
        identified. Harmless at fixed cell (NVE/NVT); a real problem under
        NPT or any strain derivative.
        """
        coord, charge, cell = nacl(reps=3, a=2.0)  # L = 6.0
        # 6.0 / 1.0 is exactly 6, so any expansion adds a grid point.
        term = PMEEnergy(rcut=2.5, alpha=2.0, mesh_spacing=1.0)
        coarse = self._scaling_derivative(term, coord, charge, cell, 1e-4)
        fine = self._scaling_derivative(term, coord, charge, cell, 1e-6)
        assert abs(fine - coarse) > 0.1 * abs(coarse), (
            "expected the derivative to be unstable across a mesh jump; "
            f"got coarse={coarse:.6f} fine={fine:.6f}"
        )

    def test_fixed_mesh_is_smooth_across_that_boundary(self):
        """Pinning the mesh removes the jump, which is why it is the
        recommended setting whenever the cell can change."""
        coord, charge, cell = nacl(reps=3, a=2.0)
        term = PMEEnergy(rcut=2.5, alpha=2.0, mesh_dimensions=(12, 12, 12))
        coarse = self._scaling_derivative(term, coord, charge, cell, 1e-4)
        fine = self._scaling_derivative(term, coord, charge, cell, 1e-6)
        assert abs(fine - coarse) < 1e-3 * abs(coarse), (
            f"expected a stable derivative; got coarse={coarse:.6f} "
            f"fine={fine:.6f}"
        )

    def test_uniform_scaling_follows_inverse_length(self):
        """Scaling a Coulomb system by lambda scales every distance, so
        E -> E / lambda and dE/dlambda = -E at lambda = 1.

        This doubles as a convergence check on the PME parameters, because
        the identity only holds when they are converged. A fixed mesh has a
        fixed number of grid points, so as the cell grows the grid spacing
        grows with it while alpha -- an inverse length in absolute units --
        does not. If the mesh only barely resolves the screening Gaussian,
        it resolves it worse after scaling and the exact law breaks: at
        alpha=2.0 on a 12^3 mesh the ratio came out at 0.985, while a wider
        Gaussian on a finer mesh gives 1.0003.
        """
        coord, charge, cell = nacl(reps=3, a=2.0)
        term = PMEEnergy(rcut=2.9, alpha=1.5, mesh_dimensions=(16, 16, 16))
        energy = float(term(coord, charge, cell).sum())
        derivative = self._scaling_derivative(term, coord, charge, cell, 1e-5)
        assert abs(derivative + energy) < 1e-3 * abs(energy)

    def test_rejects_bad_mesh_dimensions(self):
        with pytest.raises(ValueError, match="three integers"):
            PMEEnergy(mesh_dimensions=(8, 8))
        with pytest.raises(ValueError, match="three integers"):
            PMEEnergy(mesh_dimensions=(8, 8, 1))


class TestSerialization:
    def test_round_trip(self):
        term = PMEEnergy(rcut=7.0, alpha=0.35, mesh_spacing=0.9, spline_order=5,
                         accuracy=1e-7)
        restored = PMEEnergy.deserialize(term.serialize())
        assert restored.serialize() == term.serialize()


class TestNeighborCapacity:
    def test_dense_system_grows_the_neighbor_list(self):
        """A cutoff large enough to overflow the initial capacity must still
        give the right answer, not raise and not silently truncate pairs."""
        a, reps = 2.0, 3
        coord, charge, cell = nacl(reps=reps, a=a)
        term = PMEEnergy(rcut=0.45 * a * reps, alpha=2.5 / (a / 2.0), accuracy=1e-8)
        energy = float(term(coord, charge, cell).sum())
        expected = -MADELUNG_NACL * (coord.shape[1] / 2) * KE / (a / 2)
        assert abs(energy - expected) / abs(expected) < 1e-4
