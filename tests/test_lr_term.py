"""Phase 3: the long-range (reciprocal-space) electrostatic term.

`tests/test_pme_physics.py` validates the nvalchemiops engine itself. This file
validates *our wrapper*: that it computes the reciprocal half of the Ewald
decomposition in DeePMD-kit's own convention and units, that the boundary
conditions Ewald is invalid under are rejected rather than silently computed,
and that gradients reach both the coordinates and the charges.

The oracle is `reciprocal_reference` below: the textbook Ewald reciprocal
formula written out directly, with no library behind it. Matching it is the
whole specification -- the right half of the split, the right units, and the
self term subtracted rather than left in.

Nothing here compares against DeePMD-kit's DPLR. This package does not use
DPLR: its charges are fixed per element and its Wannier centroids come from a
separately trained dipole model, where ours are learned from the descriptor.
Its `ewald_reciprocal_energy` reports the *bare* k-space sum, self term
included, so matching it used to require adding a spurious self-interaction
back in.
"""

import itertools
import math

import numpy as np
import pytest
import torch
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.lr import MAX_SPLINE_ORDER, LongRangeEnergy
from deepmd_lr.lr.term import ELECTROSTATIC_CONVERSION as KE

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
    coord = torch.tensor(rng.uniform(0.0, box, size=(1, n, 3)), dtype=DT, device=DEVICE)
    charge = torch.tensor(rng.normal(size=(1, n)), dtype=DT, device=DEVICE)
    charge = charge - charge.mean()
    cell = (torch.eye(3, dtype=DT, device=DEVICE) * box).unsqueeze(0)
    return coord, charge, cell


def _method_kwargs(method: str) -> dict:
    """Backend-specific settings that make the two methods comparable."""
    if method == "pme":
        return {"method": "pme", "mesh_dimensions": FINE_MESH}
    return {"method": "ewald"}


def reciprocal_reference(coord, charge, cell, alpha, n_max: int = 14):
    r"""The Ewald reciprocal energy in eV, written out from the formula.

    .. math::
        E = \frac{2\pi}{V}\sum_{k \neq 0}
                \frac{e^{-k^2/4\alpha^2}}{k^2}\left|S(k)\right|^2
          - \frac{\alpha}{\sqrt{\pi}}\sum_i q_i^2

    with :math:`S(k) = \sum_i q_i e^{i k \cdot r_i}`. Deliberately naive --
    an explicit triple loop over Miller indices in NumPy -- so that it shares
    no code with the thing under test. ``n_max`` truncates the k-sum; 14 is
    converged to ~1e-6 eV for the cells used here.
    """
    lattice = cell[0].detach().cpu().numpy()
    positions = coord[0].detach().cpu().numpy()
    charges = charge[0].detach().cpu().numpy()
    volume = abs(np.linalg.det(lattice))
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T

    total = 0.0
    for miller in itertools.product(range(-n_max, n_max + 1), repeat=3):
        if miller == (0, 0, 0):
            continue
        k = np.asarray(miller) @ reciprocal
        k_squared = float(k @ k)
        structure = np.sum(charges * np.exp(1j * (positions @ k)))
        total += math.exp(-k_squared / (4 * alpha**2)) / k_squared * abs(
            structure
        ) ** 2
    recip = (2 * np.pi / volume) * total
    self_term = (alpha / math.sqrt(math.pi)) * float(np.sum(charges**2))
    return (recip - self_term) * KE


class TestMatchesTheEwaldFormula:
    """The specification: reproduce the reciprocal sum, self term subtracted.

    Pins three things at once -- that only the reciprocal half is computed,
    that the Gaussian self-interaction is removed, and that the eV conversion
    is right.
    """

    @pytest.mark.parametrize("alpha", [0.3, 0.4, 0.6])
    @pytest.mark.parametrize("method", ["pme", "ewald"])
    def test_matches_the_reciprocal_formula(self, alpha, method):
        coord, charge, cell = random_neutral()
        term = LongRangeEnergy(alpha=alpha, **_method_kwargs(method))
        ours = float(term(coord, charge, cell).sum())
        reference = reciprocal_reference(coord, charge, cell, alpha)
        assert abs(ours - reference) < 1e-4 * abs(reference)

    def test_the_self_term_is_subtracted_not_added(self):
        """The sign of the self term, pinned explicitly.

        Adding it instead of subtracting is a silent error: the energy stays
        smooth, differentiable and plausible, and on the LiFSI frame it
        inflated the reported long-range energy from 105.8 to 208.8 eV.
        """
        coord, charge, cell = random_neutral()
        alpha = 0.4
        term = LongRangeEnergy(alpha=alpha, mesh_dimensions=FINE_MESH)
        ours = float(term(coord, charge, cell).sum())
        self_term = (
            alpha
            / math.sqrt(math.pi)
            * float((charge[0] ** 2).sum().detach().cpu())
            * KE
        )
        bare_k_space = ours + self_term
        reference = reciprocal_reference(coord, charge, cell, alpha)
        assert abs(ours - reference) < 1e-4 * abs(reference)
        # And the other convention is genuinely different, not a rounding
        # difference -- so the test above discriminates between them.
        assert abs(bare_k_space - reference) > 0.1 * abs(reference)

    def test_matches_for_an_ionic_lattice(self):
        """A crystal, where the structure factor is sharp rather than broad.

        alpha is 1.0 here. A rocksalt lattice orders charge on a 2 A period,
        so its structure factor lives entirely at short wavelength; at
        alpha = 0.5 the Gaussian exp(-k^2/4 alpha^2) suppresses the k-space
        part to ~1e-9 eV and the energy would be the self term alone, which
        proves nothing about the sum. alpha = 1.0 is where the k-space part
        carries real weight.
        """
        coord, charge, cell = nacl()
        term = LongRangeEnergy(alpha=1.0, mesh_dimensions=FINE_MESH)
        ours = float(term(coord, charge, cell).sum())
        reference = reciprocal_reference(coord, charge, cell, 1.0)
        assert abs(reference) > 1.0, "reference is ~0; the test would be vacuous"
        assert abs(ours - reference) < 1e-4 * abs(reference)

    def test_reciprocal_term_is_small_for_short_wavelength_order(self):
        """The physical corollary, worth stating explicitly.

        The long-range term only carries weight where the charge density has
        long-wavelength structure. A tightly alternating lattice at a
        realistic alpha has almost none, so almost all of its electrostatics
        is short-ranged and the network handles it. This is why alpha is a
        modelling choice and not a free parameter.
        """
        coord, charge, cell = nacl()
        alpha = 0.5
        term = LongRangeEnergy(alpha=alpha, mesh_dimensions=FINE_MESH)
        ours = float(term(coord, charge, cell).sum())
        # Our term is `recip - self`. The claim is about the k != 0 structure,
        # so add the on-site self term back to isolate it -- this is NOT the
        # model's convention, just algebra to get at the k-sum.
        self_term = (
            alpha
            / math.sqrt(math.pi)
            * float((charge[0] ** 2).sum().detach().cpu())
            * KE
        )
        k_space = ours + self_term
        assert abs(k_space) < 1e-6, (
            f"k-space part is {k_space:.3e} eV, expected ~0 for a tightly "
            "alternating lattice"
        )
        # And the term itself is dominated by the on-site piece here, which
        # is exactly why `recip` alone is not a meaningful "long-range
        # energy" for such a system.
        assert abs(ours + self_term) < 1e-6 * abs(self_term)

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
        a = float(
            LongRangeEnergy(alpha=0.3, mesh_dimensions=FINE_MESH)(
                coord, charge, cell
            ).sum()
        )
        b = float(
            LongRangeEnergy(alpha=0.6, mesh_dimensions=FINE_MESH)(
                coord, charge, cell
            ).sum()
        )
        assert abs(a - b) > 0.1 * abs(a)

    def test_needs_no_neighbor_list(self):
        """Dropping the real-space half removes the neighbour list entirely,
        so no cutoff can exceed the cell and no capacity can overflow. This
        cell is far smaller than any sensible real-space cutoff."""
        coord, charge, cell = nacl(reps=2, a=2.0)  # L = 4 A
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        assert torch.isfinite(term(coord, charge, cell)).all()


class TestMethodChoice:
    """`pme` and `ewald` evaluate the same sum by different routes."""

    @pytest.mark.parametrize("alpha", [0.3, 0.5])
    def test_the_two_methods_agree(self, alpha):
        """The explicit sum is what makes a mesh setting checkable.

        Ewald has no interpolation error at all, so a disagreement here is a
        statement about the PME mesh, not about the physics.
        """
        coord, charge, cell = random_neutral()
        pme = float(
            LongRangeEnergy(alpha=alpha, mesh_dimensions=FINE_MESH)(
                coord, charge, cell
            ).sum()
        )
        ewald = float(
            LongRangeEnergy(alpha=alpha, method="ewald")(coord, charge, cell).sum()
        )
        assert abs(pme - ewald) < 1e-5 * abs(ewald)

    def test_ewald_is_the_more_accurate_of_the_two(self):
        """On a coarse mesh, PME drifts and Ewald does not.

        Confirms the explicit sum really is mesh-independent, which is the
        only reason to reach for it.
        """
        coord, charge, cell = random_neutral()
        reference = reciprocal_reference(coord, charge, cell, 0.4)
        coarse = float(
            LongRangeEnergy(alpha=0.4, mesh_dimensions=(8, 8, 8))(
                coord, charge, cell
            ).sum()
        )
        ewald = float(
            LongRangeEnergy(alpha=0.4, method="ewald")(coord, charge, cell).sum()
        )
        assert abs(ewald - reference) < abs(coarse - reference)

    def test_k_cutoff_is_derived_from_alpha(self):
        """2 alpha sqrt(-ln(accuracy)), the standard reciprocal cutoff."""
        term = LongRangeEnergy(alpha=0.4, method="ewald")
        expected = 2.0 * 0.4 * math.sqrt(-math.log(1e-6))
        assert term.k_cutoff == pytest.approx(expected)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            LongRangeEnergy(method="fmm")

    @pytest.mark.parametrize(
        "kwargs",
        [{"mesh_dimensions": (8, 8, 8)}, {"mesh_spacing": 1.0}, {"spline_order": 4}],
    )
    def test_rejects_pme_parameters_under_ewald(self, kwargs):
        """A mesh setting silently ignored under `ewald` would be
        indistinguishable from one that was applied."""
        with pytest.raises(ValueError, match="only apply to method='pme'"):
            LongRangeEnergy(method="ewald", **kwargs)

    @pytest.mark.parametrize("kwargs", [{"k_cutoff": 3.0}, {"accuracy": 1e-8}])
    def test_rejects_ewald_parameters_under_pme(self, kwargs):
        with pytest.raises(ValueError, match="only apply to method='ewald'"):
            LongRangeEnergy(method="pme", **kwargs)

    def test_ewald_carries_gradients(self):
        """The alternative backend has to be trainable too, not just correct."""
        coord, charge, cell = nacl(reps=2, a=2.0)
        term = LongRangeEnergy(alpha=0.5, method="ewald")
        c = coord.clone().requires_grad_(True)
        q = charge.clone().requires_grad_(True)
        energy = term(c, q, cell).sum()
        force = -torch.autograd.grad(energy, c, create_graph=True)[0]
        grad_q = torch.autograd.grad((force**2).sum(), q)[0]
        assert torch.isfinite(force).all() and torch.isfinite(grad_q).all()
        assert float(grad_q.abs().sum()) > 0.0

    def test_ewald_cell_gradient_matches_finite_difference(self):
        """Reciprocal vectors must deform with the cell for a physical virial."""
        coord, charge, cell = random_neutral(n=8, box=12.0)
        term = LongRangeEnergy(alpha=0.4, method="ewald")

        eps = torch.zeros((1, 3, 3), dtype=DT, device=DEVICE, requires_grad=True)
        strain = torch.eye(3, dtype=DT, device=DEVICE).unsqueeze(0) + eps
        energy = term(
            coord @ strain.transpose(-1, -2),
            charge,
            cell @ strain.transpose(-1, -2),
        ).sum()
        analytic = torch.autograd.grad(energy, eps)[0][0, 0, 0]

        h = 1.0e-5
        delta = torch.zeros((1, 3, 3), dtype=DT, device=DEVICE)
        delta[0, 0, 0] = h
        numeric = []
        eye = torch.eye(3, dtype=DT, device=DEVICE).unsqueeze(0)
        for sign in (1.0, -1.0):
            deform = eye + sign * delta
            numeric.append(
                term(
                    coord @ deform.transpose(-1, -2),
                    charge,
                    cell @ deform.transpose(-1, -2),
                ).sum()
            )
        finite_difference = (numeric[0] - numeric[1]) / (2.0 * h)

        assert torch.allclose(analytic, finite_difference, atol=2.0e-5, rtol=2.0e-4)

    def test_ewald_batches_correctly(self):
        term = LongRangeEnergy(alpha=0.5, method="ewald")
        coord, charge, cell = nacl(reps=2, a=2.0)
        n = 3
        totals = term(
            coord.repeat(n, 1, 1), charge.repeat(n, 1), cell.repeat(n, 1, 1)
        ).sum(dim=(1, 2))
        assert torch.allclose(totals, totals[0].expand(n), atol=1e-9)

    def test_ewald_round_trips(self):
        term = LongRangeEnergy(alpha=0.35, method="ewald", k_cutoff=3.0)
        restored = LongRangeEnergy.deserialize(term.serialize())
        assert restored.serialize() == term.serialize()
        assert restored.method == "ewald"


class TestSlabCorrection:
    """Ewald assumes 3D periodicity; a slab in a padded box does not have it.

    Without the Yeh-Berkowitz term a dipolar slab keeps interacting with its
    images through the vacuum, and the error shrinks only slowly with vacuum
    thickness -- so "add more vacuum" is not a fix. The correction is what
    makes a 2D-periodic system tractable at all.
    """

    @staticmethod
    def _dipolar_slab(box_z: float, n: int = 64, box_xy: float = 12.0):
        """Two oppositely charged sheets: a strong dipole across z."""
        rng = np.random.default_rng(0)
        xy = rng.uniform(0.0, box_xy, size=(n, 2))
        z = np.where(np.arange(n) < n // 2, 3.0, 7.0)
        q = np.where(np.arange(n) < n // 2, 1.0, -1.0).astype(float)
        coord = torch.tensor(
            np.column_stack([xy, z]), dtype=DT, device=DEVICE
        ).unsqueeze(0)
        charge = torch.tensor(q, dtype=DT, device=DEVICE).unsqueeze(0)
        cell = torch.tensor(
            np.diag([box_xy, box_xy, box_z]), dtype=DT, device=DEVICE
        ).unsqueeze(0)
        return coord, charge, cell

    def _energy(self, box_z, **kwargs):
        coord, charge, cell = self._dipolar_slab(box_z)
        term = LongRangeEnergy(alpha=0.4, mesh_spacing=0.8, **kwargs)
        return float(term(coord, charge, cell).sum())

    def test_uncorrected_slab_energy_drifts_with_vacuum(self):
        """The artefact being corrected. Not converged even at 160 A."""
        near = self._energy(40.0)
        far = self._energy(160.0)
        assert abs(far - near) > 0.05 * abs(near)

    def test_corrected_slab_energy_is_vacuum_independent(self):
        """The 2D-periodic answer, recovered without a huge box."""
        kwargs = {"slab_correction": True, "pbc": [True, True, False]}
        near = self._energy(40.0, **kwargs)
        far = self._energy(160.0, **kwargs)
        assert abs(far - near) < 1e-4 * abs(near)

    def test_correction_changes_the_answer(self):
        """Guard against a silently inactive correction."""
        plain = self._energy(40.0)
        corrected = self._energy(
            40.0, slab_correction=True, pbc=[True, True, False]
        )
        assert abs(corrected - plain) > 0.01 * abs(plain)

    def test_requires_exactly_one_open_axis(self):
        with pytest.raises(ValueError, match="exactly one non-periodic axis"):
            LongRangeEnergy(slab_correction=True, pbc=[True, True, True])
        with pytest.raises(ValueError, match="exactly one non-periodic axis"):
            LongRangeEnergy(slab_correction=True, pbc=[True, False, False])

    def test_open_axis_without_the_correction_is_refused(self):
        """Silently running a plain 3D sum on a slab is the trap."""
        with pytest.raises(ValueError, match="slab_correction: true"):
            LongRangeEnergy(pbc=[True, True, False])

    def test_round_trips(self):
        term = LongRangeEnergy(
            alpha=0.4, mesh_spacing=0.8,
            slab_correction=True, pbc=[True, True, False],
        )
        restored = LongRangeEnergy.deserialize(term.serialize())
        assert restored.serialize() == term.serialize()
        assert restored.slab_correction and restored.pbc == (True, True, False)

    def test_gradients_flow_through_the_correction(self):
        coord, charge, cell = self._dipolar_slab(40.0)
        term = LongRangeEnergy(
            alpha=0.4, mesh_spacing=0.8,
            slab_correction=True, pbc=[True, True, False],
        )
        c = coord.clone().requires_grad_(True)
        q = charge.clone().requires_grad_(True)
        energy = term(c, q, cell).sum()
        force = -torch.autograd.grad(energy, c, create_graph=True)[0]
        grad_q = torch.autograd.grad((force**2).sum(), q)[0]
        assert torch.isfinite(force).all() and torch.isfinite(grad_q).all()
        assert float(grad_q.abs().sum()) > 0.0


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

    @pytest.mark.parametrize(
        "kwargs", [{"alpha": 0.0}, {"alpha": -1.0}, {"mesh_spacing": 0.0}]
    )
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
        neighbour list: nvalchemiops' batched `naive` builder silently
        under-reported neighbours for every frame after the first. The
        reciprocal-only term has no neighbour list, so that failure mode is
        gone by construction -- but the invariant is cheap to keep.
        """
        term = LongRangeEnergy(alpha=0.5, mesh_dimensions=(32, 32, 32))
        coord, charge, cell = nacl(reps=2, a=2.0)
        n = 3
        energy = term(coord.repeat(n, 1, 1), charge.repeat(n, 1), cell.repeat(n, 1, 1))
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
        term = LongRangeEnergy(alpha=0.35, mesh_dimensions=(20, 20, 20), spline_order=5)
        restored = LongRangeEnergy.deserialize(term.serialize())
        assert restored.serialize() == term.serialize()


class TestSystemSettings:
    """How the sum is taken follows the system, not the checkpoint."""

    WATER = {
        "alpha": 0.4,
        "method": "pme",
        "slab_correction": False,
        "pbc": [True, True, True],
        "mesh_spacing": None,
        "mesh_dimensions": (10, 10, 10),
        "spline_order": 6,
    }

    def test_the_mesh_grows_with_the_cell_and_never_shrinks(self):
        from deepmd_lr.lr import system_mesh

        # The water model's mesh is right for its training box and kept there.
        assert system_mesh(self.WATER, np.eye(3) * 12.429) == (10, 10, 10)
        # A 4x4x4 supercell needs 1.25 A spacing: 40, not the stored 10.
        assert system_mesh(self.WATER, np.eye(3) * 49.716) == (40, 40, 40)
        # A mesh finer than the rule is never coarsened.
        fine = dict(self.WATER, mesh_dimensions=(30, 30, 30))
        assert system_mesh(fine, np.eye(3) * 12.429) == (30, 30, 30)

    def test_a_spacing_is_honoured_and_example_03s_mesh_is_reproduced(self):
        from deepmd_lr.lr import system_mesh

        settings = {"alpha": 0.4, "mesh_spacing": 1.25}
        assert system_mesh(settings, np.diag([14.04, 14.04, 52.0])) == (12, 12, 42)
        finer = {"alpha": 0.4, "mesh_spacing": 0.5}
        assert system_mesh(finer, np.eye(3) * 10.0) == (20, 20, 20)

    def test_pbc_picks_the_3d_or_the_slab_sum(self):
        from deepmd_lr.lr import system_reciprocal

        cell = np.diag([12.429, 12.429, 40.0])
        slab = system_reciprocal(self.WATER, cell, [True, True, False])
        assert slab["slab_correction"] is True
        assert slab["pbc"] == [True, True, False]
        assert slab["mesh_dimensions"] == [10, 10, 32]
        bulk = system_reciprocal(slab, np.eye(3) * 12.429, [True, True, True])
        assert bulk["slab_correction"] is False
        assert bulk["pbc"] == [True, True, True]
        # The rest is the model's.
        assert bulk["alpha"] == 0.4 and bulk["spline_order"] == 6

    def test_two_open_axes_are_refused_and_the_isolated_term_is_left_alone(self):
        from deepmd_lr.lr import system_reciprocal

        with pytest.raises(ValueError, match="2 axes open"):
            system_reciprocal(self.WATER, np.eye(3) * 10.0, [True, False, False])
        isolated = {"method": "isolated", "alpha": 0.4}
        assert system_reciprocal(isolated, np.eye(3) * 10.0, [True] * 3) == isolated
