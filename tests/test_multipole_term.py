# SPDX-License-Identifier: MIT
"""The long-range term with atomic dipoles and quadrupoles.

What is pinned:
- **The formula**, against a structure-factor sum written out in NumPy that
  shares no code with the term, for charges, dipoles and quadrupoles alone and
  together, in a triclinic cell.
- **The charge term is its special case**, number for number, in a neutral
  and in a charged cell.
- **Two distant dipoles** interact as point dipoles do.
- **The derivatives** autograd gives: positions, moments and cell against
  finite differences, and a second derivative exists (force training).
- **The mesh**: PME agrees with the explicit sum in a tilted cell, which
  nvalchemiops' own k-grid does not give, in energy, forces and cell
  gradient.
- **The isolated sum** (no periodic axis): against the same atoms written
  out as point charges, against a very large periodic cell, against the
  textbook point dipoles, with autograd's derivatives against finite
  differences and atoms on top of each other giving finite numbers.
- **The slab correction**: nvalchemiops' atom by atom when there are only
  charges, and with moments against the same atoms as point charges under
  the charge term's own correction.
- **What is refused**: a periodic term on a frame without a cell.
"""

import math

import numpy as np
import pytest
import torch
from deepmd.dpmodel.modifier.dipole_charge import ELECTROSTATIC_CONVERSION
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.lr.multipole import (
    IsolatedMultipoleEnergy,
    MultipoleLongRangeEnergy,
    build_term,
    isolated_multipole_term_for,
)
from deepmd_lr.lr.multipole_pairs import radial_coefficients
from deepmd_lr.lr.term import IsolatedLongRangeEnergy, LongRangeEnergy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the k-sum kernels run on GPU"
)

DT = torch.float64
KE = 14.3996453535
ALPHA = 0.4


def tensor(array) -> torch.Tensor:
    return torch.tensor(np.asarray(array), dtype=DT, device=DEVICE)


def traceless(q: np.ndarray) -> np.ndarray:
    q = 0.5 * (q + np.swapaxes(q, -1, -2))
    trace = np.trace(q, axis1=-2, axis2=-1)[..., None, None]
    return q - np.eye(3) * trace / 3.0


def system(n: int = 10, box: float = 9.0, seed: int = 0, net: float = 0.0):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * box + 0.3 * rng.normal(size=(3, 3))
    coord = rng.uniform(0.0, box, size=(n, 3))
    charge = rng.normal(size=n)
    charge += net / n - charge.mean()
    dipole = 0.5 * rng.normal(size=(n, 3))
    quadrupole = traceless(rng.normal(size=(n, 3, 3)))
    return coord, charge, dipole, quadrupole, cell


def reference(coord, charge, dipole, quadrupole, cell, alpha=ALPHA, k_max=6.0):
    """The docstring's formula, in eV."""
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    reach = int(np.ceil(k_max / np.abs(np.linalg.eigvals(reciprocal)).min())) + 2
    grid = np.arange(-reach, reach + 1)
    k = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    k = k @ reciprocal
    k2 = (k**2).sum(1)
    keep = (k2 > 1e-12) & (k2 < k_max**2)
    k, k2 = k[keep], k2[keep]
    amplitude = (
        charge[None]
        + 1j * (k @ dipole.T)
        - 0.5 * np.einsum("ka,nab,kb->kn", k, quadrupole, k)
    )
    structure = (amplitude * np.exp(1j * k @ coord.T)).sum(1)
    volume = abs(np.linalg.det(cell))
    recip = (
        2
        * np.pi
        / volume
        * (np.exp(-k2 / (4 * alpha**2)) / k2 * np.abs(structure) ** 2).sum()
    )
    root = math.sqrt(math.pi)
    own = (
        alpha / root * (charge**2).sum()
        + 2 * alpha**3 / (3 * root) * (dipole**2).sum()
        + 2 * alpha**5 / (5 * root) * (quadrupole**2).sum()
    )
    background = math.pi * charge.sum() ** 2 / (2 * alpha**2 * volume)
    return KE * (recip - own - background)


def term(**settings) -> MultipoleLongRangeEnergy:
    return MultipoleLongRangeEnergy(
        **{"alpha": ALPHA, "method": "ewald", "k_cutoff": 6.0, **settings}
    )


def energy(coord, charge, dipole, quadrupole, cell, **settings) -> torch.Tensor:
    return term(**settings)(
        tensor(coord)[None],
        tensor(charge)[None],
        tensor(cell)[None],
        dipole=None if dipole is None else tensor(dipole)[None],
        quadrupole=None if quadrupole is None else tensor(quadrupole)[None],
    )


class TestFormula:
    @pytest.mark.parametrize(
        "moments",
        ["charge", "dipole", "quadrupole", "charge+dipole", "all", "all, charged"],
    )
    def test_against_the_structure_factor_sum(self, moments):
        coord, charge, dipole, quadrupole, cell = system(
            net=1.5 if "charged" in moments else 0.0
        )
        if moments in ("dipole", "quadrupole"):
            charge = 0 * charge
        if moments in ("charge", "quadrupole"):
            dipole = 0 * dipole
        if moments in ("charge", "dipole", "charge+dipole"):
            quadrupole = 0 * quadrupole
        expected = reference(coord, charge, dipole, quadrupole, cell)
        got = float(energy(coord, charge, dipole, quadrupole, cell).sum())
        assert got == pytest.approx(expected, abs=1e-7)

    @pytest.mark.parametrize("net", [0.0, 2.0])
    def test_without_moments_it_is_the_charge_term(self, net):
        coord, charge, _, _, cell = system(net=net)
        plain = LongRangeEnergy(alpha=ALPHA, method="ewald", k_cutoff=6.0)(
            tensor(coord)[None], tensor(charge)[None], tensor(cell)[None]
        )
        ours = energy(coord, charge, None, None, cell)
        torch.testing.assert_close(ours, plain.to(DT), atol=1e-8, rtol=0)

    def test_zero_moments_are_no_moments(self):
        coord, charge, dipole, quadrupole, cell = system()
        torch.testing.assert_close(
            energy(coord, charge, 0 * dipole, 0 * quadrupole, cell),
            energy(coord, charge, None, None, cell),
            atol=1e-10,
            rtol=0,
        )

    def test_two_distant_dipoles_are_point_dipoles(self):
        """12 A apart, erf(alpha r) is 1 to 1e-11; the periodic images of the
        60 A cell add about a percent (measured: 0.7 %)."""
        box, r = 60.0, 12.0
        coord = np.array([[20.0, 30.0, 30.0], [20.0 + r, 30.0, 30.0]])
        mu = np.array([[0.3, 0.2, -0.1], [-0.2, 0.4, 0.25]])
        axis = np.array([1.0, 0.0, 0.0])
        point = KE * (mu[0] @ mu[1] - 3 * (mu[0] @ axis) * (mu[1] @ axis)) / r**3
        cell = np.eye(3) * box
        zero = np.zeros(2)
        pair = float(energy(coord, zero, mu, None, cell, k_cutoff=4.0).sum())
        alone = sum(
            float(energy(coord[i : i + 1], zero[:1], mu[i : i + 1], None, cell,
                         k_cutoff=4.0).sum())
            for i in range(2)
        )  # fmt: skip
        assert pair - alone == pytest.approx(point, rel=2e-2)


class TestDerivatives:
    def leaves(self):
        coord, charge, dipole, quadrupole, cell = system(n=6, box=8.0, seed=3)
        return [tensor(x).requires_grad_(True) for x in (coord, charge, dipole)] + [
            tensor(quadrupole),
            tensor(cell).requires_grad_(True),
        ]

    def total(self, coord, charge, dipole, quadrupole, cell) -> torch.Tensor:
        # A symmetrised, trace-free copy, so every component may be varied.
        sym = 0.5 * (quadrupole + quadrupole.transpose(-1, -2))
        eye = torch.eye(3, dtype=DT, device=DEVICE)
        free = sym - eye * sym.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] / 3
        return term()(
            coord[None], charge[None], cell[None], dipole=dipole[None],
            quadrupole=free[None],
        ).sum()  # fmt: skip

    @pytest.mark.parametrize("which", [0, 1, 2, 3, 4])
    def test_first_derivatives_against_finite_differences(self, which):
        leaves = self.leaves()
        leaves[3].requires_grad_(True)
        (analytic,) = torch.autograd.grad(self.total(*leaves), leaves[which])
        rng = np.random.default_rng(which)
        direction = tensor(rng.normal(size=tuple(leaves[which].shape)))
        h = 1e-4
        shifted = []
        for sign in (1.0, -1.0):
            moved = [x.detach().clone() for x in leaves]
            moved[which] = moved[which] + sign * h * direction
            shifted.append(float(self.total(*moved)))
        numeric = (shifted[0] - shifted[1]) / (2 * h)
        assert float((analytic * direction).sum()) == pytest.approx(
            numeric, rel=1e-5, abs=1e-7
        )

    def test_a_second_derivative_reaches_the_moments(self):
        coord, charge, dipole, quadrupole, cell = self.leaves()
        quadrupole.requires_grad_(True)
        value = self.total(coord, charge, dipole, quadrupole, cell)
        (force,) = torch.autograd.grad(value, coord, create_graph=True)
        grads = torch.autograd.grad((force**2).sum(), [charge, dipole, quadrupole])
        assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads)


class TestMesh:
    """PME against the explicit sum, in the tilted cell of ``system``."""

    MESH = {"method": "pme", "mesh_dimensions": [48, 48, 48], "k_cutoff": None}

    @pytest.mark.parametrize("moments", ["charge", "dipole", "all", "all, charged"])
    def test_energy_against_the_explicit_sum(self, moments):
        coord, charge, dipole, quadrupole, cell = system(
            net=1.5 if "charged" in moments else 0.0
        )
        if moments == "charge":
            dipole = quadrupole = None
        if moments == "dipole":
            charge, quadrupole = 0 * charge, None
        exact = energy(coord, charge, dipole, quadrupole, cell)
        mesh = energy(coord, charge, dipole, quadrupole, cell, **self.MESH)
        torch.testing.assert_close(mesh, exact, atol=2e-6, rtol=0)

    def test_without_moments_it_is_the_charge_mesh_term(self):
        coord, charge, _, _, cell = system()
        plain = LongRangeEnergy(alpha=ALPHA, mesh_dimensions=[48, 48, 48])(
            tensor(coord)[None], tensor(charge)[None], tensor(cell)[None]
        )
        ours = energy(coord, charge, None, None, cell, **self.MESH)
        torch.testing.assert_close(ours, plain.to(DT), atol=5e-6, rtol=0)

    def test_forces_and_cell_gradient_against_the_explicit_sum(self):
        coord, charge, dipole, quadrupole, cell = system()

        def gradients(**settings):
            position = tensor(coord).requires_grad_(True)
            lattice = tensor(cell).requires_grad_(True)
            total = term(**settings)(
                position[None], tensor(charge)[None], lattice[None],
                dipole=tensor(dipole)[None], quadrupole=tensor(quadrupole)[None],
            ).sum()  # fmt: skip
            return torch.autograd.grad(total, [position, lattice])

        exact, mesh = gradients(), gradients(**self.MESH)
        for a, b in zip(mesh, exact, strict=True):
            torch.testing.assert_close(a, b, atol=5e-5, rtol=0)

    def test_the_mesh_of_a_system_is_twice_as_fine(self):
        from deepmd_lr.lr.multipole import multipole_mesh
        from deepmd_lr.lr.term import system_mesh

        settings = {"alpha": ALPHA, "mesh_dimensions": [2, 2, 2]}
        cell = np.eye(3) * 12.4
        assert system_mesh(settings, cell) == (10, 10, 10)
        assert multipole_mesh(settings, cell) == (20, 20, 20)


class TestFramesAndMask:
    def test_frames_are_independent(self):
        a, b = system(seed=1), system(seed=2)
        both = term()(
            torch.stack([tensor(a[0]), tensor(b[0])]),
            torch.stack([tensor(a[1]), tensor(b[1])]),
            torch.stack([tensor(a[4]), tensor(b[4])]),
            dipole=torch.stack([tensor(a[2]), tensor(b[2])]),
            quadrupole=torch.stack([tensor(a[3]), tensor(b[3])]),
        )
        for i, frame in enumerate((a, b)):
            torch.testing.assert_close(
                both[i : i + 1], energy(*frame), atol=1e-9, rtol=0
            )

    def test_a_masked_atom_is_absent(self):
        coord, charge, dipole, quadrupole, cell = system()
        mask = torch.ones(1, len(charge), dtype=DT, device=DEVICE)
        mask[0, -1] = 0.0
        masked = term()(
            tensor(coord)[None], tensor(charge)[None], tensor(cell)[None], mask,
            dipole=tensor(dipole)[None], quadrupole=tensor(quadrupole)[None],
        )  # fmt: skip
        without = energy(coord[:-1], charge[:-1], dipole[:-1], quadrupole[:-1], cell)
        assert float(masked[0, -1]) == 0.0
        torch.testing.assert_close(masked[:, :-1], without, atol=1e-9, rtol=0)


def as_point_charges(coord, charge, dipole, quadrupole, d):
    """The same atoms as point charges: a dipole as a pair ``d`` apart, a
    quadrupole as three charges in a row along each of its principal axes.
    Right to ``d**2``. Returns positions, charges and each one's atom."""
    points, charges, owner = [], [], []
    for atom in range(len(coord)):
        points.append(coord[atom])
        charges.append(charge[atom])
        size = np.linalg.norm(dipole[atom])
        along = dipole[atom] / size
        points += [coord[atom] + along * d / 2, coord[atom] - along * d / 2]
        charges += [size / d, -size / d]
        values, axes = np.linalg.eigh(quadrupole[atom])
        for value, axis in zip(values, axes.T, strict=True):
            end = value / (2 * d * d)
            points += [coord[atom] + axis * d, coord[atom], coord[atom] - axis * d]
            charges += [end, -2 * end, end]
        owner += [atom] * 12
    return np.array(points), np.array(charges), np.array(owner)


def within_atoms(points, charges, owner, alpha=ALPHA) -> float:
    """What the point charges of one atom do to each other, in eV: a multipole
    has no such energy, its self term being subtracted."""
    distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
    kernel = np.full_like(distance, 2 * alpha / math.sqrt(math.pi))
    apart = distance > 1e-12
    kernel[apart] = np.vectorize(math.erf)(alpha * distance[apart]) / distance[apart]
    np.fill_diagonal(kernel, 0.0)
    same = owner[:, None] == owner[None]
    # The term's own constant: these are thousands of eV, taken off as many.
    pairs = (charges[:, None] * charges[None] * kernel * same).sum()
    return ELECTROSTATIC_CONVERSION * 0.5 * float(pairs)


def extrapolated(difference) -> float:
    """The d -> 0 limit of a quantity with a d**2 error, from d and d / 2."""
    coarse, fine = difference
    return (4 * fine - coarse) / 3


def isolated(coord, charge, dipole=None, quadrupole=None, **settings) -> torch.Tensor:
    return IsolatedMultipoleEnergy(alpha=ALPHA, **settings)(
        tensor(coord)[None],
        tensor(charge)[None],
        None,
        dipole=None if dipole is None else tensor(dipole)[None],
        quadrupole=None if quadrupole is None else tensor(quadrupole)[None],
    )


class TestCluster:
    def test_the_radial_coefficients(self):
        # Against autograd on erf(alpha r)/r, where that is well conditioned.
        r = tensor([1.2, 1.6, 2.4, 2.6, 5.0, 9.0]).requires_grad_(True)
        coefficients = radial_coefficients(r * r, ALPHA)
        value = torch.erf(ALPHA * r) / r
        for order in range(5):
            torch.testing.assert_close(coefficients[order], value, rtol=1e-9, atol=0)
            (slope,) = torch.autograd.grad(value.sum(), r, create_graph=True)
            value = -slope / r

    def test_without_moments_it_is_the_charge_sum(self):
        coord, charge, _, _, _ = system(seed=2, net=0.4)
        ours = isolated(coord, charge)
        theirs = IsolatedLongRangeEnergy(alpha=ALPHA, isolated_kernel="dense")(
            tensor(coord)[None], tensor(charge)[None]
        )
        torch.testing.assert_close(ours, theirs, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("moments", ["dipole", "both"])
    def test_against_the_same_atoms_as_point_charges(self, moments):
        coord, charge, dipole, quadrupole, _ = system(n=6, box=6.0, seed=5, net=0.3)
        if moments == "dipole":
            ours = float(isolated(coord, charge, dipole).sum())
            quadrupole = np.zeros_like(quadrupole)
        else:
            ours = float(isolated(coord, charge, dipole, quadrupole).sum())
        difference = []
        for d in (0.1, 0.05):
            points, charges, owner = as_point_charges(
                coord, charge, dipole, quadrupole, d
            )
            total = float(isolated(points, charges).sum())
            difference.append(ours - (total - within_atoms(points, charges, owner)))
        assert abs(difference[1]) < 0.3 * abs(difference[0])  # the d**2 error
        assert extrapolated(difference) == pytest.approx(0.0, abs=2e-6)

    def test_it_is_the_limit_of_a_large_cell(self):
        coord, charge, dipole, quadrupole, _ = system(n=6, box=6.0, seed=6)
        box = 60.0
        cluster = float(isolated(coord, charge, dipole, quadrupole).sum())
        periodic = float(
            energy(
                coord, charge, dipole, quadrupole, np.eye(3) * box,
                method="pme", mesh_dimensions=[144, 144, 144], k_cutoff=None,
            ).sum()
        )  # fmt: skip
        # A cubic lattice of the cell's dipole M lowers the energy by
        # 2 pi M^2 / 3V; what is left falls as L^-5 (and the mesh's own
        # error, 2e-4 eV at the default spacing with moments this large).
        moment = (charge[:, None] * coord).sum(0) + dipole.sum(0)
        images = KE * 2 * math.pi * float(moment @ moment) / (3 * box**3)
        assert abs(periodic - cluster) > 5e-3
        assert periodic + images == pytest.approx(cluster, abs=5e-5)

    def test_two_distant_dipoles_are_point_dipoles(self):
        first, second = np.array([0.3, -0.2, 0.5]), np.array([-0.4, 0.1, 0.2])
        separation = np.array([14.0, 3.0, -2.0])
        r = np.linalg.norm(separation)
        point = KE * (
            first @ second / r**3
            - 3 * (first @ separation) * (second @ separation) / r**5
        )
        ours = isolated(
            np.array([np.zeros(3), separation]), np.zeros(2), np.array([first, second])
        )
        assert float(ours.sum()) == pytest.approx(point, rel=1e-10)

    @pytest.mark.parametrize("which", [0, 1, 2, 3])
    def test_first_derivatives_against_finite_differences(self, which):
        coord, charge, dipole, quadrupole, _ = system(n=6, box=5.0, seed=7, net=0.2)
        leaves = [tensor(x) for x in (coord, charge, dipole, quadrupole)]
        term = IsolatedMultipoleEnergy(alpha=ALPHA)

        def total(coord, charge, dipole, quadrupole):
            sym = 0.5 * (quadrupole + quadrupole.transpose(-1, -2))
            return term(
                coord[None], charge[None], None, dipole=dipole[None],
                quadrupole=sym[None],
            ).sum()  # fmt: skip

        leaves[which].requires_grad_(True)
        (analytic,) = torch.autograd.grad(total(*leaves), leaves[which])
        direction = tensor(
            np.random.default_rng(which).normal(size=tuple(leaves[which].shape))
        )
        h = 1e-4
        shifted = []
        for sign in (1.0, -1.0):
            moved = [x.detach().clone() for x in leaves]
            moved[which] = moved[which] + sign * h * direction
            shifted.append(float(total(*moved)))
        assert float((analytic * direction).sum()) == pytest.approx(
            (shifted[0] - shifted[1]) / (2 * h), rel=1e-6, abs=1e-8
        )

    def test_it_turns_with_the_frame(self):
        coord, charge, dipole, quadrupole, _ = system(n=6, box=5.0, seed=8)
        turn, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
        before = isolated(coord, charge, dipole, quadrupole)
        after = isolated(
            coord @ turn.T, charge, dipole @ turn.T, turn @ quadrupole @ turn.T
        )
        torch.testing.assert_close(before, after, atol=1e-11, rtol=0)

    def test_atoms_on_top_of_each_other_stay_finite(self):
        coord, charge, dipole, quadrupole, _ = system(n=4, box=5.0, seed=9)
        coord[1] = coord[0]
        coord[3] = coord[2] + 1e-9
        position = tensor(coord).requires_grad_(True)
        moment = tensor(dipole).requires_grad_(True)
        total = IsolatedMultipoleEnergy(alpha=ALPHA)(
            position[None], tensor(charge)[None], None, dipole=moment[None],
            quadrupole=tensor(quadrupole)[None],
        ).sum()  # fmt: skip
        (force,) = torch.autograd.grad(total, position, create_graph=True)
        (second,) = torch.autograd.grad(force.square().sum(), moment)
        for value in (total, force, second):
            assert bool(torch.isfinite(value).all())

    def test_frames_and_a_mask(self):
        coord, charge, dipole, quadrupole, _ = system(n=5, box=5.0, seed=10)
        other = system(n=5, box=5.0, seed=11)
        term = IsolatedMultipoleEnergy(alpha=ALPHA)
        stack = lambda a, b: torch.stack([tensor(a), tensor(b)])  # noqa: E731
        mask = torch.ones(2, 5, dtype=torch.bool, device=DEVICE)
        mask[0, -1] = False
        both = term(
            stack(coord, other[0]), stack(charge, other[1]), None, mask=mask,
            dipole=stack(dipole, other[2]), quadrupole=stack(quadrupole, other[3]),
        )  # fmt: skip
        alone = isolated(coord[:-1], charge[:-1], dipole[:-1], quadrupole[:-1])
        torch.testing.assert_close(both[:1, :-1], alone, atol=1e-11, rtol=0)
        assert float(both[0, -1].abs()) == 0.0
        torch.testing.assert_close(both[1:], isolated(*other[:4]), atol=1e-11, rtol=0)

    def test_a_term_for_a_frame_without_a_cell(self):
        periodic = term(alpha=0.37)
        cluster = isolated_multipole_term_for(periodic)
        assert isinstance(cluster, IsolatedMultipoleEnergy)
        assert cluster.alpha == 0.37
        assert isolated_multipole_term_for(cluster) is cluster
        built = build_term(cluster.serialize(), multipoles=True)
        assert isinstance(built, IsolatedMultipoleEnergy)

    def test_too_many_atoms_are_refused(self):
        coord, charge, dipole, _, _ = system()
        with pytest.raises(ValueError, match="max_isolated_atoms"):
            isolated(coord, charge, dipole, max_isolated_atoms=5)


SLAB = {"slab_correction": True, "pbc": [True, True, False]}


def slab(seed: int = 12, net: float = 0.0):
    """Atoms in the lower third of a tilted cell with 20 A along its third
    vector."""
    rng = np.random.default_rng(seed)
    cell = np.array([[8.0, 0.0, 0.0], [1.5, 7.5, 0.0], [0.7, -0.9, 20.0]])
    n = 6
    coord = (rng.uniform(size=(n, 3)) * [1.0, 1.0, 0.3]) @ cell
    charge = rng.normal(scale=0.5, size=n)
    charge += net / n - charge.mean()
    dipole = 0.3 * rng.normal(size=(n, 3))
    quadrupole = traceless(0.3 * rng.normal(size=(n, 3, 3)))
    return coord, charge, dipole, quadrupole, cell


class TestSlab:
    @pytest.mark.parametrize("net", [0.0, 0.4])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_without_moments_it_is_the_charge_correction(self, net, axis):
        coord, charge, _, _, cell = slab(net=net)
        # The open axis moved to each place in turn.
        order = np.roll(np.arange(3), axis + 1)
        cell = cell[order]
        pbc = [k != int(np.argmax(order == 2)) for k in range(3)]
        settings = {"slab_correction": True, "pbc": pbc, "k_cutoff": 4.0}
        ours = energy(coord, charge, None, None, cell, **settings)
        theirs = LongRangeEnergy(alpha=ALPHA, method="ewald", **settings)(
            tensor(coord)[None], tensor(charge)[None], tensor(cell)[None]
        )
        plain = energy(coord, charge, None, None, cell, k_cutoff=4.0)
        assert float((ours - plain).abs().max()) > 1e-3
        torch.testing.assert_close(ours, theirs, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("net", [0.0, 0.3])
    def test_against_the_same_atoms_as_point_charges(self, net):
        coord, charge, dipole, quadrupole, cell = slab(net=net)
        settings = {**SLAB, "k_cutoff": 5.0}
        ours = float(energy(coord, charge, dipole, quadrupole, cell, **settings).sum())
        system_ = (coord, charge, dipole, quadrupole, cell)
        bulk = float(energy(*system_, k_cutoff=5.0).sum())
        assert abs(ours - bulk) > 1e-2
        charges_only = LongRangeEnergy(alpha=ALPHA, method="ewald", **settings)
        difference = []
        for d in (0.2, 0.1):
            points, charges, owner = as_point_charges(
                coord, charge, dipole, quadrupole, d
            )
            total = float(
                charges_only(
                    tensor(points)[None], tensor(charges)[None], tensor(cell)[None]
                ).sum()
            )
            difference.append(ours - (total - within_atoms(points, charges, owner)))
        assert abs(difference[1]) < 0.3 * abs(difference[0])
        assert extrapolated(difference) == pytest.approx(0.0, abs=2e-5)

    def test_derivatives_against_finite_differences(self):
        coord, charge, dipole, quadrupole, cell = slab(net=0.2)
        leaves = [tensor(x) for x in (coord, charge, dipole, cell)]
        slab_term = term(**SLAB, k_cutoff=4.0)

        def total(coord, charge, dipole, cell):
            return slab_term(
                coord[None], charge[None], cell[None], dipole=dipole[None],
                quadrupole=tensor(quadrupole)[None],
            ).sum()  # fmt: skip

        for which in range(4):
            moved = [x.clone().requires_grad_(k == which) for k, x in enumerate(leaves)]
            (analytic,) = torch.autograd.grad(total(*moved), moved[which])
            direction = tensor(
                np.random.default_rng(which).normal(size=tuple(leaves[which].shape))
            )
            h = 1e-4
            shifted = []
            for sign in (1.0, -1.0):
                moved = [x.clone() for x in leaves]
                moved[which] = moved[which] + sign * h * direction
                shifted.append(float(total(*moved)))
            assert float((analytic * direction).sum()) == pytest.approx(
                (shifted[0] - shifted[1]) / (2 * h), rel=1e-5, abs=1e-7
            ), which

    def test_on_the_mesh(self):
        coord, charge, dipole, quadrupole, cell = slab()
        explicit = energy(coord, charge, dipole, quadrupole, cell, **SLAB)
        mesh = energy(
            coord, charge, dipole, quadrupole, cell, method="pme", k_cutoff=None,
            mesh_dimensions=[32, 32, 64], **SLAB,
        )  # fmt: skip
        torch.testing.assert_close(mesh, explicit, atol=2e-5, rtol=0)

    def test_a_slab_needs_its_correction(self):
        with pytest.raises(ValueError, match="slab_correction"):
            term(pbc=[True, True, False])


class TestRefusals:
    def test_a_frame_without_a_cell(self):
        coord, charge, dipole, _, _ = system()
        with pytest.raises(ValueError, match="isolated_multipole_term_for"):
            term()(tensor(coord)[None], tensor(charge)[None], None,
                   dipole=tensor(dipole)[None])  # fmt: skip

    def test_a_quadrupole_without_a_dipole(self):
        coord, charge, _, quadrupole, cell = system()
        with pytest.raises(ValueError, match="dipole"):
            energy(coord, charge, None, quadrupole, cell)
