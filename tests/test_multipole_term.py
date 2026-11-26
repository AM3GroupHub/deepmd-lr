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
- **What is refused**: a mesh, a slab, a frame without a cell.
"""

import math

import numpy as np
import pytest
import torch
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.lr.multipole import MultipoleLongRangeEnergy
from deepmd_lr.lr.term import LongRangeEnergy

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


class TestRefusals:
    def test_a_mesh(self):
        with pytest.raises(ValueError, match="ewald"):
            MultipoleLongRangeEnergy(alpha=ALPHA, method="pme")

    def test_a_slab(self):
        with pytest.raises(NotImplementedError, match="three"):
            term(slab_correction=True, pbc=[True, True, False])

    def test_a_frame_without_a_cell(self):
        coord, charge, dipole, _, _ = system()
        with pytest.raises(NotImplementedError, match="isolated"):
            term()(tensor(coord)[None], tensor(charge)[None], None,
                   dipole=tensor(dipole)[None])  # fmt: skip

    def test_a_quadrupole_without_a_dipole(self):
        coord, charge, _, quadrupole, cell = system()
        with pytest.raises(ValueError, match="dipole"):
            energy(coord, charge, None, quadrupole, cell)
