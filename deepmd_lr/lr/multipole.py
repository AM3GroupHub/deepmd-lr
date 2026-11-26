# SPDX-License-Identifier: MIT
r"""The long-range term with a dipole and a quadrupole on every atom.

The same smooth term as :class:`~deepmd_lr.lr.term.LongRangeEnergy`, with more
moments. That term is the periodic energy of Gaussian charges, each of a width
that makes two of them interact as :math:`\mathrm{erf}(\alpha r)/r`, minus
their self-energy. Here every atom's Gaussian also carries a dipole
:math:`\mu_i` and a traceless quadrupole :math:`Q_i`:

.. math::
    E = \frac{2\pi k_e}{V}\sum_{k\neq 0}
            \frac{e^{-k^2/4\alpha^2}}{k^2}\,|S(k)|^2 - E_\text{self}
        - E_\text{background},
    \qquad
    S(k) = \sum_i \Bigl(q_i + i\,k\cdot\mu_i
            - \tfrac12\,k\cdot Q_i\cdot k\Bigr) e^{i k\cdot r_i}

with :math:`E_\text{self} = k_e\bigl(\tfrac{\alpha}{\sqrt\pi}\sum q_i^2
+ \tfrac{2\alpha^3}{3\sqrt\pi}\sum|\mu_i|^2
+ \tfrac{2\alpha^5}{5\sqrt\pi}\sum|Q_i|_F^2\bigr)` and the uniform background
of a charged cell, :math:`E_\text{background} = k_e\,\pi Q_\text{tot}^2 /
(2\alpha^2 V)`, as the charge term has it. The quadrupole is the traceless
second moment, its energy in a potential :math:`\tfrac12 Q:\nabla\nabla\phi`.
With no dipole and no quadrupole this is the charge term, number for number.

Two ways to the sum, as for charges. ``method: "ewald"`` runs over explicit
k-vectors, exact to its cutoff, at the cost of atoms times k-vectors.
``method: "pme"`` is nvalchemiops' multipole PME on a mesh. That routine
builds its own k-grid with the cell's inverse transposed the wrong way round
(0.5.0), which a symmetric cell does not notice and a tilted one pays 3e-3 eV
for on 12 atoms, at any mesh; the grid is therefore built here
(:func:`mesh_k_squared`) and handed in, after which it agrees with the
explicit sum to 1e-9 eV for charges on a 64^3 mesh.

A dipole is spread on the mesh with the derivative of the spline and a
quadrupole with its second derivative, each an order less smooth, so the
moments need a finer mesh than the charges for the same error: measured at
alpha 0.4 and spline order 6, a spacing of ``1/(2 alpha)`` leaves 4e-3 eV/A in
the dipoles' forces and ``1/(4 alpha)`` leaves 5e-5. The mesh a system gets
follows the second (:func:`multipole_mesh`).

Periodic in three directions only: nvalchemiops has no slab correction that
knows of atomic dipoles and no isolated multipole sum, so both are refused.
"""

import math
from typing import (
    Any,
)

import torch
from deepmd.dpmodel.modifier.dipole_charge import (
    ELECTROSTATIC_CONVERSION,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_pme_reciprocal_space,
)

from deepmd_lr.lr.term import (
    DEFAULT_ALPHA,
    LongRangeEnergy,
    _check_inputs,
    build_long_range_energy,
    system_mesh,
)


def build_term(settings: dict[str, Any], multipoles: bool) -> torch.nn.Module:
    """The long-range term of ``settings``: with ``multipoles`` the one that
    takes dipoles and quadrupoles, else what
    :func:`~deepmd_lr.lr.term.build_long_range_energy` builds."""
    if multipoles:
        return MultipoleLongRangeEnergy(**settings)
    return build_long_range_energy(settings)


def mesh_k_squared(cell: torch.Tensor, mesh: tuple[int, int, int]) -> torch.Tensor:
    """``|k|^2`` on the half-complex FFT grid of ``mesh``, ``(nx, ny, nz // 2
    + 1)``, attached to ``cell`` (rows are lattice vectors): ``k = 2 pi m
    inv(cell)^T``."""
    axes = [
        torch.fft.fftfreq(n, d=1.0 / n, device=cell.device, dtype=cell.dtype)
        for n in mesh[:2]
    ]
    axes.append(
        torch.fft.rfftfreq(
            mesh[2], d=1.0 / mesh[2], device=cell.device, dtype=cell.dtype
        )
    )
    miller = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    k = 2.0 * math.pi * (miller @ torch.linalg.inv(cell).transpose(-1, -2))
    return (k * k).sum(-1)


def multipole_mesh(settings: dict[str, Any], cell: Any) -> tuple[int, int, int]:
    """The PME mesh a cell needs with dipoles and quadrupoles on it: at
    least ``ceil(4 alpha L)`` points along each cell vector, twice what
    charges need, and never fewer than ``settings`` asks for."""
    double = {**settings, "alpha": 2.0 * float(settings.get("alpha", DEFAULT_ALPHA))}
    return system_mesh(double, cell)


#: nvalchemiops splits the Gaussian width between the density basis and its
#: Ewald factor, and the sum depends on their combination alone; this share of
#: the width goes to the basis.
_BASIS_SHARE = 0.5


class MultipoleLongRangeEnergy(LongRangeEnergy):
    """Per-atom long-range energy of charges, dipoles and quadrupoles, in eV.

    The settings of :class:`~deepmd_lr.lr.term.LongRangeEnergy`; a slab
    correction and an open axis are refused (module docstring).
    """

    def __init__(self, **settings: Any) -> None:
        if settings.get("method", "pme") not in ("pme", "ewald"):
            raise NotImplementedError(
                "dipoles and quadrupoles need a cell periodic in all three "
                f"directions: there is no {settings['method']!r} multipole sum yet"
            )
        super().__init__(**settings)
        if self.slab_correction or not all(self.pbc):
            raise NotImplementedError(
                "dipoles and quadrupoles need a cell periodic in all three "
                "directions: there is no slab correction with atomic dipoles "
                "and no isolated multipole sum yet"
            )

    def forward(
        self,
        coord: torch.Tensor,
        charge: torch.Tensor,
        box: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        dipole: torch.Tensor | None = None,
        quadrupole: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-atom long-range electrostatic energy, in eV.

        Parameters
        ----------
        coord, charge, box, mask
            As for :class:`~deepmd_lr.lr.term.LongRangeEnergy`.
        dipole
            Atomic dipoles in e angstrom, ``(nf, nloc, 3)``; None for none.
        quadrupole
            Atomic quadrupoles in e angstrom^2, symmetric and traceless,
            ``(nf, nloc, 3, 3)``; None for none. Needs ``dipole``.

        Returns
        -------
        torch.Tensor
            Per-atom energy in eV, shape ``(nf, nloc, 1)``, float64.
        """
        if box is None:
            raise NotImplementedError(
                "dipoles and quadrupoles need a cell: there is no isolated "
                "multipole sum yet"
            )
        _check_inputs(coord, charge)
        nf, nloc = charge.shape
        if quadrupole is not None and dipole is None:
            raise ValueError("a quadrupole needs a dipole beside it")
        if dipole is not None and dipole.shape != (nf, nloc, 3):
            raise ValueError(
                f"dipole must have shape {(nf, nloc, 3)}, got {tuple(dipole.shape)}"
            )
        if quadrupole is not None and quadrupole.shape != (nf, nloc, 3, 3):
            raise ValueError(
                f"quadrupole must have shape {(nf, nloc, 3, 3)}, "
                f"got {tuple(quadrupole.shape)}"
            )
        cell = box.reshape(nf, 3, 3)
        if mask is not None:
            weight = mask.to(charge.dtype)
            charge = charge * weight
            if dipole is not None:
                dipole = dipole * weight[..., None]
            if quadrupole is not None:
                quadrupole = quadrupole * weight[..., None, None]
        width = 1.0 / (2.0 * self.alpha)
        sigma = _BASIS_SHARE * width
        split = 1.0 / (2.0 * math.sqrt(width**2 - sigma**2))
        root = math.sqrt(math.pi)
        energies = []
        for frame in range(nf):
            q = charge[frame].to(torch.float64)
            mu = None if dipole is None else dipole[frame].to(torch.float64)
            theta = None if quadrupole is None else quadrupole[frame].to(torch.float64)
            moments = pack_multipole_moments(q, mu, theta)
            volume = torch.abs(torch.linalg.det(cell[frame])).to(torch.float64)
            # The charged cell's background at this term's alpha, per atom.
            background = math.pi / (2.0 * self.alpha**2) * q * q.sum() / volume
            if self.method == "pme":
                mesh = self._mesh(cell[frame : frame + 1])
                # Minus the self terms already, and minus a background taken
                # at the routine's own share of the width: that one is put
                # back and this term's subtracted.
                energy = multipole_pme_reciprocal_space(
                    coord[frame],
                    moments,
                    cell[frame],
                    sigma=sigma,
                    alpha=split,
                    mesh_dimensions=mesh,
                    spline_order=self.spline_order,
                    k_squared=mesh_k_squared(cell[frame].to(torch.float64), mesh),
                )
                theirs = background * (self.alpha / split) ** 2
                energies.append(
                    energy + ELECTROSTATIC_CONVERSION * (theirs - background)
                )
                continue
            reciprocal = multipole_reciprocal_space_energy(
                coord[frame],
                moments,
                cell[frame],
                sigma=sigma,
                alpha=split,
                k_cutoff=self.k_cutoff,
            )
            own = self.alpha / root * q * q
            if mu is not None:
                own = own + (2.0 * self.alpha**3 / (3.0 * root)) * (mu * mu).sum(-1)
            if theta is not None:
                own = own + (2.0 * self.alpha**5 / (5.0 * root)) * (theta * theta).sum(
                    (-2, -1)
                )
            energies.append(reciprocal - ELECTROSTATIC_CONVERSION * (own + background))
        return torch.stack(energies).reshape(nf, nloc, 1)
