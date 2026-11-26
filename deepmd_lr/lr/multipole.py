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

The k-sum is nvalchemiops' explicit one, exact to its cutoff. Its multipole
PME is not used: on a 12-atom cell with a 48^3 mesh it missed the explicit sum
by 4e-3 eV, where the charge PME misses by 1e-6. The cost is that of an Ewald
sum, atoms times k-vectors, which is small for training cells and grows as
the square of the size in MD.

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

from deepmd_lr.lr.term import (
    LongRangeEnergy,
    _check_inputs,
)

#: nvalchemiops splits the Gaussian width between the density basis and its
#: Ewald factor, and the sum depends on their combination alone; this share of
#: the width goes to the basis.
_BASIS_SHARE = 0.5


class MultipoleLongRangeEnergy(LongRangeEnergy):
    """Per-atom long-range energy of charges, dipoles and quadrupoles, in eV.

    The settings of :class:`~deepmd_lr.lr.term.LongRangeEnergy` with
    ``method="ewald"``; a mesh, a slab correction and an open axis are
    refused (module docstring).
    """

    def __init__(self, **settings: Any) -> None:
        if settings.get("method", "pme") != "ewald":
            raise ValueError(
                "a long-range term with dipoles or quadrupoles is summed over "
                'explicit k-vectors: set reciprocal.method to "ewald" (with '
                "k_cutoff or accuracy, and no mesh). The multipole PME of "
                "nvalchemiops is not accurate enough to stand in for it."
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
        energies = []
        for frame in range(nf):
            q = charge[frame].to(torch.float64)
            mu = None if dipole is None else dipole[frame].to(torch.float64)
            theta = None if quadrupole is None else quadrupole[frame].to(torch.float64)
            reciprocal = multipole_reciprocal_space_energy(
                coord[frame],
                pack_multipole_moments(q, mu, theta),
                cell[frame],
                sigma=sigma,
                alpha=split,
                k_cutoff=self.k_cutoff,
            )
            volume = torch.abs(torch.linalg.det(cell[frame])).to(torch.float64)
            own = self.alpha / math.sqrt(math.pi) * q * q
            if mu is not None:
                own = own + (2.0 * self.alpha**3 / (3.0 * math.sqrt(math.pi))) * (
                    mu * mu
                ).sum(-1)
            if theta is not None:
                own = own + (2.0 * self.alpha**5 / (5.0 * math.sqrt(math.pi))) * (
                    theta * theta
                ).sum((-2, -1))
            background = math.pi / (2.0 * self.alpha**2) * q * q.sum() / volume
            energies.append(reciprocal - ELECTROSTATIC_CONVERSION * (own + background))
        return torch.stack(energies).reshape(nf, nloc, 1)
