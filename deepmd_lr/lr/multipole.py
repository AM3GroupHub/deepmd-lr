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

A slab (one open axis) takes the Yeh-Berkowitz correction on top of the 3D
sum, as the charge term does. nvalchemiops' knows of charges only, so it is
written here (:func:`slab_multipole_correction`): what the correction needs of
the cell's content is its first and second moment along the slab normal, and
an atom's dipole and quadrupole add to those. A system with no periodic axis
is summed pair by pair (:class:`IsolatedMultipoleEnergy`,
:mod:`deepmd_lr.lr.multipole_pairs`).
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

from deepmd_lr.lr.multipole_pairs import (
    isolated_multipole_energy,
)
from deepmd_lr.lr.term import (
    DEFAULT_ALPHA,
    IsolatedLongRangeEnergy,
    LongRangeEnergy,
    _check_inputs,
    build_long_range_energy,
    isolated_term_for,
    system_mesh,
)


def build_term(settings: dict[str, Any], multipoles: bool) -> torch.nn.Module:
    """The long-range term of ``settings``: with ``multipoles`` the one that
    takes dipoles and quadrupoles, else what
    :func:`~deepmd_lr.lr.term.build_long_range_energy` builds."""
    if not multipoles:
        return build_long_range_energy(settings)
    if settings.get("method", "pme") == IsolatedMultipoleEnergy.METHOD:
        return IsolatedMultipoleEnergy(**settings)
    return MultipoleLongRangeEnergy(**settings)


def isolated_multipole_term_for(term: torch.nn.Module) -> "IsolatedMultipoleEnergy":
    """The isolated multipole term at ``term``'s alpha: how a frame with no
    cell is summed, whatever method the periodic frames use."""
    if isinstance(term, IsolatedMultipoleEnergy):
        return term
    return IsolatedMultipoleEnergy(**isolated_term_for(term).serialize())


def _checked_moments(
    charge: torch.Tensor,
    dipole: torch.Tensor | None,
    quadrupole: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """The three moments, their shapes checked, padding atoms zeroed and the
    quadrupole's trace taken out.

    The trace of a second moment is not a quadrupole: on a Gaussian it only
    changes the width. The multipole head has none; one handed in from
    outside is dropped, by every sum alike.
    """
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
    if quadrupole is not None:
        trace = quadrupole.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
        eye = torch.eye(3, dtype=quadrupole.dtype, device=quadrupole.device)
        quadrupole = quadrupole - trace[..., None, None] * eye
    if mask is not None:
        weight = mask.to(charge.dtype)
        charge = charge * weight
        if dipole is not None:
            dipole = dipole * weight[..., None]
        if quadrupole is not None:
            quadrupole = quadrupole * weight[..., None, None]
    return charge, dipole, quadrupole


def slab_multipole_correction(
    coord: torch.Tensor,
    charge: torch.Tensor,
    cell: torch.Tensor,
    pbc: tuple[bool, bool, bool],
    dipole: torch.Tensor | None = None,
    quadrupole: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""The Yeh-Berkowitz slab correction with atomic dipoles and
    quadrupoles, per atom in e^2/angstrom, ``(nf, nloc)`` float64.

    With :math:`z` the coordinate along the unit normal :math:`n` of the
    periodic plane, :math:`L` the cell's height along it and
    :math:`Q_\text{tot}` the cell's charge,

    .. math::
        E = \frac{2\pi}{V}\Bigl(M^2 - Q_\text{tot} M_2
            - \frac{Q_\text{tot}^2 L^2}{12}\Bigr),
        \quad M = \sum_i q_i z_i + \mu_i\cdot n,
        \quad M_2 = \sum_i q_i z_i^2 + 2 z_i\,\mu_i\cdot n + n\cdot Q_i\cdot n

    the first and second moment of the charge density along the normal. The
    charged-cell terms are those of nvalchemiops' charge-only correction
    (Ballenegger et al. 2009), which this reproduces atom by atom when no
    atom has a dipole. ``cell`` is ``(nf, 3, 3)``, rows the cell vectors, and
    ``pbc`` has exactly one open axis.
    """
    open_axis = [k for k in range(3) if not pbc[k]]
    if len(open_axis) != 1:
        raise ValueError(f"a slab has exactly one open axis, got pbc={pbc}")
    k = open_axis[0]
    cell = cell.to(torch.float64)
    coord = coord.to(torch.float64)
    q = charge.to(torch.float64)
    normal = torch.linalg.cross(cell[:, (k + 1) % 3], cell[:, (k + 2) % 3])
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)
    height = (cell[:, k] * normal).sum(-1, keepdim=True)
    volume = torch.abs(torch.linalg.det(cell))[:, None]
    z = (coord * normal[:, None, :]).sum(-1)
    first = q * z
    second = q * z * z
    if dipole is not None:
        along = (dipole.to(torch.float64) * normal[:, None, :]).sum(-1)
        first = first + along
        second = second + 2.0 * z * along
    if quadrupole is not None:
        second = second + torch.einsum(
            "fa,fiab,fb->fi", normal, quadrupole.to(torch.float64), normal
        )
    total = q.sum(-1, keepdim=True)
    energy = (
        first * first.sum(-1, keepdim=True)
        - 0.5 * (q * second.sum(-1, keepdim=True) + total * second)
        - q * total * height * height / 12.0
    )
    return (2.0 * math.pi / volume) * energy


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

    The settings of :class:`~deepmd_lr.lr.term.LongRangeEnergy`, the slab
    correction among them (:func:`slab_multipole_correction`).
    """

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
            raise ValueError(
                "The periodic long-range term needs a cell, but box is None; a "
                "frame with no cell is summed by isolated_multipole_term_for(term)."
            )
        _check_inputs(coord, charge)
        nf, nloc = charge.shape
        charge, dipole, quadrupole = _checked_moments(charge, dipole, quadrupole, mask)
        cell = box.reshape(nf, 3, 3)
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
        energy = torch.stack(energies).reshape(nf, nloc)
        if self.slab_correction:
            energy = energy + ELECTROSTATIC_CONVERSION * slab_multipole_correction(
                coord, charge, cell, self.pbc, dipole, quadrupole
            )
        return energy.reshape(nf, nloc, 1)


class IsolatedMultipoleEnergy(IsolatedLongRangeEnergy):
    """The isolated sum of charges, dipoles and quadrupoles, per atom in eV:
    :class:`~deepmd_lr.lr.term.IsolatedLongRangeEnergy` with more moments
    (:mod:`deepmd_lr.lr.multipole_pairs`).

    One dense kernel in float64, whatever ``isolated_kernel`` and
    ``kernel_dtype`` say: there is no chunked multipole sum, and a call of
    more than ``max_isolated_atoms`` atoms is refused.
    """

    def forward(
        self,
        coord: torch.Tensor,
        charge: torch.Tensor,
        box: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        dipole: torch.Tensor | None = None,
        quadrupole: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-atom energy in eV, ``(nf, nloc, 1)`` float64; arguments as
        :meth:`MultipoleLongRangeEnergy.forward`, ``box`` ignored."""
        del box
        _check_inputs(coord, charge)
        charge, dipole, quadrupole = _checked_moments(charge, dipole, quadrupole, mask)
        if charge.numel() > self.max_isolated_atoms:
            raise ValueError(
                f"{charge.numel()} atoms exceed max_isolated_atoms = "
                f"{self.max_isolated_atoms}: the isolated multipole sum has a "
                "dense O(N^2) kernel only. Raise max_isolated_atoms if the "
                "memory is there."
            )
        energy = isolated_multipole_energy(
            coord, charge, self.alpha, dipole, quadrupole
        )
        return (energy * ELECTROSTATIC_CONVERSION).unsqueeze(-1)
