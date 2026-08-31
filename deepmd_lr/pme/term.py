# SPDX-License-Identifier: LGPL-3.0-or-later
r"""The long-range (reciprocal-space) electrostatic energy, evaluated by PME.

What this computes, and what it deliberately does not
-----------------------------------------------------
Ewald splits the Coulomb energy into a screened real-space part, a smooth
reciprocal-space part, and a self term:

.. math::
    E = \underbrace{\sum_{i<j}
            \frac{q_i q_j \operatorname{erfc}(\alpha r_{ij})}{r_{ij}}
        }_{\text{short-ranged}}
      + \underbrace{\frac{2\pi}{V}\sum_{k \neq 0}
            \frac{e^{-k^2/4\alpha^2}}{k^2}|S(k)|^2
        }_{\text{long-ranged}}
      - \frac{\alpha}{\sqrt{\pi}}\sum_i q_i^2

This module supplies **only the reciprocal sum**. The real-space part decays
like ``erfc(alpha r)/r`` and is already inside the descriptor's cutoff, so the
short-range model represents it directly -- adding it here would ask the
network to learn to subtract something it can already describe.

That is the same decomposition DeePMD-kit's own DPLR uses, and this term is
checked against its ``ewald_reciprocal_energy`` (no real-space term, no self
subtraction) as the reference. nvalchemi's ``pme_reciprocal_space`` returns
the k-space sum *minus* the self term, so the self term is added back here to
land on DeePMD-kit's convention.

Because there is no real-space part, this module needs **no neighbour list**:
no cutoff, no capacity to size, no ``rcut < L/2`` constraint.

Choosing alpha
--------------
In a full Ewald sum ``alpha`` is arbitrary -- the total does not depend on it.
Here it is a real modelling choice, because it sets where the division of
labour falls: the network handles ``erfc(alpha r)/r`` and this term handles
the rest. ``alpha`` must be large enough that what is left to the network has
died inside the descriptor cutoff. At ``rcut = 6`` angstrom, ``alpha = 0.4``
leaves ``erfc(2.4) = 7e-4``, which is comfortable. Too small an ``alpha``
hands the network a tail it structurally cannot represent; too large makes
the reciprocal sum need a fine mesh. It must also match between training and
any later MD.
"""

import math
from typing import (
    Any,
)

import torch
from deepmd.dpmodel.modifier.dipole_charge import (
    ELECTROSTATIC_CONVERSION,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    pme_reciprocal_space,
)

#: Highest B-spline interpolation order nvalchemi computes correctly.
#:
#: At order >= 7 ``pme_reciprocal_space`` silently drops the entire
#: reciprocal-space sum and returns only the self-energy term, raising
#: nothing. The result stays plausible and still varies with alpha, so it
#: reads as a convergence problem rather than a bug -- in testing it produced
#: a 7% error in the NaCl Madelung constant that was invariant to mesh size.
#: Validated by ``tests/test_pme_physics.py``; a hard ceiling, not a tunable.
MAX_SPLINE_ORDER = 6

#: Default B-spline order: the highest that is correct.
DEFAULT_SPLINE_ORDER = 6

#: Default Ewald splitting parameter, in inverse angstrom. Matches the DPLR
#: default (``ewald_beta``) and leaves erfc(alpha * 6 A) = 7e-4 for a 6 A
#: descriptor cutoff.
DEFAULT_ALPHA = 0.4


class LongRangeEnergy(torch.nn.Module):
    r"""Per-atom reciprocal-space electrostatic energy, in eV.

    Parameters
    ----------
    alpha
        Ewald splitting parameter in inverse angstrom. Not arbitrary here --
        see the module docstring.
    mesh_spacing
        Reciprocal-mesh spacing in angstrom; the mesh is
        ``ceil(cell_length / mesh_spacing)``. This makes the mesh a *step
        function* of the cell, so the energy is only piecewise smooth in the
        cell -- see ``mesh_dimensions``.
    mesh_dimensions
        Fixed mesh ``(nx, ny, nz)``, overriding ``mesh_spacing``. Prefer this
        whenever the cell can change (NPT, or any strain derivative): a
        spacing-derived mesh jumps a whole grid point as a cell length crosses
        a multiple of the spacing, which makes the energy discontinuous there.
        The jump is small but real -- it moved a virial by 2% in testing.
    spline_order
        B-spline interpolation order; must not exceed :data:`MAX_SPLINE_ORDER`.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        mesh_spacing: float | None = None,
        mesh_dimensions: tuple[int, int, int] | list[int] | None = None,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Reject unknown keys rather than absorb them. This term used to take
        # a real-space `rcut`; dropping the real-space half made it
        # meaningless, and a permissive **kwargs silently accepted stale
        # configs that then ran with different physics than they asked for.
        if kwargs:
            raise TypeError(
                f"unexpected long-range parameters: {sorted(kwargs)}. "
                "Supported: alpha, mesh_spacing, mesh_dimensions, "
                "spline_order. Note `rcut` was removed: this term computes "
                "only the reciprocal-space half of the Ewald sum, which "
                "needs no real-space cutoff."
            )
        if spline_order > MAX_SPLINE_ORDER:
            raise ValueError(
                f"spline_order={spline_order} exceeds the maximum order "
                f"nvalchemi computes correctly ({MAX_SPLINE_ORDER}). At order "
                ">= 7 the reciprocal-space sum is silently dropped and only "
                "the self-energy is returned, with no error raised."
            )
        if spline_order < 3:
            raise ValueError(f"spline_order must be at least 3, got {spline_order}")
        if alpha is None or alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if mesh_dimensions is not None:
            mesh_dimensions = tuple(int(m) for m in mesh_dimensions)
            if len(mesh_dimensions) != 3 or any(m < 2 for m in mesh_dimensions):
                raise ValueError(
                    "mesh_dimensions must be three integers >= 2, got "
                    f"{mesh_dimensions}"
                )
        if mesh_dimensions is None and mesh_spacing is None:
            mesh_spacing = 1.0
        if mesh_spacing is not None and mesh_spacing <= 0:
            raise ValueError(f"mesh_spacing must be positive, got {mesh_spacing}")

        self.alpha = float(alpha)
        self.mesh_spacing = mesh_spacing
        self.mesh_dimensions = mesh_dimensions
        self.spline_order = int(spline_order)

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, mesh_spacing={self.mesh_spacing}, "
            f"mesh_dimensions={self.mesh_dimensions}, "
            f"spline_order={self.spline_order}"
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "mesh_spacing": self.mesh_spacing,
            "mesh_dimensions": self.mesh_dimensions,
            "spline_order": self.spline_order,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LongRangeEnergy":
        return cls(**data)

    def _mesh(self, cell: torch.Tensor) -> tuple[int, int, int]:
        if self.mesh_dimensions is not None:
            return self.mesh_dimensions
        lengths = torch.linalg.norm(cell.detach(), dim=-1).max(dim=0).values
        return tuple(
            max(int(math.ceil(float(length) / self.mesh_spacing)), 2)
            for length in lengths
        )

    def forward(
        self,
        coord: torch.Tensor,
        charge: torch.Tensor,
        box: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-atom long-range electrostatic energy, in eV.

        Parameters
        ----------
        coord
            Coordinates in angstrom, shape ``(nf, nloc, 3)``.
        charge
            Per-atom charges in elementary-charge units, shape ``(nf, nloc)``,
            already satisfying the total-charge constraint.
        box
            Cell vectors as rows, shape ``(nf, 3, 3)``. Required.
        mask
            Optional ``(nf, nloc)`` mask of real atoms. Padding atoms are
            given zero charge, which removes them from the reciprocal sum and
            from the self term exactly -- both are quadratic in charge.

        Returns
        -------
        torch.Tensor
            Per-atom energy in eV, shape ``(nf, nloc, 1)``.
        """
        if box is None:
            raise ValueError(
                "The long-range term requires a periodic cell, but box is "
                "None. Ewald summation is only defined under periodic "
                "boundary conditions; for an isolated system use a real-space "
                "Coulomb or DSF term instead."
            )
        if coord.ndim != 3 or coord.shape[-1] != 3:
            raise ValueError(f"coord must have shape (nf, nloc, 3), got {coord.shape}")
        if charge.shape != coord.shape[:2]:
            raise ValueError(
                f"charge must have shape (nf, nloc) = {coord.shape[:2]}, "
                f"got {charge.shape}"
            )

        nf, nloc = coord.shape[0], coord.shape[1]
        cell = box.reshape(nf, 3, 3)

        volume = torch.abs(torch.linalg.det(cell.detach()))
        if bool((volume <= 0).any()):
            raise ValueError(
                "The long-range term requires a non-degenerate cell, but at "
                "least one frame has zero volume. A collapsed or unset box "
                "cannot define periodic images."
            )

        positions = coord.reshape(nf * nloc, 3)
        charges = charge.reshape(nf * nloc)
        if mask is not None:
            charges = charges * mask.reshape(nf * nloc).to(charges.dtype)

        batch_idx = torch.repeat_interleave(
            torch.arange(nf, dtype=torch.int32, device=coord.device), nloc
        )

        # nvalchemi returns the k-space sum MINUS the self term; DeePMD-kit's
        # DPLR convention is the k-space sum alone, so add the self term back.
        # It is per-atom and charge-quadratic, so it lands naturally on the
        # per-atom axis.
        energy = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=self.alpha,
            mesh_dimensions=self._mesh(cell),
            spline_order=self.spline_order,
            batch_idx=batch_idx,
            energy_reduction="atom",
        )
        self_energy = (self.alpha / math.sqrt(math.pi)) * charges * charges
        energy = (energy + self_energy) * ELECTROSTATIC_CONVERSION
        return energy.reshape(nf, nloc, 1)


#: Backwards-compatible alias. The previous class computed the *full* Ewald
#: sum; this one computes only its long-range half, so the name changed with
#: the behaviour.
PMEEnergy = LongRangeEnergy
