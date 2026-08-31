# SPDX-License-Identifier: LGPL-3.0-or-later
"""The long-range electrostatic energy term, backed by nvalchemi's PME.

This is a thin, well-guarded wrapper. It owns three things that are easy to
get wrong and expensive to notice later: the unit convention, the PME
parameter choice, and the boundary conditions PME is and is not valid under.

Units
-----
nvalchemi computes the electrostatic energy with a prefactor of 1 --
:math:`\\sum_{i<j} q_i q_j / r_{ij}` -- so with charges in elementary-charge
units and coordinates in angstrom, its output is in units of
:math:`e^2/\\text{\\AA}`. DeePMD-kit works in eV and angstrom, so the result is
scaled by :math:`e^2/(4\\pi\\varepsilon_0)` in eV*angstrom.

That constant is imported from DeePMD-kit rather than redefined here. It is
the same one its existing DPLR path uses, so a long-range energy from this
term lands on exactly the same scale as one from ``dipole_charge`` -- and a
second definition would be a silent source of drift between them. (The value
differs from a current-CODATA evaluation in the eighth significant figure;
matching DeePMD-kit is worth more than matching CODATA.)

Boundary conditions
-------------------
PME assumes 3D periodicity. A non-periodic frame is rejected rather than
silently treated as periodic, which would fabricate interactions with images
that do not exist.
"""

import math
from typing import (
    Any,
)

import torch
from deepmd.dpmodel.modifier.dipole_charge import (
    ELECTROSTATIC_CONVERSION,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_pme_parameters,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    particle_mesh_ewald,
)
from nvalchemiops.torch.neighbors import neighbor_list

#: Highest B-spline interpolation order nvalchemi computes correctly.
#:
#: At order >= 7 ``pme_reciprocal_space`` silently drops the entire
#: reciprocal-space sum and returns only the self-energy term, raising
#: nothing. The result stays plausible and still varies with alpha, so it
#: reads as a convergence problem rather than a bug -- in testing it produced
#: a 7% error in the NaCl Madelung constant that was invariant to mesh size.
#: This is validated by ``tests/test_pme_physics.py``; treat it as a hard
#: ceiling, not a tunable.
MAX_SPLINE_ORDER = 6

#: Default B-spline order: the highest that is correct, and accurate enough
#: that the mesh is never the limiting error term in practice.
DEFAULT_SPLINE_ORDER = 6

#: Starting neighbour-list capacity per atom. The list is rebuilt with a
#: larger capacity if this overflows, so it is a performance hint and not a
#: correctness limit.
INITIAL_MAX_NEIGHBORS = 128

#: Ceiling on capacity growth, to fail loudly instead of exhausting memory on
#: a pathological (e.g. collapsed-cell) frame.
MAX_NEIGHBOR_CAPACITY = 8192

#: Neighbour-list algorithm, pinned rather than auto-selected.
#:
#: nvalchemi's ``naive`` batched builder silently under-reports neighbours for
#: every frame after the first: for two identical 216-atom frames it returned
#: 17280 pairs for frame 0 but only 10024 for frame 1, with no error and no
#: sensitivity to ``max_neighbors``. Since a training batch is exactly the
#: batched case, letting the dispatcher pick (it chooses ``naive`` at these
#: sizes) would corrupt the long-range energy of all but the first frame of
#: every batch -- and quietly, since the result stays finite and plausible.
#: ``cell_list`` returns the correct count for every frame. Regression-tested
#: by ``TestBatching`` in ``tests/test_pme_term.py``.
NEIGHBOR_LIST_METHOD = "cell_list"


class PMEEnergy(torch.nn.Module):
    r"""Per-atom long-range electrostatic energy for periodic frames.

    Parameters
    ----------
    rcut
        Real-space cutoff in angstrom. ``None`` lets nvalchemi choose one
        from the cell volume and atom count. Pinning it is usually better:
        it keeps the neighbour-list cost predictable and stops the split
        between real and reciprocal space from moving as a training batch
        changes shape.
    alpha
        Ewald splitting parameter in inverse angstrom. ``None`` derives it
        from ``rcut`` and ``accuracy`` as :math:`\sqrt{-\ln\varepsilon}/r_c`.
    mesh_spacing
        Reciprocal-mesh spacing in angstrom. ``None`` derives the mesh from
        ``accuracy``.
    spline_order
        B-spline interpolation order; must not exceed :data:`MAX_SPLINE_ORDER`.
    accuracy
        Target relative accuracy used for any parameter left as ``None``.
    """

    def __init__(
        self,
        rcut: float | None = None,
        alpha: float | None = None,
        mesh_spacing: float | None = None,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        accuracy: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if spline_order > MAX_SPLINE_ORDER:
            raise ValueError(
                f"spline_order={spline_order} exceeds the maximum order "
                f"nvalchemi computes correctly ({MAX_SPLINE_ORDER}). At order "
                ">= 7 the reciprocal-space sum is silently dropped and only "
                "the self-energy is returned, with no error raised."
            )
        if spline_order < 3:
            raise ValueError(f"spline_order must be at least 3, got {spline_order}")
        if rcut is not None and rcut <= 0:
            raise ValueError(f"rcut must be positive, got {rcut}")
        if alpha is not None and alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0 < accuracy < 1:
            raise ValueError(f"accuracy must be in (0, 1), got {accuracy}")

        self.rcut = rcut
        self.alpha = alpha
        self.mesh_spacing = mesh_spacing
        self.spline_order = int(spline_order)
        self.accuracy = float(accuracy)

    def extra_repr(self) -> str:
        return (
            f"rcut={self.rcut}, alpha={self.alpha}, "
            f"mesh_spacing={self.mesh_spacing}, "
            f"spline_order={self.spline_order}, accuracy={self.accuracy}"
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "rcut": self.rcut,
            "alpha": self.alpha,
            "mesh_spacing": self.mesh_spacing,
            "spline_order": self.spline_order,
            "accuracy": self.accuracy,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "PMEEnergy":
        return cls(**data)

    def _resolved_parameters(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> tuple[float, float, tuple[int, int, int]]:
        """Settle rcut, alpha and the mesh, filling gaps from ``accuracy``.

        nvalchemi's estimator is consulted with detached geometry: these are
        discretization choices, not differentiable quantities, and letting a
        gradient run back into the mesh size is meaningless.
        """
        estimated = estimate_pme_parameters(
            positions.detach(),
            cell.detach(),
            batch_idx=batch_idx,
            accuracy=self.accuracy,
            real_space_cutoff=self.rcut,
        )
        rcut = (
            self.rcut
            if self.rcut is not None
            else float(estimated.real_space_cutoff.min())
        )
        # A single alpha is shared across the batch: it is tied to the one
        # real-space cutoff the shared neighbour list was built with.
        alpha = (
            self.alpha if self.alpha is not None else float(estimated.alpha.min())
        )
        if self.mesh_spacing is not None:
            lengths = torch.linalg.norm(cell.detach(), dim=-1).max(dim=0).values
            mesh = tuple(
                max(int(math.ceil(float(length) / self.mesh_spacing)), 2)
                for length in lengths
            )
        else:
            mesh = tuple(int(m) for m in estimated.mesh_dimensions)
        return rcut, alpha, mesh

    def _neighbor_list(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        batch_idx: torch.Tensor,
        rcut: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the real-space neighbour list, growing capacity on overflow.

        The algorithm is pinned to :data:`NEIGHBOR_LIST_METHOD`; see that
        constant for why the auto-selected one cannot be used.

        nvalchemi allocates a fixed ``max_neighbors`` per atom and raises
        rather than truncating when it is exceeded -- truncation would
        silently drop real-space pairs and bias the energy. Doubling on
        overflow keeps that guarantee without asking the caller to guess a
        capacity that depends on density and cutoff.

        Built from detached geometry: edge indices and lattice shifts are
        discrete, so they carry no gradient. The energy is still
        differentiable in the positions through ``particle_mesh_ewald``.
        """
        from nvalchemiops.neighbors.neighbor_utils import (
            NeighborOverflowError,
        )

        capacity = INITIAL_MAX_NEIGHBORS
        while True:
            try:
                return neighbor_list(
                    positions.detach(),
                    rcut,
                    cell=cell.detach(),
                    pbc=pbc,
                    batch_idx=batch_idx,
                    return_neighbor_list=True,
                    max_neighbors=capacity,
                    method=NEIGHBOR_LIST_METHOD,
                )
            except NeighborOverflowError:
                capacity *= 2
                if capacity > MAX_NEIGHBOR_CAPACITY:
                    raise

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
            Per-atom charges in elementary-charge units, shape ``(nf, nloc)``.
            Expected to already satisfy the total-charge constraint.
        box
            Cell vectors as rows, shape ``(nf, 3, 3)``. Required.
        mask
            Optional ``(nf, nloc)`` mask of real atoms. Padding atoms are
            given zero charge, which removes them from every Ewald
            contribution exactly -- including the self term, which is
            quadratic in charge.

        Returns
        -------
        torch.Tensor
            Per-atom energy in eV, shape ``(nf, nloc, 1)``. Summing over
            atoms gives the frame's long-range energy.
        """
        if box is None:
            raise ValueError(
                "The long-range PME term requires a periodic cell, but box "
                "is None. Ewald summation is only defined under periodic "
                "boundary conditions; for an isolated system use a "
                "real-space Coulomb or DSF term instead."
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
                "The long-range PME term requires a non-degenerate cell, but "
                "at least one frame has zero volume. A collapsed or unset box "
                "cannot define periodic images."
            )

        positions = coord.reshape(nf * nloc, 3)
        charges = charge.reshape(nf * nloc)
        if mask is not None:
            charges = charges * mask.reshape(nf * nloc).to(charges.dtype)

        batch_idx = torch.repeat_interleave(
            torch.arange(nf, dtype=torch.int32, device=coord.device), nloc
        )
        pbc = torch.ones(3, dtype=torch.bool, device=coord.device)

        rcut, alpha, mesh = self._resolved_parameters(positions, cell, batch_idx)
        nlist, nptr, nshifts = self._neighbor_list(
            positions, cell, pbc, batch_idx, rcut
        )

        energy = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh,
            spline_order=self.spline_order,
            batch_idx=batch_idx,
            neighbor_list=nlist,
            neighbor_ptr=nptr,
            neighbor_shifts=nshifts,
            pbc=pbc,
            energy_reduction="atom",
        )
        energy = energy * ELECTROSTATIC_CONVERSION
        return energy.reshape(nf, nloc, 1)
