# SPDX-License-Identifier: LGPL-3.0-or-later
r"""The long-range (reciprocal-space) electrostatic energy.

Evaluated either by particle-mesh Ewald or by an explicit k-vector sum --
see *Two ways to evaluate the same sum* below. The module is named for what
it computes rather than for one of the two ways of computing it.

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

Two ways to evaluate the same sum
---------------------------------
``method="pme"`` interpolates the charges onto a mesh and uses an FFT --
``O(N log N)``, the right choice once systems get large. ``method="ewald"``
sums the k-vectors explicitly -- ``O(N * K)``, with no interpolation error at
all. They compute the *same quantity* and agree to the accuracy of whichever
is coarser, so the explicit sum makes a good reference for checking a PME
setting, and is often the faster of the two for small cells where the FFT
overhead dominates.

Slabs
-----
Ewald assumes 3D periodicity. A 2D-periodic system -- an interface, a film, a
liquid slab with vacuum above it -- is normally handled by adding vacuum and
running the 3D sum anyway, but that leaves each slab interacting with its
images through the vacuum. For a slab with a net dipole across the
non-periodic axis the error is large and converges only slowly with vacuum
thickness, so more vacuum is not a fix.

``slab_correction=True`` adds the Yeh-Berkowitz term, which removes that
image-dipole interaction and recovers the 2D-periodic answer. Pass ``pbc``
with ``False`` on the non-periodic axis. It is off by default because it is
wrong for a genuinely 3D-periodic system.

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
from nvalchemiops.torch.interactions.electrostatics.ewald import (
    ewald_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.slab import (
    compute_slab_correction,
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

#: Supported ways of evaluating the reciprocal sum.
METHODS = ("pme", "ewald")

#: Parameters that only mean something for one method. Passing one to the
#: other method raises, rather than being silently ignored -- a setting that
#: looks applied but is not is the worst kind of configuration bug.
_PME_ONLY = ("mesh_spacing", "mesh_dimensions", "spline_order")
_EWALD_ONLY = ("k_cutoff", "accuracy")


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
        method: str = "pme",
        mesh_spacing: float | None = None,
        mesh_dimensions: tuple[int, int, int] | list[int] | None = None,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        k_cutoff: float | None = None,
        accuracy: float = 1e-6,
        slab_correction: bool = False,
        pbc: tuple[bool, bool, bool] | list[bool] | None = None,
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
                f"Supported: alpha, method, {', '.join(_PME_ONLY)}, "
                f"{', '.join(_EWALD_ONLY)}. Note `rcut` was removed: this "
                "term computes only the reciprocal-space half of the Ewald "
                "sum, which needs no real-space cutoff."
            )
        if method not in METHODS:
            raise ValueError(
                f"method must be one of {METHODS}, got {method!r}"
            )
        if alpha is None or alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        # Reject cross-method parameters. Both evaluate the same sum, but
        # their knobs are not interchangeable, and a mesh setting silently
        # ignored under `ewald` would be indistinguishable from one applied.
        supplied_pme = [
            name
            for name, value in (
                ("mesh_spacing", mesh_spacing),
                ("mesh_dimensions", mesh_dimensions),
            )
            if value is not None
        ]
        if spline_order != DEFAULT_SPLINE_ORDER:
            supplied_pme.append("spline_order")
        supplied_ewald = [
            name
            for name, value in (("k_cutoff", k_cutoff),)
            if value is not None
        ]
        if accuracy != 1e-6:
            supplied_ewald.append("accuracy")

        if method == "ewald" and supplied_pme:
            raise ValueError(
                f"{supplied_pme} only apply to method='pme'; with "
                "method='ewald' the k-vectors are summed explicitly and "
                "there is no mesh. Use k_cutoff (or accuracy) instead."
            )
        if method == "pme" and supplied_ewald:
            raise ValueError(
                f"{supplied_ewald} only apply to method='ewald'; with "
                "method='pme' the reciprocal sum is evaluated on a mesh. "
                "Use mesh_dimensions (or mesh_spacing) instead."
            )

        if method == "pme":
            if spline_order > MAX_SPLINE_ORDER:
                raise ValueError(
                    f"spline_order={spline_order} exceeds the maximum order "
                    f"nvalchemi computes correctly ({MAX_SPLINE_ORDER}). At "
                    "order >= 7 the reciprocal-space sum is silently dropped "
                    "and only the self-energy is returned, with no error."
                )
            if spline_order < 3:
                raise ValueError(
                    f"spline_order must be at least 3, got {spline_order}"
                )
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
        else:
            if not 0 < accuracy < 1:
                raise ValueError(f"accuracy must be in (0, 1), got {accuracy}")
            if k_cutoff is None:
                # Standard Ewald reciprocal cutoff: the Gaussian
                # exp(-k^2 / 4 alpha^2) has fallen to `accuracy` by here.
                # Measured: alpha=0.4 with accuracy=1e-6 gives k_cutoff=3.0,
                # which reaches 1e-10 against the reference.
                k_cutoff = 2.0 * alpha * math.sqrt(-math.log(accuracy))
            if k_cutoff <= 0:
                raise ValueError(f"k_cutoff must be positive, got {k_cutoff}")

        if pbc is None:
            pbc = (True, True, True)
        pbc = tuple(bool(p) for p in pbc)
        if len(pbc) != 3:
            raise ValueError(f"pbc must have three entries, got {pbc}")
        if slab_correction and sum(pbc) != 2:
            raise ValueError(
                "slab_correction expects exactly one non-periodic axis, but "
                f"pbc={pbc}. Set pbc with False on the slab normal, e.g. "
                "[true, true, false] for a slab stacked along z."
            )
        if not slab_correction and sum(pbc) != 3:
            raise ValueError(
                f"pbc={pbc} marks a non-periodic axis but slab_correction is "
                "off. Running a plain 3D sum on a slab leaves each image "
                "interacting through the vacuum; set slab_correction: true."
            )

        self.alpha = float(alpha)
        self.method = method
        self.slab_correction = slab_correction
        self.pbc = pbc
        self.mesh_spacing = mesh_spacing
        self.mesh_dimensions = mesh_dimensions
        self.spline_order = int(spline_order)
        self.k_cutoff = None if k_cutoff is None else float(k_cutoff)
        self.accuracy = float(accuracy)

    def extra_repr(self) -> str:
        if self.method == "pme":
            return (
                f"method='pme', alpha={self.alpha}, "
                f"mesh_spacing={self.mesh_spacing}, "
                f"mesh_dimensions={self.mesh_dimensions}, "
                f"spline_order={self.spline_order}"
            )
        return (
            f"method='ewald', alpha={self.alpha}, k_cutoff={self.k_cutoff}"
        )

    def serialize(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "alpha": self.alpha,
            "method": self.method,
            "slab_correction": self.slab_correction,
            "pbc": list(self.pbc),
        }
        if self.method == "pme":
            data.update(
                mesh_spacing=self.mesh_spacing,
                mesh_dimensions=self.mesh_dimensions,
                spline_order=self.spline_order,
            )
        else:
            data.update(k_cutoff=self.k_cutoff)
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LongRangeEnergy":
        return cls(**data)

    def _mesh(self, cell: torch.Tensor) -> tuple[int, int, int]:
        """PME mesh dimensions for this cell."""
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
        pbc = torch.tensor(self.pbc, dtype=torch.bool, device=coord.device)

        # Both backends return the k-space sum MINUS the self term;
        # DeePMD-kit's DPLR convention is the k-space sum alone, so the self
        # term is added back below. It is per-atom and charge-quadratic, so
        # it lands naturally on the per-atom axis.
        if self.method == "pme":
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
        else:
            # The explicit sum wants alpha per system as a tensor, not a
            # float, and its own k-vector set for this cell.
            alpha = torch.full(
                (nf,), self.alpha, dtype=coord.dtype, device=coord.device
            )
            # Keep the reciprocal vectors connected to the strained cell.
            # Treating them as fixed Cartesian metadata gives correct energies
            # and position forces but an unphysical cell derivative/virial.
            # The integer Miller bounds are still discrete setup metadata; the
            # returned Cartesian vectors themselves must deform with the cell.
            k_vectors = generate_k_vectors_ewald_summation(cell, self.k_cutoff)
            energy = ewald_reciprocal_space(
                positions,
                charges,
                cell,
                k_vectors,
                alpha,
                batch_idx=batch_idx,
                energy_reduction="atom",
            )
        self_energy = (self.alpha / math.sqrt(math.pi)) * charges * charges
        energy = energy + self_energy

        if self.slab_correction:
            # Yeh-Berkowitz: cancels the interaction between a slab and its
            # periodic images across the vacuum. Added to the 3D result, as
            # nvalchemi's own docstring prescribes.
            energy = energy + compute_slab_correction(
                positions,
                charges,
                cell,
                pbc.expand(nf, 3),
                batch_idx=batch_idx,
                energy_reduction="atom",
            )

        energy = energy * ELECTROSTATIC_CONVERSION
        return energy.reshape(nf, nloc, 1)


#: Backwards-compatible alias. The previous class computed the *full* Ewald
#: sum; this one computes only its long-range half, so the name changed with
#: the behaviour.
PMEEnergy = LongRangeEnergy
