# SPDX-License-Identifier: MIT
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

For a system with no cell there is no reciprocal space, but there is still the
same split: see :class:`IsolatedLongRangeEnergy`, which sums the smooth half
directly. It is not an approximation of this class but its infinite-volume
limit, which is what lets one descriptor serve periodic and isolated data at
once. Choose between them with :func:`build_long_range_energy`.

**The self term stays subtracted.** The bare k-space sum counts each Gaussian
interacting with itself -- an artifact of replacing point charges with clouds,
not a physical interaction -- so the Ewald decomposition removes it. Both
backends already return ``recip - self``, and that is what this module
returns: the physically meaningful long-range contribution, leaving the
network the clean screened real-space sum and nothing else.

Verified independently against the textbook formula written out directly:
``recip = 11.37766014``, ``self = 20.63445128``, ``recip - self =
-9.25679114`` eV for a random 8-charge cell, against nvalchemiops'
``-9.25679220``.

An earlier version added the self term back, to match DeePMD-kit's DPLR
``ewald_reciprocal_energy``, which reports the bare k-space sum. That put a
spurious self-interaction inside the quantity called "long-range energy" --
on the LiFSI frame, 103.0 eV of a reported 208.8 eV -- and since it is
positive-definite in ``q_i^2`` it also biases training toward small charges.
Nothing in this package uses DPLR, so there is no convention to match.

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

import numpy as np
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

from deepmd_lr.cell import inverse_3x3
from deepmd_lr.lr.isolated import (
    DEFAULT_MAX_ISOLATED_ATOMS,
    ISOLATED_KERNELS,
    isolated_erf_energy,
)

#: Ewald's default truncation: ``exp(-k^2 / 4 alpha^2)`` falls to this at
#: ``k_cutoff``.
DEFAULT_ACCURACY = 1e-6


def _check_inputs(coord: torch.Tensor, charge: torch.Tensor) -> None:
    """Refuse coordinates and charges whose shapes do not describe frames."""
    if coord.ndim != 3 or coord.shape[-1] != 3:
        raise ValueError(f"coord must have shape (nf, nloc, 3), got {coord.shape}")
    if charge.shape != coord.shape[:2]:
        raise ValueError(
            f"charge must have shape (nf, nloc) = {coord.shape[:2]}, "
            f"got {charge.shape}"
        )


#: Highest B-spline interpolation order nvalchemiops computes correctly.
#:
#: At order >= 7 ``pme_reciprocal_space`` silently drops the entire
#: reciprocal-space sum and returns minus the self-energy term alone, raising
#: nothing. The result stays plausible and still varies with alpha, so it
#: reads as a convergence problem rather than a bug -- in testing it produced
#: a 7% error in the NaCl Madelung constant that was invariant to mesh size.
#: Validated by ``tests/test_pme_physics.py``; a hard ceiling, not a tunable.
MAX_SPLINE_ORDER = 6

#: Default B-spline order: the highest that is correct.
DEFAULT_SPLINE_ORDER = 6

#: Default Ewald splitting parameter, in inverse angstrom. Chosen so that
#: erfc(alpha * 6 A) = 7e-4 for a 6 A descriptor cutoff -- i.e. the part left
#: to the network dies inside its own reach. A modelling choice, not a fitted
#: parameter, and it must match between training and MD (see
#: docs/alpha-and-mesh.md).
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
        accuracy: float = DEFAULT_ACCURACY,
        slab_correction: bool = False,
        pbc: tuple[bool, bool, bool] | list[bool] | None = None,
        isolated_kernel: str = "auto",
        max_isolated_atoms: int = DEFAULT_MAX_ISOLATED_ATOMS,
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
        if accuracy != DEFAULT_ACCURACY:
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
                    f"nvalchemiops computes correctly ({MAX_SPLINE_ORDER}). At "
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
        # How a frame with no cell is summed (:func:`isolated_term_for`).
        self.isolated_kernel, self.max_isolated_atoms = _checked_isolated_kernel(
            isolated_kernel, max_isolated_atoms
        )
        self.slab_correction = slab_correction
        self.pbc = pbc
        #: Refuse a degenerate cell at every call. That check reads the device,
        #: so a driver that validates the cell on the host when it changes
        #: (``DistributedEvaluator``) switches it off for its own term.
        self.check_cell = True
        #: PME k-vectors fixed by :meth:`set_k_vectors`, or None to generate
        #: them at every call.
        self._k_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.mesh_spacing = mesh_spacing
        self.mesh_dimensions = mesh_dimensions
        self.spline_order = int(spline_order)
        self.k_cutoff = None if k_cutoff is None else float(k_cutoff)

    #: This term is only defined under periodic boundary conditions.
    needs_cell = True

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
            "isolated_kernel": self.isolated_kernel,
            "max_isolated_atoms": self.max_isolated_atoms,
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

    def _pbc_on(self, device) -> torch.Tensor:
        """The periodicity flags as a device tensor, uploaded once per device.

        Building them from the Python tuple at every call is a pageable
        host-to-device copy, which blocks the host until the device has
        drained -- one of the per-step synchronisations the MD engine removes.
        """
        cache = self.__dict__.setdefault("_pbc_cache", {})
        key = str(torch.device(device))
        pbc = cache.get(key)
        if pbc is None:
            pbc = torch.tensor(self.pbc, dtype=torch.bool, device=device)
            cache[key] = pbc
        return pbc

    def set_k_vectors(self, cell: torch.Tensor | None) -> None:
        """Fix the PME k-vectors for ``cell`` until the next call.

        nvalchemiops regenerates them at every evaluation, which includes a
        small upload from host memory that stalls the host until the device
        has drained. A driver that evaluates the same cell step after step --
        the MD engine -- computes them once here, when it sets the cell, and
        the term reuses them. The caller owns the cell: it must call again
        whenever the cell changes (``DistributedEvaluator.set_cell`` does),
        and pass ``None`` to go back to per-call generation. ``cell`` is
        ``(3, 3)`` or ``(1, 3, 3)`` on the device the term runs on. Only the
        PME method has them; the Ewald sum builds its own vectors per call.
        """
        if cell is None or self.method != "pme":
            self._k_cache = None
            return
        self._k_cache = self._k_vectors_for(cell.detach().reshape(1, 3, 3))

    def _k_vectors_for(
        self, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """nvalchemiops' ``generate_k_vectors_pme``, without its host upload.

        The Miller-index grid depends only on the mesh and is cached per mesh,
        device and dtype; the reciprocal cell is a 3 x 3 inverse written out
        (:func:`deepmd_lr.cell.inverse_3x3`); and the floor under ``k^2`` is a
        clamp rather than a scalar tensor built from a Python float at every
        call, which is the pageable upload a barostat step would otherwise
        pay. Same convention as nvalchemiops: ``k = h @ (2 pi inv(H))^T``
        with ``H`` rows as lattice vectors and ``h`` the Miller indices.
        """
        mesh = tuple(int(m) for m in self._mesh(cell))
        key = (mesh, str(cell.device), cell.dtype)
        cache = self.__dict__.setdefault("_miller_cache", {})
        grid = cache.get(key)
        if grid is None:
            nx, ny, nz = mesh
            kx = torch.fft.fftfreq(nx, d=1.0, device=cell.device, dtype=cell.dtype) * nx
            ky = torch.fft.fftfreq(ny, d=1.0, device=cell.device, dtype=cell.dtype) * ny
            kz = torch.fft.rfftfreq(nz, d=1.0, device=cell.device, dtype=cell.dtype)
            kz = kz * nz
            grid = torch.stack(torch.meshgrid(kx, ky, kz, indexing="ij"), dim=-1)
            cache[key] = grid
        reciprocal = 2.0 * math.pi * inverse_3x3(cell.reshape(3, 3))
        k_vectors = grid @ reciprocal.transpose(0, 1)
        k_squared = (k_vectors * k_vectors).sum(-1)
        return k_vectors, k_squared.clamp(min=1e-12)

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
                "The periodic long-range term needs a cell, but box is None; a "
                "frame with no cell is summed by isolated_term_for(term)."
            )
        _check_inputs(coord, charge)

        nf, nloc = coord.shape[0], coord.shape[1]
        cell = box.reshape(nf, 3, 3)

        if self.check_cell:
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
        # The energy is quadratic in the charges, so the unit conversion goes
        # onto them rather than onto the energy afterwards. Not cosmetic: the
        # MD driver differentiates the summed energy, and nvalchemiops'
        # backward takes its fast, read-free path only when the cotangent it
        # receives is a plain expanded constant. A multiply after the sum
        # turns that into a tensor it inspects on the host at every step.
        charges = charges * math.sqrt(ELECTROSTATIC_CONVERSION)

        batch_idx = torch.repeat_interleave(
            torch.arange(nf, dtype=torch.int32, device=coord.device), nloc
        )
        pbc = self._pbc_on(coord.device)

        # Both backends return the k-space sum MINUS the Ewald self term,
        # which is what we want: the bare k-space sum counts each Gaussian
        # interacting with itself, an artifact of the smearing rather than a
        # physical interaction. See the module docstring.
        if self.method == "pme":
            cached = self._k_cache
            energy = pme_reciprocal_space(
                positions,
                charges,
                cell,
                alpha=self.alpha,
                mesh_dimensions=self._mesh(cell),
                spline_order=self.spline_order,
                batch_idx=batch_idx,
                energy_reduction="atom",
                k_vectors=None if cached is None else cached[0],
                k_squared=None if cached is None else cached[1],
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
        if self.slab_correction:
            # Yeh-Berkowitz: cancels the interaction between a slab and its
            # periodic images across the vacuum. Added to the 3D result, as
            # nvalchemiops' own docstring prescribes.
            energy = energy + compute_slab_correction(
                positions,
                charges,
                cell,
                pbc.expand(nf, 3),
                batch_idx=batch_idx,
                energy_reduction="atom",
            )

        return energy.reshape(nf, nloc, 1)


def _checked_isolated_kernel(kernel: str, max_atoms: int) -> tuple[str, int]:
    if kernel not in ISOLATED_KERNELS:
        raise ValueError(
            f"isolated_kernel must be one of {ISOLATED_KERNELS}, got {kernel!r}"
        )
    if int(max_atoms) < 1:
        raise ValueError(f"max_isolated_atoms must be >= 1, got {max_atoms}")
    return kernel, int(max_atoms)


class IsolatedLongRangeEnergy(torch.nn.Module):
    r"""The same long-range energy, for a system with no periodic axis.

    Ewald splits Coulomb at :math:`\alpha` into a screened part the network
    learns and a smooth part this package supplies:

    .. math::
        \sum_{i<j} \frac{q_i q_j}{r_{ij}}
          = \underbrace{\sum_{i<j}
                \frac{q_i q_j \operatorname{erfc}(\alpha r_{ij})}{r_{ij}}
            }_{\text{short-ranged; the descriptor covers this}}
          + \underbrace{\sum_{i<j}
                \frac{q_i q_j \operatorname{erf}(\alpha r_{ij})}{r_{ij}}
            }_{\text{this class}}

    Under periodicity that second sum is what :class:`LongRangeEnergy`
    evaluates in reciprocal space; without a cell it is summed directly, and it
    is the *same quantity*: put a cluster in a cube of side ``L`` and
    ``recip - self`` converges onto it as :math:`1/L^3` (``-6.6e-03``,
    ``-8.3e-04``, ``-1.0e-04`` eV at ``L = 80, 160, 320`` A, pinned by
    ``tests/test_isolated_term.py``). So a shared descriptor trained on
    periodic and isolated frames sees the same ``erfc(alpha r)/r`` remainder
    in both. Three things follow:

    * **No self term.** ``erf(alpha r)/r -> 2 alpha / sqrt(pi)`` as ``r -> 0``,
      which times ``0.5 * sum_i q_i^2`` is the Ewald self term; summing
      ``j != i`` already excludes it.
    * **Charged systems need nothing extra**: no images, no neutralising
      background, so an ion is just an ion.
    * **No cutoff.** ``erf(alpha r)/r`` decays as ``1/r``; every pair is summed.

    Two kernels evaluate it (:mod:`deepmd_lr.lr.isolated`): the dense one,
    twice differentiable and O(N^2) in memory, and the chunked one, O(chunk N)
    and first derivatives only. ``isolated_kernel`` picks: ``"auto"`` is dense
    up to ``max_isolated_atoms`` atoms per call and chunked beyond. Force
    training and Hessians need ``"dense"``. ``kernel_dtype`` (a runtime
    attribute, not part of the config) is the dtype the sum runs in; ``None``
    is the inputs' own.

    Parameters
    ----------
    alpha
        Ewald splitting parameter in inverse angstrom. **Must match the value
        used by the periodic branches**; it defines where the split falls.
    """

    #: No cell required -- this is the whole reason the class exists.
    needs_cell = False

    #: Selector under ``long_range.reciprocal.method``.
    METHOD = "isolated"

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        method: str | None = None,
        mesh_spacing: float | None = None,
        mesh_dimensions: tuple[int, int, int] | list[int] | None = None,
        spline_order: int = DEFAULT_SPLINE_ORDER,
        k_cutoff: float | None = None,
        accuracy: float = DEFAULT_ACCURACY,
        slab_correction: bool = False,
        pbc: tuple[bool, bool, bool] | list[bool] | None = None,
        isolated_kernel: str = "auto",
        max_isolated_atoms: int = DEFAULT_MAX_ISOLATED_ATOMS,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del method
        # `dargs` fills the whole block in before anything is built, so an
        # isolated config arrives carrying `spline_order: 6` and `pbc: None`
        # that nobody wrote. Refuse only a value someone *chose*: a periodic
        # setting that looks applied but is not.
        supplied = [
            name
            for name, value, unset in (
                ("mesh_spacing", mesh_spacing, None),
                ("mesh_dimensions", mesh_dimensions, None),
                ("spline_order", spline_order, DEFAULT_SPLINE_ORDER),
                ("k_cutoff", k_cutoff, None),
                ("accuracy", accuracy, DEFAULT_ACCURACY),
                ("slab_correction", slab_correction, False),
            )
            if value != unset
        ]
        if pbc is not None and any(bool(p) for p in pbc):
            supplied.append("pbc")
        if kwargs:
            supplied += sorted(kwargs)
        if supplied:
            raise ValueError(
                f"{sorted(supplied)} do not apply to method='isolated', which "
                "takes `alpha`, `isolated_kernel` and `max_isolated_atoms`. "
                "Meshes, k-vector cutoffs, `slab_correction` and a periodic "
                "`pbc` describe a sum over periodic images, which an isolated "
                "system has none of."
            )
        if alpha is None or alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = float(alpha)
        self.method = self.METHOD
        self.pbc = (False, False, False)
        self.isolated_kernel, self.max_isolated_atoms = _checked_isolated_kernel(
            isolated_kernel, max_isolated_atoms
        )
        self.kernel_dtype: torch.dtype | None = None
        #: Interface parity with the periodic term; there is no cell to check.
        self.check_cell = True

    def extra_repr(self) -> str:
        return (
            f"method='{self.METHOD}', alpha={self.alpha}, "
            f"isolated_kernel='{self.isolated_kernel}'"
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "method": self.METHOD,
            "isolated_kernel": self.isolated_kernel,
            "max_isolated_atoms": self.max_isolated_atoms,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "IsolatedLongRangeEnergy":
        return cls(**data)

    def set_k_vectors(self, cell: torch.Tensor | None) -> None:
        """Nothing to fix: the direct sum has no reciprocal vectors."""

    def forward(
        self,
        coord: torch.Tensor,
        charge: torch.Tensor,
        box: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-atom long-range electrostatic energy in eV, ``(nf, nloc, 1)``.

        ``coord`` ``(nf, nloc, 3)`` in angstrom, ``charge`` ``(nf, nloc)`` in
        e, already projected onto the total charge; each frame is its own
        system. ``box`` is ignored (interchangeable with
        :class:`LongRangeEnergy`). ``mask`` zeroes the charge of padding atoms,
        which removes them exactly: every term is bilinear in the charges.
        """
        del box
        _check_inputs(coord, charge)
        q = charge if mask is None else charge * mask.to(charge.dtype)
        energy = isolated_erf_energy(
            coord,
            q,
            self.alpha,
            kernel=self.isolated_kernel,
            max_dense_atoms=self.max_isolated_atoms,
            dtype=self.kernel_dtype,
        )
        return (energy * ELECTROSTATIC_CONVERSION).unsqueeze(-1)


def isolated_term_for(term: torch.nn.Module) -> IsolatedLongRangeEnergy:
    """The isolated term with ``term``'s ``alpha`` and kernel settings: how a
    frame with no cell is summed, whatever method the model was trained with."""
    if isinstance(term, IsolatedLongRangeEnergy):
        return term
    return IsolatedLongRangeEnergy(
        alpha=term.alpha,
        isolated_kernel=term.isolated_kernel,
        max_isolated_atoms=term.max_isolated_atoms,
    )


#: Either long-range term. They are interchangeable at every call site: same
#: ``forward(coord, charge, box, mask)``, same ``(nf, nloc, 1)`` result, same
#: ``serialize``/``deserialize`` shape. What differs is ``needs_cell``.
LongRangeTerm = LongRangeEnergy | IsolatedLongRangeEnergy


#: Every way of evaluating the long-range sum, by ``method``.
TERM_CLASSES: dict[str, type[torch.nn.Module]] = {
    "pme": LongRangeEnergy,
    "ewald": LongRangeEnergy,
    IsolatedLongRangeEnergy.METHOD: IsolatedLongRangeEnergy,
}


def build_long_range_energy(settings: dict[str, Any]) -> torch.nn.Module:
    """Construct the long-range term named by ``settings["method"]``.

    The single place that maps a config or a checkpoint onto a term class, so
    the model and its deserializer cannot drift apart on which one to build.
    """
    method = settings.get("method", "pme")
    if method not in TERM_CLASSES:
        raise ValueError(
            f"method must be one of {tuple(TERM_CLASSES)}, got {method!r}"
        )
    return TERM_CLASSES[method](**settings)


def system_mesh(settings: dict[str, Any], cell: Any) -> tuple[int, int, int]:
    """The PME mesh a cell needs: at least ``ceil(2 alpha L)`` points along
    each cell vector, and never fewer than ``settings`` asks for.

    ``1 / (2 alpha)`` is the spacing the screening Gaussian needs -- 1.25 A at
    alpha 0.4. A fixed mesh is right for the cell it was chosen for and for no
    other: the water model's (10, 10, 10), exact on its 12.43 A training box,
    was 0.49 meV/atom and 0.0077 eV/A rms off on the 49.7 A 4x4x4 supercell of
    that same box, where this rule's (40, 40, 40) is exact.
    """
    lengths = np.linalg.norm(np.asarray(cell, dtype=np.float64).reshape(3, 3), axis=1)
    alpha = float(settings.get("alpha", DEFAULT_ALPHA))
    if settings.get("mesh_dimensions") is not None:
        asked = [int(n) for n in settings["mesh_dimensions"]]
    else:
        spacing = settings.get("mesh_spacing") or 1.0  # the term's own default
        asked = [math.ceil(length / spacing - 1e-9) for length in lengths]
    return tuple(
        max(n, math.ceil(2.0 * alpha * length - 1e-9), 2)
        for n, length in zip(asked, lengths, strict=True)
    )


def system_reciprocal(settings: dict[str, Any], cell: Any, pbc: Any) -> dict[str, Any]:
    """The reciprocal-space settings a system needs, from a model's own.

    The term holds no weights, so how the sum is taken is a property of the
    system, not of the checkpoint. The system's ``pbc`` picks the sum: all
    three axes periodic is the 3D sum, one open axis a slab with the
    Yeh-Berkowitz correction, no periodic axis the isolated sum (any trained
    method; ``alpha`` and the kernel settings carry over). The cell sets the
    PME mesh (:func:`system_mesh`). A wire (one periodic axis) is refused, and
    so is a periodic system for a model trained with ``method: isolated``,
    which has never seen a periodic image.
    """
    method = settings.get("method", "pme")
    periodic = [bool(p) for p in np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))]
    if not any(periodic):
        if method == IsolatedLongRangeEnergy.METHOD:
            return dict(settings)
        return {
            "alpha": float(settings.get("alpha", DEFAULT_ALPHA)),
            "method": IsolatedLongRangeEnergy.METHOD,
            "isolated_kernel": settings.get("isolated_kernel", "auto"),
            "max_isolated_atoms": settings.get(
                "max_isolated_atoms", DEFAULT_MAX_ISOLATED_ATOMS
            ),
        }
    if method == IsolatedLongRangeEnergy.METHOD:
        raise ValueError(
            f"method 'isolated' sums no periodic images, but pbc={periodic} marks "
            "a periodic axis: a model trained with method 'isolated' runs on "
            "systems with no periodic axis."
        )
    out = dict(settings)
    if all(periodic):
        out.update(slab_correction=False, pbc=[True, True, True])
    elif sum(periodic) == 2:
        out.update(slab_correction=True, pbc=periodic)
    else:
        raise ValueError(
            f"pbc={periodic} leaves two axes open: a wire (one periodic axis) is "
            "not supported. The long-range term sums a bulk cell, a slab (one "
            "open axis) and a system with no periodic axis."
        )
    if method == "pme":
        out["mesh_dimensions"] = list(system_mesh(settings, cell))
        out["mesh_spacing"] = None
    return out
