# SPDX-License-Identifier: MIT
r"""The long-range electrostatic energy: the smooth half of the Ewald split.

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

This module supplies **only the reciprocal sum, minus the self term**. The
real-space part decays like ``erfc(alpha r)/r`` inside the descriptor cutoff,
so the short-range model represents it directly; and without it there is no
neighbour list, no cutoff and no ``rcut < L/2`` constraint.

**The self term stays subtracted.** The bare k-space sum counts each Gaussian
interacting with itself, an artifact of the smearing rather than a physical
interaction. Both backends return ``recip - self``, and so does this module.
DeePMD-kit's DPLR ``ewald_reciprocal_energy`` reports the bare sum instead:
on the LiFSI frame that is 103.0 eV of self-interaction in a reported
208.8 eV, and being positive-definite in ``q_i^2`` it biases training toward
small charges. Checked against the textbook formula on a random 8-charge
cell: ``recip - self = -9.25679114`` eV, nvalchemiops ``-9.25679220``.

Three ways to evaluate the same sum
-----------------------------------
``method="pme"`` interpolates the charges onto a mesh and uses an FFT,
``O(N log N)``. ``method="ewald"`` sums the k-vectors explicitly, ``O(N K)``
with no interpolation error: the reference for checking a PME setting, and
often the faster on small cells. ``method="isolated"``
(:class:`IsolatedLongRangeEnergy`) sums the same split directly for a system
with no cell; it is the periodic sum's infinite-volume limit, which is what
lets one descriptor serve periodic and isolated data at once.
:func:`build_long_range_energy` maps settings onto a class, and
:func:`system_reciprocal` a system's ``pbc`` onto settings.

Slabs
-----
A 2D-periodic system in a padded box still interacts with its images through
the vacuum; for a slab with a net dipole across the open axis the error is
large and shrinks only slowly with more vacuum. ``slab_correction=True`` adds
the Yeh-Berkowitz term, which removes that interaction; ``pbc`` marks the
open axis ``False``. Off by default: it is wrong for a 3D-periodic system.

Choosing alpha
--------------
In a full Ewald sum ``alpha`` is arbitrary. Here it sets the division of
labour: the network handles ``erfc(alpha r)/r``, which must have died inside
the descriptor cutoff (``alpha = 0.4`` leaves ``erfc(2.4) = 7e-4`` at
``rcut = 6`` A); too large an ``alpha`` needs a fine mesh. It must match
between training and any later MD (docs/alpha-and-mesh.md).
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

from deepmd_lr.cell import inverse_3x3, pbc_flags
from deepmd_lr.lr.isolated import (
    DEFAULT_MAX_ISOLATED_ATOMS,
    ISOLATED_KERNELS,
    isolated_erf_energy,
)

#: Ewald's default truncation: ``exp(-k^2 / 4 alpha^2)`` falls to this at
#: ``k_cutoff``.
DEFAULT_ACCURACY = 1e-6

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

#: Default Ewald splitting parameter, in inverse angstrom: a modelling choice,
#: not a fitted parameter (see "Choosing alpha" above).
DEFAULT_ALPHA = 0.4

#: Supported ways of evaluating the reciprocal sum.
METHODS = ("pme", "ewald")

#: Settings that only some methods use, as ``(name, unset value, methods)``.
#: One set to anything but its unset value under another method is refused
#: rather than ignored: a setting that looks applied but is not is the worst
#: kind of configuration bug.
_METHOD_SETTINGS = (
    ("mesh_spacing", None, ("pme",)),
    ("mesh_dimensions", None, ("pme",)),
    ("spline_order", DEFAULT_SPLINE_ORDER, ("pme",)),
    ("k_cutoff", None, ("ewald",)),
    ("accuracy", DEFAULT_ACCURACY, ("ewald",)),
    ("slab_correction", False, METHODS),
    ("pbc", None, METHODS),
)

#: Every setting a term takes.
_SETTINGS = (
    "alpha",
    "method",
    *(name for name, _, _ in _METHOD_SETTINGS),
    "isolated_kernel",
    "max_isolated_atoms",
)


def _misplaced(method: str, **settings: Any) -> list[str]:
    """The names in ``settings`` set although ``method`` does not use them, in
    table order. A ``None`` marker is compared by identity, so an array value
    is never compared elementwise."""
    return [
        name
        for name, unset, methods in _METHOD_SETTINGS
        if name in settings
        and method not in methods
        and (
            settings[name] is not None
            if unset is None
            else bool(settings[name] != unset)
        )
    ]


def _checked_alpha(alpha: Any) -> float:
    if alpha is None or alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    return float(alpha)


def _checked_isolated_kernel(kernel: str, max_atoms: int) -> tuple[str, int]:
    if kernel not in ISOLATED_KERNELS:
        raise ValueError(
            f"isolated_kernel must be one of {ISOLATED_KERNELS}, got {kernel!r}"
        )
    if int(max_atoms) < 1:
        raise ValueError(f"max_isolated_atoms must be >= 1, got {max_atoms}")
    return kernel, int(max_atoms)


def _checked_mesh(
    mesh_spacing: float | None, mesh_dimensions: Any, spline_order: int
) -> tuple[float | None, tuple[int, int, int] | None]:
    """PME's ``(mesh_spacing, mesh_dimensions)``, validated, with the default
    1 A spacing when neither is given."""
    if spline_order > MAX_SPLINE_ORDER:
        raise ValueError(
            f"spline_order={spline_order} exceeds the maximum order "
            f"nvalchemiops computes correctly ({MAX_SPLINE_ORDER}). At "
            "order >= 7 the reciprocal-space sum is silently dropped "
            "and only the self-energy is returned, with no error."
        )
    if spline_order < 3:
        raise ValueError(f"spline_order must be at least 3, got {spline_order}")
    if mesh_dimensions is not None:
        mesh_dimensions = tuple(int(m) for m in mesh_dimensions)
        if len(mesh_dimensions) != 3 or any(m < 2 for m in mesh_dimensions):
            raise ValueError(
                f"mesh_dimensions must be three integers >= 2, got {mesh_dimensions}"
            )
    if mesh_dimensions is None and mesh_spacing is None:
        mesh_spacing = 1.0
    if mesh_spacing is not None and mesh_spacing <= 0:
        raise ValueError(f"mesh_spacing must be positive, got {mesh_spacing}")
    return mesh_spacing, mesh_dimensions


def _checked_k_cutoff(alpha: float, k_cutoff: float | None, accuracy: float) -> float:
    """The Ewald sum's ``k_cutoff``, by default where ``exp(-k^2 / 4 alpha^2)``
    has fallen to ``accuracy``: alpha 0.4 and 1e-6 give 3.0 /A, which reaches
    1e-10 against the reference."""
    if not 0 < accuracy < 1:
        raise ValueError(f"accuracy must be in (0, 1), got {accuracy}")
    if k_cutoff is None:
        k_cutoff = 2.0 * alpha * math.sqrt(-math.log(accuracy))
    if k_cutoff <= 0:
        raise ValueError(f"k_cutoff must be positive, got {k_cutoff}")
    return k_cutoff


def _checked_pbc(pbc: Any, slab_correction: bool) -> tuple[bool, bool, bool]:
    """``pbc`` (fully periodic if None) as three bools: all periodic, or one
    open axis with the slab correction on."""
    pbc = tuple(bool(p) for p in ((True, True, True) if pbc is None else pbc))
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
    return pbc


def _check_inputs(coord: torch.Tensor, charge: torch.Tensor) -> None:
    """Refuse coordinates and charges whose shapes do not describe frames."""
    if coord.ndim != 3 or coord.shape[-1] != 3:
        raise ValueError(f"coord must have shape (nf, nloc, 3), got {coord.shape}")
    if charge.shape != coord.shape[:2]:
        raise ValueError(
            f"charge must have shape (nf, nloc) = {coord.shape[:2]}, "
            f"got {charge.shape}"
        )


class LongRangeEnergy(torch.nn.Module):
    r"""Per-atom reciprocal-space electrostatic energy, in eV.

    Parameters
    ----------
    alpha
        Ewald splitting parameter in inverse angstrom. Not arbitrary here --
        see the module docstring.
    method
        ``"pme"`` or ``"ewald"``. A setting of the other method is refused.
    mesh_spacing
        PME only. Mesh spacing in angstrom; the mesh is
        ``ceil(cell_length / mesh_spacing)``, a *step function* of the cell,
        so the energy is only piecewise smooth in it. 1.0 when no mesh is
        given.
    mesh_dimensions
        PME only. Fixed mesh ``(nx, ny, nz)``, overriding ``mesh_spacing``.
        Prefer this whenever the cell can change (NPT, any strain
        derivative): a spacing-derived mesh jumps a whole grid point as a
        cell length crosses a multiple of the spacing, which moved a virial
        by 2% in testing.
    spline_order
        PME only. B-spline interpolation order, 3 to
        :data:`MAX_SPLINE_ORDER`.
    k_cutoff
        Ewald only. Largest ``|k|`` summed, in inverse angstrom; derived
        from ``alpha`` and ``accuracy`` when None.
    accuracy
        Ewald only. Where ``exp(-k^2 / 4 alpha^2)`` is truncated.
    slab_correction
        Add the Yeh-Berkowitz correction; needs ``pbc`` with one open axis.
    pbc
        Periodicity per axis; fully periodic when None.
    isolated_kernel, max_isolated_atoms
        How a frame with no cell is summed (:func:`isolated_term_for`).
    """

    #: This term is only defined under periodic boundary conditions.
    needs_cell = True

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
        # Unknown keys are refused, not absorbed: a stale config (`rcut`, from
        # when the term had a real-space half) would otherwise run with
        # different physics than it asked for.
        if kwargs:
            raise TypeError(
                f"unexpected long-range parameters: {sorted(kwargs)}. "
                f"Supported: {', '.join(_SETTINGS)}. There is no `rcut`: this "
                "term computes only the reciprocal-space half of the Ewald "
                "sum, which needs no real-space cutoff."
            )
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        alpha = _checked_alpha(alpha)
        misplaced = _misplaced(
            method,
            mesh_spacing=mesh_spacing,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            k_cutoff=k_cutoff,
            accuracy=accuracy,
        )
        if misplaced and method == "ewald":
            raise ValueError(
                f"{misplaced} only apply to method='pme'; with "
                "method='ewald' the k-vectors are summed explicitly and "
                "there is no mesh. Use k_cutoff (or accuracy) instead."
            )
        if misplaced:
            raise ValueError(
                f"{misplaced} only apply to method='ewald'; with "
                "method='pme' the reciprocal sum is evaluated on a mesh. "
                "Use mesh_dimensions (or mesh_spacing) instead."
            )
        if method == "pme":
            mesh_spacing, mesh_dimensions = _checked_mesh(
                mesh_spacing, mesh_dimensions, spline_order
            )
        else:
            k_cutoff = _checked_k_cutoff(alpha, k_cutoff, accuracy)
        pbc = _checked_pbc(pbc, slab_correction)

        self.alpha = alpha
        self.method = method
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

    def extra_repr(self) -> str:
        if self.method == "pme":
            return (
                f"method='pme', alpha={self.alpha}, "
                f"mesh_spacing={self.mesh_spacing}, "
                f"mesh_dimensions={self.mesh_dimensions}, "
                f"spline_order={self.spline_order}"
            )
        return f"method='ewald', alpha={self.alpha}, k_cutoff={self.k_cutoff}"

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
        """The periodicity flags as a device tensor, uploaded once per device:
        an upload from the tuple at every call is a pageable copy that blocks
        the host, one of the per-step synchronisations the MD engine removes."""
        cache = self.__dict__.setdefault("_pbc_cache", {})
        key = str(torch.device(device))
        pbc = cache.get(key)
        if pbc is None:
            pbc = torch.tensor(self.pbc, dtype=torch.bool, device=device)
            cache[key] = pbc
        return pbc

    def set_k_vectors(self, cell: torch.Tensor | None) -> None:
        """Fix the PME k-vectors for ``cell`` until the next call.

        nvalchemiops regenerates them at every evaluation, including a small
        host upload that stalls the host until the device has drained. A
        driver that evaluates the same cell step after step (the MD engine)
        computes them once here. The caller owns the cell: it must call again
        whenever the cell changes (``DistributedEvaluator.set_cell`` does), and
        pass ``None`` to go back to per-call generation. ``cell`` is
        ``(3, 3)`` or ``(1, 3, 3)`` on the term's device. A no-op for the
        Ewald sum, which builds its own vectors per call.
        """
        if cell is None or self.method != "pme":
            self._k_cache = None
            return
        self._k_cache = self._k_vectors_for(cell.detach().reshape(1, 3, 3))

    def _k_vectors_for(
        self, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """nvalchemiops' ``generate_k_vectors_pme``, without its host upload.

        The Miller-index grid is cached per mesh, device and dtype; the
        reciprocal cell is a 3 x 3 inverse written out
        (:func:`deepmd_lr.cell.inverse_3x3`); and the floor under ``k^2`` is a
        clamp rather than a scalar tensor built from a Python float. Same
        convention as nvalchemiops: ``k = h @ (2 pi inv(H))^T`` with ``H``
        rows as lattice vectors and ``h`` the Miller indices.
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
        # The unit conversion goes onto the charges (the energy is quadratic in
        # them), not onto the summed energy: nvalchemiops' backward takes its
        # read-free path only when the cotangent it receives is a plain
        # expanded constant, and a multiply after the sum makes it a tensor it
        # inspects on the host at every MD step.
        charges = charges * math.sqrt(ELECTROSTATIC_CONVERSION)
        batch_idx = torch.repeat_interleave(
            torch.arange(nf, dtype=torch.int32, device=coord.device), nloc
        )
        if self.method == "pme":
            energy = self._pme(positions, charges, cell, batch_idx)
        else:
            energy = self._ewald(positions, charges, cell, batch_idx)
        if self.slab_correction:
            # Yeh-Berkowitz, added to the 3D result as nvalchemiops prescribes.
            energy = energy + compute_slab_correction(
                positions,
                charges,
                cell,
                self._pbc_on(coord.device).expand(nf, 3),
                batch_idx=batch_idx,
                energy_reduction="atom",
            )
        return energy.reshape(nf, nloc, 1)

    def _pme(self, positions, charges, cell, batch_idx) -> torch.Tensor:
        """Per-atom ``recip - self`` on the mesh, ``(N,)`` in eV."""
        cached = self._k_cache
        return pme_reciprocal_space(
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

    def _ewald(self, positions, charges, cell, batch_idx) -> torch.Tensor:
        """Per-atom ``recip - self`` by the explicit k-vector sum, ``(N,)`` in
        eV."""
        alpha = torch.full(
            (cell.shape[0],), self.alpha, dtype=positions.dtype, device=positions.device
        )
        # The k-vectors are built from the (possibly strained) cell and stay
        # connected to it: as fixed Cartesian metadata they would give correct
        # energies and forces but a wrong cell derivative, hence virial.
        k_vectors = generate_k_vectors_ewald_summation(cell, self.k_cutoff)
        return ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            energy_reduction="atom",
        )


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

    The second sum is what :class:`LongRangeEnergy` evaluates in reciprocal
    space, here summed directly: put a cluster in a cube of side ``L`` and
    ``recip - self`` converges onto it as :math:`1/L^3` (``-6.6e-03``,
    ``-8.3e-04``, ``-1.0e-04`` eV at ``L = 80, 160, 320`` A, pinned by
    ``tests/test_isolated_term.py``). So a shared descriptor trained on
    periodic and isolated frames sees the same ``erfc(alpha r)/r`` remainder
    in both. There is no self term (``erf(alpha r)/r -> 2 alpha / sqrt(pi)``
    as ``r -> 0`` is exactly it, and ``j != i`` excludes it), nothing extra
    for a charged system (no images, no background) and no cutoff.

    ``kernel_dtype`` (a runtime attribute, not part of the config) is the
    dtype the sum runs in; ``None`` is the inputs' own.

    Parameters
    ----------
    alpha
        Ewald splitting parameter in inverse angstrom. **Must match the value
        used by the periodic branches**; it defines where the split falls.
    isolated_kernel
        The kernel (:mod:`deepmd_lr.lr.isolated`): ``"dense"`` (twice
        differentiable, O(N^2) memory: force training and Hessians),
        ``"chunked"`` (O(chunk N), first derivatives only) or ``"auto"``,
        dense up to ``max_isolated_atoms`` atoms per call.
    max_isolated_atoms
        Largest call the dense kernel takes.

    The periodic settings are accepted at their defaults -- ``dargs`` fills
    the whole block in, so an isolated config arrives carrying
    ``spline_order: 6`` nobody wrote -- and refused at any other value.
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
        misplaced = _misplaced(
            self.METHOD,
            mesh_spacing=mesh_spacing,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            k_cutoff=k_cutoff,
            accuracy=accuracy,
            slab_correction=slab_correction,
            # An all-open pbc describes this system; only a periodic axis
            # would be a setting silently ignored.
            pbc=pbc if pbc is not None and any(bool(p) for p in pbc) else None,
        ) + sorted(kwargs)
        if misplaced:
            raise ValueError(
                f"{sorted(misplaced)} do not apply to method='isolated', which "
                "takes `alpha`, `isolated_kernel` and `max_isolated_atoms`. "
                "Meshes, k-vector cutoffs, `slab_correction` and a periodic "
                "`pbc` describe a sum over periodic images, which an isolated "
                "system has none of."
            )
        self.alpha = _checked_alpha(alpha)
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
    """The long-range settings a system needs, from a model's own.

    The term holds no weights, so how the sum is taken is a property of the
    system, not of the checkpoint. ``pbc`` picks the sum: three periodic axes
    the 3D sum, two the slab with the Yeh-Berkowitz correction, none the
    isolated sum (from any trained method; ``alpha`` and the kernel settings
    carry over). The cell sets the PME mesh (:func:`system_mesh`). A wire (one
    periodic axis) is refused, and so is a periodic system for a model trained
    with ``method: isolated``, which has never seen a periodic image.
    """
    method = settings.get("method", "pme")
    periodic = list(pbc_flags(pbc))
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
