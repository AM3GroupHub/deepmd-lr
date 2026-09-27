# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spatial partitioning with a ghost halo, for partitioned evaluation.

Follows DistMLIP's design -- one process, several devices, an owned/halo split
per partition -- rather than its code: DistMLIP's partitioner is a C extension
emitting a graph format built for multi-layer message-passing networks, and
DPA4C is strictly one-hop. One halo of width ``rcut`` is all it needs, with no
per-layer exchange, so a partitioner that emits DeePMD-kit ``NeighborGraph``
objects directly is both smaller and better matched.

Two invariants make the assembled result exact rather than approximate.

**Every edge belongs to exactly one partition.** A subgraph keeps only the
edges whose *destination* is an owned atom. An edge appears in exactly one
partition -- the one owning its destination -- so summing over partitions
counts it once. Keeping ghost-to-ghost edges would double-count them, because
the partition owning those atoms holds the same edge.

**Ghost forces belong to their owner.** Edge ``(src, dst)`` pushes on both
ends, so a partition computes force contributions for its ghosts too. Those
are scattered back onto the owning partition's atoms, exactly as a
domain-decomposed MD code reverse-communicates ghost forces.

The subgraph itself is built **non-periodically**: the halo already carries
explicit periodic images, so wrapping inside a partition would double-apply
the boundary.
"""

from dataclasses import (
    dataclass,
    field,
)
from itertools import (
    product,
)

import numpy as np
import torch
from scipy.spatial import (
    cKDTree,
)

__all__ = [
    "Partition",
    "decomposition_grid",
    "default_skin",
    "halo_node_ratio",
    "partition_system",
    "skin_for_node_ratio",
]


@dataclass
class Partition:
    """One partition's atoms: owned first, then ghosts.

    Attributes
    ----------
    owned
        Global atom indices this partition owns, in node order ``[0, n_owned)``.
    ghost
        Global atom indices appearing as ghosts, in node order
        ``[n_owned, n_node)``. An index may repeat if several periodic images
        of the same atom are in range.
    ghost_shift
        Cartesian offset applied to each ghost, shape ``(n_ghost, 3)``.
    owned_shift
        Cartesian offset applied to each owned atom, shape ``(n_owned, 3)``;
        zeros unless the coordinates handed to :func:`partition_system` were
        unwrapped. Both shifts are relative to those coordinates: a node sits
        at ``coord[atom] + shift``, folded into the cell, however far the atom
        itself has wandered.
    ghost_image, owned_image
        The integer lattice vectors behind the shifts, ``shift = image @ cell``.
        With them :meth:`set_cell` moves the nodes to a deformed cell without
        repartitioning; a barostat deforms it a little every step.
    """

    owned: np.ndarray
    ghost: np.ndarray
    ghost_shift: np.ndarray
    owned_shift: np.ndarray | None = None
    ghost_image: np.ndarray | None = None
    owned_image: np.ndarray | None = None
    #: Device-resident index tensors, built once per device (see :meth:`views`).
    _views: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.owned_shift is None:
            self.owned_shift = np.zeros((len(self.owned), 3))

    @property
    def n_owned(self) -> int:
        return len(self.owned)

    @property
    def n_node(self) -> int:
        return len(self.owned) + len(self.ghost)

    def global_ids(self) -> np.ndarray:
        """Global atom index of every node, owned then ghost."""
        return np.concatenate([self.owned, self.ghost])

    def coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Node coordinates, owned then ghost, each shifted into place."""
        return np.concatenate([coord[self.owned] + self.owned_shift,
                               coord[self.ghost] + self.ghost_shift])

    def types(self, atype: np.ndarray) -> np.ndarray:
        return np.concatenate([atype[self.owned], atype[self.ghost]])

    def set_cell(self, cell: np.ndarray) -> None:
        """Re-express the node shifts in a new cell, keeping the nodes.

        The nodes stay the same atoms and lattice images; only their Cartesian
        offsets follow the cell, host copies and cached device views alike.
        Whether the halo still covers every neighbour after the change is the
        caller's to judge: the evaluator's Verlet bound counts the cell's
        change as well as the atoms'.
        """
        if self.ghost_image is None or self.owned_image is None:
            raise ValueError(
                "this partition carries no lattice images; rebuild it for the "
                "new cell instead"
            )
        cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        self.ghost_shift = self.ghost_image @ cell
        self.owned_shift = self.owned_image @ cell
        for v in self._views.values():
            if isinstance(v, dict) and "ghost_image" in v:
                shift = v["shift"]
                h = torch.as_tensor(cell, dtype=shift.dtype, device=shift.device)
                shift.copy_(v["ghost_image"] @ h)
                v["owned_shift"].copy_(v["owned_image"] @ h)

    def views(self, device, dtype=torch.float64) -> dict:
        """This partition's index tensors on ``device``, built once and cached.

        ``owned`` and ``ghost`` are global atom indices, ``ids`` their
        concatenation in node order, ``shift`` and ``owned_shift`` the ghosts'
        and the owned atoms' Cartesian offsets.
        The partitioning is reused across many steps, so uploading these
        every step would be the kind of per-step host traffic the device
        gather exists to remove.
        """
        key = (str(torch.device(device)), dtype)
        v = self._views.get(key)
        if v is None:
            dev = torch.device(device)
            ids = torch.as_tensor(self.global_ids(), dtype=torch.int64, device=dev)
            v = {
                "ids": ids,
                "owned": ids[: self.n_owned],
                "ghost": ids[self.n_owned :],
                "shift": torch.as_tensor(
                    np.asarray(self.ghost_shift, dtype=np.float64).reshape(-1, 3),
                    dtype=dtype,
                    device=dev,
                ),
                "owned_shift": torch.as_tensor(
                    np.asarray(self.owned_shift, dtype=np.float64).reshape(-1, 3),
                    dtype=dtype,
                    device=dev,
                ),
            }
            if self.ghost_image is not None and self.owned_image is not None:
                for name, image in (
                    ("ghost_image", self.ghost_image),
                    ("owned_image", self.owned_image),
                ):
                    v[name] = torch.as_tensor(
                        np.asarray(image).reshape(-1, 3), dtype=dtype, device=dev
                    )
            self._views[key] = v
        return v

    def pairs_on(self, device) -> dict | None:
        """This partition's cached pair list on ``device``, if one is live.

        Built by :func:`~deepmd_lr.nvalchemi.driver.partition_graph` at
        ``rcut`` plus a skin and filtered to ``rcut`` every step; the
        evaluator drops it (:meth:`drop_pairs`) once the atoms or the cell
        have moved far enough to invalidate it.
        """
        return self._views.get(("pairs", str(torch.device(device))))

    def store_pairs(self, device, pairs: dict) -> None:
        self._views[("pairs", str(torch.device(device)))] = pairs

    def drop_pairs(self) -> None:
        for key in [k for k in self._views if isinstance(k, tuple) and k[0] == "pairs"]:
            del self._views[key]

    def types_on(self, atype: np.ndarray, device) -> torch.Tensor:
        """Node types on ``device``, cached (the type of an atom never changes)."""
        key = ("types", str(torch.device(device)))
        t = self._views.get(key)
        if t is None:
            t = torch.as_tensor(
                self.types(atype), dtype=torch.int64, device=torch.device(device)
            )
            self._views[key] = t
        return t

    def counts_on(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(n_node, n_local)`` as one-element device tensors, built once.

        The model's lower interface takes them as tensors. Making them from
        Python ints every step is an upload from pageable memory, and that
        blocks the host until everything already queued on ``device`` has
        run -- twice per partition per step, for two integers that only change
        when the partition does.
        """
        key = ("counts", torch.device(device))
        c = self._views.get(key)
        if c is None:
            c = (
                torch.tensor([self.n_node], dtype=torch.int64, device=device),
                torch.tensor([self.n_owned], dtype=torch.int64, device=device),
            )
            self._views[key] = c
        return c


def _image_shifts(cell: np.ndarray, pbc, rcut: float) -> np.ndarray:
    """Lattice translations that could bring an image within ``rcut``.

    Derived from the perpendicular width of the cell along each axis rather
    than assumed to be +-1: a thin cell needs more images, and silently
    missing one would drop real neighbours.
    """
    widths = _perpendicular_widths(cell)
    ranges = []
    for axis in range(3):
        if not pbc[axis]:
            ranges.append([0])
            continue
        n = int(np.ceil(rcut / widths[axis]))
        ranges.append(list(range(-n, n + 1)))
    return np.array(list(product(*ranges)), dtype=int)


def _perpendicular_widths(cell: np.ndarray) -> np.ndarray:
    """The cell's width perpendicular to each pair of faces: ``V / |a_j x a_k|``."""
    volume = abs(np.linalg.det(cell))
    return np.array(
        [
            volume / np.linalg.norm(np.cross(cell[(j + 1) % 3], cell[(j + 2) % 3]))
            for j in range(3)
        ]
    )


def decomposition_grid(
    cell: np.ndarray, n_partitions: int, rcut: float
) -> tuple[int, int, int]:
    """How many pieces each cell axis is cut into, ``(n0, n1, n2)``.

    The choice that minimises one partition's halo box, ``prod(l_i / n_i +
    2 rcut)``: what a device has to hold is its owned block *plus* a shell of
    ghosts `rcut` deep on every side, and the shell is what does not shrink
    when the block does. Cutting one axis into thin slabs keeps the other two
    faces full size, so at eight partitions of 98,304 atoms each device holds
    about 2.8 times its share; cutting three axes in half, about 2.3 times. A
    count that only factors one way -- a prime -- can only be cut into slabs.

    Ties -- a cube cut in two, say -- go to the grid that cuts the lowest
    axis first.
    """
    n = max(int(n_partitions), 1)
    lengths = np.linalg.norm(np.asarray(cell, dtype=float).reshape(3, 3), axis=1)
    best = None
    for n0 in range(1, n + 1):
        if n % n0:
            continue
        for n1 in range(1, n // n0 + 1):
            if (n // n0) % n1:
                continue
            grid = (n0, n1, n // n0 // n1)
            cost = float(np.prod(lengths / np.array(grid) + 2.0 * rcut))
            key = (round(cost, 6), -grid[0], -grid[1])
            if best is None or key < best[0]:
                best = (key, grid)
    return best[1]


def _halo_edges(cell: np.ndarray, rcut: float, n_partitions: int) -> np.ndarray:
    """The three edges of one partition's halo box at zero skin."""
    lengths = np.linalg.norm(np.asarray(cell, dtype=float).reshape(3, 3), axis=1)
    grid = decomposition_grid(cell, n_partitions, rcut)
    return lengths / np.array(grid, dtype=float) + 2.0 * rcut


def _largest_skin(admissible, upper_guess: float) -> float:
    """The largest non-negative skin ``admissible`` accepts, by bisection.

    ``admissible`` must be monotone: true below some skin and false above it.
    Bracket first, then bisect. A fixed upper bound is not a bracket: for a
    very anisotropic cell -- edges [100, 100, 1], say -- even a skin of half
    the shortest edge can stay well inside the bound, and returning that cap
    would hand back an arbitrary number rather than the largest admissible one.
    The halo volume grows without bound in the skin, so doubling terminates.
    """
    if not admissible(0.0):
        return 0.0
    lo, hi = 0.0, max(float(upper_guess), 1.0)
    for _ in range(64):
        if not admissible(hi):
            break
        lo, hi = hi, 2.0 * hi
    else:  # pragma: no cover - the volume is cubic in s; 2**64 cannot survive
        raise RuntimeError("no skin is large enough to violate the bound")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if admissible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def halo_node_ratio(
    cell: np.ndarray,
    rcut: float,
    n_partitions: int,
    skin: float,
    skin_from: float = 0.0,
) -> float:
    """How many times as many nodes a halo at ``skin`` holds as one at
    ``skin_from``.

    Nodes are density times volume, and the density cancels in the ratio, which
    is why this needs no atom count. Its use is the other direction: multiply
    by a node count someone has actually *counted* and the answer is an
    absolute prediction for a different skin -- which is what a memory budget
    has to be written in. See
    :meth:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator._fit_skin_to_memory`.
    """
    base = _halo_edges(cell, rcut, n_partitions)
    return float(np.prod((base + 2.0 * skin) / (base + 2.0 * skin_from)))


def skin_for_node_ratio(
    cell: np.ndarray,
    rcut: float,
    n_partitions: int,
    ratio: float,
    skin_from: float = 0.0,
) -> float:
    """The largest skin whose halo stays within ``ratio`` of the one at
    ``skin_from`` -- the inverse of :func:`halo_node_ratio`."""
    base = _halo_edges(cell, rcut, n_partitions)
    return _largest_skin(
        lambda s: halo_node_ratio(cell, rcut, n_partitions, s, skin_from) <= ratio,
        upper_guess=float(base.min()),
    )


def default_skin(
    cell: np.ndarray,
    rcut: float,
    n_partitions: int = 1,
    tolerance: float = 0.25,
) -> float:
    r"""The skin a partitioned run should use when the caller does not pick one.

    The halo is rebuilt whenever an atom has moved more than half the skin, so
    a larger skin trades ghost work for a rarer rebuild. Which way that trade
    goes is not a constant: the rebuild is a host-side spatial sort whose cost
    grows with the atom count, while the *relative* cost of a thicker halo
    shrinks as the partition gets larger. A fixed default is therefore wrong at
    both ends -- measured on 2 A40s at 98,304 atoms, the old default of 1 A
    spent **76 % of every step** rebuilding (3.9 s per rebuild, one every 13
    steps) and capped the two-GPU speed-up at 1.14x; at 4 A the same step ran
    3.4x faster and the ceiling rose to 1.9x. On a few thousand atoms the
    ordering reverses and a thick halo is the waste.

    So the rule scales with the geometry instead: take the largest skin whose
    halo costs at most ``tolerance`` more nodes than a zero-skin halo would.
    For a partition with edges :math:`l_i`,

    .. math::
        \prod_i \frac{l_i + 2(r_c + s)}{l_i + 2 r_c} \le 1 + \text{tolerance}

    which is monotone in :math:`s` and solved here by bisection. It returns
    about 1.4 A for 1,536 atoms, 4.3 A for 98,304 and 5.3 A for 192,000 -- and
    4.3 A is where the 98k measurement put the optimum.

    The estimate treats the cell as orthorhombic and the density as uniform,
    which is what a default needs to be: cheap and roughly right. Pass an
    explicit ``skin`` when the system is neither.

    What it deliberately does not model is memory. Bytes per node belong to
    the model, not the geometry -- an uncompressed descriptor holds several
    times what a compressed one does -- so
    :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator` caps this
    value by measurement instead: a first-step probe of the cost per node, and
    a halving whenever a device runs out anyway.
    """
    return skin_for_node_ratio(cell, rcut, n_partitions, 1.0 + float(tolerance))


def partition_system(
    coord: np.ndarray,
    cell: np.ndarray,
    rcut: float,
    n_partitions: int,
    pbc=(True, True, True),
    skin: float = 0.0,
) -> list[Partition]:
    """Split atoms into ``n_partitions`` blocks, each with an ``rcut`` halo.

    The blocks form a grid, :func:`decomposition_grid` pieces per cell axis,
    chosen to make each block's halo as small as the count allows -- slabs for
    two partitions, 2 x 2 x 1 for four and 2 x 2 x 2 for eight on a cubic box.
    Each cut is by fractional coordinate at equal atom count. That is cruder
    load balancing than DistMLIP's graph partitioning, but the correctness
    contract is identical and it has no compiled dependency; the partitioner
    can be swapped later without touching anything downstream.

    Parameters
    ----------
    coord
        ``(N, 3)`` Cartesian coordinates, wrapped or not: they are folded into
        the cell on periodic axes before anything is built.
    cell
        ``(3, 3)`` cell vectors as rows.
    rcut
        Descriptor cutoff; sets the halo width.
    skin
        Extra halo width. Ghosts are collected within ``rcut + skin`` so the
        partition stays valid while atoms move, which is what lets an MD run
        reuse it across steps. Edges are still built at ``rcut``, so a skin
        changes cost but never the answer.
    n_partitions
        Number of partitions. ``1`` yields a single partition with no ghosts.
    pbc
        Per-axis periodicity.

    Returns
    -------
    list[Partition]
    """
    if n_partitions < 1:
        raise ValueError(f"n_partitions must be >= 1, got {n_partitions}")
    n_atoms = len(coord)
    if n_partitions > n_atoms:
        raise ValueError(
            f"{n_partitions} partitions requested for only {n_atoms} atoms"
        )

    inverse = np.linalg.inv(cell)
    # Fold every atom into the cell on periodic axes. Integrators never wrap,
    # so atoms that leave through opposite faces end up many cells apart, and
    # the +-ceil(halo/width) images below only bridge a spread of about one
    # halo: built from the raw coordinates, the halo silently lost neighbours
    # (1.1 eV/A on water at a spread of L + 12 A). Everything is built from the
    # folded positions, and the nodes are expressed back relative to the
    # coordinates handed in -- `owned_shift` and `ghost_shift` -- so callers
    # keep integrating unwrapped positions and nothing else changes.
    image = np.where(
        np.asarray(pbc, dtype=bool), np.floor(coord @ inverse), 0.0
    ).astype(np.int64)
    fold = image @ cell
    coord = coord - fold
    # Assign owners on a grid of blocks, `decomposition_grid` pieces per axis.
    frac = coord @ inverse
    keys = np.empty_like(frac)
    for axis in range(3):
        key = frac[:, axis]
        if pbc[axis]:
            key = key - np.floor(key)
        else:
            lo, hi = key.min(), key.max()
            key = (key - lo) / (hi - lo + 1e-12)
        keys[:, axis] = key
    # Equal-count blocks rather than equal-width: atom density is not uniform,
    # and for a slab system with vacuum equal-width would leave empty
    # partitions. Cut one axis at a time, each group by its own sort, so the
    # counts stay equal however the density varies.
    grid = decomposition_grid(cell, n_partitions, rcut)
    groups = [np.arange(n_atoms)]
    for axis in range(3):
        if grid[axis] == 1:
            continue
        groups = [
            piece
            for group in groups
            for piece in np.array_split(
                group[np.argsort(keys[group, axis], kind="stable")], grid[axis]
            )
        ]
    owner = np.empty(n_atoms, dtype=int)
    for p, members in enumerate(groups):
        owner[members] = p

    # Every periodic image that could reach into some partition. A single
    # partition needs them too: it owns every atom, but its subgraph is built
    # non-periodically, so its ghosts are the only interactions across the
    # boundary.
    halo = rcut + skin
    shifts = _image_shifts(cell, pbc, halo)
    offsets = shifts @ cell
    images = (coord[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    # Image `i` is atom `i % n_atoms` under lattice shift `shifts[i // n_atoms]`.
    image_atom = np.tile(np.arange(n_atoms), len(shifts))
    # Fractional coordinates of every image, and how far a Euclidean step of
    # `halo` can move each fractional coordinate: the distance between the
    # planes f_j = c and f_j = c + d is d times the cell's perpendicular width
    # along j, so no point within `halo` of x differs from it by more than
    # `halo / width_j` in f_j.
    image_frac = images @ inverse
    reach = halo / _perpendicular_widths(cell)
    # Upper bound handed to the nearest-neighbour query. `query` keeps only
    # distances strictly below its bound, so nudge it by one ulp: the halo is
    # a closed ball, and an image at exactly `halo` is a ghost.
    bound = np.nextafter(halo, np.inf)

    partitions = []
    for p in range(n_partitions):
        owned = np.flatnonzero(owner == p)
        # Anything within `halo` of an owned atom is needed to complete that
        # atom's neighbour shell. Two passes, both without Python-level
        # neighbour lists (which is what made this the slowest thing in a
        # rebuild: millions of list objects at 98,304 atoms):
        #   1. keep images inside the owned atoms' fractional bounding box
        #      grown by `reach` -- a superset, by the bound above;
        #   2. of those, keep the ones whose NEAREST owned atom is within
        #      `halo`, which is exactly the set a ball query would return.
        lo = frac[owned].min(axis=0) - reach
        hi = frac[owned].max(axis=0) + reach
        boxed = np.flatnonzero(
            np.all((image_frac >= lo) & (image_frac <= hi), axis=1)
        )
        if len(owned) and len(boxed):
            nearest, _ = cKDTree(coord[owned]).query(
                images[boxed], k=1, distance_upper_bound=bound, workers=-1
            )
            candidates = boxed[np.isfinite(nearest)]
        else:
            candidates = np.empty(0, dtype=int)
        # Drop the owned atoms' own unshifted copies; a shifted copy of an
        # owned atom is a legitimate ghost when the cell is small.
        is_self = np.isin(image_atom[candidates], owned) & np.all(
            shifts[candidates // n_atoms] == 0, axis=1
        )
        ghosts = candidates[~is_self]
        # Relative to the coordinates handed in, not the folded ones, and as
        # integer lattice vectors, so a deformed cell can re-express them.
        ghost_image = shifts[ghosts // n_atoms] - image[image_atom[ghosts]]
        owned_image = -image[owned]
        partitions.append(
            Partition(
                owned=owned,
                ghost=image_atom[ghosts],
                ghost_shift=ghost_image @ cell,
                owned_shift=owned_image @ cell,
                ghost_image=ghost_image,
                owned_image=owned_image,
            )
        )
    return partitions
