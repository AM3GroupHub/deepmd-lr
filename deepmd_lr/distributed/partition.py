# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spatial partitioning with a ghost halo, for distributed evaluation.

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
)
from itertools import (
    product,
)

import numpy as np
from scipy.spatial import (
    cKDTree,
)

__all__ = ["Partition", "partition_system"]


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
    """

    owned: np.ndarray
    ghost: np.ndarray
    ghost_shift: np.ndarray

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
        """Node coordinates: owned as-is, ghosts shifted into place."""
        return np.concatenate(
            [coord[self.owned], coord[self.ghost] + self.ghost_shift]
        )

    def types(self, atype: np.ndarray) -> np.ndarray:
        return np.concatenate([atype[self.owned], atype[self.ghost]])


def _image_shifts(cell: np.ndarray, pbc, rcut: float) -> np.ndarray:
    """Lattice translations that could bring an image within ``rcut``.

    Derived from the perpendicular width of the cell along each axis rather
    than assumed to be +-1: a thin cell needs more images, and silently
    missing one would drop real neighbours.
    """
    volume = abs(np.linalg.det(cell))
    ranges = []
    for axis in range(3):
        if not pbc[axis]:
            ranges.append([0])
            continue
        other = [cell[i] for i in range(3) if i != axis]
        area = np.linalg.norm(np.cross(*other))
        width = volume / area  # perpendicular width along this axis
        n = int(np.ceil(rcut / width))
        ranges.append(list(range(-n, n + 1)))
    return np.array(list(product(*ranges)), dtype=int)


def partition_system(
    coord: np.ndarray,
    cell: np.ndarray,
    rcut: float,
    n_partitions: int,
    pbc=(True, True, True),
) -> list[Partition]:
    """Split atoms into ``n_partitions`` slabs, each with an ``rcut`` halo.

    The split is along the longest cell vector, by fractional coordinate. That
    is cruder load balancing than DistMLIP's graph partitioning, but the
    correctness contract is identical and it has no compiled dependency; the
    partitioner can be swapped later without touching anything downstream.

    Parameters
    ----------
    coord
        ``(N, 3)`` Cartesian coordinates.
    cell
        ``(3, 3)`` cell vectors as rows.
    rcut
        Descriptor cutoff; sets the halo width.
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

    # Assign owners by fractional coordinate along the longest cell vector.
    inverse = np.linalg.inv(cell)
    frac = coord @ inverse
    axis = int(np.argmax(np.linalg.norm(cell, axis=1)))
    key = frac[:, axis]
    if pbc[axis]:
        key = key - np.floor(key)
    else:
        lo, hi = key.min(), key.max()
        key = (key - lo) / (hi - lo + 1e-12)
    # Equal-count slabs rather than equal-width: atom density is not uniform,
    # and for a slab system with vacuum equal-width would leave empty
    # partitions.
    order = np.argsort(key, kind="stable")
    owner = np.empty(n_atoms, dtype=int)
    for p, chunk in enumerate(np.array_split(order, n_partitions)):
        owner[chunk] = p

    # No shortcut for a single partition: it still owns every atom but still
    # needs periodic images as ghosts, because the subgraph is built
    # non-periodically. Returning an empty halo here silently dropped every
    # interaction across the boundary.

    # Every periodic image that could reach into some partition.
    shifts = _image_shifts(cell, pbc, rcut)
    offsets = shifts @ cell
    images = (coord[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    image_atom = np.tile(np.arange(n_atoms), len(shifts))
    image_shift = np.repeat(offsets, n_atoms, axis=0)
    tree = cKDTree(images)

    partitions = []
    for p in range(n_partitions):
        owned = np.flatnonzero(owner == p)
        # Anything within rcut of an owned atom is needed to complete that
        # atom's neighbour shell.
        near = tree.query_ball_point(coord[owned], r=rcut)
        candidates = np.unique(np.concatenate([np.asarray(x, dtype=int)
                                               for x in near])) if len(near) else \
            np.empty(0, dtype=int)
        # Drop the owned atoms' own unshifted copies; a shifted copy of an
        # owned atom is a legitimate ghost when the cell is small.
        is_self = (
            np.isin(image_atom[candidates], owned)
            & (np.abs(image_shift[candidates]).sum(axis=1) == 0)
        )
        ghosts = candidates[~is_self]
        partitions.append(
            Partition(
                owned=owned,
                ghost=image_atom[ghosts],
                ghost_shift=image_shift[ghosts],
            )
        )
    return partitions
