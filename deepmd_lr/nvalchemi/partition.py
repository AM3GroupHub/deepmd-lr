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
    #: Device-resident index tensors, built once per device (see :meth:`views`).
    _views: dict = field(default_factory=dict, repr=False, compare=False)

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
        return np.concatenate([coord[self.owned], coord[self.ghost] + self.ghost_shift])

    def types(self, atype: np.ndarray) -> np.ndarray:
        return np.concatenate([atype[self.owned], atype[self.ghost]])

    def views(self, device, dtype=torch.float64) -> dict:
        """This partition's index tensors on ``device``, built once and cached.

        ``owned`` and ``ghost`` are global atom indices, ``ids`` their
        concatenation in node order, ``shift`` the ghosts' Cartesian offsets.
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
            }
            self._views[key] = v
        return v

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
    value by measurement instead, halving it whenever a device runs out.
    """
    lengths = np.linalg.norm(np.asarray(cell, dtype=float).reshape(3, 3), axis=1)
    # `partition_system` slabs along the longest cell vector, so that edge --
    # and only that one -- is divided among the partitions.
    lengths[int(np.argmax(lengths))] /= max(int(n_partitions), 1)
    base = lengths + 2.0 * rcut

    def inflation(s: float) -> float:
        return float(np.prod((base + 2.0 * s) / base))

    ceiling = 1.0 + float(tolerance)
    if ceiling <= 1.0:
        return 0.0
    # Bracket first, then bisect. A fixed upper bound is not a bracket: for a
    # very anisotropic cell -- edges [100, 100, 1], say -- even a skin of half
    # the shortest edge stays well inside the tolerance, and returning that cap
    # would hand back an arbitrary number rather than the largest admissible
    # one. `inflation` grows without bound in ``s``, so doubling terminates.
    lo, hi = 0.0, max(float(lengths.min()), 1.0)
    for _ in range(64):
        if inflation(hi) > ceiling:
            break
        lo, hi = hi, 2.0 * hi
    else:  # pragma: no cover - `inflation` is cubic in s; 2**64 cannot survive
        raise RuntimeError(f"no skin reaches an inflation of {ceiling}")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if inflation(mid) <= ceiling:
            lo = mid
        else:
            hi = mid
    return lo


def partition_system(
    coord: np.ndarray,
    cell: np.ndarray,
    rcut: float,
    n_partitions: int,
    pbc=(True, True, True),
    skin: float = 0.0,
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
    halo = rcut + skin
    shifts = _image_shifts(cell, pbc, halo)
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
        near = tree.query_ball_point(coord[owned], r=halo)
        candidates = (
            np.unique(np.concatenate([np.asarray(x, dtype=int) for x in near]))
            if len(near)
            else np.empty(0, dtype=int)
        )
        # Drop the owned atoms' own unshifted copies; a shifted copy of an
        # owned atom is a legitimate ghost when the cell is small.
        is_self = np.isin(image_atom[candidates], owned) & (
            np.abs(image_shift[candidates]).sum(axis=1) == 0
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
