# SPDX-License-Identifier: LGPL-3.0-or-later
"""A neighbour list that lives on the device: nvalchemiops' cell list, compacted
into DeePMD-kit's edge graph by two small Warp kernels, with no host round trip.

The partitioned evaluator used to build its graph through DeePMD-kit's
``build_neighbor_graph_nv``: the same nvalchemiops search, then a dense-matrix
decode, float64 vectors for every candidate pair, two stable argsorts and a
per-step ``nonzero``/``bincount`` filter. On one H200 at 1.96M water atoms that
was 649 ms per search and 61 ms per step for the graph, where LAMMPS-Kokkos
spends 9.7 and 17.7, and every step read the host about ten times. The kernel
was never the problem; what surrounded it was.

This module keeps the kernel and replaces the surroundings:

- **Preallocated buffers, updated in place.** The cell-list cache, the
  ``(rows, M)`` neighbour matrix and the edge buffers are allocated once and
  resized only when a device-side flag says they overflowed.
- **The search runs every step, at ``rcut``.** No skin, no reference
  positions, no Verlet bound, no fold: ``build_cell_list`` and
  ``query_cell_list`` are cheap enough (0.96 ms at 10k atoms, 56 ms at 1.96M
  on a GB10 whose memory is some fifteen times slower than an H200's) that the
  stateless search costs less than the bookkeeping a reusable list needs.
  The brief that led here asked for both variants to be measured; the Verlet
  variant is added only if the per-step query turns out to matter.
- **The graph comes straight from the matrix.** Row ``i`` of the neighbour
  matrix already holds centre ``i``'s neighbours, so the destination CSR is a
  prefix sum of the row counts, and one Warp kernel compacts the rows into
  DeePMD-kit's ``edge_index``. The source CSR -- the one reorder a dual-CSR
  graph cannot avoid -- is built the way LAMMPS-Kokkos builds it
  (``CompactCanonicalGraphKokkos::build``): a per-node count, a prefix sum,
  and an atomic cursor that scatters each edge into its source row. No sort.
- **Nothing reads the host.** Edge counts stay device scalars; the edge axis
  has a fixed capacity and ``edge_mask`` marks where the real edges end. Only
  the first search sizes the buffers, and that one does synchronise.

Conventions, which are nvalchemiops' and DeePMD-kit's at once: the matrix row
is the centre ``i``, its entries are neighbours ``j`` with integer image
shifts ``S``, and the pair vector is ``x[j] - x[i] + S @ cell`` (cell rows are
lattice vectors). That is DeePMD-kit's ``edge_index = [src=j, dst=i]`` with
``edge_vec = r_src - r_dst``, so the shifts pass through unchanged.
nvalchemiops accepts unwrapped positions and folds the images into ``S``
itself, so an MD trajectory never has to be wrapped for this list.

The graph is a full directed list (both directions of every pair), as the
descriptor expects. Row order within a destination row follows the query
kernel, which uses atomics and is therefore not bit-stable from one call to
the next; order within a source row follows the cursor, likewise. Both move
results by summation-order rounding only, exactly as today's path does at
every rebuild.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import warp as wp
from deepmd.dpmodel.utils.neighbor_graph.graph import NeighborGraph
from nvalchemiops.torch.neighbors.cell_list import (
    build_cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list

__all__ = ["GpuNeighborList", "STATUS_MATRIX_OVERFLOW", "STATUS_EDGE_OVERFLOW"]

#: Index into :attr:`GpuNeighborList.status`: a row had more neighbours than
#: the matrix holds. The stored slots are complete up to ``M``; the rest were
#: dropped by the kernel, so the step that raised this must be redone with a
#: wider matrix.
STATUS_MATRIX_OVERFLOW = 0
#: Index into :attr:`GpuNeighborList.status`: the edges within ``rcut`` did
#: not fit the edge buffers. The prefix that fits is intact; the step must be
#: redone with a larger capacity.
STATUS_EDGE_OVERFLOW = 1
#: Length of the status word. The two spare slots are for the evaluator's own
#: device-side checks (a slab atom past an open face, for one).
STATUS_LEN = 4

#: Margin added on every side of a non-periodic node set's bounding box before
#: it becomes the cell list's box. Atoms outside the box are clamped into its
#: edge cells by the kernel, which stays correct but piles atoms into those
#: cells; a margin keeps the binning even as ghosts move between rebuilds.
_BOX_MARGIN_FACTOR = 0.5

#: Below this a cell list's search radius depends on the box in a way that a
#: barostat can invalidate (see :meth:`GpuNeighborList.cell_changed`).
_SMALL_BOX_CELLS = 4


def _torch_stream_scope(device: torch.device):
    """Make Warp launch on PyTorch's current stream for the block.

    Warp has its own default stream per device. It happens to be a blocking
    stream, which the legacy default stream PyTorch uses orders against, so
    today everything works without this -- by accident. Once no host sync
    separates the search from the kernels that consume it, the ordering has
    to be explicit, and a caller running on a side stream must not be able to
    race the model against the list. nvalchemiops' own wrappers do not scope
    their launches, so this wraps them too.
    """
    if device.type != "cuda":
        return nullcontext()
    torch_stream = torch.cuda.current_stream(device)
    wp_stream = wp.get_stream(str(device))
    if wp_stream.cuda_stream == torch_stream.cuda_stream:
        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch_stream))


# ----------------------------------------------------------------------------
# The two kernels
# ----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _fill_edges(
    pos: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    nbr: wp.array2d(dtype=wp.int32),
    shifts: wp.array2d(dtype=wp.vec3i),
    counts: wp.array(dtype=wp.int32),
    centre: wp.array(dtype=wp.int32),
    row_ptr: wp.array(dtype=wp.int64),
    cap: wp.int32,
    src: wp.array(dtype=wp.int64),
    dst: wp.array(dtype=wp.int64),
    edge_shift: wp.array(dtype=wp.vec3i),
    edge_vec: wp.array(dtype=wp.vec3f),
    edge_vec64: wp.array(dtype=wp.vec3d),
    want64: wp.int32,
    src_count: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int64),
):
    """One thread per matrix slot: copy the slot into its edge.

    Row ``r`` is centre ``centre[r]`` and its first ``counts[r]`` slots are
    valid, so slot ``k`` is edge ``row_ptr[i] + k`` -- the query's order
    within the row is kept and no scan is needed. Each edge records its
    source, destination, image shift and the float32 pair vector, and bumps
    its source node's count for the source CSR built afterwards. Writes past
    the capacity are skipped and reported; the prefix that fits is intact.
    """
    r, k = wp.tid()
    if k >= counts[r]:
        return
    i = centre[r]
    e64 = row_ptr[i] + wp.int64(k)
    if e64 >= wp.int64(cap):
        wp.atomic_max(status, 1, wp.int64(1))
        return
    e = int(e64)
    j = nbr[r, k]
    s = shifts[r, k]
    sh = wp.vec3d(wp.float64(s[0]), wp.float64(s[1]), wp.float64(s[2]))
    v = pos[j] - pos[i] + wp.transpose(cell[0]) * sh
    src[e] = wp.int64(j)
    dst[e] = wp.int64(i)
    edge_shift[e] = s
    edge_vec[e] = wp.vec3f(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))
    if want64 != 0:
        edge_vec64[e] = v
    wp.atomic_add(src_count, j, 1)


@wp.kernel(enable_backward=False)
def _source_order(
    src: wp.array(dtype=wp.int64),
    dst: wp.array(dtype=wp.int64),
    edge_shift: wp.array(dtype=wp.vec3i),
    edge_vec: wp.array(dtype=wp.vec3f),
    edge_vec64: wp.array(dtype=wp.vec3d),
    want64: wp.int32,
    n_real: wp.array(dtype=wp.int64),
    cursor: wp.array(dtype=wp.int32),
    source_order: wp.array(dtype=wp.int64),
):
    """One thread per edge slot: place real edges in their source rows.

    ``cursor`` starts as a copy of the source row pointers; an atomic add
    hands each edge the next free position in its source's row. Slots past
    the real edges map to themselves, so ``source_order`` stays a permutation
    of the whole capacity, and the first two of them are zeroed into the
    guard edges DeePMD-kit's builders append.
    """
    e = wp.tid()
    n = n_real[0]
    e64 = wp.int64(e)
    if e64 < n:
        j = int(src[e])
        slot = wp.atomic_add(cursor, j, 1)
        source_order[slot] = e64
    else:
        source_order[e] = e64
        if e64 < n + wp.int64(2):
            src[e] = wp.int64(0)
            dst[e] = wp.int64(0)
            edge_shift[e] = wp.vec3i(0, 0, 0)
            edge_vec[e] = wp.vec3f(0.0, 0.0, 0.0)
            if want64 != 0:
                edge_vec64[e] = wp.vec3d(0.0, 0.0, 0.0)


def _round_up(value: float, multiple: int = 16, floor: int = 16) -> int:
    return max(int(math.ceil(value / multiple)) * multiple, floor)


def _times_3x3(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``v @ m`` for ``(E, 3)`` by ``(3, 3)``, in ``v``'s dtype.

    In float32 the GEMM is fine. In float64 a tall-skinny GEMM with K = 3
    lands on a bad kernel (51 ms for 19M rows on a GB10, 20 ms for the
    broadcast product below, whose backward is as cheap), and the long-range
    path runs this once for the shifts and once for the strain every step.
    """
    m = m.to(v.dtype)
    if v.dtype == torch.float32:
        return v @ m
    return (v[:, :, None] * m[None]).sum(1)


class _Strained(torch.autograd.Function):
    """``vec @ strain^T`` with a backward written out.

    Autograd's own backward for this product includes the strain gradient
    ``vec^T @ g``, a ``(3, E) x (E, 3)`` reduction GEMM that cuBLAS handles
    badly at tens of millions of edges (600 ms at 19M on a GB10). Three row
    reductions do the same in a few passes, and the vector gradient is one
    more product like the forward.
    """

    @staticmethod
    def forward(ctx, vec: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(vec, strain)
        return _times_3x3(vec, strain.transpose(-1, -2))

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        vec, strain = ctx.saved_tensors
        grad_vec = _times_3x3(g, strain)
        # d/d strain[a, b] of sum_e g[e, a] vec_s[e, a] is sum_e g[e, a] vec[e, b].
        w = torch.stack([(g[:, a : a + 1] * vec).sum(0) for a in range(3)])
        return grad_vec, w.to(strain.dtype)


class _EdgeVectors(torch.autograd.Function):
    """The kernel's float64 pair vectors as a function of the node coordinates.

    The fill kernel already computed ``x[j] - x[i] + S @ H`` in float64 for
    every edge; the forward hands that buffer out, and the backward is the
    exact adjoint of the two gathers, a scatter-add of the cotangent onto the
    source and its negative onto the destination. This replaces two float64
    gathers, a cast and a matmul over the whole edge axis (37 ms at 1.96M
    atoms on an H200) with nothing in the forward and two ``index_add_`` in
    the backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, nlist) -> torch.Tensor:
        ctx.save_for_backward(nlist.edge_index[0], nlist.edge_index[1])
        ctx.n_node = x.shape[0]
        return nlist.edge_vec64

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        src, dst = ctx.saved_tensors
        gx = torch.zeros((ctx.n_node, 3), dtype=grad.dtype, device=grad.device)
        gx.index_add_(0, src, grad)
        gx.index_add_(0, dst, grad, alpha=-1.0)
        return gx, None


# ----------------------------------------------------------------------------
# The list
# ----------------------------------------------------------------------------


class GpuNeighborList:
    """The neighbour graph of one node set on one device, rebuilt every step.

    Parameters
    ----------
    n_node
        Nodes in the set (atoms, or owned atoms plus ghosts).
    rcut
        The cutoff. Pairs strictly closer than it become edges, as in
        nvalchemiops and LAMMPS-Kokkos; DeePMD-kit's dense oracle keeps the
        boundary too, which no finite-precision trajectory ever hits.
    device
        A CUDA device.
    cell
        For a periodic node set, a ``(1, 3, 3)`` float64 device tensor whose
        rows are the lattice vectors. The list keeps a reference and reads it
        at every search, so a barostat updates it in place and never through
        this object. ``None`` for a non-periodic node set (a partition's halo
        carries its images explicitly): the box is then the set's bounding box,
        recomputed on the device each step.
    pbc
        Per-axis periodicity; ignored without a cell.
    targets
        Node ids to use as search centres, an int32 device tensor, or ``None``
        for every node. A partition passes its owned atoms: ghosts are found
        as neighbours but never searched from, which is what keeps every edge
        in exactly one partition without an ownership mask.
    density
        Atoms per cubic angstrom, used to size the neighbour matrix before the
        first search; the first search then re-sizes it from what it counted.
        ``None`` estimates it from the node set itself at the first search.
    index_dtype
        ``edge_index`` dtype; int64, which every consumer accepts. The CUDA
        kernels also take int32, a later optimisation.
    float64_vectors
        Also keep the pair vectors in float64, which :meth:`edge_vec` then
        hands out as a differentiable function of the node coordinates with
        no recomputation. Needed for a model that differentiates through
        ``edge_vec`` or runs in float64; a compressed short-range model takes
        the float32 vectors and leaves this off.

    Notes
    -----
    :meth:`search` is the per-step entry: it runs the cell-list build, the
    query and the two kernels and touches the host only the first time, to
    size the buffers. :meth:`graph` then assembles a ``NeighborGraph`` over the
    persistent buffers. Both are meant to be called once per evaluation; the
    buffers are overwritten by the next :meth:`search`.
    """

    def __init__(
        self,
        n_node: int,
        rcut: float,
        device: Any,
        *,
        cell: torch.Tensor | None,
        pbc=(True, True, True),
        targets: torch.Tensor | None = None,
        density: float | None = None,
        index_dtype: torch.dtype = torch.int64,
        float64_vectors: bool = False,
    ) -> None:
        wp.init()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GpuNeighborList needs a CUDA device")
        if index_dtype is not torch.int64:
            raise NotImplementedError("edge_index is int64 for now")
        self.n_node = int(n_node)
        self.rcut = float(rcut)
        self.periodic = cell is not None
        self._cell = cell
        if cell is not None and (
            cell.shape != (1, 3, 3)
            or cell.dtype != torch.float64
            or cell.device != self.device
        ):
            raise ValueError(
                "cell must be a (1, 3, 3) float64 tensor on the list's device"
            )
        flags = (
            tuple(bool(p) for p in pbc) if cell is not None else (False, False, False)
        )
        self.pbc = flags
        self._pbc = torch.tensor(flags, dtype=torch.bool, device=self.device)
        if targets is not None:
            if targets.dtype != torch.int32 or targets.device != self.device:
                raise ValueError("targets must be an int32 tensor on the list's device")
            self._targets = targets.contiguous()
            self._identity = False
        else:
            self._targets = torch.arange(
                self.n_node, dtype=torch.int32, device=self.device
            )
            self._identity = True
        self.rows = int(self._targets.shape[0])
        self._targets_long = self._targets.to(torch.int64)
        self._density = density
        #: Keep the kernel's pair vectors in float64 as well, for a graph that
        #: must be differentiable (the charge response, the virial strain) or
        #: a model that runs in float64. Costs ``24 * capacity`` bytes.
        self.float64_vectors = bool(float64_vectors)
        #: Device status word, see the ``STATUS_*`` constants. Zeroed by every
        #: :meth:`search`; read it on the host at the end of a step, together
        #: with whatever else the step must download.
        self.status = torch.zeros(STATUS_LEN, dtype=torch.int64, device=self.device)
        self._n_node_t = torch.tensor(
            [self.n_node], dtype=torch.int64, device=self.device
        )
        self._sized = False
        self.max_neighbors = 0
        self.capacity = 0
        #: How often :meth:`search` had to synchronise to resize a buffer.
        self.resizes = 0
        # Cell-list cache and matrix: allocated by the first search, which
        # needs the positions to size them.
        self._cache = None
        self._box_cells_est = None
        #: Whether the periodic box is small enough that a cell change can
        #: invalidate the cell list's search radius (set by the first search).
        self.small_box = False
        #: The fixed search box of a non-periodic node set, and its origin.
        self._box: torch.Tensor | None = None
        self._lo: torch.Tensor | None = None

    # ------------------------------------------------------------------ setup

    def _box_for(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The cell list's positions and box.

        Periodic: the real positions and the real cell. Otherwise a box fixed
        at the first search -- the node set's bounding box plus a margin --
        and the positions shifted into it. Fixed on purpose: the number of
        cells per axis is recomputed from the box at every build, while the
        search radius is sized once with the cache, so the two agree only if
        every build sees the box the estimate saw. (A box that followed the
        nodes drifted across a cell-count boundary on an open axis, where one
        cell means a radius of zero, and silently lost every neighbour in the
        next cell.) Nodes that wander past the fixed box are clamped into its
        edge cells by the kernel, which only ever shortens cell distances, so
        the search stays complete; the margin keeps that rare.
        """
        if self.periodic:
            return x, self._cell
        if self._box is None:
            margin = _BOX_MARGIN_FACTOR * self.rcut
            lo = x.min(dim=0).values - margin
            extent = x.max(dim=0).values - lo + margin
            self._lo = lo
            self._box = torch.diag_embed(extent).unsqueeze(0)
        return x - self._lo, self._box

    def _estimate(self, box: torch.Tensor) -> None:
        """Size the cell-list cache for ``box`` (synchronises)."""
        n_cells, radius = estimate_cell_list_sizes(box, self._pbc, self.rcut)
        self._cache = allocate_cell_list(self.n_node, n_cells, radius, self.device)
        self._box_cells_est = n_cells
        if self.periodic:
            # Whether a cell change can invalidate the search radius: see
            # `cell_changed`. Decided once, from the box this estimate saw.
            h = box.detach().reshape(3, 3).cpu().numpy()
            volume = abs(np.linalg.det(h))
            widths = [
                volume / np.linalg.norm(np.cross(h[(k + 1) % 3], h[(k + 2) % 3]))
                for k in range(3)
            ]
            self.small_box = any(
                self.pbc[k] and widths[k] < (_SMALL_BOX_CELLS + 1) * self.rcut
                for k in range(3)
            )

    def _allocate_matrix(self, max_neighbors: int) -> None:
        self.max_neighbors = int(max_neighbors)
        m = self.max_neighbors
        dev = self.device
        self.neighbor_matrix = torch.zeros(
            (self.rows, m), dtype=torch.int32, device=dev
        )
        self.neighbor_matrix_shifts = torch.zeros(
            (self.rows, m, 3), dtype=torch.int32, device=dev
        )
        self.num_neighbors = torch.zeros(self.rows, dtype=torch.int32, device=dev)
        self._counts = torch.zeros(self.rows, dtype=torch.int32, device=dev)
        self._wp_nbr = wp.from_torch(
            self.neighbor_matrix, dtype=wp.int32, requires_grad=False
        )
        self._wp_shifts = wp.from_torch(
            self.neighbor_matrix_shifts, dtype=wp.vec3i, requires_grad=False
        )
        self._wp_counts = wp.from_torch(
            self._counts, dtype=wp.int32, requires_grad=False
        )

    def _allocate_edges(self, capacity: int) -> None:
        self.capacity = max(int(capacity), 2)
        cap, dev = self.capacity, self.device
        self.edge_index = torch.zeros((2, cap), dtype=torch.int64, device=dev)
        self.edge_shift = torch.zeros((cap, 3), dtype=torch.int32, device=dev)
        self.edge_vec32 = torch.zeros((cap, 3), dtype=torch.float32, device=dev)
        self.edge_vec64 = torch.zeros(
            (cap if self.float64_vectors else 1, 3), dtype=torch.float64, device=dev
        )
        self.edge_mask = torch.zeros(cap, dtype=torch.bool, device=dev)
        self.source_order = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.destination_order = torch.arange(cap, dtype=torch.int64, device=dev)
        self._arange = self.destination_order  # the same identity, read-only
        self._wp_src = wp.from_torch(
            self.edge_index[0], dtype=wp.int64, requires_grad=False
        )
        self._wp_dst = wp.from_torch(
            self.edge_index[1], dtype=wp.int64, requires_grad=False
        )
        self._wp_eshift = wp.from_torch(
            self.edge_shift, dtype=wp.vec3i, requires_grad=False
        )
        self._wp_evec = wp.from_torch(
            self.edge_vec32, dtype=wp.vec3f, requires_grad=False
        )
        self._wp_evec64 = wp.from_torch(
            self.edge_vec64, dtype=wp.vec3d, requires_grad=False
        )
        self._wp_sorder = wp.from_torch(
            self.source_order, dtype=wp.int64, requires_grad=False
        )

    def _allocate_nodes(self) -> None:
        n, dev = self.n_node, self.device
        self.destination_row_ptr = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        self.source_row_ptr = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        self._node_counts = torch.zeros(n, dtype=torch.int32, device=dev)
        self._src_count = torch.zeros(n, dtype=torch.int32, device=dev)
        self._cursor = torch.zeros(n, dtype=torch.int32, device=dev)
        self._wp_row_ptr = wp.from_torch(
            self.destination_row_ptr, dtype=wp.int64, requires_grad=False
        )
        self._wp_n_real = wp.from_torch(
            self.destination_row_ptr[n:], dtype=wp.int64, requires_grad=False
        )
        self._wp_src_count = wp.from_torch(
            self._src_count, dtype=wp.int32, requires_grad=False
        )
        self._wp_cursor = wp.from_torch(
            self._cursor, dtype=wp.int32, requires_grad=False
        )
        self._wp_centre = wp.from_torch(
            self._targets, dtype=wp.int32, requires_grad=False
        )
        self._wp_status = wp.from_torch(
            self.status, dtype=wp.int64, requires_grad=False
        )

    def _initial_max_neighbors(self, box: torch.Tensor) -> int:
        density = self._density
        if density is None:
            volume = float(torch.linalg.det(box[0]).abs())
            density = self.n_node / max(volume, 1e-12)
        sphere = 4.0 / 3.0 * math.pi * self.rcut**3
        return _round_up(1.3 * density * sphere)

    # ----------------------------------------------------------------- search

    def _query(self, pos: torch.Tensor, box: torch.Tensor) -> None:
        with _torch_stream_scope(self.device):
            build_cell_list(pos, self.rcut, box, self._pbc, *self._cache)
            self.num_neighbors.zero_()
            query_cell_list(
                pos,
                self.rcut,
                box,
                self._pbc,
                *self._cache,
                self.neighbor_matrix,
                self.neighbor_matrix_shifts,
                self.num_neighbors,
                strategy="atom_centric",
                target_indices=None if self._identity else self._targets,
            )

    def _size_from_first_search(self, pos: torch.Tensor, box: torch.Tensor) -> None:
        """The one synchronising path: query until the matrix holds every
        row, then size the edge buffers from the count."""
        self._estimate(box)
        self._allocate_nodes()
        self._allocate_matrix(self._initial_max_neighbors(box))
        while True:
            self._query(pos, box)
            found = int(self.num_neighbors.max())
            if found <= self.max_neighbors:
                break
            self._allocate_matrix(_round_up(1.2 * found))
            self.resizes += 1
        # Tight to what was counted, with room for density fluctuations.
        tight = _round_up(1.2 * max(found, 1))
        if tight < self.max_neighbors:
            self._allocate_matrix(tight)
            self._query(pos, box)
        n_edges = int(self.num_neighbors.sum())
        self._allocate_edges(_round_up(1.15 * n_edges, 64, 64) + 2)
        self._sized = True

    def grow(self) -> None:
        """Widen whichever buffer the last :meth:`search` reported as too
        small (synchronises). The caller then repeats the search."""
        flags = self.status.tolist()
        if flags[STATUS_MATRIX_OVERFLOW]:
            self._allocate_matrix(_round_up(1.25 * self.max_neighbors))
        if flags[STATUS_EDGE_OVERFLOW]:
            self._allocate_edges(_round_up(1.25 * self.capacity, 64, 64))
        self.resizes += 1

    def search(self, x: torch.Tensor, events: dict | None = None) -> None:
        """Build this step's graph from node coordinates ``x`` ``(n_node, 3)``.

        No host synchronisation after the first call. Afterwards the edge
        buffers hold the graph and :attr:`status` says whether a buffer was
        too small, in which case the step has to be redone after :meth:`grow`.

        ``events``, a dict, receives a CUDA event recorded after each stage
        (``start``, ``query``, ``rows``, ``fill``, ``source``, ``mask``) for a
        profiler to read once it has synchronised; the step itself stays
        asynchronous.
        """

        def mark(name: str) -> None:
            if events is not None:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events[name] = event

        mark("start")
        x = x.detach()
        if x.dtype != torch.float64:
            x = x.to(torch.float64)
        x = x.reshape(self.n_node, 3)
        if not x.is_contiguous():
            x = x.contiguous()
        pos, box = self._box_for(x)
        if not self._sized:
            self._size_from_first_search(pos, box)
        self.status.zero_()
        self._query(pos, box)
        mark("query")
        m = self.max_neighbors
        torch.clamp(self.num_neighbors, max=m, out=self._counts)
        self.status[STATUS_MATRIX_OVERFLOW : STATUS_MATRIX_OVERFLOW + 1] |= (
            (self.num_neighbors > m).any().to(torch.int64).reshape(1)
        )
        if self._identity:
            counts = self._counts
        else:
            self._node_counts.zero_()
            self._node_counts.index_copy_(0, self._targets_long, self._counts)
            counts = self._node_counts
        torch.cumsum(counts, 0, dtype=torch.int64, out=self.destination_row_ptr[1:])
        mark("rows")
        self._src_count.zero_()
        with _torch_stream_scope(self.device):
            wp.launch(
                _fill_edges,
                dim=(self.rows, self.max_neighbors),
                inputs=[
                    wp.from_torch(
                        x, dtype=wp.vec3d, requires_grad=False, return_ctype=True
                    ),
                    wp.from_torch(
                        box, dtype=wp.mat33d, requires_grad=False, return_ctype=True
                    ),
                    self._wp_nbr,
                    self._wp_shifts,
                    self._wp_counts,
                    self._wp_centre,
                    self._wp_row_ptr,
                    self.capacity,
                    self._wp_src,
                    self._wp_dst,
                    self._wp_eshift,
                    self._wp_evec,
                    self._wp_evec64,
                    int(self.float64_vectors),
                    self._wp_src_count,
                    self._wp_status,
                ],
                device=str(self.device),
            )
            mark("fill")
            torch.cumsum(
                self._src_count, 0, dtype=torch.int64, out=self.source_row_ptr[1:]
            )
            self._cursor.copy_(self.source_row_ptr[:-1])
            wp.launch(
                _source_order,
                dim=self.capacity,
                inputs=[
                    self._wp_src,
                    self._wp_dst,
                    self._wp_eshift,
                    self._wp_evec,
                    self._wp_evec64,
                    int(self.float64_vectors),
                    self._wp_n_real,
                    self._wp_cursor,
                    self._wp_sorder,
                ],
                device=str(self.device),
            )
        mark("source")
        torch.lt(self._arange, self.destination_row_ptr[-1], out=self.edge_mask)
        mark("mask")

    # ------------------------------------------------------------------ graph

    @property
    def n_real(self) -> torch.Tensor:
        """Edges within ``rcut`` this step, as a 0-d device tensor."""
        return self.destination_row_ptr[-1]

    def edge_vec(
        self,
        node_coord: torch.Tensor,
        strain: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The pair vectors as a differentiable function of ``node_coord``.

        ``node_coord`` is ``(n_node, 3)`` or ``(1, n_node, 3)``, typically a
        leaf, and must hold the coordinates of the last :meth:`search` (the
        vectors were computed from them); ``strain`` a ``(3, 3)`` matrix
        ``I + sym(eps)`` applied to the finished vectors, which is how the
        virial reaches them (an edge vector is ``x_j + S H - x_i``, every
        term of which deforms alike). The result spans the whole capacity,
        like :attr:`edge_mask`.

        With ``float64_vectors`` the vectors are the kernel's own, handed out
        through an autograd function whose backward scatters the cotangent
        onto the two ends. Without them they are recomputed from the compact
        edge list, in the coordinates' dtype.
        """
        x = node_coord.reshape(-1, 3)
        if self.float64_vectors:
            vec = _EdgeVectors.apply(x, self)
        else:
            src, dst = self.edge_index[0], self.edge_index[1]
            vec = x.index_select(0, src) - x.index_select(0, dst)
            if self.periodic:
                vec = vec + _times_3x3(self.edge_shift.to(x.dtype), self._cell[0])
        # The cast comes first: a compressed model's strain is then a float32
        # product, which is both what its kernel sees and the cheap GEMM.
        if dtype is not None and vec.dtype != dtype:
            vec = vec.to(dtype)
        if strain is not None:
            vec = _Strained.apply(vec, strain)
        return vec

    def graph(
        self,
        node_coord: torch.Tensor | None = None,
        strain: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
        compact: bool = False,
    ) -> NeighborGraph:
        """This step's graph as DeePMD-kit's ``NeighborGraph``.

        With ``node_coord`` the vectors are recomputed differentiably
        (:meth:`edge_vec`); without it they are the kernel's float32 values,
        which is all a short-range model needs, since DeePMD-kit assembles
        forces from the edge gradients through the CSR views and never
        differentiates through the vectors' construction.

        ``compact`` narrows every edge axis to the real edges plus two guard
        edges, which costs one host synchronisation. The compressed kernels
        walk destination rows and never see the tail, so they take the full
        buffers; the uncompressed array-API descriptor processes every edge
        and must not pay for padding.
        """
        if node_coord is not None:
            vec = self.edge_vec(node_coord, strain, dtype)
        else:
            vec = (
                self.edge_vec32
                if dtype in (None, torch.float32)
                else self.edge_vec32.to(dtype)
            )
        edge_index, mask = self.edge_index, self.edge_mask
        d_order, s_order = self.destination_order, self.source_order
        if compact:
            n = int(self.n_real) + 2
            edge_index, vec, mask = edge_index[:, :n], vec[:n], mask[:n]
            d_order, s_order = d_order[:n], s_order[:n]
        return NeighborGraph(
            n_node=self._n_node_t,
            edge_index=edge_index,
            edge_vec=vec,
            edge_mask=mask,
            destination_order=d_order,
            destination_row_ptr=self.destination_row_ptr,
            source_order=s_order,
            source_row_ptr=self.source_row_ptr,
            destination_sorted=True,
        )

    # ------------------------------------------------------------- cell change

    def cell_changed(self, cell) -> bool:
        """Re-size the cell-list cache for a new cell when the box is small
        enough for that to matter (synchronises when it does).

        ``cell`` is the host copy the caller keeps, so deciding costs no
        device readback. nvalchemiops recomputes the number of cells per axis
        on the device at every build, but the search radius -- how many
        cells out a query looks -- is fixed when the cache is sized. With at
        least four cells per axis forced, a box whose face is under ``4 rcut``
        has cells narrower than the cutoff and a radius above one; shrink it
        a little and the radius it needs can grow while the stored one does
        not. A larger box only makes the kernel merge cells, which is safe.
        So only a small periodic box re-estimates. Returns whether it did.
        """
        del cell  # the regime was decided from the box the first estimate saw
        return self.cell_changed_device()

    def cell_changed_device(self) -> bool:
        """:meth:`cell_changed` for a cell that changed on the device: nothing
        to read unless the box is small, in which case the re-estimate reads
        the device (small systems can afford that)."""
        if not self.periodic or self._cache is None or not self.small_box:
            return False
        self._estimate(self._cell)
        self.resizes += 1
        return True
