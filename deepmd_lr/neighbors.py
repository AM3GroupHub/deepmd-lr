# SPDX-License-Identifier: LGPL-3.0-or-later
"""Neighbour search on the GPU, via nvalchemiops' cell list.

DeePMD-kit's ``build_neighbor_graph`` describes itself as "an O(N^2) reference
search (correctness oracle)"; its O(N) backends are not wired up yet. That is
fine for training-sized frames and expensive for MD-sized ones -- measured on
one GB10, a periodic 6144-atom search costs 471 ms there against 3.5 ms here,
and the cell list is nearly flat in system size.

The end-to-end gain is much smaller than that ratio suggests, because the
neighbour search is only one part of a step and the rest is CPU-bound: with
``auto``, 1.00-1.77x over 768-12288 atoms and 1-4 partitions, growing with
system size and shrinking as partitions get smaller. ``auto`` is tuned to
never lose -- forcing ``"cell_list"`` everywhere costs 0.86x at 2592 atoms
split four ways.

One caveat for benchmarking: Warp JIT-compiles its kernels on first use. The
first call through the cell list costs ~1.0 s, and a second deferred
initialisation of ~0.39 s lands a few calls later; after that it is stable.
Both are one-time per process and irrelevant to a real trajectory, but they
will wreck a short timing loop that does not warm up.

Only the *search* comes from nvalchemiops. The edge vectors it returns are
produced inside a Warp kernel and carry no gradient, so they are discarded and
``neighbor_graph_from_ijs`` recomputes ``edge_vec`` from ``coord``/``box``.
Forces are ``dE/d(edge_vec)``, so using the kernel's vectors directly would
silently produce a model with no forces. DeePMD-kit's own ASE backend discards
ASE's vectors for exactly this reason.

Conventions were pinned against nvalchemiops' own returned vectors: row 0 of
its edge index is the centre ``i``, row 1 the neighbour ``j``, and the shift
sign matches DeePMD-kit's ``coord[j] + S @ box - coord[i]`` to 3.6e-15.
"""

from typing import (
    Any,
)

import torch
from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
    neighbor_graph_from_ijs,
)

__all__ = [
    "CELL_LIST_MIN_ATOMS_FREE",
    "CELL_LIST_MIN_ATOMS_PBC",
    "build_neighbor_graph_fast",
]

# The cell list costs a flat ~3.2-4.2 ms at any size; the dense search costs
# almost nothing when small and blows up when large. So `auto` is a crossover,
# and the crossover sits in a very different place depending on periodicity,
# because the dense path only materialises ghost images when there is a box.
#
# Measured on one GB10, dense vs the full wrapper (ms):
#
#   atoms    192   384   576   768  1536  2592  6144
#   PBC     1.18  2.46  4.65  7.88 29.53 82.84     -
#   free    0.38  0.33     -  0.46  1.16  3.07 17.15
#   wrapper 4.18  3.88  3.76  3.70  3.70  3.85  4.15
#
#: Periodic crossover: between 384 and 576 atoms.
CELL_LIST_MIN_ATOMS_PBC = 512
#: Non-periodic crossover -- which is what partitions use, since the halo
#: already carries the images. Between 2592 and 6144 atoms.
CELL_LIST_MIN_ATOMS_FREE = 3072


def _cell_list_edges(coord: torch.Tensor, box: torch.Tensor | None, rcut: float):
    """One frame's ``(i, j, S)`` from nvalchemiops' cell list."""
    from nvalchemiops.torch.neighbors import (
        cell_list,
        synthesize_cell_for_ss,
    )

    if box is None:
        # No cell: search inside a bounding box instead. The helper translates
        # the positions so the box starts at the origin, which changes no
        # index and no separation -- and `edge_vec` is rebuilt from the
        # original `coord` regardless.
        search_coord, cell, pbc = synthesize_cell_for_ss(coord, rcut)
    else:
        search_coord = coord
        cell = box.reshape(1, 3, 3)
        pbc = torch.ones(3, dtype=torch.bool, device=coord.device)

    # Without `return_vectors` this yields (edge_index, ptr, shifts). The
    # vectors are deliberately not requested -- they carry no gradient.
    edge_index, _ptr, shifts = cell_list(
        search_coord.contiguous(),
        rcut,
        cell.contiguous(),
        pbc,
        return_neighbor_list=True,
    )
    return edge_index[0].long(), edge_index[1].long(), shifts.long()


def build_neighbor_graph_fast(
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
    layout: Any = None,
    *,
    method: str = "auto",
):
    """Drop-in for ``build_neighbor_graph`` with an O(N) search underneath.

    ``method`` is ``"auto"`` (cell list once the frame is big enough to pay for
    it), ``"cell_list"`` (always) or ``"dense"`` (DeePMD-kit's oracle, always).
    The graph produced is identical either way -- ``tests/test_neighbors.py``
    asserts equality of the edge sets against the oracle.
    """
    if method not in ("auto", "cell_list", "dense"):
        raise ValueError(
            f"unknown neighbour method {method!r}; expected one of "
            "'auto', 'cell_list', 'dense'"
        )

    coord = coord.reshape(coord.shape[0], -1, 3)
    nf, nloc, _ = coord.shape
    atype = atype.reshape(nf, nloc)

    threshold = (
        CELL_LIST_MIN_ATOMS_FREE if box is None else CELL_LIST_MIN_ATOMS_PBC
    )
    use_cell_list = method == "cell_list" or (
        method == "auto" and nloc >= threshold and coord.device.type == "cuda"
    )
    if not use_cell_list:
        return build_neighbor_graph(coord, atype, box, rcut, layout)

    boxes = None if box is None else box.reshape(nf, 3, 3)
    i_all, j_all, s_all, frame_all = [], [], [], []
    for f in range(nf):
        i, j, s = _cell_list_edges(coord[f], None if boxes is None else boxes[f], rcut)
        # DeePMD-kit treats `type < 0` as virtual: excluded as centre and as
        # neighbour. The geometric search is type-blind, so filter afterwards.
        types = atype[f]
        keep = (types[i] >= 0) & (types[j] >= 0)
        i, j, s = i[keep], j[keep], s[keep]
        i_all.append(i)
        j_all.append(j)
        s_all.append(s)
        frame_all.append(torch.full_like(i, f))

    device = coord.device
    return neighbor_graph_from_ijs(
        torch.cat(i_all),
        torch.cat(j_all),
        torch.cat(s_all),
        coord.reshape(-1, 3),
        None if box is None else boxes,
        torch.cat(frame_all),
        torch.full((nf,), nloc, dtype=torch.int64, device=device),
        layout,
    )
