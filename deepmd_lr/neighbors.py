# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pick the neighbour-graph builder that suits the frame.

DeePMD-kit ships two relevant backends. ``build_neighbor_graph`` is the
in-tree dense search, which its own docstring calls "an O(N^2) reference
search (correctness oracle)". ``build_neighbor_graph_nv`` is backed by
nvalchemiops' GPU cell list -- the same engine this package already depends on
for the reciprocal sum.

This module only chooses between them by size. Measured on one GB10, median of
7 after warm-up (ms):

    atoms            192   384   768  1536  2592   6144
    periodic  dense 1.06  2.35  7.60 29.37 83.13 470.12
              nv    1.01  1.06  1.22  4.02  3.84   4.26
    box=None  dense 0.30  0.29  0.42  1.19  3.16  16.97
              nv    0.62  0.69  0.83  1.15  0.69   3.95

With a cell, ``nv`` wins at every size, so there is nothing to decide. Without
one -- which is what a partition uses, since its halo already carries the
images -- the dense path skips ghost materialisation and stays cheaper until
about a thousand atoms.

An earlier version of this module drove nvalchemiops directly instead of going
through DeePMD-kit's backend. That was redundant and slower: the in-tree
builder batches natively over frames, handles ``pair_excl``, and decodes the
dense neighbour matrix rather than asking for ``return_neighbor_list=True``,
which nvalchemiops' own docstring warns carries a penalty. Both produce
identical edge sets and identical ``edge_vec`` gradients.
"""

import torch
from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)

# Everything NV-related is imported inside the function below, and has to be.
# At module level, `deepmd.pt_expt.utils.nv_graph_builder` re-enters
# DeePMD-kit's `load_entry_point("deepmd.pt_expt")`, which imports this
# package again and leaves it half-initialised -- `import deepmd_lr.neighbors`
# then fails outright. `deepmd.pt.utils.nv_nlist` is no safer: reaching it
# first opens a cycle inside DeePMD-kit itself, where `nv_graph_builder` wants
# `_input_device_context` from the module still being initialised.

__all__ = [
    "CELL_LIST_MIN_ATOMS_FREE",
    "CELL_LIST_MIN_ATOMS_PBC",
    "build_neighbor_graph_fast",
]

#: With a cell the GPU cell list wins at every size measured, down to 192
#: atoms, so there is no crossover to respect.
CELL_LIST_MIN_ATOMS_PBC = 0

#: Without a cell the dense search has no ghost images to materialise and
#: stays cheaper up to about a thousand atoms. Partitions are this case.
CELL_LIST_MIN_ATOMS_FREE = 1024


def build_neighbor_graph_fast(
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
    *,
    method: str = "auto",
):
    """Build a neighbour graph, choosing the backend by frame size.

    ``method`` is ``"auto"`` (size-based, the default), ``"cell_list"`` to
    force the nvalchemiops backend, or ``"dense"`` to force DeePMD-kit's
    oracle. The graph is identical either way --
    ``tests/test_neighbors.py`` asserts equality of the edge sets.
    """
    if method not in ("auto", "cell_list", "dense"):
        raise ValueError(
            f"unknown neighbour method {method!r}; expected one of "
            "'auto', 'cell_list', 'dense'"
        )

    nloc = coord.reshape(coord.shape[0], -1, 3).shape[1]
    threshold = CELL_LIST_MIN_ATOMS_FREE if box is None else CELL_LIST_MIN_ATOMS_PBC
    use_cell_list = method == "cell_list"
    if method == "auto" and nloc >= threshold and coord.device.type == "cuda":
        from deepmd.pt.utils.nv_nlist import (
            is_nv_available,
        )

        use_cell_list = bool(is_nv_available())

    if use_cell_list:
        from deepmd.pt_expt.utils.nv_graph_builder import (
            build_neighbor_graph_nv,
        )

        return build_neighbor_graph_nv(coord, atype, box, rcut)
    return build_neighbor_graph(coord, atype, box, rcut)
