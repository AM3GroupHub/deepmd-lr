# SPDX-License-Identifier: MIT
"""The nvalchemiops cell list must produce DeePMD-kit's graph exactly.

The oracle is `build_neighbor_graph`, which DeePMD-kit's own docstring calls a
"correctness oracle". Anything the fast path does differently is a bug in the
fast path.

There is history here: nvalchemi's batched `naive` builder was found to
silently under-report neighbours for frames after the first -- 17280 against
10024 for identical frames -- so a neighbour backend from this library gets
checked against the oracle rather than trusted.
"""

import numpy as np
import pytest
import torch
from deepmd.dpmodel.utils.neighbor_graph import build_neighbor_graph

from deepmd_lr.neighbors import (
    CELL_LIST_MIN_ATOMS_FREE,
    CELL_LIST_MIN_ATOMS_PBC,
    build_neighbor_graph_fast,
)

from .test_joint_model import frame

RCUT = 6.0
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def edge_set(graph):
    """Valid edges as sorted (src, dst, vec) rows -- order-independent."""
    mask = graph.edge_mask.reshape(-1).cpu().numpy().astype(bool)
    src = graph.edge_index[0].reshape(-1).cpu().numpy()[mask]
    dst = graph.edge_index[1].reshape(-1).cpu().numpy()[mask]
    vec = graph.edge_vec.reshape(-1, 3).detach().cpu().numpy()[mask]
    rows = zip(src, dst, vec, strict=True)
    return sorted((int(a), int(b)) + tuple(np.round(v, 9)) for a, b, v in rows)


@CUDA
@pytest.mark.parametrize("n_mol,box", [(256, 36.0), (864, 54.0)])
@pytest.mark.parametrize("periodic", [True, False])
def test_matches_the_oracle(n_mol, box, periodic):
    coord, atype, cell = frame(n_mol=n_mol, box=box)
    b = cell if periodic else None
    oracle = build_neighbor_graph(coord, atype, b, RCUT)
    fast = build_neighbor_graph_fast(coord, atype, b, RCUT, method="cell_list")
    assert edge_set(fast) == edge_set(oracle)


@CUDA
def test_edge_vec_is_differentiable():
    """Forces are dE/d(edge_vec). The Warp kernel's own vectors carry no
    gradient, so the builder must rebuild them from `coord`."""
    coord, atype, cell = frame(n_mol=256, box=36.0)
    tracked = coord.clone().requires_grad_(True)
    graph = build_neighbor_graph_fast(tracked, atype, cell, RCUT, method="cell_list")
    assert graph.edge_vec.requires_grad
    valid = graph.edge_vec[graph.edge_mask.reshape(-1).bool()]
    (grad,) = torch.autograd.grad((valid**2).sum(), tracked)
    assert float(grad.norm()) > 0.0


@CUDA
@pytest.mark.parametrize("periodic", [True, False])
def test_auto_agrees_with_the_oracle_on_both_sides_of_the_crossover(periodic):
    """`auto` switches backend by size; the graph must not switch with it.

    Without a cell a partition's dense search has no ghost images to
    materialise and stays cheaper to about a thousand atoms, so the two frames
    here, 96 and 1,536 atoms, straddle the crossover. With a cell the GPU cell
    list wins at every measured size, the threshold is 0, and both frames take
    the cell list.
    """
    threshold = CELL_LIST_MIN_ATOMS_PBC if periodic else CELL_LIST_MIN_ATOMS_FREE
    for n_mol, box in [(32, 18.0), (512, 44.0)]:
        coord, atype, cell = frame(n_mol=n_mol, box=box)
        b = cell if periodic else None
        auto = build_neighbor_graph_fast(coord, atype, b, RCUT, method="auto")
        assert edge_set(auto) == edge_set(
            build_neighbor_graph(coord, atype, b, RCUT)
        ), f"{n_mol * 3} atoms, threshold {threshold}, periodic={periodic}"


def test_rejects_an_unknown_method():
    coord, atype, cell = frame(n_mol=8, box=12.0)
    with pytest.raises(ValueError, match="unknown neighbour method"):
        build_neighbor_graph_fast(coord, atype, cell, RCUT, method="octree")


@CUDA
@pytest.mark.parametrize("method", ["dense", "ase", "vesin", "nv"])
def test_only_dense_carries_the_cell_derivative(method):
    """Which builders carry a cell derivative -- a fact about DeePMD-kit.

    Only the dense builder does. `nv` is the dangerous one: it returns a cell
    gradient of 6.7e-13 -- float64 round-off, fifteen orders below the dense
    builder's 1932.8 -- rather than raising, so a virial that leaned on it
    would be silently wrong while the forces stayed right.

    This package no longer leans on it. `_predict_charges` builds its graph
    from the unstrained geometry and applies the strain to the finished edge
    vectors, so d(charge)/d(strain) flows through `edge_vec` and the box
    derivative is never needed -- which is what lets the charge graph use the
    O(N) builder (63 GB and 3 s per training step at 6520 atoms, down to
    8 GB and 0.9 s). The fact is pinned here anyway, because anyone who
    builds a strained graph from a strained box will hit it.
    """
    from deepmd.pt_expt.utils.graph_builder import build_neighbor_graph_for_method

    coord, atype, cell = frame(n_mol=32, box=18.0)
    tracked_box = cell.clone().requires_grad_(True)
    try:
        graph = build_neighbor_graph_for_method(
            method, coord, atype, tracked_box, RCUT, None
        )
        valid = graph.edge_vec[graph.edge_mask.reshape(-1).bool()]
        (grad,) = torch.autograd.grad(
            (valid**2).sum(), tracked_box, allow_unused=True
        )
        norm = 0.0 if grad is None else float(grad.norm())
    except RuntimeError:
        norm = 0.0  # cannot backward to the box at all

    if method == "dense":
        assert norm > 1.0, (
            "the dense builder must stay cell-differentiable; the long-range "
            "virial depends on it"
        )
    else:
        # The dense builder gives ~1.9e3 here, so anything below 1e-6 is
        # round-off rather than a real derivative.
        assert norm < 1e-6, (
            f"{method!r} now carries a cell gradient ({norm:.4e}). Nothing "
            "here depends on that any more, but the docstring above is now "
            "wrong and should say so."
        )
