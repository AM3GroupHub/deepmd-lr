# SPDX-License-Identifier: LGPL-3.0-or-later
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
def test_auto_keeps_small_frames_on_the_dense_search():
    """The cell list has a fixed setup cost a small frame never earns back."""
    small = 32  # 96 atoms, comfortably under the threshold
    assert small * 3 < CELL_LIST_MIN_ATOMS_PBC
    coord, atype, cell = frame(n_mol=small, box=18.0)
    auto = build_neighbor_graph_fast(coord, atype, cell, RCUT, method="auto")
    assert edge_set(auto) == edge_set(build_neighbor_graph(coord, atype, cell, RCUT))


def test_rejects_an_unknown_method():
    coord, atype, cell = frame(n_mol=8, box=12.0)
    with pytest.raises(ValueError, match="unknown neighbour method"):
        build_neighbor_graph_fast(coord, atype, cell, RCUT, method="octree")
