# SPDX-License-Identifier: LGPL-3.0-or-later
"""The GPU-direct neighbour list must produce DeePMD-kit's graph exactly.

The oracle is DeePMD-kit's dense ``build_neighbor_graph``, the O(N^2) search
its own docstring calls a correctness oracle. The list searches with
nvalchemiops' cell list and compacts the result with two Warp kernels of its
own, so every convention it relies on -- the sign of the shift, which index
is the centre, how images of a small box are counted, what happens past a
buffer's end -- is checked here against the oracle rather than trusted.
"""

import numpy as np
import pytest
import torch
from deepmd.dpmodel.utils.neighbor_graph import build_neighbor_graph
from deepmd.pt_expt.utils.env import DEVICE
from deepmd.pt_expt.utils.graph_csr import validate_graph_csr_for_export

from deepmd_lr.nvalchemi.nlist import (
    STATUS_EDGE_OVERFLOW,
    STATUS_MATRIX_OVERFLOW,
    GpuNeighborList,
)
from deepmd_lr.nvalchemi.partition import partition_system

from .test_joint_model import frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the list runs on the GPU"
)

RCUT = 6.0
PBC = (True, True, True)


def edge_set(edge_index, mask, vec, decimals=7):
    """Valid edges as sorted (src, dst, vec) rows -- order-independent."""
    m = mask.reshape(-1).cpu().numpy().astype(bool)
    src = edge_index[0].reshape(-1).cpu().numpy()[m]
    dst = edge_index[1].reshape(-1).cpu().numpy()[m]
    v = vec.reshape(-1, 3).detach().cpu().numpy()[m]
    rows = zip(src, dst, v, strict=True)
    return sorted((int(a), int(b)) + tuple(np.round(x, decimals)) for a, b, x in rows)


def oracle(coord, atype, cell):
    g = build_neighbor_graph(coord, atype, cell, RCUT)
    return edge_set(g.edge_index, g.edge_mask, g.edge_vec)


def searched(coord, cell, pbc=PBC, float64_vectors=True, **kwargs):
    """A list over ``coord`` ``(1, N, 3)``, searched twice (the first call
    sizes the buffers and is allowed to synchronise). Float64 vectors from
    the kernel by default, the path the long-range and float64 models take."""
    lst = GpuNeighborList(
        coord.shape[1],
        RCUT,
        DEVICE,
        cell=cell,
        pbc=pbc,
        float64_vectors=float64_vectors,
        **kwargs,
    )
    lst.search(coord[0])
    lst.search(coord[0])
    return lst


def assert_matches(lst, coord, want):
    """Differentiable float64 vectors equal to the oracle's; the kernel's
    float32 vectors within float32 rounding of them."""
    g = lst.graph(node_coord=coord[0])
    assert edge_set(g.edge_index, g.edge_mask, g.edge_vec) == want
    g32 = lst.graph()
    m = g32.edge_mask
    assert float((g32.edge_vec[m].double() - g.edge_vec[m]).abs().max()) < 2e-6


def assert_valid_csr(lst):
    """Both CSR views address the real edges correctly: every destination
    row holds edges into its node, every source row edges out of its node,
    ``source_order`` is a permutation, and the mask ends where the rows do."""
    g = lst.graph()
    validate_graph_csr_for_export(
        g.edge_index,
        g.edge_mask,
        g.destination_order,
        g.destination_row_ptr,
        g.source_order,
        g.source_row_ptr,
        lst.n_node,
        destination_sorted=True,
    )
    n = int(lst.n_real)
    rp = g.destination_row_ptr.cpu().numpy()
    sp = g.source_row_ptr.cpu().numpy()
    ei = g.edge_index.cpu().numpy()
    so = g.source_order.cpu().numpy()
    assert rp[-1] == n and sp[-1] == n
    for i in range(lst.n_node):
        assert (ei[1, rp[i] : rp[i + 1]] == i).all(), f"destination row {i}"
        assert (ei[0, so[sp[i] : sp[i + 1]]] == i).all(), f"source row {i}"
    assert len(set(so.tolist())) == len(so), "source_order is not a permutation"
    assert bool(g.edge_mask[:n].all()) and not bool(g.edge_mask[n:].any())
    assert torch.equal(g.destination_order, torch.arange(lst.capacity, device=DEVICE))


class TestMatchesTheOracle:
    @pytest.mark.parametrize("n_mol,box", [(16, 14.0), (4, 10.0), (64, 20.0)])
    def test_periodic_box(self, n_mol, box):
        coord, atype, cell = frame(n_mol=n_mol, box=box)
        lst = searched(coord, cell)
        assert_matches(lst, coord, oracle(coord, atype, cell))
        assert_valid_csr(lst)

    def test_a_small_box_counts_every_image(self):
        """Box 10 with rcut 6: the same pair can be in range through two
        images, and an atom can see its own image. Both are edges."""
        coord, atype, cell = frame(n_mol=4, box=10.0)
        lst = searched(coord, cell)
        g = lst.graph()
        m = g.edge_mask
        pairs = list(
            zip(g.edge_index[0][m].tolist(), g.edge_index[1][m].tolist(), strict=True)
        )
        assert len(pairs) > len(set(pairs)), "no pair appears through two images"
        assert int(lst.edge_shift[m].abs().max()) >= 1
        assert_matches(lst, coord, oracle(coord, atype, cell))

    def test_triclinic_cell(self):
        coord, atype, cell = frame(n_mol=16, box=14.0)
        cell = cell.clone()
        cell[0, 1, 0] = 3.0
        cell[0, 2, 1] = -2.0
        lst = searched(coord, cell)
        assert_matches(lst, coord, oracle(coord, atype, cell))
        assert_valid_csr(lst)

    def test_slab(self):
        """One open axis: no images along it. The oracle sees the same
        thing through a cell tall enough that no z image is within rcut."""
        coord, atype, _ = frame(n_mol=16, box=14.0)
        c = coord.clone()
        c[0, :, 2] += 10.0 - c[0, :, 2].min()
        height = float(c[0, :, 2].max() - c[0, :, 2].min()) + 20.0
        cell = torch.diag(
            torch.tensor([14.0, 14.0, height], dtype=torch.float64, device=DEVICE)
        ).unsqueeze(0)
        lst = searched(c, cell, pbc=(True, True, False))
        assert_matches(lst, c, oracle(c, atype, cell))
        assert_valid_csr(lst)

    def test_unwrapped_positions(self):
        """Atoms moved by lattice vectors, up to three cells: the kernel
        folds the images into the shifts and the graph does not change."""
        coord, atype, cell = frame(n_mol=16, box=14.0)
        rng = np.random.default_rng(0)
        idx = rng.choice(coord.shape[1], 12, replace=False)
        jumps = torch.as_tensor(
            rng.integers(-3, 4, size=(12, 3)), dtype=torch.float64, device=DEVICE
        )
        moved = coord.clone()
        moved[0, idx] += jumps @ cell[0]
        lst = searched(moved, cell)
        assert_matches(lst, moved, oracle(coord, atype, cell))

    @pytest.mark.parametrize("n_part", [2, 4])
    def test_halo_node_set_with_target_rows(self, n_part):
        """A partition's node set: owned atoms plus explicit ghost images,
        searched non-periodically with only the owned atoms as centres. The
        edges are the dense search's edges whose destination is owned.

        Four partitions of a 12 A box give node sets under two cutoffs wide
        on an open axis, where nvalchemiops uses one cell and a search radius
        of zero: the box the radius was sized for and the box each build sees
        have to be the same box (they once were not, and every neighbour in
        the next cell was lost)."""
        coord, atype, cell = frame(n_mol=8, box=12.0)
        c, a, h = (t[0].cpu().numpy() for t in (coord, atype, cell))
        for part in partition_system(c, h, RCUT, n_part, skin=1.0):
            nodes = torch.as_tensor(part.coordinates(c), device=DEVICE).unsqueeze(0)
            types = torch.as_tensor(part.types(a), device=DEVICE).unsqueeze(0)
            dense = build_neighbor_graph(nodes, types, None, RCUT)
            keep = dense.edge_mask & (dense.edge_index[1] < part.n_owned)
            want = edge_set(dense.edge_index, keep, dense.edge_vec)
            targets = torch.arange(part.n_owned, dtype=torch.int32, device=DEVICE)
            lst = searched(nodes, None, targets=targets)
            assert lst.rows == part.n_owned
            assert_matches(lst, nodes, want)
            assert_valid_csr(lst)

    def test_a_drifting_halo_set_is_still_searched_completely(self):
        """A non-periodic box is fixed at the first search. Nodes that drift
        past it are clamped into its edge cells, which only shortens cell
        distances, so even a drift well beyond the margin loses nothing."""
        coord, atype, cell = frame(n_mol=8, box=12.0)
        c, a, h = (t[0].cpu().numpy() for t in (coord, atype, cell))
        part = partition_system(c, h, RCUT, 4, skin=1.0)[1]
        nodes = torch.as_tensor(part.coordinates(c), device=DEVICE).unsqueeze(0)
        types = torch.as_tensor(part.types(a), device=DEVICE).unsqueeze(0)
        targets = torch.arange(part.n_owned, dtype=torch.int32, device=DEVICE)
        lst = searched(nodes, None, targets=targets)
        rng = np.random.default_rng(3)
        for scale in (0.5, 2.0, 5.0):
            moved = nodes + torch.as_tensor(
                rng.normal(scale=scale, size=tuple(nodes.shape)),
                dtype=torch.float64,
                device=DEVICE,
            )
            dense = build_neighbor_graph(moved, types, None, RCUT)
            keep = dense.edge_mask & (dense.edge_index[1] < part.n_owned)
            lst.search(moved[0])
            while int(lst.status.sum()):  # a drift can crowd a row past M
                lst.grow()
                lst.search(moved[0])
            assert_matches(lst, moved, edge_set(dense.edge_index, keep, dense.edge_vec))

    def test_a_walk_matches_the_oracle_at_every_step(self):
        coord, atype, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        rng = np.random.default_rng(1)
        x = coord.clone()
        for _ in range(5):
            x = x + torch.as_tensor(
                rng.normal(scale=0.3, size=x.shape), dtype=torch.float64, device=DEVICE
            )
            lst.search(x[0])
            assert_matches(lst, x, oracle(x, atype, cell))
        assert lst.resizes == 0


class TestBuffers:
    def test_a_matrix_too_narrow_is_reported_and_grown(self):
        coord, atype, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        lst._allocate_matrix(16)
        lst.search(coord[0])
        assert int(lst.status[STATUS_MATRIX_OVERFLOW]) == 1
        lst.grow()
        lst.search(coord[0])
        assert int(lst.status.sum()) == 0
        assert_matches(lst, coord, oracle(coord, atype, cell))

    def test_edge_buffers_too_small_are_reported_and_grown(self):
        coord, atype, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        lst._allocate_edges(64)
        lst.search(coord[0])
        assert int(lst.status[STATUS_EDGE_OVERFLOW]) == 1
        # The prefix that fits is still a valid graph of the first rows.
        g = lst.graph()
        rp = g.destination_row_ptr.cpu().numpy()
        full = int(np.searchsorted(rp, 64, side="right")) - 1
        ei = g.edge_index.cpu().numpy()
        for i in range(full):
            assert (ei[1, rp[i] : rp[i + 1]] == i).all()
        while int(lst.status[STATUS_EDGE_OVERFLOW]):
            lst.grow()
            lst.search(coord[0])
        assert_matches(lst, coord, oracle(coord, atype, cell))
        assert_valid_csr(lst)

    def test_a_poisoned_tail_is_never_read(self):
        """Consumers walk the CSR rows and never see the slots past the real
        edges, and the differentiable recompute reads only node ids. NaNs in
        the tail's vectors and garbage in its shifts must change nothing."""
        coord, atype, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        n = int(lst.n_real) + 2
        lst.edge_vec32[n:] = float("nan")
        lst.edge_shift[n:] = 7
        assert_matches(lst, coord, oracle(coord, atype, cell))
        lst.search(coord[0])
        assert_matches(lst, coord, oracle(coord, atype, cell))

    def test_a_warm_search_never_synchronises(self):
        coord, _, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            lst.search(coord[0] + 1e-3)
            lst.graph(node_coord=coord[0])
            lst.graph()
            lst.status.zero_()
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_a_side_stream_is_honoured(self):
        """Warp launches on PyTorch's current stream, so a caller on a side
        stream gets a consistent graph rather than a race."""
        coord, atype, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        want = oracle(coord, atype, cell)
        stream = torch.cuda.Stream(device=DEVICE)
        with torch.cuda.stream(stream):
            moved = coord * 1.0 + 0.0  # produced on the side stream
            lst.search(moved[0])
            g = lst.graph(node_coord=moved[0])
            got = edge_set(g.edge_index, g.edge_mask, g.edge_vec)
        torch.cuda.synchronize()
        assert got == want

    def test_only_a_small_box_re_estimates_on_a_cell_change(self):
        big, _, big_cell = frame(n_mol=256, box=36.0)
        lst = searched(big, big_cell)
        assert lst.cell_changed(big_cell[0].cpu().numpy() * 0.95) is False
        coord, atype, cell = frame(n_mol=16, box=14.0)
        cell = cell.clone()
        lst = searched(coord, cell)
        cell.mul_(0.95)
        x = coord * 0.95
        assert lst.cell_changed(cell[0].cpu().numpy()) is True
        lst.search(x[0])
        assert_matches(lst, x, oracle(x, atype, cell))


class TestGraphViews:
    def test_compact_view_ends_with_two_zero_guards(self):
        coord, _, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        g = lst.graph(compact=True)
        n = int(lst.n_real)
        assert g.edge_index.shape[1] == n + 2 and g.edge_vec.shape[0] == n + 2
        assert bool(g.edge_mask[:n].all()) and not bool(g.edge_mask[n:].any())
        assert torch.equal(
            g.edge_index[:, n:], torch.zeros((2, 2), dtype=torch.int64, device=DEVICE)
        )
        assert float(g.edge_vec[n:].abs().max()) == 0.0
        assert torch.equal(g.source_order[n:], torch.arange(n, n + 2, device=DEVICE))

    def test_edge_vec_reaches_the_coordinates_and_the_strain(self):
        coord, _, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        leaf = coord[0].clone().requires_grad_(True)
        eps = torch.zeros(
            (3, 3), dtype=torch.float64, device=DEVICE, requires_grad=True
        )
        strain = torch.eye(3, dtype=torch.float64, device=DEVICE) + 0.5 * (eps + eps.T)
        vec = lst.edge_vec(leaf, strain)
        plain = lst.edge_vec(coord[0])
        assert torch.equal(vec.detach(), plain)
        m = lst.edge_mask
        g_x, g_eps = torch.autograd.grad((vec[m] ** 2).sum(), [leaf, eps])
        assert float(g_x.norm()) > 0.0 and float(g_eps.norm()) > 0.0
        # d/d eps of sum |v|^2 under v -> v (I + sym eps)^T is 2 sum v v^T (symmetrised)
        want = 2.0 * plain[m].T @ plain[m]
        torch.testing.assert_close(g_eps, 0.5 * (want + want.T), rtol=0, atol=1e-8)

    def test_recomputed_vectors_equal_the_kernels(self):
        """Without float64 vectors the list recomputes them from the compact
        edge list in torch; with them it hands out the kernel's through an
        autograd function. Same vectors, same gradient."""
        coord, _, cell = frame(n_mol=16, box=14.0)
        kernel = searched(coord, cell)
        torch_path = searched(coord, cell, float64_vectors=False)
        leaf_a = coord[0].clone().requires_grad_(True)
        leaf_b = coord[0].clone().requires_grad_(True)
        va, vb = kernel.edge_vec(leaf_a), torch_path.edge_vec(leaf_b)
        ma, mb = kernel.edge_mask, torch_path.edge_mask
        assert edge_set(kernel.edge_index, ma, va) == edge_set(
            torch_path.edge_index, mb, vb
        )
        (ga,) = torch.autograd.grad((va[ma] ** 2).sum(), leaf_a)
        (gb,) = torch.autograd.grad((vb[mb] ** 2).sum(), leaf_b)
        torch.testing.assert_close(ga, gb, rtol=0, atol=1e-9)
        assert kernel.edge_vec64.shape[0] == kernel.capacity
        assert torch_path.edge_vec64.shape[0] == 1

    def test_requested_dtype(self):
        coord, _, cell = frame(n_mol=16, box=14.0)
        lst = searched(coord, cell)
        assert lst.graph().edge_vec.dtype == torch.float32
        assert lst.graph(dtype=torch.float64).edge_vec.dtype == torch.float64
        assert (
            lst.graph(node_coord=coord[0], dtype=torch.float32).edge_vec.dtype
            == torch.float32
        )
