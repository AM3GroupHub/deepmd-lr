# SPDX-License-Identifier: LGPL-3.0-or-later
"""The edge half of the Born tensor, summed along the graph's CSR rows.

:func:`deepmd_lr.born.edge_born_response` forms every edge's outer product
``r_e (x) g_e`` as an ``(E, 3, 3)`` tensor and scatter-adds it twice, onto
the destination and, negated, onto the source: at 175M edges that is a 6 GB
temporary and two passes of atomics. The graph the GPU-direct list builds
already groups the edges by destination (its physical order) and by source
(``source_order``), so one thread per node can walk its two rows and
accumulate the same sums in registers -- deterministic, no temporary, two
reads of the edge arrays.
"""

import torch
import warp as wp

from deepmd_lr.nvalchemi.nlist import _torch_stream_scope

__all__ = ["edge_born_response_csr"]


@wp.kernel(enable_backward=False)
def _born_rows(
    g: wp.array(dtype=wp.vec3f),
    vec: wp.array(dtype=wp.vec3f),
    dst_row_ptr: wp.array(dtype=wp.int64),
    source_order: wp.array(dtype=wp.int64),
    src_row_ptr: wp.array(dtype=wp.int64),
    out: wp.array(dtype=wp.mat33d),
):
    """Node ``i``: half the outer products summed over the edges into it,
    minus half those over the edges out of it."""
    i = wp.tid()
    acc = wp.mat33d()
    for e in range(int(dst_row_ptr[i]), int(dst_row_ptr[i + 1])):
        r = vec[e]
        gg = g[e]
        acc = acc + wp.outer(
            wp.vec3d(wp.float64(r[0]), wp.float64(r[1]), wp.float64(r[2])),
            wp.vec3d(wp.float64(gg[0]), wp.float64(gg[1]), wp.float64(gg[2])),
        )
    for k in range(int(src_row_ptr[i]), int(src_row_ptr[i + 1])):
        e = int(source_order[k])
        r = vec[e]
        gg = g[e]
        acc = acc - wp.outer(
            wp.vec3d(wp.float64(r[0]), wp.float64(r[1]), wp.float64(r[2])),
            wp.vec3d(wp.float64(gg[0]), wp.float64(gg[1]), wp.float64(gg[2])),
        )
    out[i] = acc * wp.float64(0.5)


def edge_born_response_csr(
    g: torch.Tensor,
    vec: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    source_row_ptr: torch.Tensor,
    n_node: int,
) -> torch.Tensor:
    """:func:`deepmd_lr.born.edge_born_response` over the CSR rows.

    ``g`` and ``vec`` are ``(E, 3)`` float32 on one CUDA device, ``E`` the
    graph's edge capacity; only the edges the row pointers address are read,
    so neither a mask nor the tail matters. The row pointers are int64 of
    length ``n_node + 1`` and ``source_order`` the source-grouped permutation,
    as the compressed kernels take them. Returns ``(n_node, 3, 3)`` float64,
    accumulated in float64.
    """
    if g.dtype != torch.float32 or vec.dtype != torch.float32:
        raise TypeError("edge_born_response_csr takes float32 g and vec")
    if g.device.type != "cuda":
        raise ValueError("edge_born_response_csr needs a CUDA device")
    g = g.detach().contiguous()
    vec = vec.detach().contiguous()
    out = torch.zeros((n_node, 3, 3), dtype=torch.float64, device=g.device)
    with _torch_stream_scope(g.device):
        wp.launch(
            _born_rows,
            dim=n_node,
            inputs=[
                wp.from_torch(
                    g, dtype=wp.vec3f, requires_grad=False, return_ctype=True
                ),
                wp.from_torch(
                    vec, dtype=wp.vec3f, requires_grad=False, return_ctype=True
                ),
                wp.from_torch(
                    destination_row_ptr.contiguous(),
                    dtype=wp.int64,
                    requires_grad=False,
                    return_ctype=True,
                ),
                wp.from_torch(
                    source_order.contiguous(),
                    dtype=wp.int64,
                    requires_grad=False,
                    return_ctype=True,
                ),
                wp.from_torch(
                    source_row_ptr.contiguous(),
                    dtype=wp.int64,
                    requires_grad=False,
                    return_ctype=True,
                ),
                wp.from_torch(out, dtype=wp.mat33d, requires_grad=False),
            ],
            device=str(g.device),
        )
    return out
