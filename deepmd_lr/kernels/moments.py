# SPDX-License-Identifier: MIT
r"""One Triton kernel for the table moments of a compressed DPA4C, and its backward.

:class:`~deepmd_lr.fitting.moment_table.MomentTable` turns each edge into a
cubic Hermite lookup :math:`G_c(r)` and :math:`W(r)`, the harmonics of the
edge's direction, and a sum onto the edge's centre:

.. math::
    R_i = \Bigl[\sum_j W,\ \sum_j G_c\,\hat u_a,\ \sum_j G_c\,B^{(2)}_m(\hat u)\Bigr].

In plain torch that is a dozen tensors of the size of the edge list going
forward and as many coming back. Here the forward is one pass over the edges
that adds into the node rows, and the backward one pass that reads the node
rows' gradient and writes the edge vectors' gradient:

.. math::
    \frac{\partial L}{\partial v} = s\,\hat u
        + \frac{h - \hat u\,(h\cdot\hat u)}{r},
    \quad s = \frac{\partial L}{\partial r}\Big|_{\hat u},
    \quad h = \frac{\partial L}{\partial \hat u}\Big|_{r} .

First derivatives only: forces, the virial and Born tensors of an evaluation.
A second derivative (training on forces) takes the torch route.
"""

import math
import os
from typing import (
    Any,
)

import torch
from torch.autograd.function import (
    once_differentiable,
)

__all__ = ["available", "table_moments"]

_BLOCK = 64

try:
    import triton
    import triton.language as tl
except ImportError:  # the torch route serves
    triton = None


def available() -> bool:
    """Whether the kernel can run here: Triton, CUDA, and not switched off
    with ``DP_LR_TRITON=0``."""
    return (
        triton is not None
        and torch.cuda.is_available()
        and os.environ.get("DP_LR_TRITON", "1").strip() != "0"
    )


if triton is not None:

    @triton.jit
    def _hermite(low, slope_low, high, slope_high, u, stride):
        # Around the lower knot, with the difference of the two values taken
        # first: it is exact in floating point, where the four-weight form
        # loses the digits its slope needs.
        v = 1.0 - u
        return (
            low
            + (high - low) * (u * u * (3.0 - 2.0 * u))
            + stride * u * v * (v * slope_low - u * slope_high)
        )

    @triton.jit
    def _hermite_slope(low, slope_low, high, slope_high, u, stride):
        # d/dr of the interpolant: (1/stride) d/du.
        return (
            (high - low) * (6.0 * u * (1.0 - u) / stride)
            + (3.0 * u * u - 4.0 * u + 1.0) * slope_low
            + (3.0 * u * u - 2.0 * u) * slope_high
        )

    @triton.jit
    def _forward(
        vec_ptr, dst_ptr, row_ptr, valid_ptr, table_ptr, out_ptr,
        n_edge, stride, knots, span, eps2,
        C1: tl.constexpr, C2: tl.constexpr, LMAX: tl.constexpr,
        WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        edge = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inside = edge < n_edge
        x = tl.load(vec_ptr + 3 * edge, mask=inside, other=0.0)
        y = tl.load(vec_ptr + 3 * edge + 1, mask=inside, other=0.0)
        z = tl.load(vec_ptr + 3 * edge + 2, mask=inside, other=0.0)
        dst = tl.load(dst_ptr + edge, mask=inside, other=0)
        row = tl.load(row_ptr + edge, mask=inside, other=0)
        keep = inside & (tl.load(valid_ptr + edge, mask=inside, other=0) != 0)
        r = tl.sqrt(x * x + y * y + z * z + eps2)
        position = tl.minimum(r / stride, span)
        cell = tl.minimum(position.to(tl.int64), knots - 1)
        u = position - cell.to(tl.float32)
        cols = C1 + 1
        base = (row + cell) * (2 * cols)
        c = tl.arange(0, C1)
        at = table_ptr + base[:, None] + c[None, :]
        wide = keep[:, None]
        g = _hermite(
            tl.load(at, mask=wide, other=0.0),
            tl.load(at + cols, mask=wide, other=0.0),
            tl.load(at + 2 * cols, mask=wide, other=0.0),
            tl.load(at + 3 * cols, mask=wide, other=0.0),
            u[:, None], stride,
        )  # fmt: skip
        last = table_ptr + base + C1
        w = _hermite(
            tl.load(last, mask=keep, other=0.0),
            tl.load(last + cols, mask=keep, other=0.0),
            tl.load(last + 2 * cols, mask=keep, other=0.0),
            tl.load(last + 3 * cols, mask=keep, other=0.0),
            u, stride,
        )  # fmt: skip
        ux, uy, uz = x / r, y / r, z / r
        node = out_ptr + dst * WIDTH
        tl.atomic_add(node, w, mask=keep)
        first = node[:, None] + 1 + c[None, :]
        tl.atomic_add(first, g * ux[:, None], mask=wide)
        tl.atomic_add(first + C1, g * uy[:, None], mask=wide)
        tl.atomic_add(first + 2 * C1, g * uz[:, None], mask=wide)
        if LMAX >= 2:
            narrow = wide & (c[None, :] < C2)
            second = first + 3 * C1
            root = 1.7320508075688772
            norm = ux * ux + uy * uy + uz * uz
            tl.atomic_add(second, g * (root * ux * uy)[:, None], mask=narrow)
            tl.atomic_add(second + C2, g * (root * uy * uz)[:, None], mask=narrow)
            tl.atomic_add(
                second + 2 * C2,
                g * (0.5 * (3.0 * uz * uz - norm))[:, None],
                mask=narrow,
            )
            tl.atomic_add(second + 3 * C2, g * (root * ux * uz)[:, None], mask=narrow)
            tl.atomic_add(
                second + 4 * C2,
                g * (0.5 * root * (ux * ux - uy * uy))[:, None],
                mask=narrow,
            )

    @triton.jit
    def _backward(
        vec_ptr, dst_ptr, row_ptr, valid_ptr, table_ptr, grad_ptr, out_ptr,
        n_edge, stride, knots, span, eps2,
        C1: tl.constexpr, C2: tl.constexpr, LMAX: tl.constexpr,
        WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        edge = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inside = edge < n_edge
        x = tl.load(vec_ptr + 3 * edge, mask=inside, other=0.0)
        y = tl.load(vec_ptr + 3 * edge + 1, mask=inside, other=0.0)
        z = tl.load(vec_ptr + 3 * edge + 2, mask=inside, other=0.0)
        dst = tl.load(dst_ptr + edge, mask=inside, other=0)
        row = tl.load(row_ptr + edge, mask=inside, other=0)
        keep = inside & (tl.load(valid_ptr + edge, mask=inside, other=0) != 0)
        r = tl.sqrt(x * x + y * y + z * z + eps2)
        scaled = r / stride
        # Beyond the last knot the lookup is clamped: no radial slope there.
        moving = keep & (scaled < span)
        position = tl.minimum(scaled, span)
        cell = tl.minimum(position.to(tl.int64), knots - 1)
        u = position - cell.to(tl.float32)
        cols = C1 + 1
        base = (row + cell) * (2 * cols)
        c = tl.arange(0, C1)
        at = table_ptr + base[:, None] + c[None, :]
        wide = keep[:, None]
        low = tl.load(at, mask=wide, other=0.0)
        slope_low = tl.load(at + cols, mask=wide, other=0.0)
        high = tl.load(at + 2 * cols, mask=wide, other=0.0)
        slope_high = tl.load(at + 3 * cols, mask=wide, other=0.0)
        g = _hermite(low, slope_low, high, slope_high, u[:, None], stride)
        dg = _hermite_slope(low, slope_low, high, slope_high, u[:, None], stride)
        last = table_ptr + base + C1
        dw = _hermite_slope(
            tl.load(last, mask=keep, other=0.0),
            tl.load(last + cols, mask=keep, other=0.0),
            tl.load(last + 2 * cols, mask=keep, other=0.0),
            tl.load(last + 3 * cols, mask=keep, other=0.0),
            u, stride,
        )  # fmt: skip
        ux, uy, uz = x / r, y / r, z / r
        node = grad_ptr + dst * WIDTH
        first = node[:, None] + 1 + c[None, :]
        gx = tl.load(first, mask=wide, other=0.0)
        gy = tl.load(first + C1, mask=wide, other=0.0)
        gz = tl.load(first + 2 * C1, mask=wide, other=0.0)
        # h = dL/du at fixed r, s = dL/dr at fixed u.
        hx = tl.sum(gx * g, axis=1)
        hy = tl.sum(gy * g, axis=1)
        hz = tl.sum(gz * g, axis=1)
        s = tl.load(node, mask=keep, other=0.0) * dw + tl.sum(
            (gx * ux[:, None] + gy * uy[:, None] + gz * uz[:, None]) * dg, axis=1
        )
        if LMAX >= 2:
            narrow = wide & (c[None, :] < C2)
            second = first + 3 * C1
            root = 1.7320508075688772
            norm = ux * ux + uy * uy + uz * uz
            q0 = tl.load(second, mask=narrow, other=0.0)
            q1 = tl.load(second + C2, mask=narrow, other=0.0)
            q2 = tl.load(second + 2 * C2, mask=narrow, other=0.0)
            q3 = tl.load(second + 3 * C2, mask=narrow, other=0.0)
            q4 = tl.load(second + 4 * C2, mask=narrow, other=0.0)
            b0 = root * ux * uy
            b1 = root * uy * uz
            b2 = 0.5 * (3.0 * uz * uz - norm)
            b3 = root * ux * uz
            b4 = 0.5 * root * (ux * ux - uy * uy)
            s += tl.sum(
                (
                    q0 * b0[:, None]
                    + q1 * b1[:, None]
                    + q2 * b2[:, None]
                    + q3 * b3[:, None]
                    + q4 * b4[:, None]
                )
                * dg,
                axis=1,
            )
            a0 = tl.sum(q0 * g, axis=1)
            a1 = tl.sum(q1 * g, axis=1)
            a2 = tl.sum(q2 * g, axis=1)
            a3 = tl.sum(q3 * g, axis=1)
            a4 = tl.sum(q4 * g, axis=1)
            hx += root * (a0 * uy + a3 * uz + a4 * ux) - a2 * ux
            hy += root * (a0 * ux + a1 * uz - a4 * uy) - a2 * uy
            hz += root * (a1 * uy + a3 * ux) + 2.0 * a2 * uz
        s = tl.where(moving, s, 0.0)
        along = hx * ux + hy * uy + hz * uz
        tl.store(out_ptr + 3 * edge, s * ux + (hx - ux * along) / r, mask=inside)
        tl.store(out_ptr + 3 * edge + 1, s * uy + (hy - uy * along) / r, mask=inside)
        tl.store(out_ptr + 3 * edge + 2, s * uz + (hz - uz * along) / r, mask=inside)


class _TableMoments(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        edge_vec: torch.Tensor,
        dst: torch.Tensor,
        row: torch.Tensor,
        valid: torch.Tensor,
        table: torch.Tensor,
        n_total: int,
        widths: tuple,
        stride: float,
        knots: int,
        eps: float,
    ) -> torch.Tensor:
        c1 = widths[0]
        c2 = widths[1] if len(widths) > 1 else c1
        width = 1 + 3 * c1 + (5 * c2 if len(widths) > 1 else 0)
        edge_vec = edge_vec.contiguous()
        valid = valid.to(torch.uint8)
        out = edge_vec.new_zeros(n_total, width)
        n_edge = edge_vec.shape[0]
        constants = {
            "C1": c1, "C2": c2, "LMAX": len(widths), "WIDTH": width,
            "BLOCK": _BLOCK,
        }  # fmt: skip
        scalars = (n_edge, stride, int(knots), float(knots), eps * eps)
        if n_edge:
            _forward[(math.ceil(n_edge / _BLOCK),)](
                edge_vec, dst, row, valid, table, out, *scalars, **constants
            )
        ctx.save_for_backward(edge_vec, dst, row, valid, table)
        ctx.launch = (scalars, constants)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple:
        edge_vec, dst, row, valid, table = ctx.saved_tensors
        scalars, constants = ctx.launch
        out = torch.zeros_like(edge_vec)
        if scalars[0]:
            _backward[(math.ceil(scalars[0] / _BLOCK),)](
                edge_vec, dst, row, valid, table, grad.contiguous(), out,
                *scalars, **constants,
            )  # fmt: skip
        return (out,) + (None,) * 9


def table_moments(
    edge_vec: torch.Tensor,
    dst: torch.Tensor,
    row: torch.Tensor,
    valid: torch.Tensor,
    table: torch.Tensor,
    n_total: int,
    widths: list[int],
    stride: float,
    knots: int,
    eps: float,
) -> torch.Tensor:
    """The unnormalised sums ``(N, 1 + 3 C_1 [+ 5 C_2])``: the envelope
    weight, then the vector and the tensor block in the descriptor's layout.

    ``edge_vec`` ``(E, 3)`` float32 on CUDA, ``dst`` and ``row`` ``(E,)``
    int64 (the centre, and the first table row of the edge's pair), ``valid``
    ``(E,)`` bool, ``table`` ``(rows, 2, C_1 + 1)`` float32 contiguous.
    Differentiable once with respect to ``edge_vec``.
    """
    return _TableMoments.apply(
        edge_vec, dst, row, valid, table, n_total, tuple(widths), stride, knots, eps
    )
