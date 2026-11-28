# SPDX-License-Identifier: MIT
r"""DPA4C's vector and tensor moments from a radial table, for a compressed model.

A multipole head on DPA4C reads the descriptor's degree-wise moments

.. math::
    X^{(\ell)}_{icm} = \Bigl(\sum_j \chi_{ij}^4 + \tfrac14\Bigr)^{-1/2}
        \sum_j \chi_{ij}\,\phi_{ijc}\,B^{(\ell)}_m(\hat u_{ij}),
    \qquad \ell = 1, 2,

which the uncompressed descriptor builds on its way to the invariant
features. The compressed kernel builds them on the device and returns the
invariant features only. This module computes the few the head needs beside
the kernel, so a model with a multipole head keeps the compressed backbone.

For one frame condition, :math:`\chi\phi_c` is a function of the distance
alone for each ordered pair of types, and the moments of degree 1 and 2 read
only the leading channels (8 and 4 of 32). So those functions and
:math:`\chi^4` are tabulated once on a uniform grid, values and radial
derivatives, by running the descriptor's own edge stage on a row of edges
per pair; nothing of that stage is repeated here. A step then costs a
cubic Hermite lookup per edge, the harmonics, and one reduction: no network
on the edge axis. The interpolant is :math:`C^1` and its gradient is taken
through the interpolation formula, so forces are the exact derivative of
the energy the table gives.

The table holds the types it has met, and grows when a new one appears: a
model with the whole periodic table in its type map tabulates the pairs of
the system at hand. Like the compressed kernel's own tables it is tied to
the frame condition folded in at compression, and is rebuilt when that
changes.

In an evaluation on CUDA the lookup, the harmonics and the reduction run as
one Triton kernel with a hand-written backward
(:mod:`deepmd_lr.kernels.moments`); ``DP_LR_TRITON=0`` selects the plain
torch route, which gives the same numbers and is the one with a second
derivative.
"""

import contextlib
from typing import (
    Any,
)

import torch
from deepmd.dpmodel.descriptor.dpa4c import DescrptDPA4C as DescrptDPA4CDP
from deepmd.dpmodel.descriptor.dpa4c_nn.geometry import (
    build_angular_basis,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)

from deepmd_lr.fitting.multipole import (
    equivariant_channels,
)
from deepmd_lr.kernels.moments import (
    available,
    table_moments,
)

__all__ = ["MomentTable", "TABLE_STRIDE"]

#: Grid spacing of the radial table, in angstrom. The Hermite error goes as
#: its fourth power (1e-10 of the moments here), while the float32 rounding
#: of the tabulated values enters the slope divided by it: finer is not better.
TABLE_STRIDE = 0.01


@contextlib.contextmanager
def _capturing_edges(descriptor: Any, into: dict) -> Any:
    """Keep what ``descriptor`` hands its moment reduction, for one pass.

    The reduction called is the class's own, and whatever the instance
    carried under that name (a model capturing its backbone hooks it too) is
    put back afterwards.
    """
    own = descriptor.__dict__.get("aggregate_moments")
    reduce = type(descriptor).aggregate_moments

    def capture(amplitude: Any, basis: Any, envelope: Any, *rest: Any) -> Any:
        into.update(amplitude=amplitude, envelope=envelope)
        return reduce(descriptor, amplitude, basis, envelope, *rest)

    descriptor.aggregate_moments = capture
    amp = getattr(descriptor, "use_amp_infer", False)
    descriptor.use_amp_infer = False  # the table is built in full precision
    try:
        yield
    finally:
        if own is None:
            del descriptor.aggregate_moments
        else:
            descriptor.aggregate_moments = own
        descriptor.use_amp_infer = amp


class MomentTable:
    """The degree 1 (and 2) moments of a DPA4C descriptor from tabulated
    radial functions.

    Parameters
    ----------
    descriptor
        The DPA4C descriptor, compressed or not; its uncompressed edge stage
        is what fills the table.
    lmax
        1 for the vectors, 2 for the tensors too.
    stride
        Grid spacing, in angstrom.
    """

    def __init__(
        self, descriptor: Any, lmax: int, stride: float = TABLE_STRIDE
    ) -> None:
        self.descriptor = descriptor
        self.lmax = int(lmax)
        #: Channels of each degree, and of the table (the widest, the first).
        self.widths = equivariant_channels(descriptor, self.lmax)
        self.rcut = float(descriptor.get_rcut())
        self.knots = int(-(-self.rcut // stride))
        self.stride = self.rcut / self.knots
        self.ntypes = int(descriptor.get_ntypes())
        self.floor = float(descriptor._DEGREE_NORM_FLOOR)
        self.eps = float(descriptor._EPS)
        #: ``(P * (knots + 1), 2, C_1 + 1)``: value and radial derivative of
        #: ``chi phi_c`` and of ``chi^4`` for each tabulated ordered pair.
        self.table: torch.Tensor | None = None
        #: ``((ntypes + 1)^2,)``: the row block of an ordered pair, -1 if
        #: not tabulated.
        self.pair_slot: torch.Tensor | None = None
        #: The types tabulated, and whether those are all of them.
        self._known: set[int] = set()
        self._complete = False
        self._condition: Any = None

    # -- building ---------------------------------------------------------

    def build(self, types: list[int], condition: Any, device: Any) -> None:
        """Tabulate every ordered pair of ``types`` at the frame condition
        ``condition`` (``[charge, multiplicity]`` or None)."""
        descriptor = self.descriptor
        count = len(types)
        kind = torch.as_tensor(types, dtype=torch.int64, device=device)
        # One centre and one neighbour node per ordered pair, and a row of
        # edges between them at the grid distances.
        center = kind.repeat_interleave(count)
        neighbor = kind.repeat(count)
        pairs = count * count
        atype = torch.stack([center, neighbor], dim=1).reshape(-1)
        rows = self.knots + 1
        radius = torch.arange(rows, dtype=torch.float64, device=device) * self.stride
        dtype = next(descriptor.parameters()).dtype
        edge_vec = torch.zeros(pairs * rows, 3, dtype=dtype, device=device)
        edge_vec[:, 0] = radius.to(dtype).repeat(pairs)
        edge_vec.requires_grad_(True)
        pair = torch.arange(pairs, device=device).repeat_interleave(rows)
        graph = NeighborGraph(
            n_node=torch.tensor([2 * pairs], device=device),
            edge_index=torch.stack([2 * pair + 1, 2 * pair]),
            edge_vec=edge_vec,
            edge_mask=torch.ones(pairs * rows, dtype=torch.bool, device=device),
        )
        seen: dict = {}
        with torch.enable_grad(), _capturing_edges(descriptor, seen):
            # The uncompressed equations, whatever the descriptor dispatches
            # to: the compressed kernel does not hand out its edges.
            DescrptDPA4CDP.call_graph(
                descriptor,
                graph,
                atype,
                type_embedding=descriptor.type_embedding.call(),
                charge_spin=None
                if condition is None
                else torch.as_tensor([condition], dtype=dtype, device=device),
            )
            envelope = seen["envelope"]
            columns = torch.cat(
                [
                    seen["amplitude"][:, : self.widths[0]] * envelope[:, None],
                    (envelope**4)[:, None],
                ],
                dim=1,
            )
            # Each edge is its own function of its own distance.
            slopes = torch.stack(
                [
                    torch.autograd.grad(
                        columns[:, c].sum(), edge_vec, retain_graph=True
                    )[0][:, 0]
                    for c in range(columns.shape[1])
                ],
                dim=1,
            )
        self.table = torch.stack([columns.detach(), slopes], dim=1).contiguous()
        slot = torch.full(
            ((self.ntypes + 1) ** 2,), -1, dtype=torch.int64, device=device
        )
        slot[center * (self.ntypes + 1) + neighbor] = torch.arange(pairs, device=device)
        self.pair_slot = slot
        self._known = set(types)
        self._complete = len(self._known) == self.ntypes
        self._condition = condition

    def ensure(self, atype: torch.Tensor, condition: Any) -> None:
        """Tabulate what ``atype`` needs and the table does not hold yet."""
        reusable = (
            self.table is not None
            and condition == self._condition
            and self.table.device == atype.device
        )
        if reusable and self._complete:
            return  # every type there is: nothing to read off the device
        present = set(atype[(atype >= 0) & (atype < self.ntypes)].unique().tolist())
        if reusable:
            if present <= self._known:
                return
            present |= self._known
        self.build(sorted(present), condition, atype.device)

    # -- evaluating -------------------------------------------------------

    def __call__(
        self,
        graph: Any,
        atype: torch.Tensor,
        condition: Any = None,
    ) -> torch.Tensor:
        """The moments on ``graph`` in the descriptor's own flat layout,
        ``(N, S)`` with the scalar block left zero: what
        :func:`~deepmd_lr.fitting.multipole.cartesian_features` reads."""
        self.ensure(atype, condition)
        edge_vec = graph.edge_vec
        n_total = atype.shape[0]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        base = self.ntypes + 1
        kinds = atype.clamp(0, self.ntypes)
        slot = self.pair_slot[kinds[dst] * base + kinds[src]]
        valid = graph.edge_mask & (slot >= 0)
        row = slot.clamp(min=0) * (self.knots + 1)
        table = self.table.to(edge_vec.dtype)
        if self.fused(edge_vec):
            reduced = table_moments(
                edge_vec, dst, row, valid, table, n_total,
                self.widths, self.stride, self.knots, self.eps,
            )  # fmt: skip
        else:
            reduced = self.reduce(edge_vec, dst, row, valid, table, n_total)
        moments = reduced[:, 1:] / torch.sqrt(reduced[:, :1] + self.floor)
        scalar = moments.new_zeros(n_total, int(self.descriptor.degree_offsets[1]))
        return torch.cat([scalar, moments], dim=1)

    def reduce(
        self,
        edge_vec: torch.Tensor,
        dst: torch.Tensor,
        row: torch.Tensor,
        valid: torch.Tensor,
        table: torch.Tensor,
        n_total: int,
    ) -> torch.Tensor:
        """The plain torch route to the unnormalised sums, ``(N, 1 + 3 C_1
        [+ 5 C_2])``: the envelope weight, the vectors, the tensors."""
        distance = torch.sqrt((edge_vec * edge_vec).sum(-1) + self.eps * self.eps)
        position = (distance / self.stride).clamp(max=self.knots)
        cell = position.detach().floor().clamp(max=self.knots - 1)
        u = (position - cell)[:, None]
        index = row + cell.to(torch.int64)
        low, high = table[index], table[index + 1]
        v = 1.0 - u
        # Around the lower knot, the difference of the two values first: it
        # is exact in floating point, and the slope autograd takes keeps it.
        value = (
            low[:, 0]
            + (high[:, 0] - low[:, 0]) * (u * u * (3.0 - 2.0 * u))
            + self.stride * u * v * (v * low[:, 1] - u * high[:, 1])
        ) * valid[:, None].to(edge_vec.dtype)
        radial, weight = value[:, :-1], value[:, -1:]
        basis = build_angular_basis(edge_vec / distance[:, None], self.lmax)
        parts = [weight, (basis[:, 1:4, None] * radial[:, None, :]).flatten(1)]
        if self.lmax >= 2:
            narrow = radial[:, None, : self.widths[1]]
            parts.append((basis[:, 4:9, None] * narrow).flatten(1))
        payload = torch.cat(parts, dim=1)
        return payload.new_zeros(n_total, payload.shape[1]).index_add_(0, dst, payload)

    def fused(self, edge_vec: torch.Tensor) -> bool:
        """Whether this evaluation takes the Triton kernel: float32 on CUDA,
        outside training (the kernel has no second derivative), Triton
        installed and not switched off with ``DP_LR_TRITON=0``."""
        return (
            edge_vec.is_cuda
            and edge_vec.dtype == torch.float32
            and not self.descriptor.training
            and available()
        )
