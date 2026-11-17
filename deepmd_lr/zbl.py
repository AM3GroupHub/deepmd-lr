# SPDX-License-Identifier: MIT
"""Short contacts on the DPA4C neighbour graph: ZBL takes over from the network.

A network fitted to condensed-phase data has seen no pair closer than its
shortest bond, and below that it extrapolates freely. On a water model an O-H
pair pushed from 0.70 to 0.60 A drops the fitted energy by 230 eV, all of it
on the two atoms in contact, and a little further in their predicted charges
reach tens of e. A few eV of pair repulsion added on top change nothing
there, so this module does two things across one window:

- **ZBL in**: the screened nuclear repulsion of Ziegler, Biersack and
  Littmark, ``V_ZBL(r) (1 - s)``;
- **the network out**: each atom's fitted energy and raw charge are
  multiplied by the envelope ``w_i = prod_j s(r_ij)``, the product over its
  neighbours (mace-lr's ``charge_net.envelope``, here on the energy too).

The window is ``[x_lo, x_hi]`` in units of the pair's covalent-radius sum,
``x = r / (R_i + R_j)`` with ASE's radii, and one C2 step
``s(x) = t^3 (10 - 15 t + 6 t^2)``, ``t = (x - x_lo) / (x_hi - x_lo)`` clamped
to [0, 1], is exactly 0 below it and exactly 1 above. Below ``x_lo`` a pair is
ZBL alone; with every neighbour beyond ``x_hi`` an atom's factor is exactly 1.
Choose ``x_hi`` at or below the shortest contact in the data, so that nothing
fitted changes, and ``x_lo`` above where the network starts to fall.

ZBL has no parameters and sees the edge vectors through no network, so its
force and virial are assembled analytically from the per-edge derivative,
through the edge-to-node reduction DeePMD-kit uses for the learned energy (its
fused CUDA kernel when ``DP_CUDA_INFER`` asks for it). On a CUDA device one
Warp kernel produces the per-edge energy and derivative; elsewhere, and under
a tracer, the same arithmetic runs as torch ops. The envelope multiplies
network outputs, so it is a differentiable function of the edge vectors and
whatever is differentiated through those outputs -- force, virial, the charge
response, Born charges -- picks its derivative up on the way.
"""

import math
from numbers import Real
from typing import Any

import torch
from ase.data import atomic_numbers, covalent_radii
from deepmd.dpmodel.model.edge_transform_output import node_ownership_mask
from deepmd.dpmodel.utils.neighbor_graph.derivatives import edge_force_virial
from deepmd.dpmodel.utils.neighbor_graph.graph import frame_id_from_n_node
from deepmd.dpmodel.utils.neighbor_graph.segment import segment_sum
from deepmd.kernels.cuda.edge_force_virial import (
    edge_force_virial as fused_edge_force_virial,
)
from deepmd.kernels.cuda.edge_force_virial import (
    op_available as fused_scatter_available,
)
from deepmd.kernels.utils import cuda_infer_level

try:
    import warp as wp
except ImportError:  # pragma: no cover - the MD extra is optional
    wp = None

__all__ = ["ZBL", "ZBL_KEY"]

#: Key of the optional block in the model config.
ZBL_KEY = "zbl"

#: Coulomb constant, eV A.
_KE = 14.3996454784255
#: ZBL universal screening: ``a = 0.46850 / (Z_i^0.23 + Z_j^0.23)`` A.
_A_PREFACTOR = 0.46850
_A_EXPONENT = 0.23
_PHI_A = (0.18175, 0.50986, 0.28022, 0.02817)
_PHI_B = (3.19980, 0.94229, 0.40290, 0.20162)

_KEYS = ("window",)
#: Keys of the first implementation (October 2026), named in the refusal so a
#: checkpoint or input that still carries them explains itself.
_RETIRED = (
    "cutoff",
    "r_inner",
    "r_outer",
    "covalent_scale",
    "inner_ratio",
    "clamp",
    "clamp_width",
)


def _tracing(t: torch.Tensor) -> bool:
    """Whether ``t`` is being traced (export, compiled training) rather than
    computed: a Warp launch is a side effect no tracer records, so the trace
    must take the torch arithmetic instead."""
    if torch.compiler.is_compiling():
        return True
    fx = torch.fx._symbolic_trace
    # `is_fx_tracing` warns since torch split it into these two questions.
    if getattr(fx, "is_fx_symbolic_tracing", fx.is_fx_tracing)():
        return True
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return isinstance(t, (FakeTensor, FunctionalTensor)) or get_proxy_mode() is not None


def check_window(window: Any) -> tuple[float, float]:
    """Validate ``window = [x_lo, x_hi]``: finite, ``0 <= x_lo < x_hi``."""
    try:
        x_lo, x_hi = window
    except (TypeError, ValueError):
        raise ValueError(
            "zbl.window must be a pair [x_lo, x_hi] in covalent-radius sums"
        ) from None
    for x in (x_lo, x_hi):
        if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x):
            raise ValueError("zbl.window must hold two finite real numbers")
    if not 0.0 <= x_lo < x_hi:
        raise ValueError(
            f"zbl.window needs 0 <= x_lo < x_hi, got [{x_lo}, {x_hi}]: ZBL is "
            "full below x_lo and gone above x_hi"
        )
    return float(x_lo), float(x_hi)


class ZBL(torch.nn.Module):
    """Screened nuclear repulsion inside a short-contact window, and the
    envelope that switches the network off there; no parameters.

    ``settings`` is the ``zbl`` block of the model config, ``{"window":
    [x_lo, x_hi]}`` in covalent-radius sums. There is no default window: it
    belongs just below the shortest contact of the data the model was fitted
    to, which differs from system to system.
    """

    def __init__(self, type_map: list[str], settings: dict[str, Any], device=None):
        super().__init__()
        if not isinstance(settings, dict):
            raise ValueError("zbl must be a configuration dictionary")
        settings = {k: v for k, v in settings.items() if not k.startswith("_")}
        unknown = set(settings) - set(_KEYS)
        if unknown:
            retired = sorted(unknown & set(_RETIRED))
            raise ValueError(
                f"zbl accepts only {list(_KEYS)}, got {sorted(unknown)}"
                + (
                    f". {retired} belonged to the first implementation; the "
                    "switch is now one window, [x_lo, x_hi] in covalent-radius "
                    "sums"
                    if retired
                    else ""
                )
            )
        if "window" not in settings:
            raise ValueError(
                "zbl.window is required: [x_lo, x_hi] in covalent-radius sums, "
                "with x_hi at or below the shortest contact in the data"
            )
        self.x_lo, self.x_hi = check_window(settings["window"])
        # A type that is not an element -- ASE's dummy "X" heads this
        # repository's periodic-table type maps -- has no nucleus: Z = 0, and
        # every pair it is in contributes nothing.
        numbers = [atomic_numbers.get(s, 0) for s in type_map]
        if not any(z > 0 for z in numbers):
            raise ValueError("zbl requires at least one element symbol in type_map")
        radii = [float(covalent_radii[z]) if z > 0 else 0.0 for z in numbers]
        #: The largest distance at which any pair is still inside its window.
        #: A plain float, fixed here: the kernels take it every step, and
        #: reading it off a device buffer would be a host synchronisation.
        self.max_outer = 2.0 * max(radii) * self.x_hi
        radius = torch.tensor(radii, dtype=torch.float64, device=device)
        real = torch.tensor([z > 0 for z in numbers], device=device)
        # A pair with a type that is no element gets an empty window, so no
        # distance is ever inside it.
        both = (real[:, None] & real[None, :]).to(torch.float64)
        pair_sum = (radius[:, None] + radius[None, :]) * both
        # Derived entirely from the serialized type_map and window: not
        # optimizer state, and no state_dict entry.
        self.register_buffer(
            "numbers",
            torch.tensor(numbers, dtype=torch.float64, device=device),
            persistent=False,
        )
        #: Per pair of types, the distances where the window starts and ends,
        #: in angstrom. Tables rather than one pair of numbers, so a window
        #: per pair needs no change below this line.
        self.register_buffer("r_lo", self.x_lo * pair_sum, persistent=False)
        self.register_buffer("r_hi", self.x_hi * pair_sum, persistent=False)

    def serialize(self) -> dict[str, Any]:
        return {"window": [self.x_lo, self.x_hi]}

    # ------------------------------------------------------------ per edge

    def _pair_window(
        self, atype: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each edge's ``(r_lo, r_hi)`` and its two types, clamped into the
        table so a padded node (type -1) indexes somewhere harmless."""
        ti = atype.clamp(0, self.numbers.numel() - 1).long()
        ts, td = ti[src], ti[dst]
        return self.r_lo[ts, td], self.r_hi[ts, td], ti

    def _edge_terms_torch(
        self,
        atype: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_vec: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Half pair energy per edge, (E,) float64, and ``dE_edge/d edge_vec``,
        (E, 3) in ``edge_vec``'s dtype. Portable torch arithmetic; the Warp
        kernel below is the same formulas in one pass."""
        r2 = edge_vec.to(torch.float64).square().sum(-1)
        # Padded edges carry a zero vector: give them a harmless distance.
        r = torch.sqrt(torch.where(valid, r2, torch.ones_like(r2)))
        ri, ro, ti = self._pair_window(atype, src, dst)
        z = self.numbers[ti]
        zi, zj = z[src], z[dst]
        a = _A_PREFACTOR / (zi.pow(_A_EXPONENT) + zj.pow(_A_EXPONENT))
        x = r / a
        phi = torch.zeros_like(r)
        dphi = torch.zeros_like(r)
        for ak, bk in zip(_PHI_A, _PHI_B, strict=True):
            term = ak * torch.exp(-bk * x)
            phi = phi + term
            dphi = dphi - bk * term
        coulomb = _KE * zi * zj / r
        pair = coulomb * phi
        dpair = coulomb * (dphi / a - phi / r)
        # An empty window (a pair with no nucleus) has no width to divide by.
        width = torch.where(ro > ri, ro - ri, torch.ones_like(ro))
        t = ((r - ri) / width).clamp(0.0, 1.0)
        switch = 1 - 10 * t.pow(3) + 15 * t.pow(4) - 6 * t.pow(5)
        dswitch = (-30 * t.pow(2) + 60 * t.pow(3) - 30 * t.pow(4)) / width
        on = valid & (r < ro)
        zero = torch.zeros_like(r)
        energy = torch.where(on, 0.5 * pair * switch, zero)
        dedr = torch.where(on, 0.5 * (dpair * switch + pair * dswitch), zero)
        g = (dedr / r).unsqueeze(-1) * edge_vec.to(torch.float64)
        return energy, g.to(edge_vec.dtype)

    def _edge_terms_warp(
        self,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        owned: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass over the edges: validity, pair terms, the energy summed
        onto its destination node (float64 atomics), the derivative per edge."""
        from deepmd_lr.nvalchemi.nlist import _torch_stream_scope

        n_edge = edge_vec.shape[0]
        energy = torch.zeros(
            atype.shape[0], dtype=torch.float64, device=edge_vec.device
        )
        g = torch.zeros_like(edge_vec)
        if n_edge == 0:
            return energy, g
        is32 = edge_vec.dtype == torch.float32
        vec_t = wp.vec3f if is32 else wp.vec3d
        kernel = _edge_terms_f32 if is32 else _edge_terms_f64
        # Bound to names, so no contiguous copy is freed before the launch.
        index = edge_index.to(torch.int64).contiguous()
        mask = edge_mask.to(torch.bool).contiguous()
        vec = edge_vec.detach().contiguous()
        types = atype.to(torch.int64).contiguous()
        own = owned.contiguous()

        def arr(t, dtype):
            return wp.from_torch(t, dtype=dtype, requires_grad=False)

        with _torch_stream_scope(edge_vec.device):
            wp.launch(
                kernel,
                dim=n_edge,
                inputs=[
                    arr(index, wp.int64),
                    arr(mask, wp.bool),
                    arr(vec, vec_t),
                    arr(types, wp.int64),
                    arr(own, wp.bool),
                    arr(self.numbers, wp.float64),
                    arr(self.r_lo, wp.float64),
                    arr(self.r_hi, wp.float64),
                    wp.float64(self.max_outer**2),
                    arr(energy, wp.float64),
                    arr(g, vec_t),
                ],
                device=str(edge_vec.device),
            )
        return energy, g

    def _use_warp(self, edge_vec: torch.Tensor) -> bool:
        return wp is not None and edge_vec.is_cuda and not _tracing(edge_vec)

    def edge_terms(
        self,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        owned: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-node energy, (N,) float64, and ``dE/d edge_vec`` per edge,
        (E, 3) in ``edge_vec``'s dtype.

        An edge counts when both ends are real atoms (a padded node has type
        -1) and its destination is owned: the directed graph lists every pair
        twice, each destination takes half, and a ghost's half belongs to the
        rank that owns it.
        """
        n_node = atype.shape[0]
        if owned is None:
            owned = torch.ones(n_node, dtype=torch.bool, device=atype.device)
        if self._use_warp(edge_vec):
            return self._edge_terms_warp(atype, edge_index, edge_vec, edge_mask, owned)
        src, dst = edge_index[0].long(), edge_index[1].long()
        real = (atype >= 0) & (atype < self.numbers.numel())
        valid = edge_mask.bool() & real[src] & real[dst] & owned[dst]
        e_edge, g = self._edge_terms_torch(atype, src, dst, edge_vec, valid)
        energy = torch.zeros(n_node, dtype=torch.float64, device=edge_vec.device)
        return energy.index_add(0, dst, e_edge), g

    def atomic_energy(self, atype, edge_index, edge_vec, edge_mask):
        """Per-node energy, (N, 1) float64: each destination's half pairs."""
        energy, _g = self.edge_terms(atype, edge_index, edge_vec, edge_mask)
        return energy.unsqueeze(-1)

    # ------------------------------------------------------------ envelope

    def _envelope_torch(
        self,
        atype: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_vec: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """The envelope as torch arithmetic, differentiable to any order."""
        r2 = edge_vec.square().sum(-1)
        # Padded edges carry a zero vector: give them a harmless distance.
        r = torch.sqrt(torch.where(valid, r2, torch.ones_like(r2)))
        ri, ro, _ti = self._pair_window(atype, src, dst)
        ri, ro = ri.to(r.dtype), ro.to(r.dtype)
        width = torch.where(ro > ri, ro - ri, torch.ones_like(ro))
        t = ((r - ri) / width).clamp(0.0, 1.0)
        s = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
        inside = valid & (r < ro)
        # A product of exact zeros and ones, summed as logarithms: an edge
        # outside its window adds log 1 = 0, and one at or below the lower
        # edge the constant -inf, which makes the product and its derivatives
        # exactly zero without differentiating log(0).
        closed = inside & (s <= 0)
        log_s = torch.where(inside & ~closed, s, torch.ones_like(s)).log()
        log_s = torch.where(closed, torch.full_like(s, -torch.inf), log_s)
        log_w = torch.zeros(atype.shape[0], dtype=r.dtype, device=r.device).index_add(
            0, dst, log_s
        )
        return torch.exp(log_w)

    def envelope(
        self,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-node factor in [0, 1] that switches the network off at short
        contacts, ``(N,)`` in ``edge_vec``'s dtype.

        ``w_i`` is the product of the step ``s`` over the edges into node
        ``i``: exactly 1 when every neighbour is beyond its window, exactly 0
        when one is at or below the window's lower edge. An edge counts when
        it is unmasked and both ends are real atoms. The graph holds every
        edge into an owned node, so an owned node's product is complete; a
        ghost's is not, and is not used.
        """
        if not self.training and self._use_warp(edge_vec):
            return _EnvelopeFunction.apply(edge_vec, self, atype, edge_index, edge_mask)
        src, dst = edge_index[0].long(), edge_index[1].long()
        real = (atype >= 0) & (atype < self.numbers.numel())
        valid = edge_mask.bool() & real[src] & real[dst]
        return self._envelope_torch(atype, src, dst, edge_vec, valid)

    def _envelope_warp(
        self,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass over the edges: the factor per node, (N,), and per edge
        ``d log s / d edge_vec``, (E, 3), both in ``edge_vec``'s dtype. The
        factor's gradient along an edge into node ``i`` is ``w_i`` times that
        edge's entry."""
        from deepmd_lr.nvalchemi.nlist import _torch_stream_scope

        n_node, n_edge = atype.shape[0], edge_vec.shape[0]
        device = edge_vec.device
        log_sum = torch.zeros(n_node, dtype=torch.float64, device=device)
        closed = torch.zeros(n_node, dtype=torch.int32, device=device)
        dlog = torch.zeros_like(edge_vec)
        if n_edge:
            is32 = edge_vec.dtype == torch.float32
            vec_t = wp.vec3f if is32 else wp.vec3d
            # Bound to names, so no contiguous copy is freed before the launch.
            index = edge_index.to(torch.int64).contiguous()
            mask = edge_mask.to(torch.bool).contiguous()
            vec = edge_vec.detach().contiguous()
            types = atype.to(torch.int64).contiguous()

            def arr(t, dtype):
                return wp.from_torch(t, dtype=dtype, requires_grad=False)

            with _torch_stream_scope(device):
                wp.launch(
                    _envelope_f32 if is32 else _envelope_f64,
                    dim=n_edge,
                    inputs=[
                        arr(index, wp.int64),
                        arr(mask, wp.bool),
                        arr(vec, vec_t),
                        arr(types, wp.int64),
                        arr(self.r_lo, wp.float64),
                        arr(self.r_hi, wp.float64),
                        wp.float64(self.max_outer**2),
                        arr(log_sum, wp.float64),
                        arr(closed, wp.int32),
                        arr(dlog, vec_t),
                    ],
                    device=str(device),
                )
        factor = torch.where(closed > 0, torch.zeros_like(log_sum), torch.exp(log_sum))
        return factor.to(edge_vec.dtype), dlog

    # -------------------------------------------------------- graph output

    def graph_output(
        self,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        n_local: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        destination_order: torch.Tensor | None = None,
        destination_row_ptr: torch.Tensor | None = None,
        source_order: torch.Tensor | None = None,
        source_row_ptr: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """The term's share of ``forward_common_lower_graph``'s result: the
        keys ``fit_output_to_model_output_graph`` writes for ``energy``, to be
        added to the learned model's."""
        n_cap = atype.shape[0]
        nf = n_node.shape[0]
        owned = (
            node_ownership_mask(n_node, n_local, n_cap).to(torch.bool)
            if n_local is not None
            else None
        )
        energy, g_e = self.edge_terms(atype, edge_index, edge_vec, edge_mask, owned)
        frame_id = frame_id_from_n_node(n_node, n_total=n_cap)
        # Ghost destinations were excluded edge by edge, so the per-node
        # energy is already the owned one.
        energy_redu = segment_sum(energy, frame_id, nf)
        if (
            cuda_infer_level() >= 1
            and fused_scatter_available()
            and destination_order is not None
            and destination_row_ptr is not None
            and source_order is not None
            and source_row_ptr is not None
        ):
            force, atom_virial, virial, _ = fused_edge_force_virial(
                g_e,
                edge_vec,
                edge_index,
                edge_mask,
                destination_order,
                destination_row_ptr,
                source_order,
                source_row_ptr,
                n_node,
                edge_vec.new_empty(0),
                n_cap,
                do_atomic_virial,
            )
        else:
            force, atom_virial, virial = edge_force_virial(
                g_e, edge_vec, edge_index, edge_mask, n_node, node_capacity=n_cap
            )
        out = {
            "energy": energy.unsqueeze(-1),
            "energy_redu": energy_redu.unsqueeze(-1),
            "energy_derv_r": force.reshape(n_cap, 1, 3),
            "energy_derv_c_redu": virial.reshape(nf, 1, 9),
        }
        if do_atomic_virial:
            out["energy_derv_c"] = atom_virial.reshape(n_cap, 1, 9)
        return out


class _EnvelopeFunction(torch.autograd.Function):
    """The envelope from the Warp kernel, with its gradient.

    The kernel leaves ``d log s / d edge_vec`` per edge, so the backward is
    one gather and one product, ``(grad_w w)[dst] dlog``: zero wherever the
    factor is flat, the closed contacts included. First derivatives only --
    a second backward is refused rather than returned wrong; training, which
    needs one, takes the torch arithmetic.
    """

    @staticmethod
    def forward(ctx, edge_vec, zbl, atype, edge_index, edge_mask):
        factor, dlog = zbl._envelope_warp(atype, edge_index, edge_vec, edge_mask)
        ctx.dlog = dlog
        ctx.dst = edge_index[1].long()
        ctx.save_for_backward(factor)
        return factor

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_factor):
        (factor,) = ctx.saved_tensors
        weight = (grad_factor * factor).index_select(0, ctx.dst)
        return weight.unsqueeze(-1) * ctx.dlog, None, None, None, None


# ----------------------------------------------------------------------------
# The Warp kernels: one pass over the edges
# ----------------------------------------------------------------------------

if wp is not None:

    @wp.func
    def _zbl_pair(
        r: wp.float64, zi: wp.float64, zj: wp.float64, ri: wp.float64, ro: wp.float64
    ) -> wp.vec2d:
        """Half the switched pair energy and its distance derivative."""
        a = wp.float64(0.46850) / (
            wp.pow(zi, wp.float64(0.23)) + wp.pow(zj, wp.float64(0.23))
        )
        x = r / a
        t1 = wp.float64(0.18175) * wp.exp(wp.float64(-3.19980) * x)
        t2 = wp.float64(0.50986) * wp.exp(wp.float64(-0.94229) * x)
        t3 = wp.float64(0.28022) * wp.exp(wp.float64(-0.40290) * x)
        t4 = wp.float64(0.02817) * wp.exp(wp.float64(-0.20162) * x)
        phi = t1 + t2 + t3 + t4
        dphi = -(
            wp.float64(3.19980) * t1
            + wp.float64(0.94229) * t2
            + wp.float64(0.40290) * t3
            + wp.float64(0.20162) * t4
        )
        coulomb = wp.float64(14.3996454784255) * zi * zj / r
        pair = coulomb * phi
        dpair = coulomb * (dphi / a - phi / r)
        t = wp.clamp((r - ri) / (ro - ri), wp.float64(0.0), wp.float64(1.0))
        t2_ = t * t
        t3_ = t2_ * t
        switch = (
            wp.float64(1.0)
            - wp.float64(10.0) * t3_
            + wp.float64(15.0) * t3_ * t
            - wp.float64(6.0) * t3_ * t2_
        )
        dswitch = (
            wp.float64(-30.0) * t2_
            + wp.float64(60.0) * t3_
            - wp.float64(30.0) * t3_ * t
        ) / (ro - ri)
        return wp.vec2d(
            wp.float64(0.5) * pair * switch,
            wp.float64(0.5) * (dpair * switch + pair * dswitch),
        )

    @wp.func
    def _edge_valid(
        e: int,
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        atype: wp.array(dtype=wp.int64),
        n_type: int,
    ) -> bool:
        """A real edge between two real atoms (a padded node has type -1)."""
        if not edge_mask[e]:
            return False
        i = atype[edge_index[0, e]]
        j = atype[edge_index[1, e]]
        if i < wp.int64(0) or i >= wp.int64(n_type):
            return False
        if j < wp.int64(0) or j >= wp.int64(n_type):
            return False
        return True

    @wp.kernel(enable_backward=False)
    def _edge_terms_f32(
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        vec: wp.array(dtype=wp.vec3f),
        atype: wp.array(dtype=wp.int64),
        owned: wp.array(dtype=wp.bool),
        numbers: wp.array(dtype=wp.float64),
        r_lo: wp.array2d(dtype=wp.float64),
        r_hi: wp.array2d(dtype=wp.float64),
        r_max2: wp.float64,
        energy: wp.array(dtype=wp.float64),
        g: wp.array(dtype=wp.vec3f),
    ):
        e = wp.tid()
        v = wp.vec3d(
            wp.float64(vec[e][0]), wp.float64(vec[e][1]), wp.float64(vec[e][2])
        )
        # The distance first: almost every edge leaves here, having read
        # nothing but its own vector.
        r2 = wp.dot(v, v)
        if r2 >= r_max2:
            return
        if not _edge_valid(e, edge_index, edge_mask, atype, numbers.shape[0]):
            return
        if not owned[edge_index[1, e]]:
            return
        i = atype[edge_index[0, e]]
        j = atype[edge_index[1, e]]
        ro = r_hi[i, j]
        r = wp.sqrt(r2)
        if r >= ro:
            return
        out = _zbl_pair(r, numbers[i], numbers[j], r_lo[i, j], ro)
        wp.atomic_add(energy, edge_index[1, e], out[0])
        s = out[1] / r
        g[e] = wp.vec3f(
            wp.float32(s * v[0]), wp.float32(s * v[1]), wp.float32(s * v[2])
        )

    @wp.kernel(enable_backward=False)
    def _edge_terms_f64(
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        vec: wp.array(dtype=wp.vec3d),
        atype: wp.array(dtype=wp.int64),
        owned: wp.array(dtype=wp.bool),
        numbers: wp.array(dtype=wp.float64),
        r_lo: wp.array2d(dtype=wp.float64),
        r_hi: wp.array2d(dtype=wp.float64),
        r_max2: wp.float64,
        energy: wp.array(dtype=wp.float64),
        g: wp.array(dtype=wp.vec3d),
    ):
        e = wp.tid()
        v = vec[e]
        r2 = wp.dot(v, v)
        if r2 >= r_max2:
            return
        if not _edge_valid(e, edge_index, edge_mask, atype, numbers.shape[0]):
            return
        if not owned[edge_index[1, e]]:
            return
        i = atype[edge_index[0, e]]
        j = atype[edge_index[1, e]]
        ro = r_hi[i, j]
        r = wp.sqrt(r2)
        if r >= ro:
            return
        out = _zbl_pair(r, numbers[i], numbers[j], r_lo[i, j], ro)
        wp.atomic_add(energy, edge_index[1, e], out[0])
        g[e] = (out[1] / r) * v

    @wp.func
    def _log_step(r: wp.float64, ri: wp.float64, ro: wp.float64) -> wp.vec2d:
        """Inside the window, ``log s`` and ``d log s / dr``; the first is
        positive (never a logarithm of a step) at or below the lower edge."""
        t = (r - ri) / (ro - ri)
        if t <= wp.float64(0.0):
            return wp.vec2d(wp.float64(1.0), wp.float64(0.0))
        s = (
            t
            * t
            * t
            * (t * (wp.float64(6.0) * t - wp.float64(15.0)) + wp.float64(10.0))
        )
        if s <= wp.float64(0.0):
            return wp.vec2d(wp.float64(1.0), wp.float64(0.0))
        one = wp.float64(1.0) - t
        ds = wp.float64(30.0) * t * t * one * one / (ro - ri)
        return wp.vec2d(wp.log(s), ds / s)

    @wp.kernel(enable_backward=False)
    def _envelope_f32(
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        vec: wp.array(dtype=wp.vec3f),
        atype: wp.array(dtype=wp.int64),
        r_lo: wp.array2d(dtype=wp.float64),
        r_hi: wp.array2d(dtype=wp.float64),
        r_max2: wp.float64,
        log_sum: wp.array(dtype=wp.float64),
        closed: wp.array(dtype=wp.int32),
        dlog: wp.array(dtype=wp.vec3f),
    ):
        e = wp.tid()
        v = wp.vec3d(
            wp.float64(vec[e][0]), wp.float64(vec[e][1]), wp.float64(vec[e][2])
        )
        r2 = wp.dot(v, v)
        if r2 >= r_max2:
            return
        if not _edge_valid(e, edge_index, edge_mask, atype, r_lo.shape[0]):
            return
        i = atype[edge_index[0, e]]
        j = atype[edge_index[1, e]]
        ro = r_hi[i, j]
        r = wp.sqrt(r2)
        if r >= ro:
            return
        out = _log_step(r, r_lo[i, j], ro)
        if out[0] > wp.float64(0.0):
            wp.atomic_add(closed, edge_index[1, e], 1)
            return
        wp.atomic_add(log_sum, edge_index[1, e], out[0])
        c = out[1] / r
        dlog[e] = wp.vec3f(
            wp.float32(c * v[0]), wp.float32(c * v[1]), wp.float32(c * v[2])
        )

    @wp.kernel(enable_backward=False)
    def _envelope_f64(
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        vec: wp.array(dtype=wp.vec3d),
        atype: wp.array(dtype=wp.int64),
        r_lo: wp.array2d(dtype=wp.float64),
        r_hi: wp.array2d(dtype=wp.float64),
        r_max2: wp.float64,
        log_sum: wp.array(dtype=wp.float64),
        closed: wp.array(dtype=wp.int32),
        dlog: wp.array(dtype=wp.vec3d),
    ):
        e = wp.tid()
        v = vec[e]
        r2 = wp.dot(v, v)
        if r2 >= r_max2:
            return
        if not _edge_valid(e, edge_index, edge_mask, atype, r_lo.shape[0]):
            return
        i = atype[edge_index[0, e]]
        j = atype[edge_index[1, e]]
        ro = r_hi[i, j]
        r = wp.sqrt(r2)
        if r >= ro:
            return
        out = _log_step(r, r_lo[i, j], ro)
        if out[0] > wp.float64(0.0):
            wp.atomic_add(closed, edge_index[1, e], 1)
            return
        wp.atomic_add(log_sum, edge_index[1, e], out[0])
        dlog[e] = (out[1] / r) * v
