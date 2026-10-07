# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fixed, smoothly truncated ZBL repulsion on the DPA4C neighbour graph.

This is an additive prior, not DeePMD's ``use_srtab`` interpolation or DPA4's
Zone/descriptor-clamping construction. Training fits the residual energy and
forces with this term present; adding it to existing weights changes the model.

Two ways to place the switch:

``cutoff: "fixed"``
    One pair of radii for every element pair, ``r_inner``/``r_outer``: DPA4's
    bridging convention (defaults 0.5/0.8 A). Mind that DPA4 pairs these
    radii with a descriptor clamp that blinds the network below ``r_outer``;
    here the network stays active, so radii this short leave the term inert
    at ordinary bond lengths.
``cutoff: "covalent"``
    Per pair, ``r_outer = covalent_scale * (R_i + R_j)`` with ASE's covalent
    radii, and ``r_inner = inner_ratio * r_outer``: MACE's convention, which
    lets the wall reach each bond and leaves the network to absorb the rest
    during training. MACE additionally softens the screening length; the
    screening here is the standard ZBL one.

The term has no parameters and sees the edge vectors through no network, so
its forces and virial are assembled analytically from the per-edge derivative
-- no second autograd pass -- through the same edge-to-node reduction DeePMD
uses for the learned energy (its fused CUDA kernel when ``DP_CUDA_INFER`` asks
for it). On a CUDA device one Warp kernel produces the per-edge energy and
derivative; elsewhere the same arithmetic runs as torch ops.
"""

import contextlib
import dataclasses
import math
from numbers import Real
from typing import Any

import numpy as np
import torch
from ase.data import atomic_numbers, covalent_radii
from deepmd.dpmodel.model.edge_transform_output import node_ownership_mask
from deepmd.dpmodel.utils.neighbor_graph.derivatives import edge_force_virial
from deepmd.dpmodel.utils.neighbor_graph.graph import frame_id_from_n_node
from deepmd.dpmodel.utils.neighbor_graph.segment import segment_sum
from deepmd.pt_expt.model.edge_transform_output import (
    cuda_infer_level,
    fused_edge_force_virial,
    fused_scatter_available,
)

try:
    import warp as wp
except ImportError:  # pragma: no cover - the MD extra is optional
    wp = None

#: Coulomb constant, eV A.
_KE = 14.3996454784255
#: ZBL universal screening: ``a = 0.46850 / (Z_i^0.23 + Z_j^0.23)`` A.
_A_PREFACTOR = 0.46850
_A_EXPONENT = 0.23
_PHI_A = (0.18175, 0.50986, 0.28022, 0.02817)
_PHI_B = (3.19980, 0.94229, 0.40290, 0.20162)

_CUTOFFS = ("fixed", "covalent")
_KEYS = {
    "cutoff",
    "r_inner",
    "r_outer",
    "covalent_scale",
    "inner_ratio",
    "clamp",
    "clamp_width",
}
#: ``clamp`` absent means off: checkpoints written before it existed are
#: reproduced as trained. The schema default for a new input is on.
_DEFAULTS = {
    "cutoff": "fixed",
    "r_inner": 0.5,
    "r_outer": 0.8,
    "covalent_scale": 1.0,
    "inner_ratio": 0.5,
    "clamp": False,
    "clamp_width": None,
}


def _septic_remap_coefficients() -> tuple[float, float, float, float]:
    """``q(t) = a4 t^4 + a5 t^5 + a6 t^6 + a7 t^7`` on [0, 1], flat to third
    order at 0 (``q = q' = q'' = q''' = 0``) and joining the identity to third
    order at 1 (``q = 1, q' = 1, q'' = q''' = 0``): the clamp's transition
    from a frozen length to the true one, C^3 at both ends like DPA4's
    ``InnerClamp``."""
    rows = [
        [1, 1, 1, 1],
        [4, 5, 6, 7],
        [12, 20, 30, 42],
        [24, 60, 120, 210],
    ]
    a = np.linalg.solve(
        np.array(rows, dtype=np.float64), np.array([1.0, 1.0, 0.0, 0.0])
    )
    return tuple(float(x) for x in a)


_REMAP = _septic_remap_coefficients()


def _tracing(t: torch.Tensor) -> bool:
    """Whether ``t`` is being traced (export, compiled training) rather than
    computed: a Warp launch is a side effect no tracer records, so the trace
    must take the torch arithmetic instead."""
    if torch.compiler.is_compiling() or torch.fx._symbolic_trace.is_fx_tracing():
        return True
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return isinstance(t, (FakeTensor, FunctionalTensor)) or get_proxy_mode() is not None


def _check_distance(key: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"zbl.{key} must be a finite real number")
    return float(value)


class ZBL(torch.nn.Module):
    """Analytical screened nuclear repulsion, with no trainable parameters."""

    def __init__(self, type_map: list[str], settings: dict[str, Any], device=None):
        super().__init__()
        if not isinstance(settings, dict):
            raise ValueError("zbl must be a configuration dictionary")
        settings = {k: v for k, v in settings.items() if not k.startswith("_")}
        if set(settings) - _KEYS:
            raise ValueError(f"zbl accepts only {sorted(_KEYS)}")
        settings = {**_DEFAULTS, **settings}
        if settings["cutoff"] not in _CUTOFFS:
            raise ValueError(f"zbl.cutoff must be one of {_CUTOFFS}")
        self.cutoff = settings["cutoff"]
        if not isinstance(settings["clamp"], bool):
            raise ValueError("zbl.clamp must be true or false")
        #: Show the descriptor nothing below the switch: a pair closer than
        #: r_outer looks like one at r_outer, and the true length returns over
        #: ``clamp_width`` beyond it (:meth:`clamping`).
        self.clamp = settings["clamp"]
        width = settings["clamp_width"]
        if width is not None and (
            isinstance(width, bool) or not isinstance(width, Real) or not width > 0
        ):
            raise ValueError("zbl.clamp_width must be a positive distance (Angstrom)")
        self._clamp_width_setting = None if width is None else float(width)
        self.r_inner = _check_distance("r_inner", settings["r_inner"])
        self.r_outer = _check_distance("r_outer", settings["r_outer"])
        self.covalent_scale = _check_distance(
            "covalent_scale", settings["covalent_scale"]
        )
        self.inner_ratio = _check_distance("inner_ratio", settings["inner_ratio"])
        if self.cutoff == "fixed" and not 0 < self.r_inner < self.r_outer:
            raise ValueError("zbl requires 0 < r_inner < r_outer")
        if self.cutoff == "covalent" and not (
            self.covalent_scale > 0 and 0 < self.inner_ratio < 1
        ):
            raise ValueError("zbl requires covalent_scale > 0 and 0 < inner_ratio < 1")
        # A type that is not an element -- ASE's dummy "X" heads this
        # repository's periodic-table type maps -- has no nucleus: Z = 0, and
        # every pair it is in contributes nothing.
        numbers = [atomic_numbers.get(s, 0) for s in type_map]
        if not any(z > 0 for z in numbers):
            raise ValueError("zbl requires at least one element symbol in type_map")
        # Derived entirely from the serialized type_map; not optimizer state.
        self.register_buffer(
            "numbers",
            torch.tensor(numbers, dtype=torch.float64, device=device),
            persistent=False,
        )
        #: Per-type half of the pair's outer radius: ``r_outer_ij`` is the sum
        #: over the two types. The fixed cutoff splits its one radius evenly.
        if self.cutoff == "covalent":
            radii = [self.covalent_scale * float(covalent_radii[z]) for z in numbers]
        else:
            radii = [0.5 * self.r_outer] * len(numbers)
        self.register_buffer(
            "half_outer",
            torch.tensor(radii, dtype=torch.float64, device=device),
            persistent=False,
        )
        #: ``r_inner_ij / r_outer_ij``, the same for every pair.
        self.inner_fraction = (
            self.inner_ratio
            if self.cutoff == "covalent"
            else self.r_inner / self.r_outer
        )
        #: The transition back to the true length, in Angstrom beyond r_outer.
        #: ``None`` means each pair's own switch width, r_outer - r_inner.
        if self._clamp_width_setting is not None:
            self.clamp_width = self._clamp_width_setting
        elif self.cutoff == "fixed":
            self.clamp_width = self.r_outer - self.r_inner
        else:
            self.clamp_width = None

    @property
    def clamp_reach(self) -> float:
        """How far out the clamp still alters a pair: the largest r_trans."""
        if self.clamp_width is None:
            return self.max_r_outer * (2.0 - self.inner_fraction)
        return self.max_r_outer + self.clamp_width

    @property
    def max_r_outer(self) -> float:
        """The largest outer radius any pair of real elements can have."""
        real = self.half_outer[self.numbers > 0]
        return 2.0 * float(real.max())

    def serialize(self) -> dict[str, Any]:
        if self.cutoff == "covalent":
            data = {
                "cutoff": "covalent",
                "covalent_scale": self.covalent_scale,
                "inner_ratio": self.inner_ratio,
            }
        else:
            data = {"r_inner": self.r_inner, "r_outer": self.r_outer}
        data["clamp"] = self.clamp
        if self._clamp_width_setting is not None:
            data["clamp_width"] = self._clamp_width_setting
        return data

    # ------------------------------------------------------------- clamp

    @staticmethod
    def clamp_distance(
        r: torch.Tensor, r_outer: torch.Tensor, r_trans: torch.Tensor
    ) -> torch.Tensor:
        """The length the descriptor is shown for a pair at ``r``: ``r_outer``
        from ``r_outer`` in, ``r`` itself from ``r_trans`` out, a C^3 septic
        remap between. Monotone, and never below ``r_outer``: the network is
        shown nothing of the region ZBL governs, which is also the region the
        data never sampled -- a frozen length *inside* the switch (DPA4's
        choice) would still show it distances it has never seen, and this
        descriptor extrapolates there by thousands of eV."""
        t = ((r - r_outer) / (r_trans - r_outer)).clamp(0.0, 1.0)
        a4, a5, a6, a7 = _REMAP
        q = t.pow(4) * (a4 + t * (a5 + t * (a6 + t * a7)))
        return torch.where(r >= r_trans, r, r_outer + (r_trans - r_outer) * q)

    def clamp_edges(
        self, edge_vec: torch.Tensor, atype: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Edge vectors rescaled radially to their clamped lengths, as a
        differentiable function of ``edge_vec``: whatever is differentiated
        through the descriptor picks the clamp's Jacobian up on the way."""
        ti = atype.clamp(0, self.numbers.numel() - 1).long()
        half = self.half_outer[ti]
        ro = (half[edge_index[0]] + half[edge_index[1]]).to(edge_vec.dtype)
        width = (
            ro * (1.0 - self.inner_fraction)
            if self.clamp_width is None
            else self.clamp_width
        )
        rt = ro + width
        r = edge_vec.norm(dim=1)
        # A padded edge is a zero vector; its scale is irrelevant but must be
        # finite.
        safe = torch.where(r > 0, r, torch.ones_like(r))
        scale = torch.where(
            r >= rt, torch.ones_like(r), self.clamp_distance(safe, ro, rt) / safe
        )
        return edge_vec * scale.unsqueeze(-1)

    @contextlib.contextmanager
    def clamping(self, descriptor):
        """Make ``descriptor.call_graph`` see clamped edge vectors for the block.

        Wrapping the descriptor rather than the model's lower keeps DeePMD's
        own ``edge_vec`` leaf in charge: the clamp sits between the leaf and
        the descriptor as ordinary torch ops, so forces, the charge response
        and the virial all differentiate through it with no further code. A
        capture opened outside this block (``_capturing_backbone``) records
        the raw vectors, which is what its consumers assume.
        """
        if not self.clamp:
            yield
            return
        unhooked = descriptor.call_graph

        def clamped(graph, atype_flat, *args, **kwargs):
            vec = self.clamp_edges(graph.edge_vec, atype_flat, graph.edge_index)
            graph = dataclasses.replace(graph, edge_vec=vec)
            return unhooked(graph, atype_flat, *args, **kwargs)

        had_own = "call_graph" in descriptor.__dict__
        descriptor.call_graph = clamped
        try:
            yield
        finally:
            if had_own:
                descriptor.call_graph = unhooked
            else:
                del descriptor.call_graph

    # ------------------------------------------------------------ per edge

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
        ti = atype.clamp(0, self.numbers.numel() - 1).long()
        z, half = self.numbers[ti], self.half_outer[ti]
        zi, zj = z[src], z[dst]
        ro = half[src] + half[dst]
        ri = self.inner_fraction * ro
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
        t = ((r - ri) / (ro - ri)).clamp(0.0, 1.0)
        switch = 1 - 10 * t.pow(3) + 15 * t.pow(4) - 6 * t.pow(5)
        dswitch = (-30 * t.pow(2) + 60 * t.pow(3) - 30 * t.pow(4)) / (ro - ri)
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

        def arr(t, dtype):
            return wp.from_torch(t.contiguous(), dtype=dtype, requires_grad=False)

        with _torch_stream_scope(edge_vec.device):
            wp.launch(
                kernel,
                dim=n_edge,
                inputs=[
                    arr(edge_index.to(torch.int64), wp.int64),
                    arr(edge_mask.to(torch.bool), wp.bool),
                    arr(edge_vec, vec_t),
                    arr(atype.to(torch.int64), wp.int64),
                    arr(owned, wp.bool),
                    arr(self.numbers, wp.float64),
                    arr(self.half_outer, wp.float64),
                    wp.float64(self.inner_fraction),
                    arr(energy, wp.float64),
                    arr(g, vec_t),
                ],
                device=str(edge_vec.device),
            )
        return energy, g

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
        if wp is not None and edge_vec.is_cuda and not _tracing(edge_vec):
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

    # -------------------------------------------------------- graph output

    def graph_output(
        self,
        model,
        atype,
        n_node,
        n_local,
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """The term's share of ``forward_common_lower_graph``'s result: the
        keys :func:`fit_output_to_model_output_graph` writes for ``energy``,
        to be added to the learned model's."""
        del model
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
        do_atomic_virial = bool(kwargs.get("do_atomic_virial", False))
        source_order = kwargs.get("source_order")
        source_row_ptr = kwargs.get("source_row_ptr")
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


# ----------------------------------------------------------------------------
# The Warp kernel: one pass over the edges
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
        phi = wp.float64(0.0)
        dphi = wp.float64(0.0)
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
        owned: wp.array(dtype=wp.bool),
        n_type: int,
    ) -> bool:
        if not edge_mask[e]:
            return False
        i = atype[edge_index[0, e]]
        j = atype[edge_index[1, e]]
        if i < wp.int64(0) or i >= wp.int64(n_type):
            return False
        if j < wp.int64(0) or j >= wp.int64(n_type):
            return False
        return owned[edge_index[1, e]]

    @wp.kernel(enable_backward=False)
    def _edge_terms_f32(
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        vec: wp.array(dtype=wp.vec3f),
        atype: wp.array(dtype=wp.int64),
        owned: wp.array(dtype=wp.bool),
        numbers: wp.array(dtype=wp.float64),
        half_outer: wp.array(dtype=wp.float64),
        inner_fraction: wp.float64,
        energy: wp.array(dtype=wp.float64),
        g: wp.array(dtype=wp.vec3f),
    ):
        e = wp.tid()
        if not _edge_valid(e, edge_index, edge_mask, atype, owned, numbers.shape[0]):
            return
        i = atype[edge_index[0, e]]
        j = atype[edge_index[1, e]]
        v = wp.vec3d(
            wp.float64(vec[e][0]), wp.float64(vec[e][1]), wp.float64(vec[e][2])
        )
        r = wp.length(v)
        ro = half_outer[i] + half_outer[j]
        if r >= ro:
            return
        out = _zbl_pair(r, numbers[i], numbers[j], inner_fraction * ro, ro)
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
        half_outer: wp.array(dtype=wp.float64),
        inner_fraction: wp.float64,
        energy: wp.array(dtype=wp.float64),
        g: wp.array(dtype=wp.vec3d),
    ):
        e = wp.tid()
        if not _edge_valid(e, edge_index, edge_mask, atype, owned, numbers.shape[0]):
            return
        i = atype[edge_index[0, e]]
        j = atype[edge_index[1, e]]
        v = vec[e]
        r = wp.length(v)
        ro = half_outer[i] + half_outer[j]
        if r >= ro:
            return
        out = _zbl_pair(r, numbers[i], numbers[j], inner_fraction * ro, ro)
        wp.atomic_add(energy, edge_index[1, e], out[0])
        g[e] = (out[1] / r) * v
