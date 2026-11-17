# SPDX-License-Identifier: MIT
"""Short contacts on the DPA4C neighbour graph: ZBL takes over from the network.

A network fitted to condensed-phase data has seen no pair closer than its
shortest bond, and below that it extrapolates freely. On a water model an O-H
pair pushed from 0.70 to 0.60 A drops the fitted energy by 230 eV, all of it
on the two atoms in contact, and a little further in their predicted charges
reach tens of e. A few eV of pair repulsion added on top change nothing
there, so this module does two things across a window per pair of elements:

- **ZBL in**: the screened nuclear repulsion of Ziegler, Biersack and
  Littmark, ``V_ZBL(r) (1 - s)``;
- **the network out**: each atom's fitted energy and raw charge are
  multiplied by the envelope ``w_i = prod_j s(r_ij)``, the product over its
  neighbours (mace-lr's ``charge_net.envelope``, here on the energy too).

A window is ``[x_lo, x_hi]`` in units of the pair's covalent-radius sum,
``x = r / (R_i + R_j)`` with ASE's radii, and one C2 step
``s(x) = t^3 (10 - 15 t + 6 t^2)``, ``t = (x - x_lo) / (x_hi - x_lo)`` clamped
to [0, 1], is exactly 0 below it and exactly 1 above. Below ``x_lo`` a pair is
ZBL alone; with every neighbour beyond ``x_hi`` an atom's factor is exactly 1.

By default every pair of elements has the same window in these units,
:data:`DEFAULT_WINDOW`, so the covalent radii place it and nothing has to be
assigned: a bond sits near one radius sum and its window just below it.
``pairs`` gives single pairs a window of their own, because what the network
has seen differs from pair to pair: two oxygens of liquid water never come
closer than 1.8 radius sums and two hydrogens than 2.0, and the network falls
right below each of those, far above a window placed for a bond. A window's
``x_hi`` belongs at or below the pair's shortest contact in the data, so that
nothing fitted changes, and its ``x_lo`` above where the network starts to
fall.

ZBL has no parameters and sees the edge vectors through no network, so its
force and virial are assembled analytically from the per-edge derivative. The
envelope multiplies network outputs, so whatever is differentiated through
those outputs -- force, virial, the charge response, Born charges -- has to
pick its derivative up on the way.

There are two ways through, with the same numbers:

- **Inference on a CUDA device** (:meth:`ZBL.contacts`). One Warp kernel
  walks the edges and leaves the per-node ZBL energy and envelope and a short
  list of the edges in contact, with each one's derivatives. In a healthy
  simulation that list is empty, and nothing the size of the edge list is
  allocated or passed over again: forces and virials are summed from the
  list, and the envelope enters the network outputs as a plain factor whose
  derivative is added from the list wherever one is taken
  (:func:`contact_output`, :func:`add_envelope_gradient`).
- **Training, a CPU, a tracer** (:meth:`ZBL.graph_output`,
  :meth:`ZBL.envelope`). The same arithmetic as torch ops over every edge,
  the envelope a differentiable function of the edge vectors, to any order.
"""

import math
from numbers import Real
from typing import Any, NamedTuple

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

__all__ = [
    "DEFAULT_WINDOW",
    "ZBL",
    "ZBL_KEY",
    "Contacts",
    "add_envelope_gradient",
    "contact_output",
]

#: Key of the optional block in the model config.
ZBL_KEY = "zbl"

#: Coulomb constant, eV A.
_KE = 14.3996454784255
#: ZBL universal screening: ``a = 0.46850 / (Z_i^0.23 + Z_j^0.23)`` A.
_A_PREFACTOR = 0.46850
_A_EXPONENT = 0.23
_PHI_A = (0.18175, 0.50986, 0.28022, 0.02817)
_PHI_B = (3.19980, 0.94229, 0.40290, 0.20162)

#: The window of every pair of elements that ``pairs`` does not name, in
#: covalent-radius sums. Measured on a DPA4C water model, whose O-H bonds stop
#: at 0.90 and whose fitted energy falls below 0.70: a window that closes at
#: 0.5 (mace-lr's, for the charges of a MACE model) is too late and leaves a
#: hole of -75 eV, one that opens at 0.9 touches the data, and this one gives
#: a wall that rises all the way in.
DEFAULT_WINDOW = (0.6, 0.85)

_KEYS = ("window", "pairs")
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


def pair_key(a: str, b: str) -> str:
    """The one name of a pair of elements, whichever way round it was written."""
    return "-".join(sorted((a, b)))


def check_window(window: Any, name: str = "window") -> tuple[float, float]:
    """Validate one window ``[x_lo, x_hi]`` in covalent-radius sums: finite,
    ``0 <= x_lo < x_hi``."""
    try:
        x_lo, x_hi = window
    except (TypeError, ValueError):
        raise ValueError(
            f"zbl.{name} must be [x_lo, x_hi] in covalent-radius sums, got {window!r}"
        ) from None
    for x in (x_lo, x_hi):
        if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x):
            raise ValueError(f"zbl.{name} must hold two finite real numbers")
    if not 0.0 <= x_lo < x_hi:
        raise ValueError(
            f"zbl.{name} needs 0 <= x_lo < x_hi, got [{x_lo}, {x_hi}]: ZBL is "
            "full below x_lo and gone above x_hi"
        )
    return float(x_lo), float(x_hi)


def check_pairs(pairs: Any) -> dict[str, tuple[float, float]]:
    """Validate ``pairs = {"A-B": [x_lo, x_hi], ...}``: pairs of elements,
    each written once, each a window. Returns them under their
    :func:`pair_key`; an empty or missing map is no pair."""
    if pairs is None:
        return {}
    if not isinstance(pairs, dict):
        raise ValueError(
            "zbl.pairs maps pairs of elements to their own window, e.g. "
            f'{{"O-O": [1.5, 1.75]}}; got {pairs!r}'
        )
    checked: dict[str, tuple[float, float]] = {}
    for name, window in pairs.items():
        if isinstance(name, str) and name.startswith("_"):
            continue  # DeePMD-kit's `_comment` annotations, as in any block
        symbols = name.split("-") if isinstance(name, str) else []
        if len(symbols) != 2 or not all(atomic_numbers.get(s, 0) > 0 for s in symbols):
            raise ValueError(f"zbl.pairs: {name!r} is not a pair of elements")
        key = pair_key(*symbols)
        if key in checked:
            raise ValueError(f"zbl.pairs names the pair {key} twice")
        checked[key] = check_window(window, f"pairs[{name!r}]")
    return dict(sorted(checked.items()))


def normalized_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """A ``zbl`` block as the model reads it: the default window written
    out, each pair under one name, numbers as floats. Two blocks that build
    the same term compare equal."""
    settings = {k: v for k, v in settings.items() if not k.startswith("_")}
    window = settings.get("window", DEFAULT_WINDOW)
    return {
        "window": None if window is None else list(check_window(window)),
        "pairs": {k: list(v) for k, v in check_pairs(settings.get("pairs")).items()},
    }


class Contacts(NamedTuple):
    """What one pass over an inference graph found (:meth:`ZBL.contacts`).

    Per node, the ZBL energy and the envelope; per edge in contact (closer
    than the upper edge of its pair's window), its ends, its vector and its
    two derivatives. The list has a fixed capacity and holds zeros beyond the
    contacts found, so every sum over it is a sum over the contacts. Float64
    throughout.
    """

    #: ``(N,)`` ZBL energy of every node: half of each pair it is the
    #: destination of, owned nodes only. NaN if the list overflowed.
    energy: torch.Tensor
    #: ``(N,)`` envelope ``w``, a plain value: its derivative is in ``dlog``.
    envelope: torch.Tensor
    #: ``(K,)`` the contact's edge, and its two ends.
    edge: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    #: ``(K, 3)`` the edge vector, ``r_src - r_dst``.
    vec: torch.Tensor
    #: ``(K, 3)`` ``d E_zbl / d edge_vec`` of that edge.
    g: torch.Tensor
    #: ``(K, 3)`` ``d log s / d edge_vec``: the envelope of the edge's
    #: destination changes by ``w_dst`` times this.
    dlog: torch.Tensor


def add_envelope_gradient(
    gradient: torch.Tensor, contacts: Contacts, coefficient: torch.Tensor
) -> torch.Tensor:
    """Add the envelope's share to a per-edge gradient, in place.

    ``gradient`` is ``d F / d edge_vec`` of some ``F`` taken with the envelope
    held fixed, ``(E, 3)``; ``coefficient`` is ``d F / d w`` per node. The
    envelope of a node depends on the edges into it only, so its share is
    ``coefficient_dst w_dst dlog`` on the edges in contact and nothing
    anywhere else.
    """
    weight = (coefficient.reshape(-1).to(torch.float64) * contacts.envelope)[
        contacts.dst
    ]
    return gradient.index_add_(
        0, contacts.edge, (weight[:, None] * contacts.dlog).to(gradient.dtype)
    )


def contact_output(
    contacts: Contacts,
    fitted: torch.Tensor,
    n_node: torch.Tensor,
    n_local: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """The short-contact share of ``forward_common_lower_graph``'s result,
    from the list of contacts: the keys ``fit_output_to_model_output_graph``
    writes for ``energy``, to be added to the learned model's.

    That is the ZBL energy with its force and virial, and the force and
    virial the envelope adds to the fitted energy: the learned model
    multiplied ``fitted``, its per-node output ``(N, 1)``, by the envelope as
    a plain factor, so its own backward gave ``w d fitted``, and ``fitted d
    w`` is added here. Sums follow DeePMD-kit's ``edge_force_virial``: the
    gradient to the destination, minus it to the source, and ``-g (x) r``
    to the frame's virial.
    """
    n_cap = contacts.energy.shape[0]
    nf = n_node.shape[0]
    coefficient = fitted.detach().reshape(-1).to(torch.float64)
    # A ghost's energy is another rank's: it is not in the sum the learned
    # model differentiated either. One frame, the case of every simulation,
    # is spelled out: this runs every step, with nothing to sum most of them.
    if n_local is not None and n_local is not n_node:
        if nf == 1:
            owned = torch.arange(n_cap, device=n_local.device) < n_local
        else:
            owned = node_ownership_mask(n_node, n_local, n_cap)
        coefficient = coefficient * owned
    weight = (coefficient * contacts.envelope)[contacts.dst]
    g = contacts.g + weight[:, None] * contacts.dlog
    force = torch.zeros((n_cap, 3), dtype=torch.float64, device=g.device)
    force.index_add_(0, contacts.dst, g).index_add_(0, contacts.src, -g)
    outer = -(g[:, :, None] * contacts.vec[:, None, :])
    if nf == 1:
        virial = outer.sum(dim=0, keepdim=True)
        energy_redu = contacts.energy.sum().reshape(1)
    else:
        frame_id = frame_id_from_n_node(n_node, n_total=n_cap)
        virial = torch.zeros((nf, 3, 3), dtype=torch.float64, device=g.device)
        virial.index_add_(0, frame_id[contacts.dst], outer)
        energy_redu = segment_sum(contacts.energy, frame_id, nf)
    return {
        "energy": contacts.energy.unsqueeze(-1),
        "energy_redu": energy_redu.unsqueeze(-1),
        "energy_derv_r": force.reshape(n_cap, 1, 3),
        "energy_derv_c_redu": virial.reshape(nf, 1, 9),
    }


class ZBL(torch.nn.Module):
    """Screened nuclear repulsion inside a short-contact window, and the
    envelope that switches the network off there; no parameters.

    ``settings`` is the ``zbl`` block of the model config, and may be empty:

    ``window``
        ``[x_lo, x_hi]`` in units of a pair's covalent-radius sum, for every
        pair of elements; :data:`DEFAULT_WINDOW` when left out. ``None``
        gives no pair a window but those in ``pairs``.
    ``pairs``
        ``{"A-B": [x_lo, x_hi], ...}``, a window of their own for single
        pairs, in the same units.
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
                    "switch is now a window in covalent-radius sums"
                    if retired
                    else ""
                )
            )
        window = settings.get("window", DEFAULT_WINDOW)
        #: The window of every pair ``pairs`` does not name, in covalent-radius
        #: sums, or None: then only those pairs have one.
        self.window = None if window is None else check_window(window)
        #: ``{"A-B": (x_lo, x_hi)}``, the pairs with a window of their own.
        self.pairs = check_pairs(settings.get("pairs"))
        if self.window is None and not self.pairs:
            raise ValueError(
                "zbl has no window: `window` is null and `pairs` names no pair"
            )
        # A type that is not an element -- ASE's dummy "X" heads this
        # repository's periodic-table type maps -- has no nucleus: Z = 0, and
        # can be in no window.
        numbers = [atomic_numbers.get(s, 0) for s in type_map]
        elements = [i for i, z in enumerate(numbers) if z > 0]
        if not elements:
            raise ValueError("zbl requires at least one element symbol in type_map")
        radius = torch.tensor(
            [float(covalent_radii[z]) for z in numbers], dtype=torch.float64
        )
        total = radius[:, None] + radius[None, :]
        n_type = len(type_map)
        r_lo = torch.zeros((n_type, n_type), dtype=torch.float64)
        r_hi = torch.zeros((n_type, n_type), dtype=torch.float64)
        if self.window is not None:
            real = torch.zeros(n_type, dtype=torch.bool)
            real[elements] = True
            both = real[:, None] & real[None, :]
            r_lo[both] = self.window[0] * total[both]
            r_hi[both] = self.window[1] * total[both]
        for name, (x_lo, x_hi) in self.pairs.items():
            a, b = name.split("-")
            if a not in type_map or b not in type_map:
                raise ValueError(
                    f"zbl.pairs names the pair {name}, but the type_map has no "
                    f"{a if a not in type_map else b}"
                )
            i, j = type_map.index(a), type_map.index(b)
            r_lo[i, j] = r_lo[j, i] = x_lo * total[i, j]
            r_hi[i, j] = r_hi[j, i] = x_hi * total[i, j]
        #: The largest distance at which any pair of this type_map is still
        #: inside its window: what the descriptor's cutoff has to cover.
        self.max_outer = float(r_hi.max())
        # Derived entirely from the serialized type_map and windows: not
        # optimizer state, and no state_dict entry.
        self.register_buffer(
            "numbers",
            torch.tensor(numbers, dtype=torch.float64, device=device),
            persistent=False,
        )
        #: Per pair of types, the distances where the window starts and ends,
        #: in angstrom; both zero for a pair without one, so that no distance
        #: is ever inside it.
        self.register_buffer("r_lo", r_lo.to(device), persistent=False)
        self.register_buffer("r_hi", r_hi.to(device), persistent=False)

    def serialize(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "window": None if self.window is None else list(self.window)
        }
        if self.pairs:
            data["pairs"] = {name: list(w) for name, w in self.pairs.items()}
        return data

    def window_of(self, a: str, b: str, type_map: list[str]) -> tuple[float, float]:
        """The window of one pair of elements, in angstrom."""
        i, j = type_map.index(a), type_map.index(b)
        return float(self.r_lo[i, j]), float(self.r_hi[i, j])

    def reach(self, atype: torch.Tensor) -> torch.Tensor:
        """The largest distance at which a pair of the types in ``atype`` is
        still inside its window, squared: a ``(1,)`` tensor on the device,
        computed there. A type_map that lists the whole periodic table has
        windows out to 4 A between its largest atoms; a pass over the edges
        of a water box should not look that far."""
        n_type = self.numbers.numel()
        # `index_fill_` takes its value as a scalar; an indexed assignment
        # would upload it, which a step of MD must not do.
        present = torch.zeros(n_type, dtype=torch.bool, device=atype.device)
        present.index_fill_(0, atype.clamp(0, n_type - 1).long(), True)
        reach = (self.r_hi * (present[:, None] & present[None, :])).max()
        return (reach * reach).reshape(1)

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
        (E, 3) in ``edge_vec``'s dtype, over every edge; the Warp kernel
        below is the same formulas for the edges in contact."""
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
        src, dst = edge_index[0].long(), edge_index[1].long()
        real = (atype >= 0) & (atype < self.numbers.numel())
        valid = edge_mask.bool() & real[src] & real[dst]
        return self._envelope_torch(atype, src, dst, edge_vec, valid)

    def inside(
        self,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[float, dict[tuple[int, int], float]]:
        """What the windows meet on one graph: the smallest envelope of any
        node, and for every pair of types with an edge inside its window the
        shortest such edge, in units of the pair's window end (below 1 is
        inside). For checking windows against data, not for a hot path."""
        src, dst = edge_index[0].long(), edge_index[1].long()
        factor = self.envelope(atype, edge_index, edge_vec, edge_mask)
        smallest = float(factor.min()) if factor.numel() else 1.0
        real = (atype >= 0) & (atype < self.numbers.numel())
        ti = atype.clamp(0, self.numbers.numel() - 1).long()
        hi = self.r_hi[ti[src], ti[dst]]
        r = edge_vec.to(torch.float64).norm(dim=1)
        found = edge_mask.bool() & real[src] & real[dst] & (r < hi)
        pairs: dict[tuple[int, int], float] = {}
        for i, j, x in zip(
            ti[src][found].tolist(),
            ti[dst][found].tolist(),
            (r[found] / hi[found]).tolist(),
            strict=True,
        ):
            key = (min(i, j), max(i, j))
            pairs[key] = min(pairs.get(key, 1.0), x)
        return smallest, pairs

    # ------------------------------------------------------------ contacts

    def explicit(self, edge_vec: torch.Tensor) -> bool:
        """Whether this graph is evaluated through :meth:`contacts`: an
        evaluation, on a CUDA device, computed rather than traced."""
        return (
            not self.training
            and wp is not None
            and edge_vec.is_cuda
            and not _tracing(edge_vec)
        )

    def contacts(
        self,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        owned: torch.Tensor | None = None,
        capacity: int | None = None,
    ) -> Contacts:
        """One pass over the edges of an inference graph: the per-node ZBL
        energy and envelope, and the list of the edges in contact.

        ``capacity`` is the length of that list, by default one slot per
        node and 4096 more. A healthy simulation uses none of it; a graph
        with more contacts than that has collapsed, and its ZBL energies are
        set to NaN rather than its forces left short of terms. Nothing is
        read back to the host, the count included.
        """
        from deepmd_lr.nvalchemi.nlist import _torch_stream_scope

        n_node, n_edge = atype.shape[0], edge_vec.shape[0]
        device = edge_vec.device
        if capacity is None:
            capacity = n_node + 4096
        if owned is None:
            owned = torch.ones(n_node, dtype=torch.bool, device=device)
        energy = torch.zeros(n_node, dtype=torch.float64, device=device)
        log_sum = torch.zeros(n_node, dtype=torch.float64, device=device)
        closed = torch.zeros(n_node, dtype=torch.int32, device=device)
        count = torch.zeros(1, dtype=torch.int32, device=device)
        # Not zeroed: only the first `count` slots are read, through `found`.
        ends = torch.empty((3, capacity), dtype=torch.int64, device=device)
        values = torch.empty((3, capacity, 3), dtype=torch.float64, device=device)
        if n_edge:
            is32 = edge_vec.dtype == torch.float32
            # Bound to names, so no contiguous copy is freed before the launch.
            index = edge_index.to(torch.int64).contiguous()
            mask = edge_mask.to(torch.bool).contiguous()
            vec = edge_vec.detach().contiguous()
            types = atype.to(torch.int64).contiguous()
            own = owned.to(torch.bool).contiguous()
            reach = self.reach(types)

            def view(t, dtype):
                # A read-only launch argument; `t` is bound above, so it
                # stays alive until the launch.
                return wp.from_torch(
                    t, dtype=dtype, requires_grad=False, return_ctype=True
                )

            def arr(t, dtype):
                return wp.from_torch(t, dtype=dtype, requires_grad=False)

            with _torch_stream_scope(device):
                wp.launch(
                    _contacts_f32 if is32 else _contacts_f64,
                    dim=n_edge,
                    inputs=[
                        view(index, wp.int64),
                        view(mask, wp.bool),
                        view(vec, wp.vec3f if is32 else wp.vec3d),
                        view(types, wp.int64),
                        view(own, wp.bool),
                        view(self.numbers, wp.float64),
                        view(self.r_lo, wp.float64),
                        view(self.r_hi, wp.float64),
                        view(reach, wp.float64),
                        capacity,
                        arr(energy, wp.float64),
                        arr(log_sum, wp.float64),
                        arr(closed, wp.int32),
                        arr(count, wp.int32),
                        arr(ends, wp.int64),
                        arr(values, wp.vec3d),
                    ],
                    device=str(device),
                )
        found = torch.arange(capacity, device=device) < count
        ends = torch.where(found, ends, torch.zeros_like(ends))
        values = torch.where(found[None, :, None], values, torch.zeros_like(values))
        envelope = torch.where(
            closed > 0, torch.zeros_like(log_sum), torch.exp(log_sum)
        )
        return Contacts(
            energy, envelope, ends[0], ends[1], ends[2], values[0], values[1], values[2]
        )

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


# ----------------------------------------------------------------------------
# The Warp kernel: one pass over the edges of an inference graph
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

    @wp.func
    def _contact(
        e: int,
        v: wp.vec3d,
        r2: wp.float64,
        edge_index: wp.array2d(dtype=wp.int64),
        atype: wp.array(dtype=wp.int64),
        owned: wp.array(dtype=wp.bool),
        numbers: wp.array(dtype=wp.float64),
        r_lo: wp.array2d(dtype=wp.float64),
        r_hi: wp.array2d(dtype=wp.float64),
        capacity: int,
        energy: wp.array(dtype=wp.float64),
        log_sum: wp.array(dtype=wp.float64),
        closed: wp.array(dtype=wp.int32),
        count: wp.array(dtype=wp.int32),
        ends: wp.array2d(dtype=wp.int64),
        values: wp.array2d(dtype=wp.vec3d),
    ):
        """One valid edge shorter than the longest window: its ZBL half
        energy and its step onto the destination, and a slot in the list."""
        src = edge_index[0, e]
        dst = edge_index[1, e]
        i = atype[src]
        j = atype[dst]
        ro = r_hi[i, j]
        r = wp.sqrt(r2)
        if r >= ro:
            return
        ri = r_lo[i, j]
        g = wp.vec3d()
        if owned[dst]:
            pair = _zbl_pair(r, numbers[i], numbers[j], ri, ro)
            wp.atomic_add(energy, dst, pair[0])
            g = (pair[1] / r) * v
        dlog = wp.vec3d()
        step = _log_step(r, ri, ro)
        if step[0] > wp.float64(0.0):
            wp.atomic_add(closed, dst, 1)
        else:
            wp.atomic_add(log_sum, dst, step[0])
            dlog = (step[1] / r) * v
        slot = wp.atomic_add(count, 0, 1)
        if slot >= capacity:
            # More contacts than the list holds: the graph has collapsed.
            # Say so in the energy rather than leave the forces short.
            wp.atomic_add(energy, dst, wp.float64(wp.NAN))
            return
        ends[0, slot] = wp.int64(e)
        ends[1, slot] = src
        ends[2, slot] = dst
        values[0, slot] = v
        values[1, slot] = g
        values[2, slot] = dlog

    @wp.kernel(enable_backward=False)
    def _contacts_f32(
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        vec: wp.array(dtype=wp.vec3f),
        atype: wp.array(dtype=wp.int64),
        owned: wp.array(dtype=wp.bool),
        numbers: wp.array(dtype=wp.float64),
        r_lo: wp.array2d(dtype=wp.float64),
        r_hi: wp.array2d(dtype=wp.float64),
        r_max2: wp.array(dtype=wp.float64),
        capacity: int,
        energy: wp.array(dtype=wp.float64),
        log_sum: wp.array(dtype=wp.float64),
        closed: wp.array(dtype=wp.int32),
        count: wp.array(dtype=wp.int32),
        ends: wp.array2d(dtype=wp.int64),
        values: wp.array2d(dtype=wp.vec3d),
    ):
        e = wp.tid()
        v = wp.vec3d(
            wp.float64(vec[e][0]), wp.float64(vec[e][1]), wp.float64(vec[e][2])
        )
        # The distance first: almost every edge leaves here, having read
        # nothing but its own vector.
        r2 = wp.dot(v, v)
        if r2 >= r_max2[0]:
            return
        if not _edge_valid(e, edge_index, edge_mask, atype, numbers.shape[0]):
            return
        _contact(
            e, v, r2, edge_index, atype, owned, numbers, r_lo, r_hi, capacity,
            energy, log_sum, closed, count, ends, values,
        )  # fmt: skip

    @wp.kernel(enable_backward=False)
    def _contacts_f64(
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        vec: wp.array(dtype=wp.vec3d),
        atype: wp.array(dtype=wp.int64),
        owned: wp.array(dtype=wp.bool),
        numbers: wp.array(dtype=wp.float64),
        r_lo: wp.array2d(dtype=wp.float64),
        r_hi: wp.array2d(dtype=wp.float64),
        r_max2: wp.array(dtype=wp.float64),
        capacity: int,
        energy: wp.array(dtype=wp.float64),
        log_sum: wp.array(dtype=wp.float64),
        closed: wp.array(dtype=wp.int32),
        count: wp.array(dtype=wp.int32),
        ends: wp.array2d(dtype=wp.int64),
        values: wp.array2d(dtype=wp.vec3d),
    ):
        e = wp.tid()
        v = vec[e]
        r2 = wp.dot(v, v)
        if r2 >= r_max2[0]:
            return
        if not _edge_valid(e, edge_index, edge_mask, atype, numbers.shape[0]):
            return
        _contact(
            e, v, r2, edge_index, atype, owned, numbers, r_lo, r_hi, capacity,
            energy, log_sum, closed, count, ends, values,
        )  # fmt: skip
