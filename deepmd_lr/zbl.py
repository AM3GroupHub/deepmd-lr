# SPDX-License-Identifier: MIT
"""Short contacts on the DPA4C neighbour graph: ZBL takes over from the network.

Across a window per pair of elements, this module combines:

- **ZBL in**: the screened nuclear repulsion of Ziegler, Biersack and
  Littmark, ``V_ZBL(r) (1 - s)``;
- **the network out**: each atom's fitted energy is multiplied by the
  envelope ``w_i = prod_j s(r_ij)``, the product over its neighbours; its
  per-type energy offset is retained. Charges use this envelope as the
  projection's weight, ``q_i = w_i (raw_i + shift)``, so a closed atom takes
  no share of the charge deficit.

A window is ``[x_lo, x_hi]`` in units of the pair's covalent-radius sum,
``x = r / (R_i + R_j)`` with ASE's radii, and one C2 step
``s(x) = t^3 (10 - 15 t + 6 t^2)``, ``t = (x - x_lo) / (x_hi - x_lo)`` clamped
to [0, 1], is exactly 0 below it and exactly 1 above. A contact below ``x_lo``
closes both atoms' envelopes; with every neighbour beyond ``x_hi`` an atom's
factor is exactly 1.

Every pair uses :data:`DEFAULT_WINDOW` unless overridden by ``pairs``.
See ``docs/zbl.md`` for window selection and validation against training data.

ZBL has no parameters and sees the edge vectors through no network, so its
force and virial are assembled analytically from the per-edge derivative. The
envelope multiplies network outputs, so whatever is differentiated through
those outputs -- force, virial, the charge response, Born charges -- has to
pick its derivative up on the way.

The term stays outside the learned model. In an evaluation the envelope is a
plain number per atom, handed to whatever evaluates the network as a weight
on its atomic energies, and everything else is added to that result from here:
the ZBL energy with its force and virial, and the force and virial of the
envelope itself, ``(w f)_i d log w_i``. DeePMD-kit's own routes take such a
weight without being changed: the autograd lower as a factor on the fitting
net's output, the fused operators of ``DP_CUDA_INFER=2`` and of the compact
compressed export as the per-node ownership they seed their backward with.

Two ways through, with the same numbers:

- **An evaluation** (:meth:`ZBL.contacts`). One pass over the edges leaves
  the per-node ZBL energy and envelope and a short list of the edges in
  contact, with each one's derivatives; forces and virials are summed from
  that list (:func:`contact_output`, :func:`add_envelope_gradient`). In a
  healthy simulation the list is empty, and nothing the size of the edge
  list is computed twice. On a CUDA device the pass is one Warp kernel that
  reads nothing back to the host; where no kernel can run (a trace, which an
  export compiles; a CPU) it is one comparison per edge and the rest on the
  edges that passed it, as torch ops.
- **Training** (:meth:`ZBL.envelope`, :meth:`ZBL.graph_output`). Torch ops
  over every edge; the envelope is a differentiable function of the edge
  vectors, to any order, and the model's own backward carries it.
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

#: Default window in covalent-radius sums; validation is in ``docs/zbl.md``.
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
    window = None if window is None else list(check_window(window))
    pairs = {k: list(v) for k, v in check_pairs(settings.get("pairs")).items()}
    if window is None and not pairs:
        raise ValueError(
            "zbl has no window: `window` is null and `pairs` names no pair"
        )
    return {"window": window, "pairs": pairs}


def _distance(vec: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Edge lengths; a padded edge carries a zero vector and is given a
    harmless distance instead."""
    r2 = vec.square().sum(-1)
    return torch.sqrt(torch.where(valid, r2, torch.ones_like(r2)))


def _lower_output(
    energy: torch.Tensor,
    energy_redu: torch.Tensor,
    force: torch.Tensor,
    virial: torch.Tensor,
    atom_virial: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """The keys ``fit_output_to_model_output_graph`` writes for ``energy``,
    from per-node energies ``(N,)``, per-frame ones ``(nf,)``, forces, frame
    virials and, if asked for, per-atom virials."""
    n_cap, nf = energy.shape[0], energy_redu.shape[0]
    out = {
        "energy": energy.unsqueeze(-1),
        "energy_redu": energy_redu.unsqueeze(-1),
        "energy_derv_r": force.reshape(n_cap, 1, 3),
        "energy_derv_c_redu": virial.reshape(nf, 1, 9),
    }
    if atom_virial is not None:
        out["energy_derv_c"] = atom_virial.reshape(n_cap, 1, 9)
    return out


class Contacts(NamedTuple):
    """What one pass over an inference graph found (:meth:`ZBL.contacts`).

    Per node, the ZBL energy and the envelope; per edge in contact (closer
    than the upper edge of its pair's window), its ends, its vector and its
    two derivatives. The kernel's list has a fixed capacity and holds zeros
    beyond the contacts found, so every sum over it is a sum over the
    contacts; the torch list is exactly as long as the contacts. Float64
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
    enveloped: torch.Tensor,
    n_node: torch.Tensor,
    n_local: torch.Tensor | None,
    do_atomic_virial: bool = False,
) -> dict[str, torch.Tensor]:
    """The short-contact share of ``forward_common_lower_graph``'s result,
    from the list of contacts: the keys ``fit_output_to_model_output_graph``
    writes for ``energy``, to be added to the learned model's.

    That is the ZBL energy with its force and virial, and the force and
    virial the envelope adds to the fitted energy. The learned model was
    evaluated with the envelope as a plain weight on its fitted energies and
    returned ``enveloped = w f`` per node, ``(N, 1)``, with the force of
    ``w d f``; ``f d w = (w f) d log w`` is added here. Sums follow
    DeePMD-kit's ``edge_force_virial``: the gradient to the destination,
    minus it to the source, and ``-g (x) r`` to the frame's virial and, per
    atom, to the edge's source.
    """
    n_cap = contacts.energy.shape[0]
    nf = n_node.shape[0]
    coefficient = enveloped.detach().reshape(-1).to(torch.float64)
    # A ghost's energy is another rank's: it is not in the sum the learned
    # model differentiated either. One frame, the case of every simulation,
    # is spelled out: this runs every step, with nothing to sum most of them.
    if n_local is not None and n_local is not n_node:
        if nf == 1:
            owned = torch.arange(n_cap, device=n_local.device) < n_local
        else:
            owned = node_ownership_mask(n_node, n_local, n_cap)
        coefficient = coefficient * owned
    g = contacts.g + coefficient[contacts.dst][:, None] * contacts.dlog
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
    atom_virial = None
    if do_atomic_virial:
        atom_virial = torch.zeros((n_cap, 3, 3), dtype=torch.float64, device=g.device)
        atom_virial.index_add_(0, contacts.src, outer)
    return _lower_output(contacts.energy, energy_redu, force, virial, atom_virial)


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
        settings = normalized_settings(settings)
        window = settings["window"]
        #: The window of every pair ``pairs`` does not name, in covalent-radius
        #: sums, or None: then only those pairs have one.
        self.window = None if window is None else tuple(window)
        #: ``{"A-B": (x_lo, x_hi)}``, the pairs with a window of their own.
        self.pairs = {k: tuple(v) for k, v in settings["pairs"].items()}
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
        present.index_fill_(0, self._types(atype)[1], True)
        reach = (self.r_hi * (present[:, None] & present[None, :])).max()
        return (reach * reach).reshape(1)

    # ------------------------------------------------------------ per edge

    def _types(self, atype: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Whether each node is a real atom, and its type clamped into the
        tables so a padded node (type -1) indexes somewhere harmless."""
        n_type = self.numbers.numel()
        return (atype >= 0) & (atype < n_type), atype.clamp(0, n_type - 1).long()

    def _valid(
        self,
        atype: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The edges that count: unmasked, between two real atoms."""
        real, _ = self._types(atype)
        return edge_mask.bool() & real[src] & real[dst]

    def _window(
        self, ti: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Where each edge of length ``r`` is in its pair's window: ``t`` in
        [0, 1] across it, the window's width and its upper end, in ``r``'s
        dtype."""
        ts, td = ti[src], ti[dst]
        ri, ro = self.r_lo[ts, td].to(r.dtype), self.r_hi[ts, td].to(r.dtype)
        # An empty window (a pair with no nucleus) has no width to divide by.
        width = torch.where(ro > ri, ro - ri, torch.ones_like(ro))
        return ((r - ri) / width).clamp(0.0, 1.0), width, ro

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
        vec = edge_vec.to(torch.float64)
        r = _distance(vec, valid)
        ti = self._types(atype)[1]
        t, width, ro = self._window(ti, src, dst, r)
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
        switch = 1 - 10 * t.pow(3) + 15 * t.pow(4) - 6 * t.pow(5)
        dswitch = (-30 * t.pow(2) + 60 * t.pow(3) - 30 * t.pow(4)) / width
        on = valid & (r < ro)
        zero = torch.zeros_like(r)
        energy = torch.where(on, 0.5 * pair * switch, zero)
        dedr = torch.where(on, 0.5 * (dpair * switch + pair * dswitch), zero)
        return energy, ((dedr / r).unsqueeze(-1) * vec).to(edge_vec.dtype)

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
        valid = self._valid(atype, src, dst, edge_mask) & owned[dst]
        e_edge, g = self._edge_terms_torch(atype, src, dst, edge_vec, valid)
        energy = torch.zeros(n_node, dtype=torch.float64, device=edge_vec.device)
        return energy.index_add(0, dst, e_edge), g

    def atomic_energy(self, atype, edge_index, edge_vec, edge_mask):
        """Per-node energy, (N, 1) float64: each destination's half pairs."""
        energy, _g = self.edge_terms(atype, edge_index, edge_vec, edge_mask)
        return energy.unsqueeze(-1)

    # ------------------------------------------------------------ envelope

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
        valid = self._valid(atype, src, dst, edge_mask)
        log_s, _ = self._step_terms_torch(atype, src, dst, edge_vec, valid)
        log_w = torch.zeros(
            atype.shape[0], dtype=log_s.dtype, device=log_s.device
        ).index_add(0, dst, log_s)
        return torch.exp(log_w)

    def _step_terms_torch(
        self,
        atype: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        vec: torch.Tensor,
        valid: torch.Tensor,
        *,
        derivative: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-edge ``log s`` in the vector's dtype, and optionally its
        analytic gradient for the contact list. Training differentiates the
        log step itself and needs no explicit gradient."""
        r = _distance(vec, valid)
        t, width, ro = self._window(self._types(atype)[1], src, dst, r)
        s = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
        inside = valid & (r < ro)
        closed = inside & (s <= 0)
        partial = inside & ~closed
        s = torch.where(partial, s, torch.ones_like(s))
        # A closed edge contributes the constant -inf: exp(sum(log s)) and
        # its derivatives are exactly zero, without differentiating log(0).
        log_s = torch.where(closed, torch.full_like(s, -torch.inf), s.log())
        if not derivative:
            return log_s, None
        # d s / d r = 30 t^2 (1 - t)^2 / width, over s for the logarithm.
        dlog = torch.where(
            partial,
            30.0 * t * t * (1.0 - t) * (1.0 - t) / (width * s),
            torch.zeros_like(s),
        )
        return log_s, (dlog / r).unsqueeze(-1) * vec

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
        ti = self._types(atype)[1]
        hi = self.r_hi[ti[src], ti[dst]]
        r = edge_vec.to(torch.float64).norm(dim=1)
        found = self._valid(atype, src, dst, edge_mask) & (r < hi)
        # The shortest edge of every pair of types, one bucket per pair.
        n_type = self.numbers.numel()
        low, high = torch.minimum(ti[src], ti[dst]), torch.maximum(ti[src], ti[dst])
        shortest = torch.full((n_type * n_type,), torch.inf, dtype=torch.float64)
        shortest = shortest.to(r.device).scatter_reduce_(
            0, (low * n_type + high)[found], (r / hi)[found], reduce="amin"
        )
        pairs = {
            (int(k) // n_type, int(k) % n_type): float(shortest[k])
            for k in torch.nonzero(torch.isfinite(shortest)).reshape(-1)
        }
        return smallest, pairs

    # ------------------------------------------------------------ contacts

    def explicit(self, edge_vec: torch.Tensor) -> bool:
        """Whether :meth:`contacts` runs its kernel on this graph: an
        evaluation, on a CUDA device, computed rather than traced. Anywhere
        else it builds the same list with torch ops."""
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

        On a CUDA device (:meth:`explicit`) the pass is a Warp kernel, and
        ``capacity`` is the length of its list, by default one slot per node
        and 4096 more. A healthy simulation uses none of it; a graph with
        more contacts than that has collapsed, and its ZBL energies are set
        to NaN rather than its forces left short of terms. Nothing is read
        back to the host, the count included. Anywhere else the list is
        built with torch ops (:meth:`_contacts_torch`) and is as long as it
        has to be.
        """
        if not self.explicit(edge_vec):
            return self._contacts_torch(
                atype,
                edge_index[0],
                edge_index[1],
                edge_vec,
                edge_mask.to(torch.bool),
                owned,
            )
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

    def canonical_contacts(
        self,
        atype: torch.Tensor,
        source: torch.Tensor,
        edge_vec: torch.Tensor,
        destination_row_ptr: torch.Tensor,
        owned: torch.Tensor | None = None,
    ) -> Contacts:
        """:meth:`contacts` for DeePMD-kit's compact canonical graph, which a
        compressed export is traced on: every edge once, grouped by
        destination, with the source index and the CSR row offsets in place
        of an edge index. ``Contacts.edge`` is the position in that stream.
        """
        return self._contacts_torch(
            atype, source, None, edge_vec, None, owned, destination_row_ptr
        )

    def _contacts_torch(
        self,
        atype: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor | None,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor | None,
        owned: torch.Tensor | None,
        destination_row_ptr: torch.Tensor | None = None,
    ) -> Contacts:
        """The list of :meth:`contacts` as torch ops, for where the kernel
        cannot run: a trace, which an export compiles, and a CPU.

        The graph holds hundreds of edges per atom and a healthy frame no
        contact at all, so the edges are narrowed twice before anything is
        computed on them. One comparison per edge, written on the three
        components so that a compiler makes a single pass of it, keeps those
        shorter than the longest window of the types present; on those few
        the pair's own window is looked up, and the terms are evaluated on
        what is left, a list exactly as long as the contacts. An edge's
        destination is ``destination`` where the graph lists it, or the row
        of ``destination_row_ptr`` its position falls in; ``edge_mask`` may
        be None for a graph without masked edges.
        """
        n_node = atype.shape[0]
        device = edge_vec.device
        if owned is None:
            owned = torch.ones(n_node, dtype=torch.bool, device=device)
        x, y, z = edge_vec.detach().unbind(-1)
        near = (x * x + y * y + z * z) < self.reach(atype).to(edge_vec.dtype)
        if edge_mask is not None:
            near = near & edge_mask
        edge = torch.nonzero(near)[:, 0]
        if source.dtype == torch.uint32:
            # The compact graph's index type, which CUDA cannot gather from;
            # node indices fit the signed type of the same width.
            source = source.view(torch.int32)
        src = source[edge].to(torch.int64)
        if destination is not None:
            dst = destination[edge].to(torch.int64)
            stored = torch.ones_like(edge, dtype=torch.bool)
        else:
            # The number of rows that end at or before the edge. On the
            # whole offsets, not `destination_row_ptr[1:]`: AOTInductor
            # (torch 2.11) compiles `searchsorted` on a sliced sequence
            # wrongly for CUDA, and an export is compiled. Storage past the
            # last row (a graph of fewer than two edges is stored as two)
            # holds no edge.
            dst = torch.searchsorted(destination_row_ptr, edge, right=True) - 1
            dst = dst.clamp(max=n_node - 1)
            stored = edge < destination_row_ptr[-1]
        vec = edge_vec.detach()[edge].to(torch.float64)
        real, ti = self._types(atype)
        longest = self.r_hi[ti[src], ti[dst]]
        contact = (
            stored & real[src] & real[dst] & (vec.square().sum(-1) < longest * longest)
        )
        kept = torch.nonzero(contact)[:, 0]
        edge, src, dst, vec = edge[kept], src[kept], dst[kept], vec[kept]
        valid = torch.ones_like(edge, dtype=torch.bool)
        e_edge, g = self._edge_terms_torch(atype, src, dst, vec, owned[dst])
        log_s, dlog = self._step_terms_torch(
            atype, src, dst, vec, valid, derivative=True
        )
        zero = torch.zeros(n_node, dtype=torch.float64, device=device)
        energy = zero.index_add(0, dst, e_edge)
        envelope = torch.exp(zero.index_add(0, dst, log_s))
        return Contacts(energy, envelope, edge, src, dst, vec, g, dlog)

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
        """ZBL's share of ``forward_common_lower_graph``'s result in
        training, as torch ops over every edge: the keys
        ``fit_output_to_model_output_graph`` writes for ``energy``, to be
        added to the learned model's, whose own backward has differentiated
        the envelope. (An evaluation adds both from :func:`contact_output`.)
        """
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
        return _lower_output(
            energy,
            energy_redu,
            force,
            virial,
            atom_virial if do_atomic_virial else None,
        )


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
    def _contacts(
        edge_index: wp.array2d(dtype=wp.int64),
        edge_mask: wp.array(dtype=wp.bool),
        vec: wp.array(dtype=Any),
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
        """One edge: if it is a contact, its ZBL half energy and its step
        onto the destination, and a slot in the list."""
        e = wp.tid()
        v = wp.vec3d(
            wp.float64(vec[e][0]), wp.float64(vec[e][1]), wp.float64(vec[e][2])
        )
        # The distance first: almost every edge leaves here, having read
        # nothing but its own vector.
        r2 = wp.dot(v, v)
        if r2 >= r_max2[0] or not edge_mask[e]:
            return
        src = edge_index[0, e]
        dst = edge_index[1, e]
        i = atype[src]
        j = atype[dst]
        # A padded node has type -1: no real atom, no contact.
        n_type = wp.int64(numbers.shape[0])
        if i < wp.int64(0) or i >= n_type or j < wp.int64(0) or j >= n_type:
            return
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

    # The graph's own precision in, float64 out.
    _contacts_f32 = wp.overload(_contacts, {"vec": wp.array(dtype=wp.vec3f)})
    _contacts_f64 = wp.overload(_contacts, {"vec": wp.array(dtype=wp.vec3d)})
