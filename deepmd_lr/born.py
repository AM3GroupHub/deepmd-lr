# SPDX-License-Identifier: MIT
r"""Born effective charge tensors from the model's own predicted charges.

The tensor, adapting 10.1021/acs.jctc.6c00146:

.. math::
    Z_{i\alpha\beta} = \tilde q_i \delta_{\alpha\beta}
      + \tfrac12 \sum_{j\neq i}\left(
            r_{ji,\alpha}\frac{\partial \tilde q_i}{\partial r_{ji,\beta}}
          - r_{ij,\alpha}\frac{\partial \tilde q_j}{\partial r_{ij,\beta}}
        \right)

What this is, and what it is not
--------------------------------
**It is not the dipole derivative.** That distinction decides whether a field
force is conservative. For a charge model carrying a global total-charge
constraint, the exact dipole derivative of an isolated system is

.. math::
    \frac{\partial M_\alpha}{\partial r_{i\beta}}
      = q_i\delta_{\alpha\beta}
      - \sum_{e:\,\mathrm{src}(e)=i}(\mathbf r_e \otimes \mathbf g_e)_{\alpha\beta}
      + (\bar{\mathbf r} - \mathbf r_i)_\alpha S_{i\beta}

with :math:`\mathbf g_e = \partial \mathrm{raw}_{\mathrm{dst}(e)}/\partial
\mathbf r_e`, :math:`S_i = \sum_k \partial\,\mathrm{raw}_k/\partial \mathbf r_i`
and :math:`\bar{\mathbf r}` the centroid. Verified against finite differences to
**5e-11** across every component.

That final term is the problem. It is proportional to the displacement from the
centroid, so it is unbounded and depends on where the system sits -- meaningless
under periodic boundary conditions, where :math:`M` itself is undefined because
position is not periodic. The formula above keeps only the local,
translation-invariant content and discards it. On a compact neutral molecule the
two differ by 1.9e-01 against a :math:`|Z|` scale of 0.38, and that gap *is* the
discarded term.

The consequence, stated plainly: :math:`F_i = Z_i^\mathsf{T} E` is **not**
:math:`-\partial(-M\cdot E)/\partial r_i`. It is the local response, which is the
only thing well defined in a periodic cell -- but a constant-field NVE run will
not conserve :math:`E_{\text{tot}} - M\cdot E`, by construction rather than by
bug. Do not read a drift there as an integration error.

Why one backward pass suffices
------------------------------
DPA4C has no message passing (``has_message_passing()`` is ``False``), so a
feature depends only on edges where its atom is the *centre*:
``d raw_i / d edge_vec_e`` is nonzero only when ``dst(e) == i``. One
``autograd.grad`` of the scalar ``sum_j raw_j`` therefore yields the complete
per-edge derivative in a single pass -- not one pass per atom, which is what a
naive Jacobian would cost.

Raw or constrained charges
--------------------------
The derivative uses the **raw**, pre-projection charges; the :math:`\delta`
term uses the **constrained** ones. ``project_to_total_charge`` subtracts a
global mean, which makes every charge depend on every position: moving an atom
32.9 A away (rcut 6.0) shifts every distant charge by a uniform 1.6e-02, an
amount comparable to genuine neighbour derivatives (6.3e-02 against 1.8e-01).
Weighted by an unbounded :math:`r_{ji}`, those terms dominate -- a literal
all-pairs evaluation differs from the neighbour-only one by 1.04 on a
:math:`|Z|` scale of 2.09, and grows with system size.

Splitting the roles keeps what the constraint is for and drops the artefact:
the acoustic sum rule :math:`\sum_i Z_i = Q\,\mathbf 1` still holds *exactly*,
because the ``dst - src`` bracket telescopes to zero whatever the derivative is.

Which is also why that sum rule is worthless as a check: it held to 1e-16 for
variants disagreeing with each other by 100%. It is a regression guard, not a
validation.
"""

import math
from numbers import (
    Real,
)
from typing import (
    Any,
)

import torch
from deepmd.dpmodel.utils.neighbor_graph import (
    segment_sum,
)

__all__ = ["born_charges", "field_force", "screening_factor"]


def screening_factor(eps_inf: Any) -> float:
    r"""The factor :math:`\sqrt{\varepsilon_\infty}` from model to screened charges.

    The model's Coulomb term is :math:`q_i q_j / (4\pi\varepsilon_0 r_{ij})`.
    In a medium with high-frequency dielectric constant
    :math:`\varepsilon_\infty` the same energy reads
    :math:`\tilde q_i \tilde q_j / (4\pi\varepsilon_0\varepsilon_\infty r_{ij})`,
    so the physical (screened) charges are
    :math:`\tilde q = \sqrt{\varepsilon_\infty}\, q`, and the Born tensors,
    being derivatives of the same charges, scale by the same factor. The
    energy and its forces are unchanged: this is a change of units on the
    reported charges, not on the interaction. ``eps_inf = 1`` is vacuum and
    leaves everything as it was.
    """
    if isinstance(eps_inf, bool) or not isinstance(eps_inf, Real):
        raise TypeError(f"eps_inf must be a positive finite number, got {eps_inf!r}")
    value = float(eps_inf)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"eps_inf must be positive and finite, got {eps_inf!r}; the vacuum "
            "value is 1.0 and a physical high-frequency dielectric constant is "
            "at least that."
        )
    return math.sqrt(value)


def edge_born_response(g, vec, edge_index, edge_mask, n_node: int) -> torch.Tensor:
    """The edge half of the Born tensor, per node: ``(n_node, 3, 3)``.

    ``g`` is ``d raw_charge / d edge_vec`` for every edge and ``vec`` the edge
    vectors themselves. Each directed edge contributes ``+1/2 r (x) g`` to its
    destination and ``-1/2 r (x) g`` to its source -- the same quantity
    gathered at opposite ends, which is why this is a difference of segment
    sums rather than a double loop over pairs. Padding edges are masked out.
    """
    g = g * edge_mask[:, None].to(g.dtype)
    # (E, 3, 3), r_ji (x) dq/dr_ji
    outer = vec[:, :, None] * g[:, None, :]
    src = edge_index[0].to(torch.int64)
    dst = edge_index[1].to(torch.int64)
    return 0.5 * (segment_sum(outer, dst, n_node) - segment_sum(outer, src, n_node))


def born_charges(
    charge: torch.Tensor,
    raw: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    create_graph: bool = False,
    eps_inf: float = 1.0,
) -> torch.Tensor:
    """Per-atom Born effective charge tensors, shape ``(nf, nloc, 3, 3)``.

    Parameters
    ----------
    charge
        Constrained per-atom charges, ``(nf, nloc)``. Supplies the
        :math:`q_i\\delta_{\\alpha\\beta}` term, so the acoustic sum rule comes
        out as the frame's actual total charge.
    raw
        Pre-projection charges on the flat node axis, ``(N,)``, still attached
        to ``edge_vec`` through the descriptor. Differentiated once.
    edge_vec
        ``(E, 3)``, ``r_src - r_dst`` (neighbour minus centre) -- the autograd
        leaf, and exactly the paper's :math:`r_{ji}`. The sign matters: the
        opposite convention was measured 6x worse against the true dipole
        derivative.
    edge_index
        ``(2, E)`` ``[src, dst]``; src is the neighbour, dst the centre.
    edge_mask
        ``(E,)`` bool, real edges versus padding.
    create_graph
        Keep the graph, for differentiating through ``Z`` itself. Off by
        default: nothing needs it yet and it costs memory.
    eps_inf
        High-frequency dielectric constant of the medium. The tensor is
        multiplied by :math:`\\sqrt{\\varepsilon_\\infty}`, the factor that
        turns the model's Coulomb charges into screened ones; see
        :func:`screening_factor`. ``1.0`` (vacuum) changes nothing.

    Returns
    -------
    torch.Tensor
        ``(nf, nloc, 3, 3)``. Not symmetric -- a Born tensor has no reason to be,
        which is why the ``polarizability`` fitting's unconditional
        symmetrization would be wrong here.
    """
    nf, nloc = charge.shape
    n_total = raw.shape[0]
    if n_total != nf * nloc:
        # Happens when `compact_nodes` drops phantom atoms from a ragged batch.
        # Refuse rather than reshape into nonsense.
        raise ValueError(
            f"raw charges hold {n_total} nodes but charge is ({nf}, {nloc}) = "
            f"{nf * nloc}. The node axis was compacted, so Born charges cannot "
            "be mapped back onto the frame layout."
        )

    # One backward pass for every edge. Summing over j is legitimate only
    # because the descriptor is one-hop: each edge feeds exactly one atom's
    # charge, its centre, so the summed cotangent carries no cross terms.
    (g,) = torch.autograd.grad(
        raw.sum(), edge_vec, create_graph=create_graph, retain_graph=True
    )
    # Defensive, not load-bearing: DPA4C multiplies its cutoff envelope by the
    # mask, so a guard edge already contributes an identically zero gradient.
    # `edge_force_virial` masks anyway, and so do we -- a future builder that
    # stopped zeroing them would otherwise corrupt Z silently.
    vec = edge_vec if create_graph else edge_vec.detach()
    response = edge_born_response(g, vec, edge_index, edge_mask, n_total)

    eye = torch.eye(3, dtype=charge.dtype, device=charge.device)
    diag = charge.reshape(-1)[:, None, None] * eye[None]
    out = (diag + response.to(charge.dtype)).reshape(nf, nloc, 3, 3)
    # Both halves are built from the same charges, so the screening factor
    # multiplies the whole tensor: Z~ = sqrt(eps_inf) Z, as q~ = sqrt(eps_inf) q.
    factor = screening_factor(eps_inf)
    if factor != 1.0:
        out = out * factor
    return out if create_graph else out.detach()


def field_force(born: torch.Tensor, efield: Any) -> torch.Tensor:
    """Force from a uniform external field, ``F_i = Z_i.T E``.

    ``born`` is ``(nf, nloc, 3, 3)`` and ``efield`` a 3-vector in V/A; the
    result is ``(nf, nloc, 3)`` in eV/A.

    This is a response, not the gradient of an energy -- see the module
    docstring. Nothing here contributes to the stress, and the field's own
    contribution to it is deliberately not guessed at.
    """
    e = torch.as_tensor(efield, dtype=born.dtype, device=born.device).reshape(3)
    # First tensor axis is polarization; second is atomic displacement.
    return torch.einsum("fiab,a->fib", born, e)
