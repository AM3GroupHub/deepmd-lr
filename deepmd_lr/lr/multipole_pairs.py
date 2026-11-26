# SPDX-License-Identifier: MIT
r"""The isolated sum of charges, dipoles and quadrupoles: every pair, no images.

The long-range term of a system with no periodic axis, for a model whose atoms
carry more than a charge. Each atom is a Gaussian density with a charge
:math:`q`, a dipole :math:`\mu` and a quadrupole :math:`Q` (second moment, its
energy in a potential :math:`\tfrac12 Q:\nabla\nabla\phi`), and two of them a
distance :math:`R = r_i - r_j` apart interact through
:math:`\phi(R) = \mathrm{erf}(\alpha R)/R`:

.. math::
    E_{ij} = \bigl(q_i + \mu_i\cdot\nabla + \tfrac12 Q_i:\nabla\nabla\bigr)
             \bigl(q_j - \mu_j\cdot\nabla + \tfrac12 Q_j:\nabla\nabla\bigr)
             \phi(R),
    \qquad e_i = \tfrac12\sum_{j\ne i} E_{ij} .

This is the infinite-volume limit of
:class:`~deepmd_lr.lr.multipole.MultipoleLongRangeEnergy` at the same alpha:
its self terms are the :math:`i = j` pair, which is left out here.

The derivatives of :math:`\phi` up to the fourth are polynomials in :math:`R`
with radial coefficients :math:`C_n = (-\tfrac1R\tfrac{d}{dR})^n\phi`,

.. math::
    \nabla_a\phi = -C_1 R_a, \qquad
    \nabla_a\nabla_b\phi = C_2 R_a R_b - C_1\delta_{ab}, \quad\ldots

and :math:`C_n = \alpha^{2n+1}\,2^{n+1}\pi^{-1/2}\,F_n(\alpha^2R^2)` with the
Boys function :math:`F_n(t) = \int_0^1 u^{2n} e^{-t u^2}\,du`. That form has
no singularity: :math:`F_n` is taken from its power series below ``t = 1`` and
from the upward recursion off :math:`\mathrm{erf}` above, so two atoms on top
of each other give a finite number and a finite gradient. Nothing in the
trace of :math:`Q` is assumed away: :math:`\phi` is not harmonic.

One dense kernel, in plain torch and float64: O(N^2) memory, differentiable to
any order by autograd (forces, force training, Hessians). A few thousand atoms.
"""

import math

import torch

__all__ = ["isolated_multipole_energy", "radial_coefficients"]

#: Below this ``t = (alpha R)^2`` the Boys function comes from its series.
_SERIES_T = 1.0
#: Terms of that series: at t = 1 the last one is 1/(21! * 51).
_SERIES_TERMS = 22
_HIGHEST = 4


def _boys(t: torch.Tensor, highest: int) -> list[torch.Tensor]:
    """``F_0 .. F_highest`` at ``t >= 0``, finite and smooth at zero."""
    small = t < _SERIES_T
    # The series for the highest order, then downward, which loses nothing.
    low = torch.where(small, t, torch.zeros_like(t))
    term = torch.ones_like(t)
    top = term / (2 * highest + 1)
    for m in range(1, _SERIES_TERMS):
        term = term * (-low / m)
        top = top + term / (2 * m + 2 * highest + 1)
    exp_low = torch.exp(-low)
    series = [top]
    for n in range(highest - 1, -1, -1):
        series.append((2.0 * low * series[-1] + exp_low) / (2 * n + 1))
    series.reverse()
    # From erf upward: each step multiplies the error by (2n + 1)/(2t), at
    # most 105/16 over the four of them at t = 1.
    high = torch.where(small, torch.ones_like(t), t)
    root = torch.sqrt(high)
    exp_high = torch.exp(-high)
    upward = [0.5 * math.sqrt(math.pi) * torch.erf(root) / root]
    for n in range(highest):
        upward.append(((2 * n + 1) * upward[-1] - exp_high) / (2.0 * high))
    return [torch.where(small, s, u) for s, u in zip(series, upward, strict=True)]


def radial_coefficients(
    r2: torch.Tensor, alpha: float, highest: int = _HIGHEST
) -> list[torch.Tensor]:
    """``C_n = (-(1/R) d/dR)^n erf(alpha R)/R`` for ``n = 0 .. highest``, from
    the squared distance ``r2``."""
    boys = _boys((alpha * alpha) * r2, highest)
    scale = alpha / math.sqrt(math.pi)
    return [scale * (2.0 ** (n + 1)) * alpha ** (2 * n) * f for n, f in enumerate(boys)]


def isolated_multipole_energy(
    coord: torch.Tensor,
    charge: torch.Tensor,
    alpha: float,
    dipole: torch.Tensor | None = None,
    quadrupole: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-atom energy of the isolated multipole sum, in e^2/angstrom.

    Parameters
    ----------
    coord
        ``(nf, nloc, 3)`` in angstrom; each frame is its own system.
    charge
        ``(nf, nloc)`` in e. An atom to leave out carries zero moments.
    alpha
        The splitting parameter, in inverse angstrom.
    dipole
        ``(nf, nloc, 3)`` in e angstrom, or None.
    quadrupole
        ``(nf, nloc, 3, 3)`` in e angstrom^2, symmetric, or None.

    Returns
    -------
    torch.Tensor
        ``(nf, nloc)``, float64.
    """
    coord = coord.to(torch.float64)
    q = charge.to(torch.float64)
    nloc = q.shape[1]
    delta = coord[:, :, None, :] - coord[:, None, :, :]
    highest = 0 if dipole is None else (2 if quadrupole is None else 4)
    c = radial_coefficients((delta * delta).sum(-1), alpha, highest)
    q_i, q_j = q[:, :, None], q[:, None, :]
    pair = c[0] * q_i * q_j
    if dipole is not None:
        mu = dipole.to(torch.float64)
        mu_i, mu_j = mu[:, :, None, :], mu[:, None, :, :]
        a_i = (mu_i * delta).sum(-1)
        a_j = (mu_j * delta).sum(-1)
        pair = pair + c[1] * (q_i * a_j - q_j * a_i + (mu_i * mu_j).sum(-1))
        pair = pair - c[2] * a_i * a_j
    if quadrupole is not None:
        theta = quadrupole.to(torch.float64)
        trace = theta.diagonal(dim1=-2, dim2=-1).sum(-1)
        t_i, t_j = trace[:, :, None], trace[:, None, :]
        # Q_i R and Q_j R, then R Q R.
        w_i = torch.einsum("fiab,fijb->fija", theta, delta)
        w_j = torch.einsum("fjab,fijb->fija", theta, delta)
        b_i = (w_i * delta).sum(-1)
        b_j = (w_j * delta).sum(-1)
        # charge - quadrupole
        pair = pair + 0.5 * q_i * (c[2] * b_j - c[1] * t_j)
        pair = pair + 0.5 * q_j * (c[2] * b_i - c[1] * t_i)
        # dipole - quadrupole
        third_ij = -c[3] * a_i * b_j + c[2] * (a_i * t_j + 2.0 * (mu_i * w_j).sum(-1))
        third_ji = -c[3] * a_j * b_i + c[2] * (a_j * t_i + 2.0 * (mu_j * w_i).sum(-1))
        pair = pair + 0.5 * (third_ij - third_ji)
        # quadrupole - quadrupole
        both = torch.einsum("fiab,fjab->fij", theta, theta)
        pair = pair + 0.25 * (
            c[4] * b_i * b_j
            - c[3] * (t_i * b_j + t_j * b_i + 4.0 * (w_i * w_j).sum(-1))
            + c[2] * (t_i * t_j + 2.0 * both)
        )
    eye = torch.eye(nloc, dtype=torch.bool, device=coord.device)
    pair = pair.masked_fill(eye, 0.0)
    return 0.5 * pair.sum(-1)
