# SPDX-License-Identifier: MIT
r"""The isolated erf sum: the long-range term of a system with no periodic axis.

.. math::
    e_i = \tfrac12 q_i \sum_{j \ne i,\ \text{same frame}} q_j
          \frac{\operatorname{erf}(\alpha r_{ij})}{r_{ij}}
    \quad [e^2/\text{A}],

no self term, no cutoff, every pair (see
:class:`~deepmd_lr.lr.term.IsolatedLongRangeEnergy` for why this is the
periodic term's infinite-volume limit). Two kernels evaluate it:

* :func:`isolated_erf_energy_dense` builds the ``(nf, nloc, nloc, 3)``
  displacement tensor and lets autograd through every intermediate, so it is
  twice differentiable (force training, Hessians). At 27,000 atoms (a 40 A
  water droplet) that tensor alone is 17 GB in float64.
* :func:`isolated_erf_energy_chunked` works in row blocks of ``chunk`` atoms
  against all N, so peak memory is O(chunk N), and takes the backward from the
  analytic first derivatives in one more such pass with the cotangent folded
  in: for ``L = sum_i g_i e_i``,

  .. math::
      \frac{\partial L}{\partial q_k}
          = \sum_{j \ne k} \tfrac12 (g_k + g_j)\, q_j\, k(r_{kj}),
      \qquad
      \frac{\partial L}{\partial \mathbf r_k}
          = q_k \sum_{j \ne k} \tfrac12 (g_k + g_j)\, q_j\,
          \frac{k'(r_{kj})}{r_{kj}}\,(\mathbf r_k - \mathbf r_j),

  with ``k(r) = erf(alpha r)/r``. Exact for any cotangent, so nothing inspects
  ``g`` on the host (the MD engine's rule: a step reads nothing back). On CUDA
  the blocks run under ``torch.compile``, where inductor fuses each into one
  reduction kernel: at 27,000 atoms in float32 on a GB10, 8 ms forward and
  37 ms backward, against 1.35 s and 1.68 s eager (0.30 s and 0.50 s compiled
  in float64). ``create_graph=True`` is refused.

:func:`isolated_erf_energy` picks between them (``isolated_kernel``).

Near coincidence both replace the analytic branch by the series
``(2 alpha/sqrt(pi)) (1 - x^2/3 + x^4/10 - x^6/42)``, ``x = alpha r``, for
``x^2 < 1e-4`` (and the chunked kernel by its derivative), so both stay finite
at r -> 0 and agree to round-off.
"""

from __future__ import annotations

import math
import warnings
from functools import cache

import torch

__all__ = [
    "COMPILE_MIN_ATOMS",
    "DEFAULT_CHUNK",
    "DEFAULT_MAX_ISOLATED_ATOMS",
    "ISOLATED_KERNELS",
    "isolated_erf_energy",
    "isolated_erf_energy_chunked",
    "isolated_erf_energy_dense",
]

#: Rows per block: each block holds a few ``(chunk, N)`` intermediates.
DEFAULT_CHUNK = 1024

#: Below this many atoms ``compile=None`` runs the blocks eagerly: the compile
#: costs seconds once per process and distinct shape, which small clusters
#: (and a test suite full of them) never earn back.
COMPILE_MIN_ATOMS = 1024

_SERIES_X2 = 1e-4
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


def _kernel_terms(
    delta: torch.Tensor, active: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``k(r) = erf(alpha r)/r`` and ``k'(r)/r`` for one block, zero where
    not ``active``."""
    r2 = (delta * delta).sum(-1)
    x2 = (alpha * alpha) * r2
    small = x2 < _SERIES_X2
    safe_r2 = torch.where(small, torch.ones_like(r2), r2)
    safe_r = torch.sqrt(safe_r2)
    erf = torch.erf(alpha * safe_r)
    a = _TWO_OVER_SQRT_PI * alpha
    series_k = a * (1.0 - x2 / 3.0 + x2 * x2 / 10.0 - x2 * x2 * x2 / 42.0)
    k = torch.where(small, series_k, erf / safe_r)
    # k'(r)/r = (2 alpha/sqrt(pi)) exp(-x^2)/r^2 - erf(alpha r)/r^3, and the
    # series' derivative, a alpha^2 (-2/3 + 2 x^2/5 - x^4/7), near r = 0.
    series_kp = (a * alpha * alpha) * (-2.0 / 3.0 + 0.4 * x2 - x2 * x2 / 7.0)
    kp = torch.where(small, series_kp, (a * torch.exp(-x2) - erf / safe_r) / safe_r2)
    zero = torch.zeros_like(k)
    return torch.where(active, k, zero), torch.where(active, kp, zero)


def _energy_block(
    pos_i: torch.Tensor,
    idx_i: torch.Tensor,
    batch_i: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    batch: torch.Tensor,
    index: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """``sum_j q_j k(r_ij)`` for the rows ``i`` of the block, shape ``(chunk,)``."""
    delta = pos_i[:, None, :] - positions[None, :, :]
    active = (idx_i[:, None] != index[None, :]) & (batch_i[:, None] == batch[None, :])
    k, _ = _kernel_terms(delta, active, alpha)
    return k @ charges


def _grad_block(
    pos_i: torch.Tensor,
    q_i: torch.Tensor,
    g_i: torch.Tensor,
    idx_i: torch.Tensor,
    batch_i: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    g: torch.Tensor,
    batch: torch.Tensor,
    index: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``dL/dq`` ``(chunk,)`` and ``dL/dr`` ``(chunk, 3)`` for the rows of the block."""
    delta = pos_i[:, None, :] - positions[None, :, :]
    active = (idx_i[:, None] != index[None, :]) & (batch_i[:, None] == batch[None, :])
    k, kp = _kernel_terms(delta, active, alpha)
    w = 0.5 * (g_i[:, None] + g[None, :]) * charges[None, :]
    dq = (w * k).sum(1)
    dr = q_i[:, None] * ((w * kp)[:, :, None] * delta).sum(1)
    return dq, dr


# ---------------------------------------------------------------- compilation
_COMPILE_OK = True


@cache
def _compiled_blocks() -> tuple:
    # Static shapes: a system has one N for the life of a run, and the
    # specialised kernel is the fast one. A new N recompiles.
    return (
        torch.compile(_energy_block, dynamic=False),
        torch.compile(_grad_block, dynamic=False),
    )


def _call(compiled_fn, eager_fn, *args):
    """The compiled block, or the eager one after a failed compile (warned once)."""
    global _COMPILE_OK  # noqa: PLW0603
    if compiled_fn is not None and _COMPILE_OK:
        try:
            return compiled_fn(*args)
        except torch.cuda.OutOfMemoryError:
            # Memory, not a missing compiler: the eager blocks need more of it,
            # and the caller's own back-off has to see the error.
            raise
        except Exception as exc:  # no inductor backend, no triton for this device ...
            _COMPILE_OK = False
            message = str(exc).split("\n", 1)[0]
            warnings.warn(
                "torch.compile of the isolated erf kernel failed "
                f"({type(exc).__name__}: {message}); running the "
                "eager blocks from now on (same result, several times slower)",
                RuntimeWarning,
                stacklevel=3,
            )
    return eager_fn(*args)


def _should_compile(positions: torch.Tensor, compile: bool | None) -> bool:
    if compile is not None:
        return bool(compile)
    return positions.device.type == "cuda" and positions.shape[0] >= COMPILE_MIN_ATOMS


class _IsolatedErf(torch.autograd.Function):
    """``e_i`` in e^2/A with the analytic first derivatives; no double backward."""

    @staticmethod
    def forward(ctx, positions, charges, batch, alpha, chunk, compiled):
        n = positions.shape[0]
        pad = (-n) % chunk
        if pad:
            # Zero-charge atoms in no graph (batch -1): every block then has
            # the same (chunk, N_pad) shape, and the padding contributes
            # nothing (inactive pairs are masked before any charge is read).
            positions = torch.cat([positions, positions.new_zeros((pad, 3))])
            charges = torch.cat([charges, charges.new_zeros(pad)])
            batch = torch.cat([batch, batch.new_full((pad,), -1)])
        n_pad = n + pad
        index = torch.arange(n_pad, device=positions.device)
        energy_fn, _ = _compiled_blocks() if compiled else (None, None)
        kq = torch.empty_like(charges)
        for start in range(0, n_pad, chunk):
            stop = start + chunk
            kq[start:stop] = _call(
                energy_fn,
                _energy_block,
                positions[start:stop],
                index[start:stop],
                batch[start:stop],
                positions,
                charges,
                batch,
                index,
                alpha,
            )
        ctx.save_for_backward(positions, charges, batch, index)
        ctx.alpha, ctx.chunk, ctx.compiled, ctx.n = alpha, chunk, compiled, n
        return (0.5 * charges * kq)[:n]

    @staticmethod
    def backward(ctx, grad_energy):
        # once_differentiable can silently detach these partials while a
        # position-dependent charge head still supplies an incomplete graph.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "the chunked isolated kernel supports first derivatives only; "
                "use isolated_kernel 'dense' for force training or Hessians, "
                "raising max_isolated_atoms if needed (dense O(N^2) memory)"
            )
        positions, charges, batch, index = ctx.saved_tensors
        n, chunk = ctx.n, ctx.chunk
        n_pad = positions.shape[0]
        g = grad_energy.to(charges.dtype)
        if n_pad > n:
            g = torch.cat([g, g.new_zeros(n_pad - n)])
        _, grad_fn = _compiled_blocks() if ctx.compiled else (None, None)
        dq = torch.empty_like(charges)
        dr = torch.empty_like(positions)
        for start in range(0, n_pad, chunk):
            stop = start + chunk
            dq[start:stop], dr[start:stop] = _call(
                grad_fn,
                _grad_block,
                positions[start:stop],
                charges[start:stop],
                g[start:stop],
                index[start:stop],
                batch[start:stop],
                positions,
                charges,
                g,
                batch,
                index,
                ctx.alpha,
            )
        return dr[:n], dq[:n], None, None, None, None


def isolated_erf_energy_chunked(
    positions: torch.Tensor,
    charges: torch.Tensor,
    alpha: float,
    *,
    batch: torch.Tensor | None = None,
    chunk: int = DEFAULT_CHUNK,
    compile: bool | None = None,
) -> torch.Tensor:
    r"""``e_i = ½ q_i Σ_{j≠i, same graph} q_j erf(α r_ij)/r_ij`` in e^2/A, ``(N,)``.

    The quantity of :func:`isolated_erf_energy_dense` on a flat atom list
    (pairs within one graph when ``batch`` is given), in blocks of ``chunk``
    rows, with analytic first derivatives with respect to ``positions`` and
    ``charges``; ``create_graph=True`` is refused. ``compile=None`` compiles
    the blocks on CUDA from :data:`COMPILE_MIN_ATOMS` atoms, ``True`` and
    ``False`` force it either way. float32 and float64 alike.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (N, 3), got {tuple(positions.shape)}"
        )
    n = positions.shape[0]
    if charges.shape != (n,):
        raise ValueError(f"charges must have shape ({n},), got {tuple(charges.shape)}")
    if not float(alpha) > 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if int(chunk) < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if n == 0:
        return torch.zeros(0, dtype=positions.dtype, device=positions.device)
    # Small clusters need only one block, without padding up to DEFAULT_CHUNK.
    chunk = min(int(chunk), n)
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=positions.device)
    else:
        if batch.shape != (n,):
            raise ValueError(f"batch must have shape ({n},), got {tuple(batch.shape)}")
        batch = batch.to(device=positions.device, dtype=torch.long)
    charges = charges.to(positions.dtype)
    return _IsolatedErf.apply(
        positions,
        charges,
        batch,
        float(alpha),
        int(chunk),
        _should_compile(positions, compile),
    )


def isolated_erf_energy_dense(
    coord: torch.Tensor, charge: torch.Tensor, alpha: float
) -> torch.Tensor:
    """``e_i`` in e^2/A, ``(nf, nloc)``, every pair of a frame at once.

    ``coord`` ``(nf, nloc, 3)``, ``charge`` ``(nf, nloc)``; a padding atom
    given zero charge drops out exactly, every term being bilinear in the
    charges. Twice differentiable.
    """
    nloc = coord.shape[1]
    delta = coord.unsqueeze(2) - coord.unsqueeze(1)  # (nf, nloc, nloc, 3)
    r2 = (delta * delta).sum(-1)
    active = ~torch.eye(nloc, dtype=torch.bool, device=coord.device)
    # The series in r^2 keeps the first and second derivatives finite at
    # coincidence; the diagonal never reaches the analytic branch's sqrt(0).
    x2 = (alpha * alpha) * torch.where(active, r2, torch.zeros_like(r2))
    small = x2 < _SERIES_X2
    safe_r = torch.sqrt(torch.where(small, torch.ones_like(r2), r2))
    series = (_TWO_OVER_SQRT_PI * alpha) * (1 - x2 / 3 + x2**2 / 10 - x2**3 / 42)
    kernel = torch.where(small, series, torch.erf(alpha * safe_r) / safe_r)
    kernel = torch.where(active, kernel, torch.zeros_like(kernel))
    # Halved: the double sum counts every pair twice, and splitting it
    # symmetrically makes this a per-atom energy.
    return 0.5 * charge * (kernel * charge.unsqueeze(1)).sum(-1)


#: ``isolated_kernel``: the dense kernel, the chunked one, or dense up to
#: ``max_isolated_atoms`` and chunked beyond.
ISOLATED_KERNELS = ("auto", "dense", "chunked")

#: Largest system, in atoms over all frames of a call, the dense kernel takes.
DEFAULT_MAX_ISOLATED_ATOMS = 5000


def isolated_erf_energy(
    coord: torch.Tensor,
    charge: torch.Tensor,
    alpha: float,
    *,
    kernel: str = "auto",
    max_dense_atoms: int = DEFAULT_MAX_ISOLATED_ATOMS,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``e_i`` in e^2/A, ``(nf, nloc)``, by the kernel ``kernel`` names.

    ``"auto"`` takes the dense kernel up to ``max_dense_atoms`` atoms in the
    call (``nf * nloc``) and the chunked one beyond; ``"dense"`` refuses a
    larger call. ``dtype`` is the dtype the sum runs in (the inputs' own by
    default); the result comes back in the inputs' dtype.
    """
    if kernel not in ISOLATED_KERNELS:
        raise ValueError(
            f"isolated_kernel must be one of {ISOLATED_KERNELS}, got {kernel!r}"
        )
    nf, nloc = charge.shape
    n = nf * nloc
    out_dtype = coord.dtype
    if dtype is not None:
        coord, charge = coord.to(dtype), charge.to(dtype)
    if kernel == "dense" and n > max_dense_atoms:
        raise ValueError(
            f"{n} isolated atoms in one call exceed max_isolated_atoms="
            f"{max_dense_atoms} for the dense O(N^2) kernel; use isolated_kernel "
            "'chunked' (first derivatives only) or raise max_isolated_atoms"
        )
    if kernel == "dense" or (kernel == "auto" and n <= max_dense_atoms):
        return isolated_erf_energy_dense(coord, charge, alpha).to(out_dtype)
    batch = torch.repeat_interleave(
        torch.arange(nf, dtype=torch.long, device=coord.device), nloc
    )
    energy = isolated_erf_energy_chunked(
        coord.reshape(n, 3), charge.reshape(n), alpha, batch=batch
    )
    return energy.reshape(nf, nloc).to(out_dtype)
