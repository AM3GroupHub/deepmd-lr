# SPDX-License-Identifier: MIT
"""Small cell helpers: 3 x 3 algebra on the device, and pbc flags.

``torch.linalg.inv`` and ``det`` check their ``info`` result, which is a read
of the device; a barostat replaces the cell every step, and the one place
that must not stall the host is exactly there. For a 3 x 3 matrix the
cofactor formulas are a handful of elementwise ops and have no such check.
"""

import numpy as np
import torch

__all__ = ["det_3x3", "inverse_3x3", "pbc_flags"]


def pbc_flags(pbc) -> tuple[bool, bool, bool]:
    """Per-axis periodicity as three Python bools; a scalar applies to all."""
    return tuple(bool(p) for p in np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)))


def _entries(h: torch.Tensor):
    h = h.reshape(3, 3)
    return (
        h[0, 0], h[0, 1], h[0, 2],
        h[1, 0], h[1, 1], h[1, 2],
        h[2, 0], h[2, 1], h[2, 2],
    )  # fmt: skip


def det_3x3(h: torch.Tensor) -> torch.Tensor:
    """Determinant of a ``(3, 3)`` matrix, as a 0-d tensor."""
    a, b, c, d, e, f, g, i, k = _entries(h)
    return a * (e * k - f * i) - b * (d * k - f * g) + c * (d * i - e * g)


def inverse_3x3(h: torch.Tensor) -> torch.Tensor:
    """Inverse of a ``(3, 3)`` matrix: the adjugate over the determinant."""
    a, b, c, d, e, f, g, i, k = _entries(h)
    adjugate = torch.stack(
        [
            torch.stack([e * k - f * i, c * i - b * k, b * f - c * e]),
            torch.stack([f * g - d * k, a * k - c * g, c * d - a * f]),
            torch.stack([d * i - e * g, b * g - a * i, a * e - b * d]),
        ]
    )
    return adjugate / det_3x3(h)
