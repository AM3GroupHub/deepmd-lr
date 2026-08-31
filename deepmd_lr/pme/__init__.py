# SPDX-License-Identifier: LGPL-3.0-or-later
"""The long-range (reciprocal-space) electrostatic term."""

from deepmd_lr.pme.term import (
    DEFAULT_ALPHA,
    DEFAULT_SPLINE_ORDER,
    MAX_SPLINE_ORDER,
    LongRangeEnergy,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_SPLINE_ORDER",
    "MAX_SPLINE_ORDER",
    "LongRangeEnergy",
]
