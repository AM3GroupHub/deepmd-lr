# SPDX-License-Identifier: LGPL-3.0-or-later
"""The long-range (reciprocal-space) electrostatic term.

Holds both ways of evaluating it -- particle-mesh Ewald and the explicit
k-vector sum -- which is why the package is ``lr`` rather than ``pme``.
"""

from deepmd_lr.lr.term import (
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
