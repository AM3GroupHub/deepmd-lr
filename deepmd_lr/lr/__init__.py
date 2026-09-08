# SPDX-License-Identifier: LGPL-3.0-or-later
"""The long-range electrostatic term.

Three ways of evaluating the same quantity -- particle-mesh Ewald, an explicit
k-vector sum, and a direct real-space sum for systems with no cell -- which is
why the package is ``lr`` rather than ``pme``. Pick one through
:func:`build_long_range_energy` rather than naming a class, so a config and a
checkpoint resolve the same way.
"""

from deepmd_lr.lr.term import (
    DEFAULT_ALPHA,
    DEFAULT_SPLINE_ORDER,
    MAX_SPLINE_ORDER,
    TERM_CLASSES,
    IsolatedLongRangeEnergy,
    LongRangeEnergy,
    LongRangeTerm,
    build_long_range_energy,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_SPLINE_ORDER",
    "MAX_SPLINE_ORDER",
    "TERM_CLASSES",
    "IsolatedLongRangeEnergy",
    "LongRangeEnergy",
    "LongRangeTerm",
    "build_long_range_energy",
]
