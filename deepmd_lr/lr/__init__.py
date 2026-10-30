# SPDX-License-Identifier: MIT
"""The long-range electrostatic term.

Three ways of evaluating the same quantity -- particle-mesh Ewald, an explicit
k-vector sum, and a direct real-space sum for systems with no periodic axis
(:mod:`deepmd_lr.lr.isolated`) -- which is why the package is ``lr`` rather
than ``pme``. Build one through :func:`build_long_range_energy`, or for a given
system through :func:`system_reciprocal`, whose ``pbc`` picks the sum.
"""

from deepmd_lr.lr.isolated import isolated_erf_energy_chunked
from deepmd_lr.lr.term import (
    DEFAULT_ALPHA,
    DEFAULT_SPLINE_ORDER,
    MAX_SPLINE_ORDER,
    TERM_CLASSES,
    IsolatedLongRangeEnergy,
    LongRangeEnergy,
    LongRangeTerm,
    build_long_range_energy,
    isolated_term_for,
    system_mesh,
    system_reciprocal,
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
    "isolated_erf_energy_chunked",
    "isolated_term_for",
    "system_mesh",
    "system_reciprocal",
]
