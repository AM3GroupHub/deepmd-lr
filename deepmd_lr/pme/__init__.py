# SPDX-License-Identifier: LGPL-3.0-or-later
"""The PME long-range electrostatic term."""

from deepmd_lr.pme.term import (
    DEFAULT_SPLINE_ORDER,
    MAX_SPLINE_ORDER,
    PMEEnergy,
)

__all__ = ["DEFAULT_SPLINE_ORDER", "MAX_SPLINE_ORDER", "PMEEnergy"]
