# SPDX-License-Identifier: MIT
"""Fitting heads contributed by this plugin."""

from deepmd_lr.fitting.charge import (
    CHARGE_VAR_NAME,
    ChargeFittingNet,
    ChargeFittingNetDP,
    project_to_total_charge,
)

__all__ = [
    "CHARGE_VAR_NAME",
    "ChargeFittingNet",
    "ChargeFittingNetDP",
    "project_to_total_charge",
]
