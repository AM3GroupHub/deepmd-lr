# SPDX-License-Identifier: MIT
"""Let ``dp_lr`` take DeePMD-kit's ``bridging_method`` keys.

DeePMD-kit rewrites a model config that carries ``bridging_method`` into its
``linear_ener`` composition before it looks at ``model.type``, and refuses
any type but ``standard`` and ``dpa4``. A ``dp_lr`` model builds the same
composition itself (:meth:`LongRangeEnergyModel.get_model`), so its config
has to reach it unexpanded.
"""

import importlib
from typing import (
    Any,
)

__all__ = ["install"]

_MARK = "_deepmd_lr_passes_dp_lr"


def install() -> None:
    """Pass ``dp_lr`` configs through DeePMD-kit's bridging expansion
    untouched. Idempotent."""
    module = importlib.import_module("deepmd.pt_expt.model.get_model")
    upstream = module.expand_bridging_method
    if getattr(upstream, _MARK, False):
        return

    def expand_bridging_method(data: dict[str, Any]) -> dict[str, Any]:
        if str(data.get("type", "standard")) == "dp_lr":
            return data
        return upstream(data)

    setattr(expand_bridging_method, _MARK, True)
    module.expand_bridging_method = expand_bridging_method
