# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compressed DPA4C inference for the ``dp_lr`` model.

What DeePMD-kit's compression is, for DPA4C
--------------------------------------------
``descriptor.enable_compression`` tabulates the radial branch into fp32
quintic-Hermite tables, freezes the pair, coupling and readout tensors as
thirteen ``compress_*`` buffers, and from then on ``DescrptDPA4C.call_graph``
dispatches the descriptor to a fused CUDA kernel,
``deepmd::dpa4c_graph_compress``. The charge head and the reciprocal-space
term are untouched -- they are siblings of the descriptor on the model -- so
compressing a ``dp_lr`` model is compressing exactly the part that costs the
time: at 6,520 atoms the descriptor is 883 ms of a step, the reciprocal sum
73 ms.

Why the CLI route does not apply
--------------------------------
``dp compress`` reads a frozen ``.pte``/``.pt2``, and ``dp freeze`` is refused
for a model with a ``long_range`` block because the exported lower never sees
the cell. So compression happens in process, on the loaded eager model,
through upstream's own ``model.enable_compression`` -- which
:class:`~deepmd_lr.model.lr_model.LongRangeEnergyModel` overrides to add the
checks below.

Why every check is loud
-----------------------
The compressed branch inside ``call_graph`` fires only when *all* of these
hold: the descriptor is compressed, eval mode, ``DP_CUDA_INFER >= 1``, the
graph carries the destination CSR views, and the compiled operator is loaded.
When any one fails it **falls through to the uncompressed math without a
word**. A model that reports itself compressed and runs at the old speed is
the failure this module exists to prevent, so each precondition is refused
where it can be seen rather than left to be discovered from a benchmark.

The one seam this module owns
-----------------------------
:func:`install` patches ``DescrptDPA4C.get_geo_compress`` to report the
descriptor's own ``compress`` flag. That method is what upstream's eager
``forward_common_graph`` (``pt_expt/model/make_model.py:945-957``) consults to
decide whether to attach the CSR views -- and DPA4C never overrides it, so a
compressed DPA4C reached through ``model(coord, atype, box)`` would ask for no
CSR and silently take the slow branch. DPA1 and DPA2 do override it; this is
the same one-boolean answer for DPA4C. Its only other reader is a DPA1-only
canonical-export check, so the blast radius is that boolean.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import (
    Any,
)

__all__ = ["compress_model", "install"]


def install() -> None:
    """Let a compressed DPA4C ask for the CSR views its kernel needs. Idempotent."""
    from deepmd.pt_expt.descriptor.dpa4c import (
        DescrptDPA4C,
    )

    if getattr(DescrptDPA4C.get_geo_compress, "_deepmd_lr_patched", False):
        return

    def get_geo_compress(self: Any) -> bool:
        """Whether the compressed kernel is in use, so the graph needs CSR views.

        Upstream's default is a constant ``False``; for DPA4C that leaves the
        compressed branch of ``call_graph`` unreachable from the eager
        forward, because the views it requires are only built when this
        returns ``True``. Installed by ``deepmd_lr``.
        """
        return bool(self.compress)

    get_geo_compress._deepmd_lr_patched = True  # type: ignore[attr-defined]
    DescrptDPA4C.get_geo_compress = get_geo_compress


def compress_model(model: Any, compress: bool | float, charge_spin: Any = None) -> Any:
    """Compress ``model`` in place as the calculator and evaluator request it.

    ``compress`` is ``True`` for DeePMD-kit's default table stride (0.01 A,
    the ``dp compress`` default) or a float for an explicit stride. The checks
    live on :meth:`LongRangeEnergyModel.enable_compression`; this only maps
    the keyword onto that call, and refuses a model that does not carry those
    checks -- a plain ``EnergyModel`` would compress through upstream's
    method with nothing standing between it and the silent fall-through.
    """
    from deepmd_lr.model.lr_model import (
        LongRangeEnergyModel,
    )

    if compress is False:
        return model
    if compress is not True:
        if isinstance(compress, bool) or not isinstance(compress, Real):
            raise TypeError(
                "compress must be a bool or a positive finite numeric stride"
            )
        if not math.isfinite(compress) or compress <= 0:
            raise ValueError("compression stride must be positive and finite")
    if not isinstance(model, LongRangeEnergyModel):
        raise TypeError(
            f"compress= needs a 'dp_lr' model, got {type(model).__name__}. A "
            "plain energy model can be compressed with its own "
            "`enable_compression()`, but nothing then checks that the "
            "compressed kernel is actually reached; this package only vouches "
            "for models it can guard."
        )
    stride = {} if compress is True else {"table_stride_1": float(compress)}
    model.enable_compression(**stride, charge_spin=charge_spin)
    return model
