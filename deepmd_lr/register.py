# SPDX-License-Identifier: LGPL-3.0-or-later
"""Entry-point target: importing this module registers the plugin's types.

``deepmd/pt_expt/__init__.py`` calls ``load_entry_point("deepmd.pt_expt")``,
whose implementation is ``[ep.load() for ep in eps]``. Note that it *loads*
each entry point and never calls it -- so registration has to be a side
effect of importing this module, not the body of a function. The entry point
in ``pyproject.toml`` therefore names this module, not a callable inside it.

Both imports below register by decorator at their own import time, so this
module stays import-only. That also keeps it cheap, which matters: it runs
during every ``deepmd.pt_expt`` import, including ones that never touch this
plugin.

Ordering is safe. The model registry is read when a model is built, and
``argcheck``'s ``ArgsPlugin`` registry is read lazily by ``model_args()`` at
input-normalization time -- both strictly after the backend import that
brings us here.
"""

from deepmd_lr import (  # schema args; `same_as`, EMA, compression hooks
    argcheck,
    compress,
    ema_default,
    same_as,
)
from deepmd_lr.fitting import (  # registers the `charge` fitting head
    charge,
)
from deepmd_lr.model import (  # registers the `dp_lr` model type
    lr_model,
)

# Unlike the three imports above, this one is not registration-by-decorator:
# it wraps three DeePMD-kit symbols. It is called last because it imports
# `deepmd.pt_expt.train.training`, and we are running *inside*
# `deepmd.pt_expt.__init__`. Kept here rather than at `same_as` import time so
# that merely importing the module -- which `argcheck` does, for one constant
# -- has no side effect on the backend.
same_as.install()
# Apply EMA defaults before CLI normalization and for direct Trainer callers.
# Install after `same_as` so the Trainer wrappers compose.
ema_default.install()
# A compressed DPA4C must ask for the CSR views its kernel needs; upstream's
# `get_geo_compress` never does for DPA4C. One boolean, see `compress.py`.
compress.install()

__all__ = ["argcheck", "charge", "compress", "ema_default", "lr_model", "same_as"]
