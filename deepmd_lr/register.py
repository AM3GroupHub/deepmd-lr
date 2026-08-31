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

from deepmd_lr import (  # noqa: F401  registers `dp_lr` input-schema args
    argcheck,
)
from deepmd_lr.model import (  # noqa: F401  registers the `dp_lr` model type
    lr_model,
)

__all__ = ["argcheck", "lr_model"]
