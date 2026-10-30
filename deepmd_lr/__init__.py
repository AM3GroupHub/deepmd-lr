# SPDX-License-Identifier: MIT
"""Long-range electrostatics for DPA4C, as a DeePMD-kit pt_expt plugin.

Registration itself lives in :mod:`deepmd_lr.register`, which DeePMD-kit's
``deepmd.pt_expt`` entry point calls when that backend is imported.

Importing ``deepmd.pt_expt`` here looks redundant -- the entry point would
fire on its own the moment any submodule touched the backend -- but the
order matters. Without it, ``import deepmd_lr.nvalchemi`` as the first
DeePMD-kit import goes:

    deepmd_lr.nvalchemi -> driver -> fitting.charge
      -> deepmd.pt_expt (mid-module)
        -> load_entry_point -> deepmd_lr.register -> argcheck
          -> fitting.charge   <- still executing, CHARGE_VAR_NAME unbound

and fails with ``ImportError: cannot import name 'CHARGE_VAR_NAME' from
partially initialized module``. The names argcheck needs are used in
module-level decorators, so they cannot be deferred into functions.

Pulling the backend in from the package root makes the entry point run to
completion before any submodule of ours starts, so nothing is ever
re-entered half-built. The cost is that importing this package imports the
backend, which any real use does anyway.
"""

import deepmd.pt_expt  # noqa: F401  see the module docstring

__version__ = "0.1.0"
