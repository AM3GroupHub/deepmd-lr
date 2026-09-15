# SPDX-License-Identifier: LGPL-3.0-or-later
"""Any module of ours must be importable first, in a fresh interpreter.

DeePMD-kit fires our entry point from inside ``deepmd.pt_expt.__init__``, so
whichever of our modules first touches that backend gets re-entered by the
registration it triggers. Before ``deepmd_lr/__init__`` imported the backend
itself, ``import deepmd_lr.nvalchemi`` failed with

    ImportError: cannot import name 'CHARGE_VAR_NAME' from partially
    initialized module 'deepmd_lr.fitting.charge'

which is exactly the snippet the README tells people to write. Import order
can only be tested in a fresh process, hence the subprocess.
"""

import subprocess
import sys

import pytest

FIRST_IMPORTS = [
    "deepmd_lr",
    "deepmd_lr.nvalchemi",
    "deepmd_lr.argcheck",
    "deepmd_lr.neighbors",
    "deepmd_lr.fitting.charge",
    "deepmd_lr.model.lr_model",
    "deepmd_lr.lr.term",
    "deepmd_lr.register",
]


@pytest.mark.parametrize("module", FIRST_IMPORTS)
def test_module_imports_first_in_a_clean_interpreter(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"`import {module}` failed as the first import:\n"
        f"{result.stderr[-1500:]}"
    )


def test_the_documented_snippet_works_standalone():
    """The first line of the README's partitioned-MD snippet, on its own."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from deepmd_lr.nvalchemi import DistributedEvaluator; "
            "print(DistributedEvaluator.__name__)",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "DistributedEvaluator" in result.stdout
