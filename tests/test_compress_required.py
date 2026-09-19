"""Strict compression qualification cannot succeed without prerequisites."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "cuda,operator", [(False, True), (True, False), (False, False)]
)
def test_required_mode_fails_collection_without_prerequisites(cuda, operator):
    # Import the backend before simulating availability. No kernels are executed.
    code = f"""
import deepmd.pt_expt
import torch
import deepmd.kernels.cuda.dpa4c.graph_compress as gc
import pytest
torch.cuda.is_available = lambda: {cuda!r}
gc.op_available = lambda: {operator!r}
raise SystemExit(pytest.main([
    '-p', 'no:cacheprovider', '--collect-only', '-q', 'tests/test_compress.py'
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "DEEPMD_LR_REQUIRE_COMPRESS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "requires a usable CUDA device" in result.stdout + result.stderr
