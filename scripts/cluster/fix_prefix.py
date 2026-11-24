# SPDX-License-Identifier: MIT
"""Rewrite a conda env's old prefix after `mv`, the way conda does: text
files substituted, binaries null-padded.

    python scripts/cluster/fix_prefix.py /old/prefix /new/prefix

The new prefix must not be longer than the old one. Used to rename the
`dpmd-cuda` env built by `build_deepmd_cuda_cluster.sh` without rebuilding.
"""

import os
import sys

from conda.core.portability import update_prefix
from conda.models.enums import FileMode

old, new = sys.argv[1], sys.argv[2]
assert len(new) <= len(old)
ob = old.encode()
n_text = n_bin = 0
for root, dirs, files in os.walk(new):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        p = os.path.join(root, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        with open(p, "rb") as fh:
            data = fh.read()
        if ob not in data:
            continue
        binary = b"\x00" in data[:8192]
        update_prefix(
            p, new, placeholder=old, mode=FileMode.binary if binary else FileMode.text
        )
        n_bin += binary
        n_text += not binary
print(f"rewrote {n_text} text and {n_bin} binary files")
