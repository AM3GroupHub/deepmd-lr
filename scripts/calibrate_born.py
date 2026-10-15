#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fit the scale and sign of a ``dp_lr`` model's Born charges to DFPT labels.

    python scripts/calibrate_born.py model.pt valid_BEC.xyz [test_BEC.xyz]

Training never supervises the charges, so the model's Born tensors are fixed
only up to a scale, ``sqrt(eps_inf)``, and a sign, the charge branch. One
least-squares fit over every frame, atom and component of the first file
gives both::

    s = sum(Z_raw * Z_dft) / sum(Z_raw * Z_raw)

The second file, if given, is scored at that ``s``. Each atom carries its DFPT
tensor as nine numbers under ``--key`` (default ``BEC``), laid out
``[polarization, displacement]`` like ``results["born_charge"]``. Born charges
are never trained on; this calibration is the whole treatment
(``docs/born-charges.md``).
"""

import argparse

import numpy as np
from ase.io import read


def born_tensors(calc, frames, key="BEC"):
    """Raw model tensors and the labels, each ``(nframes, natoms, 3, 3)``."""
    raw, ref = [], []
    for atoms in frames:
        a = atoms.copy()
        a.calc = calc
        a.get_forces()
        raw.append(np.asarray(calc.results["born_charge"]).reshape(len(a), 3, 3))
        ref.append(np.asarray(atoms.arrays[key], float).reshape(len(a), 3, 3))
    return np.array(raw), np.array(ref)


def fit_scale(z_raw, z_ref):
    """Least-squares ``s`` with ``z_ref ~ s * z_raw``; ``|s|`` is sqrt(eps_inf)."""
    return float((z_raw * z_ref).sum() / (z_raw * z_raw).sum())


def errors(z, z_ref):
    """MAE and RMSE over every component, in e."""
    d = z - z_ref
    return float(np.abs(d).mean()), float(np.sqrt((d**2).mean()))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model", help="dp_lr checkpoint")
    ap.add_argument("fit", help="labelled frames the scale is fitted on")
    ap.add_argument("test", nargs="?", help="held-out frames, scored at that scale")
    ap.add_argument("--key", default="BEC", help="per-atom array of the 9 labels")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from deepmd_lr.ase import LongRangeCalculator

    calc = LongRangeCalculator(a.model, device=a.device, efield=[0.0, 0.0, 0.0])
    sets = {a.fit: born_tensors(calc, read(a.fit, index=":"), a.key)}
    s = fit_scale(*sets[a.fit])
    sign = "+" if s > 0 else "-"
    print(f"s = {s:.4f}: sqrt(eps_inf) {abs(s):.4f}, eps_inf {s * s:.3f}, sign {sign}")
    if a.test:
        sets[a.test] = born_tensors(calc, read(a.test, index=":"), a.key)
    for path, (z_raw, z_ref) in sets.items():
        mae, rmse = errors(s * z_raw, z_ref)
        print(f"{path}: {len(z_raw)} frames, MAE {mae:.4f} e, RMSE {rmse:.4f} e")


if __name__ == "__main__":
    main()
