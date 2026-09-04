# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inspect the latent charges and the long-range term on a single frame.

For an electrode/electrolyte interface the interesting object is the *profile*
along the slab normal: a metal in contact with an electrolyte should show
charge separation near the interface. This prints that profile alongside the
energy split, and keeps the metal and electrolyte regions separate because a
point charge means different things in each (see TRAINING_LiFSI.md).
"""

import argparse
import pathlib

import numpy as np
import torch
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.ase import load_model

DT = torch.float64
METAL_Z = 30.0  # below this is the Li electrode; above it, electrolyte


def bar(value, scale, width=28):
    n = int(round(abs(value) / scale * width)) if scale else 0
    return ("+" if value >= 0 else "-") * min(n, width)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="run_joint/model.ckpt.pt")
    ap.add_argument("--data", default="data/valid")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    root = pathlib.Path(args.data)
    coord = np.load(root / "set.000/coord.npy")[args.frame]
    box = np.load(root / "set.000/box.npy")[args.frame].reshape(3, 3)
    energy_ref = np.load(root / "set.000/energy.npy")[args.frame]
    force_ref = np.load(root / "set.000/force.npy")[args.frame]
    atype = np.loadtxt(root / "type.raw", dtype=int)
    type_map = (root / "type_map.raw").read_text().split()
    n = len(atype)
    symbols = np.array([type_map[t] for t in atype])

    model = load_model(args.model)
    c = torch.tensor(coord.reshape(1, n, 3), dtype=DT, device=DEVICE)
    b = torch.tensor(box.reshape(1, 3, 3), dtype=DT, device=DEVICE)
    a = torch.tensor(atype, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    out = model(c, a, box=b)

    total = float(out["energy"].reshape(-1)[0])
    short = float(out["energy_short_range"].reshape(-1)[0])
    long_range = float(out["energy_long_range"].reshape(-1)[0])
    force = out["force"][0].detach().cpu().numpy()
    charge = out["charge"].reshape(-1).detach().cpu().numpy()
    z = coord.reshape(n, 3)[:, 2]

    print(f"frame {args.frame} of {args.data} -- {n} atoms\n")
    print("ENERGY")
    print(f"  reference (UMA)      {energy_ref:14.4f} eV")
    print(f"  model total          {total:14.4f} eV   "
          f"(error {total - energy_ref:+.4f})")
    print(f"    short-range        {short:14.4f} eV")
    print(f"    long-range         {long_range:14.4f} eV   "
          f"({abs(long_range) / abs(total) * 100:.3f}% of the total)")
    force_rmse = np.sqrt(((force - force_ref.reshape(n, 3)) ** 2).mean())
    print(f"  force RMSE           {force_rmse:14.4f} eV/A")

    print("\nLATENT CHARGE by element (e)")
    print(f"  {'elem':<5}{'n':>7}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
    for element in type_map:
        m = symbols == element
        q = charge[m]
        print(f"  {element:<5}{m.sum():>7}{q.mean():>10.4f}{q.std():>9.4f}"
              f"{q.min():>9.4f}{q.max():>9.4f}")
    print(f"  {'TOTAL':<5}{n:>7}{charge.sum():>10.2e}"
          "  <- constrained to the frame charge")

    print(f"\n  by region (metal = z < {METAL_Z:.0f} A)")
    for label, m in (("Li metal", (symbols == "Li") & (z < METAL_Z)),
                     ("electrolyte Li", (symbols == "Li") & (z >= METAL_Z)),
                     ("electrolyte, all", z >= METAL_Z)):
        if m.sum():
            print(f"    {label:<18}n={m.sum():<6} mean {charge[m].mean():+.4f}"
                  f"   net {charge[m].sum():+.3f} e")

    print("\nCHARGE PROFILE along the slab normal (5 A bins)")
    edges = np.arange(0, z.max() + 5, 5.0)
    idx = np.digitize(z, edges) - 1
    net = np.array([charge[idx == i].sum() for i in range(len(edges) - 1)])
    counts = np.array([(idx == i).sum() for i in range(len(edges) - 1)])
    scale = np.abs(net).max()
    for i, lo in enumerate(edges[:-1]):
        if counts[i] == 0:
            continue
        tag = "metal" if lo < METAL_Z else ""
        print(f"  z {lo:5.0f}-{lo+5:3.0f}  n={counts[i]:5d}  "
              f"net {net[i]:+8.3f} e  {bar(net[i], scale):<28} {tag}")

    print("\nLONG-RANGE ENERGY PROFILE (per-atom, summed in 5 A bins)")
    per_atom = model.pme(
        c, out["charge"], b,
        mask=model.atomic_model.make_atom_mask(a),
    ).reshape(-1).detach().cpu().numpy()
    binned = np.array([per_atom[idx == i].sum() for i in range(len(edges) - 1)])
    scale = np.abs(binned).max()
    for i, lo in enumerate(edges[:-1]):
        if counts[i] == 0:
            continue
        print(f"  z {lo:5.0f}-{lo+5:3.0f}  {binned[i]:+10.3f} eV  "
              f"{bar(binned[i], scale)}")


if __name__ == "__main__":
    main()
