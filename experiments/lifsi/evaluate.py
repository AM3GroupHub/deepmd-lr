# SPDX-License-Identifier: LGPL-3.0-or-later
"""Held-out evaluation of the trained models on every validation frame."""

import argparse
import pathlib

import numpy as np
import torch
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.ase import load_model

DT = torch.float64


def evaluate(checkpoint, data):
    root = pathlib.Path(data)
    coord = np.load(root / "set.000/coord.npy")
    box = np.load(root / "set.000/box.npy")
    energy = np.load(root / "set.000/energy.npy")
    force = np.load(root / "set.000/force.npy")
    atype = np.loadtxt(root / "type.raw", dtype=int)
    n = len(atype)

    model = load_model(checkpoint)
    at = torch.tensor(atype, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    de, df, elr = [], [], []
    for i in range(len(coord)):
        c = torch.tensor(coord[i].reshape(1, n, 3), dtype=DT, device=DEVICE)
        b = torch.tensor(box[i].reshape(1, 3, 3), dtype=DT, device=DEVICE)
        out = model(c, at, box=b)
        de.append((float(out["energy"].reshape(-1)[0]) - energy[i]) / n)
        df.append(out["force"][0].detach().cpu().numpy() - force[i].reshape(n, 3))
        if "energy_long_range" in out:
            elr.append(float(out["energy_long_range"].reshape(-1)[0]))
    de = np.array(de)
    df = np.concatenate(df)
    result = {
        "energy_rmse_meV_per_atom": float(np.sqrt((de**2).mean()) * 1000),
        "force_rmse": float(np.sqrt((df**2).mean())),
        "force_mae": float(np.abs(df).mean()),
    }
    if elr:
        result["long_range_energy_mean"] = float(np.mean(elr))
        result["long_range_energy_std"] = float(np.std(elr))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/valid")
    ap.add_argument("--models", nargs="+",
                    default=["baseline=run_baseline/model.ckpt.pt",
                             "joint=run_joint/model.ckpt.pt"])
    args = ap.parse_args()

    rows = {}
    for spec in args.models:
        tag, _, path = spec.partition("=")
        rows[tag] = evaluate(path, args.data)

    n_frames = len(np.load(pathlib.Path(args.data) / "set.000/energy.npy"))
    print(f"Held-out evaluation on {n_frames} frames\n")
    print(f"{'model':<10}{'E RMSE (meV/atom)':>20}{'F RMSE (eV/A)':>16}{'F MAE':>10}")
    for tag, r in rows.items():
        print(f"{tag:<10}{r['energy_rmse_meV_per_atom']:>20.3f}"
              f"{r['force_rmse']:>16.4f}{r['force_mae']:>10.4f}")
    if "baseline" in rows and "joint" in rows:
        b, j = rows["baseline"], rows["joint"]
        print(f"\nimprovement: "
              f"{b['energy_rmse_meV_per_atom'] / j['energy_rmse_meV_per_atom']:.2f}x energy, "
              f"{b['force_rmse'] / j['force_rmse']:.2f}x force")
    if "long_range_energy_mean" in rows.get("joint", {}):
        j = rows["joint"]
        print(f"\nlong-range energy: {j['long_range_energy_mean']:+.2f} "
              f"+- {j['long_range_energy_std']:.2f} eV")


if __name__ == "__main__":
    main()
