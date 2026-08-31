# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare the baseline and joint models on the held-out set.

Reports the accuracy gap, and -- the sharper test -- whether the charge head
recovered the true +-1 charges from energy and force labels alone. It was
never shown a charge.
"""

import json
import pathlib

import numpy as np
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

DT = torch.float64
TRUE_CHARGE = {0: 1.0, 1: -1.0}  # type 0 = Na (+1), type 1 = Cl (-1)


def load(run_dir):
    ckpt = torch.load(
        pathlib.Path(run_dir) / "model.ckpt.pt", map_location="cpu",
        weights_only=False,
    )
    params = ckpt["model"]["_extra_state"]["model_params"]
    model = get_model(params)
    state = {
        k[len("model.Default."):]: v
        for k, v in ckpt["model"].items()
        if k.startswith("model.Default.")
    }
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


def evaluate(model, data_dir, batch=10):
    d = pathlib.Path(data_dir)
    coord = np.load(d / "set.000/coord.npy")
    force = np.load(d / "set.000/force.npy")
    energy = np.load(d / "set.000/energy.npy")
    box = np.load(d / "set.000/box.npy")
    atype = np.loadtxt(d / "type.raw", dtype=int)
    n_frames, n_atoms = coord.shape[0], len(atype)

    de, df, charges = [], [], []
    at = torch.tensor(atype, dtype=torch.int64, device=DEVICE)
    for start in range(0, n_frames, batch):
        sl = slice(start, start + batch)
        nb = coord[sl].shape[0]
        c = torch.tensor(coord[sl].reshape(nb, n_atoms, 3), dtype=DT, device=DEVICE)
        b = torch.tensor(box[sl].reshape(nb, 3, 3), dtype=DT, device=DEVICE)
        out = model(c, at.expand(nb, -1), box=b)
        de.append(
            (out["energy"].reshape(-1).detach().cpu().numpy() - energy[sl]) / n_atoms
        )
        df.append(
            out["force"].detach().cpu().numpy().reshape(nb, -1) - force[sl]
        )
        if "charge" in out:
            charges.append(out["charge"].detach().cpu().numpy())

    de = np.concatenate(de)
    df = np.concatenate(df)
    result = {
        "energy_rmse_per_atom": float(np.sqrt((de**2).mean())),
        "force_rmse": float(np.sqrt((df**2).mean())),
    }
    if charges:
        result["charges"] = np.concatenate(charges)
        result["atype"] = atype
    return result


def main():
    rows = {}
    for tag, run in (("baseline", "run_baseline"), ("joint", "run_joint")):
        rows[tag] = evaluate(load(run), "data/valid")

    print("Held-out validation (200 frames, 32 ions, 14 A cell)\n")
    print(f"{'model':<10} {'E RMSE (meV/atom)':>19} {'F RMSE (eV/A)':>15}")
    for tag, r in rows.items():
        print(
            f"{tag:<10} {r['energy_rmse_per_atom'] * 1000:>19.2f} "
            f"{r['force_rmse']:>15.4f}"
        )
    b, j = rows["baseline"], rows["joint"]
    print(
        f"\nimprovement: {b['energy_rmse_per_atom'] / j['energy_rmse_per_atom']:.1f}x "
        f"on energy, {b['force_rmse'] / j['force_rmse']:.1f}x on force"
    )

    if "charges" in rows["joint"]:
        q = rows["joint"]["charges"]
        atype = rows["joint"]["atype"]
        print("\nLearned charges (never supplied as a label):")
        print(f"{'type':<8} {'true':>7} {'learned mean':>14} {'std':>9}")
        for t, name in ((0, "Na"), (1, "Cl")):
            sel = q[:, atype == t]
            print(
                f"{name:<8} {TRUE_CHARGE[t]:>7.2f} {sel.mean():>14.4f} "
                f"{sel.std():>9.4f}"
            )
        json.dump(
            {
                k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
                for k, v in rows.items()
            },
            open("results.json", "w"),
            indent=2,
        )


if __name__ == "__main__":
    main()
