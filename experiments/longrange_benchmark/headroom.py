# SPDX-License-Identifier: LGPL-3.0-or-later
"""How much of the label is invisible to a strictly local model?

The long-range term can only recover signal that lives beyond the descriptor
cutoff. This measures that directly: exact Ewald forces minus Coulomb forces
truncated at the cutoff. The result is the error floor no strictly local
model can go below, so it bounds what the long-range term can possibly buy.
"""

import argparse

import numpy as np
import torch
from deepmd.dpmodel.modifier.dipole_charge import ELECTROSTATIC_CONVERSION as KE
from generate_data import coulomb

DT = torch.float64


def truncated_coulomb_force(coord, charge, cell, cutoff, device):
    """Coulomb force from pairs within ``cutoff`` only, minimum image."""
    n = coord.shape[0]
    coord = coord.clone().requires_grad_(True)
    delta = coord[:, None, :] - coord[None, :, :]
    frac = delta @ torch.linalg.inv(cell)
    delta = (frac - torch.round(frac)) @ cell
    r = torch.linalg.norm(delta, dim=-1)
    real = (r < cutoff) & (~torch.eye(n, dtype=torch.bool, device=device))
    safe_r = torch.where(real, r, torch.ones_like(r))
    pair = charge[:, None] * charge[None, :] * KE / safe_r
    energy = 0.5 * (pair * real).sum()
    return -torch.autograd.grad(energy, coord)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/valid")
    ap.add_argument("--rcut", type=float, default=6.0, help="descriptor cutoff")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coords = np.load(f"{args.data}/set.000/coord.npy")
    box = np.load(f"{args.data}/set.000/box.npy")[0].reshape(3, 3)
    atype = np.loadtxt(f"{args.data}/type.raw", dtype=int)
    n_atoms = len(atype)

    charge = torch.tensor(
        np.where(atype == 0, 1.0, -1.0), dtype=DT, device=device
    )
    cell = torch.tensor(box, dtype=DT, device=device)
    label_force = np.load(f"{args.data}/set.000/force.npy")

    beyond = []
    for index in range(min(args.frames, coords.shape[0])):
        coord = torch.tensor(
            coords[index].reshape(n_atoms, 3), dtype=DT, device=device
        )
        differentiable = coord.clone().requires_grad_(True)
        exact = -torch.autograd.grad(
            coulomb(differentiable, charge, cell, device), differentiable
        )[0]
        truncated = truncated_coulomb_force(
            coord, charge, cell, args.rcut, device
        )
        beyond.append(float(((exact - truncated) ** 2).mean().sqrt()))

    total = float(np.sqrt((label_force**2).mean()))
    missing = float(np.mean(beyond))
    print(f"total label force RMS            : {total:.4f} eV/A")
    print(f"Coulomb beyond {args.rcut:.1f} A, force RMS : {missing:.4f} eV/A")
    fraction = missing / total * 100
    print(f"  -> {fraction:.1f}% of the force signal is outside the cutoff")
    print("  -> the floor a strictly local model cannot go below")


if __name__ == "__main__":
    main()
