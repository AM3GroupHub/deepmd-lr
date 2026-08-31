# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate a periodic molten-salt-like dataset where long-range matters.

Why synthetic
-------------
Answering "does the long-range term buy anything?" needs a system whose
energy provably contains a contribution a strictly local model cannot
represent. No periodic dataset was available on this machine (everything in
`/home/admin/storage/dataset` is non-periodic molecular clusters), and a real
DFT set would confound the question with everything else DFT contains.

Here the ground truth is known exactly:

    E = E_repulsion(r < 5 A) + E_Coulomb(q = +-1, full Ewald)

The repulsive part is strictly inside the 6 A descriptor cutoff, so a plain
DPA4C model can represent it exactly. The Coulomb part is not: it decays as
1/r and carries real weight far beyond any cutoff. The gap between the two
models is therefore attributable to the long-range term and nothing else.

The Coulomb reference uses nvalchemi's *exact* `ewald_summation`, not the PME
path the model uses, so the label and the model do not share an
implementation.
"""

import argparse
import pathlib

import numpy as np
import torch
from deepmd.dpmodel.modifier.dipole_charge import ELECTROSTATIC_CONVERSION as KE
from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_summation
from nvalchemiops.torch.neighbors import neighbor_list

DT = torch.float64

# Born-Mayer-like repulsion, comfortably inside the descriptor cutoff.
REPULSION_A = 3000.0  # eV
REPULSION_RHO = 0.30  # A
REPULSION_CUTOFF = 5.0  # A
MIN_SEPARATION = 2.2  # A, rejection threshold when sampling configurations


def repulsion(coord, cell, device):
    """Pairwise Born-Mayer repulsion, shifted to zero at the cutoff.

    Shifted rather than raw so the potential is continuous at the cutoff;
    a discontinuity there would put a feature in the labels that no smooth
    model can fit, muddying the comparison.
    """
    n = coord.shape[0]
    inv = torch.linalg.inv(cell)
    delta = coord[:, None, :] - coord[None, :, :]
    frac = delta @ inv
    delta = (frac - torch.round(frac)) @ cell  # minimum image
    r = torch.linalg.norm(delta, dim=-1)
    mask = (r < REPULSION_CUTOFF) & (~torch.eye(n, dtype=torch.bool, device=device))
    safe_r = torch.where(mask, r, torch.ones_like(r))
    shift = REPULSION_A * np.exp(-REPULSION_CUTOFF / REPULSION_RHO)
    pair = REPULSION_A * torch.exp(-safe_r / REPULSION_RHO) - shift
    return 0.5 * (pair * mask).sum()


def coulomb(coord, charge, cell, device):
    """Exact Ewald electrostatic energy in eV."""
    pbc = torch.ones(3, dtype=torch.bool, device=device)
    batch = torch.zeros(coord.shape[0], dtype=torch.int32, device=device)
    rc = 0.45 * float(torch.diagonal(cell).min())
    nlist, nptr, shifts = neighbor_list(
        coord.detach(), rc, cell=cell, pbc=pbc, batch_idx=batch,
        return_neighbor_list=True, max_neighbors=2048, method="cell_list",
    )
    energy = ewald_summation(
        coord, charge, cell,
        neighbor_list=nlist, neighbor_ptr=nptr, neighbor_shifts=shifts,
        batch_idx=batch, energy_reduction="system", accuracy=1e-10,
    )
    return energy.sum() * KE


def sample_configuration(rng, n_ion_pairs, box, device):
    """Random ion positions with a minimum-separation rejection rule."""
    n = 2 * n_ion_pairs
    while True:
        pos = rng.uniform(0.0, box, size=(n, 3))
        delta = pos[:, None, :] - pos[None, :, :]
        delta -= box * np.round(delta / box)
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, np.inf)
        if dist.min() > MIN_SEPARATION:
            return torch.tensor(pos, dtype=DT, device=device)


def build(n_frames, n_ion_pairs, box, seed, device, include_coulomb=True):
    rng = np.random.default_rng(seed)
    n = 2 * n_ion_pairs
    # Type 0 = cation (+1), type 1 = anion (-1). The model is never told this.
    atype = np.array([0] * n_ion_pairs + [1] * n_ion_pairs, dtype=int)
    charge = torch.tensor(
        [1.0] * n_ion_pairs + [-1.0] * n_ion_pairs, dtype=DT, device=device
    )
    cell = torch.eye(3, dtype=DT, device=device) * box

    coords, energies, forces = [], [], []
    for _ in range(n_frames):
        pos = sample_configuration(rng, n_ion_pairs, box, device)
        pos.requires_grad_(True)
        energy = repulsion(pos, cell, device)
        if include_coulomb:
            energy = energy + coulomb(pos, charge, cell, device)
        force = -torch.autograd.grad(energy, pos)[0]
        coords.append(pos.detach().cpu().numpy().reshape(-1))
        energies.append(float(energy))
        forces.append(force.cpu().numpy().reshape(-1))

    return {
        "atype": atype,
        "coord": np.array(coords),
        "energy": np.array(energies),
        "force": np.array(forces),
        "box": np.tile((np.eye(3) * box).reshape(-1), (n_frames, 1)),
    }


def write(out_dir, data, type_map):
    out_dir = pathlib.Path(out_dir)
    (out_dir / "set.000").mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "type.raw", data["atype"], fmt="%d")
    (out_dir / "type_map.raw").write_text("\n".join(type_map) + "\n")
    for key in ("coord", "energy", "force", "box"):
        np.save(out_dir / "set.000" / f"{key}.npy", data[key])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--pairs", type=int, default=16, help="ion pairs per frame")
    ap.add_argument("--box", type=float, default=14.0, help="cubic cell edge, A")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--valid", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-coulomb",
        action="store_true",
        help="ablation: repulsion only, so a local model should fit it exactly",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    type_map = ["Na", "Cl"]
    for split, n_frames, seed in (
        ("train", args.train, args.seed),
        ("valid", args.valid, args.seed + 10_000),
    ):
        data = build(
            n_frames, args.pairs, args.box, seed, device,
            include_coulomb=not args.no_coulomb,
        )
        write(pathlib.Path(args.out) / split, data, type_map)
        print(
            f"{split}: {n_frames} frames, {2 * args.pairs} ions, box {args.box} A, "
            f"E mean {data['energy'].mean():.3f} eV, "
            f"|F| rms {np.sqrt((data['force'] ** 2).mean()):.3f} eV/A"
        )


if __name__ == "__main__":
    main()
