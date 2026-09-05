# SPDX-License-Identifier: LGPL-3.0-or-later
"""MD on the LiFSI interface with nvalchemi's ensembles.

Runs a trained ``dp_lr`` model -- short-range DPA4C plus the reciprocal-space
electrostatics -- under one of nvalchemi's integrators, optionally evaluating
the model over spatial partitions.

    # constant energy, whole cell on one device
    python md_nvalchemi.py --model run_joint/model.ckpt.pt --ensemble nve

    # Langevin thermostat at 300 K, model split over 4 partitions
    python md_nvalchemi.py --ensemble nvt-langevin --temperature 300 \
        --partitions 4

``--partitions N`` on a single-GPU machine puts N partitions on that one GPU:
the arithmetic is identical to N devices, only the placement differs, which is
how the partitioning is exercised without multi-GPU hardware. Pass
``--devices cuda:0 cuda:1 ...`` when there is more than one.

NVE is the honest check on a long-range model: forces have to be the exact
gradient of an energy that includes the reciprocal term *and* the
charge-response contribution, or the total energy drifts.
"""

import argparse
import pathlib

import numpy as np
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import initialize_velocities
from nvalchemi.dynamics.integrators import NVE, NVTLangevin, NVTNoseHoover
from nvalchemi.neighbors import compute_neighbors

from deepmd_lr.ase import load_model
from deepmd_lr.nvalchemi import LongRangeNVAlchemiModel

ENSEMBLES = {"nve": NVE, "nvt-langevin": NVTLangevin, "nvt-nose-hoover": NVTNoseHoover}
BOLTZMANN = 8.617333262e-5  # eV/K, only for reporting T from kinetic energy.
# nvalchemi takes temperatures in KELVIN throughout -- both `initialize_velocities`
# and the thermostats -- not in energy units.


def load_frame(data_dir, frame, device, dtype=torch.float64):
    """One periodic frame from a DeePMD-kit npy system, as atomic numbers."""
    from ase.data import atomic_numbers as ATOMIC_NUMBERS

    root = pathlib.Path(data_dir)
    subset = sorted(root.glob("set.*"))[0]
    coord = np.load(subset / "coord.npy")[frame].reshape(-1, 3)
    cell = np.load(subset / "box.npy")[frame].reshape(3, 3)
    type_index = np.loadtxt(root / "type.raw", dtype=int).reshape(-1)
    type_map = (root / "type_map.raw").read_text().split()
    z = np.array([ATOMIC_NUMBERS[type_map[t]] for t in type_index], dtype=np.int64)
    return (
        torch.tensor(z, dtype=torch.long, device=device),
        torch.tensor(coord, dtype=dtype, device=device),
        torch.tensor(cell, dtype=dtype, device=device).reshape(1, 3, 3),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="run_joint/model.ckpt.pt")
    ap.add_argument("--data", default="data/valid")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--ensemble", choices=sorted(ENSEMBLES), default="nve")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--timestep", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--friction", type=float, default=0.01,
                    help="Langevin friction, 1/fs")
    ap.add_argument("--partitions", type=int, default=1)
    ap.add_argument("--devices", nargs="*", default=None,
                    help="one device per partition; overrides --partitions")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    devices = args.devices or ([str(device)] * args.partitions)
    wrapped = LongRangeNVAlchemiModel(
        load_model(args.model, device),
        devices=None if len(devices) == 1 and args.partitions == 1 else devices,
    ).eval()

    numbers, positions, cell = load_frame(args.data, args.frame, device)
    n_atoms = len(numbers)
    data = AtomicData(
        atomic_numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
        forces=torch.zeros_like(positions),
        energy=torch.zeros((1, 1), dtype=positions.dtype, device=device),
        velocities=torch.zeros_like(positions),
    )
    batch = Batch.from_data_list([data], device=device)
    compute_neighbors(batch, config=wrapped.model_config.neighbor_config)

    # nvalchemi's initializer is a torch op over tensors, and writes into
    # `velocities` in place.
    initialize_velocities(
        batch.velocities,
        batch.atomic_masses,
        torch.tensor([args.temperature], dtype=positions.dtype, device=device),
        batch.batch_idx,
        random_seed=args.seed,
    )

    kwargs = {"dt": args.timestep}
    if args.ensemble != "nve":
        kwargs["temperature"] = args.temperature
    if args.ensemble == "nvt-langevin":
        kwargs["friction"] = args.friction
    dynamics = ENSEMBLES[args.ensemble](wrapped, **kwargs)

    print(f"model      : {args.model}  (rcut={wrapped.rcut} A)")
    print(f"system     : {n_atoms} atoms from {args.data}, frame {args.frame}")
    print(f"ensemble   : {args.ensemble}, dt={args.timestep} fs, "
          f"{args.steps} steps")
    print(f"evaluation : {len(devices)} partition(s) on {sorted(set(devices))}")
    print(f"\n{'step':>7} {'E_pot [eV]':>16} {'E_tot [eV]':>16} {'T [K]':>9}")

    def report(step):
        kinetic = float(
            (0.5 * batch.atomic_masses.unsqueeze(-1) * batch.velocities**2).sum()
        )
        potential = float(batch.energy.sum())
        temperature = 2.0 * kinetic / (3.0 * n_atoms * BOLTZMANN)
        print(f"{step:>7} {potential:>16.4f} {potential + kinetic:>16.4f} "
              f"{temperature:>9.1f}")

    dynamics.compute(batch)
    report(0)
    reference = float(batch.energy.sum()) + float(
        (0.5 * batch.atomic_masses.unsqueeze(-1) * batch.velocities**2).sum()
    )
    for step in range(1, args.steps + 1):
        dynamics.step(batch)
        if step % args.log_every == 0 or step == args.steps:
            report(step)

    if args.ensemble == "nve":
        kinetic = float(
            (0.5 * batch.atomic_masses.unsqueeze(-1) * batch.velocities**2).sum()
        )
        drift = (float(batch.energy.sum()) + kinetic - reference) / n_atoms
        picoseconds = args.steps * args.timestep / 1000
        print(f"\nNVE drift: {drift * 1000:+.4f} meV/atom "
              f"({drift * 1000 / picoseconds:+.4f} meV/atom/ps over "
              f"{picoseconds:.3f} ps)")


if __name__ == "__main__":
    main()
