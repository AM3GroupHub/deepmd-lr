# SPDX-License-Identifier: LGPL-3.0-or-later
"""MD on the LiFSI interface with ASE's ensembles.

Runs a trained ``dp_lr`` model -- short-range DPA4C plus the reciprocal-space
electrostatics -- under one of ASE's integrators, optionally evaluating the
model over spatial partitions.

    # constant energy, whole cell on one device
    python md.py --model run_joint/model.ckpt.pt --ensemble nve

    # Langevin thermostat at 300 K, model split over 4 partitions
    python md.py --ensemble nvt-langevin --temperature 300 --partitions 4

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
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from deepmd_lr.distributed import DistLongRangeCalculator

ENSEMBLES = ("nve", "nvt-langevin", "nvt-nose-hoover")


def load_frame(data_dir, frame=0):
    """One periodic frame from a DeePMD-kit npy system, as an ASE Atoms."""
    root = pathlib.Path(data_dir)
    subset = sorted(root.glob("set.*"))[0]
    type_map = (root / "type_map.raw").read_text().split()
    types = np.loadtxt(root / "type.raw", dtype=int).reshape(-1)
    return Atoms(
        symbols=[type_map[t] for t in types],
        positions=np.load(subset / "coord.npy")[frame].reshape(-1, 3),
        cell=np.load(subset / "box.npy")[frame].reshape(3, 3),
        pbc=True,
    )


def build_dynamics(args, atoms):
    """The requested ensemble, with ASE's unit conventions applied."""
    dt = args.timestep * units.fs
    if args.ensemble == "nve":
        return VelocityVerlet(atoms, timestep=dt)
    if args.ensemble == "nvt-langevin":
        # The CLI takes friction in 1/fs; ASE wants it per ASE time unit.
        return Langevin(
            atoms,
            timestep=dt,
            temperature_K=args.temperature,
            friction=args.friction / units.fs,
        )
    return NoseHooverChainNVT(
        atoms,
        timestep=dt,
        temperature_K=args.temperature,
        tdamp=args.tdamp * units.fs,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="run_joint/model.ckpt.pt")
    ap.add_argument("--data", default="data/valid")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--ensemble", choices=ENSEMBLES, default="nve")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--timestep", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--friction", type=float, default=0.01,
                    help="Langevin friction, 1/fs")
    ap.add_argument("--tdamp", type=float, default=100.0,
                    help="Nose-Hoover damping time, fs")
    ap.add_argument("--partitions", type=int, default=1)
    ap.add_argument("--devices", nargs="*", default=None,
                    help="one device per partition; overrides --partitions")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    atoms = load_frame(args.data, args.frame)
    n_atoms = len(atoms)

    calc = DistLongRangeCalculator(args.model)
    calc.enable_distributed_mode(args.devices or args.partitions)
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    dynamics = build_dynamics(args, atoms)

    placement = args.devices or [str(calc.device)] * args.partitions
    print(f"model      : {args.model}")
    print(f"system     : {n_atoms} atoms from {args.data}, frame {args.frame}")
    print(f"ensemble   : {args.ensemble}, dt={args.timestep} fs, "
          f"{args.steps} steps")
    print(f"evaluation : {len(placement)} partition(s) on {sorted(set(placement))}")
    print(f"\n{'step':>7} {'E_pot [eV]':>16} {'E_tot [eV]':>16} {'T [K]':>9}")

    def report(step):
        print(f"{step:>7} {atoms.get_potential_energy():>16.4f} "
              f"{atoms.get_total_energy():>16.4f} "
              f"{atoms.get_temperature():>9.1f}")

    report(0)
    reference = atoms.get_total_energy()
    for step in range(1, args.steps + 1):
        dynamics.run(1)
        if step % args.log_every == 0 or step == args.steps:
            report(step)

    if args.ensemble == "nve":
        drift = (atoms.get_total_energy() - reference) / n_atoms
        picoseconds = args.steps * args.timestep / 1000
        print(f"\nNVE drift: {drift * 1000:+.4f} meV/atom "
              f"({drift * 1000 / picoseconds:+.4f} meV/atom/ps over "
              f"{picoseconds:.3f} ps)")


if __name__ == "__main__":
    main()
