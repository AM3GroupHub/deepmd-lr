# SPDX-License-Identifier: LGPL-3.0-or-later
"""NVE energy conservation for a trained dp_lr model.

The sharpest end-to-end test that forces really are the gradient of the
energy, long-range term included: a non-conservative force injects or drains
energy, and constant-energy dynamics puts that on display over time in a way
no single-point check does.
"""

import argparse
import pathlib

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from deepmd_lr.ase import LongRangeCalculator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="run_joint/model.ckpt.pt")
    ap.add_argument("--data", default="data/valid",
                    help="DeePMD system to take the starting frame from")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--timestep", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the initial velocities, so a timestep sweep "
                         "compares the same trajectory rather than different ones")
    args = ap.parse_args()
    np.random.seed(args.seed)

    root = pathlib.Path(args.data)
    coord = np.load(root / "set.000/coord.npy")[0]
    box = np.load(root / "set.000/box.npy")[0].reshape(3, 3)
    atype = np.loadtxt(root / "type.raw", dtype=int)
    type_map = (root / "type_map.raw").read_text().split()
    symbols = [type_map[t] for t in atype]

    atoms = Atoms(
        symbols=symbols, positions=coord.reshape(-1, 3), cell=box, pbc=True
    )
    atoms.calc = LongRangeCalculator(args.model)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    dyn = VelocityVerlet(atoms, timestep=args.timestep * units.fs)
    n_atoms = len(atoms)
    trace = []

    def record() -> None:
        trace.append(
            (
                atoms.get_potential_energy() + atoms.get_kinetic_energy(),
                atoms.get_temperature(),
                float(atoms.calc.results.get("energy_long_range", np.nan)),
            )
        )

    record()
    dyn.attach(record, interval=max(args.steps // 100, 1))
    dyn.run(args.steps)

    total = np.array([t[0] for t in trace])
    temperature = np.array([t[1] for t in trace])
    long_range = np.array([t[2] for t in trace])

    picoseconds = args.steps * args.timestep / 1000
    drift = (total[-1] - total[0]) / n_atoms
    print(f"{n_atoms} atoms from {args.data}, {picoseconds:.2f} ps "
          f"at dt = {args.timestep} fs")
    print(f"  mean T                : {temperature.mean():.1f} K")
    print(f"  total energy drift    : {drift * 1000:+.4f} meV/atom")
    print(f"  drift rate            : {drift * 1000 / picoseconds:+.4f} meV/atom/ps")
    print(f"  total energy std      : {total.std() / n_atoms * 1000:.4f} meV/atom")
    print(f"  long-range energy     : {long_range.mean():.3f} +- "
          f"{long_range.std():.3f} eV  (it is doing work, not sitting still)")


if __name__ == "__main__":
    main()
