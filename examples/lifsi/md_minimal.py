# SPDX-License-Identifier: LGPL-3.0-or-later
"""Distributed MD in about ten lines.

    python md_minimal.py

Everything below the loader is the actual API. `md.py` is the same thing with
a CLI, a choice of ensembles, and energy reporting.
"""

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from deepmd_lr.distributed import DistLongRangeCalculator

MODEL = "run_joint/model.ckpt.pt"
DATA = "data/valid"
PARTITIONS = 4


def load_frame(data_dir, frame=0):
    """First frame of a DeePMD-kit npy system, as an ASE Atoms."""
    import pathlib

    root = pathlib.Path(data_dir)
    subset = sorted(root.glob("set.*"))[0]
    type_map = (root / "type_map.raw").read_text().split()
    types = np.loadtxt(root / "type.raw", dtype=int)
    return Atoms(
        symbols=[type_map[t] for t in types],
        positions=np.load(subset / "coord.npy")[frame].reshape(-1, 3),
        cell=np.load(subset / "box.npy")[frame].reshape(3, 3),
        pbc=True,
    )


atoms = load_frame(DATA)

# The whole API: build the calculator, tell it how many partitions to use.
# An integer means "that many partitions on the current device" -- identical
# arithmetic to that many GPUs, which is how this is exercised on a one-GPU
# machine. With real hardware pass devices instead:
#     .enable_distributed_mode(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
atoms.calc = DistLongRangeCalculator(MODEL).enable_distributed_mode(PARTITIONS)

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dynamics = VelocityVerlet(atoms, timestep=0.5 * units.fs)

print(f"{len(atoms)} atoms, {PARTITIONS} partitions")
print(f"{'step':>6} {'E_pot [eV]':>15} {'E_tot [eV]':>15} {'T [K]':>8}")
for step in range(0, 101, 20):
    if step:
        dynamics.run(20)
    print(
        f"{step:>6} {atoms.get_potential_energy():>15.4f} "
        f"{atoms.get_total_energy():>15.4f} {atoms.get_temperature():>8.1f}"
    )
