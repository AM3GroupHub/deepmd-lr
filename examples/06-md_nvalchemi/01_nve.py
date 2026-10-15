"""NVE: velocity Verlet.

The simplest run, and the one worth doing first on any new model: with no
thermostat the total energy is conserved, which directly tests that the forces
are the gradient of the reported energy. Two numbers to read at the end:
- the rms fluctuation of Etot, which falls about 4x when dt is halved (it is
  O(dt^2));
- the drift, which should be tiny: about 0.01 meV/atom/ps for the bundled
  water box.
A force that is not the energy's gradient leaves a drift no smaller timestep
removes.

    python 01_nve.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
from ase import units
from ase.md.velocitydistribution import Stationary, thermalize_momenta

from deepmd_lr.nvalchemi import (
    VelocityVerlet,
    build_evaluator,
    read_structure,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K, initial")
    ap.add_argument("--structure", default="water_192.xyz",
                    help="anything ase.io.read understands; needs a cell")
    ap.add_argument("--index", type=int, default=-1,
                    help="frame to take from a multi-frame file")
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms)
    thermalize_momenta(atoms, temperature_K=a.temperature, rng=np.random.default_rng(1))
    Stationary(atoms)
    md = VelocityVerlet(ev, atoms, a.dt * units.fs, logfile="-",
                        loginterval=max(a.steps // 20, 1))
    print(f"NVE  {len(atoms)} atoms  dt {a.dt} fs  {a.steps} steps")
    md.run(a.steps)

    etot = np.array([r["Etot[eV]"] for r in md.rows]) * 1e3 / len(atoms)
    print(f"\n  {md.performance()}")
    print(f"  Etot fluctuation {etot.std():.4f} meV/atom rms, "
          f"drift {md.drift('Etot[eV]'):+.4f} meV/atom/ps")
    print("  Halve --dt: the fluctuation should fall ~4x. A drift that does not")
    print("  shrink means the forces are not the exact gradient of the energy --")
    print("  which no thermostat would hide.")


if __name__ == "__main__":
    main()
