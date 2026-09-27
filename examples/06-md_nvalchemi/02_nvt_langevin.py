"""NVT: Langevin, BAOAB splitting.

The thermostat of choice when you want the canonical distribution and do not
care about dynamics: BAOAB has the best configurational sampling of the common
Langevin splittings, and unlike Nose-Hoover it needs no chain state.

`friction` is in INVERSE ASE TIME UNITS, and one such unit is 10.18 fs, so the
relaxation time is `1/friction` of those:

    friction 0.01  ->  1018 fs      too weak to equilibrate a short run
    friction 0.1   ->   102 fs      the default here
    friction 0.5   ->    20 fs      very strong; perturbs dynamics

Start from a poorly equilibrated structure and the first few hundred fs are
equilibration, not sampling -- the excess potential energy turns into heat
faster than a weak thermostat removes it.

The noise is seeded by `seed + step`, so a restarted run replays exactly the
noise an uninterrupted one would have drawn. `fixcm=True`, ASE's default,
removes the centre-of-mass motion the noise adds, so the temperature's
3N - 3 degrees of freedom are exact.

    python 02_nvt_langevin.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
from ase import units

from deepmd_lr.nvalchemi import (
    Langevin,
    build_evaluator,
    maxwell_boltzmann,
    read_structure,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--friction", type=float, default=0.1,
                    help="inverse ASE time units; 1/friction is the relaxation time")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms)
    maxwell_boltzmann(atoms, temperature_K=a.temperature, seed=a.seed)
    md = Langevin(ev, atoms, a.dt * units.fs, temperature_K=a.temperature,
                  friction=a.friction, seed=a.seed, logfile="-",
                  loginterval=max(a.steps // 20, 1))
    print(f"Langevin (BAOAB)  {len(atoms)} atoms  dt {a.dt} fs  "
          f"T {a.temperature} K  friction {a.friction}")
    md.run(a.steps)

    half = md.rows[len(md.rows) // 2:]
    print(f"\n  {md.performance()}")
    print(f"  mean T over the second half: "
          f"{np.mean([r['T[K]'] for r in half]):.1f} K (target {a.temperature})")
    print("  There is no conserved quantity here -- the thermostat exchanges")
    print("  energy with a bath on purpose. Judge it by the temperature.")


if __name__ == "__main__":
    main()
