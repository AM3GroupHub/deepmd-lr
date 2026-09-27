"""NVT: velocity rescaling.

The bluntest thermostat there is -- multiply every velocity by
`sqrt(T_target / T_now)`. It samples no correct ensemble and destroys the
dynamics, so never use it for production. It is genuinely useful for one
thing: dragging a badly prepared structure to temperature before handing it to
a real thermostat.

`VelocityRescaling` is velocity Verlet plus nvalchemiops' `velocity_rescale`
kernel, applied only when the temperature strays from the target by more than
`tolerance` (as LAMMPS' fix temp/rescale does), and only every `interval`
steps. A policy of your own fits an observer: `md.attach(fn, interval)` sees
`atoms` synced from the GPU, and what it writes back into `atoms` -- new
momenta, say -- is uploaded before the next step.

    python 04_nvt_rescale.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
from ase import units

from deepmd_lr.nvalchemi import (
    VelocityRescaling,
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
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="K; rescale only when |T - target| exceeds this")
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms)
    maxwell_boltzmann(atoms, temperature_K=a.temperature, seed=1)
    md = VelocityRescaling(ev, atoms, a.dt * units.fs, a.temperature,
                           tolerance=a.tolerance, logfile="-",
                           loginterval=max(a.steps // 20, 1),
                           extra_columns={"rescaled": lambda m: m.rescales})
    print(f"velocity rescaling  {len(atoms)} atoms  dt {a.dt} fs  "
          f"T {a.temperature} K  tolerance {a.tolerance} K")
    md.run(a.steps)

    half = md.rows[len(md.rows) // 2:]
    print(f"\n  {md.performance()}")
    print(f"  mean T over the second half: "
          f"{np.mean([r['T[K]'] for r in half]):.1f} K   ({md.rescales} rescalings)")
    print("  Use this to equilibrate, then switch to Langevin or Nose-Hoover")
    print("  before collecting anything you intend to report.")


if __name__ == "__main__":
    main()
