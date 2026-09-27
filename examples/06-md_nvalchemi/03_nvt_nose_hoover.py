"""NVT: Nose-Hoover chain.

Deterministic and time-reversible, so unlike Langevin it leaves the dynamics
alone -- the right choice when you want transport properties or vibrational
spectra out of the trajectory. The price is chain state to carry and a
conserved quantity that is not the physical energy: the log's `Econs` column
(physical energy plus the chain's own) is what should stay flat.

It does, to O(dt^2), at any coupling strength. An earlier version of this
script computed the chain energy with the kernel's own `eta`, which runs twice
as fast as `eta_dot` implies, and reported a timestep-independent drift of a few
meV/atom per 20 fs as unexplained. The class uses `eta/2`; the dynamics never
depended on it. The kernel-level traps (full dt, `total_scale`, ndof dtypes)
are written down in this folder's README.

    python 03_nvt_nose_hoover.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
from ase import units

from deepmd_lr.nvalchemi import (
    NoseHooverChainNVT,
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
    ap.add_argument("--tau", type=float, default=100.0,
                    help="thermostat time constant, fs")
    ap.add_argument("--chain-length", type=int, default=3)
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms)
    maxwell_boltzmann(atoms, temperature_K=a.temperature, seed=1)
    md = NoseHooverChainNVT(ev, atoms, a.dt * units.fs, temperature_K=a.temperature,
                            tdamp=a.tau * units.fs, tchain=a.chain_length,
                            logfile="-", loginterval=max(a.steps // 20, 1))
    print(f"Nose-Hoover chain  {len(atoms)} atoms  dt {a.dt} fs  "
          f"T {a.temperature} K  tau {a.tau} fs  chain {a.chain_length}")
    md.run(a.steps)

    half = md.rows[len(md.rows) // 2:]
    print(f"\n  {md.performance()}")
    print(f"  mean T over the second half: "
          f"{np.mean([r['T[K]'] for r in half]):.1f} K (target {a.temperature})")
    print(f"  Econs drift {md.drift('Econs[eV]'):+.4f} meV/atom/ps -- watch this,")
    print("  not Epot + Ekin, which the thermostat moves on purpose.")


if __name__ == "__main__":
    main()
