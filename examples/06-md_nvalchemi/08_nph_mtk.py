"""NPH: the Martyna-Tobias-Klein barostat alone.

Constant enthalpy and pressure, with no thermostat: the temperature goes where
the dynamics take it. What stays constant is the log's `Econs`, the enthalpy
Epot + Ekin + P V + W |eps_dot|^2 / 2 -- the check that the barostat is
integrated right, as NVE's total energy is for velocity Verlet.

As for NPT, the barostat needs the **virial**, `--cell` picks what the cell
may do, and the new cell is pushed to the evaluator every step, which
repartitions. The barostat mass W = 3N kT tau_P^2 still needs a temperature:
the atoms' own at the start, unless `temperature_K` is given.

With no thermostat and no field, nothing here is first order: halving dt
shrinks the fluctuation of `Econs` about 4x in every cell mode. (A stiff
barostat, tau_P of a few hundred fs, erodes that: nvalchemiops moves the cell
by a first-order step, whose error grows with the strain rate squared.)

    python 08_nph_mtk.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
from ase import units

from deepmd_lr.nvalchemi import (
    MTKNPH,
    build_evaluator,
    maxwell_boltzmann,
    read_structure,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0,
                    help="K, of the initial velocities")
    ap.add_argument("--pressure", type=float, default=0.0, help="GPa")
    ap.add_argument("--tau-p", type=float, default=1000.0, help="fs")
    ap.add_argument("--cell", choices=("isotropic", "orthorhombic", "full"),
                    default="isotropic")
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms, compute_virial=True)
    maxwell_boltzmann(atoms, temperature_K=a.temperature, seed=1)
    md = MTKNPH(ev, atoms, a.dt * units.fs, pressure_au=a.pressure * units.GPa,
                pdamp=a.tau_p * units.fs, cell=a.cell, logfile="-",
                loginterval=max(a.steps // 20, 1))
    print(f"NPH (MTK, {a.cell} cell)  {len(atoms)} atoms  dt {a.dt} fs  "
          f"P {a.pressure} GPa  tau_P {a.tau_p} fs")
    md.run(a.steps)

    econs = np.array([r["Econs[eV]"] for r in md.rows]) * 1e3 / len(atoms)
    half = md.rows[len(md.rows) // 2:]
    print(f"\n  {md.performance()}")
    print(f"  Econs fluctuation {np.std(econs):.4f} meV/atom rms, "
          f"drift {md.drift('Econs[eV]'):+.4f} meV/atom/ps")
    print(f"  second half: T {np.mean([r['T[K]'] for r in half]):.1f} K, "
          f"P {np.mean([r['P[GPa]'] for r in half]):+.4f} GPa, "
          f"V {np.mean([r['V[A^3]'] for r in half]):.1f} A^3")
    print("  Halve --dt: the fluctuation should fall ~4x. Judge a drift over")
    print("  0.5 ps or more; a short run's fit mostly measures the fluctuation.")


if __name__ == "__main__":
    main()
