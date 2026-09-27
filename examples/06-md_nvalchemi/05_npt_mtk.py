"""NPT: Martyna-Tobias-Klein, with a Nose-Hoover chain thermostat.

The ensemble that asks most of the model: the barostat is driven by the
pressure tensor, which needs the **virial**, so the evaluator must compute it
-- `compute_virial=True` here, or `evaluator.set_compute_virial(True)` on an
evaluator that ran FIRE or NVT first. It costs a strain leaf per partition and
an extra target in the charge-response backward.

`--cell` picks what the cell may do: `isotropic` (the default here) scales it
uniformly, `orthorhombic` moves the three lengths of a diagonal cell, `full`
lets all of it fluctuate. A liquid has no shear stiffness, so keep isotropic
for water; `full` is for crystals.

What the class does that a hand-written loop has to get right (see
`07_custom_loop.py` for the loop itself):
- **The cell.** After the barostat moves the cell it is pushed to the
  evaluator. Its partitions follow the cell and rebuild only once the cell or
  the atoms have moved enough; reach the density here, then produce in NVT.
- **Fresh inputs.** The kinetic energy and pressure are refreshed right before
  every barostat/thermostat half-step.
- **Degrees of freedom.** The kernels count 3N - 3.
- **The convention.** nvalchemiops' cell update reads the columns of a cell
  as its lattice vectors, where ASE's are rows, so the kernels get the cell
  transposed. Only an isotropic barostat is indifferent to that.

The log's `Econs` is the MTK conserved quantity. It is second order in dt
while the thermostat is near equilibrium; nvalchemiops applies the thermostat
drag inside the kick to first order, which shows when the thermostat works
hard.

    python 05_npt_mtk.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
from ase import units

from deepmd_lr.nvalchemi import (
    MTKNPT,
    build_evaluator,
    maxwell_boltzmann,
    read_structure,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--pressure", type=float, default=0.0, help="GPa")
    ap.add_argument("--tau-t", type=float, default=100.0, help="fs")
    ap.add_argument("--tau-p", type=float, default=1000.0, help="fs")
    ap.add_argument("--chain-length", type=int, default=3)
    ap.add_argument("--cell", choices=("isotropic", "orthorhombic", "full"),
                    default="isotropic")
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms, compute_virial=True)
    maxwell_boltzmann(atoms, temperature_K=a.temperature, seed=1)
    md = MTKNPT(ev, atoms, a.dt * units.fs, temperature_K=a.temperature,
                pressure_au=a.pressure * units.GPa, tdamp=a.tau_t * units.fs,
                pdamp=a.tau_p * units.fs, cell=a.cell, tchain=a.chain_length,
                logfile="-", loginterval=max(a.steps // 20, 1))
    print(f"NPT (MTK, {a.cell} cell)  {len(atoms)} atoms  dt {a.dt} fs  "
          f"T {a.temperature} K  P {a.pressure} GPa  tau_T {a.tau_t} fs  "
          f"tau_P {a.tau_p} fs")
    md.run(a.steps)

    half = md.rows[len(md.rows) // 2:]
    print(f"\n  {md.performance()}")
    print(f"  second half: T {np.mean([r['T[K]'] for r in half]):.1f} K, "
          f"P {np.mean([r['P[GPa]'] for r in half]):+.4f} GPa, "
          f"V {np.mean([r['V[A^3]'] for r in half]):.1f} A^3")
    print("  A barostat equilibrates on tau_P, much longer than a demo run --")
    print("  expect the volume still to be moving here.")


if __name__ == "__main__":
    main()
