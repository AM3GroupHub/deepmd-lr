"""NVT: Langevin, BAOAB splitting.

The thermostat of choice when you want the canonical distribution and do not
care about dynamics: BAOAB has the best configurational sampling of the common
Langevin splittings, and unlike Nose-Hoover it needs no chain state.

`friction` is in INVERSE ASE TIME UNITS, and one such unit is 10.18 fs, so the
relaxation time is `1/friction` of those:

    friction 0.01  ->  1018 fs      too weak to equilibrate a short run
    friction 0.1   ->   102 fs      the default here
    friction 0.5   ->    20 fs      very strong; perturbs dynamics

Start from a lattice like the one below and the first few hundred fs are
equilibration, not sampling -- the artificial structure dumps potential energy
into heat faster than a weak thermostat removes it.

    python 02_nvt_langevin.py --model ../../path/to/model.ckpt.pt
"""

import argparse

from nvalchemiops.dynamics.integrators import (
    langevin_baoab_finalize,
    langevin_baoab_half_step,
)

from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure


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
    th = Thermo(
        ev, atoms.get_positions(), timestep=a.dt, temperature=a.temperature,
        columns=("step", "time", "temp", "epot", "ekin"),
        every=max(a.steps // 10, 1),
    )
    friction = th.scalar(a.friction)

    print(f"Langevin (BAOAB)  {th.n_atoms} atoms  dt {a.dt} fs  "
          f"T {a.temperature} K  friction {a.friction}")
    th.evaluate()
    for step in range(a.steps + 1):
        th.log(step)
        if step == a.steps:
            break
        # B-A-O-A: half-kick, half-drift, thermostat, half-drift -- one kernel.
        # The seed advances per step so the noise is not repeated.
        langevin_baoab_half_step(th.positions, th.velocities, th.forces,
                                 th.masses, th.dt, th.kT, friction,
                                 a.seed + step)
        th.evaluate()
        # The final B: half-kick with the new forces.
        langevin_baoab_finalize(th.velocities, th.forces, th.masses, th.dt)
    half = len(th.rows) // 2
    rows = th.rows[half:]
    print(f"\n  mean T over the second half: "
          f"{sum(r[2] for r in rows)/len(rows):.1f} K (target {a.temperature})")
    print("  There is no conserved quantity here -- the thermostat exchanges")
    print("  energy with a bath on purpose. Judge it by the temperature.")


if __name__ == "__main__":
    main()
