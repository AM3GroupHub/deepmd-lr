"""NVT: velocity rescaling.

The bluntest thermostat there is -- multiply every velocity by
`sqrt(T_target / T_now)`. It samples no correct ensemble and destroys the
dynamics, so never use it for production. It is genuinely useful for one
thing: dragging a badly prepared structure to temperature before handing it to
a real thermostat, which is exactly what the lattice these examples build
needs.

`velocity_rescale` takes the scale factor, not a temperature, so the policy is
yours -- here, rescale only when the drift exceeds a tolerance.

    python 04_nvt_rescale.py --model ../../path/to/model.ckpt.pt
"""

import argparse

from nvalchemiops.dynamics.integrators import (
    velocity_rescale,
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)

from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure


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
    th = Thermo(
        ev, atoms.get_positions(), timestep=a.dt, temperature=a.temperature,
        columns=("step", "time", "temp", "epot"),
        every=max(a.steps // 10, 1),
    )
    scale = th.scalar(1.0)
    rescales = 0

    print(f"velocity rescaling  {th.n_atoms} atoms  dt {a.dt} fs  "
          f"T {a.temperature} K  tolerance {a.tolerance} K")
    th.evaluate()
    for step in range(a.steps + 1):
        temperature = th.temperature
        th.log(step, rescaled=rescales)
        if step == a.steps:
            break
        if abs(temperature - a.temperature) > a.tolerance and temperature > 0:
            scale.fill_((a.temperature / temperature) ** 0.5)
            velocity_rescale(th.velocities, scale)
            rescales += 1
        velocity_verlet_position_update(th.positions, th.velocities,
                                        th.forces, th.masses, th.dt)
        th.evaluate()
        velocity_verlet_velocity_finalize(th.velocities, th.forces,
                                          th.masses, th.dt)
    half = len(th.rows) // 2
    rows = th.rows[half:]
    print(f"\n  mean T over the second half: "
          f"{sum(r[2] for r in rows)/len(rows):.1f} K   ({rescales} rescalings)")
    print("  Use this to equilibrate, then switch to Langevin or Nose-Hoover")
    print("  before collecting anything you intend to report.")


if __name__ == "__main__":
    main()
