"""NVE: velocity Verlet.

The simplest loop, and the one worth running first on any new model: with no
thermostat the total energy is conserved, so its drift is a direct test that
the forces really are the gradient of the reported energy. A force that is not
leaves a drift floor no timestep reduction removes.

    python 01_nve.py --model ../../path/to/model.ckpt.pt
"""

import argparse

from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)

from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure


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
    th = Thermo(
        ev, atoms.get_positions(), timestep=a.dt, temperature=a.temperature,
        columns=("step", "time", "temp", "epot", "ekin", "etot"),
        every=max(a.steps // 10, 1),
    )

    print(f"NVE  {th.n_atoms} atoms  dt {a.dt} fs  {a.steps} steps")
    th.evaluate()
    for step in range(a.steps + 1):
        th.log(step)
        if step == a.steps:
            break
        # B-A: half-kick and drift, both inside one kernel.
        velocity_verlet_position_update(
            th.positions, th.velocities, th.forces, th.masses, th.dt
        )
        th.evaluate()
        # B: the second half-kick, with the new forces.
        velocity_verlet_velocity_finalize(
            th.velocities, th.forces, th.masses, th.dt
        )

    print(f"\n  total-energy drift {th.drift('etot'):+.4f} meV/atom/ps")
    print("  Halve dt and this should fall roughly 4x (it is O(dt^2)).")
    print("  A floor that does not move means the forces are not the exact")
    print("  gradient of the energy -- which no thermostat will hide.")


if __name__ == "__main__":
    main()
