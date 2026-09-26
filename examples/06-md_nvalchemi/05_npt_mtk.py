"""NPT: Martyna-Tobias-Klein, isotropic cell.

The ensemble that asks most of the model: the barostat is driven by the
pressure tensor, which needs the **virial**, so the evaluator must be built
with `compute_virial=True`. That costs a strain leaf per partition and an
extra target in the charge-response backward.

The part specific to this package is the cell. `DistributedEvaluator` builds
its spatial partitioning against the cell it was given, and the Verlet
criterion that decides when to rebuild watches *atom displacement only* -- it
cannot see the cell move. So after the barostat moves it, `thermo.set_cell()`
pushes the new cell back, invalidates the partitioning and refreshes the
volume, in that order. Skip it and the halo goes stale with ghost images at
the wrong offsets: forces quietly wrong, nothing raising.

Two things the kernel signatures do not say:

  * Refresh the kinetic energy and the pressure right before each barostat
    and thermostat half-step, including after the second kick, as LAMMPS'
    fix nh does. The barostat must see the current state, not a value cached
    earlier in the step.
  * The kernels compute N_f = 3 * (the atom count they are given). The centre
    of mass is at rest and stays there, so the moving degrees of freedom are
    3N - 3: pass N - 1. Otherwise the system is held 3N/(3N-3) hot.

    python 05_npt_mtk.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
import warp as wp
from ase.units import GPa, fs
from nvalchemiops.dynamics.integrators import (
    npt_barostat_half_step,
    npt_cell_update,
    npt_position_update,
    npt_thermostat_half_step,
    npt_velocity_half_step,
)

from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure


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
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    # compute_virial is not optional here: no virial, no pressure, no barostat.
    ev = build_evaluator(a.model, atoms, compute_virial=True)
    th = Thermo(
        ev, atoms.get_positions(), timestep=a.dt, temperature=a.temperature,
        columns=("step", "time", "temp", "press", "vol", "epot"),
        every=max(a.steps // 10, 1),
    )

    # Ensemble state -- the thermostat chain and the barostat. This belongs to
    # the loop, not to Thermo, which is why it is allocated here.
    L = a.chain_length
    eta = th.zeros((1, L))
    eta_dot = th.zeros((1, L))
    # Q_0 = ndof kT tau^2, Q_k = kT kT tau^2, as a 2-D (system, chain) array.
    ndof = 3 * th.n_atoms - 3
    q = np.full((1, L), float(th.kT.numpy()[0]) * (a.tau_t * fs) ** 2)
    q[0, 0] *= ndof
    th_mass = wp.array(q, dtype=wp.float64, device=th.device)
    cell_v = th.zeros(1, dtype=wp.mat33d)
    # Barostat mass, W = (ndof + 3) kT tau_p^2.
    cell_mass = th.scalar(
        (ndof + 3) * float(th.kT.numpy()[0]) * (a.tau_p * fs) ** 2
    )
    target_p = th.scalar(a.pressure * GPa)
    # N - 1: the kernels count N_f = 3 * this, and the COM does not move.
    dof_count = wp.array([th.n_atoms - 1], dtype=wp.int32, device=th.device)

    th.evaluate()

    print(f"NPT (MTK)  {th.n_atoms} atoms  dt {a.dt} fs  T {a.temperature} K  "
          f"P {a.pressure} GPa  tau_T {a.tau_t} fs  tau_P {a.tau_p} fs")
    for step in range(a.steps + 1):
        th.log(step)
        if step == a.steps:
            break

        # First half: thermostat, barostat, then the velocity half-kick that
        # already knows about the cell's motion -- all on fresh KE and pressure.
        th.update_kinetic_energy()
        th.update_pressure()
        npt_thermostat_half_step(eta, eta_dot, th.ke_w, th.kT, th_mass,
                                 dof_count, L, th.dt)
        npt_barostat_half_step(cell_v, th.press_w, target_p, th.volume_w,
                               cell_mass, th.ke_w, dof_count, th.dt)
        npt_velocity_half_step(th.velocities, th.masses, th.forces, cell_v,
                               eta_dots=eta_dot, num_atoms=dof_count, dt=th.dt)
        # Drift the atoms and the cell together, then re-sync the evaluator.
        npt_position_update(th.positions, th.velocities, th.cell, cell_v, th.dt)
        npt_cell_update(th.cell, cell_v, th.dt)
        th.set_cell()
        th.evaluate()

        # Second half, in reverse order: the kick, then KE and pressure after it.
        npt_velocity_half_step(th.velocities, th.masses, th.forces, cell_v,
                               eta_dots=eta_dot, num_atoms=dof_count, dt=th.dt)
        th.update_kinetic_energy()
        th.update_pressure()
        npt_barostat_half_step(cell_v, th.press_w, target_p, th.volume_w,
                               cell_mass, th.ke_w, dof_count, th.dt)
        npt_thermostat_half_step(eta, eta_dot, th.ke_w, th.kT, th_mass,
                                 dof_count, L, th.dt)

    half = len(th.rows) // 2
    rows = th.rows[half:]
    print(f"\n  {th.performance()}")
    print(f"  second half: T {sum(r[2] for r in rows)/len(rows):.1f} K, "
          f"P {sum(r[3] for r in rows)/len(rows):+.4f} GPa, "
          f"V {sum(r[4] for r in rows)/len(rows):.1f} A^3")
    print("  A barostat equilibrates on tau_P, much longer than a demo run --")
    print("  expect the volume still to be moving here.")


if __name__ == "__main__":
    main()
