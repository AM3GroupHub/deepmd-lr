"""Your own integrator: a kernel-level loop, and the same idea as a class.

The classes cover NVE, Langevin, Nose-Hoover and isotropic NPT. For anything
else you drive nvalchemiops' kernels yourself, in one of two ways:

1. **A raw loop on a `Thermo`.** Thermo holds the device state (positions,
   velocities, masses, the forces view, the cell) and the measurements. It
   does not own time or the ensemble, so you make those scalars yourself:
   ``dt = th.scalar(0.5 * units.fs)``, ``kT = th.scalar(units.kB * T)``.
2. **A `MolecularDynamics` subclass.** Implement ``step()`` -- the kernels around
   exactly one ``self.thermo.evaluate()`` -- and get logging, trajectories,
   observers and checkpoints for free.

The traps the kernels do not announce, all found the hard way:
- **The full dt goes to every kernel.** Each applies its own factor of 1/2.
  Passing dt/2 to the Nose-Hoover kick halves every force: frequencies drop by
  sqrt(2), and configurations sample about 2T while the kinetic temperature
  still reads T.
- **The ndof dtypes differ.** ndof is int32 for ``nhc_compute_masses`` and
  float64 for ``nhc_thermostat_chain_update``. A mismatch is a Warp launch
  error, not a cast.
- **Reset ``total_scale`` to 1** before every chain update. The kernel
  multiplies into it, and starting it at 0 freezes the system solid.
- **The Nose-Hoover chain energy needs eta/2.** Each chain update scales the
  velocities by a half step but moves eta by a full eta_dot * dt, so eta runs
  twice as fast as eta_dot implies. The dynamics are right either way; only
  the conserved quantity is off.
- **The NPT kernels count N_f = 3 x (the atom count you pass).** The centre of
  mass does not move, so pass N - 1.
- **Refresh KE and pressure** (``update_kinetic_energy``, ``update_pressure``)
  right before every barostat/thermostat half-step. Thermo's getters never
  write those buffers, so logging cannot change a run.
- **Never rebind a Thermo array.** Kernels write into the bound buffers; load
  new values with ``th.set_state(...)``.

This script runs one Nose-Hoover NVT twice -- as a raw loop and through the
class -- and then a Berendsen thermostat written as a subclass.

    python 07_custom_loop.py --model ../../path/to/model.ckpt.pt
"""

import argparse
import math

import numpy as np
import warp as wp
from ase import units
from nvalchemiops.dynamics.integrators import (
    nhc_compute_masses,
    nhc_position_update,
    nhc_thermostat_chain_update,
    nhc_velocity_half_step,
    velocity_rescale,
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)

from deepmd_lr.nvalchemi import (
    MolecularDynamics,
    NoseHooverChainNVT,
    Thermo,
    build_evaluator,
    maxwell_boltzmann,
    read_structure,
)


def raw_nose_hoover(ev, atoms, dt_fs, temperature, tau_fs, steps, chain=3):
    """Nose-Hoover NVT by hand, on the shared evaluator. Returns the Thermo."""
    th = Thermo(ev, atoms.get_positions(), velocities=atoms.get_velocities(),
                masses=atoms.get_masses())
    th.remove_com_motion()
    dt = th.scalar(dt_fs * units.fs)                 # the loop owns time ...
    kT = th.scalar(units.kB * temperature)           # ... and the ensemble
    ndof = 3 * th.n_atoms - 3
    ndof_f64 = th.scalar(float(ndof))
    ndof_i32 = wp.array(np.array([ndof]), dtype=wp.int32, device=th.device)
    eta, eta_dot, mass = th.zeros(chain), th.zeros(chain), th.zeros(chain)
    nhc_compute_masses(ndof_i32, kT, th.scalar(tau_fs * units.fs), chain, mass)
    ke2, step_scale, dt_chain = th.zeros(1), th.zeros(1), th.zeros(1)
    total_scale = th.scalar(1.0)

    def chain_half():
        total_scale.fill_(1.0)
        nhc_thermostat_chain_update(th.velocities, th.masses, eta, eta_dot, mass,
                                    kT, dt, ndof_f64, ke2, total_scale,
                                    step_scale, dt_chain, nloops=3)

    th.evaluate()
    for _ in range(steps):
        chain_half()
        nhc_velocity_half_step(th.velocities, th.forces, th.masses, dt)  # full dt
        nhc_position_update(th.positions, th.velocities, dt)
        th.evaluate()
        nhc_velocity_half_step(th.velocities, th.forces, th.masses, dt)
        chain_half()
    return th


class Berendsen(MolecularDynamics):
    """Berendsen weak coupling: velocity Verlet, then rescale every velocity by
    ``sqrt(1 + dt/tau (T0/T - 1))``. Not canonical -- a demonstration."""

    def __init__(self, evaluator, atoms, timestep, temperature_K, taut, **kwargs):
        super().__init__(evaluator, atoms, timestep, **kwargs)
        self.temperature_K, self.taut = float(temperature_K), float(taut)
        self._scale = self.thermo.scalar(1.0)

    def step(self):
        th = self.thermo
        velocity_verlet_position_update(th.positions, th.velocities, th.forces,
                                        th.masses, self.dt)
        th.evaluate()                        # exactly one evaluation per step
        velocity_verlet_velocity_finalize(th.velocities, th.forces, th.masses,
                                          self.dt)
        t = th.temperature                   # a host read each step: fine here
        if t > 0:
            self._scale.fill_(math.sqrt(
                1.0 + self.timestep / self.taut * (self.temperature_K / t - 1.0)))
            velocity_rescale(th.velocities, self._scale)

    def _params(self):                       # what a checkpoint records
        return {"temperature_K": self.temperature_K, "taut": self.taut}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--tau", type=float, default=100.0, help="fs")
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms)
    maxwell_boltzmann(atoms, temperature_K=a.temperature, seed=1)

    raw = raw_nose_hoover(ev, atoms.copy(), a.dt, a.temperature, a.tau, a.steps)
    cls = NoseHooverChainNVT(ev, atoms.copy(), a.dt * units.fs, a.temperature,
                             a.tau * units.fs)
    cls.run(a.steps)
    diff = np.abs(raw.positions.numpy() - cls.thermo.positions.numpy()).max()
    print(f"raw loop vs NoseHooverChainNVT, {a.steps} steps on one evaluator: "
          f"max |dx| {diff:.2e} A")

    berendsen = Berendsen(ev, atoms, a.dt * units.fs, a.temperature,
                          a.tau * units.fs, logfile="-",
                          loginterval=max(a.steps // 10, 1))
    berendsen.run(a.steps)
    print(f"Berendsen subclass: {berendsen.performance()}")


if __name__ == "__main__":
    main()
