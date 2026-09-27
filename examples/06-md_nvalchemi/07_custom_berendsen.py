"""A thermostat of your own: Berendsen weak coupling, written as a subclass.

The MD classes cover every integrator nvalchemiops ships. For anything else,
subclass `MolecularDynamics` and implement `step()`: nvalchemiops kernels on
`self.thermo` -- the device state (positions, velocities, masses, the forces
view, the cell) -- around exactly one `self.thermo.evaluate()`. The log,
trajectories, observers and checkpoints then work as for the built-in classes.

Berendsen runs a velocity-Verlet step, then scales every velocity by
`sqrt(1 + dt/tau (T0/T - 1))`, which pulls the temperature towards T0 with
time constant tau. It does not sample the canonical ensemble: use it to bring
a structure to temperature, then switch to Langevin or Nose-Hoover (02, 03).

Writing kernels yourself, read "API conventions you cannot guess" in this
folder's README first; the one that matters here is that the full dt goes to
every kernel, which applies its own factor of 1/2.

    python 07_custom_berendsen.py --model ../../path/to/model.ckpt.pt
"""

import argparse
import math

import numpy as np
from ase import units
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from nvalchemiops.dynamics.integrators import (
    velocity_rescale,
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)

from deepmd_lr.nvalchemi import (
    MolecularDynamics,
    build_evaluator,
    read_structure,
)


class Berendsen(MolecularDynamics):
    """Velocity Verlet, then every velocity scaled by sqrt(1 + dt/tau (T0/T - 1))."""

    def __init__(self, evaluator, atoms, timestep, temperature_K, taut, **kwargs):
        super().__init__(evaluator, atoms, timestep, **kwargs)
        self.temperature_K, self.taut = float(temperature_K), float(taut)
        self._scale = self.thermo.scalar(1.0)       # a device scalar the kernel reads

    def step(self):
        th = self.thermo
        velocity_verlet_position_update(th.positions, th.velocities, th.forces,
                                        th.masses, self.dt)
        th.evaluate()                               # exactly one evaluation per step
        velocity_verlet_velocity_finalize(th.velocities, th.forces, th.masses,
                                          self.dt)
        t = th.temperature                          # kelvin; a host read every step
        if t > 0:
            self._scale.fill_(math.sqrt(
                1.0 + self.timestep / self.taut * (self.temperature_K / t - 1.0)))
            velocity_rescale(th.velocities, self._scale)

    def _params(self):                              # what a checkpoint records
        return {"temperature_K": self.temperature_K, "taut": self.taut}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K, the target")
    ap.add_argument("--start-temperature", type=float, default=100.0,
                    help="K, the initial velocities")
    ap.add_argument("--tau", type=float, default=100.0, help="fs, the coupling time")
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms)
    thermalize_momenta(atoms, temperature_K=a.start_temperature,
                       rng=np.random.default_rng(1))
    Stationary(atoms)
    md = Berendsen(ev, atoms, a.dt * units.fs, a.temperature, a.tau * units.fs,
                   logfile="-", loginterval=max(a.steps // 20, 1))
    print(f"Berendsen  {len(atoms)} atoms  {a.start_temperature:g} -> "
          f"{a.temperature:g} K  tau {a.tau:g} fs")
    md.run(a.steps)

    last = md.rows[-max(len(md.rows) // 4, 1):]
    print(f"\n  {md.performance()}")
    print(f"  last quarter: <T> {np.mean([r['T[K]'] for r in last]):.1f} K "
          f"(target {a.temperature:g} K)")


if __name__ == "__main__":
    main()
