"""NVT: Nose-Hoover chain.

Deterministic and time-reversible, so unlike Langevin it leaves the dynamics
alone -- the right choice when you want transport properties or vibrational
spectra out of the trajectory. The price is chain state to carry and a
conserved quantity that is not the physical energy.

Three API details that are not guessable from the signatures:

  * `ndof` is **int32** for `nhc_compute_masses` and **float64** for
    `nhc_thermostat_chain_update`. Passing one where the other is wanted is a
    Warp launch error, not a cast, so both are built below.
  * the half-step factors differ between kernels. `nhc_velocity_half_step`
    applies its own 1/2; `nhc_position_update` does not. So the FULL `dt` goes
    to every call and the kernels take care of their own factors.
  * the chain update is called once at EACH END of the step, not once per step.

    python 03_nvt_nose_hoover.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
import warp as wp
from ase.units import fs
from nvalchemiops.dynamics.integrators import (
    nhc_compute_chain_energy,
    nhc_compute_masses,
    nhc_position_update,
    nhc_thermostat_chain_update,
    nhc_velocity_half_step,
)

from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--tau", type=float, default=100.0,
                    help="thermostat time constant, fs")
    ap.add_argument("--chain-length", type=int, default=3)
    ap.add_argument("--nloops", type=int, default=3,
                    help="Yoshida-Suzuki sub-steps")
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

    # The dtype split: int32 to build the chain masses, float64 to propagate.
    # It is a Warp launch error, not a cast, so both forms are needed.
    ndof = 3 * th.n_atoms - 3
    ndof_i32 = wp.array(np.array([ndof]), dtype=wp.int32, device=th.device)
    ndof_f64 = th.scalar(float(ndof))

    L = a.chain_length
    eta = th.zeros(L)
    eta_dot = th.zeros(L)
    eta_mass = th.zeros(L)
    nhc_compute_masses(ndof_i32, th.kT, th.scalar(a.tau * fs), L, eta_mass)

    # Scratch the chain update needs; total_scale must start at one.
    ke2 = th.zeros(1)
    total_scale = th.scalar(1.0)
    step_scale = th.zeros(1)
    dt_chain = th.zeros(1)
    ke_chain = th.zeros(1)
    pe_chain = th.zeros(1)

    def chain():
        total_scale.fill_(1.0)
        nhc_thermostat_chain_update(
            th.velocities, th.masses, eta, eta_dot, eta_mass, th.kT, th.dt,
            ndof_f64, ke2, total_scale, step_scale, dt_chain, nloops=a.nloops,
        )

    def invariant():
        """The MTK conserved quantity: physical energy plus the chain's own.

        The physical total is NOT conserved -- the thermostat moves it on
        purpose. This is what to watch instead.
        """
        nhc_compute_chain_energy(eta, eta_dot, eta_mass, th.kT, ndof_f64,
                                 ke_chain, pe_chain)
        return (th.total_energy + float(ke_chain.numpy()[0])
                + float(pe_chain.numpy()[0]))

    th.evaluate()
    print(f"Nose-Hoover chain  {th.n_atoms} atoms  dt {a.dt} fs  "
          f"T {a.temperature} K  tau {a.tau} fs  chain {L}  nloops {a.nloops}")
    for step in range(a.steps + 1):
        th.log(step, invariant=invariant())
        if step == a.steps:
            break
        chain()                                              # first half
        nhc_velocity_half_step(th.velocities, th.forces, th.masses, th.dt)
        nhc_position_update(th.positions, th.velocities, th.dt)
        th.evaluate()
        nhc_velocity_half_step(th.velocities, th.forces, th.masses, th.dt)
        chain()                                              # second half

    half = len(th.rows) // 2
    rows = th.rows[half:]
    print(f"\n  mean T over the second half: "
          f"{sum(r[2] for r in rows)/len(rows):.1f} K (target {a.temperature})")
    print(f"  invariant drift {th.drift('invariant'):+.4f} meV/atom/ps")
    print("  Watch the invariant, not E_pot+E_kin. Note it drifts a few")
    print("  meV/atom/ps at strong coupling INDEPENDENTLY of dt -- so that")
    print("  drift is not integration error. Weak coupling is the safe regime,")
    print("  and NVE (01_nve.py) is the trustworthy conservation check.")


if __name__ == "__main__":
    main()
