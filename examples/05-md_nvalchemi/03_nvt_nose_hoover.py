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

    python 03_nvt_nose_hoover.py --model ../../experiments/water_joint/model.ckpt.pt
"""

import argparse

import numpy as np
import torch
import warp as wp
from ase.units import fs
from common import (
    Reporter,
    build_evaluator,
    kelvin,
    kt,
    maxwell_boltzmann,
    scalar,
    vec,
    water_box,
)
from nvalchemiops.dynamics.integrators import (
    nhc_compute_chain_energy,
    nhc_compute_masses,
    nhc_position_update,
    nhc_thermostat_chain_update,
    nhc_velocity_half_step,
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
    ap.add_argument("--nloops", type=int, default=3,
                    help="Yoshida-Suzuki sub-steps")
    ap.add_argument("--n-side", type=int, default=6)
    a = ap.parse_args()

    pos, symbols, cell = water_box(a.n_side)
    ev, masses = build_evaluator(a.model, symbols, cell)
    dev = str(ev.home)
    n = len(pos)

    x = torch.tensor(pos, dtype=torch.float64, device=dev)
    v = maxwell_boltzmann(masses, a.temperature, device=dev)
    m = torch.tensor(masses, dtype=torch.float64, device=dev)
    ndof = 3 * n - 3

    wx, wv, wm = vec(x), vec(v), wp.from_torch(m)
    wdt = scalar(a.dt * fs, dev)
    wkT = scalar(kt(a.temperature), dev)

    # The dtype split: int32 to build the masses, float64 to propagate.
    ndof_i32 = wp.array(np.array([ndof]), dtype=wp.int32, device=dev)
    ndof_f64 = scalar(float(ndof), dev)

    L = a.chain_length
    eta = wp.zeros(L, dtype=wp.float64, device=dev)
    eta_dot = wp.zeros(L, dtype=wp.float64, device=dev)
    eta_mass = wp.zeros(L, dtype=wp.float64, device=dev)
    nhc_compute_masses(ndof_i32, wkT, scalar(a.tau * fs, dev), L, eta_mass)

    # Scratch the chain update needs; total_scale must start at one.
    ke2 = wp.zeros(1, dtype=wp.float64, device=dev)
    total_scale = scalar(1.0, dev)
    step_scale = wp.zeros(1, dtype=wp.float64, device=dev)
    dt_chain = wp.zeros(1, dtype=wp.float64, device=dev)
    ke_chain = wp.zeros(1, dtype=wp.float64, device=dev)
    pe_chain = wp.zeros(1, dtype=wp.float64, device=dev)

    def chain():
        total_scale.fill_(1.0)
        nhc_thermostat_chain_update(
            wv, wm, eta, eta_dot, eta_mass, wkT, wdt, ndof_f64, ke2,
            total_scale, step_scale, dt_chain, nloops=a.nloops,
        )

    def invariant(e_pot: float, e_kin: float) -> float:
        """The MTK conserved quantity: physical energy plus the chain's own.

        The physical total is NOT conserved -- the thermostat moves it on
        purpose. This is what to watch instead.
        """
        nhc_compute_chain_energy(eta, eta_dot, eta_mass, wkT, ndof_f64,
                                 ke_chain, pe_chain)
        return e_pot + e_kin + float(ke_chain.numpy()[0]) + float(pe_chain.numpy()[0])

    energy, f = ev(x)
    wf = vec(f)

    print(f"Nose-Hoover chain  {n} atoms  dt {a.dt} fs  T {a.temperature} K  "
          f"tau {a.tau} fs  chain {L}  nloops {a.nloops}")
    log = Reporter("T_K", "E_pot", "E_kin", "invariant",
                   every=max(a.steps // 10, 1))
    for step in range(a.steps + 1):
        ke = 0.5 * float((m[:, None] * v**2).sum())
        log(step, step * a.dt, T_K=kelvin(v, m, ndof), E_pot=energy,
            E_kin=ke, invariant=invariant(energy, ke))
        if step == a.steps:
            break
        chain()                                   # first half of the step
        nhc_velocity_half_step(wv, wf, wm, wdt)   # applies its own 1/2
        nhc_position_update(wx, wv, wdt)          # full dt, no internal 1/2
        energy, f = ev(x)
        wf = vec(f)
        nhc_velocity_half_step(wv, wf, wm, wdt)
        chain()                                   # second half

    half = len(log.rows) // 2
    mean_T = sum(r[2] for r in log.rows[half:]) / max(len(log.rows) - half, 1)
    print(f"\n  mean T over the second half: {mean_T:.1f} K "
          f"(target {a.temperature})")
    print(f"  invariant drift {log.drift('invariant', per_atom=n):+.4f} "
          f"meV/atom/ps")
    print("  Watch the invariant, not E_pot+E_kin. Note it drifts a few")
    print("  meV/atom/ps at strong coupling INDEPENDENTLY of dt -- so that")
    print("  drift is not integration error. Weak coupling is the safe regime,")
    print("  and NVE (01_nve.py) is the trustworthy conservation check.")


if __name__ == "__main__":
    main()
