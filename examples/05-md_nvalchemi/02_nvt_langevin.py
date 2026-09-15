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

    python 02_nvt_langevin.py --model ../../experiments/water_joint/model.ckpt.pt
"""

import argparse

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
    langevin_baoab_finalize,
    langevin_baoab_half_step,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--friction", type=float, default=0.1,
                    help="inverse ASE time units")
    ap.add_argument("--n-side", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
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
    # k_B T in eV, NOT kelvin. See common.kt.
    wkT = scalar(kt(a.temperature), dev)
    wfric = scalar(a.friction, dev)

    energy, f = ev(x)
    wf = vec(f)

    print(f"Langevin (BAOAB)  {n} atoms  dt {a.dt} fs  T {a.temperature} K  "
          f"friction {a.friction}")
    log = Reporter("T_K", "E_pot", "E_kin", every=max(a.steps // 10, 1))
    for step in range(a.steps + 1):
        ke = 0.5 * float((m[:, None] * v**2).sum())
        log(step, step * a.dt, T_K=kelvin(v, m, ndof), E_pot=energy, E_kin=ke)
        if step == a.steps:
            break
        # B-A-O-A: half-kick, half-drift, thermostat, half-drift -- one kernel.
        # The seed advances per step so the noise is not repeated.
        langevin_baoab_half_step(wx, wv, wf, wm, wdt, wkT, wfric,
                                 a.seed + step)
        energy, f = ev(x)
        wf = vec(f)
        # The final B: half-kick with the new forces.
        langevin_baoab_finalize(wv, wf, wm, wdt)

    half = len(log.rows) // 2
    mean_T = sum(r[2] for r in log.rows[half:]) / max(len(log.rows) - half, 1)
    print(f"\n  mean T over the second half: {mean_T:.1f} K "
          f"(target {a.temperature})")
    print("  There is no conserved quantity here -- the thermostat exchanges")
    print("  energy with a bath on purpose. Judge it by the temperature.")


if __name__ == "__main__":
    main()
