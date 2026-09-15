"""NVT: velocity rescaling.

The bluntest thermostat there is -- multiply every velocity by
`sqrt(T_target / T_now)`. It samples no correct ensemble and destroys the
dynamics, so never use it for production. It is genuinely useful for one
thing: dragging a badly prepared structure to temperature before handing it to
a real thermostat, which is exactly what the lattice these examples build
needs.

`velocity_rescale` takes the scale factor, not a temperature, so the policy is
yours -- here, rescale only when the drift exceeds a tolerance.

    python 04_nvt_rescale.py --model ../../experiments/water_joint/model.ckpt.pt
"""

import argparse

import torch
import warp as wp
from ase.units import fs
from common import (
    Reporter,
    build_evaluator,
    kelvin,
    maxwell_boltzmann,
    scalar,
    vec,
    water_box,
)
from nvalchemiops.dynamics.integrators import (
    velocity_rescale,
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="K; rescale only when |T - target| exceeds this")
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
    wscale = scalar(1.0, dev)

    energy, f = ev(x)
    wf = vec(f)

    print(f"velocity rescaling  {n} atoms  dt {a.dt} fs  T {a.temperature} K  "
          f"tolerance {a.tolerance} K")
    log = Reporter("T_K", "E_pot", "rescaled", every=max(a.steps // 10, 1))
    rescales = 0
    for step in range(a.steps + 1):
        T = kelvin(v, m, ndof)
        log(step, step * a.dt, T_K=T, E_pot=energy, rescaled=rescales)
        if step == a.steps:
            break
        if abs(T - a.temperature) > a.tolerance and T > 0:
            wscale.fill_((a.temperature / T) ** 0.5)
            velocity_rescale(wv, wscale)
            rescales += 1
        velocity_verlet_position_update(wx, wv, wf, wm, wdt)
        energy, f = ev(x)
        wf = vec(f)
        velocity_verlet_velocity_finalize(wv, wf, wm, wdt)

    half = len(log.rows) // 2
    mean_T = sum(r[2] for r in log.rows[half:]) / max(len(log.rows) - half, 1)
    print(f"\n  mean T over the second half: {mean_T:.1f} K   "
          f"({rescales} rescalings)")
    print("  Use this to equilibrate, then switch to Langevin or Nose-Hoover")
    print("  before collecting anything you intend to report.")


if __name__ == "__main__":
    main()
