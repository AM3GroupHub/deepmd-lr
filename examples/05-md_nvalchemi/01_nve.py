"""NVE: velocity Verlet.

The simplest loop, and the one worth running first on any new model: with no
thermostat the total energy is conserved, so its drift is a direct test that
the forces really are the gradient of the reported energy. A force that is not
leaves a drift floor no timestep reduction removes.

    python 01_nve.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import torch
import warp as wp
from ase.units import fs
from common import (
    Reporter,
    Thermo,
    build_evaluator,
    maxwell_boltzmann,
    read_structure,
    scalar,
    vec,
)
from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)


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

    pos, symbols, cell = read_structure(a.structure, a.index)
    ev, masses = build_evaluator(a.model, symbols, cell)
    dev = str(ev.home)
    n = len(pos)

    x = torch.tensor(pos, dtype=torch.float64, device=dev)
    v = maxwell_boltzmann(masses, a.temperature, device=dev)
    m = torch.tensor(masses, dtype=torch.float64, device=dev)

    # Warp views. These alias the torch tensors, so the kernels update them in
    # place and nothing round-trips through the host.
    wx, wv, wm = vec(x), vec(v), wp.from_torch(m)
    thermo = Thermo(n, dev)
    wdt = scalar(a.dt * fs, dev)

    energy, f = ev(x)
    wf = vec(f)

    print(f"NVE  {n} atoms  dt {a.dt} fs  {a.steps} steps")
    log = Reporter("T_K", "E_pot", "E_kin", "E_tot", every=max(a.steps // 10, 1))
    for step in range(a.steps + 1):
        ke = thermo.kinetic_energy(wv, wm)
        log(step, step * a.dt, T_K=thermo.kelvin(wv, wm), E_pot=energy,
            E_kin=ke, E_tot=energy + ke)
        if step == a.steps:
            break
        # B-A: half-kick and drift, both inside one kernel.
        velocity_verlet_position_update(wx, wv, wf, wm, wdt)
        energy, f = ev(x)
        wf = vec(f)
        # B: the second half-kick, with the new forces.
        velocity_verlet_velocity_finalize(wv, wf, wm, wdt)

    d = log.drift("E_tot", per_atom=n)
    print(f"\n  total-energy drift {d:+.4f} meV/atom/ps")
    print("  Halve dt and this should fall roughly 4x (it is O(dt^2)).")
    print("  A floor that does not move means the forces are not the exact")
    print("  gradient of the energy -- which no thermostat will hide.")


if __name__ == "__main__":
    main()
