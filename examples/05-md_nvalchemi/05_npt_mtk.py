"""NPT: Martyna-Tobias-Klein, isotropic cell.

The ensemble that needs the most from the model: the barostat is driven by the
pressure tensor, which needs the **virial**, so the evaluator must be built
with `compute_virial=True`. That costs essentially nothing here -- DeePMD-kit's
lower graph already produces the virial alongside the force.

The part specific to this package: `DistributedEvaluator` takes the cell at
construction, because the spatial partitioning is built against it. NPT moves
the cell every step, so the cell has to be pushed back in and the partitioning
invalidated:

    ev.cell = new_cell
    ev._partitions = None      # force a repartition against it

That is cheap -- it re-runs `partition_system`, not the per-device model
copies. Rebuilding the whole evaluator each step would deepcopy the model and
is what you want to avoid.

    python 05_npt_mtk.py --model ../../path/to/model.ckpt.pt
"""

import argparse

import numpy as np
import torch
import warp as wp
from ase.units import GPa, fs
from common import (
    Reporter,
    Thermo,
    build_evaluator,
    kt,
    maxwell_boltzmann,
    read_structure,
    scalar,
    vec,
)
from nvalchemiops.dynamics.integrators import (
    compute_pressure_tensor,
    npt_barostat_half_step,
    npt_cell_update,
    npt_position_update,
    npt_thermostat_half_step,
    npt_velocity_half_step,
    vec9d,
)


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
    ap.add_argument("--repartition-every", type=int, default=1,
                    help="steps between repartitions as the cell moves")
    ap.add_argument("--structure", default="water_192.xyz",
                    help="anything ase.io.read understands; needs a cell")
    ap.add_argument("--index", type=int, default=-1,
                    help="frame to take from a multi-frame file")
    a = ap.parse_args()

    pos, symbols, cell = read_structure(a.structure, a.index)
    # compute_virial is not optional here: no virial, no pressure, no barostat.
    ev, masses = build_evaluator(a.model, symbols, cell, compute_virial=True)
    dev = str(ev.home)
    n = len(pos)

    x = torch.tensor(pos, dtype=torch.float64, device=dev)
    v = maxwell_boltzmann(masses, a.temperature, device=dev)
    m = torch.tensor(masses, dtype=torch.float64, device=dev)
    ndof = 3 * n - 3

    wx, wv, wm = vec(x), vec(v), wp.from_torch(m)
    thermo = Thermo(n, dev)
    wdt = scalar(a.dt * fs, dev)
    wkT = scalar(kt(a.temperature), dev)
    # eV/A^3 is the model's pressure unit; ase.units.GPa converts into it.
    wP = scalar(a.pressure * GPa, dev)
    natoms_i = wp.array(np.array([n]), dtype=wp.int32, device=dev)

    cells = wp.array(np.asarray(cell)[None], dtype=wp.mat33d, device=dev)
    cell_v = wp.zeros(1, dtype=wp.mat33d, device=dev)
    # The tensor kernels want a flat vec9 layout, and `kinetic_tensors` is
    # 2-D (B, 9) rather than an array of 3x3 -- only `cells` is mat33.
    virial_w = wp.zeros(1, dtype=vec9d, device=dev)
    press_w = wp.zeros(1, dtype=vec9d, device=dev)
    kin_w = wp.zeros((1, 9), dtype=wp.float64, device=dev)
    vols = wp.zeros(1, dtype=wp.float64, device=dev)
    ke_w = wp.zeros(1, dtype=wp.float64, device=dev)

    L = a.chain_length
    eta = wp.zeros((1, L), dtype=wp.float64, device=dev)
    eta_dot = wp.zeros((1, L), dtype=wp.float64, device=dev)
    # Q_0 = ndof kT tau^2, Q_k = kT tau^2 -- the same form nhc_compute_masses
    # uses, written out because this kernel wants a 2-D (system, chain) array.
    q = np.full((1, L), kt(a.temperature) * (a.tau_t * fs) ** 2)
    q[0, 0] *= ndof
    th_mass = wp.array(q, dtype=wp.float64, device=dev)
    # Barostat mass, W = (ndof + 3) kT tau_p^2.
    cell_mass = scalar((ndof + 3) * kt(a.temperature) * (a.tau_p * fs) ** 2, dev)

    def refresh(coord):
        """Forces and virial at the current cell, with the pressure tensor."""
        energy, f = ev(coord)
        virial_w.assign(np.asarray(ev.virial, dtype=np.float64).reshape(1, 9))
        vols.fill_(ev.volume)
        compute_pressure_tensor(wv, wm, virial_w, cells, kin_w, press_w, vols)
        return energy, f

    energy, f = refresh(x)
    wf = vec(f)

    print(f"NPT (MTK)  {n} atoms  dt {a.dt} fs  T {a.temperature} K  "
          f"P {a.pressure} GPa  tau_T {a.tau_t} fs  tau_P {a.tau_p} fs")
    log = Reporter("T_K", "P_GPa", "volume", "E_pot", every=max(a.steps // 10, 1))
    for step in range(a.steps + 1):
        ke = thermo.kinetic_energy(wv, wm)
        ke_w.fill_(ke)
        p = press_w.numpy()[0]
        p_iso = float(p[0] + p[4] + p[8]) / 3.0
        log(step, step * a.dt, T_K=thermo.kelvin(wv, wm), P_GPa=p_iso / GPa,
            volume=ev.volume, E_pot=energy)
        if step == a.steps:
            break

        # First half: thermostat, barostat, then the velocity half-kick that
        # already knows about the cell's motion.
        npt_thermostat_half_step(eta, eta_dot, ke_w, wkT, th_mass, natoms_i, L, wdt)
        npt_barostat_half_step(cell_v, press_w, wP, vols, cell_mass, ke_w,
                               natoms_i, wdt)
        npt_velocity_half_step(wv, wm, wf, cell_v, volumes=vols,
                               eta_dots=eta_dot, num_atoms=natoms_i, dt=wdt)
        # Drift the atoms and the cell together.
        npt_position_update(wx, wv, cells, cell_v, wdt)
        npt_cell_update(cells, cell_v, wdt)

        # Hand the moved cell back to the evaluator, then re-partition.
        new_cell = cells.numpy()[0]
        ev.cell = np.asarray(new_cell, dtype=np.float64)
        if step % a.repartition_every == 0:
            ev._partitions = None
        energy, f = refresh(x)
        wf = vec(f)

        # Second half, in reverse order.
        ke_w.fill_(thermo.kinetic_energy(wv, wm))
        npt_velocity_half_step(wv, wm, wf, cell_v, volumes=vols,
                               eta_dots=eta_dot, num_atoms=natoms_i, dt=wdt)
        npt_barostat_half_step(cell_v, press_w, wP, vols, cell_mass, ke_w,
                               natoms_i, wdt)
        npt_thermostat_half_step(eta, eta_dot, ke_w, wkT, th_mass, natoms_i, L, wdt)

    half = len(log.rows) // 2
    rows = log.rows[half:]
    print(f"\n  second half: T {sum(r[2] for r in rows)/len(rows):.1f} K, "
          f"P {sum(r[3] for r in rows)/len(rows):+.4f} GPa, "
          f"V {sum(r[4] for r in rows)/len(rows):.1f} A^3")
    print("  A barostat equilibrates on tau_P, which is much longer than a")
    print("  demo run -- expect the volume still to be moving here.")


if __name__ == "__main__":
    main()
