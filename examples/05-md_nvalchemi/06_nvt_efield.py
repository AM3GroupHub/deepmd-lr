"""NVT under a uniform external electric field.

The capability this package exists for: the charge head predicts
geometry-dependent charges, their derivative is a Born effective charge tensor
Z, and `F_ib = E_a Z_iab` is the force a field exerts. Pass `efield` to the
evaluator and every returned force already carries it.

Two things to understand before reading a trajectory from this:

  * **There is no conserved quantity.** `F = Z^T E` is a local response, not
    the gradient of a `-M.E` energy -- under periodic boundaries `M` is not
    even defined. Do not read energy drift as an integrator problem. Use a
    thermostat and look at structure and response.
  * **Couple the thermostat to the field.** The field does work with nothing
    bounding it, so the thermostat is the only sink. Measured on bulk water at
    2 V/A: friction 0.01 runs away (Z grows on the distorted geometry, the
    force grows with it, and the feedback closes), friction 0.5 is stable.

The net force is `Q.E`, exactly zero for a neutral cell, so nothing drifts.

    python 06_nvt_efield.py --model ../../path/to/model.ckpt.pt \
        --efield 0.5
"""

import argparse

import numpy as np
from nvalchemiops.dynamics.integrators import (
    langevin_baoab_finalize,
    langevin_baoab_half_step,
)

from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure


def alignment(atoms) -> float:
    """Mean projection of each molecule's HOH bisector on +x.

    0 for an isotropic liquid, 1 for perfect alignment -- the observable a
    field run is for. Minimum image assumes an orthorhombic cell.
    """
    p = atoms.get_positions()
    L = np.diag(np.asarray(atoms.get_cell()))
    sym = np.array(atoms.get_chemical_symbols())
    oxy = np.where(sym == "O")[0]
    hyd = np.where(sym == "H")[0]
    d = p[hyd][:, None, :] - p[oxy][None, :, :]
    d -= np.round(d / L) * L
    own = np.argmin((d**2).sum(-1), axis=1)
    mu = np.zeros((len(oxy), 3))
    for k in range(len(oxy)):
        mu[k] = d[own == k, k].sum(axis=0)
    n = np.linalg.norm(mu, axis=1)
    n[n == 0] = 1.0
    return float((mu[:, 0] / n).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--efield", type=float, default=0.5, help="V/angstrom, along x")
    ap.add_argument("--friction", type=float, default=0.5,
                    help="raise this with the field; see the module docstring")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--structure", default="water_192.xyz")
    ap.add_argument("--index", type=int, default=-1)
    a = ap.parse_args()

    atoms = read_structure(a.structure, a.index)
    ev = build_evaluator(a.model, atoms, efield=[a.efield, 0.0, 0.0])
    th = Thermo(
        ev, atoms.get_positions(), timestep=a.dt, temperature=a.temperature,
        columns=("step", "time", "temp", "epot"),
        every=max(a.steps // 10, 1),
    )
    friction = th.scalar(a.friction)

    th.evaluate()
    print(f"NVT + field  {th.n_atoms} atoms  E_x {a.efield} V/A  "
          f"T {a.temperature} K  friction {a.friction}")
    for step in range(a.steps + 1):
        Z = ev.results["born_charge"]
        th.log(step, cos_x=alignment(th.atoms()), maxZ=float(np.abs(Z).max()))
        if step == a.steps:
            break
        langevin_baoab_half_step(th.positions, th.velocities, th.forces,
                                 th.masses, th.dt, th.kT, friction,
                                 a.seed + step)
        th.evaluate()
        langevin_baoab_finalize(th.velocities, th.forces, th.masses, th.dt)

    half = len(th.rows) // 2
    rows = th.rows[half:]
    print(f"\n  second half: <cos_x> {sum(r[4] for r in rows)/len(rows):+.4f}, "
          f"T {sum(r[2] for r in rows)/len(rows):.1f} K")
    net = np.abs(ev.results["field_force"].sum(axis=0)).max()
    print(f"  net field force {net:.2e} eV/A  (zero for a neutral cell)")
    print("  If maxZ climbs above a few, the field is distorting the structure")
    print("  into configurations the model never saw -- raise the friction or")
    print("  lower the field.")


if __name__ == "__main__":
    main()
