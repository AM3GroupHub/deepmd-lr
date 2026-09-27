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
from ase import units

from deepmd_lr.nvalchemi import (
    Langevin,
    build_evaluator,
    maxwell_boltzmann,
    read_structure,
)


def alignment(p, oxy, hyd, L) -> float:
    """Mean projection of each molecule's HOH bisector on +x.

    0 for an isotropic liquid, 1 for perfect alignment -- the observable a
    field run is for. Minimum image assumes an orthorhombic cell.

    Takes raw positions rather than an ASE `Atoms`, and sums with `add.at`
    rather than a per-molecule loop: this runs every step, and built the
    long way it cost more than the MD it was measuring.
    """
    d = p[hyd][:, None, :] - p[oxy][None, :, :]
    d -= np.round(d / L) * L
    own = np.argmin((d**2).sum(-1), axis=1)
    mu = np.zeros((len(oxy), 3))
    np.add.at(mu, own, d[np.arange(len(hyd)), own])
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
    maxwell_boltzmann(atoms, temperature_K=a.temperature, seed=a.seed)
    sym = np.array(atoms.get_chemical_symbols())
    oxy, hyd = np.where(sym == "O")[0], np.where(sym == "H")[0]
    lengths = np.diag(np.asarray(atoms.get_cell()))
    md = Langevin(
        ev, atoms, a.dt * units.fs, temperature_K=a.temperature, friction=a.friction,
        seed=a.seed, logfile="-", loginterval=max(a.steps // 20, 1),
        extra_columns={
            "cos_x": lambda m: alignment(m.thermo.positions.numpy(), oxy, hyd, lengths),
            "maxZ": lambda m: float(np.abs(m.results["born_charge"]).max()),
        })
    print(f"NVT + field  {len(atoms)} atoms  E_x {a.efield} V/A  "
          f"T {a.temperature} K  friction {a.friction}")
    md.run(a.steps)

    half = md.rows[len(md.rows) // 2:]
    print(f"\n  {md.performance()}")
    print(f"  second half: <cos_x> {np.mean([r['cos_x'] for r in half]):+.4f}, "
          f"T {np.mean([r['T[K]'] for r in half]):.1f} K")
    net = np.abs(ev.results["field_force"].sum(axis=0)).max()
    print(f"  net field force {net:.2e} eV/A  (zero for a neutral cell)")
    print("  If maxZ climbs above a few, the field is distorting the structure")
    print("  into configurations the model never saw -- raise the friction or")
    print("  lower the field.")


if __name__ == "__main__":
    main()
