"""deepmd-lr through ASE: one calculator for everything ASE can drive.

`LongRangeCalculator` is an ordinary ASE calculator for a `dp_lr` model, the
long-range term included, so single points, ASE's optimizers and integrators,
NEB or phonon workflows take it as they take any other calculator. It
evaluates the whole system on one device: the right tool for small systems and
for ASE-based workflows. For MD at scale, on one or several GPUs, use the
classes in ../06-md_nvalchemi -- the same model, stepped without leaving the GPU.

In order, this script runs:
  1. a single point: the energy with its short- and long-range parts, forces,
     stress and atomic charges;
  2. Born effective charges, which the calculator reports when it is built with
     a field; a zero field changes no force;
  3. a relaxation with ASE's FIRE;
  4. NVT with ASE's Langevin.

    python ase_calculator.py --model ../../path/to/model.ckpt.pt
    python ase_calculator.py --model ... --compress   # needs the CUDA operators
"""

import argparse

import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary
from ase.optimize import FIRE

try:  # ASE >= 3.29; the older name still works there, with a warning
    from ase.md.velocitydistribution import thermalize_momenta
except ImportError:
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution as thermalize_momenta,
    )

from deepmd_lr.ase import LongRangeCalculator


def per_element(values, symbols) -> str:
    return ", ".join(f"{el} {values[symbols == el].mean():+.3f}"
                     for el in sorted(set(symbols)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--structure", default="../06-md_nvalchemi/water_192.xyz")
    ap.add_argument("--compress", action="store_true",
                    help="the compressed DPA4C kernel; needs deepmd-kit built "
                         "from source")
    ap.add_argument("--eps-inf", type=float, default=1.0,
                    help="report charges screened by sqrt(eps_inf); energies and "
                         "forces do not change")
    ap.add_argument("--fmax", type=float, default=0.05,
                    help="eV/A, the relaxation target")
    ap.add_argument("--md-steps", type=int, default=200)
    a = ap.parse_args()

    atoms = read(a.structure)
    atoms.calc = LongRangeCalculator(a.model, compress=a.compress, eps_inf=a.eps_inf)
    symbols = np.array(atoms.get_chemical_symbols())

    # 1. A single point. The long-range split rides in `results`.
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)             # eV/A^3, ASE's sign
    charges = atoms.get_charges()
    r = atoms.calc.results
    print(f"1. {len(atoms)} atoms: energy {energy:.4f} eV = short-range "
          f"{r['energy_short_range']:.4f} + long-range {r['energy_long_range']:.4f}")
    print(f"   max |F| {np.abs(forces).max():.3f} eV/A, static pressure "
          f"{-np.trace(stress) / 3 / units.GPa:+.3f} GPa")
    print(f"   mean charge {per_element(charges, symbols)} e; "
          f"total {charges.sum():+.1e} e")

    # 2. Born effective charges. A field switches them on; a zero one moves no
    #    atom. Under a field the stress is withheld: its share is not derived.
    field = LongRangeCalculator(a.model, compress=a.compress, eps_inf=a.eps_inf,
                                efield=[0.0, 0.0, 0.0])
    probe = atoms.copy()
    probe.calc = field
    probe.get_forces()
    z = field.results["born_charge"]    # (N, 3, 3): [atom, polarization, displacement]
    iso = np.trace(z, axis1=1, axis2=2) / 3
    print(f"2. isotropic Born charge {per_element(iso, symbols)} e; "
          f"acoustic sum rule |sum_i Z_i| {np.abs(z.sum(axis=0)).max():.1e} e")

    # 3. A relaxation, with any ASE optimizer.
    relaxed = atoms.copy()
    relaxed.calc = atoms.calc
    opt = FIRE(relaxed, logfile=None)
    converged = opt.run(fmax=a.fmax, steps=500)
    print(f"3. FIRE {'converged' if converged else 'did NOT converge'} in {opt.nsteps} "
          f"steps: energy {relaxed.get_potential_energy() - energy:+.3f} eV, "
          f"max |F| {np.abs(relaxed.get_forces()).max():.3f} eV/A")

    # 4. MD, with any ASE integrator. FixCom holds the centre of mass, which is
    #    what Langevin's own fixcm=True does less exactly.
    atoms.set_constraint(FixCom())
    thermalize_momenta(atoms, temperature_K=300, rng=np.random.default_rng(1))
    Stationary(atoms)
    dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.01 / units.fs,
                   fixcm=False, rng=np.random.default_rng(2))
    temperatures = []
    dyn.attach(lambda: temperatures.append(atoms.get_temperature()), interval=10)
    dyn.run(a.md_steps)
    print(f"4. Langevin NVT, {a.md_steps} steps of 0.5 fs: T {temperatures[0]:.0f} -> "
          f"{temperatures[-1]:.0f} K")


if __name__ == "__main__":
    main()
