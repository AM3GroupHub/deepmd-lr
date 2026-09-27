# Running MD

> **Do not `dp freeze` a model with a `long_range` block.** It is refused. The
> exported artifact is traced from the graph lower, which receives edge vectors
> and no cell, so it cannot evaluate the reciprocal-space term. Before this was
> guarded, `dp freeze` succeeded silently and the frozen model returned
> **−93.74 eV** where the eager model gives **−55.12** — the entire 38.6 eV
> long-range contribution gone, with no warning. Freezing a short-range-only
> model (no `long_range` block) still works.
>
> The same refusal covers `DeepPot`, `deepmd.calculator.DP` and `dp test` on
> a checkpoint, and training with `enable_compile`: all of them call the graph
> lower directly and would report the short-range energy alone.

Use the ASE calculator, which drives the model's full `forward` and therefore
has the cell:

```python
from ase.md.verlet import VelocityVerlet
from ase import units
from deepmd_lr.ase import LongRangeCalculator

atoms.calc = LongRangeCalculator("run_joint/model.ckpt.pt")
VelocityVerlet(atoms, timestep=0.5 * units.fs).run(10000)
```

### Check energy conservation before trusting a trajectory

Measured on the water model (192 atoms, 340 K):

| timestep | drift |
|---------:|------:|
| 1.00 fs  | +10.7 meV/atom/ps |
| 0.50 fs  | −0.40 meV/atom/ps |
| 0.25 fs  | −0.005 meV/atom/ps |

What matters is that drift **vanishes as the timestep shrinks**. A force that
is not the exact gradient of the reported energy leaves a floor no timestep
reduction removes; here the drift falls by ~2000× from 1.0 to 0.25 fs. (The
1 fs figure is the usual story for unconstrained water — O–H stretches need a
short step — and is nothing to do with the long-range term.)


## On the GPU

The calculator above runs on one device, and every step copies the positions
to the host and back. For MD that never leaves the GPU, and for several GPUs,
the classes in `deepmd_lr.nvalchemi` follow the same ASE pattern on
nvalchemiops kernels:

```python
from ase import units
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from deepmd_lr.nvalchemi import FIRE2, NoseHooverChainNVT, build_evaluator

ev = build_evaluator("run_joint/model.ckpt.pt", atoms, devices=["cuda:0"])
FIRE2(ev, atoms).run(fmax=0.05)
thermalize_momenta(atoms, temperature_K=300)
Stationary(atoms)
NoseHooverChainNVT(ev, atoms, 0.5 * units.fs, temperature_K=300,
                   tdamp=100 * units.fs, logfile="md.log",
                   loginterval=100).run(20_000)
```

`VelocityVerlet` (NVE); `Langevin`, `NoseHooverChainNVT` and
`VelocityRescaling` (NVT); `MTKNPT` and `IsotropicMTKNPT` (NPT); `MTKNPH` and
`IsotropicMTKNPH` (NPH); and the relaxations `FIRE` and `FIRE2` share that one
evaluator, so a relaxation and several MD stages run on one copy of the
model. Between them they cover every integrator nvalchemiops ships. [Partitioned MD](partitioned-md.md#the-md-classes)
covers the API and what each piece owns.
[Example 06](../examples/06-md_nvalchemi) has a script per ensemble, and
[example 07](../examples/07-ase-calculator) covers the ASE calculator.

## Traps

Each of these has cost a run.

- **A freshly packed box collapses.** The model has no repulsive wall below
  the shortest contacts it was trained on, so a minimiser, or the first MD
  steps, can pull clashing atoms together instead of pushing them apart.
  `FIRE` and `FIRE2` stop with a `RelaxationError` once forces grow tenfold
  or more than 2 eV/atom is released. Pre-relax with a classical force field
  ([below](#relaxing-a-packed-box-first)).
- **Check conservation before trusting a model**, with NVE: the drift has to
  vanish as dt shrinks ([above](#check-energy-conservation-before-trusting-a-trajectory)).
  Langevin conserves nothing, and under a field nothing is conserved at all.
- **Kelvin where `kT` is expected.** nvalchemiops' kernels take `k_B T` in eV.
  The classes take `temperature_K` and convert; a custom loop that passes
  kelvin thermostats to ~11,600× the target.
- **`friction` is per ASE time unit**, which is 10.18 fs: `friction=0.01`
  relaxes in 1 ps, and `friction=0.01 / units.fs` in 100 fs.
- **The full `dt` goes to every kernel** in a custom loop; each applies its own
  factor of ½. Passing `dt/2` to the Nosé–Hoover kick halved every force:
  frequencies came out √2 low, and the configurations sampled ~600 K while the
  kinetic temperature read 300 K.
- **The Nosé–Hoover invariant needs `eta / 2`.** With the kernel's own `eta`
  it climbs by the same amount at every timestep, which reads as an integrator
  fault and is only bookkeeping
  ([measured](partitioned-md.md#what-is-conserved)).
- **Temperatures count `3N − 3` degrees of freedom**, which assumes the
  centre of mass is at rest. The classes remove its motion at start-up (with
  a warning when it was large), and Langevin's `fixcm` keeps it at rest. A
  custom loop has to do the same, or the drift's kinetic energy reads as heat.
- **NPT needs the virial.** Reach the density with NPT, then produce with
  NVT. NPT under a field is refused. A cell
  relaxation (`FIRE2(..., relax_cell="isotropic")`) reaches the zero-kelvin
  density far faster, and needs the virial too.
- **Relax a liquid's cell isotropically.** A liquid has no shear stiffness, so
  `relax_cell="full"` lets its cell shear; keep that for crystals.
- **An MD object built before an NPT stage refuses to step** once that stage
  has moved the shared evaluator's cell. Build a new one from the stage's
  `atoms`.
- **Unwrapped positions are fine.** Integrators never wrap, trajectories are
  written unwrapped so diffusion can be read off them, and every halo rebuild
  folds the atoms into the cell itself. Before that fold, water that had
  diffused a few picoseconds lost neighbours: 1.1 eV/Å force errors, with
  nothing raised.

## Relaxing a packed box first

A box from a packing tool starts with clashes. A packed 64-water box here had
O–O contacts down to 1.56 Å, and FIRE2 with the model diverged from it: the
molecules slid into the short-range hole. A classical force field has the
wall, so push the molecules apart with rigid TIP3P first, then refine with the
model:

```python
import numpy as np
from ase import Atoms
from ase.calculators.tip3p import TIP3P, angleHOH, rOH
from ase.constraints import FixBondLengths
from ase.geometry import get_distances
from ase.optimize import FIRE as AseFIRE


def tip3p_prerelax(atoms, fmax=0.05, steps=5000):
    """Push a packed water box apart with rigid TIP3P. Positions only; the
    cell and the atom order are kept."""
    sym = np.array(atoms.get_chemical_symbols())
    O, H = np.flatnonzero(sym == "O"), np.flatnonzero(sym == "H")
    owner = get_distances(atoms.positions[H], atoms.positions[O],
                          cell=atoms.cell, pbc=True)[1].argmin(axis=1)
    if not (np.bincount(owner, minlength=len(O)) == 2).all():
        raise ValueError("every O must own exactly two H")
    order, pos = [], []
    for k, o in enumerate(O):          # TIP3P wants O,H,H triplets, each H
        hs = H[owner == k]             # in the minimum image of its own O
        oh = get_distances(atoms.positions[o], atoms.positions[hs],
                           cell=atoms.cell, pbc=True)[0][0]
        order += [o, *hs]
        pos += [atoms.positions[o], *(atoms.positions[o] + oh)]
    w = Atoms("OH2" * len(O), positions=pos, cell=atoms.cell, pbc=True)
    hh = 2 * rOH * np.sin(np.radians(angleHOH) / 2)
    pairs = [p for i in range(0, len(w), 3)
             for p in ((i, i + 1), (i, i + 2), (i + 1, i + 2))]
    w.set_constraint(FixBondLengths(pairs, bondlengths=[rOH, rOH, hh] * len(O)))
    w.calc = TIP3P(rc=min(atoms.cell.lengths()) / 2 - 0.1)
    AseFIRE(w, logfile=None).run(fmax=fmax, steps=steps)
    out = atoms.copy()
    out.positions[order] = w.positions
    return out


atoms = tip3p_prerelax(atoms)
FIRE2(ev, atoms).run(fmax=0.05)            # the model's own minimum, on the GPU
```

{{TIP3P}}
