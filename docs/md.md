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
use the classes in `deepmd_lr.nvalchemi`: on one GPU the nodes are the atoms,
the neighbour list is built on the device every step, and the only thing a
step downloads is the energy (with the virial and a few status flags). [Partitioned MD](partitioned-md.md#the-md-classes)
shows the workflow and class API; [example 06](../examples/06-md_nvalchemi)
has a script per ensemble. [Example 07](../examples/07-ase-calculator)
covers the ASE calculator.

## Switching small charges off: `charge_filter`

A charge head can leave a small charge on every atom of a bulk material:
+2e-4 e per C and -6e-4 e per Si in a C/Si model. Each is negligible, but in a
slab periodic in x and y every atomic layer acts as an infinite charged sheet,
and their fields add up instead of cancelling. The separated charge, and with
it the potential drop across the slab, grows with the thickness. On a 32 nm
C/Si cell the drop reached 80-280 V, and the charge-response forces broke MD
within a picosecond.

`charge_filter=(c_t, delta)` removes such charge in simulation. After the
usual projection, every charge with `|q| <= c_t - delta` is set exactly to
zero, every charge with `|q| >= c_t + delta` is kept, and a C^3 polynomial
joins the two, so forces and the Hessian stay continuous. The total charge is
then restored on the atoms that keep charge, in proportion to their weight,
so an atom switched off stays at zero:

```python
ev = build_evaluator("model.ckpt.pt", atoms, charge_filter=(3.5e-3, 1.5e-3))
atoms.calc = LongRangeCalculator("model.ckpt.pt", charge_filter=(3.5e-3, 1.5e-3))
```

Measured on that C/Si model (`ener_lr`, `pref_qn` 100) with (3.5e-3, 1.5e-3):

| cell | filter | bulk q per C / Si (e) | charge below the interface | E_lr per atom | 2 ps+ MD at 300 K |
|---|---|---|---:|---:|---|
| 11 nm | off | +2.1e-4 / -6.1e-4 | 2.19 e | 0.63 meV | stable |
| 32 nm | off | +1.9e-4 / -6.2e-4 | 5.25 e | 6.33 meV | blew up near 0.8 ps |
| 11 nm | on | 0 / 0 | 0.47 e | 0.056 meV | |
| 32 nm | on | 0 / 0 | 0.47 e | 0.061 meV | stable for 5 ps |

- **It is a simulation setting, not part of the model.** It is never saved,
  and training refuses it. The model was trained without it, so it changes
  the interface electrostatics the model saw: on the 11 nm cell the forces
  moved by 0.09 eV/A rms at the interface and 0.06 at the surfaces (bulk
  0.003). Validate it on a cell where the unfiltered model is trusted, for
  example by comparing the property you are after with and without it.
- **It can change that property.** On the 11 nm cell, four NEMD seeds each
  way gave an interfacial conductance of 131 +- 7 MW m^-2 K^-1 without the
  filter and 106 +- 8 with it: the same temperature jump, 20 % less heat
  flux. Switching off most of the interfacial charge changed how the
  interface carries heat. So there it stabilises the long cells but is not
  fit for a published conductance; the fix belongs in the model.
- **Choose `c_t` from the bulk charges at the temperature you run.** At
  300 K the C/Si bulk charges reached 2-2.5e-3 e, so the switch-off point
  `c_t - delta` sits at 2e-3. Too low, and thermal fluctuations switch bulk
  atoms on with the bulk's bias; too high, and most of the interfacial
  charge goes too. A model whose bulk charges sit inside the transition
  (the same C/Si model before `ener_lr`, at -2.5e-3 e on Si) is not cleaned
  by it.
- **Born charges and fields are refused with it** for now: the Born tensor
  is built from the unfiltered charges.

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
  With fixed atoms the count is three per mobile atom, and nothing is
  removed. `Thermo.ndof` holds whichever count applies, and a thermostat of
  your own must use it.
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
