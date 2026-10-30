# 6 — MD with nvalchemiops

Eight short scripts: one per ensemble, a field run on every visible GPU, and
a thermostat of your own. The MD classes in `deepmd_lr.nvalchemi` do the
stepping, ASE-style -- parameters in `__init__`, then `run(steps)` -- so each
script is mostly the physics worth knowing about its ensemble. Between them
the classes cover every integrator nvalchemiops ships;
[`07_custom_berendsen.py`](07_custom_berendsen.py) adds one of your own as a
subclass, and [example 08](../08-nemd) a larger one: Langevin baths on named
atom groups, with the heat each injects accounted on the device, for
thermal-conductance NEMD. Trajectories, observers and checkpoints are
arguments of every class (`trajectory=`, `attach()`, `checkpoint=` and
`Cls.from_checkpoint`).

## Prerequisite

A trained `dp_lr` checkpoint. [Example 01](../01-bulk) makes one; any of these
then works against it:

```bash
python 01_nve.py --model ../../path/to/model.ckpt.pt
```

The starting structure defaults to `water_192.xyz`, shipped here — one
equilibrated frame of 64 H₂O in a 12.429 Å cell, taken from example 01's
validation set, so it matches the model that example trains.

Any structure `ase.io.read` understands works instead:

```bash
python 02_nvt_langevin.py --model ... --structure my_system.cif
python 02_nvt_langevin.py --model ... --structure traj.xyz --index 250
```

`--index` picks the frame from a multi-frame file and defaults to the last,
usually the most equilibrated. **A cell is required** along every periodic
axis. A slab, whose `pbc` leaves one axis open, runs too: `read_structure` pads
that axis to at least 20 Å of vacuum (`vacuum=`) and centres the slab, and the
long-range sum takes its slab form. A molecule or cluster (`pbc` all false)
runs on one GPU with no cell: the long-range term is the direct isolated sum,
and NPT is refused ([non-periodic](../../docs/non-periodic.md#running-md)).

## The scripts

| | ensemble | thermostat / barostat | use it for |
|---|---|---|---|
| [`01_nve.py`](01_nve.py) | NVE | — | **checking the model first.** Energy conservation tests that forces are the gradient of the energy |
| [`02_nvt_langevin.py`](02_nvt_langevin.py) | NVT | Langevin, BAOAB | canonical sampling when dynamics do not matter |
| [`03_nvt_nose_hoover.py`](03_nvt_nose_hoover.py) | NVT | Nosé–Hoover chain | transport, spectra — anything where dynamics matter |
| [`04_nvt_rescale.py`](04_nvt_rescale.py) | — | velocity rescaling | equilibration only, never production |
| [`05_npt_mtk.py`](05_npt_mtk.py) | NPT | MTK + Nosé–Hoover chain; `--cell` isotropic, orthorhombic or full | equilibrating density; needs the virial |
| [`06_nvt_efield.py`](06_nvt_efield.py) | NVT | Langevin | response to an external electric field, on every visible GPU |
| [`07_custom_berendsen.py`](07_custom_berendsen.py) | NVT | Berendsen, your own | an integrator of your own, written as a subclass |
| [`08_nph_mtk.py`](08_nph_mtk.py) | NPH | MTK alone; `--cell` as for NPT | constant enthalpy: the check that the barostat is integrated right |

```python
from ase import units
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from deepmd_lr.nvalchemi import Langevin, build_evaluator, read_structure

atoms = read_structure("water_192.xyz")
ev = build_evaluator("model.ckpt.pt", atoms)       # built once, shared
thermalize_momenta(atoms, temperature_K=300)
Stationary(atoms)
md = Langevin(ev, atoms, 0.5 * units.fs, temperature_K=300, friction=0.1,
              logfile="-", loginterval=50)
md.run(1000)
```

The evaluator is built once and shared: a FIRE relaxation, an NVT stage and
an NPT stage can run one after another on it, while each MD object builds its
own `Thermo` (the device state the kernels mutate, plus the measurements) and
owns the timestep, ensemble parameters, log and step count
([sharing one evaluator](../../docs/partitioned-md.md#sharing-one-evaluator)).
Temperatures count `3N − 3` degrees of freedom, so the classes remove the
centre-of-mass motion at start-up; with fixed atoms the count is three per
mobile atom instead ([below](#fixed-atoms)).

## Fixed atoms

Attach ASE's `FixAtoms` to `atoms`, as for ASE's own MD. Every class here
honours it, and refuses any other constraint rather than ignoring it:

```python
from ase.constraints import FixAtoms

atoms = read_structure("slab.extxyz")                    # pbc T T F
z = atoms.positions[:, 2]
atoms.set_constraint(FixAtoms(mask=z < z.min() + 3.0))   # the bottom 3 A
ev = build_evaluator("model.ckpt.pt", atoms)
thermalize_momenta(atoms, temperature_K=300)             # fixed atoms stay at rest
Langevin(ev, atoms, 0.5 * units.fs, temperature_K=300, friction=0.1).run(1000)
```

- **They stay exactly where they start, at rest.** The kernels see zero force
  on them. Langevin's kernel gives every atom noise, so `Thermo.evaluate` also
  puts them back before every force call.
- **The temperature counts three degrees of freedom per mobile atom**, as
  ASE's `atoms.get_temperature()` does, and the thermostats target that same
  count. No centre-of-mass motion is removed, since the fixed atoms hold the
  system in place: skip `Stationary`, and Langevin's `fixcm` does nothing.
- **Output keeps the model's force on them.** `atoms.calc` and the trajectory
  hold it, and the trajectory carries the constraint too (`move_mask` in
  extxyz). `atoms.get_forces()` zeroes it, as ASE does, and
  `get_forces(apply_constraint=False)` returns it.
- **A structure file can carry them.** POSCAR's selective dynamics `F F F` and
  extxyz's `move_mask` read back as `FixAtoms`, and `read_structure` moves
  them with the slab when it centres one.
- **NPT, NPH and cell relaxation refuse them**, because the cell's strain moves
  every position. Reach the density first, then fix the atoms for NVT.

## Units

Angstrom, eV, amu and ASE time units, as in ASE: `timestep` is
`0.5 * units.fs`, `friction` is in *inverse* ASE time units (one is 10.181 fs,
so `friction=0.1` relaxes in 102 fs), and `temperature_K` is in kelvin
([units and friction](../../docs/partitioned-md.md#the-md-classes)). In a
custom loop, the nvalchemiops kernels take **`k_B T` in eV, not kelvin**:
`kT = th.scalar(units.kB * T)`, or you thermostat to ~11,600× the target.
`07` and [your own loop](../../docs/partitioned-md.md#your-own-loop) list the
other kernel conventions the classes handle for you.

## Choosing a thermostat

`04` (rescaling) samples no correct ensemble and destroys the dynamics. Use it
to drag a bad structure to temperature, then switch.

Between the two real ones, choose by purpose:
- **Langevin** (`02`) is robust and needs no state. Its noise and friction
  perturb the dynamics, so use it for sampling and equilibration. There is no
  conserved quantity; judge it by the temperature it holds.
- **Nosé–Hoover** (`03`) is deterministic and leaves the dynamics alone, so
  use it for diffusion and spectra. Its `Econs` column, the extended energy,
  is conserved to O(dt²) at any coupling: halving dt shrinks its fluctuation
  4x, at 20 fs as at 100 fs coupling
  ([what is conserved](../../docs/partitioned-md.md#what-is-conserved)).

**NVE is still the cleanest conservation check**, which is why `01` exists.

## NPT moves the cell

The NPT and NPH classes push the barostat's new cell to the evaluator every
step; with several partitions the halo is re-expressed in the new cell and
rebuilt once atoms or cell have moved enough
([reusing a partitioning](../../docs/partitioned-md.md#reusing-a-partitioning-across-steps)).
NPT costs a little more than NVT ([throughput](#throughput)). Reach the
density with NPT, then produce with NVT.

The barostat needs the virial: build the evaluator with
`compute_virial=True`, or call `ev.set_compute_virial(True)` on one that
already ran FIRE or NVT. `MTKNPT` and `MTKNPH` take `cell="isotropic"`,
`"orthorhombic"` or `"full"` (the default, as in ASE); keep `"isotropic"` for
a liquid such as water, which has no shear stiffness
([relaxing the cell](../../docs/partitioned-md.md#relaxing-the-cell)).

## Running under a field

`06` passes `efield` to the evaluator and every returned force already carries
`F_ib = E_a Z_iab`. There is no conserved quantity, no virial and so no NPT
under a field ([read this first](../../docs/born-charges.md#read-this-before-interpreting-a-field-run)).
What this script adds:

- **Couple the thermostat to the field.** The field does work with nothing
  bounding it, so the thermostat is the only sink. At 2 V/Å, friction 0.01 runs
  away — Z grows on the distorted geometry, the force grows with it, and the
  feedback closes; friction 0.5 is stable.
- **Check the net force and `max|Z|`.** `06` prints the net field force, `Q·E`,
  exactly zero for a neutral cell, and `max|Z|`: if that climbs above a few,
  the field is distorting the structure into configurations the model never
  saw.

## Several GPUs

`build_evaluator` takes the device list, so partitioning across cards is one
argument. `06` uses every visible GPU unless told otherwise (`--gpus`):

```python
ev = build_evaluator(model, atoms, devices=["cuda:0", "cuda:1"])
```

Each device gets one spatial partition with a halo of ghost atoms, and the
forces, charges and Born tensors are assembled on the first device. Whether a
second card pays depends on where a step spends its time: uncompressed, two
A40s help from about ten thousand water atoms; compressed, one card is as
fast below ~100,000 ([measurements](../../docs/partitioned-md.md#limits)).

## Throughput

Every script ends with `md.performance()`:

```
  192 atoms, dt 0.5 fs: 17.1 ms/step -> 2.526 ns/day
```

`ns/day` scales with the timestep, so it is a statement about *this run's* `dt`
and not about the model alone — at 1 fs the same model doubles it.

The first logged interval is excluded, because it carries kernel compilation:
~1 s against a ~17 ms step here. Fitting through that point is not enough — a
least-squares slope over all points still lands 2× high — so it is dropped
outright.

It measures the run as it actually went, the log included. Logging every step
means a device synchronisation every step, and any `extra_columns` or observer
runs at its own interval too. Measured on one GB10, 192 atoms at 0.5 fs, a
trained water model uncompressed, median of three runs (this machine is
shared, and single runs scatter by about 20 %):

| | ms/step | ns/day |
|---|---:|---:|
| NVE, NVT (Langevin, Nosé–Hoover, rescaling) | 16.8–17.5 | ~2.5 |
| NPT, NPH | 19.2–19.7 | ~2.2 |
| NVT + field | 20.7 | 2.09 |

NPT pays for the virial and the barostat; the field pays for the extra
backward pass Born charges need. Both match the costs in
[Cost](../../docs/cost.md).

## What these runs are not

The defaults are sized to finish in a minute or two, not to converge anything.
`water_192.xyz` is equilibrated for the example-01 model, so the temperature
settles quickly — but start from *your* equilibrated structure for anything
real, and expect the barostat in `05` to need many times `tau_P` before the
volume stops moving. A structure that is not equilibrated for the model you
are using will dump potential energy into heat faster than a weak thermostat
removes it, which looks like a thermostat problem and is not one.
