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

**Who owns what.**
- **The evaluator** holds the model on every device and is expensive to build,
  so it is built once. A FIRE relaxation, an NVT stage and an NPT stage can run
  one after another on the same one: `md.evaluator` *is* the object you built.
- **Each MD object builds a `Thermo`**: the device state the nvalchemiops
  kernels mutate (positions, velocities, masses, the forces view, the cell),
  plus the measurements (`md.thermo.temperature`, `.pressure`, ...).
- **The MD object owns time and the ensemble**: the timestep, target
  temperature, friction, time constants, pressure, chain state, the log and
  the step count.

Under the hood `Thermo` calls nvalchemiops' own utilities rather than
reimplementing them:
- `remove_com_motion` handles the centre of mass on the GPU;
- `compute_kinetic_energy` gives the kinetic energy, and the temperature is
  `2 KE / (ndof k_B)`;
- `compute_pressure_tensor` gives the pressure.

`Thermo.ndof` is `3N − 3`, and every thermostat here counts the same. That
count assumes the centre of mass is at rest, so the MD classes remove its
motion at start-up; Langevin's `fixcm=True` keeps it at rest. With fixed atoms
it is three per mobile atom instead ([below](#fixed-atoms)).

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

## Units, and the one that bites

The model works in **angstrom, eV, amu and ASE time units**, which are mutually
consistent — no conversion factors appear in any step.

**Temperature is `k_B T` in energy units, not kelvin**, at every nvalchemiops
kernel. Pass kelvin where `kT` is expected and you thermostat to ~11,600× the
temperature you meant. The classes take `temperature_K` in kelvin and build
`kT` themselves; `md.thermo.temperature` reads back in kelvin. In a custom loop,
`kT = th.scalar(units.kB * T)` is the one line where the conversion is
written.

Time is the other one. The `timestep` is in ASE time units (`0.5 * units.fs`),
one ASE time unit is **10.181 fs**, and `friction` is in *inverse* ASE time
units, so the relaxation time is `1/friction` of those:

| friction | relaxation time | |
|---:|---:|---|
| 0.01 | 1018 fs | too weak to equilibrate a short run |
| 0.1 | 102 fs | the default in `02` |
| 0.5 | 20 fs | very strong; use under a field |

## API conventions you cannot guess

They matter when you drive the kernels yourself, as `07` does; the classes handle
them. All were found by running into them:

- **`ndof` dtypes disagree.** `nhc_compute_masses` wants **int32**;
  `nhc_thermostat_chain_update` wants **float64**. That is a Warp launch error,
  not a cast.
- **Half-step factors are inconsistent between kernels.**
  `nhc_velocity_half_step` and `velocity_verlet_position_update` apply their own
  factor of ½; `nhc_position_update` does not. So the **full `dt`** goes to
  every call and each kernel applies its own convention.
- **The tensor kernels want a flat `vec9` layout**, and `kinetic_tensors` is
  2-D `(B, 9)`. Only `cells` is `mat33`. `vec9d` comes from nvalchemiops, not
  from warp — `wp.vec9d` does not exist. `Thermo.update_pressure` owns this
  mess so no script has to.
- **The NPT kernels count `3N` degrees of freedom** unless told otherwise:
  each computes `N_f = 3 × (the atom count passed)`. The centre of mass does
  not move, so pass `N − 1` to all three, as the NPT class does. Pass `N` and the
  system is held `3N/(3N−3)` hot.
- **Refresh right before the barostat reads.** `ke_w` and `press_w` hold
  whatever was last computed into them, so call `update_kinetic_energy()` and
  `update_pressure()` before each barostat/thermostat half-step, after the
  second kick too. `Thermo`'s getters never write them, so logging cannot
  change a run.

The NHC chain update is also called once at **each end** of the step, not once
per step, and its chain energy needs **eta/2**: each call scales the velocities
by a half step but moves eta by a full `eta_dot dt`, so eta runs twice as fast
as eta_dot implies.

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
  4x, at 20 fs as at 100 fs coupling.

An earlier version of this page reported an invariant drift independent of
the timestep (111 meV/atom/ps at `tau` 100 fs). It came from computing the
chain energy with the kernel's own eta, which runs twice as fast as eta_dot
implies and so counts the heat the thermostat removes twice. Measured on
water at `tau` 20 fs: with the kernel's eta the invariant climbs 9.95, 10.48
and 10.61 meV/atom over 100 fs at dt 0.5, 0.25 and 0.125 fs; with eta/2 it
fluctuates by 0.022, 0.0057 and 0.0014 meV/atom rms. The dynamics were never
affected.

**NVE is still the cleanest conservation check**, which is why `01` exists.

## NPT moves the cell, and the evaluator has to be told

With two or more devices `DistributedEvaluator` partitions the atoms against
its cell (on one device nothing is partitioned: the list searches the real
cell every step), and NPT moves the cell every step. So after the barostat runs, the NPT and NPH classes push
the new cell to the evaluator (`Thermo.set_cell`). The partitions keep their
nodes -- the same atoms and lattice images -- and re-express the images'
offsets in the new cell. The evaluator's Verlet bound counts the cell's change
as well as the atoms', so they rebuild only once either has moved enough.

This is **correctness, not caching**. A halo left as it was built for the old
cell would put ghost images at the wrong offsets, and the forces would be
quietly wrong with nothing raising. Until the bound counted the cell, the only
safe answer was to repartition every step: at 12,288 water atoms that made NPT
75 % slower than NVT. Now it costs about what NVT does. Reach the density with
NPT, then produce with NVT.

The barostat needs the virial, since it is driven by the pressure tensor. It
costs a strain leaf per partition and an extra target in the charge-response
backward. Build the evaluator with `compute_virial=True`, or call
`ev.set_compute_virial(True)` on one that already ran FIRE or NVT.

`MTKNPT` and `MTKNPH` take `cell=`, one of three things the cell may do:

| `cell=` | the cell |
|---|---|
| `"isotropic"` | scales uniformly |
| `"orthorhombic"` | the three lengths of a diagonal cell move independently |
| `"full"` (the default, as in ASE) | the whole cell fluctuates, without rotating |

A liquid has no shear stiffness, so `"full"` lets its cell shear; keep
`"isotropic"` for water.

nvalchemiops' cell update, `h + dt eps_dot h`, reads the columns of `h` as the
lattice vectors, where ASE's cell has them as rows, so the classes hand the
kernels the transpose. An isotropic barostat is indifferent to the
convention; the others are not. On a sheared cell, the untransposed cell broke
NPH's conservation outright: its error was the same at every timestep.

## Running under a field

`06` passes `efield` to the evaluator and every returned force already carries
`F_ib = E_a Z_iab`. Two things to know:

- **There is no conserved quantity.** The field force is a local response, not
  the gradient of a `−M·E` energy — under periodic boundaries `M` is not even
  defined. Do not read energy drift as an integrator problem.
- **Couple the thermostat to the field.** The field does work with nothing
  bounding it, so the thermostat is the only sink. At 2 V/Å, friction 0.01 runs
  away — Z grows on the distorted geometry, the force grows with it, and the
  feedback closes; friction 0.5 is stable.

The net field force is `Q·E`, exactly zero for a neutral cell, so nothing
drifts. `06` prints it as a check, along with `max|Z|`: if that climbs above a
few, the field is distorting the structure into configurations the model never
saw.

Note that `compute_virial` and `efield` **cannot be combined** — the field's
contribution to the stress is a separate derivation that does not exist yet, so
NPT under a field is refused rather than quietly wrong.

## Several GPUs

`build_evaluator` takes the device list, so partitioning across cards is one
argument. `06` uses every visible GPU unless told otherwise (`--gpus`):

```python
ev = build_evaluator(model, atoms, devices=["cuda:0", "cuda:1"])
```

Each device gets one spatial partition with a halo of ghost atoms, and the
forces, charges and Born tensors are assembled on the first device. Whether a
second card pays depends on where a step spends its time. NVT on water, 2x
A40, ms/step:

| atoms | uncompressed, 1 GPU | 2 GPUs | compressed, 1 GPU | 2 GPUs |
|---:|---:|---:|---:|---:|
| 12,288 | 184 | 145 | 14.8 | 18.4 |
| 41,472 | 574 | 399 | 36.2 | 36.5 |
| 98,304 | — | — | 75.9 | 72.3 |

Uncompressed, the model's work on the device dominates, and two cards help
from about ten thousand atoms. Compressed, the host's share dominates below
~100,000 atoms, and one card is as fast. More in
[partitioned MD](../../docs/partitioned-md.md#limits).

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
