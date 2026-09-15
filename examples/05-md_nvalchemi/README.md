# 5 — MD with nvalchemiops

nvalchemiops ships integrator **kernels** and no driver, so the loop belongs to
you. That is the point of these six scripts: each is a complete, runnable loop
to copy, not a wrapper hiding one.

Every script was run to produce the numbers below.

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
usually the most equilibrated. **A cell is required** — these are periodic
simulations, and a structure without one is refused rather than quietly
treated as a molecule in vacuum.

## The scripts

| | ensemble | thermostat / barostat | use it for |
|---|---|---|---|
| [`01_nve.py`](01_nve.py) | NVE | — | **checking the model first.** Energy drift is a direct test that forces are the gradient of the energy |
| [`02_nvt_langevin.py`](02_nvt_langevin.py) | NVT | Langevin, BAOAB | canonical sampling when dynamics do not matter |
| [`03_nvt_nose_hoover.py`](03_nvt_nose_hoover.py) | NVT | Nosé–Hoover chain | transport, spectra — anything where dynamics matter |
| [`04_nvt_rescale.py`](04_nvt_rescale.py) | — | velocity rescaling | equilibration only, never production |
| [`05_npt_mtk.py`](05_npt_mtk.py) | NPT | MTK, isotropic | equilibrating density; needs the virial |
| [`06_nvt_efield.py`](06_nvt_efield.py) | NVT | Langevin | response to an external electric field |

There is no `common.py`: the shared setup now lives in the package itself, so
these scripts demonstrate the API you would actually use rather than a helper
file sitting beside them.

```python
from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure

atoms = read_structure("water_192.xyz")
ev = build_evaluator("model.ckpt.pt", atoms)
th = Thermo(ev, atoms.get_positions(), timestep=0.5, temperature=300.0)
```

`Thermo` **owns the state** the integrator kernels mutate — `th.positions`,
`th.velocities`, `th.forces`, `th.masses`, `th.cell`, `th.dt` are warp arrays,
bound once — and reports the thermodynamics, which nvalchemiops does not do at
all. It deliberately does *not* own the ensemble: thermostat chains and
barostat masses stay in these scripts, which is what lets one class serve all
six.

Under the hood it calls nvalchemiops' own utilities rather than reimplementing
them — `initialize_velocities` for the Maxwell–Boltzmann draw (COM motion
removed on the GPU, verified to 1e-14), `compute_kinetic_energy` /
`compute_temperature` for the measurements, `compute_pressure_tensor` for the
pressure. `compute_temperature` returns `k_B T` in energy units and already
divides by `3N − 3` rather than `3N`, assuming the COM motion was removed,
which `initialize_velocities` does. Both conventions were checked against the
kernel's output rather than assumed.

## Units, and the one that bites

The model works in **angstrom, eV, amu and ASE time units**, which are mutually
consistent — no conversion factors appear in any loop.

**Temperature is `k_B T` in energy units, not kelvin.** Every nvalchemiops
dynamics entry point takes it that way, which is the *opposite* of the heavy
`nvalchemi-toolkit`, whose thermostats took kelvin. Pass kelvin where `kT` is
expected and you thermostat to ~11,600× the temperature you meant; pass `kT`
where kelvin is expected and you get a system at rest.

The defence is structural rather than a helper function: `Thermo` takes
`temperature` in **kelvin** and builds `th.kT` itself, so no script has any
reason to write the conversion. Hand a kernel `th.kT` and it is right by
construction. `th.temperature` reads back in kelvin — a different name, a
different type, a different role, so the two cannot be swapped at a call
site.

Time is the other one. One ASE time unit is **10.181 fs**, and `friction` is in
*inverse* ASE time units, so the relaxation time is `1/friction` of those:

| friction | relaxation time | |
|---:|---:|---|
| 0.01 | 1018 fs | too weak to equilibrate a short run |
| 0.1 | 102 fs | the default in `02` |
| 0.5 | 20 fs | very strong; use under a field |

## Three API conventions you cannot guess

Found by running into them:

- **`ndof` dtypes disagree.** `nhc_compute_masses` wants **int32**;
  `nhc_thermostat_chain_update` wants **float64**. That is a Warp launch error,
  not a cast, so `03` builds both.
- **Half-step factors are inconsistent between kernels.**
  `nhc_velocity_half_step` and `velocity_verlet_position_update` apply their own
  factor of ½; `nhc_position_update` does not. So the **full `dt`** goes to
  every call and each kernel applies its own convention.
- **The tensor kernels want a flat `vec9` layout**, and `kinetic_tensors` is
  2-D `(B, 9)`. Only `cells` is `mat33`. `vec9d` comes from nvalchemiops, not
  from warp — `wp.vec9d` does not exist. `Thermo.update_pressure` owns this
  mess so no script has to.

The NHC chain update is also called once at **each end** of the step, not once
per step.

## Choosing a thermostat

`04` (rescaling) samples no correct ensemble and destroys the dynamics. Use it
to drag a bad structure to temperature, then switch.

Between the two real ones, the tradeoff is measured. Nosé–Hoover's conserved
quantity — the MTK invariant, which `03` reports — drifts, and the drift scales
with coupling strength (375-atom water, 400 steps):

| `tau` | mean T reached | invariant drift |
|---:|---:|---:|
| 100 fs | 420 K | 111 meV/atom/ps |
| 500 fs | 487 K | 14.4 meV/atom/ps |
| 2000 fs | 498 K | 0.94 meV/atom/ps |

The drift is **independent of timestep**, so it is not integration error. Weak
coupling is the safe regime — but weak coupling also thermostats slowly, which
is the whole tension. Langevin has no such quantity to watch and no such
problem; judge it by the temperature it holds. **NVE is the trustworthy
conservation check**, which is why `01` exists.

## NPT moves the cell, and the evaluator has to be told

`DistributedEvaluator` takes the cell at construction, because the spatial
partitioning is built against it. NPT moves the cell every step, so `05` hands
it back after the barostat has run:

```python
npt_cell_update(th.cell, cell_v, th.dt)
th.set_cell()              # device cell -> evaluator, repartition, new volume
```

This is **correctness, not caching**. The Verlet criterion that decides when to
rebuild watches *atom displacement only* and cannot see the cell move, so a
stale halo puts ghost images at the wrong offsets and the forces are quietly
wrong with nothing raising. `set_cell` therefore invalidates unconditionally.
It re-runs `partition_system` only — **do not rebuild the evaluator**, which
would deepcopy the model onto every device.

NPT also needs `compute_virial=True`, since the barostat is driven by the
pressure tensor. That costs essentially nothing: DeePMD-kit's lower graph
already produces the virial alongside the force (measured at under 1% of a
step).

For **NPH**, drop the two `npt_thermostat_half_step` calls from `05` and use
the `nph_*` kernels, which take the same arguments without the thermostat
state.

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

Every script takes its device list through `common.build_evaluator`, so
partitioning across cards is one argument:

```python
ev, masses = build_evaluator(model, symbols, cell,
                             devices=["cuda:0", "cuda:1"])
```

Worth it above a few thousand atoms per card and not below — see the
[main README](../../README.md#limits) for the measured scaling.

## Throughput

Every script ends with a throughput line:

```
  192 atoms, dt 0.5 fs: 32.7 ms/step -> 1.320 ns/day
```

`ns/day` scales with the timestep, so it is a statement about *this run's* `dt`
and not about the model alone — at 1 fs the same model doubles it.

The first logged interval is excluded, because it carries kernel compilation:
~1 s against a ~30 ms step here. Fitting through that point is not enough — a
least-squares slope over all points still lands 2× high — so it is dropped
outright, as the benchmarks in `experiments/` do.

It measures the loop as it actually ran, `log` included. Logging every step
means a device synchronisation every step, and any analysis you pass as a
keyword runs every step too. Measured on this machine, 192 atoms at 0.5 fs:

| | ms/step | ns/day |
|---|---:|---:|
| NVE, NVT (all three thermostats) | 32.7–33.8 | ~1.29 |
| NPT | 38.3 | 1.13 |
| NVT + field | 44.9 | 0.96 |

NPT pays for repartitioning every step as the cell moves; the field pays for
the extra backward pass Born charges need. Both match the costs measured in
the main README's benchmarks.

## What these runs are not

The defaults are sized to finish in a minute or two, not to converge anything.
`water_192.xyz` is equilibrated for the example-01 model, so the temperature
settles quickly — but start from *your* equilibrated structure for anything
real, and expect the barostat in `05` to need many times `tau_P` before the
volume stops moving. A structure that is not equilibrated for the model you
are using will dump potential energy into heat faster than a weak thermostat
removes it, which looks like a thermostat problem and is not one.
