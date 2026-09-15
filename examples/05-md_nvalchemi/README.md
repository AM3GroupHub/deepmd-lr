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

Each builds its own jittered water lattice, so no data files are needed. Pass
`--n-side` to change the size (6 → 648 atoms, 8 → 1536).

## The scripts

| | ensemble | thermostat / barostat | use it for |
|---|---|---|---|
| [`01_nve.py`](01_nve.py) | NVE | — | **checking the model first.** Energy drift is a direct test that forces are the gradient of the energy |
| [`02_nvt_langevin.py`](02_nvt_langevin.py) | NVT | Langevin, BAOAB | canonical sampling when dynamics do not matter |
| [`03_nvt_nose_hoover.py`](03_nvt_nose_hoover.py) | NVT | Nosé–Hoover chain | transport, spectra — anything where dynamics matter |
| [`04_nvt_rescale.py`](04_nvt_rescale.py) | — | velocity rescaling | equilibration only, never production |
| [`05_npt_mtk.py`](05_npt_mtk.py) | NPT | MTK, isotropic | equilibrating density; needs the virial |
| [`06_nvt_efield.py`](06_nvt_efield.py) | NVT | Langevin | response to an external electric field |

`common.py` holds the shared setup: warp interop, units, the water box, the
evaluator, and a small reporter.

## Units, and the one that bites

The model works in **angstrom, eV, amu and ASE time units**, which are mutually
consistent — no conversion factors appear in any loop.

**Temperature is `k_B T` in energy units, not kelvin.** Every nvalchemiops
dynamics entry point takes it that way, which is the *opposite* of the heavy
`nvalchemi-toolkit`, whose thermostats took kelvin. Pass kelvin where `kT` is
expected and you thermostat to ~11,600× the temperature you meant; pass `kT`
where kelvin is expected and you get a system at rest. `common.kt(300.0)` does
the conversion.

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
  from warp — `wp.vec9d` does not exist.

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
partitioning is built against it. NPT moves the cell every step, so `05` pushes
it back in and invalidates the partitioning:

```python
ev.cell = new_cell
ev._partitions = None      # force a repartition against it
```

That re-runs `partition_system` only. **Do not rebuild the evaluator** — that
deepcopies the model onto every device.

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

## What these runs are not

Each script starts from a jittered lattice, so the first few hundred fs are
equilibration rather than sampling: the artificial structure dumps potential
energy into heat faster than a weak thermostat removes it. The defaults are
sized to finish in a minute or two, not to converge anything. Start production
runs from an equilibrated structure and expect the barostat in `05` to need
many times `tau_P` before the volume settles.
