# Running MD over partitions

`LongRangeCalculator` is single-process and single-device. For larger systems,
`DistributedEvaluator` evaluates the model over spatial partitions — one per
GPU:

```python
from deepmd_lr.nvalchemi import DistributedEvaluator

evaluator = DistributedEvaluator(
    model, atype, cell,
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    compute_virial=False,       # True also fills .virial and .stress
)
energy, forces = evaluator(positions)      # eV, eV/angstrom
```

**That is the whole object.** It owns the partitioning, the ghost halo, the
per-device replicas and the decision of when to rebuild — and nothing else.
No timestep, no ensemble, no thermostat, no velocities: those belong to
whoever is integrating. Swapping NVE for NPT changes the integrator and not
this.

`positions` may be a torch tensor — including the one an integrator is
mutating in place — or anything `np.asarray` accepts. Forces come back as a
float64 tensor on the model's device, which `wp.from_torch` turns into a
zero-copy Warp view, so **the trajectory never leaves the GPU** for the
integrator's sake. `evaluator.results` carries the energy split and charges;
`evaluator.masses()` reads the element order off the model's own `type_map`.

**There is deliberately no ASE calculator in this path.** ASE's interface
returns energy and forces one configuration at a time and its integrators
cannot touch on-device state, which is the point of this path.

### The MD classes

The integrators follow ASE: the evaluator and an `ase.Atoms` come first, the
parameters go to `__init__`, and `run(steps)` advances. One evaluator carries
a whole workflow:

```python
from ase import units
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from deepmd_lr.nvalchemi import (
    FIRE2, IsotropicMTKNPT, NoseHooverChainNVT, build_evaluator, read_structure,
)

atoms = read_structure("start.xyz")
ev = build_evaluator("model.ckpt.pt", atoms, devices=["cuda:0", "cuda:1"])
FIRE2(ev, atoms).run(fmax=0.05)
thermalize_momenta(atoms, temperature_K=300)
Stationary(atoms)
NoseHooverChainNVT(ev, atoms, 0.5 * units.fs, temperature_K=300,
                   tdamp=100 * units.fs, logfile="nvt.log",
                   loginterval=100).run(20_000)
ev.set_compute_virial(True)                     # only NPT pays for the virial
npt = IsotropicMTKNPT(ev, atoms, 0.5 * units.fs, temperature_K=300,
                      pressure_au=1e-4 * units.GPa, tdamp=100 * units.fs,
                      pdamp=1000 * units.fs, logfile="npt.log",
                      trajectory="npt.traj", traj_interval=500)
npt.run(20_000)
```

| class | what | the `Econs` column |
|---|---|---|
| `VelocityVerlet` | NVE | Etot |
| `Langevin` | NVT, BAOAB; `fixcm=True` keeps the centre of mass at rest | none |
| `NoseHooverChainNVT` | NVT, Nosé–Hoover chain | Etot + the chain energy |
| `VelocityRescaling` | velocity Verlet, rescaled to the target whenever the temperature leaves a window; equilibration only | none |
| `MTKNPT` | NPT, MTK barostat + Nosé–Hoover chain; `cell="full"` (default), `"orthorhombic"` or `"isotropic"` | the MTK H′ |
| `IsotropicMTKNPT` | `MTKNPT` with an isotropic cell, under ASE's name | the MTK H′ |
| `MTKNPH` | NPH, the MTK barostat alone; `cell` as for `MTKNPT` | the enthalpy H′ |
| `IsotropicMTKNPH` | `MTKNPH` with an isotropic cell | the enthalpy H′ |
| `FIRE` | relaxes the positions, and the cell with `relax_cell`; classic FIRE with ASE's defaults | — |
| `FIRE2` | the same with FIRE 2.0 and nvalchemiops' more cautious defaults | — |

`FIRE` and `FIRE2` are ASE's two relaxations, each on its nvalchemiops kernel,
with ASE's parameter names and, as in ASE, unit masses. FIRE caps each atom's
step on its own. FIRE2 kicks before it measures the power, takes half a step
back after an uphill step, and scales the whole step so that no atom moves
more than `maxstep`. On water, from an MD snapshot, `FIRE` reached 0.05 eV/Å
in 199 steps and `FIRE2` in 380; from a TIP3P-relaxed packed box, in 149 and
160.

### Relaxing the cell

`relax_cell` chooses what else may move. The target is `pressure_au` (eV/Å³),
and `run(smax=...)` sets how close the stress must come to it; the default is
0.01 GPa.

| `relax_cell` | the cell | for |
|---|---|---|
| `None` (default) | fixed | — |
| `"isotropic"` | scales uniformly, keeping its shape | liquids, which have no shear stiffness: `"full"` lets their cell shear |
| `"orthorhombic"` | the three lengths of a diagonal cell move independently | orthorhombic crystals |
| `"full"` | all nine entries move | crystals of any symmetry |

```python
ev.set_compute_virial(True)                     # the stress drives the cell
FIRE2(ev, atoms, relax_cell="isotropic", pressure_au=1e-4 * units.GPa).run(fmax=0.05)
```

It follows nvalchemiops' variable-cell FIRE example (`07_fire_variable_cell`):
the cell's rows join the atoms in one extended array, which the FIRE kernel
steps like any other degrees of freedom. Three things differ, and each was
measured:

- **The cell force** is −∂(E + PV)/∂H = −V H⁻ᵀ (σ + P) for the rows of ASE's
  cell. nvalchemiops' `stress_to_cell_force` computes −V σ H⁻ᵀ, the gradient
  for a cell whose columns are the lattice vectors. The two agree for a
  diagonal cell only. On a sheared water cell, this one matched the
  finite-difference gradient to 2e-3 eV/Å and nvalchemiops' was 0.70 eV/Å off.
  With no alignment step, the structure is never rotated either.
- **The atoms ride the cell affinely** before their own step, as nvalchemiops'
  FIRE2 variable-cell step does; in the example they stand still while the cell
  moves.
- **The cell force is not divided by the atom count.** That makes the cell
  about as heavy as one atom, as the example's cell mass does.

On water stretched by 4%, the undivided force converged fastest for both
algorithms, at 192 and at 12,288 atoms alike (to 0.05 eV/Å and 0.01 GPa):

| steps | isotropic | orthorhombic | full |
|---|---:|---:|---:|
| `FIRE2` | 390 | 440 | 390 |
| `FIRE` | 750 | 610 | 740 |
| `FIRE2`, cell force ÷ N | 1000 | 1480 | 1570 |

The new cell goes to the evaluator every step, as under NPT; the partitions
follow it without a rebuild (below). An MD object built before the
relaxation then refuses to step, because the cell it knew has moved. Build it
from the relaxed `atoms`.

The units are exactly ASE's: `timestep` in ASE time (`0.5 * units.fs`),
`friction` in 1/ASE time, `tdamp` and `pdamp` in ASE time, `pressure_au` in
eV/Å³, `temperature_K` in kelvin.

**Who owns what.**

| object | built by | owns |
|---|---|---|
| `DistributedEvaluator` | you, once | the model replicas, the partitioning and skin, the forces buffer, the cell |
| `Thermo` | each MD object, in its `__init__` | positions, velocities, masses and the forces view, on the device; the measurements |
| the MD object | you | the timestep and ensemble parameters, the thermostat and barostat state, the step count, the log, the trajectory, observers, checkpoints |

`md.evaluator` is a read-only property returning `md.thermo.evaluator`, which
is the object you built. No class ever builds or copies an evaluator. Sharing
it between stages is safe by construction:

- **Forces.** The stages share one forces buffer. An object that another has
  evaluated since re-evaluates before its first kick (`Thermo.fresh`).
- **Cell.** An object built before an NPT stage moved the cell refuses to
  step (`Thermo.cell_is_current`).
- **Atoms.** `atoms` must match the evaluator's atom order and cell. ASE
  constraints and non-periodic axes are refused rather than ignored.
- **Virial.** `set_compute_virial` switches it on for the stage that needs
  it; it is refused under a field.

Two objects stepping one evaluator at the same time are not supported; one
after another is the design.

**`atoms` stays in step with the device, both ways.**
- **Device to `atoms`:** positions (unwrapped), momenta, the cell and a
  `SinglePointCalculator` are written back at log, trajectory and observer
  intervals, and at the end of every `run`, even one that raised.
- **`atoms` to device:** edits made between runs or inside an observer are
  uploaded before the next step. A cheap every-step observer can skip the
  sync with `attach(fn, 1, sync_atoms=False)`.

Nothing adds a host synchronisation per step beyond the evaluator's own.
Measured on water on one GB10, the classes cost what a hand-written kernel
loop costs:

| ms/step | 192 atoms | 12,288 atoms, compressed |
|---|---:|---:|
| hand-written velocity Verlet | 35.8 | 36.1 |
| `VelocityVerlet` | 35.7 | 36.4 |
| hand-written Nosé–Hoover chain | 40.0 | 36.0 |
| `NoseHooverChainNVT` | 40.5 | 36.1 |

These are medians of three interleaved runs; the runs spread by about ±3 ms at
192 atoms and ±1 ms at 12,288, and logging every 100 steps changes nothing
measurable.

**Checkpoints** hold the state, the class and its parameters, and a
fingerprint of the model and system. `Cls.from_checkpoint(ev, atoms, path)`
continues exactly: Langevin's noise is seeded per step, so a restart draws
what an uninterrupted run would. After a hard kill it trims the log and the
trajectory back to the checkpoint.

### Your own loop

For an integrator the classes lack, drive nvalchemiops' kernels on a
`Thermo` directly, or subclass `MolecularDynamics` and write `step()` — the
kernels around exactly one `self.thermo.evaluate()` — to get the log,
trajectory, observers and checkpoints for free.
[`07_custom_berendsen.py`](../examples/06-md_nvalchemi/07_custom_berendsen.py)
does the second, for a Berendsen thermostat. NVE on a `Thermo`:

```python
from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update, velocity_verlet_velocity_finalize,
)

th = Thermo(ev, atoms.get_positions(), velocities=atoms.get_velocities(),
            masses=atoms.get_masses())
dt = th.scalar(0.5 * units.fs)                   # the loop owns time
th.evaluate()
for _ in range(steps):
    velocity_verlet_position_update(
        th.positions, th.velocities, th.forces, th.masses, dt)
    th.evaluate()
    velocity_verlet_velocity_finalize(th.velocities, th.forces, th.masses, dt)
```

Available kernels: velocity Verlet (NVE), Langevin BAOAB, Nosé–Hoover chain,
NPT and NPH (MTK), and velocity rescaling. Each has a class above.

The classes handle these conventions. None can be guessed from the
signatures, and all were found by running into them:

- **Temperature is `k_B T` in energy units, not Kelvin.** Every nvalchemiops
  dynamics entry point documents it that way. For eV,
  `kT = ase.units.kB * T[K]`.
- **Half-step factors are inconsistent between kernels.** Measured:
  `nhc_velocity_half_step` and `velocity_verlet_position_update` apply their
  own factor of ½ (`dv = (F/m) dt/2`), `nhc_position_update` does not
  (`dr = v dt`), and `nhc_thermostat_chain_update` scales the velocities by
  a half step, `exp(−eta_dot dt/2)`. So the **full `dt`** goes to every call,
  and the chain update is called once at each end of the step. Passing `dt/2`
  to the Nosé–Hoover kick halves every force.
- **`ndof` dtypes disagree.** `nhc_compute_masses` requires int32 while
  `nhc_thermostat_chain_update` and `nhc_compute_chain_energy` require
  float64. That is a Warp launch error, not a cast, so keep both.
- **`total_scale` accumulates.** Reset it to 1 before every chain update; the
  kernel multiplies into it.
- **The Nosé–Hoover chain energy needs `eta / 2`.** While scaling the
  velocities by a half step, the chain update moves `eta` by `eta_dot dt/2`
  twice, a full `eta_dot dt`; called at both ends of the step, `eta` runs
  twice as fast as `eta_dot` implies. The dynamics are right either way; only
  the conserved quantity is off.
- **The NPT kernels count `N_f = 3 ×` the atom count they are given.** The
  centre of mass does not move, so pass `N − 1` for `3N − 3`, the count
  `Thermo.temperature` uses; otherwise the system is held `3N/(3N−3)` hot.
  Refresh `ke_w` and `press_w` (`update_kinetic_energy`, `update_pressure`)
  right before each barostat half-step; `Thermo`'s getters never write them.
- **Never rebind a `Thermo` array.** The kernels write into the bound
  buffers; `th.set_state(...)` copies new values in.

Units: the model works in angstrom and eV. Pair that with masses in amu and a
timestep in ASE time units (`A sqrt(amu/eV)`) and the set is self-consistent.
Take both constants from ASE — `from ase.units import fs, kB`.

### What is conserved

With a thermostat, watch the extended energy — the `Econs` column — rather
than the physical total, which a thermostat moves on purpose. For the
Nosé–Hoover chain it is

```
E_pot + E_kin + sum_k Q_k eta_dot_k^2 / 2 + N_f kT eta_1 + kT sum_{k>1} eta_k
```

with `eta` half the kernel's own array, `Q_1 = N_f kT tdamp²` and
`Q_k = kT tdamp²` for `k > 1`. Measured on water (192 atoms, 100 fs from one
start), it is conserved as dt² at any coupling:

| `tdamp` | dt 0.5 fs | 0.25 fs | 0.125 fs |
|---|---:|---:|---:|
| 20 fs, `eta / 2` | 0.0224 | 0.0057 | 0.0014 |
| 100 fs, `eta / 2` | 0.0226 | 0.0057 | 0.0014 |
| 20 fs, the kernel's `eta` | +9.95 | +10.48 | +10.61 |

The first two rows are the rms fluctuation, in meV/atom. The last is the
change over the 100 fs: with the kernel's own `eta`, the invariant counts the
heat the thermostat removes twice, and climbs by the same amount at every
timestep. An earlier version of this page reported that as an unresolved,
timestep-independent drift.

For MTK NPT, H′ adds `P V + W |eps_dot|² / 2`, with `W = 3N kT pdamp²`. The
NPT chain update moves `eta` by one `eta_dot dt/2` per half step, so there is
no halving there. H′ is O(dt²) only while the thermostat is nearly idle. The
NPT velocity kernel applies the drag `−(alpha eps_dot + eta_dot) v`
explicitly inside the kick, which is first order. On equilibrated water at
`tdamp` 100 fs, halving dt shrinks H′'s fluctuation 4.0× (0.0218 → 0.0054
meV/atom rms); a system heating fast enough to keep the thermostat busy gets
about 1.8×.

NPH conserves `Epot + Ekin + P V + W |eps_dot|² / 2`, the enthalpy, with no
thermostat term to be first order. Its barostat mass still needs a
temperature: the atoms' own at the start, unless `temperature_K` is given.
Measured on water, 0.5 ps from an equilibrated snapshot, rms in meV/atom:

| dt 0.5 → 0.25 fs | isotropic | orthorhombic | full |
|---|---:|---:|---:|
| NPH | 0.0218 → 0.0054 (4.05×) | 0.0207 → 0.0055 (3.74×) | 0.0202 → 0.0054 (3.71×) |
| NPT, `tdamp` 100 fs | 0.0218 → 0.0054 (4.0×) | 0.0216 → 0.0051 (4.21×) | 0.0206 → 0.0052 (3.94×) |

nvalchemiops moves the cell by a first-order step, `h + dt eps_dot h`, rather
than `exp(dt eps_dot) h`. Its error grows with the strain rate squared, so a
stiff barostat exposes it: on the test fixture, NPH's ratio fell from 3.8 at
`pdamp` 1 ps to 2.5 at 200 fs. Its cell update also reads the columns of `h`
as the lattice vectors, where ASE's are rows, so the classes hand the kernels
the transpose. With the rows passed as they are, NPH on a sheared cell kept an
error that did not shrink with dt: a ratio of 1.07, and ten times the
fluctuation.

NVE, the Nosé–Hoover NVT invariant, and NPH's enthalpy are the trustworthy
conservation checks.

### How the atoms are split

Into a grid of blocks, not slabs. What a device has to hold is its owned block
plus a shell of ghosts `rcut + skin` deep on every side, and that shell is the
part that does not shrink when the block does. Slabs keep two faces at full
cell size however thin they get; cutting more axes keeps every face smaller.
`decomposition_grid` picks, per partition count, the factorisation that
minimises one block's halo box: slabs for two, 2 x 2 x 1 for four, 2 x 2 x 2
for eight on a cubic box, and slabs again along a long axis when the box is
elongated enough. Each cut is at equal atom count, one axis at a time.

The positions may be **unwrapped** -- integrators never wrap them, and a
trajectory that is to yield diffusion should not be wrapped either. Every
rebuild folds the atoms into the cell first, builds the owners, images and
ghosts from the folded positions, and expresses each node back relative to the
positions it was given, so a node sits in the cell however far its atom has
wandered. Before this, the halo was built from the raw positions with only the
nearest images, and atoms that had diffused out through opposite faces lost
their neighbours: 1.1 eV/Å on water once the unwrapped cloud spanned L + 12 Å,
which a few picoseconds of diffusion reaches. Nothing needs folding between
rebuilds: a node moves with its atom, and the Verlet check below reads raw
displacement, so a caller that does wrap an atom (a jump by a lattice vector)
simply triggers a rebuild.

What that buys, as nodes on the busiest device, water at the auto skin:

| atoms | partitions | slabs | grid | change |
|---|---|---:|---:|---:|
| 12,288 | 4 | 9,761 | 8,855 | -9 % |
| 12,288 | 8 | 7,192 | 5,034 | -30 % |
| 41,472 | 4 | 26,015 | 23,555 | -9 % |
| 41,472 | 8 | 17,600 | 13,067 | -26 % |
| 98,304 | 4 | 52,898 | 49,135 | -7 % |
| 98,304 | 8 | 33,942 | 26,623 | -22 % |

Two partitions are unchanged, since a slab is the only way to cut in two. An
uncompressed step is device-bound -- about 5 ms of host work at 41,472 atoms
against hundreds on the device -- and its work follows the node count, so the
busiest device's count is what sets the step.

### What is split and what is not

| piece | cost at 6520 atoms | treatment |
|---|---:|---|
| descriptor + short-range | 883 ms | partitioned |
| charge prediction | (same pass) | partitioned |
| reciprocal sum + charge projection | 73 ms | gathered onto one device |
| charge-response force | — | partitioned |

The reciprocal sum is gathered rather than decomposed because it does not
survive spatial decomposition — splitting it gave 2.0% error on 2 ranks and
36.5% on 8. That is affordable because it is under 10% of the step and the
round trip is ~230 KB. Only the descriptor is worth splitting.

Most of that 73 ms is the gather and the charge projection, not the FFT: the
PME evaluation timed on its own is **1.5 ms**, and it stays there as the system
grows (2.3% of a 6,000-atom step, 0.3% of a 65,856-atom one) because its cost
follows the mesh rather than the atom count.

The charge-response force, `-Σⱼ (dE/dqⱼ) ∂qⱼ/∂rᵢ`, stays partitioned: `∂qⱼ/∂rᵢ`
is nonzero only within `rcut`, so although the *energy* is long-ranged its
*charge derivative* is short-ranged.

### Reusing a partitioning across steps

Ghosts are gathered within `rcut + skin`, and the partitioning is rebuilt only
once atoms have moved far enough to threaten it (the Verlet criterion,
`2·max displacement > skin`). In a 40-step test that meant 3 rebuilds instead
of 40, with results identical to rebuilding every step. `repartition_every=1`
forces a rebuild each step; it changes timing, never results — there is a test
asserting exactly that.

**A changing cell keeps the partitioning too.** Every node is an atom's folded
position `f_i = r_i − n_i H` plus a lattice shift `S H`, and each partition
keeps `n_i` and `S` as integers. `set_cell` re-expresses the shifts in the
new cell, and the Verlet bound counts what the cell's change does to a node:
each atom's motion in its folded frame, `Δr_i − n_i ΔH`, plus
`(max|S| + √3) ‖ΔH‖` for the ghosts and the images just outside the halo.
With the cell fixed that is the raw displacement, as before. Before this,
`set_cell` threw the partitioning away, so NPT repartitioned at every step:
66 against 38 ms/step for NVT at 12,288 water atoms, 154 against 97 at
41,472. It now rebuilds when the atoms or the cell have moved enough, about
as often as NVT does.

### Reusing each partition's neighbour search

Every step each partition used to search its nodes for pairs within `rcut`
afresh and sort the edges twice for the compressed kernel's CSR views: half
of every evaluation at 12,288 atoms and up (27 % search, 23 % sorting). Each
partition now keeps a pair list searched at `rcut + pair_skin`, sorted by
centre. Every step filters it to `rcut` and derives both CSR views from
prefix sums, with no search and no sort, until some node may have moved
`pair_skin / 2`. `pair_skin="auto"` is 1 Å, capped by the halo's skin, and
`pair_skin=0` searches every step. The edge set is the fresh search's (a
test compares them, and the CSR against `attach_edge_csr`). Only the edge
order differs, which moves results by summation-order rounding. With the
float32 water model, 200 steps of NVE drifted 1.6e-6 Å from the fresh-search
run, against 1.2e-6 Å between two fresh-search runs.

Both together, compressed water on one GB10, ms/step:

| atoms | NVT, search every step | NVT | NPT, repartitioning every step | NPT |
|---:|---:|---:|---:|---:|
| 192 | 13.1 | 10.6 | 14.7 | 12.9 |
| 12,288 | 35.6 | 24.9 | 66.3 | 26.9 |
| 41,472 | 97.0 | 67.9 | 154.5 | 70.4 |

### The knob that decides multi-GPU scaling

**`skin` (default `"auto"`).** The decomposition is reused until an atom has
moved half the skin, then rebuilt on the host by a spatial sort whose cost
grows with the atom count. Measured on two A40s at 98,304 atoms:

| skin (Å) | ms/step | rebuild | serial fraction | 2-GPU Amdahl ceiling |
|---|---|---|---|---|
| 1.0 (the old default) | 347 | every 13 steps | 76 % | 1.14x |
| 2.0 | 206 | every 40 steps | 54 % | 1.30x |
| 4.0 | 101 | none in 40 steps | 5 % | 1.90x |
| 8.0 | 189 | none | 4 % | 1.92x |

A constant is wrong at both ends -- a few thousand atoms rebuild cheaply and
pay for a thick halo -- so `"auto"` takes the largest skin whose halo costs at
most 25 % more nodes than a zero-skin halo, from the cell, the cutoff and the
partition count (`partition.default_skin`). That is 1.4 Å at 1,536 atoms,
4.3 Å at 98,304 and 5.3 Å at 192,000, and 4.3 Å is where the measurement put
the optimum. Pass a number to override; `evaluator.skin` reports what was used.

**What the rule cannot see: bytes.** The skin is sized from the cell, the
cutoff and the partition count, because those decide how much ghost work it
buys and how often the rebuild runs. How many bytes a node costs is a property
of the *model* -- an uncompressed descriptor holds several times what a
compressed one does -- and of the step, so no geometric rule can predict it. It
showed up as an out-of-memory at 98,304 atoms on eight devices with an
uncompressed long-range model, where the old fixed 1 A skin had fitted: the
rule grants 25 % more nodes on top of a halo that is already 2.5x the owned
count when the blocks are that small.

Memory therefore enters as a cap that is **measured**, not as a term in the
formula. The first evaluation is a probe: it runs at the floor skin, the
smallest halo there is, and its peak allocation divided by its node count is
what a node costs. That coefficient spans a factor of thirty between models,
which is exactly why no formula can carry it:

| model | bytes per node | linear fit error |
|---|---|---|
| uncompressed long-range | 755 kB | 0.3 % |
| compressed long-range | 24.3 kB | 1.2 % |

The footprint is linear in the node count, so the coefficient extrapolates to
any other skin, and the skin becomes the largest value whose predicted
footprint fits `memory_headroom` (0.8 by default) of what this process could
hold on the device -- free memory plus its own cached blocks, so a co-tenant
process is accounted for. It is only ever a ceiling: the geometry rule still
decides whenever memory is not the binding constraint. Measured at 41,472
atoms uncompressed, predicted against actual peak: 47.4 against 47.23 GiB.

Density is what turns a skin into a node count, and it enters through the node
count the probe actually **counted**, not through `N/V`: for a slab, an
interface or anything with a void, the counted occupancy is right where a
uniform density is not.

The reactive path remains underneath, because a prediction can still be wrong
when another process grows on the same device. On an out-of-memory the
evaluator halves the skin, rebuilds and retries the step, and that value
becomes a ceiling too, `set_cell` under a barostat included.
`evaluator.skin_backoffs` counts the reductions and each one warns. An explicitly chosen skin is never overridden, and the
backoff stops at 0.1 A, below which the halo is not what is filling the
device. Lowering the skin instead of raising it in the rule was rejected on
measurement: an owned-count-relative rule gives 1.2 A at eight partitions,
which is the old fixed default and puts the step straight back into the
rebuild-dominated regime the auto skin exists to escape.

### Why the launch phase is single-threaded

A thread per device once recovered part of the overlap that the neighbour
search's blocking host reads take away -- 1.31x to 1.88x of an ideal 2x on two
A40s at 98,304 atoms. It was removed (`406f775`): the lazily-initialised
libraries underneath -- deepmd-kit's fake-op registration, torch's CUDA
linear-algebra loader, warp's code generation -- raced on first use, and a
4-GPU V100 scaling matrix kept losing rows. The loop now runs one partition at
a time, in order. Two things from that work stay, because they are about
correctness rather than overlap:

- **`_current_device` around each partition.** The compressed operator binds
  the *current* device rather than its arguments' device: without this, a
  compressed model whose buffers sit on `cuda:1` returns forces wrong by
  5 eV/A while `cuda:0` is current.
- **One coordinate copy per device, taken before the loop.** Every partition
  on a device reads the same full array and gathers its own rows, so this is
  one upload per device instead of one per partition.

## What the driver returns

`evaluate_partitioned` assembles on the gather device and returns **device
tensors** (scalars as 0-d tensors). There is no host-side assembly path: the
per-partition forces, energies, charges, charge-response forces and Born
tensors are scatter-added with `index_add_` on that device, which is the
ghost-force reverse communication of a domain-decomposed code done as a device
primitive. Convert explicitly when you want numpy:

```python
from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy

out = evaluate_partitioned(model, coord, atype, cell, n_partitions=2, devices=[...])
out["force"]                      # torch tensor on the gather device
to_numpy(out)["force"]            # numpy (N, 3), one explicit download
```

`DistributedEvaluator` keeps the tensor path: the forces go straight into the
buffer the warp integrator aliases, so they never leave the GPU between steps,
and `results` converts an entry to numpy only when it is read
(`results_on_device=True` skips even that). One download per step remains, the
energy scalar, because the call returns it as a Python float.

## Limits

- **`stress` is available, opt-in.** Pass `compute_virial=True` and read
  `evaluator.virial` (eV) or `evaluator.stress` (eV/Å³, ASE's sign convention).
  It is off by default because it costs a strain leaf per partition and an
  extra target in the charge-response backward; the `(energy, forces)` return
  signature is unchanged either way.

  All three contributions decompose. The short-range virial is the sum of the
  per-partition `energy_derv_c_redu` — each edge survives in exactly one
  partition, the same ownership argument that makes the forces exact. The
  reciprocal virial is a strain derivative taken on the gather device. The
  charge-response virial comes from straining each partition's node coordinates
  and backpropagating `dE/dq` into the strain leaf. The trick that makes the
  last one work: a ghost's Cartesian offset is a lattice-vector combination
  `n A`, which deforms as `n A (I+ε)ᵀ` — the same transform as a coordinate. So
  straining the assembled node set is exact, and the strain reaches `edge_vec`
  through the coordinates rather than through a box. Verified against the
  serial virial — itself finite-difference checked — to ~6e-15, symmetric, and
  identical at 1, 2, 4 and 8 partitions.

- **Multi-GPU pays only where the devices are the bottleneck.** Results are
  identical either way: on two distinct A40s, every returned quantity at 2, 3,
  4 and 8 partitions (grids 2x1x1 to 2x2x2) matches the undivided model at the
  run-to-run floor, and NVE energy traces from the same start agree to 1e-5
  meV/atom. Speed depends on where a step spends its time, measured on two
  A40s:

  | case | 1 GPU | 2 GPUs | limited by |
  |---|---:|---:|---|
  | compressed long-range, 12,288 atoms | 20.4 ms | 24.8 ms | host (device busy 3.8 ms of the step) |
  | compressed long-range, 41,472 atoms | 48.7 ms | 47.6 ms | host (device busy 10.7 ms) |
  | uncompressed short-range, 12,288 atoms | 222 ms | 141 ms | devices (121 ms each) |

  So compressed runs below roughly 100,000 atoms belong on one GPU, and an
  uncompressed run overlaps at 80–86 % across two. The fixed per-partition
  host cost -- the neighbour search and eager-PyTorch dispatch -- is what more
  partitions multiply.

- **Neighbour lists.** Which builder runs for the descriptor and for the
  charge head, why a rebuilt model must re-resolve `neighbor_graph_method`,
  and why the virial never needs the O(N²) builder, is in
  [Which neighbour list runs](neighbour-lists.md).
