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
    FIRE2, MTKNPT, NoseHooverChainNVT, build_evaluator, read_structure,
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
npt = MTKNPT(ev, atoms, 0.5 * units.fs, temperature_K=300,
             pressure_au=1e-4 * units.GPa, tdamp=100 * units.fs,
             pdamp=1000 * units.fs, cell="isotropic", logfile="npt.log",
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
| `MTKNPH` | NPH, the MTK barostat alone; `cell` as for `MTKNPT` | the enthalpy H′ |
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
- **Atoms.** `atoms` must match the evaluator's atom order, cell and pbc.
  ASE's `FixAtoms` holds its atoms in place, at rest, and the temperature
  counts three degrees of freedom per mobile atom. Any other ASE constraint is
  refused rather than ignored.
- **Boundary.** The structure's `pbc` decides it: a bulk cell, or a slab with
  one open axis. Along an open axis the evaluator makes no images, and the
  long-range sum takes its slab form (Yeh–Berkowitz) whatever the model was
  trained with, since the term holds no weights. `read_structure` pads a
  slab's open axis to at least 20 Å of vacuum (`vacuum=`) and centres the
  slab. An atom that comes within half a skin of an open face is refused,
  because past it the sum would put the atom across the vacuum. NPT, NPH and
  cell relaxation are refused for a slab. An `isolated` term, and molecules
  and clusters, belong to the ASE calculator
  ([non-periodic](non-periodic.md#running-md)).
- **PME mesh.** Set from the cell when the evaluator is built, at least
  `ceil(2 alpha L)` points per axis, and kept for the run
  ([alpha and mesh](alpha-and-mesh.md)).
- **The model.** Each evaluator builds its own long-range term for its cell
  and boundary (`ev.long_range_energy`) and never changes the model, so
  evaluators for different systems, and ASE calculators, can share one.
- **Checkpoints** record the boundary with the rest of the physics, so a
  bulk run's checkpoint does not restore onto a slab evaluator, or the
  reverse.
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

**On one device they are not.** The node set is the system itself
(`partition.whole_system`): every atom owned, no ghosts, and the GPU-direct
neighbour list below searches the real cell with the real periodicity,
recording each pair's lattice image as an integer shift. The nodes are the
atoms -- 1.96M of them where the periodic-image halo used to add 0.8M more
on one H200 -- and the host-side spatial sort never runs. The rest of this
section is about two or more devices.

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
simply triggers a rebuild. On one device the question does not arise: the
cell-list kernel folds the images into the pair shifts itself, at every step,
and a lattice jump changes nothing.

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

The bound is evaluated on the device and read back as one bool per step (it
used to download two reductions). On one device nothing is reused and nothing
is rebuilt: `rebuild_count` only counts the rebuilds `repartition_every`
forces.

**A barostat never touches the host.** The cell it integrates is a device
tensor, and `Thermo.set_cell_device` hands it on as one: the evaluator copies
it device-to-device, inverts it there (a 3 x 3 inverse written out, since
`torch.linalg.inv` reads the device to check its result), re-expresses the
partitions' image shifts and the long-range term's k-vectors on the device,
and leaves its host copy stale until a log row, a checkpoint or a repartition
reads `evaluator.cell`. The pressure kernel reads the evaluator's device
virial instead of a re-upload of the host copy, and `cell_is_current` is a
version counter rather than a comparison of host copies. An NPT step thus
downloads exactly what an NVE step does (the energy word; one more for an
uncompressed model's compact edge count), where it used to download the cell
twice, upload it once and re-upload the virial. `set_cell` with a host cell
still works for everything else (relaxations, tests, `read_structure`).
`tests/test_sync.py` pins the NPT count for both large and small periodic
boxes. After a cell change, the search radius is refreshed on the device
from the new cell and the actual bin counts, using the existing buffers.

### The neighbour list lives on the device

Every step, each node set is searched afresh at `rcut` by
`GpuNeighborList` (`deepmd_lr/nvalchemi/nlist.py`): nvalchemiops' cell list
over the nodes, with only the owned atoms as search centres, so no ownership
mask is needed and every edge lands in exactly one partition. The kernel
leaves a neighbour matrix -- row `i` holds centre `i`'s neighbours and their
image shifts -- and two small Warp kernels turn it into DeePMD-kit's graph
without a sort: the destination CSR is a prefix sum of the row counts and a
fill kernel compacts the rows into `edge_index`; the source CSR is built the
way LAMMPS-Kokkos builds it, a per-node count, a prefix sum and an atomic
cursor. The pair vector is `x[j] − x[i] + S·H`, nvalchemiops' and
DeePMD-kit's convention at once, so the shifts pass straight through, and
unwrapped coordinates are fine: the kernel folds the images into `S`.

Nothing is cached between steps and nothing reads the host: the buffers are
allocated once (sized by the first search, which is the one that
synchronises), the edge count stays a device scalar, `edge_mask` marks where
the real edges end, and a device status word reports a matrix row or an edge
buffer that overflowed, in which case the evaluator widens it by a quarter,
repeats the step, warns, and counts it in `evaluator.list_growths`. The
first search sizes the matrix at 1.2x the widest row it counted and the edge
buffers at 1.15x the edges, so growth is rare: the total edge count moves by
well under a percent along a water trajectory, while the widest row drifts
up as the run samples more configurations (101 to 113 neighbours over 440
steps at 319k atoms, which is what the 1.2x absorbs). A compressed model takes the
kernel's float32 vectors over the full buffers -- its kernels walk the CSR
rows and never see the tail -- unless the charge response needs `edge_vec`
differentiable, in which case the vectors are recomputed from the node
coordinates and the strain; an uncompressed model gets vectors in its own
precision on a compact edge axis, at the cost of one more download (the edge
count), because its descriptor processes every edge it is handed.

What this replaced: DeePMD-kit's builder over every node, ghosts included,
then a dense-matrix decode, float64 vectors for every candidate pair, two
stable argsorts and a per-step `nonzero`/`bincount` filter of a cached pair
list, about ten host synchronisations per step. On one H200 at 1.96M water
atoms that was 649 ms per search and 61 ms per step for the graph, against
9.7 and 17.7 for LAMMPS-Kokkos, and on one GPU the periodic-image halo made
the model run on 2.76M nodes instead of 1.96M. nvalchemiops' search itself
scales linearly -- 0.96, 6.6 and 56 ms at 10k, 169k and 1.96M atoms on a
GB10, whose memory is some fifteen times slower than an H200's -- so a
stateless search every step costs less than the bookkeeping a reusable list
needs. A Verlet-reuse variant (search at `rcut + skin`, filter per step) is
measured against it in `experiments/scaling/profile_nlist.py` and will be
added only if the per-step search turns out to matter on the H200.

Measured 2026-10-02, water, the compressed SR model (`DP_CUDA_INFER=2`),
ms per MD step and per piece (the old path's 1.96M numbers are the H200
profile the brief was written from; its profiler ran out of the H200's 139 GB
at that size):

| GPU | atoms | path | step | graph (+ search) | forward | nodes | peak GiB |
|---|---:|---|---:|---:|---:|---:|---:|
| GB10 | 10,368 | pair list | 10.2 | 4.0 (+17 per rebuild, every ~13 steps) | 4.2 | 23,529 | 0.9 |
| GB10 | 10,368 | GPU list | 3.6 | 1.2 | 2.1 | 10,368 | 0.1 |
| GB10 | 169,344 | pair list | 146 | 68.5 (+227 per rebuild) | 57.5 | 277,074 | 11.1 |
| GB10 | 169,344 | GPU list | 53 | 13.6 | 39.1 | 169,344 | 1.9 |
| H200 | 10,368 | pair list | 4.3 | 0.7 (+3.7 per rebuild, 28 ms repartition) | 1.2 | 23,422 | — |
| H200 | 10,368 | GPU list | 2.6 | 0.8 | 1.2 | 10,368 | 0.2 |
| H200 | 1,956,864 | pair list | 213 | 61 (+704 per rebuild, 3.8 s repartition) | 95 | 2,757,158 | 115 |
| H200 | 1,956,864 | GPU list | 97 | 26 | 68 | 1,956,864 | 22 |

The list's own stages on the H200 at 1.96M: query 12.8 ms, fill 7.1, source
CSR 6.2, 26 ms in all for 175M edges (89 per atom), the same standalone,
inside the evaluator and after 50 MD steps on the moved, unwrapped positions
(hopper-29; one earlier run on hopper-26 had the query kernel at 54 ms and the
step at 139 ms with the same binary and inputs -- a node, not the code; if a
step looks slow, compare nodes before anything else). The step is then 97 ms
against LAMMPS-Kokkos's 90 on the same problem, with the model's kernels at
68 ms. A search at `rcut + 1 A`, what a Verlet variant would pay per rebuild,
costs 44 ms for 280M pairs; with the per-step query at 12 ms, a Verlet list
would save a few ms per step at the price of reference positions, fold
images, a rebuild bound and the cell-change bookkeeping, so it is not built.

A graph that is differentiated -- the charge response, the virial strain --
or a model that runs in float64 asks the list for `float64_vectors`: the fill
kernel then keeps its float64 vectors too, and `edge_vec` hands them out
through an autograd function whose backward is the exact adjoint of the two
gathers, a scatter-add onto the source and its negative onto the destination.
The strain product has its own backward as well, because autograd's version
reduces `vec^T @ g` with a `(3, E) x (E, 3)` GEMM that cuBLAS mishandles at
tens of millions of edges. On the GB10 at 169k atoms the long-range path's
vectors, strain and backward went from 141 ms (recomputed in torch, with a
`(E, 3) x (3, 3)` float64 matmul on a bad kernel) to 64 ms in float64 and
36 ms in float32; without the strain, 13 ms.

### The knob that decides multi-GPU scaling

**`skin` (default `"auto"`).** Two or more devices only: on one device
there is no halo to size, the node count is the atom count whatever the
skin, and the memory probe below does not run. The decomposition is reused
until an atom has moved half the skin, then rebuilt on the host by a spatial
sort whose cost grows with the atom count. Measured on two A40s at 98,304 atoms:

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
and `results` converts an entry to numpy only when it is read. One download
per step remains, because the call returns the energy as a Python float: the
energy, the neighbour list's status flags and, with `compute_virial`, the
virial come back in one small copy. On one device that is the step's only
host synchronisation; two or more devices add the one bool of the halo's
Verlet check; an uncompressed model adds its compact edge count.
`tests/test_sync.py` pins these counts.

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
  host cost -- eager-PyTorch dispatch; the neighbour search no longer reads
  the host -- is what more partitions multiply.

- **Neighbour lists.** Which builder runs for the descriptor and for the
  charge head, why a rebuilt model must re-resolve `neighbor_graph_method`,
  and why the virial never needs the O(N²) builder, is in
  [Which neighbour list runs](neighbour-lists.md).
