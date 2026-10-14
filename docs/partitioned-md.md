# Running MD over partitions

`LongRangeCalculator` is single-process and single-device. `DistributedEvaluator`
evaluates the model over spatial partitions, one per GPU:

```python
from deepmd_lr.nvalchemi import DistributedEvaluator

evaluator = DistributedEvaluator(
    model, atype, cell,
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    compute_virial=False,       # True also fills .virial and .stress
)
energy, forces = evaluator(positions)      # eV, eV/angstrom
```

It owns the partitioning, the ghost halo, the per-device replicas and the
decision of when to rebuild, and nothing else: no timestep, ensemble,
thermostat or velocities. `positions` may be a torch tensor, including one an
integrator mutates in place, or anything `np.asarray` accepts. Forces come
back as a float64 tensor on the model's device, which `wp.from_torch` views
without a copy, so the trajectory never leaves the GPU. `evaluator.results`
carries the energy split and charges; `evaluator.masses()` reads the element
order off the model's `type_map`. There is no ASE calculator on this path on
purpose: ASE's integrators cannot touch on-device state.

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

`FIRE` and `FIRE2` are ASE's two relaxations on nvalchemiops kernels, with
ASE's parameter names and unit masses. FIRE caps each atom's step on its own;
FIRE2 kicks before measuring the power, backs off half a step after an uphill
step, and scales the whole step so no atom moves more than `maxstep`. On water
from an MD snapshot they reached 0.05 eV/Å in 199 and 380 steps; from a
TIP3P-relaxed packed box, in 149 and 160.

The units are ASE's: `timestep` in ASE time (`0.5 * units.fs`), `friction` in
1/ASE time, `tdamp` and `pdamp` in ASE time, `pressure_au` in eV/Å³,
`temperature_K` in kelvin.

### Relaxing the cell

`relax_cell` chooses what else may move. The target is `pressure_au` and
`run(smax=...)` sets how close the stress must come to it (default 0.01 GPa).

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

The cell's rows join the atoms in one extended array that the FIRE kernel
steps like any other degrees of freedom (nvalchemiops' `07_fire_variable_cell`),
with three differences, each measured:

- **The cell force** is −∂(E + PV)/∂H = −V H⁻ᵀ (σ + P) for the rows of ASE's
  cell. nvalchemiops' `stress_to_cell_force` gives −V σ H⁻ᵀ, the gradient for
  a cell whose columns are the lattice vectors; the two agree for a diagonal
  cell only. On a sheared water cell this one matched the finite-difference
  gradient to 2e-3 eV/Å and nvalchemiops' was 0.70 eV/Å off. The structure is
  never rotated.
- **The atoms ride the cell affinely** before their own step.
- **The cell force is not divided by the atom count**, which makes the cell
  about as heavy as one atom.

On water stretched by 4%, at 192 and 12,288 atoms alike, to 0.05 eV/Å and
0.01 GPa:

| steps | isotropic | orthorhombic | full |
|---|---:|---:|---:|
| `FIRE2` | 390 | 440 | 390 |
| `FIRE` | 750 | 610 | 740 |
| `FIRE2`, cell force ÷ N | 1000 | 1480 | 1570 |

The new cell goes to the evaluator every step, as under NPT, and the
partitions follow it without a rebuild. An MD object built before the
relaxation refuses to step afterwards; build it from the relaxed `atoms`.

### Sharing one evaluator

| object | built by | owns |
|---|---|---|
| `DistributedEvaluator` | you, once | the model replicas, the partitioning and skin, the forces buffer, the cell |
| `Thermo` | each MD object, in its `__init__` | positions, velocities, masses and the forces view, on the device; the measurements |
| the MD object | you | the timestep and ensemble parameters, the thermostat and barostat state, the step count, the log, the trajectory, observers, checkpoints |

`md.evaluator` returns `md.thermo.evaluator`, the object you built; no class
builds or copies one. Sharing it between stages, one after another (two at
once is not supported), is safe by construction:

- **Forces.** One shared buffer; an object another has evaluated since
  re-evaluates before its first kick (`Thermo.fresh`).
- **Cell.** An object built before an NPT stage moved the cell refuses to step
  (`Thermo.cell_is_current`).
- **Atoms.** `atoms` must match the evaluator's atom order, cell and pbc.
  `FixAtoms` holds its atoms in place, at rest, with three degrees of freedom
  per mobile atom in the temperature; any other constraint is refused.
- **Boundary.** The structure's `pbc` decides it: bulk, or a slab with one
  open axis. Along an open axis there are no images and the long-range sum
  takes its slab form (Yeh–Berkowitz) whatever the model was trained with,
  since the term holds no weights. `read_structure` pads the open axis to at
  least 20 Å of vacuum (`vacuum=`) and centres the slab. An atom within half a
  skin of an open face is refused. NPT, NPH and cell relaxation are refused for
  a slab. An `isolated` term, molecules and clusters belong to the ASE
  calculator ([non-periodic](non-periodic.md#running-md)).
- **PME mesh.** Set from the cell when the evaluator is built, at least
  `ceil(2 alpha L)` points per axis, and kept for the run
  ([alpha and mesh](alpha-and-mesh.md)).
- **The model.** Each evaluator builds its own long-range term
  (`ev.long_range_energy`) and never changes the model, so evaluators for
  different systems, and ASE calculators, can share one.
- **Checkpoints** record the boundary, so a bulk checkpoint does not restore
  onto a slab evaluator or the reverse.
- **Virial.** `set_compute_virial` switches it on for the stage that needs it;
  it is refused under a field.

**`atoms` stays in step with the device, both ways.** Positions (unwrapped),
momenta, the cell and a `SinglePointCalculator` are written back at log,
trajectory and observer intervals and at the end of every `run`, even one that
raised. Edits made between runs or inside an observer are uploaded before the
next step; a cheap every-step observer can skip that with
`attach(fn, 1, sync_atoms=False)`.

The classes add no host synchronisation beyond the evaluator's own and cost
what a hand-written kernel loop costs (water, one GB10, medians of three
interleaved runs; logging every 100 steps changes nothing measurable):

| ms/step | 192 atoms | 12,288 atoms, compressed |
|---|---:|---:|
| hand-written velocity Verlet | 35.8 | 36.1 |
| `VelocityVerlet` | 35.7 | 36.4 |
| hand-written Nosé–Hoover chain | 40.0 | 36.0 |
| `NoseHooverChainNVT` | 40.5 | 36.1 |

**Checkpoints** hold the state, the class and its parameters, and a
fingerprint of the model and system. `Cls.from_checkpoint(ev, atoms, path)`
continues exactly: Langevin's noise is seeded per step, so a restart draws what
an uninterrupted run would, and after a hard kill the log and trajectory are
trimmed back to the checkpoint.

### Your own loop

For an integrator the classes lack, drive nvalchemiops' kernels on a `Thermo`
directly, or subclass `MolecularDynamics` and write `step()`, the kernels
around exactly one `self.thermo.evaluate()`, to get the log, trajectory,
observers and checkpoints for free
([`07_custom_berendsen.py`](../examples/06-md_nvalchemi/07_custom_berendsen.py)
does the second). NVE on a `Thermo`:

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

Kernels exist for velocity Verlet, Langevin BAOAB, the Nosé–Hoover chain, MTK
NPT and NPH, and velocity rescaling; each has a class above. The kernels'
conventions, which the classes handle and the signatures do not reveal:

- **Temperature is `k_B T` in energy units, not kelvin**: `kT = ase.units.kB * T`.
- **Half-step factors differ between kernels.** `nhc_velocity_half_step` and
  `velocity_verlet_position_update` apply their own ½ (`dv = (F/m) dt/2`),
  `nhc_position_update` does not (`dr = v dt`), and
  `nhc_thermostat_chain_update` scales the velocities by a half step,
  `exp(−eta_dot dt/2)`. So the full `dt` goes to every call and the chain
  update is called once at each end of the step. Passing `dt/2` to the
  Nosé–Hoover kick halves every force.
- **`ndof` dtypes disagree**: `nhc_compute_masses` wants int32,
  `nhc_thermostat_chain_update` and `nhc_compute_chain_energy` float64, and a
  mismatch is a Warp launch error, not a cast.
- **`total_scale` accumulates**; reset it to 1 before every chain update.
- **The Nosé–Hoover chain energy needs `eta / 2`.** The chain update moves
  `eta` by a full `eta_dot dt` while scaling the velocities by a half step, so
  called at both ends `eta` runs twice as fast as `eta_dot` implies. The
  dynamics are right; only the conserved quantity is off.
- **The NPT kernels count `N_f = 3 ×` the atom count they are given.** Pass
  `N − 1` for `3N − 3`, the count `Thermo.temperature` uses, or the system is
  held `3N/(3N−3)` hot. Refresh `ke_w` and `press_w` (`update_kinetic_energy`,
  `update_pressure`) right before each barostat half-step; `Thermo`'s getters
  never write them.
- **Never rebind a `Thermo` array.** The kernels write into the bound buffers;
  `th.set_state(...)` copies new values in.

Units: angstrom, eV, amu and ASE time (`Å sqrt(amu/eV)`) are self-consistent;
take `fs` and `kB` from `ase.units`.

### What is conserved

With a thermostat, watch the extended energy, the `Econs` column, rather than
the physical total. For the Nosé–Hoover chain it is

```
E_pot + E_kin + sum_k Q_k eta_dot_k^2 / 2 + N_f kT eta_1 + kT sum_{k>1} eta_k
```

with `eta` half the kernel's own array, `Q_1 = N_f kT tdamp²` and
`Q_k = kT tdamp²` for `k > 1`. On water (192 atoms, 100 fs from one start) it
is conserved as dt² at any coupling:

| `tdamp` | dt 0.5 fs | 0.25 fs | 0.125 fs |
|---|---:|---:|---:|
| 20 fs, `eta / 2` | 0.0224 | 0.0057 | 0.0014 |
| 100 fs, `eta / 2` | 0.0226 | 0.0057 | 0.0014 |
| 20 fs, the kernel's `eta` | +9.95 | +10.48 | +10.61 |

The first two rows are the rms fluctuation in meV/atom; the last is the change
over the 100 fs, which with the kernel's own `eta` counts the heat the
thermostat removes twice and climbs at every timestep.

For MTK NPT, H′ adds `P V + W |eps_dot|² / 2`, with `W = 3N kT pdamp²`. The
NPT chain update moves `eta` by one `eta_dot dt/2` per half step, so there is
no halving there. H′ is O(dt²) only while the thermostat is nearly idle,
because the NPT velocity kernel applies the drag `−(alpha eps_dot + eta_dot) v`
explicitly inside the kick, which is first order: on equilibrated water at
`tdamp` 100 fs, halving dt shrinks H′'s fluctuation 4.0× (0.0218 → 0.0054
meV/atom rms); a system heating fast enough to keep the thermostat busy gets
about 1.8×.

NPH conserves the enthalpy `Epot + Ekin + P V + W |eps_dot|² / 2`, with no
first-order thermostat term. Its barostat mass still needs a temperature: the
atoms' own at the start, unless `temperature_K` is given. On water, 0.5 ps
from an equilibrated snapshot, rms in meV/atom:

| dt 0.5 → 0.25 fs | isotropic | orthorhombic | full |
|---|---:|---:|---:|
| NPH | 0.0218 → 0.0054 (4.05×) | 0.0207 → 0.0055 (3.74×) | 0.0202 → 0.0054 (3.71×) |
| NPT, `tdamp` 100 fs | 0.0218 → 0.0054 (4.0×) | 0.0216 → 0.0051 (4.21×) | 0.0206 → 0.0052 (3.94×) |

nvalchemiops moves the cell by a first-order step, `h + dt eps_dot h`, whose
error grows with the strain rate squared, so a stiff barostat exposes it: NPH's
ratio fell from 3.8 at `pdamp` 1 ps to 2.5 at 200 fs. Its cell update also
reads the columns of `h` as the lattice vectors, where ASE's are rows, so the
classes hand the kernels the transpose; with the rows passed as they are, NPH
on a sheared cell kept an error that did not shrink with dt.

NVE, the Nosé–Hoover NVT invariant, and NPH's enthalpy are the trustworthy
conservation checks.

### How the atoms are split

**On one device they are not.** The node set is the system itself
(`partition.whole_system`): every atom owned, no ghosts, and the neighbour
list below searches the real cell with the real periodicity, recording each
pair's lattice image as an integer shift. The host-side spatial sort never
runs. The rest of this section is about two or more devices.

Into a grid of blocks, not slabs. A device holds its owned block plus a shell
of ghosts `rcut + skin` deep on every side, and that shell does not shrink
when the block does: slabs keep two faces at full cell size however thin they
get, while cutting more axes keeps every face smaller. `decomposition_grid`
picks, per partition count, the factorisation that minimises one block's halo
box (slabs for two, 2 × 2 × 1 for four, 2 × 2 × 2 for eight on a cubic box,
slabs again along a long axis when the box is elongated enough), each cut at
equal atom count. Nodes on the busiest device, water at the auto skin:

| atoms | partitions | slabs | grid | change |
|---|---|---:|---:|---:|
| 12,288 | 4 | 9,761 | 8,855 | -9 % |
| 12,288 | 8 | 7,192 | 5,034 | -30 % |
| 41,472 | 4 | 26,015 | 23,555 | -9 % |
| 41,472 | 8 | 17,600 | 13,067 | -26 % |
| 98,304 | 4 | 52,898 | 49,135 | -7 % |
| 98,304 | 8 | 33,942 | 26,623 | -22 % |

An uncompressed step is device-bound (about 5 ms of host work at 41,472 atoms
against hundreds on the device) and follows the node count, so the busiest
device sets the step.

The positions may be **unwrapped**. Every rebuild folds the atoms into the
cell, builds the owners, images and ghosts from the folded positions, and
expresses each node back relative to the positions it was given, so a node
sits in the cell however far its atom has wandered. Between rebuilds nothing
needs folding: a node moves with its atom, and the Verlet check reads raw
displacement, so a caller that wraps an atom (a jump by a lattice vector)
simply triggers a rebuild. On one device the cell-list kernel folds the images
into the pair shifts itself, every step.

### What is split and what is not

| piece | cost at 6520 atoms | treatment |
|---|---:|---|
| descriptor + short-range | 883 ms | partitioned |
| charge prediction | (same pass) | partitioned |
| reciprocal sum + charge projection | 73 ms | gathered onto one device |
| charge-response force | — | partitioned |

The reciprocal sum is gathered because it does not survive spatial
decomposition (2.0% error on 2 ranks, 36.5% on 8), and that is affordable
because it is under 10% of the step and the round trip is ~230 KB. Most of the
73 ms is the gather and the projection; the PME itself is 1.5 ms and stays
there as the system grows ([Cost](cost.md)). The charge-response force,
`-Σⱼ (dE/dqⱼ) ∂qⱼ/∂rᵢ`, stays partitioned: the energy is long-ranged, but
`∂qⱼ/∂rᵢ` is nonzero only within `rcut`.

### Reusing a partitioning across steps

Ghosts are gathered within `rcut + skin`, and the partitioning is rebuilt only
once atoms have moved far enough to threaten it (`2·max displacement > skin`).
`repartition_every=1` forces a rebuild each step; it changes timing, never
results, and a test asserts exactly that.

**A changing cell keeps the partitioning.** Every node is an atom's folded
position `f_i = r_i − n_i H` plus a lattice shift `S H`, with `n_i` and `S`
kept as integers. `set_cell` re-expresses the shifts in the new cell, and the
Verlet bound counts each atom's motion in its folded frame, `Δr_i − n_i ΔH`,
plus `(max|S| + √3) ‖ΔH‖` for the ghosts and the images just outside the halo.
NPT therefore rebuilds about as often as NVT does; repartitioning at every
step cost 66 against 38 ms/step at 12,288 water atoms and 154 against 97 at
41,472.

The bound is evaluated on the device and read back as one bool per step. On
one device nothing is reused and nothing is rebuilt; `rebuild_count` only
counts the rebuilds `repartition_every` forces.

**A barostat never touches the host.** The cell it integrates is a device
tensor, handed on by `Thermo.set_cell_device`: the evaluator copies it
device-to-device, inverts it there (a 3 × 3 inverse written out, since
`torch.linalg.inv` reads the device to check its result), re-expresses the
partitions' image shifts and the long-range term's k-vectors on the device,
and leaves its host copy stale until a log row, a checkpoint or a repartition
reads `evaluator.cell`. The pressure kernel reads the evaluator's device
virial, and `cell_is_current` is a version counter. An NPT step therefore
downloads exactly what an NVE step does. `set_cell` with a host cell still
works for relaxations, tests and `read_structure`. After a cell change the
search radius is refreshed on the device from the new cell and the bin counts,
in the existing buffers.

### The neighbour list lives on the device

Every step, each node set is searched afresh at `rcut` by `GpuNeighborList`
(`deepmd_lr/nvalchemi/nlist.py`): nvalchemiops' cell list over the nodes, with
only the owned atoms as search centres, so every edge lands in exactly one
partition. The kernel leaves a neighbour matrix (row `i` holds centre `i`'s
neighbours and their image shifts) and two small Warp kernels turn it into
DeePMD-kit's graph without a sort: the destination CSR is a prefix sum of the
row counts plus a fill kernel that compacts the rows into `edge_index`; the
source CSR is a per-node count, a prefix sum and an atomic cursor, as
LAMMPS-Kokkos builds it. The pair vector `x[j] − x[i] + S·H` is both
libraries' convention, so the shifts pass straight through, and the kernel
folds unwrapped coordinates into `S`.

Nothing is cached between steps and nothing reads the host. The buffers are
allocated once, sized by the first search (the one that synchronises) at 1.2×
the widest row and 1.15× the edge count; the edge count stays a device scalar,
`edge_mask` marks where the real edges end, and a device status word reports
an overflowed row or edge buffer, on which the evaluator widens it by a
quarter, repeats the step, warns and counts it in `evaluator.list_growths`.
Growth is rare: along a water trajectory the edge count moves by well under a
percent, and the widest row drifts up slowly (101 to 113 neighbours over 440
steps at 319k atoms), which the margin absorbs. A compressed model takes the
kernel's float32 vectors over the full buffers, since its kernels walk the CSR
rows and never see the tail, unless the charge response needs `edge_vec`
differentiable; an uncompressed model gets vectors in its own precision on a
compact edge axis, at the cost of one more download (the edge count), because
its descriptor processes every edge it is handed.

Measured 2026-10-02 on water with the compressed SR model (`DP_CUDA_INFER=2`),
ms per MD step, against the cached pair list it replaced (DeePMD-kit's builder
over every node, ghosts included, a dense-matrix decode, two argsorts and a
per-step `nonzero`/`bincount` filter, about ten host synchronisations per
step):

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
inside the evaluator and after 50 MD steps on the moved, unwrapped positions.
The step is 97 ms against LAMMPS-Kokkos's 90 on the same problem, with the
model's kernels at 68 ms. nvalchemiops' search scales linearly (0.96, 6.6 and
56 ms at 10k, 169k and 1.96M atoms on a GB10), so a stateless search every
step costs less than the bookkeeping a Verlet list needs: a search at
`rcut + 1 Å` costs 44 ms for 280M pairs per rebuild, against a 12 ms per-step
query, so a Verlet variant would save a few ms per step for reference
positions, fold images, a rebuild bound and cell-change bookkeeping, and is
not built. If a step looks slow, compare nodes before anything else: one H200
node ran the same binary and inputs at 139 ms with the query kernel at 54 ms.

A graph that is differentiated (the charge response, the virial strain) or a
model that runs in float64 asks the list for `float64_vectors`: the fill
kernel keeps float64 vectors too, and `edge_vec` hands them out through an
autograd function whose backward is the exact adjoint of the two gathers, a
scatter-add onto the source and its negative onto the destination. The strain
product has its own backward as well, because autograd's version reduces
`vec^T @ g` with a `(3, E) × (E, 3)` GEMM that cuBLAS mishandles at tens of
millions of edges. On the GB10 at 169k atoms the long-range path's vectors,
strain and backward went from 141 ms (recomputed in torch) to 64 ms in float64
and 36 ms in float32; without the strain, 13 ms.

### The knob that decides multi-GPU scaling

**`skin` (default `"auto"`)**, two or more devices only: on one device there is
no halo to size and the memory probe below does not run. The decomposition is
reused until an atom has moved half the skin, then rebuilt on the host by a
spatial sort whose cost grows with the atom count. Two A40s at 98,304 atoms:

| skin (Å) | ms/step | rebuild | serial fraction | 2-GPU Amdahl ceiling |
|---|---|---|---|---|
| 1.0 (the old default) | 347 | every 13 steps | 76 % | 1.14x |
| 2.0 | 206 | every 40 steps | 54 % | 1.30x |
| 4.0 | 101 | none in 40 steps | 5 % | 1.90x |
| 8.0 | 189 | none | 4 % | 1.92x |

A constant is wrong at both ends (a few thousand atoms rebuild cheaply and pay
for a thick halo), so `"auto"` takes the largest skin whose halo costs at most
25% more nodes than a zero-skin halo, from the cell, the cutoff and the
partition count (`partition.default_skin`): 1.4 Å at 1,536 atoms, 4.3 Å at
98,304, where the measurement put the optimum, and 5.3 Å at 192,000. Pass a
number to override; `evaluator.skin` reports what was used.

**Memory is a measured cap, not a term in the formula.** How many bytes a node
costs is a property of the model and the step, not of the geometry, and spans
a factor of thirty between models:

| model | bytes per node | linear fit error |
|---|---|---|
| uncompressed long-range | 755 kB | 0.3 % |
| compressed long-range | 24.3 kB | 1.2 % |

The first evaluation is a probe at the floor skin, the smallest halo there is;
its peak allocation per node is the coefficient, the footprint is linear in
the node count, and the skin becomes the largest value whose predicted
footprint fits `memory_headroom` (0.8 by default) of what this process could
hold on the device, free memory plus its own cached blocks, so a co-tenant is
accounted for. It is only a ceiling: the geometry rule decides whenever memory
is not binding. Predicted against actual peak at 41,472 atoms uncompressed:
47.4 against 47.23 GiB. The node count the probe counted, not `N/V`, turns the
skin into nodes, which is right for a slab, an interface or a void where a
uniform density is not.

On an out-of-memory the evaluator halves the skin, rebuilds and retries the
step, and that value becomes a ceiling too, `set_cell` under a barostat
included; `evaluator.skin_backoffs` counts the reductions and each one warns.
An explicitly chosen skin is never overridden, and the backoff stops at 0.1 Å,
below which the halo is not what is filling the device.

### The launch phase is single-threaded

A thread per device raced on the lazily initialised libraries underneath
(deepmd-kit's fake-op registration, torch's CUDA linear-algebra loader, warp's
code generation), so the loop runs one partition at a time, in order. Two
things from that work stay, because they are about correctness:

- **`_current_device` around each partition.** The compressed operator binds
  the *current* device rather than its arguments' device: without this, a
  compressed model whose buffers sit on `cuda:1` returns forces wrong by
  5 eV/Å while `cuda:0` is current.
- **One coordinate copy per device, taken before the loop.** Every partition
  on a device gathers its own rows from it, one upload per device instead of
  one per partition.

## What the driver returns

`evaluate_partitioned` assembles on the gather device and returns device
tensors (scalars as 0-d tensors): the per-partition forces, energies, charges,
charge-response forces and Born tensors are scatter-added with `index_add_`
there, the ghost-force reverse communication of a domain-decomposed code done
as a device primitive. Convert explicitly when you want numpy:

```python
from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy

out = evaluate_partitioned(model, coord, atype, cell, n_partitions=2, devices=[...])
out["force"]                      # torch tensor on the gather device
to_numpy(out)["force"]            # numpy (N, 3), one explicit download
```

`DistributedEvaluator` keeps the tensor path: the forces go straight into the
buffer the warp integrator aliases, and `results` converts an entry to numpy
only when it is read. One download per step remains, because the call returns
the energy as a Python float: the energy, the neighbour list's status flags
and, with `compute_virial`, the virial come back in one small copy. On one
device that is the step's only host synchronisation; two or more devices add
the one bool of the halo's Verlet check; an uncompressed model adds its
compact edge count. `tests/test_sync.py` pins these counts.

## Limits

- **`stress` is available, opt-in.** Pass `compute_virial=True` and read
  `evaluator.virial` (eV) or `evaluator.stress` (eV/Å³, ASE's sign
  convention). It is off by default because it costs a strain leaf per
  partition and an extra target in the charge-response backward; the
  `(energy, forces)` return is unchanged either way.

  All three contributions decompose. The short-range virial is the sum of the
  per-partition `energy_derv_c_redu`, since each edge survives in exactly one
  partition. The reciprocal virial is a strain derivative taken on the gather
  device. The charge-response virial comes from straining each partition's
  node coordinates and backpropagating `dE/dq` into the strain leaf, which is
  exact because a ghost's offset is a lattice-vector combination `n A` that
  deforms as `n A (I+ε)ᵀ`, the same transform as a coordinate. Verified
  against the serial virial, itself finite-difference checked, to ~6e-15,
  symmetric, and identical at 1, 2, 4 and 8 partitions.

- **Multi-GPU pays only where the devices are the bottleneck.** Results are
  identical either way: on two distinct A40s, every returned quantity at 2, 3,
  4 and 8 partitions (grids 2x1x1 to 2x2x2) matches the undivided model at the
  run-to-run floor, and NVE energy traces from the same start agree to 1e-5
  meV/atom. Speed depends on where a step spends its time, on two A40s:

  | case | 1 GPU | 2 GPUs | limited by |
  |---|---:|---:|---|
  | compressed long-range, 12,288 atoms | 20.4 ms | 24.8 ms | host (device busy 3.8 ms of the step) |
  | compressed long-range, 41,472 atoms | 48.7 ms | 47.6 ms | host (device busy 10.7 ms) |
  | uncompressed short-range, 12,288 atoms | 222 ms | 141 ms | devices (121 ms each) |

  So compressed runs below roughly 100,000 atoms belong on one GPU, and an
  uncompressed run overlaps at 80–86% across two. The fixed per-partition host
  cost, eager-PyTorch dispatch, is what more partitions multiply.

- **Neighbour lists.** Which builder runs for the descriptor and for the
  charge head, why a rebuilt model must re-resolve `neighbor_graph_method`,
  and why the virial never needs the O(N²) builder:
  [Which neighbour list runs](neighbour-lists.md).
