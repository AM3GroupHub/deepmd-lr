# Running MD over partitions

`DistributedEvaluator` computes energies and forces; the MD classes integrate
positions and velocities on the GPU. Build it once with `build_evaluator` and
share it across stages. The default uses the model's single device. Pass
`devices=["cuda:0", "cuda:1"]` to split the system over two GPUs.

Each entry in `devices` defines a partition; repeated device names also create
multiple partitions. A single entry uses the whole system without halo atoms.
See [partitioning](#how-the-atoms-are-split) and the
[low-level interface](#what-the-driver-returns) for details.

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
ev = build_evaluator("model.ckpt.pt", atoms)
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
| `LBFGS` | the same with limited-memory BFGS (nvalchemiops 0.5): a quasi-Newton step under a `maxstep` trust region, no line search; ASE's `memory` (default 6) and `maxstep` (0.2 Å) | — |

`FIRE`, `FIRE2` and `LBFGS` use nvalchemiops kernels with ASE-style interfaces
and unit masses. FIRE caps each atom's step on its own; FIRE2 kicks before
measuring the power, backs off half a step after an uphill step, and scales
the whole step so no atom moves more than `maxstep`; LBFGS takes a quasi-Newton
step from the last `memory` position and force differences, scaled the same
way, reads no energy (which may rise on a step while the force converges) and
restarts from steepest descent when a direction stops descending. Steps to
0.05 eV/Å on 192 water atoms (64 molecules) with the trained model,
nvalchemiops 0.5.0:

| from | `FIRE` | `FIRE2` | `LBFGS` |
|---|---:|---:|---:|
| an MD snapshot (`water_192.xyz`) | 370 | 444 | 215 |
| a packed box pre-relaxed with TIP3P | 149 | 145 | 211 |

From the packed box LBFGS ended 1.3 eV below FIRE2. Its `memory` barely
matters here (3 to 100 pairs all landed within 195 to 224 steps on the
snapshot) and `maxstep` 0.1 against 0.2 changed seven steps.

The units are ASE's: `timestep` in ASE time (`0.5 * units.fs`), `friction` in
1/ASE time, `tdamp` and `pdamp` in ASE time, `pressure_au` in eV/Å³,
`temperature_K` in kelvin. One ASE time unit is 10.181 fs, so the Langevin
relaxation time is `1/friction` of those:

| `friction` | relaxation time | |
|---:|---:|---|
| 0.01 | 1018 fs | too weak to equilibrate a short run |
| 0.1 | 102 fs | the default in example 06's `02` |
| 0.5 | 20 fs | very strong; use under a field |
| `0.01 / units.fs` | 100 fs | |

### Relaxing the cell

`relax_cell` chooses what else may move. The target is `pressure_au` and
`run(smax=...)` sets how close the stress must come to it (default 0.01 GPa).

| `relax_cell` | the cell | for |
|---|---|---|
| `None` (default) | fixed | — |
| `"isotropic"` | scales uniformly, keeping its shape | liquids, which have no shear stiffness: `"full"` lets their cell shear |
| `"orthorhombic"` | the three lengths of a diagonal cell move independently | orthorhombic crystals |
| `"full"` | six cell degrees of freedom, keeping the orientation | crystals of any symmetry |

```python
ev.set_compute_virial(True)                     # the stress drives the cell
FIRE2(ev, atoms, relax_cell="isotropic", pressure_au=1e-4 * units.GPa).run(fmax=0.05)
```

The cell moves on nvalchemiops' own variable-cell kernels: FIRE2's coupled
coordinate-cell step and L-BFGS's coordinate-cell step relax atoms and cell
together, and FIRE, which has no coupled kernel upstream, steps the extended
array of nvalchemiops' cell filter and then lets the atoms ride the cell
affinely, as the coupled kernels do. Each carries six cell parameters, one
triangle of the cell matrix, so the cell cannot rotate while it relaxes.
Around the kernels:

- **The cell force** given to FIRE and FIRE2 is −∂(E + PV)/∂H = −V H⁻ᵀ (σ + P)
  for the rows of ASE's cell, projected onto the mode's directions; a test
  pins it to the finite-difference gradient on a sheared cell. L-BFGS takes
  the stress, projected the same way.
- **The frame.** FIRE and FIRE2 want the lower triangle of ASE's cell, its
  standard form; L-BFGS's kernel holds the lattice vectors as columns and
  wants the upper. A cell not already in the form its kernel needs is rotated
  into it for the run, positions with it, and the result, the log rows and
  the trajectory frames come back in the orientation of the start: the first
  lattice vector keeps its direction and the second its plane. A diagonal
  cell is never rotated; triclinic cells in any orientation are accepted.
  L-BFGS applies `maxstep` in its kernel frame. Restoring the orientation of
  a shearing cell adds a rigid rotation, so reported displacements can exceed
  that cap. Failed runs also restore the frame and discard stale results.
- **The cell's weight.** The cell force is not divided by the atom count
  (`cell_force_scale = 1/N` for FIRE2's kernel), which makes the cell about
  as heavy as one atom, and the L-BFGS chart's scale is the cube root of the
  starting volume for the same reason. In the measurement that chose it (192
  and 12,288 atoms alike) FIRE2 took 390 steps from a 4% stretch and 1000 to
  1570 with the force divided by N.

On 192 water atoms stretched by 4%, to 0.05 eV/Å and 0.01 GPa, with the trained
model on nvalchemiops 0.5.0:

| steps | isotropic | orthorhombic | full |
|---|---:|---:|---:|
| `LBFGS` | 167 | 208 | 239 |
| `FIRE2` | 377 | 344 | 395 |
| `FIRE` | 687 | 362 | 798 |

The new cell goes to the evaluator every step, as under NPT. Partitions are
reused while the [displacement and cell-deformation bound](#reusing-a-partitioning-across-steps)
holds. An MD object built before the relaxation refuses to step afterwards;
build it from the relaxed `atoms`.

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
- **Boundary.** The structure's `pbc` picks the sum, whatever the model was
  trained with ([non-periodic](non-periodic.md)). Along a slab's open axis
  there are no images; `read_structure` pads it to at least 20 Å of vacuum
  (`vacuum=`) and centres the slab. Atoms must stay inside the open faces;
  multiple partitions also require a half-skin margin at each rebuild. NPT,
  NPH and cell relaxation are refused for a slab. A cluster runs on one device
  with no cell and refuses the virial
  ([running MD on a cluster](non-periodic.md#running-md)).
- **PME mesh.** Set from the cell when the evaluator is built, at least
  `ceil(2 alpha L)` points per axis, and kept for the run
  ([alpha and mesh](alpha-and-mesh.md)).
- **The model.** Each evaluator owns its long-range term
  (`ev.long_range_energy`), so its boundary and mesh settings do not overwrite
  those of another evaluator sharing the model. Compression does modify the
  descriptor; shared users must agree on the folded `charge_spin`.
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

Between output and observer intervals, integration adds no host
synchronisation beyond the evaluator's own. Measured against a hand-written
loop on water, one GB10, medians of three interleaved runs (logging every 100
steps had no measurable cost):

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
from deepmd_lr.nvalchemi import Thermo
from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update, velocity_verlet_velocity_finalize,
)

th = Thermo(ev, atoms.get_positions(), velocities=atoms.get_velocities(),
            masses=atoms.get_masses())
dt = th.scalar(0.5 * units.fs)                   # the loop owns time
th.evaluate()
for _ in range(10_000):
    velocity_verlet_position_update(
        th.positions, th.velocities, th.forces, th.masses, dt)
    th.evaluate()
    velocity_verlet_velocity_finalize(th.velocities, th.forces, th.masses, dt)
```

Kernels exist for velocity Verlet, Langevin BAOAB, the Nosé–Hoover chain, MTK
NPT and NPH, and velocity rescaling; each has a class above. The kernels'
conventions, which the classes handle and the signatures do not reveal:

- **Temperature is `k_B T` in energy units, not kelvin**: `kT = ase.units.kB * T`.
- **Temperatures count `3N − 3` degrees of freedom**, assuming the centre of
  mass is at rest. The classes remove its motion at start-up (warning when it
  was large) and Langevin's `fixcm` keeps it at rest; a custom loop must do
  the same or the drift's kinetic energy reads as heat. With fixed atoms the
  count is three per mobile atom and nothing is removed. `Thermo.ndof` holds
  whichever applies.
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
- **The tensor kernels want a flat `vec9` layout**, and `kinetic_tensors` is
  2-D `(B, 9)`; only `cells` is `mat33`. `vec9d` comes from nvalchemiops, not
  from Warp (`wp.vec9d` does not exist). `Thermo.update_pressure` handles it.
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

The evaluator selects its path from `len(devices)`, even if entries name the
same physical GPU:

| device entries | nodes | role of `skin` |
|---|---|---|
| one (default) | real atoms only; periodic images are edge shifts | inactive; no halo or memory probe |
| two or more | owned atoms plus explicit ghosts within `rcut + skin` | allows the partitioning to be reused |

Both paths search for neighbours at **`rcut` every evaluation**. Skin caches
the partition membership, not the pair list.

For multiple partitions, `decomposition_grid` minimises each block's halo
volume and cuts at equal atom count: 2 × 2 × 1 for four partitions of a cubic
box, 2 × 2 × 2 for eight, or slabs along a sufficiently long axis. Nodes on
the busiest device, water at the auto skin:

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
simply triggers a rebuild. With one partition the cell-list kernel folds
the images into the pair shifts itself, every step.

### What is split and what is not

| piece | cost at 6520 atoms | treatment |
|---|---:|---|
| descriptor + short-range | 883 ms | partitioned |
| charge prediction | (same pass) | partitioned |
| reciprocal sum + charge projection | 73 ms | gathered onto one device |
| charge-response force | — | partitioned |

Independent reciprocal sums per partition miss interactions between
partitions. Charges are therefore gathered for one global sum. In the
measurement above, most of the 73 ms was gathering and projection; PME itself
took 1.5 ms ([Cost](cost.md)). The charge-response backward through the raw
charge head stays partitioned because each raw charge depends only on edges
within `rcut`. The global total-charge projection is included in the gather
stage.

### Reusing a partitioning across steps

With multiple partitions, ghosts are gathered within `rcut + skin`. At fixed
cell the rebuild criterion is `2·max displacement > skin`.
`repartition_every=1` forces a rebuild each step; it changes timing, never
results, and a test asserts exactly that.

**Cell deformation also counts toward the bound.** Every node is an atom's folded
position `f_i = r_i − n_i H` plus a lattice shift `S H`, with `n_i` and `S`
kept as integers. `set_cell` re-expresses the shifts in the new cell, and the
Verlet bound counts each atom's motion in its folded frame, `Δr_i − n_i ΔH`,
plus `(max|S| + √3) ‖ΔH‖` for the ghosts and the images just outside the halo.
Small cell changes can reuse the partitioning; larger changes rebuild it.
In measured water runs, rebuilding every step cost 66 against 38 ms/step at
12,288 atoms and 154 against 97 at 41,472.

The bound is evaluated on the device and read back as one bool per step.
The whole-system path skips this check and reuses its node set and buffers.
Its `rebuild_count` counts initialisation and any forced rebuilds.

**Cell and pressure updates stay on the device.** The barostat's cell is a device
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

The search runs every step, reusing buffers sized by the first search at 1.2×
the widest row and 1.15× the edge count; the edge count stays a device scalar,
`edge_mask` marks where the real edges end, and a device status word reports
an overflowed row or edge buffer, on which the evaluator widens it by a
quarter, repeats the step, warns and counts it in `evaluator.list_growths`.
Initial sizing and overflow recovery synchronise with the host. A warm
compressed search fits within the evaluator's existing status readback.
A compressed model takes the kernel's float32 vectors over the full buffers,
since its kernels walk the CSR
rows and never see the tail, unless the charge response needs `edge_vec`
differentiable; an uncompressed model gets vectors in its own precision on a
compact edge axis, at the cost of one edge-count download per partition, because
its descriptor processes every edge it is handed.

Measured on water with the compressed short-range model
(`DP_CUDA_INFER=2`), ms per MD step, one device:

| GPU | atoms | step | neighbour graph | forward | peak GiB |
|---|---:|---:|---:|---:|---:|
| GB10 | 10,368 | 3.6 | 1.2 | 2.1 | 0.1 |
| GB10 | 169,344 | 53 | 13.6 | 39.1 | 1.9 |
| H200 | 10,368 | 2.6 | 0.8 | 1.2 | 0.2 |
| H200 | 1,956,864 | 97 | 26 | 68 | 22 |

At 1.96M atoms on the H200, the list's 26 ms comprises 12.8 ms for the query,
7.1 for the fill and 6.2 for source CSR, over 175M edges.

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

**`skin` (default `"auto"`)** controls halo reuse with two or more partitions.
It is inactive on the whole-system path. Increasing it trades fewer host
rebuilds for more ghost nodes. Two A40s at 98,304 atoms:

| skin (Å) | ms/step | rebuild | serial fraction | 2-GPU Amdahl ceiling |
|---|---|---|---|---|
| 1.0 | 347 | every 13 steps | 76 % | 1.14x |
| 2.0 | 206 | every 40 steps | 54 % | 1.30x |
| 4.0 | 101 | none in 40 steps | 5 % | 1.90x |
| 8.0 | 189 | none | 4 % | 1.92x |

`"auto"` takes the largest skin whose halo costs at most
25% more nodes than a zero-skin halo, from the cell, the cutoff and the
partition count (`partition.default_skin`): 1.4 Å at 1,536 atoms, 4.3 Å at
98,304, where the measurement put the optimum, and 5.3 Å at 192,000. Pass a
number to override; `evaluator.skin` reports what was used.

**Memory caps the geometry-based skin.** Measured bytes per node differ
substantially between models:

| model | bytes per node | linear fit error |
|---|---|---|
| uncompressed long-range | 755 kB | 0.3 % |
| compressed long-range | 24.3 kB | 1.2 % |

The first evaluation probes peak allocation at the minimum auto skin. The
measured cost per node caps the halo at `memory_headroom` (default 0.8) of
available memory, including this process's cached blocks. The measured node
count accounts for slabs and voids without assuming uniform density. At
41,472 atoms uncompressed, the predicted peak was 47.4 GiB against 47.23 GiB
measured. The geometry rule still applies when memory is not limiting.

On an out-of-memory the evaluator halves the skin, rebuilds and retries the
step, and that value becomes a ceiling too, `set_cell` under a barostat
included; `evaluator.skin_backoffs` counts the reductions and each one warns.
An explicitly chosen skin is never overridden, and the backoff stops at 0.1 Å,
below which the halo is not what is filling the device.

### The launch phase is single-threaded

One host thread queues the partitions in order, avoiding races in lazy
library initialisation; device work can overlap across GPUs. Two details
are required for correctness and performance:

- **`_current_device` around each partition.** The compressed operator binds
  the *current* device rather than its arguments' device: without this, a
  compressed model whose buffers sit on `cuda:1` returns forces wrong by
  5 eV/Å while `cuda:0` is current.
- **One coordinate copy per device, taken before the loop.** Every partition
  on a device gathers its own rows from it, one upload per device instead of
  one per partition.

## What the driver returns

`DistributedEvaluator` accepts a torch tensor (including one mutated in place
by an integrator) or an array-like of positions. It returns energy in eV as a
Python float and forces in eV/Å as a float64 tensor on the model's device.
`wp.from_torch` aliases the force buffer without copying. `results` includes
the energy split and charges and converts entries to NumPy only when read.
`masses()` derives masses from the model's `type_map`.

```python
from deepmd_lr.nvalchemi import DistributedEvaluator

ev = DistributedEvaluator(model, atype, cell, devices=["cuda:0"])
energy, forces = ev(positions)
```

`evaluate_partitioned` assembles on the gather device and returns device
tensors (scalars as 0-d tensors): the per-partition forces, energies, charges,
charge-response forces and Born tensors are scatter-added with `index_add_`
there, the ghost-force reverse communication of a domain-decomposed code done
as a device primitive. Convert explicitly when you want numpy:

```python
from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy

out = evaluate_partitioned(model, coord, atype, cell, n_partitions=2,
                           devices=["cuda:0", "cuda:1"])
out["force"]                      # torch tensor on the gather device
to_numpy(out)["force"]            # numpy (N, 3), one explicit download
```

Without supplied `partitions`, this low-level helper calls `partition_system`,
including explicit periodic ghosts even for `n_partitions=1`. Use
`DistributedEvaluator` for whole-system MD and persistent buffers.

In a warm evaluation without rebuilding or buffer growth, the evaluator reads
energy and list status in one small copy (also the virial if requested).
Multiple partitions add one bool for the halo check; an uncompressed model
also reads one compact edge count per partition. Logging, observers and
checkpoints can download more. [`test_sync.py`](../tests/test_sync.py) checks
the evaluation counts.

## Limits

- **`stress` is available, opt-in.** Pass `compute_virial=True` and read
  `evaluator.virial` (eV) or `evaluator.stress` (eV/Å³, ASE's sign
  convention). It is off by default because it costs a strain leaf per
  partition and an extra target in the charge-response backward; the
  `(energy, forces)` return is unchanged either way.

  All three contributions decompose. The short-range virial is the sum of the
  per-partition `energy_derv_c_redu`, since each edge survives in exactly one
  partition. The reciprocal virial is a strain derivative taken on the gather
  device. The charge-response virial strains each partition's edge vectors
  and backpropagates `dE/dq` into the strain leaf. This includes periodic
  offsets because every term of `r_j + S H − r_i` deforms alike. Verified
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

  These compressed cases gain little from a second GPU; the uncompressed
  run overlaps at 80–86% across two. More partitions multiply eager-PyTorch
  dispatch costs. A separate NVT series on water, two A40s, ms/step:

  | atoms | uncompressed, 1 GPU | 2 GPUs | compressed, 1 GPU | 2 GPUs |
  |---:|---:|---:|---:|---:|
  | 12,288 | 184 | 145 | 14.8 | 18.4 |
  | 41,472 | 574 | 399 | 36.2 | 36.5 |
  | 98,304 | — | — | 75.9 | 72.3 |

  Uncompressed, two cards help from about ten thousand atoms; compressed, the
  host's share dominates below ~100,000 atoms and one card is as fast.
  Measure the intended system and hardware before adding GPUs.

- **Neighbour lists.** Which builder runs for the descriptor and for the
  charge head, why a rebuilt model must re-resolve `neighbor_graph_method`,
  and why the virial never needs the O(N²) builder:
  [Which neighbour list runs](neighbour-lists.md).
