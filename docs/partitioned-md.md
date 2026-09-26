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

`Thermo` is the other half an MD loop needs — it owns the state those kernels
mutate (positions, velocities, masses, forces, cell, timestep, each a warp
array bound once) and reports the thermodynamics, which nvalchemiops does not
do at all. It takes `temperature` in kelvin and builds the `k_B T` array
itself, so nothing downstream has a reason to write that conversion.
`examples/06-md_nvalchemi` has a loop per ensemble.

```python
from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure

atoms = read_structure("start.xyz")
ev = build_evaluator("model.ckpt.pt", atoms, compute_virial=True)
th = Thermo(ev, atoms.get_positions(), timestep=0.5, temperature=300.0,
            columns=("step", "time", "temp", "press", "vol"))
```

**That is the whole object.** It owns the partitioning, the ghost halo, the
per-device replicas and the decision of when to rebuild — and nothing else.
No timestep, no ensemble, no thermostat, no velocities: those belong to
whoever is integrating. Swapping NVE for NPT changes your loop and not this.

`positions` may be a torch tensor — including the one an integrator is
mutating in place — or anything `np.asarray` accepts. Forces come back as a
float64 tensor on the model's device, which `wp.from_torch` turns into a
zero-copy Warp view, so **the trajectory never leaves the GPU** for the
integrator's sake. `evaluator.results` carries the energy split and charges;
`evaluator.masses()` reads the element order off the model's own `type_map`.

**There is deliberately no ASE calculator in this path.** ASE's interface
returns energy and forces one configuration at a time and its integrators
cannot touch on-device state, which is the point of this path.

### Driving it with nvalchemiops

nvalchemiops ships half-step kernels and no driver, so the loop is yours; for
a ready-made one with restarts and a GPU-count benchmark, see
[examples/07-multi-gpu-md](../examples/07-multi-gpu-md). NVE is four lines:

```python
from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update, velocity_verlet_velocity_finalize,
)
vec = lambda t: wp.from_torch(t, dtype=wp.vec3d)

energy, forces = evaluator(positions)
for _ in range(steps):
    velocity_verlet_position_update(
        vec(positions), vec(velocities), vec(forces),
        wp.from_torch(masses), wp.from_torch(dt),
    )
    energy, forces = evaluator(positions)
    velocity_verlet_velocity_finalize(
        vec(velocities), vec(forces), wp.from_torch(masses), wp.from_torch(dt),
    )
```

Available kernels: velocity Verlet (NVE), Langevin BAOAB, Nosé–Hoover chain,
NPT and NPH (MTK), and velocity rescaling.

Three conventions to respect, none of them guessable from the signatures, all
found by running into them:

- **Temperature is `k_B T` in energy units, not Kelvin.** Every nvalchemiops
  dynamics entry point documents it that way. For eV,
  `kT = ase.units.kB * T[K]`.
- **Half-step factors are inconsistent between kernels.** Measured:
  `nhc_velocity_half_step` and `velocity_verlet_position_update` apply their
  own factor of ½ (`dv = (F/m) dt/2`), `nhc_position_update` does not
  (`dr = v dt`), and `nhc_thermostat_chain_update` advances `eta` by
  `half_dt` internally. So the **full `dt`** goes to every call, and the
  chain update is called once at each end of the step.
- **`ndof` dtypes disagree.** `nhc_compute_masses` requires int32 while
  `nhc_thermostat_chain_update` and `nhc_compute_chain_energy` require
  float64. That is a Warp launch error, not a cast, so keep both.

Units: the model works in angstrom and eV. Pair that with masses in amu and a
timestep in ASE time units (`A sqrt(amu/eV)`) and the set is self-consistent.
Take both constants from ASE — `from ase.units import fs, kB`.

With a thermostat, watch the **MTK invariant** rather than the physical total
— a thermostat moves the latter on purpose:

```
E_pot + E_kin + sum_k Q_k eta_dot_k^2 / 2 + N_dof kT eta_1 + kT sum_{k>1} eta_k
```

> One caveat, measured and unresolved. That invariant drifts a few meV/atom
> per 20 fs at strong Nosé–Hoover coupling, and the drift is **independent of
> the timestep** — 2.2237, 2.2355, 2.2385 meV/atom at dt = 0.5, 0.25, 0.125
> fs — and unchanged by the Yoshida–Suzuki `nloops`. So it is not integration
> error. The same system under NVE conserves cleanly and as `dt^2` (−0.0159,
> −0.0040, −0.0010), and the chain-energy expression matches the analytic
> formula exactly, so it is neither the forces nor the bookkeeping. It scales
> as the square of the coupling strength, so weak coupling is the safe regime.
> Treat NVE as the trustworthy conservation check.

### How the atoms are split

Into a grid of blocks, not slabs. What a device has to hold is its owned block
plus a shell of ghosts `rcut + skin` deep on every side, and that shell is the
part that does not shrink when the block does. Slabs keep two faces at full
cell size however thin they get; cutting more axes keeps every face smaller.
`decomposition_grid` picks, per partition count, the factorisation that
minimises one block's halo box: slabs for two, 2 x 2 x 1 for four, 2 x 2 x 2
for eight on a cubic box, and slabs again along a long axis when the box is
elongated enough. Each cut is at equal atom count, one axis at a time.

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
