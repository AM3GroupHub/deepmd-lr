# Running MD over partitions

`LongRangeCalculator` is single-process and single-device. For larger systems,
`DistributedEvaluator` evaluates the model over spatial partitions — one per
GPU:

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

```python
from deepmd_lr.nvalchemi import DistributedEvaluator

evaluator = DistributedEvaluator(
    model, atype, cell,
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    skin=1.0,
    compute_virial=False,       # True also fills .virial and .stress
)
energy, forces = evaluator(positions)      # eV, eV/angstrom
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

nvalchemiops ships half-step kernels and no driver, so the loop is yours. NVE
is four lines:

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
  dynamics entry point documents it that way — the *opposite* of the heavy
  `nvalchemi-toolkit`, whose thermostats took Kelvin, where passing kT
  silently gave a system at rest. For eV, `kT = ase.units.kB * T[K]`.
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
count when the slabs are that thin.

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

### The search is cached too

The decomposition was Verlet-cached from the start; the neighbour search
inside it was not, and ran every step for every partition. It is the expensive
half of a step -- a cell list plus eight blocking device-to-host reads -- so
that fixed cost is what makes more GPUs slower than fewer at small atom counts.

It does not have to run. A partition is rebuilt when some atom has moved half
the skin, and until then every pair within `rcut` of an owned atom is inside a
list searched at `rcut + skin`. So the search happens once per rebuild, at the
padded cutoff, and what it returns -- the integer topology and the static part
of the mask -- is kept on the partition. Each later step recomputes only the
geometry: the edge vectors from the current node coordinates, and the cutoff
mask over them. The cache lives on the `Partition` object, so it is invalidated
by the same criterion that invalidates the halo, with no second rule to keep in
step with the first.

Two details make it exact rather than nearly exact. The edge set is identical:
a padded search masked back to `rcut` gives the same pairs as a plain search at
`rcut`, checked directly. And an edge the cutoff drops has its vector zeroed
rather than left long, which matches what upstream padding already does, so the
descriptor's tables are never asked about a distance beyond the range they were
built for.

Measured on one GB10 with a compressed long-range model, against the same run
searching every step:

| atoms | searched every step | cached | saved |
|---|---|---|---|
| 1,536 | 17.9 ms | 12.5 ms | 30 % |
| 5,184 | 41.2 ms | 29.1 ms | 29 % |
| 12,288 | 92.2 ms | 67.1 ms | 27 % |

Uncompressed the same saving is there but invisible, about 3 %, because the
descriptor dominates everything else.

### Why the launch phase is single-threaded

The neighbour search performs eight blocking device-to-host reads per
partition per step (three in `_max_radius_tuple`, two device queries, the
cell-list size estimate, the cell-list query, and the capacity check in
`nv_search_matrix`). From one thread those serialise the devices whatever the
CUDA streams do, because the host cannot queue partition k+1 while partition
k's search is still running. Giving each *device* its own thread did recover
much of it: on two A40s the launch phase went from 54.7 to 38.1 ms at 98,304
atoms, a parallel gain of 1.31x to 1.88x against an ideal 2x.

It was removed anyway. Every lazily-initialised library under this phase turned
out to be unsafe while it initialises, and each one cost whole rows of a
multi-GPU scaling matrix:

| what is lazy | how it failed |
|---|---|
| deepmd-kit's fake-op registration | `register_fake: the operator ... already has an fake impl registered` |
| torch's CUDA linear-algebra library, loaded by the cell inverse in the neighbour search | `lazy wrapper should be called at most once` |
| warp's code generation | `'Adjoint' object has no attribute ...`, a half-built kernel read by another thread |

Four fixes in -- warming two of them by name, serialising the first pass over
each device, going back to a first pass on every rebuild, retrying a pass that
lost the race -- a 4-GPU V100 run still lost rows, and the short-range case at
3,456 atoms ran at 1612 ms per step against 101 ms single-threaded, with
12,288 atoms hanging past thirteen minutes. A constant factor under 2x does not
justify that. The loop runs one partition at a time, in order, and the whole
class of failure goes with it.

What stayed from that work, because it is load-bearing for correctness rather
than for overlap:

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

Measured effect of removing the host gather, compressed, 98,304 atoms on one
GB10: long-range 344 to 315 ms/step, long-range with Born tensors 442 to 358.
Short-range is unchanged, because its step had only one transfer to remove. On
two H200s the difference is within run-to-run scatter. The gather was never the
ceiling: at 98k atoms a compressed step is 48 % host-side partitioning, 19 %
neighbour search, 17 % edge sort for the compressed kernel, 16 % network, and
under 1 % gather.

## Limits

- **`stress` is available, opt-in.** Pass `compute_virial=True` and read
  `evaluator.virial` (eV) or `evaluator.stress` (eV/Å³, ASE's sign convention).
  It is off by default because it costs a strain leaf per partition and a
  second backward through the charge head; the `(energy, forces)` return
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

- **Multi-GPU pays above a few thousand atoms per card, and not below.**
  Results are identical either way — verified on 2 × A40 (ΔE = 0 at 768 and
  2592 atoms, 9.1e-13 at 6144) and on 3 × RTX PRO 6000 Blackwell, where Born
  charges match the single-GPU answer to ~1e-15 at 1, 2 and 3 real devices and
  stay flat from 81 to 5184 atoms.

  Speed depends entirely on size, because the parallelism is real but the fixed
  per-partition cost is Python. Measured on RTX PRO 6000:

  | system | 1 GPU | 2 GPUs | speedup |
  |---|---:|---:|---:|
  | 12,000 atoms | 121.7 ms | 78.6 ms | 1.55× |
  | 131,712 atoms (weak scaling vs 65,856 on one) | 403.8 ms | 421.5 ms | 96% efficiency |

  Below that it is worthless and can be negative: an older 2 × A40 measurement
  at ≤ 6,144 atoms gave **0.92–1.03×**, i.e. none. The reason is measured, not
  guessed. Put every partition on one device, so total GPU work is constant and
  only the number of launch iterations changes, and the wall time still grows —
  49 → 184 ms going from 1 to 8 partitions — while the GPU work outstanding
  when the call returns is 0.04 ms at every count. The ~11 ms per partition is
  eager-PyTorch dispatch, not GPU compute. At 6,000 atoms per card that is most
  of the step; at 65,856 it is ~2% of it, which is why efficiency climbs from
  81% to 96% across that range. Partitioning multiplies that fixed cost, so
  fewer, larger partitions stay cheaper than many small ones.

- **The short-range path needs `neighbor_graph_method` set for inference.**
  DeePMD-kit resolves it once at *training* startup (default `auto`, which is
  `nv` on CUDA) and does not record it in the checkpoint. A model rebuilt for
  inference therefore falls back to the `dense` O(N²) builder. On the
  6520-atom LiFSI frame that is **48.2 GB and 724 ms** against **4.0 GB and
  201 ms** for `nv`, at a bit-identical energy — and 48 GB does not fit on a
  48 GB card, which is how it first showed up, as an OOM. `load_model` now sets
  it via DeePMD-kit's own `resolve_neighbor_graph_method("auto", device)`, so
  CPU or a missing nvalchemiops degrades to `dense` instead of raising.

- **Only the `dense` builder carries a cell derivative — and nothing here
  needs one any more.** Measured `d(edge_vec)/d(box)`: `dense` 1932.8, `nv`
  **6.7e-13**, `ase` and `vesin` raise. `nv` returning round-off rather than an
  error is the dangerous part: forces stay correct and a virial that leaned on
  it would go silently wrong. `_long_range` gets the virial by differentiating
  the charge prediction through a strain, and now builds the graph from the
  *unstrained* geometry and applies the strain to the finished edge vectors.
  That is exact — the strain is affine and an edge vector is
  `coord[j] + S·box − coord[i]`, every term of which transforms by the same
  `(I+ε)ᵀ` — so the derivative flows through `edge_vec` alone and the O(N)
  builder is used: **8.3 GB and 0.92 s per step**, against 63 GB and 3.0 s when
  it routed through the box.

- **The neighbour-graph choice in `deepmd_lr/neighbors.py`.** DeePMD-kit's
  `build_neighbor_graph` calls itself a "correctness oracle" and is O(N²);
  `deepmd_lr/neighbors.py` swaps in the nvalchemiops GPU cell list wherever it
  pays — see [Which neighbour list runs](neighbour-lists.md#which-neighbour-list-runs) for the
  thresholds. Only the *search* is taken from nvalchemiops: its edge vectors
  come out of a Warp kernel with no gradient, so they are discarded and
  DeePMD-kit's `neighbor_graph_from_ijs` recomputes `edge_vec` from
  `coord`/`box`. Using the kernel's vectors directly would produce a model
  with no forces. Edge sets are identical to the oracle — same count, same
  vectors to 9 decimals, periodic and not. End-to-end this is 1.00–1.77× over
  768–12288 atoms.

