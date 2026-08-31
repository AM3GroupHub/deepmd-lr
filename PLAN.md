# Long-Range Electrostatics for DPA4C via nvalchemi PME

## Context

DPA4C is a strictly local descriptor: an atom's energy depends only on neighbours within
`rcut`, so the model cannot represent Coulomb interactions beyond the cutoff. We want to add
an explicit long-range electrostatic term computed by NVIDIA's PME implementation
(`nvalchemi-toolkit-ops`), trained jointly with the short-range DPA4C energy.

The central question was whether this requires forking deepmd-kit. **It does not.** Verified
against the released `deepmd-kit==3.2.0` aarch64 wheel:

- `deepmd/pt_expt/__init__.py` calls `load_entry_point("deepmd.pt_expt")` — a third-party pip
  package can register itself into the pt_expt backend at import time.
- `deepmd/dpmodel/model/model_factory.py:430` ends model-type dispatch with
  `base_model.get_class_by_type(model_type).get_model(data)` — a plugin-registered
  `model.type` is reachable from a stock install with no patch.
- `deepmd/utils/argcheck.py` exposes `descrpt_args_plugin`, `fitting_args_plugin`,
  `model_args_plugin`, `loss_args_plugin` — the JSON-schema side is extensible too. (Schema
  rigidity is normally what forces a fork; here it doesn't.)

The one real architectural constraint: `NeighborGraph`
(`deepmd/dpmodel/utils/neighbor_graph/graph.py:36`) states *"Geometry enters the model ONLY
through `edge_vec`"*, and neither `forward_common_atomic` nor `call_common_lower` receives the
cell. PME needs absolute coordinates + cell, so the long-range term **must attach at the
top-level model**, where `call_common(coord, atype, box, ...)`
(`deepmd/dpmodel/model/make_model.py:278`) does have the box. That is why this plan registers a
new *model type* rather than a new *atomic model*.

**Decisions taken** (confirmed with the user):
- Charges: environment-dependent charge head on the DPA4C descriptor, projected to satisfy the
  total-charge constraint.
- Training: joint end-to-end, one loss over `E_sr + E_lr`.
- MD target: nvalchemi / ASE (both drive the top-level forward, which has the box). LAMMPS is
  explicitly out of scope — its pair-style path cannot see the cell.

## Environment

Verified on this machine:

| Property | Value |
|---|---|
| Arch | `aarch64` (GB10 / DGX Spark class) |
| GPU | NVIDIA GB10, compute capability **12.1** (sm_121) |
| CUDA toolkit | 13.0 |
| glibc | 2.39 |
| RAM | 119 GB unified |
| `dpmd` env | conda, Python **3.11.16**, currently empty |

Constraints this imposes:
- `deepmd-kit==3.2.0` pins `torch==2.11.0.*`; its wheel is built against torch 2.11.0.
- sm_121 requires a CUDA 12.8+/13 build → use the **cu130** PyTorch index.
- `torch-2.11.0+cu130-cp311-cp311-manylinux_2_28_aarch64.whl` **exists** — confirmed.
- The PyPI deepmd-kit aarch64 wheel is `DP_VARIANT=cpu` (its `run_config.ini` confirms; no CUDA
  in `libdeepmd_op_pt.so`). This is fine: training runs in **eager PyTorch** on GPU, and the
  compiled CUDA mega-kernels are opt-in via `DP_CUDA_INFER` and inference-only
  (`deepmd/kernels/utils.py:cuda_infer_level`). No source build needed.

### Install into `dpmd`

```bash
# 1. PyTorch 2.11.0 + CUDA 13 (aarch64) — version pinned by deepmd-kit 3.2.0
conda run -n dpmd pip install --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.11.0+cu130"

# 2. deepmd-kit (prebuilt aarch64 wheel; core deps only, avoids pulling mpich)
conda run -n dpmd pip install "deepmd-kit==3.2.0"

# 3. deepmd's torch-path extras, explicitly.
#    mpich is REQUIRED, not optional: the PyPI wheel is cibuildwheel-built, so
#    deepmd/pt/cxx_op.py calls load_mpi_library() unconditionally at import and
#    resolves libmpi.so through mpich's package metadata. Without it,
#    `import deepmd.pt_expt.descriptor` raises PackageNotFoundError.
conda run -n dpmd pip install "e3nn>=0.5.9" "vesin[torch]" mpich

# 4. nvalchemi PME engine (pure-python wheel + aarch64 warp)
conda run -n dpmd pip install "nvalchemi-toolkit-ops>=0.4.1" "warp-lang>=1.13.0"

# 5. data / MD / dev tooling
conda run -n dpmd pip install "dpdata>=0.2.7" "ase>=3.23.0" \
    pytest pytest-cov ruff pre-commit

# 5b. reference PME implementation, used only to validate our own PME usage
conda run -n dpmd pip install "torch-pme==0.3.2"

# 6. optional: nvalchemi MD driver (for the ASE/nvalchemi MD path later)
conda run -n dpmd pip install nvalchemi-toolkit
```

Wheel availability confirmed for all of the above on `cp311` / `linux_aarch64`:
`nvalchemi_toolkit_ops-0.4.1-py3-none-any.whl` (pure python),
`warp_lang-1.17.0-py3-none-manylinux_2_34_aarch64.whl` (glibc 2.39 ≥ 2.34 ✓),
`deepmd_kit-3.2.0-py37-none-manylinux_2_28_aarch64.whl`.

**Smoke test before writing any code:**

```bash
conda run -n dpmd python -c "
import torch, warp, deepmd
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
print('pme ok')
from deepmd.pt_expt.descriptor.dpa4c import *   # dpa4c reachable
print('dpa4c ok')
"
```

The riskiest line is warp compiling for sm_121; if it fails, `warp.init()` will say so and the
fix is a newer `warp-lang`, not a change to this design.

## Architecture

Register a new top-level model type `dp_lr` in our own package. It owns a DPA4C descriptor, two
heads, and the PME term:

```
                 ┌─> ener fitting ──────> E_sr,i
descriptor(dpa4c)┤
                 └─> charge fitting ────> q~_i ──project──> q_i   (Σq_i = Q_total)
                                                             │
                            coord, box ──────────────────────┤
                                                             v
                                       particle_mesh_ewald(...) -> E_lr,i

E_total = Σ_i (E_sr,i + E_lr,i);  F = -dE/dr, virial from dE/dcell  (autograd)
```

Sharing one descriptor between both heads is the point: charges become environment-dependent for
free, and the marginal cost over plain DPA4C is one extra fitting network.

### Why these specific reuse points

| Need | Reuse | Path |
|---|---|---|
| Model registration | `BaseModel.register("dp_lr")` + `get_model` classmethod | `deepmd/dpmodel/model/base_model.py:284` |
| Backend model class | subclass `EnergyModel` | `deepmd/pt_expt/model/ener_model.py:36` |
| Charge head | subclass `InvarFitting` (`dim_out=1`) | `deepmd/dpmodel/fitting/invar_fitting.py:32` |
| Total charge per frame | existing `charge_spin` (nf × 2) tensor | `deepmd/utils/charge_state.py` |
| JSON schema | `model_args_plugin` / `fitting_args_plugin` | `deepmd/utils/argcheck.py:3389`, `:2767` |
| Loss | stock `ener` loss — output keys unchanged | no new code |
| PME | `particle_mesh_ewald(...)` | `nvalchemiops.torch.interactions.electrostatics.pme` |

Note the PyTorch bindings live at `nvalchemiops.torch.interactions.electrostatics`; the
`nvalchemiops/interactions/electrostatics` path is the framework-agnostic Warp layer and its
torch re-exports there are deprecated.

`particle_mesh_ewald` gives us exactly what the deepmd contract wants:
- `energy_reduction="atom"` → per-atom energies, matching the atomic-energy contract.
- `compute_charge_gradients=True` → dE/dq, to backprop into the charge head.
- `_pme_convolve_double_backward` exists → double backward works, which is required to train
  against **force** labels through an autograd-derived force.

## Repository layout

```
deepmd-lr/
├── pyproject.toml            # entry point: [project.entry-points."deepmd.pt_expt"]
├── README.md
├── PLAN.md                   # this document
├── deepmd_lr/
│   ├── __init__.py
│   ├── register.py           # entry-point target; import-only (see Phase 1)
│   ├── argcheck.py           # model_args_plugin / fitting_args_plugin registrations
│   ├── model/lr_model.py     # LongRangeEnergyModel(EnergyModel), registered "dp_lr"
│   ├── fitting/charge.py     # ChargeFittingNet + project_to_total_charge
│   └── pme/term.py           # PMEEnergy: particle_mesh_ewald + eV conversion
├── tests/
│   ├── test_pme_physics.py
│   ├── test_plugin_registration.py
│   ├── test_charge_head.py
│   └── test_pme_term.py
└── examples/water_lr/input.json
```

`pyproject.toml` entry point:

```toml
[project.entry-points."deepmd.pt_expt"]
deepmd_lr = "deepmd_lr.register"   # a module, NOT a callable
```

## Implementation phases

### Phase 0 — environment + baseline — **DONE, verified on this machine**
1. Install per the block above. Verified working:
   `torch 2.11.0+cu130`, `deepmd-kit 3.2.0`, `warp-lang 1.17.0`,
   `nvalchemi-toolkit-ops 0.4.1`. `torch.cuda` sees the GB10 at capability
   `(12, 1)`; warp initialises against `sm_121` and runs kernels correctly.
   DPA4C imports and the pt_expt registries populate.
2. `nvalchemi`'s own PME test-suite passes here: 78 CUDA tests, plus 146
   reference tests against `torch-pme` once that package is installed (they
   silently skip without it -- install it, or the tests that check *absolute*
   energies never run).
3. PME validated against the NaCl Madelung constant: **6.7e-7** relative error
   (216 atoms, alpha=2.5, spline order 6, 96^3 mesh; residual is real-space
   cutoff truncation). `gradcheck` passes for positions and for charges, and
   `gradgradcheck` passes with `nondet_tol=1e-8` -- so the double backward
   needed to train against force labels is sound.

> **API hazard found during this validation -- `spline_order` must be 3-6.**
> At `spline_order >= 7` `pme_reciprocal_space` **silently drops the entire
> reciprocal-space sum** and returns only the self-energy term
> (-alpha/sqrt(pi) * sum q_i^2), with no error raised. The result looks
> plausible and varies with alpha, so it is easy to mistake for a convergence
> problem: it produced a 7% Madelung error that was invariant to mesh size.
> Pin `spline_order=6` and treat it as a validated constant, not a tunable.

Other usage notes established here:
- PyTorch bindings are `nvalchemiops.torch.interactions.electrostatics.*`.
- `particle_mesh_ewald` **requires** a neighbour list for its real-space part
  (`neighbor_list` + `neighbor_ptr` + `neighbor_shifts`, or the matrix form);
  build it with `nvalchemiops.torch.neighbors.neighbor_list`, which needs an
  explicit `max_neighbors` or it raises `NeighborOverflowError`.
- Both `mesh_spacing` and `mesh_dimensions` work and agree.
- `estimate_pme_mesh_dimensions` expects a batched (array) `alpha`; passing a
  Python float raises `TypeError`.

### Phase 1 — plugin skeleton — **DONE**
The package registers a `dp_lr` model type that delegates to plain DPA4C with
no long-range term, so plugin wiring is proven independently of the physics.
`dp --pt-expt train examples/water_lr/input.json` runs end to end on the water
dataset, the loss descends, and the checkpoint records `type: "dp_lr"` and
rebuilds as `LongRangeEnergyModel` with 0 missing/unexpected weights.

Two things worth knowing before Phase 2 touches this machinery:

> **The entry point is loaded, never called.** `load_entry_point` is
> `[ep.load() for ep in eps]` — it imports the target and discards the result.
> Registration therefore has to be a *module import side effect*; an entry
> point naming a `register()` function silently does nothing, because the
> function object is loaded and dropped. `pyproject.toml` names the module
> `deepmd_lr.register`, whose body is import-only.

> **Registering the model class is only half of it.** An input file is
> validated against the `dargs` schema before any model is built, so a type
> missing from `model_args_plugin` is rejected at `dp train` time no matter
> what the model registry says. Both registrations live in
> `deepmd_lr/register.py`'s two imports.

**Correction to the phase gate.** This plan originally required Phase 1 to
reproduce the baseline *bit-for-bit*. That is not achievable and the gate was
wrong: the descriptor's segment/scatter reductions sum in a nondeterministic
order on GPU, so evaluating one model twice on identical inputs already
differs by ~3.5e-15. The tests instead measure that floor in-process and
require the `dp_lr`-vs-standard difference to sit at or below it — a stronger
statement anyway, since it says the two models differ by no more than one
model differs from itself. State dicts are separately asserted bitwise equal.

Note also that two *training runs* of the same config do not produce identical
loss curves (batch sampling and initialization are not fully seed-locked
across processes), so the end-to-end run is a sanity check; the equivalence
claim rests on the unit test, which fixes weights and inputs.

### Phase 2 — charge head — **DONE**
`ChargeFittingNet` (a `dim_out=1` `InvarFitting`, `var_name="charge"`) is
registered as fitting type `charge` in both the pt_expt fitting registry and
the `dargs` schema. Charges from a real DPA4C descriptor are verified
invariant under translation, rotation and periodic wrapping, and equivariant
under permutation — with explicit guards that those tests cannot pass
vacuously (the output must have real spread across atoms and must respond to
moving an atom, so neither a constant head nor fixed per-element charges
would satisfy them).

> **The projection cannot live inside the fitting.** The plan assumed the
> head would apply the total-charge constraint itself. It cannot: on the
> graph route `call_graph` receives the flat node axis `(N,)` and reuses the
> dense forward by treating *each node as its own single-atom frame*, so it
> has no way to tell which atoms share a frame — and `charge_spin` is never
> passed to a fitting at all. A per-frame reduction is therefore
> inexpressible there.
>
> This is the same shape of constraint as the cell: frame-level information
> lives at the model, not below it. So the head emits **raw** charges, and
> `project_to_total_charge` is a separate pure function applied at the model
> level in Phase 4, where both frame structure and `charge_spin` are
> available. Splitting it this way also made the projection testable on its
> own, independently of any network.

The projection is
`q_i = q̃_i + (Q_total - Σ_j q̃_j) / N`, enforced *exactly* rather than as a
penalty: a penalty would let a frame drift off-neutral, and a non-neutral
cell makes the Ewald reciprocal sum conditionally convergent — precisely the
pathology the long-range term exists to avoid. It is uniform (so charge
*differences* survive), permutation-equivariant, `gradcheck`-clean, and
guarded against NaN on a fully-padded frame.

### Phase 3 — long-range term — **DONE** (revised)
`deepmd_lr/pme/term.py` exposes `LongRangeEnergy`, returning per-atom
long-range electrostatic energy in eV for `(nf, nloc, 3)` coordinates,
`(nf, nloc)` charges and `(nf, 3, 3)` cells.

> **It supplies only the reciprocal half of the Ewald split.** The first
> implementation computed the *full* sum (real + reciprocal + self). That was
> wrong for this design: the screened real-space part decays as
> `erfc(alpha r)/r` and lives entirely inside the descriptor cutoff, so the
> short-range network already represents it. Including it asked the network
> to learn to *subtract* something it could describe directly. This is the
> same decomposition DeePMD-kit's own DPLR uses.
>
> The correction paid off measurably: the benchmark force gain went from
> 2.6x to **4.9x**, and an energy regression on real water (0.79x) turned
> into a 1.29x improvement.
>
> It also simplified the term considerably. With no real-space part there is
> **no neighbour list**: no cutoff, no capacity to size, no `rcut < L/2`
> constraint — and the batched-`naive` neighbour-list bug found in the first
> pass is now unreachable by construction. The test suite got 3x faster.

**Validation.** The oracle is DeePMD-kit's own `ewald_reciprocal_energy` —
the function its DPLR path uses — matched to 1e-7. That single comparison
pins the split, the units, and the self-term convention at once (nvalchemi's
`pme_reciprocal_space` returns k-space *minus* self; DeePMD-kit keeps it in,
so the term adds it back).

**`alpha` is a modelling choice here, not a free parameter.** In a full Ewald
sum the total is alpha-independent. For the reciprocal half alone, alpha
decides how the work is divided: the network handles `erfc(alpha r)/r` and
this term handles the rest. It must be large enough that the network's share
has died inside the descriptor cutoff — `alpha = 0.4` leaves `erfc(2.4) =
7e-4` at 6 Å — and must match between training and any later MD.

A physical corollary worth knowing: the reciprocal term only carries weight
where the charge density has *long-wavelength* structure. A tight rocksalt
lattice at `alpha = 0.5` has a reciprocal energy of ~1e-9 eV, because its
charge ordering is short-wavelength and `exp(-k^2/4 alpha^2)` suppresses it.

**Other guards.** `box=None` and zero-volume cells are rejected with messages
naming the physical reason; `spline_order >= 7` is refused at construction
(Phase 0's silent-failure mode); unknown parameters raise rather than being
absorbed (a stale `rcut` in a config silently ran different physics until
this was tightened); padding atoms get zero charge, which removes them from
both the reciprocal sum and the self term exactly.

### Phase 4 — joint model — **DONE**
`model.type: "dp_lr"` with an optional `long_range` block containing
`charge_net` and `pme`. Omit the block and the model is exactly the Phase 1
model, which is what keeps "the long-range term is the only thing that
changed" a testable property. The forward returns the usual keys plus
`charge`, `energy_short_range` and `energy_long_range` as diagnostics.

> **The long-range energy cannot reuse DeePMD-kit's force machinery.** The
> obvious implementation — add the PME per-atom energy to the atomic
> output — is silently wrong. On the graph route force and virial come from
> `edge_force_virial`, which differentiates with respect to `edge_vec`. The
> PME energy is not a function of `edge_vec` (it depends on absolute
> coordinates and the cell), so it would contribute exactly **zero** force
> and virial while still shifting the energy: energies move, forces do not,
> and the model quietly stops being conservative.
>
> The long-range branch therefore carries its own derivatives — `-dE/dr` for
> the force, and `-dE/dε` under a symmetric strain applied to coordinates and
> cell together for the virial, matching the convention read off
> `edge_force_virial`. Both are verified against finite differences of the
> *total* energy. The same finite-difference check on the pure short-range
> model reproduces DeePMD-kit's own virial to 2e-10, which is what makes it a
> trustworthy oracle for the joint one.

> **A spacing-derived PME mesh makes the energy discontinuous in the cell.**
> The mesh is `ceil(L / mesh_spacing)`, so a cell length sitting on a
> multiple of the spacing gains a whole grid point under an infinitesimal
> strain. This first appeared as a 2% virial error that looked like a bug in
> the strain derivative; the derivative was right and the energy was
> genuinely discontinuous where the finite difference sampled it. `PMEEnergy`
> now accepts explicit `mesh_dimensions`, which is the right setting whenever
> the cell can change (NPT, or any strain derivative). Harmless at fixed cell.

**Cost.** A frame runs the descriptor twice: the short-range path consumes
and discards its own descriptor output, and the charge head needs it again.
Measured at ~3x per training step on small systems (24 atoms: 15 ms → 45 ms;
96 atoms: 14 ms → 46 ms), though these sizes are launch-overhead dominated so
the ratio should improve with system size. Sharing one pass means giving the
atomic model a second, non-reducible `charge` output; the carry-through works
(`fit_output_to_model_output_graph` does `model_ret = dict(fit_ret.items())`
and skips reduction for non-reducible keys), but `init_out_stat` sizes bias
and std buffers from *every* output key, which would entangle charges with
the out-stat machinery that has no charge labels. Deferred as a profiling-led
optimization rather than done speculatively.

**A gap the unit tests missed.** The trainer passes coordinates flattened to
`(nf, nloc * 3)` and the cell to `(nf, 9)`, while eager callers pass
`(nf, nloc, 3)` and `(nf, 3, 3)`. The tests only exercised the eager shapes,
so the first real training step failed with `batch1 must be a 3D tensor`.
Both conventions are now pinned by `TestCallingConventions`.

### Phase 5 — training + validation — **DONE**
Full write-up in `experiments/RESULTS.md`. Headlines:

**The data situation is a finding in itself.** No periodic dataset exists on
this machine: `omol25`, `opoly26` and all three `electrolyte` sets are
non-periodic molecular clusters (`pbc = [F F F]`). PME needs 3D periodicity,
so none of it can drive this work as built. See the note at the end of this
section.

**Controlled benchmark.** A periodic molten-salt-like system (32 ions, 14 Å
cell) whose ground truth is exactly `E_repulsion(<5 Å) + E_Coulomb(±1)`, with
labels from the *exact* Ewald path rather than the PME one the model uses.
17.7% of the force signal lies beyond the 6 Å cutoff — a 0.37 eV/Å floor for
any strictly local model.

| model    | E RMSE (meV/atom) | F RMSE (eV/Å) |
|----------|------------------:|--------------:|
| baseline |            187.31 |        0.4075 |
| joint    |             34.04 |        0.0826 |

The baseline lands at 0.41 eV/Å — essentially exactly the predicted floor.
The joint model reaches 0.083, *five times below* it, which requires
representing interaction beyond the cutoff.

**The capacity control matters more than the headline.** The joint model has
an extra network, so the honest objection is that it wins on parameters. Re-run
on an identical dataset with Coulomb removed: baseline 0.0243 vs joint 0.0257
eV/Å — **no benefit at all** (0.94×). The gain is specifically long-range
physics, not capacity.

**Learned charges recover the truth up to an exact symmetry.** The head, never
shown a charge, returns ±0.9604 against a true ±1: magnitude within 4%,
relative assignment correct. The global sign is a genuine gauge freedom — this
run happened to land on the physical branch, an earlier one on the inverted
branch (∓0.9364), and both are equally optimal. The freedom is exact — the Coulomb energy is bilinear in charge, so `q → −q` leaves
every `q_i q_j` unchanged and, for a neutral cell, the branches are exactly
degenerate (verified to 1.4e-14, pinned by `TestChargeSignDegeneracy`).
Anyone comparing learned charges to a Bader/Mulliken analysis must break the
symmetry themselves. The residual 6.4% is a second, softer degeneracy: the
short-range/long-range split is not unique, since DPA4C can absorb Coulomb
inside its cutoff.

**Real water is suggestive, not demonstrated.** 240 frames of the DeePMD-kit
water example gives 1.29× on energy and 1.07× on force — too small a dataset
to resolve an effect that size on neutral water. What it establishes is that
the long-range term neither breaks nor destabilizes training on real DFT
data.

> **Open question for the target science.** If the systems of interest are
> the electrolyte clusters in `/home/admin/storage/dataset/electrolyte`, they
> are non-periodic and charged, and **PME is the wrong engine for them** —
> Ewald is only defined under PBC. nvalchemi ships `coulomb` and `dsf`
> (damped shifted force) for exactly that case, and both are already
> differentiable with charge gradients. The charge head, the projection, the
> model wiring and the derivative handling are all engine-agnostic; only
> `deepmd_lr/pme/term.py` would need a sibling. Worth settling before Phase 6,
> because it changes what "MD deployment" should target.

### Phase 6 — MD via nvalchemi/ASE
Wire the trained model into the nvalchemi driver, following
`examples/water/dpa4/nvalchemi/relax.py` and `doc/third-party/nvalchemi.md` in the deepmd-kit
tree. Confirm energy conservation in NVE — this is the sharpest test that forces are the exact
gradient of the energy, PME term included.

## Verification

```bash
# unit
conda run -n dpmd pytest tests/ -v

# physics: PME against analytic Madelung; gradcheck in float64
conda run -n dpmd pytest tests/test_pme_physics.py -v

# plugin wiring: is the type registered from a stock deepmd?
conda run -n dpmd python -c "
from deepmd.pt_expt.model.model import BaseModel
print('dp_lr' in BaseModel.get_plugins())"

# end-to-end
conda run -n dpmd dp --pt-expt train examples/water_lr/input.json
conda run -n dpmd dp --pt-expt test -m model.ckpt.pt -s /path/to/validation

# conservation
conda run -n dpmd python examples/water_lr/nve_drift.py   # NVE energy drift over 10 ps
```

Gate for each phase: Phase 1 must match the standard model to within the GPU nondeterminism
floor (see Phase 1 — the original "bit-for-bit" wording was wrong); Phase 4 must reduce to
Phase 1 when charges are forced to zero. Both are cheap regression tests worth keeping.

## Risks

| Risk | Mitigation |
|---|---|
| ~~warp-lang cannot JIT for sm_121~~ | **Resolved.** warp 1.17.0 initialises on `sm_121` and its kernels give correct results. |
| Coulomb prefactor / unit mismatch | Madelung check passed at 6.7e-7 in nvalchemi's native units (prefactor 1). The eV/Angstrom conversion for deepmd is still to be applied and tested in Phase 3. |
| Joint training unstable from random charge init | Fall back to the staged scheme: fit charges first, then unfreeze. Design already supports it. |
| Double-counting: DPA4C absorbs short-range Coulomb | Expected and acceptable — PME real-space is screened within `rcut`. Watch for charge drift toward unphysical magnitudes; regularize charges if so. |
| deepmd-kit 3.2.0 lags master | The plugin hooks are all present in 3.2.0 (verified by import). Pin 3.2.0; revisit only if a needed DPA4C fix is master-only. |
| Silent-wrong-answer APIs in nvalchemi | `spline_order>=7` proved that wrong values can come back with no error raised. Every PME parameter change must be re-validated against the Madelung test, which is why that test belongs in the committed suite. |
