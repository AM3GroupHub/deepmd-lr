# deepmd-lr

Long-range electrostatics for **DPA4C**, computed with NVIDIA's PME
implementation (`nvalchemi-toolkit-ops`) and trained jointly with the
short-range model.

This is a **plugin package, not a fork** of DeePMD-kit. It registers itself
through the `deepmd.pt_expt` entry point that DeePMD-kit already exposes, so
it works against a stock `pip install deepmd-kit==3.2.0`.

Every number in this document was measured on this setup, not taken from
folklore. Where a choice is a judgement call rather than a constraint, it
says so.

## Contents

- [Install](#install) · [Usage](#usage) · [Examples](#examples)
- [Does it apply to your data?](#does-it-apply-to-your-data)
- [Choosing `alpha` and the mesh](#choosing-alpha-and-the-mesh) ·
  [`pme` or `ewald`](#pme-or-ewald)
- [Non-periodic systems](#non-periodic-systems) ·
  [Multi-head training](#several-levels-of-theory-multi-head-training) ·
  [Mixed boundaries (`same_as`)](#one-theory-two-boundaries-same_as)
- [Dataset layout](#dataset-layout) ·
  [Training schedule](#the-training-schedule) ·
  [What to watch](#what-to-watch) · [Validating](#validating-a-trained-model)
- [Troubleshooting](#troubleshooting) · [Cost](#cost)
- [Running MD](#running-md) · [Born charges and external fields](#born-effective-charges-and-md-in-an-external-field) ·
  [MD over partitions](#running-md-over-partitions)
- [Which neighbour list runs](#which-neighbour-list-runs) · [Tests](#tests) · [Results](#results)

## Install

Into an environment with `deepmd-kit==3.2.0` and a matching `torch`:

```bash
pip install -e . --no-deps
```

`--no-deps` is deliberate: the dependency pins are recorded in
`pyproject.toml` for documentation, but the environment is built explicitly
(a CUDA-13 aarch64 `torch` from the PyTorch index, which pip should not try
to re-resolve).

### Building that environment

`deepmd-kit==3.2.0` pins `torch==2.11.0.*`, and an sm_121 GPU needs a CUDA
12.8+/13 build, so torch comes from the cu130 index rather than PyPI:

```bash
# 1. PyTorch 2.11.0 + CUDA 13 -- the version is pinned by deepmd-kit 3.2.0
pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0+cu130"

# 2. deepmd-kit itself (prebuilt wheel)
pip install "deepmd-kit==3.2.0"

# 3. deepmd's torch-path extras, explicitly
pip install "e3nn>=0.5.9" "vesin[torch]" mpich

# 4. the PME engine
pip install "nvalchemi-toolkit-ops>=0.4.1" "warp-lang>=1.13.0"

# 5. data / MD / dev tooling
pip install "dpdata>=0.2.7" "ase>=3.23.0" pytest pytest-cov ruff pre-commit
```

**`mpich` is required, not optional.** The PyPI wheel is cibuildwheel-built,
so `deepmd/pt/cxx_op.py` calls `load_mpi_library()` unconditionally at import
and resolves `libmpi.so` through mpich's package metadata. Without it,
`import deepmd.pt_expt.descriptor` raises `PackageNotFoundError`.

The PyPI `deepmd-kit` aarch64 wheel is a `DP_VARIANT=cpu` build, which is
fine: training runs in eager PyTorch on the GPU, and the compiled CUDA
mega-kernels are opt-in through `DP_CUDA_INFER` and inference-only. No source
build is needed.

Smoke-test before anything else — the riskiest part is warp compiling for
sm_121, and if it fails `warp.init()` says so:

```bash
python -c "
import torch, warp, deepmd
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
from deepmd.pt_expt.descriptor.dpa4c import DescrptDPA4C
print('ok')
"
```

## Usage

Select the model type in the input file:

```json
{
  "model": {
    "type": "dp_lr",
    "type_map": ["O", "H"],
    "descriptor": { "type": "dpa4c", "rcut": 6.0 },
    "fitting_net": { "neuron": [192, 192, 192] }
  }
}
```

Add a `long_range` block to switch the electrostatics on:

```json
{
  "model": {
    "type": "dp_lr",
    "type_map": ["O", "H"],
    "descriptor": { "type": "dpa4c", "rcut": 6.0 },
    "fitting_net": { "neuron": [192, 192, 192] },
    "long_range": {
      "charge_net": { "neuron": [64, 64] },
      "reciprocal": { "alpha": 0.4, "mesh_dimensions": [16, 16, 16] }
    }
  }
}
```

```bash
dp --pt-expt train input.json
```

Without the `long_range` block the model is exactly the built-in standard
model. Only the long-range (reciprocal-space) half of the Ewald sum is
computed here; the short-range half is already covered by the descriptor.

**Always write the baseline twin.** Copy the input file, delete
`"type": "dp_lr"` and the `long_range` block, change nothing else. Without it
you cannot tell whether the long-range term helped or the run merely went
well — this is the single most important methodological point in this
document.

```bash
dp --pt-expt train input_joint.json
dp --pt-expt train input_baseline.json     # same settings, no long_range
```

## Examples

`examples/` is the step-by-step material: data preparation, an annotated
`input.json`, and a README per case. Each was run.

| | case | teaches |
|---|---|---|
| [01](examples/01-bulk) | bulk liquid water, 3D-periodic | the basic loop: xyz → dataset → `input.json` → train |
| [02](examples/02-molecules) | four molecules, no cell | `nopbc`, `charge_spin.npy`, `method: "isolated"`, one system per composition |
| [03](examples/03-slab) | Li/LiFSI interface, 2D-periodic | vacuum padding, `slab_correction` + `pbc` |
| [04](examples/04-multi-head) | all three at once | `shared_dict`/`model_dict`, and the gradient clipping it needs |
| [05](examples/05-pseudo-head) | bulk + slab, one weight set | `same_as`, and setting the long-range type at load |
| [06](examples/06-md_nvalchemi) | MD with nvalchemiops | NVE, NVT, NPT and a field, one runnable loop each |

`scripts/xyz_to_deepmd.py` converts a trajectory ASE can read into a DeePMD
dataset, including the `--split-by-composition` and `--vacuum` cases the
examples use. Composition splitting also sorts atoms and their per-atom labels
into a consistent element order. Choose a new or empty output directory:
the converter refuses to overwrite existing datasets, preventing stale charge
labels or boundary markers from being mixed into a new conversion.

## Does it apply to your data?

Two prerequisites, then one measurement worth doing before spending GPU time.

**Periodic data uses Ewald; non-periodic data uses the direct sum.** Both are
supported — see [Non-periodic systems](#non-periodic-systems) — but they are
different code paths and the choice is not automatic.

**You need energies and forces. You do not need charges.** The charge head is
trained through the electrostatic energy, never against charge labels.

**Measure the headroom.** The long-range term can only recover the part of the
signal that lives beyond the descriptor cutoff. If that is negligible, nothing
downstream will help. Compare exact Ewald forces against Coulomb truncated at
your cutoff: on the benchmark system **17.7% of the force signal lies beyond
6 Å**, which corresponds to a 0.37 eV/Å floor no local model can beat — and
the baseline duly plateaued at 0.41. If your equivalent number is ~1%, expect
a correspondingly small return.

## Choosing `alpha` and the mesh

`alpha` is **not** a free parameter. This term supplies only the reciprocal
half of the Ewald split, so `alpha` decides the division of labour:

```
full Coulomb  =  erfc(alpha·r)/r     +     the reciprocal sum
                 ^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^^^^^^
                 the network's job         this term's job
```

### Constraint 1 — `alpha` large enough that the network's share fits inside its cutoff

`erfc(alpha·r_cut)` — what is still unresolved at the descriptor cutoff:

| alpha | rc = 4 Å | rc = 5 Å | rc = 6 Å | rc = 8 Å |
|------:|---------:|---------:|---------:|---------:|
| 0.2   |  2.6e-01 |  1.6e-01 |  9.0e-02 |  2.4e-02 |
| 0.3   |  9.0e-02 |  3.4e-02 |  1.1e-02 |  6.9e-04 |
| 0.4   |  2.4e-02 |  4.7e-03 | **6.9e-04** | 6.0e-06 |
| 0.5   |  4.7e-03 | **4.1e-04** | 2.2e-05 | 1.5e-08 |
| 0.7   | **7.5e-05** | 7.4e-07 | 2.9e-09 | 2.4e-15 |

Aim for ≲ 1e-3. Too small an `alpha` hands the network a tail it structurally
cannot represent — the exact failure the long-range term exists to fix.

### Constraint 2 — mesh fine enough to resolve the screening Gaussian

Relative error in the reciprocal energy vs mesh spacing `h`:

| alpha | 1/(2·alpha) | h = 1.5 | h = 1.25 | h = 1.0 | h = 0.75 |
|------:|------------:|--------:|---------:|--------:|---------:|
| 0.3   |        1.67 | 1.9e-06 |  3.4e-06 | 5.6e-07 |  1.9e-08 |
| 0.4   |        1.25 | 5.3e-05 |  2.8e-05 | 3.3e-06 |  5.8e-08 |
| 0.5   |        1.00 | 1.1e-04 |  1.6e-04 | 1.8e-05 |  8.3e-07 |
| 0.7   |        0.71 | 5.9e-03 |  5.5e-04 | 2.9e-04 |  2.2e-05 |

**`h ≈ 1/(2·alpha)`** lands around 1e-4; go one step finer for ~1e-6.

### Recommended pairings

| descriptor `rcut` | `alpha` | mesh spacing | for a 20 Å cell |
|------------------:|--------:|-------------:|----------------:|
| 6 Å               |     0.4 |     ≤ 1.25 Å | `[16, 16, 16]`  |
| 5 Å               |     0.5 |     ≤ 1.00 Å | `[20, 20, 20]`  |
| 4 Å               |     0.7 |     ≤ 0.75 Å | `[28, 28, 28]`  |

Larger `alpha` costs mesh but leaves the network less to do. **Whatever you
pick must match between training and any later MD** — `alpha` defines the
model, not just the numerics.

**Prefer `mesh_dimensions` over `mesh_spacing`.** A spacing-derived mesh is
`ceil(L / spacing)`, a step function of the cell: a cell length crossing a
multiple of the spacing gains a whole grid point and the energy jumps. It is
harmless at fixed cell (NVE/NVT) and a real problem under NPT or any strain
derivative — it showed up here as a 2% virial error that looked like a bug in
the strain derivative.

## `pme` or `ewald`

Both compute the *same* reciprocal sum, by different routes.

| | `pme` (default) | `ewald` |
|---|---|---|
| how | charges onto a mesh, then an FFT | explicit sum over k-vectors |
| scaling | `O(N log N)` | `O(N · K)`, and `K` grows with cell volume |
| error | mesh interpolation, ~1e-6 at a sane mesh | none — exact for its `k_cutoff` |
| knobs | `mesh_dimensions` / `mesh_spacing`, `spline_order` | `k_cutoff` (or `accuracy`) |

Measured (energy + backward, alpha 0.4, 1.25 Å mesh):

| atoms | cell | `pme` | `ewald` | k-vectors |
|------:|-----:|------:|--------:|----------:|
|    32 | 14 Å | 2.9 ms |  3.3 ms |     1,687 |
|   128 | 22 Å | 2.8 ms |  5.0 ms |     6,083 |
|   512 | 35 Å | 2.7 ms | 25.5 ms |    21,437 |
|  2048 | 56 Å | 2.9 ms | 113.6 ms |   83,187 |

**Use `pme` for training.** It is faster at every size tested — including the
smallest, so there is no small-system crossover to exploit — and the gap grows
to 40× by 2048 atoms.

**Use `ewald` to check a `pme` setting.** Because it has no interpolation
error, disagreement between the two is a statement about your mesh, not about
the physics:

```python
from deepmd_lr.lr import LongRangeEnergy
pme   = LongRangeEnergy(alpha=0.4, mesh_dimensions=[16, 16, 16])
ewald = LongRangeEnergy(alpha=0.4, method="ewald")
a, b = pme(coord, q, box).sum(), ewald(coord, q, box).sum()
print(abs(a - b) / abs(b))       # want <= 1e-4; if not, refine the mesh
```

Parameters belonging to the other method are **rejected**, not ignored — a
mesh setting silently dropped under `ewald` would be indistinguishable from
one that was applied.

## Non-periodic systems

Worked example: [`examples/02-molecules`](examples/02-molecules).

Mark a system with an empty `nopbc` file in its directory and DeePMD-kit's
loader hands the model `box = None`. A short-range head already works that way
— a `dp_lr` branch with no `long_range` block, or a plain `standard` model,
runs on `box=None` frames with no special handling.

To add long-range electrostatics to such a branch, set the method to
`isolated`:

```jsonc
"long_range": {
  "charge_net": { "neuron": [64, 64], "precision": "float32" },
  "reciprocal": { "method": "isolated", "alpha": 0.4 }
}
```

`alpha` must match whatever the periodic branches use. Nothing else applies —
meshes, k-vector cutoffs, `slab_correction` and `pbc` all describe a sum over
periodic images, and passing one is an error rather than a silent no-op.

**What that term computes is fixed, not a choice.** The periodic path splits
Coulomb at the screening parameter α:

```
Σ qᵢqⱼ/r  =  Σ qᵢqⱼ erfc(αr)/r     ← short-ranged; the network learns this
           + Σ qᵢqⱼ  erf(αr)/r     ← smooth and long-ranged
```

`LongRangeEnergy` returns the second sum, evaluated in reciprocal space. So for
an isolated system the consistent term is that same sum evaluated directly, at
the **same α**:

```
Eᵢ = ½ Σ_{j≠i} qᵢqⱼ erf(α rᵢⱼ) / rᵢⱼ
```

This is not an analogy. Put a cluster in a cube of side L and the periodic
`recip − self` converges onto the direct `erf` sum as 1/L³ — measured at
−6.6e-03, −8.3e-04 and −1.0e-04 eV for L = 80, 160 and 320 Å. It is the same
physical quantity, which is what lets one shared backbone serve both branches:
the remainder left to the network has the same `erfc(αr)/r` form either way.

Three consequences worth knowing:

- **No self term to subtract.** `erf(αr)/r → 2α/√π` as `r → 0`, and the Ewald
  self term is exactly that times `½Σqᵢ²`. Summing `j ≠ i` already excludes it.
- **Charged molecules are easier here than in a cell** — no neutralising
  background is needed, so an ion is just an ion.
- **No cutoff.** `erf(αr)/r` decays as `1/r`; truncating it would be wrong.

It is summed densely rather than through `nvalchemiops`. Both of that
library's candidates are unusable here: `dsf_coulomb` is a shifted-force
potential with a charge-dependent self term and no position autograd, and
`coulomb_energy` raises under `create_graph=True` — so neither survives
force-loss training, which differentiates the force a second time. The dense
form is exact, twice differentiable, and costs about 1 ms at 1000 atoms. Its
one limit is memory: the pair tensor is `O(N²)`, or 8 MiB at 1000 atoms in
float64. Fine for molecules, and the reason this is not the method for a
periodic cell even in principle.

Two other things behave differently without a cell, both deliberately:

- **The virial is still computed**, and is still meaningful — it is the pair
  sum `Σ rᵢⱼ ⊗ fᵢⱼ`, which needs no periodicity. DeePMD-kit's own energy model
  returns one for a `box=None` frame, so this term matches it rather than
  leaving the total inconsistent with the short-range part.
- **Stress is not reported.** Force per unit area needs a volume to divide by,
  and there is none. `LongRangeCalculator` simply omits it.

## Several levels of theory: multi-head training

Worked example: [`examples/04-multi-head`](examples/04-multi-head).

If your datasets differ not just in composition but in **what produced the
labels** — one at PBE, one at a hybrid functional, one at coupled cluster — a
single energy head cannot fit them. The references and scales differ, so
whatever the head learns is a compromise between incompatible targets. The
same applies when the datasets differ in *boundary condition*: a slab, a bulk
cell and an isolated molecule each need their own long-range settings.

Both cases are the same fix: **one shared descriptor, several fitting heads**.
DeePMD-kit calls this multi-task, and `dp_lr` works inside it unchanged.
Multi-task is triggered by the presence of `model_dict` — there is no CLI flag.

### What is shared and what is not

Only three keys can be shared: `type_map`, `descriptor`, `fitting_net`. Write
the value once in `shared_dict` and reference it by name from each branch.

Sharing works by **aliasing the module objects**, not by copying and
resynchronising: after setup the branches hold literally the same parameter
tensors, so one optimizer step updates the backbone for all of them.

`long_range.charge_net` is **not** shareable, so every branch gets its own
charge head. That is the right default when the branches are different levels
of theory — the charges genuinely differ — but it does mean the branches are
only tied through the backbone. When the branches are the *same* theory and
differ only in boundary condition, see [`same_as`](#one-theory-two-boundaries-same_as)
below, which ties everything instead.

DPA4C supports `shared_level: 0` only (share everything). It also checks that
the structural parameters match *exactly* — `rcut`, `channels`, `lmax`,
`n_radial`, `radial_modes`, `basis_type`, `type_map`, precision and
`add_chg_spin_ebd`. That is why the descriptor is written once in
`shared_dict` rather than copy-pasted per branch: a single typo becomes
`ValueError: DPA4C parameter sharing requires identical structural parameters`.

### Set `gradient_max_norm`

**Not optional with heterogeneous branches.** Branches push the shared
backbone with wildly different gradient norms. Measured at initialization on
the three example datasets:

| branch | descriptor `stddev` min | amplification | grad norm on shared backbone |
|---|---:|---:|---:|
| bulk water | 3.6e-05 | ×28,000 | **5.0e+09** |
| molecules | 7.8e-03 | ×128 | 1.9e+06 |
| slab | 1.9e-03 | ×538 | 4.8e+05 |

DPA4C's `compute_input_stats` divides each descriptor coordinate by its spread
on that branch's own data — a fixed preconditioner set once at initialization.
Bulk liquid water is dense and isotropic, so ~20 of its 144 coordinates have a
spread below 1e-3, and dividing by 3.6e-05 amplifies those directions ×28,000
on the way back out. Standalone that is harmless, because the branch's head is
matched to its own preconditioner. **Shared it is not**: the calibration is
per-branch, the weights are not, and the bulk branch simply overwrites what the
others learn. Measured at 600 steps:

| `gradient_max_norm` | bulk `rmse_f` | molecules | slab |
|---|---:|---:|---:|
| *(unset)* | 0.403 | 1.43 | **15.0** |
| 5.0 | 0.221 | 1.24 | **1.00** |
| 1.0 | 0.216 | 1.06 | 1.43 |

Unclipped, the slab sits 150× above its own labels' force scale and stays
there. Two ways to recognise the problem: a branch stuck orders of magnitude
above the others while training fine on its own, and results that depend on
the order branches are listed in.

### Rules that will bite you

- **One `type_map` for all heads.** DeePMD-kit asserts it. The map is the union
  of elements across every dataset; per-system `type.raw` indices stay local
  and are remapped through each system's own `type_map.raw`.
- **`pt_expt` does not inherit top-level defaults into branches.** Unlike the
  `pt` backend, nothing cascades — combined with strict validation, every
  option must be written out in every branch.
- **`alpha` must be the same in every branch.** It sets the split between what
  the network learns and what the term adds; one backbone cannot serve two
  different splits. `mesh_dimensions` *should* differ per branch, because the
  cells do.
- **`data_stat_nbatch` is a model key, per branch** — not a `training` key.
  Putting it in `training` fails normalization outright. The default of 10
  builds a dense neighbour matrix over 10 frames at once, which is tens of GiB
  at a few hundred atoms.
- **Set `start_pref_v: 0` for any branch without a cell.** There is no virial
  to fit.
- **Charged systems need `add_chg_spin_ebd: true`** on the descriptor, plus a
  `charge_spin.npy` of shape `(nframes, 2)` holding `[charge, multiplicity]`.
  Because that flag is part of the structural signature, turning it on for one
  branch means turning it on for **all** of them — decide before the first long
  run, since retrofitting it invalidates existing checkpoints.
- **Then set `default_chg_spin: [0, 1]` too.** Once the descriptor is
  charge-conditioned it demands a `charge_spin` for *every* frame, so the
  neutral branches would otherwise fail with `A charge-conditioned DPA4C
  requires a frame charge_spin`. The default supplies the neutral singlet
  fallback. Unlike `add_chg_spin_ebd`, this key is deliberately excluded from
  the structural signature, so it does not interfere with sharing.

### Running and reading it

One head is chosen per optimizer step by a weighted random draw over
`model_prob` (defaulting to weights proportional to each task's system count),
and a single optimizer steps all parameters. Every head is still evaluated at
each `disp_freq`, so `lcurve.out` gains per-head columns —
`rmse_e_trn_interface`, `rmse_f_val_bulk`, and so on.

```bash
dp --pt-expt train input.json
dp --pt-expt show  model.ckpt.pt model-branch     # list the heads
```

Long-range branches cannot be frozen; for MD, select the branch from the
training checkpoint through the eager ASE calculator:

```python
from deepmd_lr.ase import LongRangeCalculator
atoms.calc = LongRangeCalculator("model.ckpt.pt", head="interface")
```

`head` is required for multi-head checkpoints; branch aliases are accepted,
and the ASE periodic axes must match that branch's boundary configuration.
Each head carries its own energy zero, so do not mix heads within one
simulation.

Not available in multi-task mode: `zero_stage`, `validating.full_validation`,
`lora`, and `change-bias`.

**Is it working?** Compare each branch's validation RMSE against a single-task
model trained on that dataset alone. Sharing should be roughly
neutral-to-better per branch; if one branch is clearly worse, reweight with
`model_prob` or train it separately — `shared_level` cannot be relaxed for
DPA4C.

## One theory, two boundaries: `same_as`

Worked example: [`examples/05-pseudo-head`](examples/05-pseudo-head).

Multi-head above answers "different levels of theory". A different question:
your dataset is **one** level of theory, but contains both periodic bulk cells
and 2D-periodic slabs. Separate heads would be wrong — they would each fit
their own energy zero and each see half the data.

They do not have to be separate, because **the long-range term holds no
trainable weights**. Every parameter lives in the descriptor, the energy head
and the charge head; the reciprocal sum has none. So one set of weights serves
both boundary conditions, and only the sum over periodic images differs.

`same_as` declares a branch to be another branch with a different long-range
term:

```json
"model_dict": {
  "bulk": { "type": "dp_lr", "type_map": "type_map_all",
            "descriptor": {...}, "fitting_net": {...},
            "long_range": { "charge_net": {...},
                            "reciprocal": {"alpha": 0.4,
                                           "slab_correction": false,
                                           "pbc": [true, true, true]} } },

  "slab": { "same_as": "bulk",
            "long_range": { "reciprocal": {"alpha": 0.4,
                                           "slab_correction": true,
                                           "pbc": [true, true, false]} } }
}
```

Unlike `shared_dict`, which ties a named list of components, this ties
*everything* parameter-bearing: descriptor, energy head, charge head, and the
per-type energy offsets — the same Python objects, so one gradient and one
update per step whichever branch it was drawn from. Those offsets are fitted
once over both datasets **pooled**, which is what stops any gap between them
being absorbed as a spurious surface energy.

The two mechanisms compose: `shared_dict` for genuinely different heads,
`same_as` for heads that are the same model. A multi-task config still needs
exactly one shared `type_map` reference, so the `shared_dict` stays even here.

**Do you need it?** The correction a slab branch adds is Yeh–Berkowitz,
`2π M_z²/V`, which is *identically zero* for a non-polar slab. If your slabs
carry no dipole across the vacuum, one ordinary 3D branch already serves both.
It earns its keep for polar slabs — an electrode under electrolyte, an
asymmetric interface — where the correction is large and decays only as `1/L_z`.

A `same_as` branch may set only `long_range`, `model_branch_alias` and `info`.
Anything else is refused rather than ignored, because it would be built and
then discarded when the modules are aliased — training would converge to a
model nobody asked for. So are a differing `alpha` (the shared descriptor
learned one split), an overridden `charge_net` (it holds weights, so it must be
the same object), `dim_case_embd > 0` (it gives each branch an index in one
shared buffer), and a `same_as` chain.

> Remove `stat_file` when you add or change `same_as`: the offsets are restored
> from it when present, bypassing the pooled fit.

### Setting the long-range type at load

The same fact — no weights in the term — makes the boundary treatment a
property of the *simulation* rather than of the checkpoint:

```python
from deepmd_lr.ase import LongRangeCalculator, load_model

model = load_model("model.ckpt.pt", head="slab")
model.set_long_range({"alpha": 0.4, "mesh_dimensions": [10, 10, 10],
                      "slab_correction": False, "pbc": [True, True, True]})
```

`long_range=` is also a keyword on `load_model`, `LongRangeCalculator` and
`build_evaluator`, so a swap needs no extra line. It **replaces** the reciprocal
block rather than merging into it — merging cannot work, since
`method: "isolated"` refuses every periodic parameter. `alpha` is inherited from
the checkpoint and a different one raises: it is where Coulomb is split, and the
network learned the remainder of *that* split.

## Dataset layout

`scripts/xyz_to_deepmd.py` handles the common cases; this section is the
reference for what it produces and why.

**One system = one composition and one fixed atom ordering.** `type.raw` and
`type_map.raw` live at the *system root*, not inside `set.000`, so every
`set.*` in a system shares them — the sets are frame batches, not different
compositions:

```
data/water/type.raw          <- here, shared by all sets
data/water/type_map.raw      <- here
data/water/set.000/coord.npy
data/water/set.001/coord.npy
```

### The indices in `type.raw` are LOCAL to that system

`type.raw` indexes that system's *own* `type_map.raw`, and DeePMD-kit remaps
into the model's global `type_map` at load time
(`deepmd/utils/data.py`: `old_to_new_type_idx[self.atom_type]`). So each
system's `type_map.raw` may be shorter, and in a different order, than the
model's — it only has to contain the elements that system actually uses.

With model `type_map = ["Cl", "H", "Na", "O"]`:

| system | `type_map.raw` | `type.raw` | remapped | elements |
|---|---|---|---|---|
| water | `["H", "O"]` | `[1, 0, 0]` | `[3, 1, 1]` | O H H |
| nacl | `["Cl", "Na"]` | `[0, 1, 0, 1]` | `[0, 2, 0, 2]` | Cl Na Cl Na |

- **Always write `type_map.raw`.** Without it there is nothing to remap
  against, so the indices are taken as *global* — and this fails **silently**.
  The water system above, minus its `type_map.raw`, loads as H-Cl-Cl instead
  of O-H-H with no error at all.
- An element in a system's `type_map.raw` that the model's `type_map` lacks
  fails loudly: `ValueError: Elements ['Xe'] are not present in the provided
  type_map`.

### Several compositions in one training set

Different compositions go in different systems. `dpdata` will group them:

```python
import dpdata
ms = dpdata.MultiSystems()
for frame in whatever_you_have:      # any dpdata LabeledSystem
    ms.append(frame)
ms.to_deepmd_npy("data")             # or ms.to_deepmd_hdf5("data.hdf5")
```

**Point `systems` at the parent, not at each child.** `expand_sys_str`
recurses and takes every directory containing a `type.raw`
(`deepmd/common.py:204`), so nesting depth is free:

```json
"training_data": { "systems": ["./data"], "batch_size": "auto:8" }
```

Two knobs matter once the systems differ in size:

- **`"batch_size": "auto:N"`** targets ~N atoms per batch rather than N
  frames. A fixed `batch_size` makes the large systems dominate each step.
- **`auto_prob`** (default `prob_sys_size`) sets how often each system is
  drawn, proportional to its batch count. Use `"auto_prob": "prob_uniform"`
  to weight every system equally instead, or `sys_probs` to set them by hand.

### Composition that varies frame to frame

If the *atom count* is fixed but which element sits at each site changes per
frame, that is one system in **`mixed_type`** format: add
`real_atom_types.npy` of shape `(nframes, natoms)` to each set, holding local
indices, with `-1` for a virtual (padding) atom. Its presence is what selects
the mode, it must be present in *every* set of the system, and
`type_map.raw` becomes mandatory. `type.raw` is still required — it fixes
`natoms`.

### Directories or HDF5

DeePMD-kit reads either. The mapping is one-for-one — **a directory becomes an
HDF5 Group, a file becomes a Dataset** (`deepmd/utils/path.py`, `DPH5Path`) —
so the tree inside the file is identical to the directory layout. A system is
addressed with `#` rather than a path separator, and nothing else changes:

```json
"training_data": { "systems": ["./data.hdf5#/train"], "batch_size": 1 }
```

`type_map.raw` is the one thing to get wrong: it is read back with
`load_txt(dtype=str)`. Fixed-length bytes (`S8`) and both variable-length
string dtypes decode correctly through that; pickled Python objects do not.

**Compression is not worth enabling.** Measured on 75.1 MB of float64
coordinates, boxes, energies and forces:

| filter | size | ratio | write | read all |
|---|---:|---:|---:|---:|
| none | 75.2 MB | 1.00× | 0.04 s | 0.01 s |
| lzf | 74.8 MB | 1.00× | 0.47 s | 0.09 s |
| gzip | 70.0 MB | 1.07× | 1.65 s | 0.24 s |

MD coordinates are high-entropy, so gzip costs a 24× slower read to save 7%.
If a set is genuinely too large, storing coordinates as float32 halves it — a
real decision about precision, not a free win like compression appeared to be.

Choose HDF5 for the **file count**, not the size: one file instead of a dozen
per system matters on a cluster filesystem with inode quotas, and makes a
dataset atomic to `rsync`. Throughput is comparable — measured 0.79 s/batch
from HDF5 against 0.72 from directories, on 6520-atom frames.

## The training schedule

### Plan A — joint from scratch (default)

Train everything at once, LR 1e-3 → 1e-5 with exponential decay. This is what
the benchmark used and it converged without trouble: 4.9× better forces than
the baseline, charges settling near ±1 within a few thousand steps. Use it
unless you hit one of the symptoms in [Troubleshooting](#troubleshooting).

### Plan B — staged (fallback when Plan A is unstable)

Warm up the short-range model first, then switch the long-range term on.

| stage | steps | LR | what trains |
|-------|------:|----|-------------|
| 1. warm-up | 10–20% of budget | 1e-3 → 1e-4 | descriptor + energy head (no `long_range` block) |
| 2. graft   | — | — | copy weights across, scale down the charge head |
| 3. joint   | remainder | 5e-4 → 1e-5 | everything |

Stage 2 needs a helper, because `dp train --init-model` loads the state dict
**strictly** and refuses a checkpoint missing the charge head's weights:

```bash
dp --pt-expt train input_stage1.json          # long_range block removed

python scripts/graft_short_range.py \
    --short-range run_stage1/model.ckpt.pt \
    --input       input_joint.json \
    --out         init_joint.ckpt.pt

dp --pt-expt train input_joint.json --init-model init_joint.ckpt
```

The script scales the charge head's output layer by 0.01 by default, and the
scale has to be *small but not zero*. Both failure modes are real:

- **Too large** (a freshly initialized head) predicts random charges and
  injects a large spurious long-range energy, undoing the warm start. Measured
  on the water set: the warm-up ended at 0.176 eV/Å and grafting an unscaled
  head restarted training at **0.789**.
- **Exactly zero** looks safe and is worse, because it fails silently. The
  electrostatic energy is quadratic in charge, so at `q = 0` the gradient
  `dE/dq` is exactly zero — and with it every gradient reaching the charge
  head. Measured: a zeroed head gives `sum|grad| = 0.000e+00`. The head is
  frozen at zero permanently, the long-range term never turns on, and training
  looks perfectly healthy because the short-range model simply carries on
  alone. The script rejects zero.

At the default 0.01 the graft is continuous and the head is alive:

| | long-range energy | max abs charge | training force RMSE |
|---|---:|---:|---:|
| warm-up ended | — | — | 0.176 eV/Å |
| grafted init  | 0.0002 eV | 4.1e-03 | 0.168 eV/Å |
| after stage 3 | 0.0492 eV | 8.7e-02 | 0.175 eV/Å |

## What to watch

The model returns three diagnostics beyond the usual outputs:

```python
out = model(coord, atype, box=box)
out["energy_short_range"]   # DPA4C
out["energy_long_range"]    # the reciprocal term, self term subtracted
out["charge"]               # per-atom, after the total-charge projection
```

| signal | healthy | trouble |
|--------|---------|---------|
| `energy_long_range / energy_short_range` | a few % to tens of % | ~0 (term is doing nothing) or dominant (it is absorbing short-range physics) |
| charge magnitudes | stable, physically plausible | growing without bound, or drifting run to run |
| charge spread across frames | small for chemically equivalent atoms | large — the head is fitting noise |
| val force RMSE vs the baseline | clearly below | at or above — the term is not paying for itself |

> **Convention, and a breaking change.** `energy_long_range` is
> `recip - self`: the reciprocal sum with the Ewald self term
> `(alpha/sqrt(pi)) * sum q_i^2` **subtracted**, which is the standard Ewald
> decomposition. Its complement is then exactly the screened real-space sum
> `sum_{i<j} q_i q_j erfc(alpha r_ij) / r_ij`, and nothing else, for the
> network to represent.
>
> Before 2026-09-07 the term added the self piece back, to match DeePMD-kit's
> DPLR `ewald_reciprocal_energy`, which reports the bare k-space sum. That was
> wrong for two reasons. It put a purely on-site quantity inside a number
> called "long-range energy" — on the LiFSI frame, 103.0 eV of a reported
> 208.8 eV. And `+(alpha/sqrt(pi)) sum q_i^2` is positive-definite and grows
> with charge magnitude, so it biases the optimiser toward small charges,
> which is the direction the LiFSI charge head actually collapsed in.
>
> **A checkpoint trained before that change is not comparable with one after
> it.** Retrain, or pin the old commit.

## Validating a trained model

**Against the baseline.** The only claim that matters. Compare held-out
energy and force RMSE against the identically configured run without
`long_range`.

**The charges.** Expect the right magnitudes and relative signs. Do **not**
expect the right overall sign: the energy is bilinear in charge, so `q → −q`
leaves every `q_i q_j` unchanged and, for a neutral cell, the two branches are
*exactly* degenerate (verified to 1.4e-14). Two runs of the same benchmark
converged to `+0.96/−0.96` and `−0.94/+0.94`; both are equally optimal. If you
need physically signed charges — to compare against Bader or Mulliken — you
must break the symmetry yourself with a charge label or a sign constraint.

**Energy conservation.** The sharpest end-to-end test that forces really are
the gradient of the energy, long-range term included: run NVE and watch the
drift.

## Troubleshooting

| symptom | likely cause | fix |
|---------|--------------|-----|
| No improvement over baseline | little signal beyond the cutoff | check the [headroom](#does-it-apply-to-your-data); if it is ~1%, the term cannot help |
| No improvement, but headroom is large | `alpha` too small, so the network is still being handed a long tail | raise `alpha` |
| Loss spikes when long-range switches on | charge head initialized random | use `scripts/graft_short_range.py` |
| Charges grow without bound | short-range and long-range trading off against each other | raise the force weight, or add a charge regularizer |
| Virial wrong by a few % | mesh jumping with the cell | use `mesh_dimensions`, not `mesh_spacing` |
| One multi-head branch stuck orders of magnitude high | branches pushing the shared backbone with very different gradient norms | set `gradient_max_norm` |
| `TypeError: unexpected long-range parameters: ['rcut']` | stale config | `rcut` was removed; only the reciprocal half is computed, so there is no real-space cutoff |
| `ValueError: ... only apply to method='pme'` | mesh knob set with `method: ewald` | use `k_cutoff`, or drop `method` to stay on PME |
| `pme` and `ewald` disagree by more than ~1e-4 | mesh too coarse for this `alpha` | refine the mesh, or lower `alpha` |
| Energies fine, forces bad | `alpha`/mesh mismatch | check the mesh resolves `1/(2·alpha)` |
| CUDA OOM during statistics | `data_stat_nbatch` default of 10 on large frames | set `data_stat_nbatch: 1` in the model block |

## Cost

The charge head reads the same descriptor features the energy head produced,
so the backbone runs **once** — the serial path via `_capturing_backbone`, the
partitioned path via the equivalent capture in `nvalchemi/driver.py`. Both
keep the old second pass as a fallback for when the capture cannot carry the
gradient.

Measured against an identically configured short-range twin (same descriptor,
same fitting net, `long_range` deleted):

| | short-range | long-range | ratio |
|---|---:|---:|---:|
| training, 192-atom water | 0.0431 s/batch | 0.0967 s/batch | **2.24×** |
| MD, 6,000 atoms | 34.6 ms/step | 55.5 ms/step | **1.60×** |
| MD, 65,856 atoms | 256.7 ms/step | 403.8 ms/step | **1.57×** |

Training pays more than MD because the force loss differentiates a second time,
and that double backward goes through the charge head too.

**The reciprocal sum is not what costs.** Timed on its own it is 1.5 ms — 2.3%
of a 6,000-atom step and **0.3%** of a 65,856-atom one — and it barely grows
with the system, because the FFT cost follows the mesh rather than the atom
count. Refining the mesh is therefore close to free; see
[Choosing `alpha` and the mesh](#choosing-alpha-and-the-mesh). What remains of
the long-range overhead is the charge head and the charge-response backward,
which is work rather than duplication.

An external field adds a further backward for `Z`: **+25–30%** on the step, and
no extra memory, since it reuses the captured features. Memory for the
long-range model is ~0.65 MiB/atom (42.6 GiB at 65,856 atoms), with or without
a field.

## Running MD

> **Do not `dp freeze` a model with a `long_range` block.** It is refused. The
> exported artifact is traced from the graph lower, which receives edge vectors
> and no cell, so it cannot evaluate the reciprocal-space term. Before this was
> guarded, `dp freeze` succeeded silently and the frozen model returned
> **−93.74 eV** where the eager model gives **−55.12** — the entire 38.6 eV
> long-range contribution gone, with no warning. Freezing a short-range-only
> model (no `long_range` block) still works.

Use the ASE calculator, which drives the model's full `forward` and therefore
has the cell:

```python
from ase.md.verlet import VelocityVerlet
from ase import units
from deepmd_lr.ase import LongRangeCalculator

atoms.calc = LongRangeCalculator("run_joint/model.ckpt.pt")
VelocityVerlet(atoms, timestep=0.5 * units.fs).run(10000)
```

`atoms.calc.results` carries `energy_short_range`, `energy_long_range` and
`charges` alongside energy/forces/stress, so the split can be watched along a
trajectory without re-running the model. `experiments/run_md.py` is a minimal
driver.

This is eager PyTorch, not a compiled artifact, so it is slower than a frozen
short-range model. That is the price of computing physics the exported path
cannot express.

### Check energy conservation before trusting a trajectory

Measured on the water model (192 atoms, 340 K):

| timestep | drift |
|---------:|------:|
| 1.00 fs  | +10.7 meV/atom/ps |
| 0.50 fs  | −0.40 meV/atom/ps |
| 0.25 fs  | −0.005 meV/atom/ps |

What matters is that drift **vanishes as the timestep shrinks**. A force that
is not the exact gradient of the reported energy leaves a floor no timestep
reduction removes; here the drift falls by ~2000× from 1.0 to 0.25 fs. (The
1 fs figure is the usual story for unconstrained water — O–H stretches need a
short step — and is nothing to do with the long-range term.)

## Born effective charges and MD in an external field

A trained `dp_lr` model predicts geometry-dependent charges, and the derivative
of those charges is what a Born effective charge tensor is made of. No extra
training, no extra labels:

```python
calc = LongRangeCalculator(model, efield=[0.0, 0.0, 0.05])   # V/angstrom
atoms.calc = calc
atoms.get_forces()          # includes F_i = Z_i.T E (polarization axis contracted)
calc.results["born_charge"] # (natoms, 3, 3)
```

The tensor computed is

```
Z_iab = q_i d_ab + 1/2 sum_{j!=i} ( r_ji,a dq_i/dr_ji,b  -  r_ij,a dq_j/dr_ij,b )
```

following 10.1021/acs.jctc.6c00146.

### Read this before interpreting a field run

The first axis of `Z_iab` is polarization and the second is displacement,
so the field contraction is `F_ib = sum_a E_a Z_iab`, or `F = Z.T·E`.

**`F = Z.T·E` is not the gradient of `−M·E`.** That is not a bug and not an
approximation that gets better with a smaller timestep.

For a charge model with a total-charge constraint, the exact dipole derivative
of an isolated system is

```
dM_a/dr_ib = q_i d_ab  -  sum_{e: src=i} (r_e (x) g_e)_ab  +  (centroid - r_i)_a S_ib
```

verified against finite differences to **5e-11**. The final term is
proportional to the displacement from the centroid: unbounded, and meaningless
in a periodic cell where `M` itself is undefined because position is not
periodic. The formula above keeps the local, translation-invariant content and
discards that term — which is exactly why it works under PBC at all. On a
compact molecule the two differ by ~50%, and that gap *is* the discarded term.

- **A constant-field NVE run will not conserve `E − M·E`.** Do not read the
  drift as an integrator problem; there is no energy whose gradient this force
  is. Use a thermostat and look at structure and response, not at a conserved
  quantity.
- **No stress is reported under a field.** The field's contribution to the
  stress is a separate derivation, so `LongRangeCalculator` withholds the
  stress entirely rather than returning the field-free value, which would be
  quietly wrong.
- **Net force is exactly `Q·E`** — zero for a neutral cell, so nothing drifts.
  That follows from `sum_i Z_i = Q·I`, which holds identically.

### Implementation notes

**It costs one extra backward pass, not one per atom.** DPA4C has no message
passing, so an atom's charge depends only on edges where it is the centre. A
single VJP therefore yields every per-edge derivative at once. A naive Born
implementation is `O(natoms)` backward passes; this is `O(1)`.

**The derivative uses the raw charges, the `q_i d_ab` term uses the constrained
ones.** `project_to_total_charge` subtracts a global mean, so every charge
depends on every position — moving an atom 32.9 Å away (rcut 6.0) shifts every
distant charge by a uniform amount comparable to a genuine neighbour
derivative. Weighted by an unbounded `r_ji`, those terms would dominate and
grow with system size. Splitting the two roles keeps the sum rule exact while
keeping the tensor local.

**Caveats.** It needs the shared-backbone path: under `DP_CUDA_INFER >= 2` a
fused kernel bypasses it and the model **raises** rather than returning a wrong
tensor, so unset that variable for field simulations. And `Z` is only as
meaningful as the charge head — if `mean|q|` has collapsed toward zero during
training, the Born tensor and the whole field response are meaningless.

It is available on the partitioned evaluator too, either as the tensor alone
or as a field to run under:

```python
ev = DistributedEvaluator(model, atype, cell, devices=[...], compute_born=True)
ev(positions)
ev.results["born_charge"]          # (natoms, 3, 3)

# or: run under a uniform field, in V/angstrom
ev = DistributedEvaluator(model, atype, cell, devices=[...], efield=[0, 0, 0.05])
energy, forces = ev(positions)     # forces include F_ib = E_a Z_iab
ev.results["field_force"]          # the field term alone, (natoms, 3)
```

`efield` implies `compute_born`, and the returned `forces` carry the response
term while `results["force"]` stays the field-free force. Two deliberate
refusals: a model with no `long_range` block has no charges to build `Z` from,
and `compute_virial` cannot be combined with a field, because the field's
contribution to the stress is a separate derivation and reporting the
field-free virial under a field would be quietly wrong.

Verified against the serial path at 1, 2 and 4 partitions, and the net field
force comes out as `Q·E` to 1e-15 — zero for a neutral cell, so nothing
drifts.

The global projection is no obstacle there. It only enters the `q_i delta`
term, which is already projected by the time the partitions are assembled,
while the derivative half uses the raw charges and is strictly local. What the
partitioned assembly does need is a scatter-add: an atom's tensor sums over
edges where it is the *source* as well as the destination, and its source edges
point at neighbours that may live in another partition. Those contributions are
returned to their owners exactly as ghost forces are. Verified to agree with
the serial tensor to 5e-16 at 1, 2 and 4 partitions.

## Running MD over partitions

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

### Limits

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
  pays — see [Which neighbour list runs](#which-neighbour-list-runs) for the
  thresholds. Only the *search* is taken from nvalchemiops: its edge vectors
  come out of a Warp kernel with no gradient, so they are discarded and
  DeePMD-kit's `neighbor_graph_from_ijs` recomputes `edge_vec` from
  `coord`/`box`. Using the kernel's vectors directly would produce a model
  with no forces. Edge sets are identical to the oracle — same count, same
  vectors to 9 decimals, periodic and not. End-to-end this is 1.00–1.77× over
  768–12288 atoms.

## Which neighbour list runs

Two neighbour graphs are built per step, chosen independently.

**The descriptor's**, which DeePMD-kit owns, is selected by
`neighbor_graph_method`. It resolves once at *training* startup — default
`auto`, which is `nv` (the nvalchemiops GPU cell list) on CUDA — but it is
**not recorded in the checkpoint**, so a model rebuilt for inference falls
back to the `dense` O(N²) builder. `deepmd_lr.ase.load_model` re-resolves it,
so anything loaded through `load_model` or `LongRangeCalculator` gets `nv`
again; a model you rebuild by hand does not. That gap is the 48.2 GB / 724 ms
versus 4.0 GB / 201 ms difference described under
[Running MD over partitions](#limits).

**The charge head's**, which this package owns, comes from
`build_neighbor_graph_fast`. With `method="auto"` it uses the cell list when
all three of these hold: the frame is on CUDA, nvalchemiops is available, and
the atom count clears the threshold for its boundary condition.

| frame | threshold | so in practice |
|---|---:|---|
| with a cell (`box` given) | `CELL_LIST_MIN_ATOMS_PBC = 0` | **always** — any size |
| no cell (`box=None`) | `CELL_LIST_MIN_ATOMS_FREE = 1024` | 1024 atoms and up |

Partitions are the second case: the halo already carries the images, so the
partitioned driver builds its graph with `box=None`.

The thresholds come from measurement, median of 7 after warm-up on one GB10
(ms):

```
atoms            192   384   768  1536  2592   6144
periodic  dense 1.06  2.35  7.60 29.37 83.13 470.12
          nv    1.01  1.06  1.22  4.02  3.84   4.26
box=None  dense 0.30  0.29  0.42  1.19  3.16  16.97
          nv    0.62  0.69  0.83  1.15  0.69   3.95
```

With a cell the cell list wins at every size measured, down to 192 atoms, so
there is no crossover to respect. Without one, the dense path has no ghost
images to materialise and stays cheaper until about a thousand atoms.

Pass `method="cell_list"` or `method="dense"` to force either; the graph is
identical either way, and `tests/test_neighbors.py` asserts it. On CPU, or
without nvalchemiops, `auto` always yields `dense` rather than raising.

Computing the virial does **not** force the dense builder. The strain is
applied to the finished edge vectors rather than to the coordinates and cell,
so the derivative never flows through the box — which is what lets the O(N)
builder be used on the charge graph at all.

## Tests

```bash
pytest tests/ -q
```

- `tests/test_pme_physics.py` — PME against the NaCl Madelung constant,
  `gradcheck`/`gradgradcheck`, and a guard for the `spline_order >= 7`
  silent-failure mode.
- `tests/test_plugin_registration.py` — entry-point registration, input-schema
  acceptance, equivalence to the standard model, and serialization round-trip.
- `tests/test_charge_head.py` — the total-charge projection (exactness,
  masking, differentiability) and the charge head's physical symmetries
  through a real DPA4C descriptor.
- `tests/test_lr_term.py` — the eV unit conversion (against analytic Coulomb
  and Madelung), boundary-condition guards, batching, masking, and gradients
  including the double backward that force training needs.
- `tests/test_joint_model.py` — the combined model: force and virial against
  finite differences of the total energy, energy decomposition, both calling
  conventions, and serialization of the long-range term.
- `tests/test_isolated_term.py`, `tests/test_born.py`,
  `tests/test_multi_task.py`, `tests/test_neighbors.py`,
  `tests/test_xyz_to_deepmd.py` — the non-periodic term, Born charges,
  multi-head sharing, the neighbour-graph swap, and the dataset converter.

The tests import only `deepmd`, never `deepmd_lr`, so they fail if the entry
point stops working rather than silently passing.

## Results

A controlled periodic benchmark where the joint model beats the strictly local
baseline by 4.9× on force RMSE and breaks through the floor a 6 Å cutoff
imposes; a control showing the gain disappears when there is no long-range
physics to capture; and the charge head recovering the true charge magnitudes
(up to the exact global-sign degeneracy above).

- `experiments/codex_test/` — independent correctness experiments: is the
  electrostatic energy the intended formula, and are the reported derivatives
  actually derivatives of it?
- `experiments/hopper_bench/RESULTS.md` — MD throughput in ns/day across the
  ASE, nvalchemiops and partitioned paths.

## Status

| Phase | | |
|---|---|---|
| 0 | Environment + PME validation | **done** |
| 1 | Plugin skeleton (`dp_lr` model type, no long-range term yet) | **done** |
| 2 | Charge head (`charge` fitting type + total-charge projection) | **done** |
| 3 | Long-range term (`LongRangeEnergy`, reciprocal-only, eV) | **done** |
| 4 | Joint model (`long_range` block, conservative forces) | **done** |
| 5 | Training + validation (benchmark + control) | **done** |
| 6 | MD deployment (ASE calculator, export guard) | **done** |
