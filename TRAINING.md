# Training a DPA4C model with long-range electrostatics

A practical recipe. Every number here was measured on this setup, not taken
from folklore — where a choice is a judgement call rather than a constraint,
it says so.

---

## 0. First: will it help at all?

Two prerequisites, then one measurement worth doing before spending GPU time.

**Your data must be periodic.** Ewald summation is only defined under PBC.
If your frames have no cell (`pbc = F F F` — molecular clusters, most OMol25-
style datasets), this term does not apply and you want a real-space
alternative instead; see *Non-periodic systems* at the bottom.

**You need energies and forces. You do not need charges.** The charge head is
trained through the electrostatic energy, never against charge labels.

**Measure the headroom.** The long-range term can only recover the part of
the signal that lives beyond the descriptor cutoff. If that is negligible,
nothing downstream will help. Compare exact Ewald forces against Coulomb
truncated at your cutoff:

```bash
python experiments/longrange_benchmark/headroom.py
```

On the benchmark system this reports **17.7% of the force signal beyond 6 Å**,
which corresponds to a 0.37 eV/Å floor no local model can beat — and the
baseline duly plateaued at 0.41. If your equivalent number is ~1%, expect a
correspondingly small return.

---

## 1. Choosing the method: `pme` or `ewald`

Both compute the *same* reciprocal sum, by different routes.

| | `pme` (default) | `ewald` |
|---|---|---|
| how | charges onto a mesh, then an FFT | explicit sum over k-vectors |
| scaling | `O(N log N)` | `O(N · K)`, and `K` grows with cell volume |
| error | mesh interpolation, ~1e-6 at a sane mesh | none — exact for its `k_cutoff` |
| knobs | `mesh_dimensions` / `mesh_spacing`, `spline_order` | `k_cutoff` (or `accuracy`) |

Measured on this machine (energy + backward, alpha 0.4, 1.25 Å mesh):

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
the physics. A cheap sanity check before a long run:

```python
from deepmd_lr.lr import LongRangeEnergy
pme   = LongRangeEnergy(alpha=0.4, mesh_dimensions=[16, 16, 16])
ewald = LongRangeEnergy(alpha=0.4, method="ewald")
a, b = pme(coord, q, box).sum(), ewald(coord, q, box).sum()
print(abs(a - b) / abs(b))       # want <= 1e-4; if not, refine the mesh
```

Set it in the input file with `"method": "ewald"`. Parameters belonging to
the other method are **rejected**, not ignored — a mesh setting silently
dropped under `ewald` would be indistinguishable from one that was applied.

## 2. Choosing `alpha` and the mesh

`alpha` is **not** a free parameter here. This term supplies only the
reciprocal half of the Ewald split, so `alpha` decides the division of labour:

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

---

## 3. The input file

Annotated; the `_comment` keys are ignored by DeePMD-kit.

```jsonc
{
  "model": {
    "type": "dp_lr",
    "type_map": ["Na", "Cl"],

    "descriptor": {
      "type": "dpa4c",
      "rcut": 6.0,
      "channels": 32,
      "lmax": 2,
      "precision": "float64"
    },

    "fitting_net": {
      "_comment": "The short-range energy head. Unchanged from a normal run.",
      "neuron": [128, 128, 128],
      "activation_function": "silu",
      "precision": "float64"
    },

    "long_range": {
      "_comment": "Omit this whole block and you get the plain DPA4C model.",

      "charge_net": {
        "_comment": "Predicts per-atom charges from the same descriptor. Small is fine -- it outputs one scalar per atom.",
        "neuron": [64, 64],
        "activation_function": "silu",
        "precision": "float64"
      },

      "pme": {
        "_comment": "method defaults to 'pme'; set 'ewald' for the exact k-sum (see section 1).",
        "_comment_alpha": "alpha 0.4 pairs with rcut 6.0 (erfc(2.4) = 7e-4).",
        "alpha": 0.4,
        "_comment_mesh": "Fixed mesh, ~1.25 A spacing for a 20 A cell.",
        "mesh_dimensions": [16, 16, 16],
        "spline_order": 6
      }
    }
  },

  "learning_rate": { "type": "exp", "start_lr": 1e-3, "stop_lr": 1e-5 },

  "loss": {
    "_comment": "Force-weighted. The long-range signal shows up far more clearly in forces than in the total energy.",
    "type": "ener",
    "start_pref_e": 1, "limit_pref_e": 1,
    "start_pref_f": 100, "limit_pref_f": 100
  },

  "training": {
    "training_data":   { "systems": ["./data/train"], "batch_size": 4 },
    "validation_data": { "systems": ["./data/valid"], "batch_size": 4, "numb_btch": 10 },
    "numb_steps": 200000,
    "disp_freq": 500,
    "seed": 1
  }
}
```

**Always write the baseline twin.** Copy the file, delete `"type": "dp_lr"`
and the `long_range` block, change nothing else. Without it you cannot tell
whether the long-range term helped or the run merely went well — this is the
single most important methodological point in this document.

```bash
dp --pt-expt train input_joint.json
dp --pt-expt train input_baseline.json     # same settings, no long_range
```

---

## 4. The training schedule

### Plan A — joint from scratch (default)

Train everything at once. This is what the Phase 5 benchmark used and it
converged without trouble: 4.9× better forces than the baseline, charges
settling near ±1 within a few thousand steps.

| stage | steps | LR | what trains |
|-------|------:|----|-------------|
| joint | full budget | 1e-3 → 1e-5 (exp decay) | descriptor, energy head, charge head |

Use it unless you hit one of the symptoms in §7.

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
# 1. warm up
dp --pt-expt train input_stage1.json          # long_range block removed

# 2. graft the warm-up weights into the joint architecture
python scripts/graft_short_range.py \
    --short-range run_stage1/model.ckpt.pt \
    --input       input_joint.json \
    --out         init_joint.ckpt.pt

# 3. continue jointly
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
  alone.

At the default 0.01 the graft is continuous and the head is alive. Measured on
the water set:

| | long-range energy | max abs charge | training force RMSE |
|---|---:|---:|---:|
| warm-up ended | — | — | 0.176 eV/Å |
| grafted init  | 0.0002 eV | 4.1e-03 | 0.168 eV/Å |
| after stage 3 | 0.0492 eV | 8.7e-02 | 0.175 eV/Å |

The long-range term starts negligible, preserving the warm start, then grows
by ~20× as the head learns. Use `--charge-init-scale` to tune it; the script
rejects zero.

---

## 5. What to watch

The model returns three diagnostics beyond the usual outputs:

```python
out = model(coord, atype, box=box)
out["energy_short_range"]   # DPA4C
out["energy_long_range"]    # the reciprocal term
out["charge"]               # per-atom, after the total-charge projection
```

| signal | healthy | trouble |
|--------|---------|---------|
| `energy_long_range / energy_short_range` | a few % to tens of % | ~0 (term is doing nothing) or dominant (it is absorbing short-range physics) |
| charge magnitudes | stable, physically plausible | growing without bound, or drifting run to run |
| charge spread across frames | small for chemically equivalent atoms | large — the head is fitting noise |
| val force RMSE vs the baseline | clearly below | at or above — the term is not paying for itself |

---

## 6. Validating a trained model

**Against the baseline.** The only claim that matters. Compare held-out
energy and force RMSE against the identically configured run without
`long_range`.

**The charges.** Expect the right magnitudes and relative signs. Do **not**
expect the right overall sign: the energy is bilinear in charge, so `q → −q`
leaves every `q_i q_j` unchanged and, for a neutral cell, the two branches are
*exactly* degenerate (verified to 1.4e-14). Two runs of the same benchmark
here converged to `+0.96/−0.96` and `−0.94/+0.94` respectively; both are
equally optimal. If you need physically signed charges — to compare against
Bader or Mulliken — you must break the symmetry yourself with a charge label
or a sign constraint.

**Energy conservation.** The sharpest end-to-end test that forces really are
the gradient of the energy, long-range term included: run NVE and watch the
drift.

---

## 7. Troubleshooting

| symptom | likely cause | fix |
|---------|--------------|-----|
| No improvement over baseline | little signal beyond the cutoff | run the headroom check; if it is ~1%, the term cannot help |
| No improvement, but headroom is large | `alpha` too small, so the network is still being handed a long tail | raise `alpha` per §2 |
| Loss spikes when long-range switches on | charge head initialized random | use `graft_short_range.py` without `--keep-charge-init` |
| Charges grow without bound | short-range and long-range trading off against each other | raise the force weight, or add a charge regularizer |
| Virial wrong by a few % | mesh jumping with the cell | use `mesh_dimensions`, not `mesh_spacing` |
| `TypeError: unexpected long-range parameters: ['rcut']` | stale config | `rcut` was removed; only the reciprocal half is computed, so there is no real-space cutoff |
| `ValueError: ... only apply to method='pme'` | mesh knob set with `method: ewald` | use `k_cutoff`, or drop `method` to stay on PME |
| `pme` and `ewald` disagree by more than ~1e-4 | mesh too coarse for this `alpha` | refine the mesh per §2, or lower `alpha` |
| Energies fine, forces bad | `alpha`/mesh mismatch | check the mesh resolves `1/(2·alpha)` |

---

## 8. Cost

The joint model runs the descriptor twice — the short-range path consumes and
discards its own descriptor output, and the charge head needs it again.
Measured at **~3× per training step** on small systems (24 atoms: 15 → 45 ms;
96 atoms: 14 → 46 ms). Those sizes are launch-overhead dominated, so expect
the ratio to improve with system size. Budget accordingly; PLAN.md Phase 4
describes the single-pass optimization if it becomes the constraint.

---

## 9. Running MD with the trained model

> **Do not `dp freeze` a model with a `long_range` block.** The exported
> artifact is traced from the graph lower, which receives edge vectors and no
> cell, so it cannot evaluate the reciprocal-space term. Before this was
> guarded, `dp freeze` succeeded silently and the frozen model returned
> **−93.74 eV** where the eager model gives **−55.12** — the entire 38.6 eV
> long-range contribution gone, with no warning. It now raises instead.
> Freezing a short-range-only model (no `long_range` block) still works.

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
trajectory without re-running the model.

This is eager PyTorch, not a compiled artifact, so it is slower than a frozen
short-range model. That is the price of computing physics the exported path
cannot express.

### Check energy conservation before trusting a trajectory

The sharpest end-to-end test that the forces really are the gradient of the
energy, long-range term included:

```bash
python examples/lifsi/md.py \
    --model run_joint/model.ckpt.pt --ensemble nve \
    --steps 1000 --timestep 0.5
```

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

---

## Non-periodic systems

If your frames have no cell, none of the above applies: Ewald needs
periodicity, and `LongRangeEnergy` will refuse a `box=None` frame rather than
silently pretend. The charge head, the total-charge projection, the model
wiring and the force/virial handling are all engine-agnostic — only the
energy term itself would need a sibling built on `nvalchemiops`'s `coulomb`
or `dsf` (damped shifted force), both of which are already differentiable
with charge gradients. See PLAN.md, Phase 5.

---

## 10. Running MD over partitions

`deepmd_lr.ase.LongRangeCalculator` is single-process. For larger systems,
`DistLongRangeCalculator` evaluates the same model over spatial partitions:

```python
from deepmd_lr.distributed import DistLongRangeCalculator

atoms.calc = DistLongRangeCalculator("run_joint/model.ckpt.pt")
atoms.calc.enable_distributed_mode(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
```

or wrap a calculator you already have:

```python
atoms.calc = DistLongRangeCalculator.from_existing(calc).enable_distributed_mode(4)
```

Passing an integer means "that many partitions on the current device" — which
is how the partitioning is exercised where only one GPU exists. Results are
identical either way; only placement differs.

### What is split and what is not

| piece | cost at 6520 atoms | treatment |
|---|---:|---|
| descriptor + short-range | 883 ms | partitioned |
| charge prediction | (same pass) | partitioned |
| reciprocal sum + charge projection | 73 ms | gathered onto one device |
| charge-response force | — | partitioned |

The reciprocal sum is gathered rather than decomposed because it does not
survive spatial decomposition — splitting it gave 2.0% error on 2 ranks and
36.5% on 8 (`LAMMPS.md`). That is affordable because it is under 10% of the
step and the round trip is ~230 KB. Only the descriptor is worth splitting.

The charge-response force, `-Σⱼ (dE/dqⱼ) ∂qⱼ/∂rᵢ`, stays partitioned: `∂qⱼ/∂rᵢ`
is nonzero only within `rcut`, so although the *energy* is long-ranged its
*charge derivative* is short-ranged. Each partition backpropagates its own
atoms' cotangents on its own device.

### Reusing a partitioning across steps

Ghosts are gathered within `rcut + skin`, and the partitioning is rebuilt only
once atoms have moved far enough to threaten it (the Verlet criterion,
`2·max displacement > skin`). In a 40-step test that meant 3 rebuilds instead
of 40, with results identical to rebuilding every step.

```python
DistLongRangeCalculator(model, skin=1.0, repartition_every=None)
```

`repartition_every=1` forces a rebuild each step; it changes timing, never
results — there is a test asserting exactly that.

### Limits

- **`stress` is not provided.** The partitioned path assembles energy and
  forces only. Use `LongRangeCalculator` when the virial is needed.
- **Multi-GPU is correct but not yet faster.** Verified on 2 × A40: results
  are identical across `cuda:0`, `cuda:0`×2 and `cuda:0`+`cuda:1` (ΔE = 0 at
  768 and 2592 atoms, 9.1e-13 at 6144), and the 43 partition-invariance tests
  pass on real devices. But the measured speedup from a second GPU is
  **0.92–1.03×** — i.e. none.

  The reason is measured, not guessed: with the partition loop staged so that
  all devices are launched before any host transfer, **the step is still
  ~100% CPU-bound** — at 768 atoms there is 0 ms of GPU work outstanding when
  the launch loop returns, and 2.4 ms (8%) at 2592. The ~11 ms per partition
  is eager-PyTorch dispatch in the descriptor and charge head, not GPU
  compute, so a second device has nothing to overlap with.

  Adding GPUs will not help until that per-partition CPU cost comes down —
  through a compiled descriptor, CUDA graphs, or per-device launch threads.
  Partitioning also *multiplies* that fixed cost, so fewer, larger partitions
  are cheaper than many small ones until then.

- **The neighbour search uses nvalchemiops' GPU cell list above a size
  threshold.** DeePMD-kit's `build_neighbor_graph` calls itself a "correctness
  oracle" and is O(N²); `deepmd_lr/neighbors.py` swaps in
  `nvalchemiops.torch.neighbors.cell_list` once a frame is big enough to pay
  for it (512 atoms periodic, 3072 non-periodic — partitions are the latter,
  since the halo already carries the images).

  Only the *search* is taken from nvalchemiops. Its edge vectors come out of a
  Warp kernel with no gradient, so they are discarded and DeePMD-kit's
  `neighbor_graph_from_ijs` recomputes `edge_vec` from `coord`/`box`. Using the
  kernel's vectors directly would produce a model with no forces.

  Edge sets are identical to the oracle — same count, same vectors to 9
  decimals, periodic and not (`tests/test_neighbors.py`). End-to-end this is
  1.00–1.77× over 768–12288 atoms, growing with system size.
