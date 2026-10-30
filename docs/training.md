# The training schedule

### Plan A: joint from scratch (default)

Train everything at once, LR 1e-3 → 1e-5 with exponential decay. The benchmark
used it and converged without trouble: 4.9× better forces than the baseline,
charges settling near ±1 within a few thousand steps. Use it unless you hit a
symptom in [Troubleshooting](#troubleshooting).

### Plan B: staged (when Plan A is unstable)

| stage | steps | LR | what trains |
|-------|------:|----|-------------|
| 1. warm-up | 10–20% of budget | 1e-3 → 1e-4 | descriptor + energy head (no `long_range` block) |
| 2. graft   | — | — | copy weights across, scale down the charge head |
| 3. joint   | remainder | 5e-4 → 1e-5 | everything |

Stage 2 needs a helper, because `dp train --init-model` loads the state dict
strictly and refuses a checkpoint missing the charge head's weights:

```bash
dp --pt-expt train input_stage1.json          # long_range block removed

python scripts/graft_short_range.py \
    --short-range run_stage1/model.ckpt.pt \
    --input       input_joint.json \
    --out         init_joint.ckpt.pt

dp --pt-expt train input_joint.json --init-model init_joint.ckpt
```

The script scales the charge head's output layer by 0.01, and the scale has to
be small but not zero. Too large (a fresh head) predicts random charges whose
spurious long-range energy undoes the warm start: on the water set the warm-up
ended at 0.176 eV/Å and an unscaled graft restarted at **0.789**. Exactly zero
fails silently: the electrostatic energy is quadratic in charge, so at `q = 0`
every gradient reaching the charge head is zero (`sum|grad| = 0.000e+00`), the
head stays frozen, and training looks healthy because the short-range model
carries on alone. The script rejects zero. At 0.01 the graft is continuous and
the head alive:

| | long-range energy | max abs charge | training force RMSE |
|---|---:|---:|---:|
| warm-up ended | — | — | 0.176 eV/Å |
| grafted init  | 0.0002 eV | 4.1e-03 | 0.168 eV/Å |
| after stage 3 | 0.0492 eV | 8.7e-02 | 0.175 eV/Å |

### EMA is always on

Training keeps an exponential moving average of the parameters,
`ema = 0.999 * ema + 0.001 * param` after every optimizer step, written next to
the live checkpoint as `model_ema.ckpt-<step>.pt` (pointer `model_ema.ckpt.pt`).
While the plugin is installed `enable_ema` is forced to `true` for every
training run, `dp_lr` or not; `ema_decay` defaults to 0.999 and a decay you
write is kept. The saved `out.json` shows what ran; your input file is not
rewritten.

Why: the default loss schedule weights forces a thousand times more than
energy at the start and hands the weight to energy as the learning rate decays,
so late in training the force-shaped parameters drift on a small, noisy
gradient. Averaging the last ~1000 steps (the window is about `1/(1-decay)`)
gives a model that does not depend on which step training stopped at.

- **Deploy the EMA snapshot, restart from the live one.** `model_ema.ckpt.pt`
  holds averaged parameters plus the live buffers and no optimizer or EMA
  state; the calculator, the evaluator and `load_model` take it like any
  checkpoint. `--restart` needs `model.ckpt.pt`.
- **A new run is a new average.** `--init-model` drops the accumulator and
  starts from the loaded weights; only `--restart` continues it.
- **Short runs: shorten the window.** After 200 steps a 0.999 average is still
  82% initial weights; below a few thousand steps use 0.99 (~100 steps).
- **`zero_stage >= 2` is refused**: DeePMD-kit supports EMA only below it.

### Keeping the raw charges neutral: the `ener_lr` loss

The raw charges are made to sum to the frame's total charge by adding one
constant to every atom, so a constant offset in the raw charges costs nothing
and each material ends training with an offset of its own. A C/Si interface
model came out at +0.84963 e per atom in bulk diamond and +0.84664 e in bulk
silicon; at the interface the projection removes the common mean and leaves
the 0.3% difference as +q on every C and −q on every Si. That charge grows with
the thickness of each material, and on longer cells its forces broke MD.

`ener_lr` is DeePMD-kit's `ener` loss plus a term on the shift the projection
applied, `delta_f = (Q_f - sum_i raw_i) / N_f`:

```jsonc
"loss": {
  "type": "ener_lr",
  "start_pref_e": 1, "limit_pref_e": 1, "start_pref_f": 1000, "limit_pref_f": 1,
  "start_pref_qn": 1, "limit_pref_qn": 1
}
```

The term is `pref_qn * mean_f delta_f^2` (`mean_f |delta_f|` under
`loss_func: "mae"`) on the usual prefactor schedule. Every single-material
frame then pins its offset near zero, so two materials cannot disagree by more
than the term allows. Inference is unchanged: the projection still enforces
the total charge exactly. Every other `ener` argument works as before; in a
multi-task run set it in each branch's `loss_dict` entry.

- **Watch `rmse_qn`** in `lcurve.out`, the rms shift in e per atom. It bounds
  the offset difference between materials; the bulk charge of an interface
  scales with that difference and the long-range energy with its square, so
  1e-4 e against the 3e-3 e above is about 1,000 times less energy.
- **Prefactor.** Match the energy term's scale: under `mae`, `pref_qn = 1`
  weighs a 1e-3 e shift like a 1e-3 eV/atom energy error; under `mse`, take it
  of the order of `pref_e`. On example 01's water, `pref_qn = 100` took
  `rmse_qn` from 0.23 to 0.005 e in 300 steps, energies and forces improving
  alongside.
- **Fine-tuning with `--init-model` is enough**; the offset falls within the
  first few hundred steps.
- **Limits.** `validating.full_validation` accepts only `ener` and
  `ener_spin`; a penalty-only loss (every `ener` prefactor zero) fails inside
  DeePMD-kit's own loss, so keep an energy term.

## What to watch

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
| charge spread across frames | small for chemically equivalent atoms | large: the head is fitting noise |
| val force RMSE vs the baseline | clearly below | at or above: the term is not paying for itself |

`energy_long_range` is `recip − self`, the reciprocal sum with the Ewald self
term `(alpha/sqrt(pi)) * sum q_i^2` subtracted, so its complement is exactly
the screened real-space sum `sum_{i<j} q_i q_j erfc(alpha r_ij) / r_ij` the
network has to represent. Unlike DeePMD-kit's DPLR `ewald_reciprocal_energy`,
the self piece is not added back: it is an on-site quantity (103.0 eV of
208.8 eV on the LiFSI frame) and, being positive-definite in the charges, it
would bias the optimiser toward small charges.

## Validating a trained model

**Against the baseline**, the only claim that matters: held-out energy and
force RMSE against the identically configured run without `long_range`.

**The charges.** Expect the right magnitudes and relative signs, not the
overall sign: the energy is bilinear in charge, so `q → −q` leaves every
`q_i q_j` unchanged and for a neutral cell the two branches are exactly
degenerate (to 1.4e-14). Two runs of the benchmark converged to `+0.96/−0.96`
and `−0.94/+0.94`. The sign, and the scale, of the Born tensors are fitted
afterwards
([calibration](born-charges.md#calibrating-the-scale-and-sign-against-dfpt-labels)).

**Energy conservation.** Run NVE and watch the drift
([md](md.md#check-energy-conservation-before-trusting-a-trajectory)).

## Troubleshooting

| symptom | likely cause | fix |
|---------|--------------|-----|
| No improvement over baseline | little signal beyond the cutoff | check the [headroom](alpha-and-mesh.md#does-it-apply-to-your-data); if it is ~1%, the term cannot help |
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
| `RuntimeError: compression needs the compiled operator ...` | the PyPI wheel has no `dpa4c_graph_compress` kernel | [build the CUDA operators](compression.md#building-the-cuda-operators), or drop `compress=` |
| `RuntimeError: DP_CUDA_INFER=2 selects the fused energy/force kernel ...` when compressing, or `DP_CUDA_INFER=2 on a compressed model` at run time | level 2 bypasses `call_graph` and the shared backbone (refused only with `long_range`) | `export DP_CUDA_INFER=1` |
| `RuntimeError: ... needs the destination CSR views` | a hand-built graph reached a compressed model | build it with `build_neighbor_graph_fast(..., with_csr=True)` |
| `ValueError: charge_spin ... differs from the state ... folded into the compressed tables` | the run's condition is not the one compressed with | pass that `charge_spin` to the calculator/evaluator, which folds it in |
| `NotImplementedError: The graph-lower interface (DeepPot, dp test, ...)` | a long-range checkpoint evaluated through DeepPot/`dp test` | use `LongRangeCalculator` or the nvalchemi evaluator; only they see the cell |
