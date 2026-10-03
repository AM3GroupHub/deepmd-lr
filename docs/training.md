# The training schedule

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

### EMA is always on

Training keeps an exponential moving average of the parameters,
`ema = 0.999 * ema + 0.001 * param` after every optimizer step, and writes it
next to the live checkpoint as `model_ema.ckpt-<step>.pt` (pointer
`model_ema.ckpt.pt`). While the plugin is installed this is enforced for every
training run, `dp_lr` or a plain DeePMD-kit model alike: `enable_ema` is set to
`true` even when the input says `false`. `ema_decay` defaults to 0.999, and a
decay you write yourself is kept. The change is applied to a copy of the raw
input before normalization, so the saved `out.json` shows what actually ran;
your input file is not rewritten.

Why: the default loss schedule weights forces a thousand times more than
energy at the start and hands the weight to energy as the learning rate
decays; late in training the force-shaped parameters drift on a small, noisy
gradient. Averaging the last ~1000 steps (the window is about `1/(1-decay)`)
gives a production model that does not depend on which step training happened
to stop at, for the price of one extra parameter copy.

What to know:

- **Deploy the EMA snapshot, restart from the live one.** `model_ema.ckpt.pt`
  holds averaged parameters plus the live buffers (`out_bias`, `out_std`,
  descriptor statistics) and carries neither optimizer nor EMA state; the
  ASE calculator, the partitioned evaluator and `load_model` take it like any
  checkpoint. `--restart` needs `model.ckpt.pt`.
- **A new run is a new average.** `--init-model` (Plan B's stage 3, a graft,
  a fine-tune) drops the previous accumulator and starts averaging from the
  loaded weights; only `--restart` continues it.
- **Short runs: shorten the window.** After 200 steps a 0.999 average is still
  82 % initial weights. EMA cannot be switched off, so below a few thousand
  steps use a smaller decay: 0.99 averages ~100 steps.
- **`zero_stage >= 2` is refused.** DeePMD-kit supports EMA only for
  `zero_stage < 2`, and the plugin raises rather than train without it.

The examples write both keys explicitly so a reader sees what will run.

### Keeping the raw charges neutral: the `ener_lr` loss

The charge head's raw charges are made to sum to the frame's total charge by
adding one constant to every atom. So a constant in the raw charges costs
nothing, and a frame of a single material never sets it: each material ends
training with a raw offset of its own. A C/Si interface model came out with
+0.84963 e per atom in bulk diamond and +0.84664 e in bulk silicon. At the
interface the projection removes their common mean, and the 0.3 % difference
is left as +q on every C atom and -q on every Si atom. That charge grows with
the thickness of each material, not with the interface area, and on longer
cells its forces broke MD.

`ener_lr` is DeePMD-kit's `ener` loss plus a term on the shift the projection
applied to every atom of a frame, `delta_f = (Q_f - sum_i raw_i) / N_f`:

```jsonc
"loss": {
  "type": "ener_lr",
  "start_pref_e": 1, "limit_pref_e": 1, "start_pref_f": 1000, "limit_pref_f": 1,
  "start_pref_qn": 1, "limit_pref_qn": 1
}
```

The term is `pref_qn * mean_f delta_f^2`, or `mean_f |delta_f|` under
`loss_func: "mae"`, on the usual prefactor schedule. It drives the raw charges
to satisfy the constraint by themselves. Every single-material frame then pins
its material's offset near zero, and two materials can no longer disagree by
more than the term allows. Nothing in the architecture changes, and inference
does not either: the projection still enforces the total charge exactly. Every
other `ener` argument works as before. In a multi-task run, set it in each
branch's `loss_dict` entry.

- **Watch `rmse_qn`** in `lcurve.out`, the rms shift in e per atom. It bounds
  the offset difference between materials. The bulk charge of an interface
  scales with that difference, and the long-range energy with its square: at
  1e-4 e against the 3e-3 e above, the energy is about 1,000 times smaller.
- **Choosing the prefactor.** Match the energy term's scale. Under `mae`,
  `pref_qn = 1` weighs a 1e-3 e shift like a 1e-3 eV/atom energy error. Under
  `mse`, take `pref_qn` of the order of `pref_e`. Measured on example 01's
  water, `pref_qn = 100` took `rmse_qn` from 0.23 to 0.005 e in 300 steps,
  with energies and forces improving alongside.
- **Fine-tuning an existing model** with `--init-model` is enough; its offset
  starts large (about 0.85 e above) and falls within the first few hundred
  steps.
- **Limits.** `validating.full_validation` is DeePMD-kit's and accepts only
  `ener` and `ener_spin`. A penalty-only loss (every `ener` prefactor zero)
  fails inside DeePMD-kit's own loss; keep an energy term.

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

