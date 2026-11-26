# 11-sr-lr-ft: the SR → LR → FT curriculum

Trains the water model of [example 01](../01-bulk) in three stages instead of
jointly, on the same dataset (build `../01-bulk/data` there first).

```bash
./train.sh        # sr/, lr/, ft/ are created here; the model is ft/model_ema.ckpt.pt
```

| stage | file | what trains | learning rate | steps |
|---|---|---|---|---|
| SR | `sr.json` | descriptor and energy head; no `long_range` block | 1e-3 → 1e-5 | 100,000 |
| LR | `lr.json` | only the charge head, on the SR model frozen (`long_range.train_charge_head_only: true`) | 1e-3 → 1e-5 | 50,000 |
| FT | `ft.json` | everything, from the LR model | 1e-4 → 1e-6 | 50,000 |

The three files are example 01's `input.json` with these differences only:
the `long_range` block (absent in SR, with the freeze key in LR), the learning
rate and the step count. The data paths are written relative to the stage
folders.

Between SR and LR, `scripts/graft_short_range.py` writes the checkpoint the LR
stage starts from: the SR weights plus a charge head whose output is scaled to
0.01, small so the first step is not thrown off and nonzero so the head has a
gradient. From LR to FT `--init-model` takes the live checkpoint as it is.

What to look at:

- `lr/lcurve.out`: the force error starts at the SR stage's final value and
  can only fall by what the charges explain; a flat curve means the
  short-range model left nothing electrostatic within reach of the loss.
- the charges after LR and after FT (`calc.results["charges"]`, see
  [example 07](../07-ase-calculator)): small after LR, growing in FT.

Whether the curriculum beats joint training is to be measured per system; see
[training](../../docs/training.md#plan-b-sr--lr--ft-staged). For a DPA4 model
change `descriptor` and `fitting_net` in all three files alike
([dpa4](../../docs/dpa4.md)).
