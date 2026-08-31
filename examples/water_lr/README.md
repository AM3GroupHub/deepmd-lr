# Water example

Periodic liquid water, using DeePMD-kit's own example dataset.

| file | what it is |
|------|------------|
| `input_joint.json`    | DPA4C + long-range electrostatics |
| `input_baseline.json` | the same model with `long_range` removed — the comparison that makes the result meaningful |
| `input_stage1.json`   | short-range warm-up, for the staged schedule in `TRAINING.md` §3 |

The data is not vendored here. Copy DeePMD-kit's water set in:

```bash
cp -r /path/to/deepmd-kit/examples/water/data ./data
```

Then, from a run directory:

```bash
mkdir -p run_joint && cd run_joint && ln -s ../data data
dp --pt-expt train ../input_joint.json
```

Train `input_baseline.json` the same way and compare held-out force RMSE.
See `../../TRAINING.md` for choosing `alpha` and the mesh, the staged
schedule, and what to watch during training.
