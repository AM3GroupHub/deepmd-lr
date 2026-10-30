# 5 — Bulk and slab, one set of weights

Your dataset has both: periodic bulk cells and 2D-periodic slabs, from the same
calculations. A single ordinary model cannot take both, because
`slab_correction` and `pbc` are fixed when the model is built. Training two
models wastes half the data on each.

You do not have to. **The long-range term holds no trainable weights.** Every
parameter lives in the descriptor, the energy head and the charge head; the
reciprocal-space sum has none. One set of weights genuinely serves both
boundary conditions — only the sum over periodic images differs.

`same_as` says that:

```json
"model_dict": {
  "bulk": { "type": "dp_lr", "type_map": "type_map_all",
            "descriptor": {...}, "fitting_net": {...},
            "long_range": { "charge_net": {...},
                            "reciprocal": {"alpha": 0.4, "slab_correction": false,
                                           "pbc": [true, true, true]} } },

  "slab": { "same_as": "bulk",
            "long_range": { "reciprocal": {"alpha": 0.4, "slab_correction": true,
                                           "pbc": [true, true, false]} } }
}
```

The `slab` branch **is** the `bulk` model. Not a copy with tied weights — the
same Python objects, so one gradient and one update per step whichever branch
the step was drawn from, and one set of per-type energy offsets fitted over
both datasets pooled.

## First: do you actually need this?

The correction a slab branch adds is Yeh–Berkowitz, `2π M_z² / V`. For a
**non-polar** slab `M_z = 0` and it is identically zero — an ordinary 3D branch
already handles both, and this file buys you nothing.

It earns its keep when the slab is **polar**: an electrode under electrolyte, an
asymmetric interface, anything with a net dipole across the vacuum. There the
correction is large and decays only as `1/L_z`, so no amount of padding removes
it.

## What the two subsets must have in common

This is the requirement that makes one weight set valid, and it is not
negotiable:

- **One level of theory.** Same functional, same basis, same code, same
  settings. The branches share their energy zero, because they share the
  fitting net's `bias_atom_e`.
- **One chemistry**, or at least one `type_map` covering both.
- **The same `alpha`.** It is where Coulomb is split, and the one shared
  descriptor can only learn the `erfc(αr)/r` remainder of a single split. A
  mismatch is refused rather than silently honoured.

If your two subsets come from *different* theory, you want separate heads
sharing a descriptor — that is [example 04](../04-multi-head), and `same_as`
would be wrong there.

## Step 1 — build the two datasets

This example ships no data. Convert your own into `data/bulk` and `data/slab`,
or point `data_dict` wherever they already live:

```bash
python ../../scripts/xyz_to_deepmd.py bulk.xyz -o data/bulk --valid-fraction 0.1
python ../../scripts/xyz_to_deepmd.py slab.xyz -o data/slab --vacuum 52.0 \
    --valid-fraction 0.15
```

`--vacuum` matters for the slab and is explained in [example 03](../03-slab):
the gap must exceed `2 × rcut`, or atoms near one face fall inside the cutoff of
the next image's other face.

Then size the two meshes. Each must beat `1 / (2α) = 1.25 Å` in every
direction — for the padded slab that means a much larger `z`.

## Step 2 — train

```bash
dp --pt-expt train input.json
```

`lcurve.out` carries one `rmse_f` column per branch. Watch **both**: the point
of this example is that one weight set serves both, and the way that fails is
one column falling while the other stalls. If it does, `model_prob` is the knob
— it is the probability a step is drawn from that branch, not a data-size
weight.

## What is checked for you

A pseudo-head that quietly diverged would still train and still converge; it
would just not be the model you asked for. So the divergences are refused, each
naming the branch:

| Written in a `same_as` branch | Why it is refused |
|---|---|
| `descriptor`, `fitting_net`, `type_map` | would be built, then discarded when the modules are aliased |
| `long_range.charge_net` | holds weights, so it must be the same object — not an equal copy |
| a different `alpha` | the shared descriptor learned one split |
| `fitting_net.dim_case_embd > 0` | gives each branch its own index in one shared buffer |
| `same_as` pointing at another `same_as` branch | leaves it ambiguous which model is the original |

A `same_as` branch may set only `long_range`, `model_branch_alias` and `info`.

> **Remove `stat_file` if you add or change `same_as`.** The per-type offsets
> are restored from it when present, which would bypass the pooled fit and
> leave them fitted on one branch's frames.

## Step 3 — use it

Load either head; the weights are the same tensors, and either one runs a bulk
cell or a slab, because the frame's `pbc` picks the sum:

```python
from deepmd_lr.ase import LongRangeCalculator

atoms.calc = LongRangeCalculator("model.ckpt.pt", head="slab")   # any pbc
```

The same holds for `build_evaluator` in an MD run
([example 06](../06-md_nvalchemi)). To change how the sum is taken, see
[setting the long-range type at load](../../docs/same-as.md#setting-the-long-range-type-at-load).
