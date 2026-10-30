# 5 — Bulk and slab, one set of weights

Your dataset has both periodic bulk cells and 2D-periodic slabs, from the same
calculations. `input.json` trains them as one set of weights with `same_as`:
the `slab` branch **is** the `bulk` model, with the slab form of the
long-range sum. [`same_as`](../../docs/same-as.md) explains why one weight set
is valid, whether you need it at all (only for polar slabs), what the two
subsets must share (one level of theory, one `type_map`, one `alpha`), and
what a `same_as` branch may not set. If your subsets come from different
theories, you want [example 04](../04-multi-head) instead.

> **Remove `stat_file` if you add or change `same_as`**, or the per-type
> offsets come back fitted on one branch's frames.

## Step 1 — build the two datasets

This example ships no data. Convert your own into `data/bulk` and `data/slab`,
or point `data_dict` wherever they already live:

```bash
python ../../scripts/xyz_to_deepmd.py bulk.xyz -o data/bulk --valid-fraction 0.1
python ../../scripts/xyz_to_deepmd.py slab.xyz -o data/slab --vacuum 52.0 \
    --valid-fraction 0.15
```

`--vacuum` matters for the slab ([example 03](../03-slab#why---vacuum-is-not-optional)).
Then size the two meshes: each must beat `1 / (2α) = 1.25 Å` in every
direction, so the padded slab needs a much larger `z`
([alpha and the mesh](../../docs/alpha-and-mesh.md#choosing-alpha-and-the-mesh)).

## Step 2 — train

```bash
dp --pt-expt train input.json
```

`lcurve.out` carries one `rmse_f` column per branch. Watch **both**: the point
of this example is that one weight set serves both, and the way that fails is
one column falling while the other stalls. If it does, `model_prob` is the knob
— it is the probability a step is drawn from that branch, not a data-size
weight.

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
