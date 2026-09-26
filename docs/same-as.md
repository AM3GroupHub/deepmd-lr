# One theory, two boundaries: `same_as`

Worked example: [`examples/05-pseudo-head`](../examples/05-pseudo-head).

[Multi-head](multi-head.md) answers "different levels of theory". A different question:
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

