# One theory, two boundaries: `same_as`

Worked example: [`examples/05-pseudo-head`](../examples/05-pseudo-head).

[Multi-head](multi-head.md) answers "different levels of theory". `same_as`
answers a different question: one level of theory whose dataset holds both
periodic bulk cells and 2D-periodic slabs. In training a branch's
`slab_correction` and `pbc` are fixed when it is built, so one ordinary branch
cannot take both. Separate heads would each fit their own energy zero and each
see half the data, and they are unnecessary, because
**the long-range term holds no trainable weights**. Every parameter lives in
the descriptor, the energy head and the charge head, so one set of weights
serves both boundary conditions and only the sum over periodic images differs.

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
everything parameter-bearing: descriptor, energy head, charge head and the
per-type energy offsets, as the same Python objects, so one gradient and one
update per step whichever branch the frame came from. The offsets are fitted
once over both datasets pooled, which stops a gap between them being absorbed
as a spurious surface energy. The two mechanisms compose: `shared_dict` for
genuinely different heads, `same_as` for heads that are the same model. A
multi-task config still needs exactly one shared `type_map` reference, so the
`shared_dict` stays.

**Do you need it?** The slab branch adds the Yeh–Berkowitz correction,
`2π M_z²/V`, which is identically zero for a non-polar slab. If your slabs
carry no dipole across the vacuum, one 3D branch already serves both. It earns
its keep for polar slabs (an electrode under electrolyte, an asymmetric
interface), where the correction is large and decays only as `1/L_z`.

**What the two subsets must share.** One level of theory (same functional,
basis, code and settings: the branches share the fitting net's `bias_atom_e`,
so their energy zero), one `type_map` covering both, and the same `alpha`. If
the subsets come from different theories, use separate heads sharing a
descriptor ([multi-head](multi-head.md)); `same_as` would be wrong there.

A `same_as` branch may set only `long_range`, `model_branch_alias` and `info`.
A pseudo-head that quietly diverged would still train and converge, just not
as the model you asked for, so every divergence is refused, naming the branch:

| written in a `same_as` branch | why it is refused |
|---|---|
| `descriptor`, `fitting_net`, `type_map`, ... | would be built, then discarded when the modules are aliased |
| `long_range.charge_net` | holds weights, so it must be the same object, not an equal copy |
| a different `alpha` | the shared descriptor learned one split |
| `fitting_net.dim_case_embd > 0` | gives each branch its own index in one shared buffer |
| `same_as` pointing at another `same_as` branch | leaves it ambiguous which model is the original |

> Remove `stat_file` when you add or change `same_as`: the offsets are restored
> from it when present, bypassing the pooled fit and leaving them fitted on
> one branch's frames.

### Setting the long-range type at load

The same fact, no weights in the term, makes the boundary treatment a property
of the simulation rather than of the checkpoint. At inference either head runs
a bulk cell, a slab or a molecule: the frame's `pbc` picks the 3D, the slab or
the isolated sum ([non-periodic](non-periodic.md)), and its cell the PME mesh.
What remains to set by hand is how the sum is taken:

```python
from deepmd_lr.ase import LongRangeCalculator, load_model

model = load_model("model.ckpt.pt", head="slab")
model.set_long_range({"method": "ewald"})
```

`long_range=` is also a keyword on `load_model`, `LongRangeCalculator` and
`build_evaluator`. It replaces the reciprocal block rather than merging into
it, since `method: "isolated"` refuses every periodic parameter. `alpha` is
inherited from the checkpoint and a different one raises: it is where Coulomb
is split, and the network learned the remainder of that split.
