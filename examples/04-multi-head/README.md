# 4 — Multi-head: all three at once

Examples 01, 02 and 03 trained as **one model with one shared descriptor**.

| branch | from | boundary | labels | atoms |
|---|---|---|---|---:|
| `bulk` | [01](../01-bulk) | 3D-periodic | RPBE-D3 | 192 |
| `molecules` | [02](../02-molecules) | no cell | OMol25 DFT | 8–32 |
| `slab` | [03](../03-slab) | 2D-periodic + Yeh–Berkowitz | UMA MLIP | 468 |

Three things forbid one head. The **boundary treatment** differs; the **levels
of theory** differ; the **energy zeros** are unrelated. What they can share is
the descriptor — local geometry is local geometry, and a backbone that has
seen liquid water, charged molecules and a metal/electrolyte interface is a
better backbone than one that has seen any of them alone.

Everything below was run. The important part is the section on gradients:
this example does not work without it.

## Prerequisite

This example ships no data. Run **step 1 of each of 01, 02 and 03** first —
`input.json` reads their `data/` folders in place.

```bash
ls ../01-bulk/data/train ../02-molecules/data/*/train ../03-slab/data/train
```

## The shape of a multi-task config

Four things change relative to a single-head file:

| single head | multi-head |
|---|---|
| `model` | `model.shared_dict` + `model.model_dict` |
| `loss` | `loss_dict`, keyed by branch |
| `training.training_data` | `training.data_dict`, keyed by branch |
| — | `training.model_prob`, keyed by branch |

Multi-task is triggered by the presence of `model_dict`. Note that `pt_expt`
does **not** cascade top-level defaults into branches: whatever a branch
needs, a branch must state.

### Sharing is aliasing, not copying

```jsonc
"shared_dict": {
  "type_map_all": ["C", "Cl", "F", "H", "Li", "N", "O", "P", "S", "Si"],
  "backbone": { "type": "dpa4c", ... }
},
"model_dict": {
  "bulk": { "type_map": "type_map_all", "descriptor": "backbone", ... }
}
```

A branch writes the **name** where a dict would go. Every branch that names
`backbone` ends up holding the same parameter tensors — verified: all 15
descriptor parameters are literally the same objects across the three
branches. One backbone, updated by whichever branch each step was drawn from.

Only **`type_map`, `descriptor` and `fitting_net`** can be shared. `long_range`
cannot, which is exactly what lets the boundary treatment differ per branch.
The fitting nets here are deliberately *not* shared — three levels of theory
need three heads.

### One `type_map`, the union

Multi-task asserts a single `type_map`, so it is the union of all three:
`[C, Cl, F, H, Li, N, O, P, S, Si]`. No dataset is renumbered — each system
keeps its own local `type_map.raw` and DeePMD-kit remaps at load time.

### Same `alpha` in every branch

`alpha: 0.4` in all three, with three different boundary treatments around it:

```jsonc
"bulk":      { "alpha": 0.4, "mesh_dimensions": [10,10,10], "slab_correction": false, "pbc": [true,true,true] }
"molecules": { "alpha": 0.4, "method": "isolated" }
"slab":      { "alpha": 0.4, "mesh_dimensions": [12,12,42], "slab_correction": true,  "pbc": [true,true,false] }
```

The shared backbone learns the `erfc(αr)/r` remainder. **One backbone can
only learn one remainder**, so branches that split Coulomb at different
alphas would be asking it for two functions at once.

### `add_chg_spin_ebd` must be true everywhere

It is part of DPA4C's **structural signature** — the set of options that fix
the module's shape. Branches whose signatures disagree cannot be aliased, so
turning it on for the molecules means turning it on for all three. Writing it
once in `shared_dict` makes that automatic. `default_chg_spin: [0, 1]` then
covers `bulk` and `slab`, whose data carries no `charge_spin.npy`, and is
deliberately **not** part of the signature.

### `rcut` is now a global constraint

6.0 Å must stay below half of the *smallest* cell in *any* branch — bulk water
at 12.429 Å gives 6.21 Å. Adding a branch with a smaller cell would force
`rcut` down for everything, because there is only one descriptor.

### `data_stat_nbatch` is a model key

It sits in each branch of `model_dict`, not in `training`. Putting it in
`training` fails normalization outright:

```
undefined key `data_stat_nbatch` is not allowed in strict mode
```

The default of 10 builds a dense neighbour matrix over 10 frames at once; at
468 atoms that is tens of GiB, so the slab wants 1.

## The part that actually matters: gradient imbalance

**Without `gradient_max_norm`, this model does not train.** Measured at
initialization, the gradient norm each branch puts on the shared backbone:

| branch | descriptor `stddev` min | amplification | grad norm on shared backbone |
|---|---:|---:|---:|
| `bulk` | 3.6e-05 | ×28,000 | **5.0e+09** |
| `molecules` | 7.8e-03 | ×128 | 1.9e+06 |
| `slab` | 1.9e-03 | ×538 | 4.8e+05 |

Four orders of magnitude. One bulk step moves the shared weights further than
ten thousand slab steps, so the slab branch is simply overwritten.

The cause is DPA4C's calibration. `compute_input_stats` measures the spread of
each descriptor coordinate on that branch's own data and divides by it — a
fixed preconditioner set once at initialization. Bulk liquid water is dense and
isotropic, so **about 20 of its 144 coordinates have a spread below 1e-3**
(19–21 depending on the frames sampled), the tightest at 3.6e-05. Dividing by that amplifies those directions ×28,000, and
gradients come back out through the same factor.

Standalone this is harmless: example 01's head is matched to its own
preconditioner (it is ×48,000 there even with its own 2-element `type_map`, so
the union is not the cause). **Shared, it is not harmless** — each branch keeps
its own calibration, but they all write into one set of weights.

Measured at 600 steps, one seed (branch draws are random, so
expect the slab column to move by ~50% run to run — not by the factor of 15
that clipping buys):

| `gradient_max_norm` | bulk `rmse_f` | molecules | slab |
|---|---:|---:|---:|
| *(unset)* | 0.403 | 1.43 | **15.0** |
| 5.0 | 0.221 | 1.24 | **1.00** |
| 1.0 | 0.216 | 1.06 | 1.43 |

Unclipped, the slab sits 150× above its own labels' force scale (rms 0.10
eV/Å) and stays there — it is not a transient. Clipping fixes the slab and
improves the other two as well. This example uses **5.0**.

Two ways to see the same problem if you hit it with other data:

- A branch whose error is stuck orders of magnitude above the others, while
  that branch trains fine on its own.
- Branch order mattering. With `bulk` first the slab's *first* reported
  `rmse_f` is 23–530 across three seeds; with `slab` first it is 1.8–4.5 —
  even though the two models are identical at initialization (verified: the
  same initial |F|max of 27 eV/Å either way). Order dependence before
  training has meaningfully begun is the signature of one branch overpowering
  the rest.

## `model_prob` is not a data-size weight

```jsonc
"model_prob": { "bulk": 0.4, "molecules": 0.4, "slab": 0.2 }
```

It is the probability that a given **step** is drawn from that branch. The
slab has 29 training frames against the bulk's 544, so 0.2 already revisits
each slab frame far more often than each bulk frame. Raise it and the slab
memorizes; lower it and it never learns.

## Train

```bash
dp --pt-expt train input.json
```

Measured: **0.07 s/batch** across the three branches.

## Using one head afterwards

A multi-head checkpoint holds three models. Selecting one is mandatory —
there is no default:

```python
from deepmd_lr.ase import load_model
model = load_model("model.ckpt.pt", head="slab")
```

Asking for none raises `Select a checkpoint head from ['bulk', 'molecules',
'slab']`. Each head carries its own energy zero and its own level of theory,
so **do not mix heads within one simulation** — the energies are not
comparable.

## Caveats

- **The three datasets are toy-sized** (604 / 670 / 34 frames) and their
  labels come from three different methods, one of which is an MLIP. The
  numbers above are evidence that the machinery works, not that the model does.
- Adding a branch changes the other branches. The backbone is shared, so a new
  branch's gradients land on weights the others depend on — expect to retune
  `gradient_max_norm` and `model_prob` rather than to bolt one on for free.
