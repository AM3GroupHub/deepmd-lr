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
this example does not work without it. The concepts are in
[multi-head](../../docs/multi-head.md); this page covers the choices this
`input.json` makes.

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

Multi-task is triggered by the presence of `model_dict`. The rules for the
shared parts are in [multi-head](../../docs/multi-head.md#rules-that-will-bite-you);
below is how this file meets them.

### Sharing is aliasing, not copying

```jsonc
"shared_dict": {
  "type_map_all": ["X", "H", "He", ..., "Og"],   // the full periodic table
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

The fitting nets here are deliberately *not* shared: three levels of theory
need three heads. `type_map_all` is the full periodic table every dataset was
converted against, so no dataset is renumbered.

### Same `alpha` in every branch

`alpha: 0.4` in all three, with three different boundary treatments around it:

```jsonc
"bulk":      { "alpha": 0.4, "mesh_dimensions": [10,10,10], "slab_correction": false, "pbc": [true,true,true] }
"molecules": { "alpha": 0.4, "method": "isolated" }
"slab":      { "alpha": 0.4, "mesh_dimensions": [12,12,42], "slab_correction": true,  "pbc": [true,true,false] }
```

The shared backbone learns the `erfc(αr)/r` remainder of one split, so the
branches cannot split Coulomb at different alphas.

### The other shared settings

- `add_chg_spin_ebd: true` sits once in `shared_dict`, so all three branches
  carry it, as the molecules need; `default_chg_spin: [0, 1]` covers `bulk`
  and `slab`, whose data has no `charge_spin.npy`.
- `rcut = 6.0` stays below half the smallest cell, bulk water's 12.429 Å.
- `data_stat_nbatch: 1` sits in each branch of `model_dict`; the default of
  10 needs tens of GiB on the 468-atom slab.

## The part that actually matters: gradient imbalance

**Without `gradient_max_norm`, this model does not train.** At
initialization the bulk branch puts a gradient norm of 5e+09 on the shared
backbone against the slab's 5e+05, and unclipped the slab's `rmse_f` stays at
15 eV/Å, 150× its labels' force scale. This example uses **5.0**. The cause,
the measurements and how to recognise it with other data are in
[set `gradient_max_norm`](../../docs/multi-head.md#set-gradient_max_norm).

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

A multi-head checkpoint holds three models, and selecting one is mandatory:

```python
from deepmd_lr.ase import load_model
model = load_model("model.ckpt.pt", head="slab")
```

Do not mix heads within one simulation: each has its own energy zero and level
of theory ([multi-head](../../docs/multi-head.md#running-and-reading-it)).

## Caveats

- **The three datasets are toy-sized** (604 / 670 / 34 frames) and their
  labels come from three different methods, one of which is an MLIP. The
  numbers above are evidence that the machinery works, not that the model does.
- Adding a branch changes the other branches. The backbone is shared, so a new
  branch's gradients land on weights the others depend on — expect to retune
  `gradient_max_norm` and `model_prob` rather than to bolt one on for free.
