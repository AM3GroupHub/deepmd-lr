# Several levels of theory: multi-head training

Worked example: [`examples/04-multi-head`](../examples/04-multi-head).

Datasets that differ in what produced the labels (PBE, a hybrid, coupled
cluster) cannot share one energy head: the references and scales differ. The
same holds for datasets that differ in boundary condition, since a slab, a bulk
cell and a molecule each need their own long-range settings. Both get the same
fix, **one shared descriptor, several fitting heads**, which DeePMD-kit calls
multi-task; `dp_lr` works inside it unchanged, and the presence of
`model_dict` is what switches it on.

### What is shared and what is not

Three keys can be shared, `type_map`, `descriptor` and `fitting_net`: write the
value once in `shared_dict` and reference it by name from each branch. Sharing
aliases the module objects, so the branches hold the same parameter tensors
and one optimizer step updates the backbone for all of them.

`long_range.charge_net` is not shareable, so every branch gets its own charge
head. That is right when the branches are different levels of theory, and it
means the branches are tied only through the backbone. Branches that are the
same theory and differ only in boundary condition use
[`same_as`](same-as.md#one-theory-two-boundaries-same_as), which ties
everything; different theories that should share one electrostatics use
`charge_same_as`, below.

DPA4C supports `shared_level: 0` only, and requires the structural parameters
to match exactly (`rcut`, `channels`, `lmax`, `n_radial`, `radial_modes`,
`basis_type`, `type_map`, precision, `add_chg_spin_ebd`). Write the descriptor
once in `shared_dict`; a copy with one typo is `ValueError: DPA4C parameter
sharing requires identical structural parameters`.

### Set `gradient_max_norm`

**Not optional with heterogeneous branches.** DPA4C's `compute_input_stats`
divides each descriptor coordinate by its spread on that branch's own data, a
fixed preconditioner set once at initialization. Bulk water is dense and
isotropic, so about 20 of its 144 coordinates (19–21 depending on the frames
sampled) have a spread below 1e-3, and dividing by 3.6e-05 amplifies those
directions ×28,000, on the way in and on the way back. Alone that is harmless,
because the head is matched to its own preconditioner (example 01 trained
alone, with a 2-element `type_map`, shows ×48,000, so the union of elements is
not the cause); shared, each branch keeps its own calibration but all write
into one set of weights, and the bulk branch overwrites what the others learn.
At initialization on the three datasets of
[example 04](../examples/04-multi-head):

| branch | descriptor `stddev` min | amplification | grad norm on shared backbone |
|---|---:|---:|---:|
| bulk water | 3.6e-05 | ×28,000 | **5.0e+09** |
| molecules | 7.8e-03 | ×128 | 1.9e+06 |
| slab | 1.9e-03 | ×538 | 4.8e+05 |

Four orders of magnitude: one bulk step moves the shared weights further than
ten thousand slab steps. At 600 steps, one seed (branch draws are random, so
expect the slab column to move by ~50% run to run, not by the factor of 15
that clipping buys):

| `gradient_max_norm` | bulk `rmse_f` | molecules | slab |
|---|---:|---:|---:|
| *(unset)* | 0.403 | 1.43 | **15.0** |
| 5.0 | 0.221 | 1.24 | **1.00** |
| 1.0 | 0.216 | 1.06 | 1.43 |

Unclipped, the slab sits 150× above its labels' force scale (rms 0.10 eV/Å)
and stays there; it is not a transient. Clipping fixes the slab and improves
the other two as well. Two ways to see the same problem with other data:

- A branch whose error is stuck orders of magnitude above the others, while
  that branch trains fine on its own.
- Branch order mattering. With `bulk` listed first the slab's *first* reported
  `rmse_f` is 23–530 across three seeds; with `slab` first it is 1.8–4.5,
  although the two models are identical at initialization (the same initial
  |F|max of 27 eV/Å either way). Order dependence before training has
  meaningfully begun is the signature of one branch overpowering the rest.

### Rules that will bite you

- **One `type_map` for all heads**; DeePMD-kit asserts it. With the converter's
  default that is the whole periodic table, so there is no union to compute;
  per-system `type.raw` indices stay local.
- **`pt_expt` does not inherit top-level defaults into branches.** Every
  option must be written out in every branch.
- **`alpha` must be the same in every branch**: one backbone cannot serve two
  splits. `mesh_dimensions` should differ, because the cells do.
- **`data_stat_nbatch` is a model key, per branch**, not a `training` key;
  in `training` it fails normalization with ``undefined key
  `data_stat_nbatch` is not allowed in strict mode``. Its default of 10
  builds a dense neighbour matrix over 10 frames at once, tens of GiB at a few
  hundred atoms, so large frames want 1.
- **`rcut` is a global constraint.** There is one descriptor, so its cutoff
  must suit the smallest cell of any branch.
- **`start_pref_v: 0` for any branch without a cell.**
- **Charged systems need `add_chg_spin_ebd: true`** on the descriptor plus a
  `charge_spin.npy` of shape `(nframes, 2)` holding `[charge, multiplicity]`.
  The flag is part of the structural signature, so it is on for all branches
  or none (writing it once in `shared_dict` makes that automatic); decide
  before the first long run, since retrofitting it invalidates checkpoints.
- **Then set `default_chg_spin: [0, 1]` too**, or the neutral branches fail
  with `A charge-conditioned DPA4C requires a frame charge_spin`. This key is
  excluded from the signature and does not interfere with sharing.

### Running and reading it

One head is chosen per optimizer step by a weighted draw over `model_prob`
(default: proportional to each task's system count) and a single optimizer
steps all parameters. Every head is evaluated at each `disp_freq`, so
`lcurve.out` gains per-head columns (`rmse_e_trn_interface`,
`rmse_f_val_bulk`, ...).

```bash
dp --pt-expt train input.json
dp --pt-expt show  model.ckpt.pt model-branch     # list the heads
```

Long-range branches cannot be frozen; for MD, select the branch from the
training checkpoint:

```python
from deepmd_lr.ase import LongRangeCalculator
atoms.calc = LongRangeCalculator("model.ckpt.pt", head="interface")
```

`head` is required for a multi-head checkpoint (none raises `Select a
checkpoint head from [...]`), and aliases are accepted. Any head runs any
boundary: the frame's `pbc` picks the sum ([non-periodic](non-periodic.md)).
Each head carries its own energy zero and level of theory, so do not mix heads
within one simulation. Not available in multi-task mode: `zero_stage`,
`validating.full_validation`, `lora`, `change-bias`.

**Is it working?** Compare each branch's validation RMSE against a single-task
model trained on that dataset alone; sharing should be neutral-to-better per
branch. If one branch is clearly worse, reweight with `model_prob` or train it
separately; `shared_level` cannot be relaxed for DPA4C.

### One charge head under two theories: `charge_same_as`

When different levels of theory should share one electrostatics (the charges
are a property of the geometry, only the short-range energy differs), put
`charge_same_as: "<owner branch>"` on the borrowing base branch. It keeps its
own `fitting_net` and energy zero; its `charge_fitting` becomes the owner's
module, aliased, and the `same_as` links of either branch follow it. The pair
must be one charge model, and each way it could not be is refused:

- **One descriptor**: both branches' `descriptor` point at one `shared_dict`
  entry. An inline copy with the same settings is a second model, trained
  apart. Checked on the built modules, when the trainer shares the parameters.
- **An identical `charge_net` block** (defaults filled in) and **the same
  `alpha`**.
- **The owner is a base branch**: not a `same_as` link, not borrowing itself.

Everything but the descriptor is checked as the config is read. "Sharing the
long-range term" means only that both theories use the same `alpha`, method,
mesh rule and boundaries, since the term holds no weights. With `ttt` and
`ttf` naming the periodic axes (bulk and slab):

```json
"ttt_dft":  {"type": "dp_lr", "descriptor": "backbone", "fitting_net": {...},
             "long_range": {"charge_net": {...}, "reciprocal": {...3D...}}},
"ttf_dft":  {"same_as": "ttt_dft",
             "long_range": {"reciprocal": {...slab...}}},
"ttt_mace": {"type": "dp_lr", "descriptor": "backbone", "fitting_net": {...},
             "charge_same_as": "ttt_dft",
             "long_range": {"charge_net": {...identical...},
                            "reciprocal": {...3D...}}},
"ttf_mace": {"same_as": "ttt_mace",
             "long_range": {"reciprocal": {...slab...}}}
```
