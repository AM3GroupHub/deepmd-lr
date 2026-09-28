# Several levels of theory: multi-head training

Worked example: [`examples/04-multi-head`](../examples/04-multi-head).

If your datasets differ not just in composition but in **what produced the
labels** — one at PBE, one at a hybrid functional, one at coupled cluster — a
single energy head cannot fit them. The references and scales differ, so
whatever the head learns is a compromise between incompatible targets. The
same applies when the datasets differ in *boundary condition*: a slab, a bulk
cell and an isolated molecule each need their own long-range settings.

Both cases are the same fix: **one shared descriptor, several fitting heads**.
DeePMD-kit calls this multi-task, and `dp_lr` works inside it unchanged.
Multi-task is triggered by the presence of `model_dict` — there is no CLI flag.

### What is shared and what is not

Only three keys can be shared: `type_map`, `descriptor`, `fitting_net`. Write
the value once in `shared_dict` and reference it by name from each branch.

Sharing works by **aliasing the module objects**, not by copying and
resynchronising: after setup the branches hold literally the same parameter
tensors, so one optimizer step updates the backbone for all of them.

`long_range.charge_net` is **not** shareable, so every branch gets its own
charge head. That is the right default when the branches are different levels
of theory — the charges genuinely differ — but it does mean the branches are
only tied through the backbone. When the branches are the *same* theory and
differ only in boundary condition, see [`same_as`](same-as.md#one-theory-two-boundaries-same_as)
below, which ties everything instead.

DPA4C supports `shared_level: 0` only (share everything). It also checks that
the structural parameters match *exactly* — `rcut`, `channels`, `lmax`,
`n_radial`, `radial_modes`, `basis_type`, `type_map`, precision and
`add_chg_spin_ebd`. That is why the descriptor is written once in
`shared_dict` rather than copy-pasted per branch: a single typo becomes
`ValueError: DPA4C parameter sharing requires identical structural parameters`.

### Set `gradient_max_norm`

**Not optional with heterogeneous branches.** Branches push the shared
backbone with wildly different gradient norms. Measured at initialization on
the three example datasets:

| branch | descriptor `stddev` min | amplification | grad norm on shared backbone |
|---|---:|---:|---:|
| bulk water | 3.6e-05 | ×28,000 | **5.0e+09** |
| molecules | 7.8e-03 | ×128 | 1.9e+06 |
| slab | 1.9e-03 | ×538 | 4.8e+05 |

DPA4C's `compute_input_stats` divides each descriptor coordinate by its spread
on that branch's own data — a fixed preconditioner set once at initialization.
Bulk liquid water is dense and isotropic, so ~20 of its 144 coordinates have a
spread below 1e-3, and dividing by 3.6e-05 amplifies those directions ×28,000
on the way back out. Standalone that is harmless, because the branch's head is
matched to its own preconditioner. **Shared it is not**: the calibration is
per-branch, the weights are not, and the bulk branch simply overwrites what the
others learn. Measured at 600 steps:

| `gradient_max_norm` | bulk `rmse_f` | molecules | slab |
|---|---:|---:|---:|
| *(unset)* | 0.403 | 1.43 | **15.0** |
| 5.0 | 0.221 | 1.24 | **1.00** |
| 1.0 | 0.216 | 1.06 | 1.43 |

Unclipped, the slab sits 150× above its own labels' force scale and stays
there. Two ways to recognise the problem: a branch stuck orders of magnitude
above the others while training fine on its own, and results that depend on
the order branches are listed in.

### Rules that will bite you

- **One `type_map` for all heads.** DeePMD-kit asserts it. With the
  converter's default that is simply the whole periodic table, so there is no
  union to compute and no element to forget; per-system `type.raw` indices
  stay local and are remapped through each system's own `type_map.raw`.
- **`pt_expt` does not inherit top-level defaults into branches.** Unlike the
  `pt` backend, nothing cascades — combined with strict validation, every
  option must be written out in every branch.
- **`alpha` must be the same in every branch.** It sets the split between what
  the network learns and what the term adds; one backbone cannot serve two
  different splits. `mesh_dimensions` *should* differ per branch, because the
  cells do.
- **`data_stat_nbatch` is a model key, per branch** — not a `training` key.
  Putting it in `training` fails normalization outright. The default of 10
  builds a dense neighbour matrix over 10 frames at once, which is tens of GiB
  at a few hundred atoms.
- **Set `start_pref_v: 0` for any branch without a cell.** There is no virial
  to fit.
- **Charged systems need `add_chg_spin_ebd: true`** on the descriptor, plus a
  `charge_spin.npy` of shape `(nframes, 2)` holding `[charge, multiplicity]`.
  Because that flag is part of the structural signature, turning it on for one
  branch means turning it on for **all** of them — decide before the first long
  run, since retrofitting it invalidates existing checkpoints.
- **Then set `default_chg_spin: [0, 1]` too.** Once the descriptor is
  charge-conditioned it demands a `charge_spin` for *every* frame, so the
  neutral branches would otherwise fail with `A charge-conditioned DPA4C
  requires a frame charge_spin`. The default supplies the neutral singlet
  fallback. Unlike `add_chg_spin_ebd`, this key is deliberately excluded from
  the structural signature, so it does not interfere with sharing.

### Running and reading it

One head is chosen per optimizer step by a weighted random draw over
`model_prob` (defaulting to weights proportional to each task's system count),
and a single optimizer steps all parameters. Every head is still evaluated at
each `disp_freq`, so `lcurve.out` gains per-head columns —
`rmse_e_trn_interface`, `rmse_f_val_bulk`, and so on.

```bash
dp --pt-expt train input.json
dp --pt-expt show  model.ckpt.pt model-branch     # list the heads
```

Long-range branches cannot be frozen; for MD, select the branch from the
training checkpoint through the eager ASE calculator:

```python
from deepmd_lr.ase import LongRangeCalculator
atoms.calc = LongRangeCalculator("model.ckpt.pt", head="interface")
```

`head` is required for multi-head checkpoints; branch aliases are accepted,
and the ASE periodic axes must match that branch's boundary configuration.
Each head carries its own energy zero, so do not mix heads within one
simulation.

Not available in multi-task mode: `zero_stage`, `validating.full_validation`,
`lora`, and `change-bias`.

**Is it working?** Compare each branch's validation RMSE against a single-task
model trained on that dataset alone. Sharing should be roughly
neutral-to-better per branch; if one branch is clearly worse, reweight with
`model_prob` or train it separately — `shared_level` cannot be relaxed for
DPA4C.

### One charge head under two theories: `charge_same_as`

When the branches are different levels of theory that should share one
electrostatics — the charges are a property of the geometry, and only the
short-range energy differs — put `charge_same_as: "<owner branch>"` on the
borrowing base branch. It keeps its own `fitting_net` and energy zero; its
`charge_fitting` becomes the owner's module (aliased, not copied), and the
`same_as` links of either branch follow it.

The pair must be one charge model, and each way it could not be is refused:

- **One descriptor**, since the charge head reads its features: point both
  branches' `descriptor` at one `shared_dict` entry. An inline copy with the
  same settings is a second model, trained apart. This is checked on the
  built modules, when the trainer shares the parameters.
- **An identical `charge_net` block** (compared with defaults filled in) and
  **the same `alpha`**.
- **The owner is a base branch**: not a `same_as` link, and not borrowing
  itself (no chains). `charge_same_as` on a link belongs on its base.

Everything but the descriptor is checked as the config is read, before any
model is built. The long-range term holds no weights, so "sharing the
long-range term" means only that both theories use the same `alpha`, method,
mesh rule and boundaries. Below, `ttt` and `ttf` name the periodic axes: `pbc`
true-true-true (bulk) and true-true-false (a slab).

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

Tests: `tests/test_charge_same_as.py` — object identity, the optimizer's
parameter count, gradient routing per theory, the state-dict round trip,
equal charges under both theories, links differing by the slab correction
only, and each refusal.
