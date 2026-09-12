# 2 — Training on non-periodic molecules

Four different molecules, with four different net charges and a spin
multiplicity that varies frame to frame — 670 frames from
[OMol25](https://arxiv.org/abs/2505.08762). No cells anywhere.

This is the case example 01 could not show: **several compositions in one
dataset**, each carrying its own charge and spin.

Everything below was run.

## The data

```
molecules.xyz        670 frames, 1.2 MB
```

| composition | charge | atoms | frames | multiplicities |
|---|---:|---:|---:|---|
| `C2Cl1H5Si1` | 0 | 9 | 180 | 166 singlet, 14 triplet |
| `C1H1O4S2` | −1 | 8 | 180 | 161 singlet, 19 triplet |
| `C4H8N1O5P1S1` | −2 | 20 | 130 | 126 singlet, 4 triplet |
| `C11H20N1` | +1 | 32 | 180 | 164 singlet, 16 triplet |

Every header carries `charge=N spin=M pbc="F F F"`, which is where the
per-frame condition comes from.

> Frames were **sorted by atomic number** during extraction. OMol25 stores the
> same composition in more than one atom order, and one `type.raw` has to
> describe a whole system. The split keys on the *sorted* symbols, so those
> orders land in one bucket rather than two — but it does not reorder atoms
> for you, and an unsorted bucket is then refused with `differs at atom N`.

## Step 1 — split by composition, then convert

**A DeePMD system is one fixed composition and one fixed atom count.** Four
molecules therefore means four systems, and the file has to be split:

```bash
python ../../scripts/xyz_to_deepmd.py molecules.xyz -o data \
    --split-by-composition --valid-fraction 0.1
```

```
molecules.xyz -> data
  670 frames -> 4 system(s)
  C2Cl1H5Si1
  180 frames x 9 atoms
  type_map ['C', 'Cl', 'H', 'Si']
  non-periodic (nopbc)  labels: energy, force  +charge_spin
  data/C2Cl1H5Si1/train  162 frames
  data/C2Cl1H5Si1/valid  18 frames
  note: charge_spin is read only when the descriptor sets `add_chg_spin_ebd: true`
  ... three more, one per composition
```

Without `--split-by-composition` the converter refuses the file rather than
writing a `type.raw` that is wrong for most frames:

```
frames are not a single system -- 9 atoms in frame 0, 8 in frame 180.
```

Note that **charge and spin are not part of the split**. They vary per frame
and live in `charge_spin.npy`, so the same molecule in two charge states is
still one system.

## Step 2 — what was written

```
data/C1H1O4S2/train/
  nopbc                       <- empty marker; there is NO box.npy
  type.raw                    <- indices into THIS system's type_map.raw
  type_map.raw                <- C H O S
  set.000/charge_spin.npy     <- (nframes, 2) = [charge, multiplicity]
  set.000/{coord,energy,force}.npy
```

**Four systems, four different `type_map.raw`.** The model has *one* global
`type_map` — the union, `[C, Cl, H, N, O, P, S, Si]` — and each system's
`type.raw` indexes its own local list, which DeePMD-kit remaps at load time.
That indirection is what lets systems with different element subsets share one
model without renumbering anything.

**`nopbc`** is an empty file whose *presence* is the mechanism: the loader sees
it and hands the model `box = None`.

**`charge_spin.npy`** holds e.g. `{(-1, 1): 144, (-1, 3): 18}` for the anion.

## Step 3 — `input.json`

### One model, four systems

```jsonc
"training_data": {
  "systems": ["./data/C11H20N1/train", "./data/C1H1O4S2/train",
              "./data/C2Cl1H5Si1/train", "./data/C4H8N1O5P1S1/train"],
  "batch_size": 8,
  "auto_prob": "prob_sys_size"
}
```

DeePMD-kit draws each batch from **one** system, which is why 8-atom and
32-atom molecules cost nothing to mix — no padding is ever needed across
compositions. `auto_prob` weights systems by how many batches each holds;
`"prob_uniform"` would weight them equally, which matters when one molecule
dominates the frame count.

### `add_chg_spin_ebd: true` — and `default_chg_spin: [0, 1]`

```jsonc
"descriptor": { "add_chg_spin_ebd": true, "default_chg_spin": [0, 1] }
```

The first is what makes the model **read `charge_spin.npy` at all**; without it
the file is ignored silently and every molecule is treated as neutral. The
second supplies a fallback so systems *without* one still run.

Two consequences worth knowing before a long run:

- `add_chg_spin_ebd` is part of DPA4C's **structural signature**. In a
  multi-head model it must be `true` for *every* branch or the shared
  descriptor refuses to link — and turning it on later invalidates existing
  checkpoints.
- `default_chg_spin` is deliberately **not** in that signature, which is what
  lets neutral branches coexist with charged ones.

### The charges sum to the frame's total, not to zero

The charge head's raw output is projected onto the frame's total charge, taken
from column 0 of `charge_spin`. So predicted charges sum to 0, −1, −2 or +1
depending on which system the batch came from.

### `method: "isolated"`

```jsonc
"reciprocal": { "method": "isolated", "alpha": 0.4 }
```

Ewald splits Coulomb at `alpha` into a screened part the network learns and a
smooth part this term supplies. Under periodicity the smooth part is the
reciprocal sum; with no cell it is summed directly:

```
E = 1/2 sum_{j != i} q_i q_j erf(alpha r_ij) / r_ij
```

Not an analogue of the periodic term — the **same quantity**. Put a cluster in
a box and grow it, and the periodic `recip − self` converges onto this sum as
1/L³. So **`alpha` must match your periodic systems** (0.4, as in `01-bulk`
and `03-slab`): one descriptor cannot serve two different splits.

Nothing else from the periodic block applies. Meshes, spline orders,
`slab_correction` and `pbc` all describe a sum over periodic images, and
passing one is an **error**, not a silent no-op. There is no cutoff either —
`erf(αr)/r` decays as `1/r`, so truncating it would be wrong. Every pair is
summed: O(N²), and 8 MiB at 1000 atoms.

### `start_pref_v: 0`

No cell, so no virial to fit.

## Step 4 — train

```bash
dp --pt-expt train input.json
```

Measured: **0.063 s/batch** across the four systems at batch size 8, one GPU.

## Caveats

- **670 frames of four molecules trains nothing useful.** The example
  demonstrates the mechanics. A real molecular model needs thousands of
  compositions.
- OMol25 is 558 GB, so this shipped subset is a convenience rather than a
  reproducible pipeline. For data with a citable provenance and a small
  download, see
  [rMD17](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)
  — ten molecules, ~20 MB each — though it is neutral throughout and so cannot
  exercise `charge_spin`.
