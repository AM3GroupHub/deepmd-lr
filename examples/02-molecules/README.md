# 2 — Training on non-periodic molecules

An isolated molecular **anion**: CHO₄S₂⁻, 383 frames of 8 atoms, taken from
[OMol25](https://arxiv.org/abs/2505.08762). No cell, a net charge of −1, and a
spin multiplicity that varies between frames — which is exactly the combination
example 01 could not show.

Everything below was run.

## The data

```
CHO4S2_anion.xyz        383 frames, 8 atoms, 351 KB
```

```
composition : C1 H1 O4 S2      pbc = False
charge      : -1  (every frame)
spin        : 1 x348,  3 x35
energy      : -30916.15 .. -30910.31 eV   (spread 5.84)
forces      : |F|max 7.10,  rms 1.23 eV/A
```

Each header carries `charge=-1 spin=N pbc="F F F"`, which is where the
converter picks up the per-frame condition.

> The frames were extracted from OMol25's `val` shard and **sorted by atomic
> number**. OMol25 stores this composition in two different atom orders, and one
> `type.raw` has to describe every frame — `scripts/xyz_to_deepmd.py` refuses a
> trajectory whose atom order changes, rather than writing a silently wrong one.

## Step 1 — convert

```bash
python ../../scripts/xyz_to_deepmd.py CHO4S2_anion.xyz -o data --valid-fraction 0.1
```

```
383 frames x 8 atoms
type_map ['C', 'H', 'O', 'S']
non-periodic (nopbc)  labels: energy, force  +charge_spin
data/train  345 frames
data/valid   38 frames
```

Two things differ from example 01:

```
data/train/
  nopbc                       <- empty marker; NO box.npy
  set.000/charge_spin.npy     <- (nframes, 2) = [charge, multiplicity]
```

**`nopbc`** is an empty file whose *presence* is the whole mechanism: the loader
sees it and hands the model `box = None`. There is no `box.npy`.

**`charge_spin.npy`** holds `[-1, 1]` for 315 frames and `[-1, 3]` for 30.

## Step 2 — `input.json`

### `add_chg_spin_ebd: true` — and `default_chg_spin: [0, 1]`

```jsonc
"descriptor": {
  "add_chg_spin_ebd": true,
  "default_chg_spin": [0, 1]
}
```

The first is what makes the model **read `charge_spin.npy` at all**; without it
the file is ignored silently. The second supplies a neutral-singlet fallback so
frames or systems *without* one still run.

Two consequences worth knowing before a long run:

- `add_chg_spin_ebd` is part of DPA4C's **structural signature**. In a
  multi-head model it must be `true` for *every* branch or the shared
  descriptor refuses to link — and turning it on later invalidates existing
  checkpoints. Decide before you train.
- `default_chg_spin` is deliberately **not** in that signature, which is what
  lets neutral branches coexist with charged ones.

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

This is not an analogue of the periodic term — it is the **same quantity**. Put
a cluster in a box and grow it, and the periodic `recip − self` converges onto
this sum as 1/L³. So **`alpha` must match whatever your periodic systems use**
(0.4 here, as in examples 01 and 03); a shared descriptor cannot serve two
different splits.

Nothing else from the periodic block applies. Meshes, spline orders,
`slab_correction` and `pbc` all describe a sum over periodic images, and passing
one is an **error**, not a silent no-op. There is no cutoff either — `erf(αr)/r`
decays as `1/r`, so truncating it would be wrong. Every pair is summed, which is
O(N²) and costs 8 MiB at 1000 atoms.

### The charges sum to −1, not 0

The charge head's output is projected onto the frame's total charge, read from
column 0 of `charge_spin`. For this system they sum to −1.

### `start_pref_v: 0`

No cell, so no virial to fit.

## Step 3 — train

```bash
dp --pt-expt train input.json
```

Measured: **0.084 s/batch** at batch size 8.

## If your file holds several molecules

This example is one composition, so it is one system. A file mixing species
must be **split, one directory per composition** — batches are drawn from a
single system, which is why differing atom counts then cost nothing. The
converter refuses a mixed file rather than guessing:

```
frames are not a single system -- 8 atoms in frame 0, 15 in frame 3.
A DeePMD system is one fixed composition and atom count.
```

## Caveats

- **383 frames of one small anion trains nothing useful.** The example
  demonstrates the settings. A real non-periodic model needs many molecules
  across many compositions.
- OMol25 is 558 GB, so this shipped subset is a convenience, not a reproducible
  pipeline. For data with a citable provenance and a small download, see
  [rMD17](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)
  — ten molecules, ~20 MB each — though it is neutral throughout and so cannot
  exercise `charge_spin`.
