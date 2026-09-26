# 3 — Training on a 2D-periodic slab

A Li-metal electrode under LiFSI/EC/DMC electrolyte: 468 atoms, 34 frames of
MD. This is the case the long-range term exists for — a **dipolar interface**,
where the electrostatics do not decay however much vacuum you add.

> **The labels here are UMA machine-learned predictions, not DFT**, and the
> energy includes a harmonic wall that acted above z = 30.285 Å in the original
> cell. This example teaches the settings a slab needs. Whatever error it
> reaches means nothing about accuracy, and the model it produces is not
> useful.

## The data

```
Li_LiFSI_interface.xyz     34 frames, 468 atoms, 1.6 MB
```

```
composition : C84 F4 H152 Li130 N2 O92 S4
cell        : 14.04 x 14.04 x 32.285     pbc = [True True False]
z occupancy : 0.00 .. 30.31  ->  only 1.98 A of headroom
energy      : -2229.69 .. -2185.63 eV
forces      : |F|max 0.4429 eV/A
```

The trajectory is already 2D-periodic — `pbc = [True, True, False]`, so the
reference calculation contained **no z-periodicity at all**.

## Step 1 — convert, and pad the vacuum

```bash
python ../../scripts/xyz_to_deepmd.py Li_LiFSI_interface.xyz -o data \
    --vacuum 52.0 --valid-fraction 0.15
```

```
34 frames x 468 atoms
type_map: the full periodic table (119 entries); present here: ['C', 'F', 'H', 'Li', 'N', 'O', 'S']
padded z to 52.0 A; slab spans 30.31 A, vacuum gap 21.69 A
periodic  labels: energy, force
data/train  29 frames
data/valid   5 frames
```

### Why `--vacuum` is not optional

The labels have no z-periodicity. The **model**, however, is handed a
3D-periodic cell — that is simply how the descriptor works. With the original
32.285 Å cell there are only **1.98 Å** of headroom, and `rcut = 6.0`. Atoms
near the top face would fall inside the cutoff of the next image's bottom face,
and the model would be asked to reproduce forces from an interaction that is
not in the reference at all.

The gap must exceed **2 × rcut = 12 Å**. Padding to 52 Å gives 21.7 Å, and the
script re-centres the slab so the vacuum is shared between both faces:

```
cell diag   : [14.04 14.04 52.0]
slab z span : 10.81 .. 41.19       (10.81 A below, 10.81 A above)
```

## Step 2 — `input.json`

### `slab_correction` and `pbc` are a pair

```jsonc
"reciprocal": {
  "alpha": 0.4,
  "mesh_dimensions": [12, 12, 42],
  "slab_correction": true,
  "pbc": [true, true, false]
}
```

`pbc` marks z as the open axis; `slab_correction` applies the Yeh–Berkowitz
correction, which cancels the interaction between the slab and its periodic
images across the vacuum. They are enforced together — `slab_correction`
without exactly one `false` entry is refused at construction, and a `false`
entry without the correction is refused too.

**Padding is not a substitute for the correction.** They fix different things:

| | what it stops |
|---|---|
| `--vacuum` | the **descriptor** seeing across the gap into the next image |
| `slab_correction` | the **electrostatics** doing so |

The second does not go away with more vacuum. A slab with a net dipole
perpendicular to its surface interacts with its images through a term that
decays barely at all — which is exactly what an electrode/electrolyte
interface has.

### The mesh follows the *padded* cell

`ceil(14.04 / 1.25) = 12` and `ceil(52 / 1.25) = 42`, giving `[12, 12, 42]`,
where 1.25 Å is the `1/(2·alpha)` spacing the screening Gaussian needs. Using
the original 32.285 would under-resolve the reciprocal sum along z.

### `alpha` matches the other examples

0.4, as in 01 and 02. It fixes where Coulomb is split, so a model trained on
several systems — or a multi-head model sharing one descriptor — must use the
same value everywhere.

## Step 3 — train

```bash
dp --pt-expt train input.json
```

Measured: **0.139 s/batch** at 468 atoms, batch size 1.

## Using it in MD afterwards

The `Atoms` object must carry `pbc=[True, True, False]` to match the
`long_range` block, or `LongRangeCalculator` refuses the frame:

```
Atoms pbc=[True, True, True] does not match the selected long-range
branch pbc=[True, True, False]
```

Pad the cell the same way you padded it for training.

## Caveats

- **34 frames trains nothing.** This is a miniature of a production set of
  300 frames of 6520 atoms with DFT-quality labels, which is not shipped here.
- The harmonic wall in the source energies is an artifact of how the
  trajectory was generated, not physics you want a model to learn. It is
  another reason to treat the numbers here as illustrative only.
- For a public slab dataset with citable provenance, see
  [OC20](https://opencatalystproject.org) — its smallest S2EF split is 200K
  structures (344 MB compressed), in LMDB rather than xyz.
