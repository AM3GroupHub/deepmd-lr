# Training on the LiFSI / EC-DMC interface

How this dataset was built and trained, and why each choice was forced. The
general guide is [`../../TRAINING.md`](../../TRAINING.md); this covers what is
specific to a **slab**.

## What the system is

`LiFSI_EC_DMC_1M_T300.xyz` — 1461+ frames from UMA, 6520 atoms each,
`pbc="T T F"`, cell 28.08 × 28.08 × 106.33 Å, with energies and forces.

It is not bulk electrolyte. Reading the composition and the z-profile:

| region | contents |
|---|---|
| z 0–30 Å | ~1026 Li, nearest-neighbour 2.71 Å → **lithium metal electrode** |
| z 30–105 Å | ~38 Li⁺, 40 FSI⁻ (N=40, S=80, F=80), EC/DMC solvent |

N/S/F fix the salt at exactly 40 FSI⁻, so charge balance accounts for only 40
of the 1064 Li. The rest is electrode. This is an **anode–electrolyte
interface**.

> **What the learned charges mean here.** In the electrolyte they are a
> sensible effective-charge picture. In *metallic* Li they are not: a metal
> screens fields and its charge is delocalised, so a point charge per Li atom
> is a fitting device rather than a physical observable. Treat the metal-side
> charges as latent variables — which is what they are in any case, since the
> electrostatic energy fixes them only up to a global sign (see
> `../RESULTS.md`). Report metal and electrolyte separately.

## Preparing the data

```bash
python prepare_dataset.py --frames 300 --skip 100 --vacuum 20 --out data
```

Three things happen, and two of them are not optional.

**Sampling.** 300 frames spread evenly across the trajectory (not a block —
consecutive MD frames are correlated), after discarding the first 100 as
equilibration. Split 240 train / 60 validation.

**Wrapping x and y.** The raw coordinates run outside the cell (x spans
−8.9 … 32.7 in a 28.08 box), so they are wrapped back into the periodic
directions. z is left exactly as it was: it is the open direction.

**Padding z with vacuum.** The slab is 104.34 Å thick in a 106.33 Å box — a
1.99 Å gap. Declaring that cell 3D-periodic would put **427 atoms at the top
face within the 6 Å descriptor cutoff of 256 atoms at the bottom face**,
neighbours that the UMA labels never saw because they were computed with
`pbc="T T F"`. The model would be asked to reproduce forces from an
interaction absent from the reference. The gap must exceed twice the
descriptor cutoff; 20 Å is comfortable. The slab is centred so both faces get
the same gap. Final cell: **28.08 × 28.08 × 124.38 Å**.

## The slab correction is required, not optional

Padding makes the cell 3D-periodic. It does **not** make the physics
3D-periodic: the slab still interacts with its images across the vacuum, and
for a dipolar interface that error is large and shrinks only slowly with
thickness. Measured on a dipolar bilayer, total long-range energy in eV:

| vacuum | 20 Å | 40 Å | 80 Å | 160 Å |
|---|---:|---:|---:|---:|
| uncorrected | 1238.3 | 1495.7 | 1624.4 | 1688.7 |
| corrected | 1753.037 | 1753.043 | 1753.043 | 1753.043 |

Adding vacuum is not a fix — the uncorrected sum has not converged at 160 Å.
So the `long_range` block sets:

```json
"pme": {
  "alpha": 0.4,
  "mesh_dimensions": [24, 24, 100],
  "spline_order": 6,
  "slab_correction": true,
  "pbc": [true, true, false]
}
```

`alpha = 0.4` pairs with `rcut = 6.0` (`erfc(2.4) = 7e-4` left to the
network). The mesh gives 1.17 / 1.17 / 1.24 Å spacing, all finer than
`1/(2·alpha) = 1.25 Å`. It is fixed rather than spacing-derived so the energy
stays smooth in the cell.

## Why the staged schedule, here specifically

Measured cost per training step at 6520 atoms (float32, energy + force +
backward):

| model | ms/step | peak memory |
|---|---:|---:|
| short-range only | 883 | 48 GB |
| joint | 4465 | 63 GB |

The long-range term itself is **not** the expense — profiled alone it is
73 ms. The 3.6 s gap is the second descriptor pass and its *double* backward,
which is what keeps the long-range force differentiable in the parameters.

So the short-range model trains ~5× faster, and the staged schedule spends the
bulk of the budget where it is cheap:

```bash
# 1. short-range warm-up (~0.9 s/step)
cd run_stage1 && dp --pt-expt train ../input_stage1.json --skip-neighbor-stat

# 2. graft into the joint architecture, charge head scaled down
python ../../../scripts/graft_short_range.py \
    --short-range run_stage1/model.ckpt.pt \
    --input input_joint.json --out run_joint/init_joint.ckpt.pt

# 3. joint fine-tune (~4.5 s/step)
cd run_joint && dp --pt-expt train ../input_joint.json \
    --init-model init_joint.ckpt --skip-neighbor-stat
```

`--skip-neighbor-stat` is required: the statistics pass is an O(N²) all-pairs
scan that tries to allocate **6 TB** at this system size. DPA4C is
graph-native and has no `sel`, so the statistics it would compute are unused.

Grafting matters more than usual here. At random initialisation the charge
head puts the long-range energy at **+9484 eV** against a total energy of
−34065 eV — the slab correction amplifies a random dipole across a 104 Å slab.
Starting from a scaled-down head keeps that term small and lets it grow.

## The comparison that makes it meaningful

`input_baseline.json` is the same model with the `long_range` block removed
and the *combined* step budget of stages 1 and 2. Train it and compare
held-out force RMSE. Without it there is no way to separate "the long-range
term helped" from "the run went well".
