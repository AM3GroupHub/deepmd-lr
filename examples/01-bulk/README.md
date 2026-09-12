# 1 — Training on a 3D-periodic system

Bulk liquid water: 64 H₂O in a 12.429 Å cubic cell, RPBE-D3 energies and
forces. We go from one extxyz file to a running training, and explain every
number that depends on *this* system — those are the ones you must revisit for
your own.

Nothing here needs to be run to be useful, but everything here was run.

## The data

```
train-H2O_RPBE-D3.xyz
```

```
frames      : 604
composition : {'H': 128, 'O': 64}  (192 atoms)
cell        : [12.429 12.429 12.429]   pbc=[True True True]
density     : 0.1000 atoms/A^3
energy      : -940.83 .. -931.47 eV   (-4.8774 eV/atom)
forces      : |F|max 7.04  rms 0.836 eV/A
closest pair: 0.930 A
```

Two properties make this the simplest possible case, and both are worth
checking in your own data before anything else:

- **Every frame holds the same atoms in the same order.** That is what lets the
  whole trajectory be *one* DeePMD system. If your file mixes species or atom
  counts, it must be split first — see `examples/02-*` if present, or
  `scripts/xyz_to_deepmd.py`, which refuses such a file with a clear message
  rather than writing a silently wrong `type.raw`.
- **The cell is fixed and fully periodic.** No `nopbc`, no slab correction.

## Step 1 — convert to a DeePMD dataset

```bash
python ../../scripts/xyz_to_deepmd.py train-H2O_RPBE-D3.xyz -o data --valid-fraction 0.1
```

```
604 frames x 192 atoms
type_map ['H', 'O']
periodic  labels: energy, force
data/train  544 frames
data/valid   60 frames
```

Reading goes through `ase.io.read`, so the same command works for `.traj`,
OUTCAR, CP2K output and anything else ASE knows (`--format` overrides the
extension guess, `--index` subsamples, e.g. `--index ::5`).

## Step 2 — what it wrote, and why

```
data/train/
  type.raw          192 lines: 0 0 0 ... 1 1 1     <- LOCAL indices
  type_map.raw      H O                            <- what those indices mean
  set.000/
    coord.npy   (544, 576)    = (nframes, natoms*3)
    force.npy   (544, 576)
    energy.npy  (544,)
    box.npy     (544, 9)      <- present because the frames are periodic
```

Three things here trip people up:

**`type.raw` is local, not global.** Its integers index *this system's*
`type_map.raw`, not the model's `type_map`. DeePMD-kit remaps them at load
time. That is what lets a water system (`[H, O]`) and an electrolyte system
(`[C, F, H, Li, N, O, S]`) share one model without renumbering anything.

**`box.npy` present ⇔ periodic.** For an isolated molecule there is no
`box.npy` and instead an empty file named `nopbc`, whose presence makes the
loader hand the model `box=None`.

**DeePMD-kit sorts atoms by type internally** and permutes coordinates and
forces to match. Labels stay aligned, so this needs nothing from you — but do
not assume row *i* of your xyz is row *i* of what the model sees when
debugging.

## Step 3 — `input.json`, decision by decision

The full file is here; this section explains only the values that are tied to
this particular system.

### `descriptor.rcut = 6.0`

**Must stay below half the cell length.** Here 12.429 / 2 = 6.21 Å, so 6.0 fits
— but only just. With a smaller cell you must reduce `rcut`, or an atom sees
two periodic images of the same neighbour and the descriptor becomes
ill-defined.

### `long_range.reciprocal.alpha = 0.4`

`alpha` splits the Coulomb interaction in two:

```
  q_i q_j / r   =   q_i q_j erfc(alpha r)/r     <- short-ranged, the network learns it
                  + q_i q_j  erf(alpha r)/r     <- this term supplies it
```

So `alpha` must be large enough that the network's share has *died inside its
own cutoff*: `erfc(0.4 × 6.0) = 7e-4`. Too small and you ask the network to
represent something reaching past `rcut`; too large and the reciprocal sum
needs an impractically fine mesh.

**`alpha` must be identical everywhere it appears** — across branches of a
multi-head model, and between training and any later MD. It defines what the
network learned to do.

### `mesh_dimensions = [10, 10, 10]`

The mesh must resolve the screening Gaussian, so spacing must beat
`1/(2·alpha) = 1.25 Å`:

```
ceil(12.429 / 1.25) = 10
```

**Use `mesh_dimensions`, never `mesh_spacing`.** A spacing-derived mesh is
`ceil(L / spacing)` — a *step function* of the cell. Under NPT or any strain
derivative it jumps a whole grid point as a cell length crosses a multiple of
the spacing, which makes the energy discontinuous exactly there. It moved a
virial by 2% in testing.

### `slab_correction = false`, `pbc = [true, true, true]`

3D-periodic bulk. The Yeh–Berkowitz correction is for a slab with vacuum, and
these two settings are a *pair*: `slab_correction` without exactly one `false`
entry in `pbc` is refused, and a `false` entry without the correction is
refused too.

### `loss.start_pref_f = 1000 -> limit_pref_f = 1`

Forces dominate early: there are `3 × 192` of them per frame against a single
energy, and they carry the local information the descriptor has to learn. The
prefactor decays so the energy matters more as training converges.
`start_pref_v = 0` because these labels carry no virial.

### `training_data.batch_size = 4`

192 atoms is small, so several frames per step keeps the GPU busy. For a
6,000-atom system you would use 1.

## Step 4 — train

```bash
dp --pt-expt train input.json
```

Measured here: **0.16 s/batch**, so the configured 200,000 steps is about
**9 hours** on one GPU.

To warm-start from a pretrained DPA4C instead, see the repository README —
note that
`--finetune` refits the energy bias to your data at startup, which is a
separate decision from training itself.

## Step 5 — what to watch in `lcurve.out`

```
step  rmse_val  rmse_trn  rmse_e_val  rmse_e_trn  rmse_f_val  rmse_f_trn  lr
```

- **`rmse_f` is the number that matters.** This dataset has a force rms of
  **0.836 eV/Å**, so a model predicting zero everywhere scores ~0.836. Anything
  near that means nothing has been learned yet. A converged water model should
  reach a few tens of meV/Å.
- **`rmse_e` is per atom.** Divide by nothing; DeePMD-kit already has.
- **Watch the charges.** The long-range term is only as meaningful as the
  charges feeding it. If `mean|q|` collapses toward zero, the model has learned
  to switch the term off rather than use it, and you have paid for it without
  benefit. "What to watch" in the repository README covers how to check.

## Is the long-range term earning its place?

Train a second model with the `long_range` block deleted — everything else
identical — and compare `rmse_f`. That is the only honest test, and it is worth
doing before a long production run. Water is a genuinely hard case for this:
it is neutral, homogeneous and well screened, so the term has less to do than
in an electrolyte or at an interface.

## Files

| | |
|---|---|
| `train-H2O_RPBE-D3.xyz` | the source trajectory (not tracked — `*.xyz` is gitignored) |
| `input.json` | the training config, annotated |
| `data/` | generated by step 1 (not tracked — `*.npy` is gitignored) |

Both the source and the generated dataset stay out of git deliberately; step 1
regenerates `data/` from the xyz in seconds.
