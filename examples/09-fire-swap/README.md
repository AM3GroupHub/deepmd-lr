# 9 — FIRE-Swap: B-site order in PMN by relaxed-swap Monte Carlo

Metropolis Monte Carlo over Mg ↔ Nb swaps in Pb(Mg₁/₃Nb₂/₃)O₃. Every trial
swap is followed by a FIRE relaxation of the atoms and the cell, and the
relaxed energy decides. The protocol is that of Xu, Cai, Shi, Zhong & Xie,
Phys. Rev. B **114**, 024204 (2026), Appendix B. Here it runs on the GPU
classes of [example 06](../06-md_nvalchemi) with one evaluator for the whole
run.

```bash
python fireswap.py --model pmn-lr-joint.pt --repeat 2 --steps 10000 --compress --out run12
python fireswap.py --model pmn-lr-joint.pt --repeat 2 --steps 20000 --compress --out run12 --resume
python analyze_order.py run12/fireswap.extxyz run12/order.dat
```

## The model and the structure

`pmn-lr-joint.pt` is the PMN release model: DPA4C with rcut 7 Å plus the PME
term (α 0.4), trained on PBEsol energies and forces of 15-, 30- and 135-atom
cells. Its hold-out force RMSE is 0.044 eV/Å; the same model without the
long-range term gets 0.052. The checkpoint is not in the repository
(`*.pt` is ignored), so pass its path. It was trained on **energies and
forces only, no stress**. The cell relaxation below therefore follows a
stress that nothing in training constrained. `--relax-cell none` keeps the
cell fixed if that matters for your question.

`pmn1080.xyz` is the 6×6×6 starting cell, 1080 atoms: frame 124 of a CACE-LR
FIRE-Swap run, with no Mg–Mg nearest neighbours. `--repeat 2` tiles it to
the 12×12×12 cell (8640 atoms) and `--repeat 3` to the 18×18×18 cell (29,160
atoms). Those are the two production sizes. The PME mesh follows the cell,
so the supercells need no setting.

## The protocol

| | |
|---|---|
| move | a random Mg, swapped with a random Nb within `--swap-cutoff` 6.3 Å (any Nb if none) |
| relaxation | FIRE, dt 0.1, maxstep 0.2 Å, until fmax 0.02 eV/Å or `--max-relax` 200 steps |
| cell | relaxed with the atoms (`--relax-cell full`) to `--smax` 0.02 GPa |
| acceptance | Metropolis at `--temperature` 600 K on the relaxed energy |
| stage A | the input is relaxed fully first (up to 10,000 steps), since it was relaxed with a different model |

What the script does that the ASE driver did differently:

- **The swap exchanges positions, not species.** The evaluator's types are
  fixed for its life, and moving each atom to the other's site gives the same
  configuration (checked: the two energies agree to 5 µeV). The jump of
  about 4 Å triggers the evaluator's neighbour-list rebuild by itself.
- **The cell force is divided by 10** (`SwapFIRE.CELL_FACTOR`), and the
  stress is relaxed to 0.02 GPa rather than to FrechetCellFilter's
  fmax · N / V ≈ 0.24 GPa. The same 10 trials at 8640 atoms (6 at 29,160)
  were each relaxed to 0.02 GPa three ways, with at most 400 steps:

  | cell force | 1080 atoms | 8640 atoms | 29,160 atoms |
  |---|---:|---:|---:|
  | whole (the package's FIRE) | 89 | 151, one trial unconverged | 178, one unconverged |
  | ÷ 10 (this script) | 83 | 95 | 82 |
  | ÷ N (FrechetCellFilter) | none converged | none converged | none converged |

  The whole force suits a box far from its density. After one swap,
  though, it makes the cell the stiff mode, and its stiffness grows with
  the cell length. Divided by N, the cell becomes so soft that
  it barely moves. FrechetCellFilter's own looser tolerance hides that:
  every 29,160-atom trial passed it with the cell still 0.6 eV above the
  energy the tight tolerance reaches, and the trials' relaxed energies
  scattered by up to 0.03 eV (0.6 kT) about the tight values. With the
  factor 10, all three sizes converge in under 100 steps and agree with
  the whole-force energies.
- **A rejected trial rolls back** to the last accepted state. That includes
  the cell, which is pushed to the evaluator.
- **Frames carry the accepted state's energy.** The ASE driver could label
  a frame with a rejected trial's energy.
- **Step t draws from `default_rng([seed, t])`.** So `--resume` continues
  the same chain. A resumed run reproduced an uninterrupted one swap for
  swap and FIRE step for FIRE step, with energies within 1e-5 eV.

## Output (`--out`)

| file | contents |
|---|---|
| `fireswap.log` | CSV per step: `step, E[eV], dE[eV], accepted, fire_steps, converged, trial[s], wall[s]` |
| `fireswap.extxyz` | the accepted state at step 0 and every `--save-every` 100 steps |
| `best.extxyz` | the lowest-energy state so far |
| `checkpoint.extxyz`, `checkpoint.json` | every `--checkpoint-every` 10 steps; `--resume` trims the log and trajectory back to it |

`analyze_order.py` reads the trajectory and writes one row per frame. Each
row has the Mg–Mg, Nb–Nb and Nb–Mg nearest-neighbour counts, the Mg on β_II,
the order parameter O(s), the Nb clusters (Nb with at most one Mg
neighbour), and, for a supercell, how much of the 6-site replica period has
been broken.

## Speed

All of the following was measured on one GB10, starting from the same cells.
The CACE row uses the original ASE driver: ASE FIRE on a FrechetCellFilter,
with memory patch A on.

| 8640 atoms | per force call | FIRE steps / trial | per trial | peak memory |
|---|---:|---:|---:|---:|
| CACE-LR, ASE | 2.31 s | 65–69 | 161–168 s | 27 GB (45 unpatched) |
| dp_lr, this script | 0.34 s | 100 | 34 s | 6.1 GB |
| dp_lr, this script, `--compress` | **0.020 s** | 89 | **2.0 s** | **0.9 GB** |

Compressed, at the three sizes (50, 50 and 20 MC steps, every trial converged):

| atoms | ms / FIRE step | FIRE steps / trial | s / trial | peak memory |
|---:|---:|---:|---:|---:|
| 1080 | 12.5 | 93 | 1.2 | |
| 8640 | 23.0 | 89 | 2.0 | 0.9 GB |
| 29,160 | 63.5 | 97 | 6.2 | 2.5 GB |

For reference, the CACE production runs on H200 took 49 s per trial at
8640 atoms on one GPU. At 29,160 atoms they took about 60 s per MC step on
four GPUs, which needed 90 GB per GPU.

### Why the CACE runs are slow

One trial is about 80 FIRE steps, and each step is one force-and-stress
call on the whole cell, so the cost of a call sets everything. On the
8640-atom cell, one 2.3 s call broke down as follows:

- **The CACE representation's backward pass, 1.8 s (78 %).** These are
  unfused eager-mode gathers, scatters and elementwise kernels over 578k
  edges with 1296 features per node. The forward pass is another 0.41 s.
- **The Ewald sum, 0.20 s (9 %).** It is a direct k-sum over about 30,000
  k-vectors (dl 2 Å, σ 1 Å), which costs O(N·K), so O(N²) at fixed density.
  At 29,160 atoms it grows to 2.4 s and 94 GB on its own. Patch B's
  chunking is what made that size fit in memory.
- **Host data preparation, 0.06 s here.** That is the matscipy neighbour
  list rebuilt every step, plus building the input dict and the copy to
  the device. On the H200 nodes it was 0.22 s (31 %), because their x86
  host is slower than the GB10's Grace.

The trial-parallel driver hides none of this per trial. Its four GPUs each
relax a whole cell, so it buys at most about 3.8 MC steps per round.

On the dp_lr path the compressed DPA4C backbone takes the place of the
representation. PME is O(N log N) on a 40³ mesh at 8640 atoms and a 59³ mesh
at 29,160. The neighbour list is built and reused on the GPU, and FIRE never
leaves the device.

## Limits worth knowing

- **The convergence test runs every 10 FIRE steps**, so trial lengths come
  in multiples of 10 and about 5 steps per trial are spent past
  convergence.
- **Relaxed energies carry up to 0.02 eV of scatter.** That is how far
  apart the same trials came out when relaxed by different routes to the
  same fmax 0.02 eV/Å and 0.02 GPa. It is 0.4 kT at 600 K. Lowering
  `--fmax` shrinks it, at the cost of more steps.
- **One GPU, one chain.** A 29,160-atom cell runs at about 64 ms per FIRE
  step on one device. For more throughput, run independent chains with
  different `--seed` values.
