# Short contacts: `zbl`

Worked example: [`examples/10-zbl`](../examples/10-zbl).

A network fitted to condensed-phase data has seen no pair closer than the
shortest contact in that data, and below it extrapolates freely. On the water
model of [example 01](../examples/01-bulk), with one O–H pair pushed together
inside a 192-atom frame:

| O–H (Å) | fitted energy | long-range energy | charges of the pair (e) |
|---|---|---|---|
| 0.95 (the frame) | 0 | 0 | −0.23, +0.12 |
| 0.70 | +2.8 eV | −0.2 eV | −0.32, +0.13 |
| 0.60 | −224 eV | −3.3 eV | +0.98, −0.55 |
| 0.52 | −1052 eV | −341 eV | +0.19, −14.5 |

The whole fall sits on the two atoms in contact (every other atom together:
+0.16 eV). A pair repulsion added on top does not help: ZBL is 21 eV at 0.60 Å.
The `zbl` block therefore does two things across a window, and needs nothing
assigned:

```json
"model": {
  "type": "dp_lr",
  "zbl": {}
}
```

A window is `[x_lo, x_hi]` in units of the pair's covalent-radius sum,
`x = r / (R_i + R_j)` with ASE's radii, and `s(x)` is a C² step: exactly 0 below
`x_lo`, exactly 1 above `x_hi`.

- **ZBL in.** The pair contributes `V_ZBL(r) (1 − s)`: the full screened nuclear
  repulsion below `x_lo`, none above `x_hi`.
- **The network out.** Every atom has an envelope `w_i = Π_j s(r_ij)`, the
  product over its neighbours. Its fitted energy is multiplied by it (the
  per-type energy offsets are not), and its charge is weighted by it in the
  projection onto the total charge, `q_i = w_i (raw_i + shift)`.

Below `x_lo` a pair is its energy offsets plus ZBL and carries no charge. With
every neighbour beyond its window an atom's envelope is exactly 1, and the model
is the one that was trained.

## The windows

| key | default | |
|---|---|---|
| `window` | `[0.6, 0.85]` | every pair's window, in covalent-radius sums; `null` for none |
| `pairs` | `{}` | a window of their own for single pairs, `{"O-O": [1.5, 1.75]}`, same units |

**The default is placed by the covalent radii**, just below a bond: 0.58–0.82 Å
for O–H. It was chosen on the water model, whose O–H bonds stop at 0.90 radius
sums and whose fitted energy falls below 0.70:

| every pair's window | the O–H bond pushed to 0.57 Å |
|---|---|
| `[0.5, 0.8]` (mace-lr's, for the charges of a MACE model) | −75 eV: closes too late |
| `[0.6, 0.85]` | +24.7 eV, rising all the way in |
| `[0.7, 0.9]` | +24.7 eV, but opens at the shortest bond in the data |

**A pair needs a window of its own when its data stop far from a bond
length.** Two oxygens of liquid water never come closer than 1.78 radius sums
and two hydrogens than 2.05, and the network falls right below each of those,
far above a window placed for a bond. With the default alone the O–H bond is
covered and paths that close an H–H or an O–O contact keep holes of −115 and
−19 eV; with

```json
"zbl": {"pairs": {"H-H": [1.45, 2.00], "O-O": [1.50, 1.75]}}
```

all three hold. `scan.py --data train.xyz` prints every pair's shortest contact
in radius sums; a window's `x_hi` belongs at or below it, and its `x_lo` above
where the trained model starts to fall (1.95 Å for two water molecules pushed
together along O–O).

**Training checks the windows against its data.** A window that covers fitted
geometries would have the network switched off on them, so when the statistics
are taken the sampled frames are checked, and a window that reaches into them
is refused with the pair and a window that fits:

```
a zbl window reaches into the training data: the network would be switched down
to 0.94 of its output on a fitted geometry. Shortest contacts inside a window,
in covalent-radius sums: H-O 0.90 (window 0.70-0.95). Give these pairs a window
below their data: "pairs": {"H-O": [0.64, 0.89]}
```

The radii are those of single bonds, and bonds can be shorter: a triple bond
(C≡C is 0.79 radius sums, N≡N 0.77), a metal–oxygen bond in an off-centred
octahedron, a contact of a melt. The shortest Nb–O of a PMN set sits at 0.77
radius sums, the shortest C–C and Si–Si of a set with amorphous carbon and
silicon at 0.75 and 0.82, all inside the default
([other models](#other-models)). This is where they show up. Each window must
also end inside `descriptor.rcut`; with a type_map that lists the whole
periodic table the default reaches 4.4 Å between its largest atoms.

## A trained model

The term holds no weights and adds no `state_dict` entry, so a trained model
can be given one:

```python
from deepmd_lr.ase import LongRangeCalculator, load_model
from deepmd_lr.nvalchemi import build_evaluator

model = load_model("model.ckpt.pt")
model.set_zbl({})                                  # the default for every pair
model.set_zbl({"pairs": {"H-H": [1.45, 2.00], "O-O": [1.50, 1.75]}})  # water
atoms.calc = LongRangeCalculator(model)            # ASE
ev = build_evaluator(model, atoms, compress=True)  # GPU MD
```

With every window below its pair's data this changes nothing fitted: on the
water test set the energy moved by 5e-6 eV, forces by 2e-6 eV/Å and charges by
4e-7 e, which is float32 noise. For the same reason training with the block
gives the same model on the data; `set_zbl(None)` removes the term.

`set_zbl` has no data to check the windows against. The check training makes
is `model.check_zbl_against(sampled)`, on samples in DeePMD-kit's statistics
format, and `scan.py --data train.xyz` runs it on a structure file: run it
before trusting the default on a model trained without the block.

## DeePMD-kit's own routes: level 2, `dp freeze`, LAMMPS

The term stays outside the learned model. DeePMD-kit evaluates the network by
whichever of its routes is selected, unchanged, and takes the envelope as a
weight per atom at the one place each route has for one; ZBL and the
envelope's own force and virial are added to its result from the list of
contacts.

| DeePMD-kit's route | when | where the envelope goes in |
|---|---|---|
| the autograd lower | training; inference at `DP_CUDA_INFER` 0 or 1 | a factor on the fitting net's output |
| the fused energy/force operators | a compressed model at `DP_CUDA_INFER=2` | the per-node ownership they seed their backward with |
| the compact canonical lower | a compressed `.pt2` export | the same ownership |

So a short-range model with the block freezes and compresses like any other,
and LAMMPS runs the file with `pair_style deepmd`, Kokkos and several ranks
included. (A model with `long_range` exports nothing, with or without the
block: see [MD](md.md).)

```bash
dp --pt-expt freeze -c model.ckpt.pt -o model.pt2        # the graph lower
dp --pt-expt compress -i model.pt2 -o model_c.pt2        # the compact canonical lower
```

Checked on the short-range water model with one O–H pair pushed to 0.70 Å and
another to 0.50 Å, where the term is worth 378 eV and the largest force is
132 eV/Å, against the same model evaluated in process at level 1:

| largest difference in | energy | force |
|---|---|---|
| compressed, `DP_CUDA_INFER=2` | 3e-6 eV | 4e-5 eV/Å |
| `model.pt2` read by `DeepPot` | 3e-6 eV | 6e-5 eV/Å |
| `model_c.pt2` read by `DeepPot` | 3e-6 eV | 6e-5 eV/Å |
| `model_c.pt2` in LAMMPS-Kokkos, 1,536 atoms, 1, 2 and 4 ranks | 3e-8 eV/atom | 8e-5 eV/Å |

That is float32 on a 132 eV/Å force; the same frame without contacts agrees
to 1e-6 eV/Å, and an NVE run of 12,288 atoms is the same trajectory with and
without the block, since no pair enters a window.

## What it does, measured

[`scan.py`](../examples/10-zbl/scan.py) pushes one pair of each kind together
inside a validation frame: the O–H bond by moving the hydrogen along it, the
other two by moving the neighbour's whole molecule, rigid. The reference is a
MACE-LR model trained on the same data, evaluated on the same geometries; the
block is the default with the two pairs above:

| energy change | trained model | with `zbl` | MACE-LR |
|---|---|---|---|
| O–H at 0.62 Å | −124 eV | +15.9 eV | +7.1 eV |
| O–H at 0.29 Å | −1.5e6 eV | +112 eV | +50 eV |
| H–H at 0.97 Å | −149 eV | +22.8 eV | +9.0 eV |
| O–O at 1.56 Å | −130 eV | +24.9 eV | +10.8 eV |
| largest charge of a pair on the three paths | 1.3e4 e | 0.27 e | |

With the term the energy rises all the way in on each path and has the
reference's shape. Where a path passes a third atom its energy falls again
behind that atom, in the reference too, so the check is not a steady rise but
that the energy never drops back below where it stood at the pair's shortest
fitted contact.

On a 192-atom water box an NVE run conserves energy as before (0.032 meV/atom
rms without the term, 0.031 with it), since no pair enters a window.

### Other models

The same scan over every pair of elements of two more long-range models, first
with the bare default, then with the windows the check proposed:

| | water | PMN (Pb, Mg, Nb, O) | silicon/diamond |
|---|---|---|---|
| pairs scanned | 3 | 10 | 3 |
| trained model, lowest energy on a path | −7.4e8 eV | −3.1e10 eV | −2.1 eV |
| trained model, largest charge of a pair | 1.3e4 e | 1,450 e | 0.75 e |
| data inside the default window | none | Nb–O, from 0.77 | C–C from 0.75, Si–Si from 0.82 |
| what the default does to fitted frames | nothing | 16–41 meV/atom, up to 16 eV/Å, in all 30 frames | up to 118 meV/atom, 30 eV/Å, in 890 of 4,552 frames |
| `pairs` the check proposes | | `"Nb-O": [0.52, 0.77]` | `"C-C": [0.5, 0.75]`, `"Si-Si": [0.57, 0.82]` |
| fitted frames with those | | unchanged | unchanged |
| paths that hold | 1 of 3; 3 of 3 with the two pairs above | 10 of 10 | 3 of 3 |
| largest charge of a pair with the term | 0.27 e | 0.67 e | 0.09 e |

The short-range model fitted to the same water data, which has no charges to
lose, falls later and less far (to −4.8e4 eV) and holds on all three paths
with the same block.

Three things to take from it:

- **The default is a starting point that the data confirm or move.** Two of
  three data sets hold bonds shorter than 0.85 radius sums; the check names
  them, and with its windows the fitted frames are back to float32 noise.
  (PMN was checked against 30 DFPT structures rather than its training set.)
- **How badly a model fails depends on its data.** The silicon/diamond set
  holds amorphous carbon and silicon from melt-quench runs, and its model
  never collapses: its walls only fall back, by 7, 6 and 54 eV on the three
  paths, where with the term they rise all the way in. The water and PMN sets
  hold nothing of the kind, and their models fall by orders of magnitude.
- **A path shows whatever is met first.** In a perovskite an oxygen sits
  between two neighbouring B-site cations, so the Nb–Nb path is an Nb–O
  collision (64,000 eV of ZBL at 0.05 Å, where the trained model stands at
  −3.1e10 eV). And only water has the default fail on a non-bonded pair: in
  PMN the network alone carries O–O from its data at 2.47 Å down to the window
  at 1.12 Å, rising to 158 eV on the way, which is luck rather than design.

## What it does not do

- **It is a guard for a pair contact in otherwise normal surroundings.** A
  badly packed box still collapses: FIRE from a 64-water box with O–O contacts
  at 1.56 Å slides to −14,100 eV in 600 steps with the two pairs above
  (−640,000 eV with the default alone; the trained model reaches −1.2e9 in
  25). Once two oxygens are switched off, their hydrogens are still described
  by the network, which has never seen such surroundings. Pre-relax with a
  classical force field ([MD](md.md#relaxing-a-packed-box-first)).
- **Inside a window the wall is stiffer than the reference.** Standard ZBL is
  2–3 times MACE-LR there (O–H at 0.62 Å: 15.9 eV against 7.1 eV). The term
  keeps a run alive through a collision; it does not make the energy of that
  collision right.
- **The default covers bonds, not every pair.** Pairs that never bond in the
  data need a window of their own, as above; the default alone leaves them to
  the network between their data and a bond length.

## Refused

Each of these would silently leave the term out, or meet it half way:

| | why |
|---|---|
| the dense (neighbour-list) lower, `forward_lower` | the term is evaluated on the graph lowers, which are what `dp freeze` and a compressed export trace for this model |
| a Hessian | DeePMD-kit's Hessian wrapper evaluates the atomic model directly |
| `use_srtab`, atom or pair exclusions, a descriptor other than DPA4C | not supported with the term |
| `charge_same_as` branches with different `zbl` blocks | one charge head has one envelope |

An MD checkpoint records the block, and restores only onto an evaluator whose
model has the same one.

## Cost

An evaluation finds the short contacts in one pass over the edges, and sums
forces, virials and the envelope's derivatives from the list it leaves, which
in a healthy simulation is empty. Nothing the size of the edge list is
computed twice.

- **In process, on a CUDA device**, the pass is one Warp kernel. Nothing is
  read back to the host: an MD step makes the same number of downloads as
  without the term. The list holds one contact per atom and 4096 more; a
  graph with more than that has collapsed, and its energy is NaN, which stops
  a run.
- **In an export, or on a CPU**, where no kernel of ours can run, the pass is
  torch arithmetic: one comparison per edge, which the compiler makes a
  single kernel of, and the terms on the edges that pass it. The list is as
  long as the contacts.
- **Training** uses torch ops over every edge, with the envelope
  differentiable to any order.

Measured on the GB10 with the compressed water models, milliseconds per
evaluation (per step in LAMMPS):

| | atoms | without | with `zbl` |
|---|---|---|---|
| long-range model, level 1 | 192 | 10.8 | 11.6 |
| | 12,288 | 13.8 | 14.6 |
| | 98,304 | 62.3 | 64.3 |
| | 98,304, with Born tensors | 71.5 | 74.9 |
| short-range model, level 1 | 12,288 | 7.9 | 8.8 |
| | 98,304 | 34.1 | 35.8 |
| short-range model, level 2 | 12,288 | 6.6 | 7.3 |
| | 98,304 | 33.3 | 35.4 |
| LAMMPS-Kokkos, compressed `.pt2` | 12,288 | 2.93 | 3.37 |
| | 98,304 | 26.0 | 27.7 |

In process that is about a millisecond per evaluation and another per 100,000
atoms. In LAMMPS it is a quarter of a millisecond of small kernels and one
pass over the edge vectors, 1.4 ms at 98,304 atoms on this machine's memory.
