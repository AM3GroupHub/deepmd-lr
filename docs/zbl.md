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
The `zbl` block therefore does two things across a window per pair of elements:

```json
"model": {
  "type": "dp_lr",
  "zbl": {
    "window": {"O-H": [0.70, 0.90], "H-H": [1.45, 2.00], "O-O": [1.50, 1.75]}
  }
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

## Choosing the windows

What the network has seen differs from pair to pair, so each pair has its own
window, and a pair that is not listed has no term:

- **`x_hi` at or below the pair's shortest contact in the data**, so that
  nothing fitted changes. `scan.py --data train.xyz` prints these in radius
  sums.
- **`x_lo` above where the network starts to fall.** Scan the pair with the
  trained model first; the fall is abrupt.
- Each window must end inside `descriptor.rcut`.

For the water model:

| pair | shortest in the data | the trained model falls below | window |
|---|---|---|---|
| O–H | 0.873 Å = 0.90 | 0.68 Å = 0.70 (the bond compressed) | `[0.70, 0.90]` |
| O–O | 2.350 Å = 1.78 | 1.95 Å = 1.48 (two molecules pushed together) | `[1.50, 1.75]` |
| H–H | 1.272 Å = 2.05 | not separable: an O–H contact closes first | `[1.45, 2.00]` |

A window has to sit against its own pair's data. One covalent window for every
pair, `[0.7, 0.9]`, covered the O–H bond and nothing else: paths that close an
H–H or an O–O contact kept holes of −190 and −51 eV between that window and
those pairs' floors.

## A trained model

The term holds no weights and adds no `state_dict` entry, so a trained model
can be given one:

```python
from deepmd_lr.ase import LongRangeCalculator, load_model
from deepmd_lr.nvalchemi import build_evaluator

model = load_model("model.ckpt.pt")
model.set_zbl({"window": {"O-H": [0.70, 0.90], "H-H": [1.45, 2.00], "O-O": [1.50, 1.75]}})
atoms.calc = LongRangeCalculator(model)            # ASE
ev = build_evaluator(model, atoms, compress=True)  # GPU MD
```

With every window below its pair's data this changes nothing fitted: on the
water test set the energy moved by 5e-6 eV, forces by 2e-6 eV/Å and charges by
4e-7 e, which is float32 noise. For the same reason training with the block
gives the same model on the data; `set_zbl(None)` removes the term.

## What it does, measured

[`scan.py`](../examples/10-zbl/scan.py) pushes one pair of each kind together
inside a validation frame: the O–H bond by moving the hydrogen along it, the
other two by moving the neighbour's whole molecule, rigid. The reference is a
MACE-LR model trained on the same data, evaluated on the same geometries:

| energy change | trained model | with `zbl` | MACE-LR |
|---|---|---|---|
| O–H at 0.62 Å | −124 eV | +20.2 eV | +7.1 eV |
| O–H at 0.29 Å | −1.5e6 eV | +112 eV | +50 eV |
| H–H at 0.97 Å | −149 eV | +25.7 eV | +9.0 eV |
| O–O at 1.56 Å | −130 eV | +26.2 eV | +10.8 eV |
| largest charge of a pair on the three paths | 1.3e4 e | 0.25 e | |

With the term the energy rises all the way in on each path and has the
reference's shape. Where a path passes a third atom its energy falls again
behind that atom, in the reference too, so the check is not a steady rise but
that the energy never drops back below where it stood at the pair's shortest
fitted contact.

On a 192-atom water box an NVE run conserves energy as before (0.032 meV/atom
rms without the term, 0.031 with it), since no pair enters a window.

## What it does not do

- **It is a guard for a pair contact in otherwise normal surroundings.** A
  badly packed box still collapses: FIRE from a 64-water box with O–O contacts
  at 1.56 Å slides to −13,900 eV in 600 steps (the trained model reaches −1.2e9
  in 25). Once two oxygens are switched off, their hydrogens are still
  described by the network, which has never seen such surroundings. Pre-relax
  with a classical force field ([MD](md.md#relaxing-a-packed-box-first)).
- **Inside a window the wall is stiffer than the reference.** Standard ZBL is
  2–4 times MACE-LR there (O–H at 0.71 Å: 13.4 eV against 3.2 eV). The term
  keeps a run alive through a collision; it does not make the energy of that
  collision right.
- **A window above a pair's data changes the model on the data.** Bond floors
  differ between materials (0.75 radius sums for Nb–O in PMN, 0.77 for C–C in
  SiC), which is why there is no default window.

## Refused

Each of these would silently leave the term out, or meet it half way:

| | why |
|---|---|
| `DP_CUDA_INFER=2` | its fused energy/force kernel never calls the atomic model the envelope acts on; use 1 |
| `forward_lower`, the canonical compressed export | the term is evaluated on the graph lower; the graph export of a short-range model (`dp freeze`) keeps it |
| a Hessian | DeePMD-kit's Hessian wrapper evaluates the atomic model directly |
| `use_srtab`, atom or pair exclusions, a descriptor other than DPA4C | not supported with the term |
| `charge_same_as` branches with different `zbl` blocks | one charge head has one envelope |

An MD checkpoint records the block, and restores only onto an evaluator whose
model has the same one.

## Cost

In inference on a CUDA device one Warp kernel pass over the edges leaves the
per-node ZBL energy and envelope and a short list of the edges in contact,
which in a healthy simulation is empty. Forces, virials and the envelope's
derivatives are summed from that list, so nothing the size of the edge list is
allocated or passed over a second time, and nothing is read back to the host:
an MD step makes the same number of downloads as without the term. The list
holds one contact per atom and 4096 more; a graph with more than that has
collapsed, and its energy is NaN, which stops a run. Training, a CPU and any
traced path use the same arithmetic as torch ops over every edge.

Measured on the GB10 with the compressed water model, per evaluation:

| atoms | without | with `zbl` |
|---|---|---|
| 192 | 13.0 ms | 14.2 ms |
| 12,288 | 15.3 ms | 16.3 ms |
| 98,304 | 62.3 ms | 64.5 ms |
| 98,304, with Born tensors | 71.6 ms | 75.2 ms |

That is about a millisecond per evaluation, and another per 100,000 atoms.
