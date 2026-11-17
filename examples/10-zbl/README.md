# 10 — Short contacts: `zbl`

A model fitted to condensed-phase data falls apart when two atoms are pushed
closer than anything in that data. The `zbl` block hands such a pair over to
the ZBL nuclear repulsion and switches the network off for the atoms in
contact ([short contacts](../../docs/zbl.md)). Nothing has to be assigned: the
covalent radii place a window below every bond.

```json
"zbl": {}
```

It holds no weights, so a trained model can be given one:

```python
model = load_model("model.ckpt.pt")
model.set_zbl({})
```

It stays outside the network, so a short-range model keeps it through
`dp freeze`, DeePMD-kit's compression and into LAMMPS
([how](../../docs/zbl.md#deepmd-kits-own-routes-level-2-dp-freeze-lammps)).

[`scan.py`](scan.py) checks it on your model, and shows which pairs need a
window of their own. It prints the shortest contact of every pair in the
training data, in covalent-radius sums, then pushes one pair of each kind
together inside a frame, with the trained model and with the term:

```bash
python scan.py model.ckpt.pt frame.xyz --data train.xyz --png scan.png
```

With `--data` it also makes the check training makes: a window that reaches
into the data, which would switch the network down on geometries it was fitted
to, is reported with a window that fits (`--head` picks the branch of a
multi-task checkpoint):

```
WARNING: a zbl window reaches into the training data: ... Give these pairs a
window below their data: "pairs": {"C-C": [0.5, 0.75], "Si-Si": [0.57, 0.82]}
```

On water the default covers the O–H bond; two oxygens and two hydrogens never
come anywhere near a bond length in the data (1.78 and 2.05 radius sums), and
the model falls right below those, so they get a window there:

```bash
python scan.py model.ckpt.pt frame.xyz --data train.xyz \
    --pairs H-H:1.45:2.0 O-O:1.5:1.75 --png scan.png
```

```json
"zbl": {"pairs": {"H-H": [1.45, 2.00], "O-O": [1.50, 1.75]}}
```

A bonded pair is compressed by moving one atom along the bond; for any other
pair the neighbour's whole molecule moves, rigid. Each pair gets a table (the
energy change and its short- and long-range parts, the two charges, the pair's
ZBL energy, the closest other contact) and a verdict: below the pair's
shortest fitted contact the energy must never fall back, must end at no less
than half the pair's ZBL energy, and the long-range part and the charges must
stay bounded.

## A reference line

To see what a path should look like, evaluate the same geometries with
something you trust, DFT or another model, and hand them back:

```bash
python scan.py model.ckpt.pt frame.xyz --frames scan_frames.xyz
```

```python
frames = ase.io.read("scan_frames.xyz", ":")
for atoms in frames:
    atoms.calc = trusted
    atoms.get_potential_energy()
ase.io.write("scan_reference.xyz", frames)
```

```bash
python scan.py model.ckpt.pt frame.xyz \
    --reference scan_reference.xyz --reference-label "DFT" --png scan.png
```

## What to expect

On the water model of [example 01](../01-bulk), against a MACE-LR model trained
on the same data: the trained model falls by hundreds of eV below each pair's
data, the model with the term rises all the way in, with the reference's shape
and two to three times its height (standard ZBL is stiffer than the real
repulsion at these distances). The term guards a pair contact in otherwise
normal surroundings; a badly packed box still needs a classical pre-relaxation
([MD](../../docs/md.md#relaxing-a-packed-box-first)).
