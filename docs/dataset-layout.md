# Dataset layout

`scripts/xyz_to_deepmd.py` handles the common cases; this section is the
reference for what it produces and why.

**One system = one composition and one fixed atom ordering.** `type.raw` and
`type_map.raw` live at the *system root*, not inside `set.000`, so every
`set.*` in a system shares them — the sets are frame batches, not different
compositions:

```
data/water/type.raw          <- here, shared by all sets
data/water/type_map.raw      <- here
data/water/set.000/coord.npy
data/water/set.001/coord.npy
```

### The indices in `type.raw` are LOCAL to that system

`type.raw` indexes that system's *own* `type_map.raw`, and DeePMD-kit remaps
into the model's global `type_map` at load time
(`deepmd/utils/data.py`: `old_to_new_type_idx[self.atom_type]`). So each
system's `type_map.raw` may be shorter, and in a different order, than the
model's — it only has to contain the elements that system actually uses.

With model `type_map = ["Cl", "H", "Na", "O"]`:

| system | `type_map.raw` | `type.raw` | remapped | elements |
|---|---|---|---|---|
| water | `["H", "O"]` | `[1, 0, 0]` | `[3, 1, 1]` | O H H |
| nacl | `["Cl", "Na"]` | `[0, 1, 0, 1]` | `[0, 2, 0, 2]` | Cl Na Cl Na |

- **Always write `type_map.raw`.** Without it there is nothing to remap
  against, so the indices are taken as *global* — and this fails **silently**.
  The water system above, minus its `type_map.raw`, loads as H-Cl-Cl instead
  of O-H-H with no error at all.
- An element in a system's `type_map.raw` that the model's `type_map` lacks
  fails loudly: `ValueError: Elements ['Xe'] are not present in the provided
  type_map`.

**The remap above is machinery you can now mostly ignore.**
`scripts/xyz_to_deepmd.py` writes the full periodic table into `type_map.raw`
by default, arranged so that **a type index is the atomic number** — hydrogen 1,
carbon 6, oxygen 8 — in every dataset it produces, which makes the remap the
identity. Index 0 is ASE's dummy `"X"` and is never assigned to an atom; it is
there so the indices are the numbers everyone already uses rather than one less.
That is what makes a dataset reusable with a model it was not converted for, and
it is why pretrained DPA models carry the full table.

The trade is the last bullet above, now unmissable rather than rare: the
model's `type_map` must be the full table too, or every element the data
declares and the model lacks is reported at load. Elements with no atoms in
the data are harmless — a row each in the type embedding and the energy bias,
never a gradient. Pass `--type-map present` to get the old per-dataset map.

### Several compositions in one training set

Different compositions go in different systems. `dpdata` will group them:

```python
import dpdata
ms = dpdata.MultiSystems()
for frame in whatever_you_have:      # any dpdata LabeledSystem
    ms.append(frame)
ms.to_deepmd_npy("data")             # or ms.to_deepmd_hdf5("data.hdf5")
```

**Point `systems` at the parent, not at each child.** `expand_sys_str`
recurses and takes every directory containing a `type.raw`
(`deepmd/common.py:204`), so nesting depth is free:

```json
"training_data": { "systems": ["./data"], "batch_size": "auto:8" }
```

Two knobs matter once the systems differ in size:

- **`"batch_size": "auto:N"`** targets ~N atoms per batch rather than N
  frames. A fixed `batch_size` makes the large systems dominate each step.
- **`auto_prob`** (default `prob_sys_size`) sets how often each system is
  drawn, proportional to its batch count. Use `"auto_prob": "prob_uniform"`
  to weight every system equally instead, or `sys_probs` to set them by hand.

### Composition that varies frame to frame

If the *atom count* is fixed but which element sits at each site changes per
frame, that is one system in **`mixed_type`** format: add
`real_atom_types.npy` of shape `(nframes, natoms)` to each set, holding local
indices, with `-1` for a virtual (padding) atom. Its presence is what selects
the mode, it must be present in *every* set of the system, and
`type_map.raw` becomes mandatory. `type.raw` is still required — it fixes
`natoms`.

### Directories or HDF5

DeePMD-kit reads either. The mapping is one-for-one — **a directory becomes an
HDF5 Group, a file becomes a Dataset** (`deepmd/utils/path.py`, `DPH5Path`) —
so the tree inside the file is identical to the directory layout. A system is
addressed with `#` rather than a path separator, and nothing else changes:

```json
"training_data": { "systems": ["./data.hdf5#/train"], "batch_size": 1 }
```

`type_map.raw` is the one thing to get wrong: it is read back with
`load_txt(dtype=str)`. Fixed-length bytes (`S8`) and both variable-length
string dtypes decode correctly through that; pickled Python objects do not.

**Compression is not worth enabling.** Measured on 75.1 MB of float64
coordinates, boxes, energies and forces:

| filter | size | ratio | write | read all |
|---|---:|---:|---:|---:|
| none | 75.2 MB | 1.00× | 0.04 s | 0.01 s |
| lzf | 74.8 MB | 1.00× | 0.47 s | 0.09 s |
| gzip | 70.0 MB | 1.07× | 1.65 s | 0.24 s |

MD coordinates are high-entropy, so gzip costs a 24× slower read to save 7%.
If a set is genuinely too large, storing coordinates as float32 halves it — a
real decision about precision, not a free win like compression appeared to be.

Choose HDF5 for the **file count**, not the size: one file instead of a dozen
per system matters on a cluster filesystem with inode quotas, and makes a
dataset atomic to `rsync`. Throughput is comparable — measured 0.79 s/batch
from HDF5 against 0.72 from directories, on 6520-atom frames.

