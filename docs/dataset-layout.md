# Dataset layout

`scripts/xyz_to_deepmd.py` handles the common cases; this page is the
reference for what it produces and why.

**One system = one composition and one fixed atom ordering.** `type.raw` and
`type_map.raw` live at the system root, shared by every `set.*`, which are
frame batches, not compositions:

```
data/water/type.raw          <- here, shared by all sets
data/water/type_map.raw      <- here
data/water/set.000/coord.npy
data/water/set.001/coord.npy
```

### The indices in `type.raw` are local to that system

`type.raw` indexes that system's own `type_map.raw`, and DeePMD-kit remaps
into the model's global `type_map` at load (`deepmd/utils/data.py`,
`old_to_new_type_idx[self.atom_type]`), so a system's `type_map.raw` may be
shorter than the model's and in another order. With model
`type_map = ["Cl", "H", "Na", "O"]`:

| system | `type_map.raw` | `type.raw` | remapped | elements |
|---|---|---|---|---|
| water | `["H", "O"]` | `[1, 0, 0]` | `[3, 1, 1]` | O H H |
| nacl | `["Cl", "Na"]` | `[0, 1, 0, 1]` | `[0, 2, 0, 2]` | Cl Na Cl Na |

- **Always write `type_map.raw`.** Without it the indices are taken as global,
  silently: the water system above, minus its `type_map.raw`, loads as H-Cl-Cl
  with no error.
- An element the model's `type_map` lacks fails loudly: `ValueError: Elements
  ['Xe'] are not present in the provided type_map`.

The converter makes the remap the identity: by default it writes the full
periodic table into `type_map.raw`, arranged so that **a type index is the
atomic number** (hydrogen 1, carbon 6, oxygen 8; index 0 is ASE's dummy `"X"`
and is never assigned). That is what makes a dataset reusable with a model it
was not converted for, and why pretrained DPA models carry the full table. The
model's `type_map` must then be the full table too, or every element the data
declares and the model lacks is reported at load. Elements with no atoms in
the data cost a row each in the type embedding and the energy bias and never
a gradient. `--type-map present` gives the old per-dataset map.

### Several compositions in one training set

Different compositions go in different systems; `dpdata` groups them:

```python
import dpdata
ms = dpdata.MultiSystems()
for frame in whatever_you_have:      # any dpdata LabeledSystem
    ms.append(frame)
ms.to_deepmd_npy("data")             # or ms.to_deepmd_hdf5("data.hdf5")
```

**Point `systems` at the parent, not at each child.** `expand_sys_str`
recurses and takes every directory containing a `type.raw`
(`deepmd/common.py:204`):

```json
"training_data": { "systems": ["./data"], "batch_size": "auto:8" }
```

Once the systems differ in size, `"batch_size": "auto:N"` targets ~N atoms per
batch rather than N frames (a fixed `batch_size` lets the large systems
dominate), and `auto_prob` (default `prob_sys_size`, draws proportional to
batch count) can be set to `"prob_uniform"`, or `sys_probs` by hand.

### Composition that varies frame to frame

A fixed atom count with the element at each site changing per frame is one
system in `mixed_type` format: add `real_atom_types.npy`, shape
`(nframes, natoms)`, of local indices with `-1` for a padding atom, to every
set. Its presence selects the mode, `type_map.raw` becomes mandatory, and
`type.raw` is still required because it fixes `natoms`.

### Directories or HDF5

DeePMD-kit reads either, one-for-one: a directory becomes an HDF5 Group, a
file a Dataset (`deepmd/utils/path.py`, `DPH5Path`), and a system is addressed
with `#`:

```json
"training_data": { "systems": ["./data.hdf5#/train"], "batch_size": 1 }
```

`type_map.raw` is read back with `load_txt(dtype=str)`: fixed-length bytes
(`S8`) and both variable-length string dtypes decode, pickled Python objects
do not.

**Compression is not worth enabling.** On 75.1 MB of float64 coordinates,
boxes, energies and forces:

| filter | size | ratio | write | read all |
|---|---:|---:|---:|---:|
| none | 75.2 MB | 1.00× | 0.04 s | 0.01 s |
| lzf | 74.8 MB | 1.00× | 0.47 s | 0.09 s |
| gzip | 70.0 MB | 1.07× | 1.65 s | 0.24 s |

MD coordinates are high-entropy, so gzip costs a 24× slower read to save 7%;
float32 coordinates halve a set that is genuinely too large, at a real cost in
precision. Choose HDF5 for the file count, not the size: one file instead of a
dozen per system matters under inode quotas and makes a dataset atomic to
`rsync`. Throughput is comparable, 0.79 s/batch from HDF5 against 0.72 from
directories on 6520-atom frames.
