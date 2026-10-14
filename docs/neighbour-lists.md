# Which neighbour list runs

Three neighbour graphs exist, chosen independently: the descriptor's and the
charge head's for the serial model (training, the ASE calculator), and the MD
engine's, which lives on the device.

**The descriptor's**, which DeePMD-kit owns, is selected by
`neighbor_graph_method`. It resolves once at training startup (default `auto`,
which is `nv`, the nvalchemiops GPU cell list, on CUDA) but is **not recorded
in the checkpoint**, so a model rebuilt for inference falls back to the `dense`
O(N²) builder. `deepmd_lr.ase.load_model` re-resolves it with DeePMD-kit's own
`resolve_neighbor_graph_method("auto", device)`, so anything loaded through
`load_model` or `LongRangeCalculator` gets `nv` again (CPU or a missing
nvalchemiops degrades to `dense` instead of raising); a model you rebuild by
hand does not. On the 6520-atom LiFSI frame that gap is **48.2 GB and 724 ms**
for `dense` against **4.0 GB and 201 ms** for `nv`, at a bit-identical energy,
and 48 GB does not fit a 48 GB card.

**The charge head's**, which this package owns, comes from
`build_neighbor_graph_fast`. With `method="auto"` it uses the cell list when
the frame is on CUDA, nvalchemiops is available, and the atom count clears the
threshold for its boundary condition:

| frame | threshold | so in practice |
|---|---:|---|
| with a cell (`box` given) | `CELL_LIST_MIN_ATOMS_PBC = 0` | always |
| no cell (`box=None`) | `CELL_LIST_MIN_ATOMS_FREE = 1024` | 1024 atoms and up |

Median of 7 after warm-up on one GB10, ms:

```
atoms            192   384   768  1536  2592   6144
periodic  dense 1.06  2.35  7.60 29.37 83.13 470.12
          nv    1.01  1.06  1.22  4.02  3.84   4.26
box=None  dense 0.30  0.29  0.42  1.19  3.16  16.97
          nv    0.62  0.69  0.83  1.15  0.69   3.95
```

With a cell the cell list wins at every size, down to 192 atoms. Without one
the dense path has no ghost images to materialise and stays cheaper until
about a thousand atoms. `method="cell_list"` or `"dense"` forces either; the
graph is identical, and `tests/test_neighbors.py` asserts it.

Only the search is taken from nvalchemiops: its edge vectors come out of a Warp
kernel with no gradient, so they are discarded and DeePMD-kit's
`neighbor_graph_from_ijs` recomputes `edge_vec` from `coord` and `box`. Using
the kernel's vectors directly would give a model with no forces.

**The virial never needs the dense builder.** The strain is applied to the
finished edge vectors, not to the coordinates and cell, so the derivative
never flows through the box. That matters because only `dense` carries a cell
derivative: `d(edge_vec)/d(box)` is 1932.8 for `dense` and **6.7e-13** for
`nv`, while `ase` and `vesin` raise, and `nv` returning round-off instead of an
error is what would have made a box-routed virial silently wrong. Straining
the edge vectors is exact (every term of `coord[j] + S·box − coord[i]`
transforms by the same `(I+ε)ᵀ`) and costs 8.3 GB and 0.92 s per step against
63 GB and 3.0 s through the box.

**The MD engine's**, `GpuNeighborList` in `deepmd_lr/nvalchemi/nlist.py`,
which `DistributedEvaluator` uses for every node set: nvalchemiops' cell list
every step at `rcut` into preallocated buffers, compacted into DeePMD-kit's
graph by two Warp kernels, with no sort and no host synchronisation after the
first, sizing search. On one device it searches the real cell with the real
periodicity (no halo, nodes = atoms); on a partition, the owned atoms plus
their explicit ghost images, non-periodically, with only the owned atoms as
centres. Here the kernel's float32 vectors are used when nothing
differentiates through them (a compressed short-range model); the charge
response and the virial get vectors recomputed from the node coordinates and
the strain. The same CSR rows let the Born tensor's edge half be summed per
node by one kernel ([Born charges](born-charges.md#implementation-notes)).
Mechanics and measured costs:
[the device list](partitioned-md.md#the-neighbour-list-lives-on-the-device).
Small boxes, slabs, triclinic cells and lattice jumps are checked against the
dense oracle in `tests/test_nlist.py`.
