# Which neighbour list runs

Three neighbour graphs exist, chosen independently: the descriptor's and the
charge head's for the serial model (training, the ASE calculator), and the MD
engine's own, which lives on the device.

**The descriptor's**, which DeePMD-kit owns, is selected by
`neighbor_graph_method`. It resolves once at *training* startup — default
`auto`, which is `nv` (the nvalchemiops GPU cell list) on CUDA — but it is
**not recorded in the checkpoint**, so a model rebuilt for inference falls
back to the `dense` O(N²) builder. `deepmd_lr.ase.load_model` re-resolves it,
so anything loaded through `load_model` or `LongRangeCalculator` gets `nv`
again; a model you rebuild by hand does not. On the 6520-atom LiFSI frame
that gap is **48.2 GB and 724 ms** for `dense` against **4.0 GB and 201 ms**
for `nv`, at a bit-identical energy -- and 48 GB does not fit on a 48 GB card,
which is how it first showed up, as an out-of-memory. `load_model` uses
DeePMD-kit's own `resolve_neighbor_graph_method("auto", device)`, so CPU or a
missing nvalchemiops degrades to `dense` instead of raising.

**The charge head's**, which this package owns, comes from
`build_neighbor_graph_fast`. With `method="auto"` it uses the cell list when
all three of these hold: the frame is on CUDA, nvalchemiops is available, and
the atom count clears the threshold for its boundary condition.

| frame | threshold | so in practice |
|---|---:|---|
| with a cell (`box` given) | `CELL_LIST_MIN_ATOMS_PBC = 0` | **always** — any size |
| no cell (`box=None`) | `CELL_LIST_MIN_ATOMS_FREE = 1024` | 1024 atoms and up |

The partitioned driver used to be the second case (a halo with explicit
images, searched with `box=None`); it now has its own list, below.

The thresholds come from measurement, median of 7 after warm-up on one GB10
(ms):

```
atoms            192   384   768  1536  2592   6144
periodic  dense 1.06  2.35  7.60 29.37 83.13 470.12
          nv    1.01  1.06  1.22  4.02  3.84   4.26
box=None  dense 0.30  0.29  0.42  1.19  3.16  16.97
          nv    0.62  0.69  0.83  1.15  0.69   3.95
```

With a cell the cell list wins at every size measured, down to 192 atoms, so
there is no crossover to respect. Without one, the dense path has no ghost
images to materialise and stays cheaper until about a thousand atoms.

Pass `method="cell_list"` or `method="dense"` to force either; the graph is
identical either way, and `tests/test_neighbors.py` asserts it. On CPU, or
without nvalchemiops, `auto` always yields `dense` rather than raising.

Computing the virial does **not** force the dense builder. The strain is
applied to the finished edge vectors rather than to the coordinates and cell,
so the derivative never flows through the box — which is what lets the O(N)
builder be used on the charge graph at all. That matters because only `dense`
carries a cell derivative: measured `d(edge_vec)/d(box)` is 1932.8 for
`dense` and **6.7e-13** for `nv`, while `ase` and `vesin` raise. `nv`
returning round-off rather than an error is the dangerous part -- forces stay
correct and a virial that leaned on it would go silently wrong. Straining the
edge vectors is exact (an edge vector is `coord[j] + S·box − coord[i]`, every
term of which transforms by the same `(I+ε)ᵀ`) and cost **8.3 GB and 0.92 s
per step**, against 63 GB and 3.0 s when the derivative routed through the box.

Only the *search* is taken from nvalchemiops: its edge vectors come out of a
Warp kernel with no gradient, so they are discarded and DeePMD-kit's
`neighbor_graph_from_ijs` recomputes `edge_vec` from `coord`/`box`. Using the
kernel's vectors directly would produce a model with no forces.

**The MD engine's**, `GpuNeighborList` in `deepmd_lr/nvalchemi/nlist.py`,
which `DistributedEvaluator` uses for every node set. It calls nvalchemiops'
cell list directly, every step at `rcut`, into preallocated buffers, and
compacts the neighbour matrix into DeePMD-kit's graph with two Warp kernels
(the destination CSR by a prefix sum, the source CSR by an atomic cursor as
LAMMPS-Kokkos does), with no sort and no host synchronisation after the
first, sizing search. The kernel's `r = x[j] − x[i] + S·cell` is DeePMD-kit's
`edge_index = [j, i]`, `edge_vec = r_src − r_dst`, so the integer shifts pass
through; unwrapped positions are folded into `S` by the kernel. On one device
the list searches the real cell with the real periodicity (no halo, nodes =
atoms); on a partition it searches the owned atoms plus their explicit ghost
images non-periodically, with only the owned atoms as centres. Here the
kernel's float32 vectors *are* used when nothing differentiates through them
(a compressed short-range model: DeePMD-kit assembles forces from the edge
gradients through the CSR views); the charge response and the virial, which
do, get vectors recomputed from the node coordinates and the strain. The
mechanics, the measured costs and the one-download-per-step contract are in
[Running MD over partitions](partitioned-md.md#the-neighbour-list-lives-on-the-device).
Small boxes (several images of one pair, an atom seeing its own image),
slabs, triclinic cells and lattice jumps are checked against the dense oracle
in `tests/test_nlist.py`.

