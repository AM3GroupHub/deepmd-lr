# Which neighbour list runs

Two neighbour graphs are built per step, chosen independently.

**The descriptor's**, which DeePMD-kit owns, is selected by
`neighbor_graph_method`. It resolves once at *training* startup — default
`auto`, which is `nv` (the nvalchemiops GPU cell list) on CUDA — but it is
**not recorded in the checkpoint**, so a model rebuilt for inference falls
back to the `dense` O(N²) builder. `deepmd_lr.ase.load_model` re-resolves it,
so anything loaded through `load_model` or `LongRangeCalculator` gets `nv`
again; a model you rebuild by hand does not. That gap is the 48.2 GB / 724 ms
versus 4.0 GB / 201 ms difference described under
[Running MD over partitions](partitioned-md.md#limits).

**The charge head's**, which this package owns, comes from
`build_neighbor_graph_fast`. With `method="auto"` it uses the cell list when
all three of these hold: the frame is on CUDA, nvalchemiops is available, and
the atom count clears the threshold for its boundary condition.

| frame | threshold | so in practice |
|---|---:|---|
| with a cell (`box` given) | `CELL_LIST_MIN_ATOMS_PBC = 0` | **always** — any size |
| no cell (`box=None`) | `CELL_LIST_MIN_ATOMS_FREE = 1024` | 1024 atoms and up |

Partitions are the second case: the halo already carries the images, so the
partitioned driver builds its graph with `box=None`.

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
builder be used on the charge graph at all.

