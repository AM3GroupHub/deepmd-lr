# Can this model run in LAMMPS? Can it run in parallel?

Short answer: **the short-range half is LAMMPS-ready today; the long-range
half is not, and cannot be without changing where the reciprocal sum is
computed.** Two independent blockers, tested below.

## What works

The model without a `long_range` block is an ordinary DPA4C model. It freezes
and loads through the same path `pair_style deepmd` uses:

```
$ dp --pt-expt freeze -o model.pth
Saved frozen model to model.pt2 (lower_kind=graph)

>>> DeepPot('model.pt2').eval(coord, box, atype)
E = -34095.589 eV, |F|rms = 0.5963 eV/A
```

So a short-range DPA4C model trained here drops straight into LAMMPS.

## Blocker 1 — the export path cannot see the cell

`dp freeze` traces `forward_lower_graph_exportable`, whose inputs are edge
vectors, atom types and a neighbour graph. The reciprocal sum needs absolute
positions and the cell, neither of which is in that signature.

Note this is *not* simply "LAMMPS passes no cell" — `pair_deepmd.cpp` does
build a `dbox` from `domain->h` and hand it to `DeepPot::compute`. The cell
reaches the C++ API. It is the **exported artifact** that has no way to
receive it: the box is consumed to build the neighbour graph, and the traced
graph consumes edges only.

Freezing is therefore refused rather than allowed to produce a wrong file:

```
NotImplementedError: Graph-lower export (`dp freeze`) is not supported for a
'dp_lr' model with a long_range block. ... the artifact would silently contain
only the short-range energy.
```

That refusal was added after measuring the alternative: the frozen artifact
came back at **-93.74 eV** where the eager model gives **-55.12**, having
dropped the entire long-range contribution with no warning.

## Blocker 2 — the reciprocal sum does not decompose across MPI ranks

This one is not an implementation gap; it is the shape of the physics.

LAMMPS domain-decomposes. Each rank holds `nlocal + nghost` atoms — a spatial
subdomain plus a halo — and deepmd evaluates the model once per rank on that
subdomain. A reciprocal-space sum is a structure factor
`S(k) = sum_i q_i exp(i k . r_i)` over **every atom in the cell**, so it is not
a sum of independent spatial pieces.

Measured on the LiFSI frame (6520 atoms), splitting along z exactly as a 1-D
decomposition would, and summing the per-subdomain long-range energies:

| ranks | subdomain sum | error vs whole system |
|---|---:|---:|
| 1 (whole) | 2357.40 eV | — |
| 2 | 2311.17 eV | 2.0% |
| 4 | 1668.99 eV | 29.2% |
| 8 | 1497.89 eV | 36.5% |

The error grows with rank count. **No halo width fixes this** — deepmd's
multi-rank machinery (`comm_dict`, `deepmd_export::border_op`) exchanges ghost
*descriptor features*, which is exactly the right tool for a short-range
message-passing descriptor and the wrong shape for a global FFT.

## The path that does work: the DPLR architecture

LAMMPS already has a parallel long-range solver. The answer is not to compute
k-space inside the model but to let LAMMPS do it:

1. the model predicts **charges** (a local, per-atom quantity that decomposes
   across ranks perfectly well);
2. LAMMPS's own `kspace_style` computes the reciprocal sum with its parallel
   PPPM;
3. the resulting forces are mapped back onto atoms.

This is precisely how DeePMD-kit's existing DPLR works — `source/lmp/
pppm_dplr.cpp` and `fix_dplr.cpp` — and it is why DPLR is usable in parallel
MD while an in-model PME is not.

Adapting this model to that architecture means:

- exporting a **charge-predicting** model rather than an energy one (the
  charge head is already a separate fitting net, so the piece exists);
- a LAMMPS fix that reads those charges, hands them to `kspace_style`, and
  adds the returned forces;
- matching `alpha` between training and `kspace_modify gewald`, since with a
  reciprocal-only term `alpha` defines the model rather than being a free
  numerical knob.

That is a substantial piece of C++ work, not a configuration change.

## Meanwhile

For MD with the long-range term today, use `deepmd_lr.ase.LongRangeCalculator`
(see `TRAINING.md` §9), which drives the eager model through its full forward
and therefore has both the cell and every atom. It is single-process: fine for
system sizes up to a few thousand atoms, not a replacement for parallel LAMMPS.
