# Does it apply to your data?

Two prerequisites, then one measurement worth doing before spending GPU time.

**Periodic data uses Ewald; non-periodic data uses the direct sum.** Both are
supported — see [Non-periodic systems](non-periodic.md#non-periodic-systems) — but they are
different code paths and the choice is not automatic.

**You need energies and forces. You do not need charges.** The charge head is
trained through the electrostatic energy, never against charge labels.

**Measure the headroom.** The long-range term can only recover the part of the
signal that lives beyond the descriptor cutoff. If that is negligible, nothing
downstream will help. Compare exact Ewald forces against Coulomb truncated at
your cutoff: on the benchmark system **17.7% of the force signal lies beyond
6 Å**, which corresponds to a 0.37 eV/Å floor no local model can beat — and
the baseline duly plateaued at 0.41. If your equivalent number is ~1%, expect
a correspondingly small return.

## Choosing `alpha` and the mesh

`alpha` is **not** a free parameter. This term supplies only the reciprocal
half of the Ewald split, so `alpha` decides the division of labour:

```
full Coulomb  =  erfc(alpha·r)/r     +     the reciprocal sum
                 ^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^^^^^^
                 the network's job         this term's job
```

### Constraint 1 — `alpha` large enough that the network's share fits inside its cutoff

`erfc(alpha·r_cut)` — what is still unresolved at the descriptor cutoff:

| alpha | rc = 4 Å | rc = 5 Å | rc = 6 Å | rc = 8 Å |
|------:|---------:|---------:|---------:|---------:|
| 0.2   |  2.6e-01 |  1.6e-01 |  9.0e-02 |  2.4e-02 |
| 0.3   |  9.0e-02 |  3.4e-02 |  1.1e-02 |  6.9e-04 |
| 0.4   |  2.4e-02 |  4.7e-03 | **6.9e-04** | 6.0e-06 |
| 0.5   |  4.7e-03 | **4.1e-04** | 2.2e-05 | 1.5e-08 |
| 0.7   | **7.5e-05** | 7.4e-07 | 2.9e-09 | 2.4e-15 |

Aim for ≲ 1e-3. Too small an `alpha` hands the network a tail it structurally
cannot represent — the exact failure the long-range term exists to fix.

### Constraint 2 — mesh fine enough to resolve the screening Gaussian

Relative error in the reciprocal energy vs mesh spacing `h`:

| alpha | 1/(2·alpha) | h = 1.5 | h = 1.25 | h = 1.0 | h = 0.75 |
|------:|------------:|--------:|---------:|--------:|---------:|
| 0.3   |        1.67 | 1.9e-06 |  3.4e-06 | 5.6e-07 |  1.9e-08 |
| 0.4   |        1.25 | 5.3e-05 |  2.8e-05 | 3.3e-06 |  5.8e-08 |
| 0.5   |        1.00 | 1.1e-04 |  1.6e-04 | 1.8e-05 |  8.3e-07 |
| 0.7   |        0.71 | 5.9e-03 |  5.5e-04 | 2.9e-04 |  2.2e-05 |

**`h ≈ 1/(2·alpha)`** lands around 1e-4; go one step finer for ~1e-6.

### Recommended pairings

| descriptor `rcut` | `alpha` | mesh spacing | for a 20 Å cell |
|------------------:|--------:|-------------:|----------------:|
| 6 Å               |     0.4 |     ≤ 1.25 Å | `[16, 16, 16]`  |
| 5 Å               |     0.5 |     ≤ 1.00 Å | `[20, 20, 20]`  |
| 4 Å               |     0.7 |     ≤ 0.75 Å | `[28, 28, 28]`  |

Larger `alpha` costs mesh but leaves the network less to do. **Whatever you
pick must match between training and any later MD** — `alpha` defines the
model, not just the numerics.

**Prefer `mesh_dimensions` over `mesh_spacing`.** A spacing-derived mesh is
`ceil(L / spacing)`, a step function of the cell: a cell length crossing a
multiple of the spacing gains a whole grid point and the energy jumps. It is
harmless at fixed cell (NVE/NVT) and a real problem under NPT or any strain
derivative — it showed up here as a 2% virial error that looked like a bug in
the strain derivative.

**At inference the mesh follows the cell.** A fixed mesh is right for the cell
it was chosen for and for no other. The water model's `[10, 10, 10]` is exact
on its 12.43 Å training box, but on the 49.7 Å 4×4×4 supercell of that same
box it left 0.49 meV/atom and 0.0077 eV/Å rms of error. So both inference
paths give each axis at least `ceil(2 alpha L)` points, the `1/(2 alpha)`
spacing of the table above, and never fewer than the model asks for; that
supercell gets `[40, 40, 40]` and is exact. The GPU evaluator sets the mesh
once, when it is built, so NPT keeps the smooth energy a fixed mesh buys. The
ASE calculator sets it per frame. Training is unchanged.

## `pme` or `ewald`

Both compute the *same* reciprocal sum, by different routes.

| | `pme` (default) | `ewald` |
|---|---|---|
| how | charges onto a mesh, then an FFT | explicit sum over k-vectors |
| scaling | `O(N log N)` | `O(N · K)`, and `K` grows with cell volume |
| error | mesh interpolation, ~1e-6 at a sane mesh | none — exact for its `k_cutoff` |
| knobs | `mesh_dimensions` / `mesh_spacing`, `spline_order` | `k_cutoff` (or `accuracy`) |

Measured (energy + backward, alpha 0.4, 1.25 Å mesh):

| atoms | cell | `pme` | `ewald` | k-vectors |
|------:|-----:|------:|--------:|----------:|
|    32 | 14 Å | 2.9 ms |  3.3 ms |     1,687 |
|   128 | 22 Å | 2.8 ms |  5.0 ms |     6,083 |
|   512 | 35 Å | 2.7 ms | 25.5 ms |    21,437 |
|  2048 | 56 Å | 2.9 ms | 113.6 ms |   83,187 |

**Use `pme` for training.** It is faster at every size tested — including the
smallest, so there is no small-system crossover to exploit — and the gap grows
to 40× by 2048 atoms.

**Use `ewald` to check a `pme` setting.** Because it has no interpolation
error, disagreement between the two is a statement about your mesh, not about
the physics:

```python
from deepmd_lr.lr import LongRangeEnergy
pme   = LongRangeEnergy(alpha=0.4, mesh_dimensions=[16, 16, 16])
ewald = LongRangeEnergy(alpha=0.4, method="ewald")
a, b = pme(coord, q, box).sum(), ewald(coord, q, box).sum()
print(abs(a - b) / abs(b))       # want <= 1e-4; if not, refine the mesh
```

Parameters belonging to the other method are **rejected**, not ignored — a
mesh setting silently dropped under `ewald` would be indistinguishable from
one that was applied.

