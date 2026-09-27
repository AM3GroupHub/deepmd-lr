# Non-periodic systems

Worked example: [`examples/02-molecules`](../examples/02-molecules).

Mark a system with an empty `nopbc` file in its directory and DeePMD-kit's
loader hands the model `box = None`. A short-range head already works that way
— a `dp_lr` branch with no `long_range` block, or a plain `standard` model,
runs on `box=None` frames with no special handling.

To add long-range electrostatics to such a branch, set the method to
`isolated`:

```jsonc
"long_range": {
  "charge_net": { "neuron": [64, 64], "precision": "float32" },
  "reciprocal": { "method": "isolated", "alpha": 0.4 }
}
```

`alpha` must match whatever the periodic branches use. Nothing else applies —
meshes, k-vector cutoffs, `slab_correction` and `pbc` all describe a sum over
periodic images, and passing one is an error rather than a silent no-op.

**What that term computes is fixed, not a choice.** The periodic path splits
Coulomb at the screening parameter α:

```
Σ qᵢqⱼ/r  =  Σ qᵢqⱼ erfc(αr)/r     ← short-ranged; the network learns this
           + Σ qᵢqⱼ  erf(αr)/r     ← smooth and long-ranged
```

`LongRangeEnergy` returns the second sum, evaluated in reciprocal space. So for
an isolated system the consistent term is that same sum evaluated directly, at
the **same α**:

```
Eᵢ = ½ Σ_{j≠i} qᵢqⱼ erf(α rᵢⱼ) / rᵢⱼ
```

This is not an analogy. Put a cluster in a cube of side L and the periodic
`recip − self` converges onto the direct `erf` sum as 1/L³ — measured at
−6.6e-03, −8.3e-04 and −1.0e-04 eV for L = 80, 160 and 320 Å. It is the same
physical quantity, which is what lets one shared backbone serve both branches:
the remainder left to the network has the same `erfc(αr)/r` form either way.

Three consequences worth knowing:

- **No self term to subtract.** `erf(αr)/r → 2α/√π` as `r → 0`, and the Ewald
  self term is exactly that times `½Σqᵢ²`. Summing `j ≠ i` already excludes it.
- **Charged molecules are easier here than in a cell** — no neutralising
  background is needed, so an ion is just an ion.
- **No cutoff.** `erf(αr)/r` decays as `1/r`; truncating it would be wrong.

It is summed densely rather than through `nvalchemiops`. Both of that
library's candidates are unusable here: `dsf_coulomb` is a shifted-force
potential with a charge-dependent self term and no position autograd, and
`coulomb_energy` raises under `create_graph=True` — so neither survives
force-loss training, which differentiates the force a second time. The dense
form is exact, twice differentiable, and costs about 1 ms at 1000 atoms. Its
one limit is memory: the pair tensor is `O(N²)`, or 8 MiB at 1000 atoms in
float64. Fine for molecules, and the reason this is not the method for a
periodic cell even in principle.

Two other things behave differently without a cell, both deliberately:

- **The virial is still computed**, and is still meaningful — it is the pair
  sum `Σ rᵢⱼ ⊗ fᵢⱼ`, which needs no periodicity. DeePMD-kit's own energy model
  returns one for a `box=None` frame, so this term matches it rather than
  leaving the total inconsistent with the short-range part.
- **Stress is not reported.** Force per unit area needs a volume to divide by,
  and there is none. `LongRangeCalculator` simply omits it.

## Running MD

Through the ASE calculator, with any ASE integrator. The GPU MD classes in
`deepmd_lr.nvalchemi` are for 3D-periodic cells and refuse anything else.

```python
from ase import units
from ase.md.verlet import VelocityVerlet
from deepmd_lr.ase import LongRangeCalculator

atoms.calc = LongRangeCalculator("model.ckpt.pt")   # trained with method "isolated"
VelocityVerlet(atoms, 0.25 * units.fs).run(2000)
```

The term holds no weights, so a model trained on periodic cells can be asked
for an isolated system by swapping the term at load; `alpha` is inherited
([same-as](same-as.md)):

```python
LongRangeCalculator("bulk.ckpt.pt", long_range={"method": "isolated"})
```

That makes the electrostatics exact for the cluster. Whether the short-range
part transfers is a question of training data: a model trained on bulk has
never seen a surface. Slabs run the same way, with `slab_correction` and `pbc`
in `long_range` ([example 03](../examples/03-slab)).

