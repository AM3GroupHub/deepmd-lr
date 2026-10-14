# Non-periodic systems

Worked example: [`examples/02-molecules`](../examples/02-molecules).

An empty `nopbc` file in a system's directory makes DeePMD-kit's loader hand
the model `box = None`. A short-range head (a `dp_lr` branch with no
`long_range` block, or a plain `standard` model) already runs on such frames.
To add long-range electrostatics, set the method to `isolated`:

```jsonc
"long_range": {
  "charge_net": { "neuron": [64, 64], "precision": "float32" },
  "reciprocal": { "method": "isolated", "alpha": 0.4 }
}
```

`alpha` must match the periodic branches. Nothing else applies: meshes,
k-vector cutoffs, `slab_correction` and `pbc` describe a sum over periodic
images, and passing one is an error rather than a silent no-op.

**What the term computes is fixed, not a choice.** The periodic path splits
Coulomb at α:

```
Σ qᵢqⱼ/r  =  Σ qᵢqⱼ erfc(αr)/r     ← short-ranged; the network learns this
           + Σ qᵢqⱼ  erf(αr)/r     ← smooth and long-ranged
```

`LongRangeEnergy` returns the second sum in reciprocal space, so for an
isolated system the consistent term is the same sum evaluated directly, at the
same α:

```
Eᵢ = ½ Σ_{j≠i} qᵢqⱼ erf(α rᵢⱼ) / rᵢⱼ
```

Put a cluster in a cube of side L and the periodic `recip − self` converges
onto this sum as 1/L³: −6.6e-03, −8.3e-04 and −1.0e-04 eV at L = 80, 160 and
320 Å. It is the same quantity, which is what lets one backbone serve both
branches: the remainder left to the network is `erfc(αr)/r` either way.
Consequences:

- **No self term to subtract.** `erf(αr)/r → 2α/√π` as `r → 0`, which times
  `½Σqᵢ²` is the Ewald self term; summing `j ≠ i` already excludes it.
- **Charged molecules are easier than in a cell**: no neutralising background,
  so an ion is just an ion.
- **No cutoff.** `erf(αr)/r` decays as `1/r`; truncating it would be wrong.

It is summed densely, not through nvalchemiops, whose two candidates both fail
force-loss training: `dsf_coulomb` is a shifted-force potential with a
charge-dependent self term and no position autograd, and `coulomb_energy`
raises under `create_graph=True`. The dense form is exact, twice
differentiable, and costs about 1 ms at 1000 atoms; its pair tensor is
`O(N²)` (8 MiB at 1000 atoms in float64), fine for molecules and the reason it
is not the method for a periodic cell.

Without a cell, two things differ on purpose: **the virial is still computed**
(the pair sum `Σ rᵢⱼ ⊗ fᵢⱼ`, which needs no periodicity, and which DeePMD-kit's
own energy model also returns for a `box=None` frame), and **stress is not
reported**, since there is no volume to divide by.

## Running MD

Through the ASE calculator, with any ASE integrator; the GPU MD classes in
`deepmd_lr.nvalchemi` run bulk cells and slabs and refuse a molecule.

```python
from ase import units
from ase.md.verlet import VelocityVerlet
from deepmd_lr.ase import LongRangeCalculator

atoms.calc = LongRangeCalculator("model.ckpt.pt")   # trained with method "isolated"
VelocityVerlet(atoms, 0.25 * units.fs).run(2000)
```

The term holds no weights, so a model trained on periodic cells can be asked
for an isolated system by swapping the term at load, `alpha` inherited
([same-as](same-as.md)):

```python
LongRangeCalculator("bulk.ckpt.pt", long_range={"method": "isolated"})
```

That makes the electrostatics exact for the cluster; whether the short-range
part transfers is a question of training data, since a bulk model has never
seen a surface. A slab needs no override: its `pbc`, with one axis `false`,
switches on the slab sum in the ASE calculator and the GPU classes alike
([example 03](../examples/03-slab)).
