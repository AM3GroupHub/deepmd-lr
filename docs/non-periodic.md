# Non-periodic systems

Worked example: [`examples/02-molecules`](../examples/02-molecules).

The long-range term follows the system's boundary. A frame with no cell (an
empty `nopbc` file in a training system, `box=None` in the model, `pbc` all
`False` in ASE or the GPU engine) is summed directly, **whatever method the
model was trained with**: `pme` and `ewald` describe how a periodic sum is
taken, and a molecule has none. So one `dp_lr` model trains on, and runs, bulk
cells, slabs and molecules alike:

| `pbc` | sum |
|---|---|
| `T T T` | the 3D Ewald/PME reciprocal sum minus the self term |
| two periodic, one open | the same plus the Yeh–Berkowitz slab correction |
| `F F F` | the isolated sum below, no cutoff, no self term |
| one periodic (a wire) | refused |

`method: "isolated"` remains for a model that never sees a periodic frame: it
makes the isolated sum the only one, takes `alpha` and the kernel settings
below, and refuses every periodic setting and every periodic frame.

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

Put a neutral cluster in a cube of side L and the periodic `recip − self` converges
onto this sum as 1/L³: −6.6e-03, −8.3e-04 and −1.0e-04 eV at L = 80, 160 and
320 Å. It is the same quantity, which is what lets one backbone serve both:
the remainder left to the network is `erfc(αr)/r` either way. Consequences:

- **No self term to subtract.** `erf(αr)/r → 2α/√π` as `r → 0`, which times
  `½Σqᵢ²` is the Ewald self term; summing `j ≠ i` already excludes it.
- **Charged molecules are easier than in a cell**: no neutralising background,
  so an ion is just an ion.
- **No cutoff.** `erf(αr)/r` decays as `1/r`; truncating it would be wrong.

## Two kernels

The sum is O(N²) in pairs. `deepmd_lr.lr.isolated` evaluates it two ways:

| kernel | memory | derivatives |
|---|---|---|
| dense | every pair at once, O(N²) | any order: force training, Hessians |
| chunked | row blocks of 1024 atoms against all N, O(1024 N) | first only: `create_graph=True` is refused |

The dense displacement tensor alone uses 24 MB at 1000 atoms in float64 and
17.5 GB at 27,000; kernel intermediates and autograd need additional memory.

On CUDA, from 1024 atoms, the chunked kernel runs each block as one fused
reduction under `torch.compile` (a few seconds once per process and system
size): at 27,000 atoms in float32 on a GB10, 8 ms forward and 37 ms backward,
against 1.35 s and 1.68 s eager. Two `reciprocal` keys choose, in training and
the ASE calculator:

- `isolated_kernel: auto` (the default) is dense up to `max_isolated_atoms`
  atoms per call (all frames of a batch; default 5000) and chunked beyond;
  `dense` refuses a larger call; `chunked` always takes the chunked kernel.
- Force training needs the dense kernel: keep the isolated batch within the
  cap, or raise it if memory permits.

nvalchemiops' own candidates fail force training: `dsf_coulomb` is a
shifted-force potential with no position autograd, and `coulomb_energy` raises
under `create_graph=True`.

Without a cell, two things differ on purpose: **the virial is still computed**
in training and the ASE calculator (the pair sum `Σ rᵢⱼ ⊗ fᵢⱼ`, which needs no
periodicity, and which DeePMD-kit's own energy model also returns for a
`box=None` frame), and **stress is not reported**, since there is no volume to
divide by.

## Running MD

Any `dp_lr` model, through the ASE calculator with any ASE integrator:

```python
from ase import units
from ase.md.verlet import VelocityVerlet
from deepmd_lr.ase import LongRangeCalculator

atoms.pbc = False
atoms.calc = LongRangeCalculator("model.ckpt.pt")
VelocityVerlet(atoms, 0.25 * units.fs).run(2000)
```

or on the GPU engine, which runs a cluster on one device with the chunked
kernel:

```python
from deepmd_lr.nvalchemi import Langevin, build_evaluator, read_structure

atoms = read_structure("droplet.xyz")    # pbc F F F: returned as read, no cell needed
ev = build_evaluator("model.ckpt.pt", atoms)
Langevin(ev, atoms, 0.5 * units.fs, temperature_K=300,
         friction=0.01 / units.fs).run(10000)
```

On the engine a cluster has no cell: the neighbour list boxes the atoms itself
(their bounding box plus a margin, fixed at the first search; an atom that
later leaves it is clamped into an edge cell, which keeps the search
complete), and the file's cell, zero or a dummy, is never read. One device
only (the pair sum does not partition), and no virial, so NPT, NPH and cell
relaxation are refused; NVE, NVT and position relaxations run. The all-pairs
sum runs in the charge head's precision; `long_range={"isolated_dtype":
"float64"}` asks for float64 (0.8 s per step at 27,000 atoms on a GB10,
against 45 ms in float32).

Whether the short-range part of a bulk-trained model transfers to a molecule
is a question of training data, since a bulk model has never seen a surface;
the electrostatics are exact either way.
