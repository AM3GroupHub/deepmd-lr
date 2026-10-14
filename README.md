# DeepMD-LR: A long-range electrostatics plugin for DPA4C

A charge head on the shared DPA4C descriptor and a reciprocal-space Ewald/PME
term from `nvalchemi-toolkit-ops`, trained jointly with the short-range model.
A **plugin, not a fork**: it registers through DeePMD-kit's `deepmd.pt_expt`
entry point and works against a stock `pip install deepmd-kit==3.2.0`.

On the periodic water benchmark the joint model beats the strictly local
baseline by **4.9× in force RMSE**, the floor a 6 Å cutoff imposes; with the
compressed backbone MD runs **13.7× faster** than uncompressed.

## Install

Into an environment with `deepmd-kit==3.2.0` and a matching `torch`:

```bash
pip install -e . --no-deps
```

Building that environment, the `mpich` trap of the PyPI wheel and the smoke
test: [docs/install.md](docs/install.md). The wheel's CPU-only operators
suffice for training and MD; only the
[compressed backbone](#compressed-backbone) needs a source build.

## Train

```json
{
  "model": {
    "type": "dp_lr",
    "type_map": ["O", "H"],
    "descriptor": { "type": "dpa4c", "rcut": 6.0 },
    "fitting_net": { "neuron": [192, 192, 192] },
    "long_range": {
      "charge_net": { "neuron": [64, 64] },
      "reciprocal": { "alpha": 0.4, "mesh_dimensions": [16, 16, 16] }
    }
  }
}
```

```bash
dp --pt-expt train input.json
```

Without the `long_range` block the model is exactly the standard one, and
**always train that baseline twin too** (delete `"type": "dp_lr"` and
`long_range`, change nothing else): it is the only way to know the term
helped. `alpha` and the mesh are not free parameters
([choosing them](docs/alpha-and-mesh.md)). `type_map` is abbreviated above;
the converter writes the whole periodic table by default and the model's must
match ([dataset layout](docs/dataset-layout.md)).

## Examples

Each case in `examples/` was run and comes with its data preparation, an
annotated `input.json` and a README.

| | case | teaches |
|---|---|---|
| [01](examples/01-bulk) | bulk water, 3D-periodic | xyz → dataset → `input.json` → train |
| [02](examples/02-molecules) | molecules, no cell | `nopbc`, `charge_spin.npy`, `method: "isolated"` |
| [03](examples/03-slab) | Li/LiFSI interface, 2D-periodic | vacuum padding, `slab_correction` + `pbc` |
| [04](examples/04-multi-head) | all three at once | `shared_dict`/`model_dict`, and the gradient clipping it needs |
| [05](examples/05-pseudo-head) | bulk + slab, one weight set | `same_as`; the long-range type set at load |
| [06](examples/06-md_nvalchemi) | MD on nvalchemiops | NVE, NVT, NPT, NPH; a field run on every GPU; your own thermostat |
| [07](examples/07-ase-calculator) | the ASE calculator | single points, charges, Born tensors, ASE's optimizers and MD |
| [08](examples/08-nemd) | thermal conductance by NEMD | a region thermostat with an exact heat account, fixed layers, a slab on the GPU path |
| [09](examples/09-fire-swap) | B-site order in PMN by FIRE-Swap | relaxed-swap Metropolis on one evaluator, rollback and resume |

## Run MD

Only this package's own paths see the cell, so only they evaluate the
long-range term: `dp freeze`, `DeepPot`, `dp test` and LAMMPS are **refused**
for a long-range model rather than allowed to return the short-range energy
silently ([why](docs/md.md)).

```python
from ase.md.verlet import VelocityVerlet
from ase import units
from deepmd_lr.ase import LongRangeCalculator

atoms.calc = LongRangeCalculator("run_joint/model.ckpt.pt")
VelocityVerlet(atoms, timestep=0.5 * units.fs).run(10000)
```

For MD that never leaves the GPU, on one or several, the same pattern runs on
nvalchemiops kernels through one evaluator shared by every stage:

```python
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from deepmd_lr.nvalchemi import FIRE2, Langevin, build_evaluator

ev = build_evaluator("run_joint/model.ckpt.pt", atoms, devices=["cuda:0", "cuda:1"])
FIRE2(ev, atoms).run(fmax=0.05)
thermalize_momenta(atoms, temperature_K=300)
Stationary(atoms)
Langevin(ev, atoms, 0.5 * units.fs, temperature_K=300, friction=0.01 / units.fs,
         logfile="md.log", loginterval=100).run(10000)
```

`VelocityVerlet`, `Langevin`, `NoseHooverChainNVT`, `VelocityRescaling`,
`MTKNPT`, `MTKNPH`, `FIRE` and `FIRE2` all work this way, with trajectories,
observers and checkpoints ([partitioned MD](docs/partitioned-md.md)). A field
run, Born effective charges and their screening by ε∞:
[born-charges](docs/born-charges.md).

### Compressed backbone

DeePMD-kit's DPA4C compression (radial tables + one fused CUDA kernel) covers
the backbone, about 90% of a step. At 6,144 atoms:

| | ms/step | ns/day | peak GiB |
|---|---:|---:|---:|
| long-range | 373.3 | 0.116 | 6.41 |
| long-range, compressed | **27.2** | **1.590** | **0.34** |

```python
atoms.calc = LongRangeCalculator("model.ckpt.pt", compress=True)     # ASE
ev = build_evaluator("model.ckpt.pt", atoms, compress=True)          # nvalchemi MD
```

The tabulation error (1e-5 eV/Å in force, 1e-7 e in charge) is three orders
below the model's error against DFT. Compression needs a deepmd-kit built from
source with the CUDA operators, is inference-only, and folds the run's
`charge_spin` into the tables; each rule is refused rather than warned about
when violated ([docs/compression.md](docs/compression.md)).

## Scripts

- `scripts/xyz_to_deepmd.py` converts any trajectory ASE can read into a
  DeePMD dataset (`--split-by-composition`, `--vacuum`); it refuses to
  overwrite an existing dataset.
- `scripts/graft_short_range.py` turns a short-range checkpoint into the
  initial state of a joint model, with the charge head scaled down
  ([training](docs/training.md)).
- `scripts/calibrate_born.py` fits the scale and sign of a trained model's
  Born tensors to DFPT labels, the one post-training calibration the charges
  need ([why](docs/born-charges.md)).

## Documentation

| | |
|---|---|
| [install.md](docs/install.md) | building the environment, the `mpich` trap, the smoke test |
| [alpha-and-mesh.md](docs/alpha-and-mesh.md) | does the term apply to your data; choosing `alpha` and the mesh; `pme` vs `ewald` |
| [non-periodic.md](docs/non-periodic.md) | molecules: `nopbc`, `method: "isolated"`, what that term computes |
| [multi-head.md](docs/multi-head.md) | several levels of theory: `shared_dict`, `gradient_max_norm`, the rules that bite |
| [same-as.md](docs/same-as.md) | one theory, bulk + slab: `same_as`; setting the long-range type at load |
| [dataset-layout.md](docs/dataset-layout.md) | systems, `type_map.raw`, the full-periodic-table convention, HDF5 |
| [training.md](docs/training.md) | schedules, grafting, EMA, what to watch, validating, troubleshooting |
| [cost.md](docs/cost.md) | step time and memory, compressed and not |
| [md.md](docs/md.md) | why export is refused; energy-conservation check; the GPU classes, the traps, relaxing a packed box |
| [compression.md](docs/compression.md) | compressed backbone: requirements, what changes, building the CUDA operators |
| [born-charges.md](docs/born-charges.md) | Born effective charges, MD in an external field, calibrating against DFPT |
| [partitioned-md.md](docs/partitioned-md.md) | `DistributedEvaluator`, the MD classes and one shared evaluator, your own kernel loop, what is conserved, scaling |
| [neighbour-lists.md](docs/neighbour-lists.md) | which neighbour list runs: the serial model's, and the MD engine's GPU-direct one |
| [tests.md](docs/tests.md) | what the suite pins |

```bash
pytest tests/ -q          # compressed-kernel tests need a CUDA build
```
