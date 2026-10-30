# DeepMD-LR: A long-range electrostatics plugin for DPA4C

A charge head on the shared DPA4C descriptor and a reciprocal-space Ewald/PME
term from `nvalchemi-toolkit-ops`, trained jointly with the short-range model.
It registers through DeePMD-kit's `deepmd.pt_expt` entry point and works with
stock `deepmd-kit==3.2.0`.

On the periodic water benchmark the joint model improves force RMSE by
**4.9×** over the local baseline ([training](docs/training.md)). The compressed
backbone gives **13.7× faster MD** in the
[6,144-atom GB10 benchmark](docs/cost.md#with-the-compressed-backbone).

## Install

From this repository, with the dependencies installed in a compatible environment:

```bash
pip install -e . --no-deps
```

See [installation](docs/install.md) for dependencies and an import check.
The [compressed backbone](#compressed-backbone) needs a CUDA source build of
DeePMD-kit.

## Train

The plugin adds `type: "dp_lr"` and `long_range` to the model configuration.
This is a **model fragment**, not a complete training input:

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

Use [example 01](examples/01-bulk) for data preparation and a complete
`input.json`. Train a baseline with `"type": "dp_lr"` and `long_range` removed
to measure the improvement. Choose [alpha and the mesh](docs/alpha-and-mesh.md)
together. The abbreviated `type_map` above must be expanded if using the
converter's default full periodic table ([dataset layout](docs/dataset-layout.md)).

## Examples

Each directory has a README with its inputs and commands; training examples
include `input.json`, and simulation examples include runnable scripts.

| | case | teaches |
|---|---|---|
| [01](examples/01-bulk) | bulk water, 3D-periodic | xyz → dataset → `input.json` → train |
| [02](examples/02-molecules) | molecules, no cell | `nopbc`, `charge_spin.npy`, the isolated sum |
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

For GPU integration, share one evaluator across relaxation and MD. Supply a
trained checkpoint and a bulk or slab structure:

```python
from ase import units
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from deepmd_lr.nvalchemi import FIRE2, Langevin, build_evaluator, read_structure

atoms = read_structure("start.xyz")
ev = build_evaluator("run_joint/model.ckpt.pt", atoms)  # one GPU by default
FIRE2(ev, atoms).run(fmax=0.05)
thermalize_momenta(atoms, temperature_K=300)
Stationary(atoms)
Langevin(ev, atoms, 0.5 * units.fs, temperature_K=300, friction=0.01 / units.fs,
         logfile="md.log", loginterval=100).run(10000)
```

NVE, NVT, NPT, NPH, cell relaxation, multiple GPUs and checkpoints are covered
in [partitioned MD](docs/partitioned-md.md). For ASE integrators and optimizers,
attach `LongRangeCalculator` instead ([ASE workflow](docs/md.md)). For an
external electric field, see [Born charges](docs/born-charges.md).

### Compressed backbone

DeePMD-kit's DPA4C compression replaces the backbone with radial tables and a
fused CUDA kernel:

```python
ev = build_evaluator("model.ckpt.pt", atoms, compress=True)
```

`LongRangeCalculator` also accepts `compress=True`. Compression is for
inference and fixes the run's `charge_spin` in the tables. See
[requirements and build instructions](docs/compression.md) and
[measured speed, memory and tabulation error](docs/cost.md#with-the-compressed-backbone).

## Scripts

- `scripts/xyz_to_deepmd.py` converts any trajectory ASE can read into a
  DeePMD dataset (`--split-by-composition`, `--vacuum`); it refuses to
  overwrite an existing dataset.
- `scripts/graft_short_range.py` turns a short-range checkpoint into the
  initial state of a joint model, with the charge head scaled down
  ([training](docs/training.md)).
- `scripts/calibrate_born.py` fits the scale and sign of a trained model's
  Born tensors to DFPT labels ([calibration](docs/born-charges.md#calibrating-the-scale-and-sign-against-dfpt-labels)).

## Documentation

| | |
|---|---|
| [install.md](docs/install.md) | dependencies, environment setup and import check |
| [alpha-and-mesh.md](docs/alpha-and-mesh.md) | does the term apply to your data; choosing `alpha` and the mesh; `pme` vs `ewald` |
| [non-periodic.md](docs/non-periodic.md) | molecules and clusters: the isolated sum, its two kernels, MD on the GPU |
| [multi-head.md](docs/multi-head.md) | several levels of theory: sharing, clipping and branch configuration |
| [same-as.md](docs/same-as.md) | one theory, bulk + slab: `same_as`; setting the long-range type at load |
| [dataset-layout.md](docs/dataset-layout.md) | systems, `type_map.raw`, the full-periodic-table convention, HDF5 |
| [training.md](docs/training.md) | schedules, grafting, EMA, what to watch, validating, troubleshooting |
| [cost.md](docs/cost.md) | step time and memory, compressed and not |
| [md.md](docs/md.md) | ASE workflow, energy conservation, charge filtering and initial relaxation |
| [compression.md](docs/compression.md) | compressed backbone: requirements, what changes, building the CUDA operators |
| [born-charges.md](docs/born-charges.md) | Born effective charges, MD in an external field, calibrating against DFPT |
| [partitioned-md.md](docs/partitioned-md.md) | GPU MD and its units, cell updates, skin and halos, custom integrators and scaling |
| [neighbour-lists.md](docs/neighbour-lists.md) | which neighbour list runs: the serial model's, and the MD engine's GPU-direct one |
| [tests.md](docs/tests.md) | running the tests and coverage by module |

```bash
pytest tests/ -q          # compressed-kernel tests need a CUDA build
```

## License

MIT, see [LICENSE](LICENSE). DeePMD-kit (LGPL-3.0-or-later) and nvalchemiops
(Apache-2.0) are dependencies the plugin imports, not code it includes.
