# deepmd-lr

Long-range electrostatics for **DPA4C**: a charge head on the shared
descriptor, a reciprocal-space Ewald/PME term from `nvalchemi-toolkit-ops`,
trained jointly with the short-range model. A **plugin, not a fork** — it
registers through DeePMD-kit's `deepmd.pt_expt` entry point and works against
a stock `pip install deepmd-kit==3.2.0`.

On the periodic benchmark the joint model beats the strictly local baseline by
**4.9× in force RMSE**, breaking the floor a 6 Å cutoff imposes; with the
compressed backbone it runs MD **13.7× faster** than uncompressed. Every number
here was measured on this setup.

## Install

Into an environment with `deepmd-kit==3.2.0` and a matching `torch`:

```bash
pip install -e . --no-deps
```

`--no-deps` is deliberate: the dependency pins are recorded in
`pyproject.toml` for documentation, but the environment is built explicitly
(a CUDA-13 aarch64 `torch` from the PyTorch index, which pip should not try
to re-resolve).

### Building that environment

`deepmd-kit==3.2.0` pins `torch==2.11.0.*`, and an sm_121 GPU needs a CUDA
12.8+/13 build, so torch comes from the cu130 index rather than PyPI:

```bash
# 1. PyTorch 2.11.0 + CUDA 13 -- the version is pinned by deepmd-kit 3.2.0
pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0+cu130"

# 2. deepmd-kit itself (prebuilt wheel)
pip install "deepmd-kit==3.2.0"

# 3. deepmd's torch-path extras, explicitly
pip install "e3nn>=0.5.9" "vesin[torch]" mpich

# 4. the PME engine
pip install "nvalchemi-toolkit-ops>=0.4.1" "warp-lang>=1.13.0"

# 5. data / MD / dev tooling
pip install "dpdata>=0.2.7" "ase>=3.29" pytest pytest-cov ruff
```

**`mpich` is required, not optional.** The PyPI wheel is cibuildwheel-built,
so `deepmd/pt/cxx_op.py` calls `load_mpi_library()` unconditionally at import
and resolves `libmpi.so` through mpich's package metadata. Without it,
`import deepmd.pt_expt.descriptor` raises `PackageNotFoundError`.

The PyPI wheel is a CPU-only operator build, which is fine for training and
MD. The one thing it cannot do is the [compressed backbone](#compress-the-model),
whose CUDA kernel needs a [source build](docs/compression.md#building-the-cuda-operators).

Smoke-test before anything else — the riskiest part is warp compiling for
sm_121, and if it fails `warp.init()` says so:

```bash
python -c "
import torch, warp, deepmd
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
from deepmd.pt_expt.descriptor.dpa4c import DescrptDPA4C
print('ok')
"
```

## Usage

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

Without `long_range` or `zbl`, the model is exactly the standard one — and
**always train that baseline twin too** (delete `"type": "dp_lr"` and
`long_range`, change nothing else). It is the only way to know the term
helped. `alpha` and the mesh are not free parameters: see
[choosing them](docs/alpha-and-mesh.md). `type_map` is abbreviated above; the
converter writes the whole periodic table by default and the model's must
match ([dataset layout](docs/dataset-layout.md)).

## Optional ZBL repulsion

DPA4C and DPA4C-LR can include a fixed, smoothly truncated nuclear-repulsion
prior through `model.zbl`. Both use `type: "dp_lr"`; omit `long_range` for
pure DPA4C. ZBL is disabled unless configured and adds no trainable parameters.
See [configuration, training and limitations](examples/10-zbl/README.md).

## Examples

`examples/` is the step-by-step material: data preparation, an annotated
`input.json`, and a README per case. Each was run.

| | case | teaches |
|---|---|---|
| [01](examples/01-bulk) | bulk liquid water, 3D-periodic | the basic loop: xyz → dataset → `input.json` → train |
| [02](examples/02-molecules) | four molecules, no cell | `nopbc`, `charge_spin.npy`, `method: "isolated"`, one system per composition |
| [03](examples/03-slab) | Li/LiFSI interface, 2D-periodic | vacuum padding, `slab_correction` + `pbc` |
| [04](examples/04-multi-head) | all three at once | `shared_dict`/`model_dict`, and the gradient clipping it needs |
| [05](examples/05-pseudo-head) | bulk + slab, one weight set | `same_as`, and setting the long-range type at load |
| [06](examples/06-md_nvalchemi) | MD with nvalchemiops | NVE, NVT, NPT and NPH, one script each; a field run on every visible GPU; a thermostat of your own |
| [07](examples/07-ase-calculator) | the ASE calculator | single points, charges and Born tensors, ASE's optimizers and MD |
| [08](examples/08-nemd) | interfacial thermal conductance by NEMD | a region thermostat of your own with an exact heat account, fixed layers, a slab on the GPU path, resume, and the fits that give G |
| [09](examples/09-fire-swap) | B-site order in PMN by FIRE-Swap Monte Carlo | relaxed-swap Metropolis on one evaluator, cell relaxation after a local change, rollback and resume |

`scripts/xyz_to_deepmd.py` converts a trajectory ASE can read into a DeePMD
dataset, including the `--split-by-composition` and `--vacuum` cases the
examples use. Composition splitting also sorts atoms and their per-atom labels
into a consistent element order. Choose a new or empty output directory:
the converter refuses to overwrite existing datasets, preventing stale charge
labels or boundary markers from being mixed into a new conversion.

## Running MD

Only this package's own paths see the cell, so only they evaluate the
long-range term. `dp freeze`, `DeepPot`, `dp test` and LAMMPS are **refused**
for a long-range model rather than allowed to return the short-range energy
silently ([why](docs/md.md)).

```python
from ase.md.verlet import VelocityVerlet
from ase import units
from deepmd_lr.ase import LongRangeCalculator

atoms.calc = LongRangeCalculator("run_joint/model.ckpt.pt")
VelocityVerlet(atoms, timestep=0.5 * units.fs).run(10000)
```

For MD that never leaves the GPU, and for several GPUs, the same pattern runs
on nvalchemiops kernels. The evaluator holds the model on every device, is
built once, and is shared by every stage that follows:

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

Every ensemble nvalchemiops provides has a class that works the same way,
with a trajectory, observers and checkpoints: `VelocityVerlet` (NVE),
`Langevin`, `NoseHooverChainNVT` and `VelocityRescaling` (NVT), `MTKNPT` and
`MTKNPH` in isotropic, orthorhombic or full-cell modes (NPT, NPH). `FIRE` is
the classic relaxation beside `FIRE2`, and both can relax the cell.
`examples/06-md_nvalchemi` has a script per ensemble, one of them on every
visible GPU ([partitioned MD](docs/partitioned-md.md)), and
`examples/07-ase-calculator` walks through the ASE calculator. A field run,
Born effective charges, and reporting them screened by a medium's ε∞
(`eps_inf=`): [born-charges](docs/born-charges.md).

### Compress the model

DeePMD-kit's DPA4C compression (radial tables + one fused CUDA kernel) applies
to the backbone, which is ~90% of a step. Measured at 6,144 atoms:

| | ms/step | ns/day | peak GiB |
|---|---:|---:|---:|
| long-range | 373.3 | 0.116 | 6.41 |
| long-range, compressed | **27.2** | **1.590** | **0.34** |

```python
atoms.calc = LongRangeCalculator("model.ckpt.pt", compress=True)     # ASE
ev = build_evaluator("model.ckpt.pt", atoms, compress=True)          # nvalchemi MD
model.enable_compression(table_stride_1=0.01)                        # a loaded model, on GPU, eval mode
```

`compress=True` is the default 0.01 Å table stride (a float sets another; the
finer 0.002 buys nothing). The tabulation error is 1e-5 eV/Å in force and 1e-7 e
in charge — three orders below the model's error against DFT.

Three things to know, each refused rather than warned about when violated:

- **It needs a deepmd-kit built from source with the CUDA operators.** The
  PyPI wheel lacks the kernel, and DeePMD-kit would then silently run the
  uncompressed math — so on the wheel `compress=True` raises, naming the fix.
  The build takes about two minutes:
  [docs/compression.md](docs/compression.md#building-the-cuda-operators).
- **It is inference-only**, and `DP_CUDA_INFER` becomes `1`. Level 2 is
  refused for a long-range model (it bypasses the shared backbone) and allowed
  for a short-range one, where its fused energy/force path is faster.
- **The charge state is folded into the tables**: the calculator/evaluator fold
  in the run's `charge_spin`; a forward with a different one is refused.

Details, the source-build traps, and what changes for multi-GPU and
persistence: [docs/compression.md](docs/compression.md).

## Documentation

| | |
|---|---|
| [alpha-and-mesh.md](docs/alpha-and-mesh.md) | does the term apply to your data; choosing `alpha` and the mesh; `pme` vs `ewald` |
| [non-periodic.md](docs/non-periodic.md) | molecules: `nopbc`, `method: "isolated"`, what that term computes |
| [multi-head.md](docs/multi-head.md) | several levels of theory: `shared_dict`, `gradient_max_norm`, the rules that bite |
| [same-as.md](docs/same-as.md) | one theory, bulk + slab: `same_as`; setting the long-range type at load |
| [dataset-layout.md](docs/dataset-layout.md) | systems, `type_map.raw`, the full-periodic-table convention, HDF5 |
| [training.md](docs/training.md) | schedules, grafting, EMA, what to watch, validating, troubleshooting |
| [cost.md](docs/cost.md) | step time and memory, compressed and not |
| [md.md](docs/md.md) | why export is refused; energy-conservation check; the GPU classes, the traps, relaxing a packed box |
| [compression.md](docs/compression.md) | compressed backbone: requirements, what changes, building the CUDA operators |
| [born-charges.md](docs/born-charges.md) | Born effective charges and MD in an external field |
| [partitioned-md.md](docs/partitioned-md.md) | `DistributedEvaluator`, the MD classes and one shared evaluator, your own kernel loop, what is conserved, scaling |
| [neighbour-lists.md](docs/neighbour-lists.md) | which neighbour list runs: the serial model's, and the MD engine's GPU-direct one |
| [tests.md](docs/tests.md) | what the suite pins |

```bash
pytest tests/ -q          # compressed-kernel tests need a CUDA build
```
