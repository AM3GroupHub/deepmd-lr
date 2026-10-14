# Installing

Into an environment with `deepmd-kit==3.2.0` and a matching `torch`:

```bash
pip install -e . --no-deps
```

`--no-deps` is deliberate: the dependency pins in `pyproject.toml` are
documentation, and the environment is built explicitly below (a CUDA-13
aarch64 `torch` from the PyTorch index, which pip should not try to
re-resolve).

## Building the environment

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
MD. The one thing it cannot do is the compressed backbone, whose CUDA kernel
needs a [source build](compression.md#building-the-cuda-operators).

## Smoke test

The riskiest part is warp compiling for sm_121; if it fails, `warp.init()`
says so:

```bash
python -c "
import torch, warp, deepmd
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
from deepmd.pt_expt.descriptor.dpa4c import DescrptDPA4C
print('ok')
"
```
