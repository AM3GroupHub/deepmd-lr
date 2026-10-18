# Installing

Use Python 3.11–3.13. From the repository root, after installing the
dependencies below:

```bash
pip install -e . --no-deps
```

The requirements in [`pyproject.toml`](../pyproject.toml) are enforced by pip
unless `--no-deps` is given. Here dependencies are installed explicitly so pip
does not replace a selected CUDA build of torch or DeePMD-kit.

## Building the environment

The recipe below is for the tested aarch64 GB10 environment with CUDA 13.
Use a torch build compatible with your hardware and DeePMD-kit; version
3.2.0 requires `torch==2.11.0.*`.

```bash
# 1. PyTorch 2.11.0 + CUDA 13 -- the version is pinned by deepmd-kit 3.2.0
pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0+cu130"

# 2. deepmd-kit itself (prebuilt wheel)
pip install "deepmd-kit==3.2.0"

# 3. deepmd's torch-path extras, explicitly
pip install "e3nn>=0.5.9" "vesin[torch]" mpich

# 4. the PME engine
pip install "nvalchemi-toolkit-ops>=0.5.0" "warp-lang>=1.16.0"

# 5. data / MD / dev tooling
pip install "dpdata>=0.2.7" "ase>=3.29" pytest pytest-cov ruff
```

**The tested PyPI wheel needs `mpich`.** It is cibuildwheel-built,
so `deepmd/pt/cxx_op.py` calls `load_mpi_library()` unconditionally at import
and resolves `libmpi.so` through mpich's package metadata. Without it,
`import deepmd.pt_expt.descriptor` raises `PackageNotFoundError`.

That wheel has CPU-only DeePMD operators, which suffice for the uncompressed
path even when torch and nvalchemiops run on CUDA. The compressed backbone
needs a [source build](compression.md#building-the-cuda-operators).

## Smoke test

Check imports and GPU discovery. This does not compile or exercise the PME
or compressed descriptor kernels; see [tests](tests.md) for runtime checks.

```bash
python -c "
import torch, warp, deepmd, deepmd_lr
warp.init()
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
from deepmd.pt_expt.descriptor.dpa4c import DescrptDPA4C
print('ok')
"
```
