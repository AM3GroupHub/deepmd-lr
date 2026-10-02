# Compressed backbone

DeePMD-kit can compress DPA4C — radial tables plus one fused CUDA kernel —
and the descriptor is where the time goes, so the long-range model gets the
full benefit: **13.7× per step, 19× less memory** ([Cost](cost.md#with-the-compressed-backbone)).
The charge head and the reciprocal term are untouched.

**How to compress**

```python
# ASE
atoms.calc = LongRangeCalculator("model.ckpt.pt", compress=True)

# nvalchemi MD (examples/06-md_nvalchemi)
ev = build_evaluator("model.ckpt.pt", atoms, compress=True)

# a model you already hold, on the GPU and in eval mode
model.enable_compression(table_stride_1=0.01)
```

`compress=True` is DeePMD-kit's default 0.01 Å table stride; pass a float for
another; a numeric stride must be finite and strictly positive (`0.0` is
invalid). Only the boolean `False` disables compression. Compress once per
process, after loading. It is inference-only. Failed compression preserves
the original descriptor state and leaves `DP_CUDA_INFER` unchanged, so the
caller can correct the inputs and retry.

**Requirements**

- A deepmd-kit **built from source with the CUDA operators** — see below. The
  PyPI wheel lacks the kernel, and DeePMD-kit's compressed branch then runs
  the uncompressed math without saying so, so on the wheel `compress=True` is
  refused: `RuntimeError: compression needs the compiled operator
  torch.ops.deepmd.dpa4c_graph_compress ...`
- `DP_CUDA_INFER=1`. Set for you if unset. Level 2 is refused for a model
  with `long_range`: its fused energy/force path bypasses the shared backbone,
  the charge response and the Born charges. A model without `long_range` may
  set `DP_CUDA_INFER=2` for that fused path (force as a value, no autograd).
- A `float32` descriptor with no `exclude_types` (upstream's rule).

**What it changes**

| | |
|---|---|
| charge state | folded into the tables; the calculator/evaluator fold in the run's `charge_spin`, and a forward with a different one is refused. The descriptor records the folded state, so the refusal holds however the tables were built: through upstream's `enable_compression` (which folds `default_chg_spin`), a later `apply_charge_state`, or a model with no `long_range` block |
| persistence | none — tables are rebuilt at load; a `state_dict` saved from a compressed model is rejected by `load_model` |
| multi-GPU | the evaluator compresses once, before making replicas |
| precision | none: the fp32 descriptor already sees fp32 edge vectors; the only new error is the tabulation, measured above |

## Building the CUDA operators

Verified on an aarch64 GB10 (cc 12.1), CUDA 13.0, cmake 3.28, gcc 13,
torch 2.11.0+cu130; about two minutes on 20 cores.

```bash
conda create --clone dpmd -n dpmd-cuda && conda activate dpmd-cuda
pip install "scikit-build-core>=1" setuptools_scm dependency_groups packaging ninja hatch-fancy-pypi-readme
git clone --branch v3.2.0 --depth 1 https://github.com/deepmodeling/deepmd-kit.git && cd deepmd-kit
export DP_VARIANT=cuda CUDAToolkit_ROOT=/usr/local/cuda DP_ENABLE_TENSORFLOW=0 \
       TORCH_CUDA_ARCH_LIST="12.0" SETUPTOOLS_SCM_PRETEND_VERSION=3.2.0 \
       SKBUILD_CMAKE_DEFINE="CMAKE_CUDA_ARCHITECTURES=120;DEEPMD_CUDA_PORTABLE_PTX=OFF;CMAKE_DISABLE_FIND_PACKAGE_MPI=ON;CMAKE_CUDA_FLAGS=-gencode arch=compute_120,code=sm_120"
pip install --no-build-isolation --no-deps -v .
pip install --no-deps -e /path/to/deepmd-lr
```

Each of these silently yields a kernel-less build if omitted: `--no-build-isolation`
(else a CPU torch is found and no CUDA is compiled); the arch triple (torch's
CMake resets `CMAKE_CUDA_ARCHITECTURES`, after which DeePMD-kit compiles every
architecture or PTX only — use your GPU's `sm_`); MPI off (a pip `mpich` gets
auto-linked and the library fails to load); `--no-deps` on both installs (else
the `==3.2.0` pin pulls the wheel back).

Verify: `python -c "import deepmd.pt_expt; from deepmd.kernels.cuda.dpa4c.graph_compress import op_available; print(op_available())"`
prints `True`, and `DEEPMD_LR_REQUIRE_COMPRESS=1 pytest tests/` passes — that
variable fails collection if either CUDA or the operator is unavailable.
Without it, CPU-capable checks still run and only CUDA execution tests skip.

