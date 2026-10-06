# ZBL validation

Validated on 2026-10-06, based on upstream `dev-md` commit
`2b09991a17693e094d87bfa9ec3cbdea198916fa`.

Environment: DeePMD-kit 3.2.0 with the pt_expt/DPA4C kernels, PyTorch 2.11.0,
nvalchemi-toolkit-ops 0.4.1; one NVIDIA GeForce RTX 5090 for CUDA checks.
The installed scientific environment was not replaced: tests imported this
checkout through `PYTHONPATH`.

## Results

- CPU suite: 102 passed, 17 GPU-only neutrality-loss cases skipped.
- GPU suite: 23 passed, including those 17 neutrality-loss cases.
- Total: 125 distinct passing tests; no failures in the final suites.
- Ruff on the changed Python modules/tests and `git diff --check`: passed.

The new tests cover fixed parameter counts, input normalization and invalid
settings, screened pair energies, switching continuity through second derivative,
force/virial finite differences, optimizer gradients, serialization, real
`dp --pt-expt train` runs and restart, EMA checkpoint loading, `same_as`,
charge-filter independence, masked/empty edges, and pure-DPA4C graph export.
The LR CLI test uses `ener_lr` with nonzero neutrality-loss prefactors.
Existing registration, EMA, sharing and neutrality-loss tests provide regression
coverage for models without ZBL.

CUDA comparisons cover compressed and uncompressed DPA4C at inference levels 1
and 2, and DPA4C-LR at level 1, including PME. E/F/virial are compared with the
serial uncompressed evaluator at absolute tolerance 2e-4 in their respective
native units (eV, eV/Angstrom, eV), and relative tolerance 2e-5. Two spatial
partitions run on the same physical GPU to test ownership/halo accounting.
This is not a multi-GPU scaling measurement.

## Reproduce

Use an environment with the repository's scientific dependencies and run from
the repository root:

```bash
CUDA_VISIBLE_DEVICES='' WARP_CACHE_PATH=/tmp/deepmd-lr-zbl-warp \
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest -q \
  tests/test_zbl.py tests/test_plugin_registration.py \
  tests/test_ema_default.py tests/test_charge_same_as.py \
  tests/test_same_as.py tests/test_neutrality_loss.py

OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest -q \
  tests/test_zbl_gpu.py tests/test_neutrality_loss.py
```

GPU compression cases skip if the compiled DPA4C operator is unavailable; they
all executed in the reported run. The tiny training dataset exercises software
contracts, not model accuracy. No H200 result, long production MD trajectory,
collision-physics validation or throughput percentage is claimed. See the
[configuration guide](../examples/10-zbl/README.md) for supported paths and limits.
