# Tests

```bash
pytest tests/ -q
```

Most files need a CUDA device; those that cannot run without one skip as a
whole. The compressed-kernel tests need deepmd-kit built with the CUDA
operators ([Compression](compression.md#building-the-cuda-operators)):

```bash
DP_CUDA_INFER=1 DEEPMD_LR_REQUIRE_COMPRESS=1 pytest tests/ -q
```

`DEEPMD_LR_REQUIRE_COMPRESS=1` turns the kernel skips into failures, so a
build that silently lost its operator cannot pass.

Two sets of tests are opt-in, since they need a trained water checkpoint:

- **25 ps of water MD**, never wrapped. Every 500 steps it checks that the
  partitioned forces match a fresh evaluation of the wrapped positions. It
  takes about 10 minutes compressed on one GPU.
- **The cell relaxation on water**: a box stretched by 4% comes back to zero
  stress in each mode, in about 20 seconds.

```bash
DEEPMD_LR_WATER_MODEL=path/to/model.ckpt.pt pytest tests/test_nvalchemi.py -k long_water -s
DEEPMD_LR_WATER_MODEL=path/to/model.ckpt.pt pytest tests/test_md.py -k water_relaxes
```

Add `DEEPMD_LR_WATER_COMPRESS=1` (with the CUDA-operator build) to compress
the model first.

**The model and its long-range term**

- `test_lr_term.py` — the reciprocal-space term against a hand-written Ewald
  sum; `pme` against `ewald`; the slab correction; boundary-condition guards,
  batching, masking, and gradients including the double backward force
  training needs; the mesh and the sum a system's cell and pbc pick.
- `test_isolated_term.py` — the no-cell term: its periodic limit, invariances,
  differentiability, configuration, and use inside the model.
- `test_pme_physics.py` — nvalchemiops' PME against the NaCl Madelung
  constant, spline orders, and the double backward.
- `test_joint_model.py` — the combined model: force and virial against finite
  differences of the total energy, the shared backbone, training gradients,
  guards, and serialization.
- `test_charge_head.py` — the total-charge projection (exactness, masking,
  differentiability) and the charge head's physical symmetries.
- `test_charge_contracts.py` — `charge_spin` and `exclude_types`, serial
  against partitioned.
- `test_born.py`, `test_eps_inf.py` — Born effective charges against a
  brute-force derivative, their symmetries, the field force, and the
  `sqrt(eps_inf)` screening on every path.
- `test_multi_task.py`, `test_same_as.py` — multi-head sharing and the
  `same_as` expansion.
- `test_neighbors.py` — the cell-list graph against the dense builder.

**Running it**

- `test_ase_calculator.py`, `test_review_regressions.py` — the ASE calculator
  against the model, its pbc guards and export refusal, and NVE drift; each
  frame's pbc picking its sum and its cell the mesh; a slab with too little
  vacuum refused.
- `test_read_structure.py` — the file's pbc kept; a slab's open axis padded to
  the vacuum, centred, and made whole when the file wrapped it; fixed atoms
  read back and moved with the slab; molecules and wires refused.
- `test_nvalchemi.py` — the partitioner and halo (including unwrapped
  coordinates), the skin rules and memory probe, the device gather, the
  gathered long-range term and virial, and `DistributedEvaluator`, whose
  reused partitions (across cell changes) and cached pair lists must match a
  fresh evaluation at every call. A slab matches the serial model at 1, 2 and
  4 partitions, its pbc switches the slab sum on, and an atom near an open
  face is refused; a slab term on a bulk cell takes the 3D sum, and an
  `isolated` term is refused. A supercell gets the mesh its cell needs and
  reproduces its unit cell.
- `test_thermo.py` — `Thermo`'s measurements, warp aliasing, cell sync, the
  guards that make one evaluator safe to share (`fresh`,
  `cell_is_current`, `set_state`), and fixed atoms: read from `FixAtoms` by
  `from_atoms`, put back at rest before every force call, zero force for the
  kernels and the model's for output, and left out of the temperature.
- `test_md.py` — the MD classes, `FIRE` and `FIRE2`: each integrator and each
  relaxation against a hand-written kernel loop, one evaluator carrying a
  whole workflow, `atoms` kept in step both ways, run semantics and
  observers, checkpoints, the log, conservation (NVE, NPT and NPH as dt² in
  every cell mode, the Nosé–Hoover invariant), the barostats' cell handed to
  the kernels transposed, the relaxations' refusals, the cell relaxation
  (its force against finite differences of the enthalpy, every mode reaching
  the target pressure), the dipole current, and slabs: NVT keeps one in its
  cell, while barostats and cell relaxation refuse it, and frames keep its
  pbc. Fixed atoms stay exactly in place in every integrator and relaxation,
  the temperature and the Nosé–Hoover target count three degrees of freedom
  per mobile atom, NVE matches ASE's own `VelocityVerlet` with the same
  `FixAtoms`, and output, checkpoints and refusals carry the constraint.
- `test_compress.py`, `test_compress_arguments.py`,
  `test_compress_atomic_state.py`, `test_compress_required.py` — compressed
  DPA4C inference: every refusal, the graph-lower guard, argument validation,
  state left intact by a failed compression, and, with the operator, kernel
  dispatch, compressed-vs-uncompressed agreement, and the cached pair lists'
  CSR driving the kernel to the fresh search's answer.

**Plumbing**

- `test_plugin_registration.py`, `test_import_order.py` — entry-point
  registration, schema acceptance, and that each module imports cleanly first.
- `test_ema_default.py` — EMA enforcement for training.
- `test_xyz_to_deepmd.py`, `test_graft_short_range.py` — the two scripts.
