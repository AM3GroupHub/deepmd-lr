# Tests

```bash
pytest tests/ -q                                                # most files need a CUDA device
DP_CUDA_INFER=1 DEEPMD_LR_REQUIRE_COMPRESS=1 pytest tests/ -q   # with the CUDA-operator build
```

`DEEPMD_LR_REQUIRE_COMPRESS=1` turns the compressed-kernel skips into failures,
so a build that lost its operator cannot pass
([building it](compression.md#building-the-cuda-operators)).

Two tests are opt-in because they need a trained water checkpoint; add
`DEEPMD_LR_WATER_COMPRESS=1` to compress it first:

```bash
DEEPMD_LR_WATER_MODEL=model.ckpt.pt pytest tests/test_nvalchemi.py -k long_water -s   # 25 ps never wrapped, forces checked every 500 steps; ~10 min
DEEPMD_LR_WATER_MODEL=model.ckpt.pt pytest tests/test_md.py -k water_relaxes          # a 4% stretched box back to zero stress in each mode; ~20 s
```

| file | pins |
|---|---|
| `test_lr_term.py` | the reciprocal term against a hand-written Ewald sum; `pme` against `ewald`; the slab correction; boundary guards, batching, masking, gradients including the double backward; the mesh and sum a cell and pbc pick |
| `test_isolated_term.py` | the no-cell term: its periodic limit, invariances, differentiability, configuration, use inside the model |
| `test_pme_physics.py` | nvalchemiops' PME against the NaCl Madelung constant, spline orders, the double backward |
| `test_joint_model.py` | force and virial against finite differences of the energy, the shared backbone, training gradients, guards, serialization |
| `test_charge_head.py` | the total-charge projection (exactness, masking, differentiability) and the charge head's symmetries |
| `test_born.py`, `test_eps_inf.py` | Born charges against a brute-force derivative, their symmetries, the field force, `sqrt(eps_inf)` on every path |
| `test_multi_task.py`, `test_same_as.py`, `test_charge_same_as.py` | multi-head sharing, the `same_as` expansion, one charge head under two theories |
| `test_neighbors.py` | the cell-list graph against the dense builder |
| `test_nlist.py` | the GPU-direct list against the dense oracle: periodic, small (several images of one pair), triclinic and slab cells, unwrapped positions, a halo node set, a set drifting past its box; both CSR views valid; an overflowed matrix or edge buffer reported and grown; a poisoned tail never read; a warm search without a synchronising op, correct on a side stream; differentiable vectors reaching the coordinates and strain; the Born row-sum kernel equal to the array-API response and deterministic |
| `test_ase_calculator.py` | the calculator against the model, pbc guards, export refusal, NVE drift; each frame's pbc picking its sum and its cell the mesh; too little vacuum refused; a multi-head branch loaded by name |
| `test_read_structure.py` | the file's pbc kept; a slab's open axis padded, centred, made whole; fixed atoms read and moved with the slab; molecules and wires refused |
| `test_nvalchemi.py` | the partitioner and halo (unwrapped too), skin rules and memory probe, the device gather, the gathered term and virial; `DistributedEvaluator` matching a fresh evaluation at every call, across cell changes; slabs at 1, 2 and 4 partitions, an atom near an open face refused, an `isolated` term refused; a supercell's mesh; evaluators sharing one model; `charge_spin` and `exclude_types` serial and partitioned alike |
| `test_sync.py` | one download per evaluation (one bool more on two partitions, one more for an uncompressed model's edge count); no pageable upload, `nonzero` or `bincount` |
| `test_thermo.py` | `Thermo`'s measurements, warp aliasing, cell sync, the guards that make one evaluator safe to share, fixed atoms at rest and out of the temperature |
| `test_md.py` | the MD classes and relaxations against hand-written kernel loops; one evaluator across stages; `atoms` in step both ways; observers, checkpoints, the log; conservation as dt² in every cell mode and the Nosé–Hoover invariant; the barostats' transposed cell; the cell relaxation against finite differences of the enthalpy; the dipole current; slabs kept, refused by barostats, restored only onto their boundary; fixed atoms in every integrator |
| `test_compress.py` | every refusal, the graph-lower guard, argument validation, state intact after a failed compression, the strict mode; with the operator, kernel dispatch, compressed against uncompressed, and the GPU list's CSR views driving the kernel at 1, 2 and 4 partitions |
| `test_charge_filter.py` | exact zeros below the threshold, the total kept, a C³ weight, forces equal to minus the gradient on one and two partitions, serial = partitioned = ASE, serialization, training gradients, overrides and refusals |
| `test_neutrality_loss.py` | the `ener_lr` loss: the reported shift, the term and its schedule, its gradient reaching head and backbone, schema, factory, serialization |
| `test_ema_default.py` | EMA enforced for training |
| `test_plugin_registration.py`, `test_import_order.py` | entry-point registration, schema acceptance, each module importing cleanly first |
| `test_xyz_to_deepmd.py`, `test_graft_short_range.py`, `test_calibrate_born.py` | the three scripts |
