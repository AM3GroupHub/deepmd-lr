# Tests

Run from the repository root in the [prepared environment](install.md):

```bash
pytest tests/ -q
DP_CUDA_INFER=1 DEEPMD_LR_REQUIRE_COMPRESS=1 pytest tests/ -q
```

Most test modules need CUDA. The second command requires the
[compiled compression operator](compression.md#building-the-cuda-operators)
and turns missing CUDA/operator skips into failures.

## Checks with a trained checkpoint

These workflows are opt-in. Supply a trained water checkpoint; add
`DEEPMD_LR_WATER_COMPRESS=1` to compress it first:

```bash
DEEPMD_LR_WATER_MODEL=model.ckpt.pt pytest tests/test_nvalchemi.py -k long_water -s
DEEPMD_LR_WATER_MODEL=model.ckpt.pt pytest tests/test_md.py -k water_relaxes
```

The first runs 25 ps without wrapping and checks forces every 500 steps
(about 10 minutes). The second relaxes a box stretched by 4% in each cell mode
(about 20 seconds). Timings depend on hardware.

## Physics and model coverage

| tests | checks |
|---|---|
| [lr_term](../tests/test_lr_term.py), [pme_physics](../tests/test_pme_physics.py) | Ewald reference, PME, NaCl Madelung constant, slab correction, gradients and double backward |
| [isolated_term](../tests/test_isolated_term.py) | periodic limit, invariances, differentiability and no-cell inference |
| [joint_model](../tests/test_joint_model.py) | finite-difference forces and virial, shared backbone, training and serialization |
| [charge_head](../tests/test_charge_head.py) | total-charge projection, masking and symmetries |
| [born](../tests/test_born.py), [eps_inf](../tests/test_eps_inf.py) | tensor derivatives, sum rules, field forces and screening on each path |
| [charge_filter](../tests/test_charge_filter.py) | exact zeros, total charge, C³ weights, force gradients and agreement across interfaces |
| [neutrality_loss](../tests/test_neutrality_loss.py) | `ener_lr` penalty, schedule, gradients, schema and serialization |
| [multi_task](../tests/test_multi_task.py), [same_as](../tests/test_same_as.py), [charge_same_as](../tests/test_charge_same_as.py) | branch expansion and parameter sharing |

## Inference and MD coverage

| tests | checks |
|---|---|
| [neighbors](../tests/test_neighbors.py) | fallback cell-list graph against the dense builder |
| [nlist](../tests/test_nlist.py) | GPU graph, periodic images, small/triclinic/slab cells, cell changes, CSR, overflow recovery, gradients and deterministic Born reduction |
| [nvalchemi](../tests/test_nvalchemi.py) | partition ownership, halos, skin, cell deformation, memory probing, gathered quantities and agreement with fresh evaluation |
| [sync](../tests/test_sync.py) | host-transfer counts, no pageable uploads, `nonzero` or `bincount` in the warm path |
| [thermo](../tests/test_thermo.py) | measurements, Warp views, cell updates, evaluator sharing and fixed atoms |
| [md](../tests/test_md.py) | integrators and the three relaxations against kernel loops, conservation, cell conventions and the cell gradient (L-BFGS chart included), stage sharing, checkpoints, observers and constraints |
| [compress](../tests/test_compress.py) | requirements, failed-compression rollback, kernel dispatch and agreement with uncompressed evaluation |
| [ase_calculator](../tests/test_ase_calculator.py) | model agreement, boundaries, export guards, NVE drift, mesh selection and head loading |
| [read_structure](../tests/test_read_structure.py) | periodicity, slab padding and centring, fixed atoms and unsupported structures |

## Configuration and tools

| tests | checks |
|---|---|
| [ema_default](../tests/test_ema_default.py) | EMA training defaults |
| [plugin_registration](../tests/test_plugin_registration.py), [import_order](../tests/test_import_order.py) | entry point, schema and independent module imports |
| [xyz_to_deepmd](../tests/test_xyz_to_deepmd.py) | dataset conversion |
| [graft_short_range](../tests/test_graft_short_range.py) | checkpoint grafting |
| [calibrate_born](../tests/test_calibrate_born.py) | signed scale fitting and calibration CLI |
