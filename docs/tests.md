# Tests

```bash
pytest tests/ -q
```

- `tests/test_pme_physics.py` — PME against the NaCl Madelung constant,
  `gradcheck`/`gradgradcheck`, and a guard for the `spline_order >= 7`
  silent-failure mode.
- `tests/test_plugin_registration.py` — entry-point registration, input-schema
  acceptance, equivalence to the standard model, and serialization round-trip.
- `tests/test_compress.py` — compressed DPA4C inference. On any install: the
  `get_geo_compress` seam, every refusal (no kernel, wrong `DP_CUDA_INFER`,
  fallback paths, unpersisted tables), and the graph-lower guard that keeps
  `DeepPot`/`dp test` from returning short-range-only energies. With the
  compiled operator: that the kernel is entered exactly once per forward, and
  compressed-vs-uncompressed agreement for energy, force, virial, charge and
  Born tensors, serial and partitioned. `DEEPMD_LR_REQUIRE_COMPRESS=1` turns
  the kernel skips into failures.
- `tests/test_charge_head.py` — the total-charge projection (exactness,
  masking, differentiability) and the charge head's physical symmetries
  through a real DPA4C descriptor.
- `tests/test_lr_term.py` — the eV unit conversion (against analytic Coulomb
  and Madelung), boundary-condition guards, batching, masking, and gradients
  including the double backward that force training needs.
- `tests/test_joint_model.py` — the combined model: force and virial against
  finite differences of the total energy, energy decomposition, both calling
  conventions, and serialization of the long-range term.
- `tests/test_isolated_term.py`, `tests/test_born.py`,
  `tests/test_multi_task.py`, `tests/test_neighbors.py`,
  `tests/test_xyz_to_deepmd.py` — the non-periodic term, Born charges,
  multi-head sharing, the neighbour-graph swap, and the dataset converter.

The tests import only `deepmd`, never `deepmd_lr`, so they fail if the entry
point stops working rather than silently passing.

