# Results

A controlled periodic benchmark where the joint model beats the strictly local
baseline by 4.9× on force RMSE and breaks through the floor a 6 Å cutoff
imposes; a control showing the gain disappears when there is no long-range
physics to capture; and the charge head recovering the true charge magnitudes
(up to the exact global-sign degeneracy above).

- `experiments/codex_test/` — independent correctness experiments: is the
  electrostatic energy the intended formula, and are the reported derivatives
  actually derivatives of it?
- `experiments/hopper_bench/RESULTS.md` — MD throughput in ns/day across the
  ASE, nvalchemiops and partitioned paths.

## Status

| Phase | | |
|---|---|---|
| 0 | Environment + PME validation | **done** |
| 1 | Plugin skeleton (`dp_lr` model type, no long-range term yet) | **done** |
| 2 | Charge head (`charge` fitting type + total-charge projection) | **done** |
| 3 | Long-range term (`LongRangeEnergy`, reciprocal-only, eV) | **done** |
| 4 | Joint model (`long_range` block, conservative forces) | **done** |
| 5 | Training + validation (benchmark + control) | **done** |
| 6 | MD deployment (ASE calculator, export guard) | **done** |
