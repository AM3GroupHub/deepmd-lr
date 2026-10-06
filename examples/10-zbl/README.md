# Fixed ZBL repulsion for DPA4C and DPA4C-LR

Both variants use the plugin's `model.type: "dp_lr"` construction path with a
`dpa4c` descriptor. Omit `long_range` for a pure short-range DPA4C model; retain
it for DPA4C-LR. The built-in `standard` model does not consume this plugin's
`zbl` block. This example does not change that upstream registry.

Add the following block inside `model` in a working training input:

```json
"zbl": {"r_inner": 0.5, "r_outer": 0.8}
```

Distances are in Angstrom. An empty `"zbl": {}` block enables the same
defaults: `r_inner=0.5`, `r_outer=0.8`. Either radius can be overridden
independently; explicit values in existing inputs and checkpoints are preserved.
These radii match DeePMD DPA4 bridging defaults; the coupling method remains
additive and is not equivalent to DPA4 bridging. They are not calibrated
Si/C or universal radii. Choose and validate them against close-contact data,
normal bond distances and the intended collision energies. Require
`0 < r_inner < r_outer <= descriptor.rcut`. Omit the block to disable ZBL.

The energy is `E_learned + E_ZBL` (DPA4C), or
`E_learned + E_ZBL + E_long_range` (DPA4C-LR). ZBL uses the element's atomic
number, not the learned charge. It adds no trainable parameters. Its settings
survive checkpoint/EMA load, restart and model serialization; the element
lookup is reconstructed from the stored type map.

This is an **additive prior**, not DeePMD's `use_srtab` interpolation and not
DPA4's Zone/descriptor-clamping method. Do not set `use_srtab` together with
this block. The existing charge network, projection/filter and reciprocal
energy remain unchanged for fixed weights. Training changes those weights
through the total E/F loss as usual. `energy_short_range` includes the ZBL term.

## Training and model use

1. Add `zbl` before training so the network learns with its contribution present.
   The supplied E/F labels remain the total reference values; do not subtract
   ZBL from the data as well.
2. Run `dp --pt-expt train input.json` as for the existing model. Standard energy
   loss and the LR `ener_lr` loss retain their existing meanings. ZBL has no
   separate loss or optimization settings. An existing checkpoint trained without
   ZBL is a different potential; fine-tuning with ZBL requires the normal
   initialization workflow and validation, not just toggling an MD option.
3. Load the resulting checkpoint with `LongRangeCalculator` or
   `DistributedEvaluator`. Eager graph inference and graph compression retain ZBL.
   The same neighbour graph supplies its pair distances; no separate search or
   CPU geometry transfer is introduced.
4. Validate E/F/virial on compressed and uncompressed models, near both switching
   radii and on close contacts. Recheck MD timestep and energy drift for the
   intended temperature/collision regime. ZBL does not guarantee extrapolation
   accuracy, nor repair an incorrect long-range charge model.

`same_as` branches inherit the base branch's ZBL configuration. Independent heads
can specify their own configuration. A pure DPA4C baseline that retains ZBL must
also retain `type: "dp_lr"`; remove only `long_range`.

## Implementation and limits

For each pair, use the screened nuclear potential documented in
[LAMMPS ZBL](https://docs.lammps.org/pair_zbl.html), with screening length
`0.46850 / (Z_i**0.23 + Z_j**0.23)` Angstrom and Coulomb constant
14.3996454784255 eV Angstrom. Inside `r_inner`, the full repulsion applies.
Between the radii multiply it by `1 - 10*t**3 + 15*t**4 - 6*t**5`, where
`t = (r-r_inner)/(r_outer-r_inner)`; beyond `r_outer` it is zero. This switching
convention is not LAMMPS's switching polynomial. Energy, force and curvature
are continuous at the endpoints. Exact overlapping real atoms remain singular.

The directed graph contributes half the pair energy to each destination.
DeePMD's edge derivatives and owned/ghost reductions assemble conservative forces
and virials. The current implementation adds a separate tensor evaluation and
edge-derivative pass; it is not a fused ZBL CUDA kernel, and no MD throughput
percentage is claimed.

Atom/pair exclusions and non-DPA4C descriptors are rejected when ZBL is enabled.
Dense-lower and compact canonical compressed export are rejected rather than
silently dropping ZBL. Ordinary graph inference remains available. Existing
LR graph-export restrictions remain in force. Extending this implementation to
DPA4-style descriptor clamping or table blending requires a separate design.

Validation commands and results: [ZBL validation](../../docs/zbl-validation.md).
