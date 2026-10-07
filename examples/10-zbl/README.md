# ZBL repulsion for DPA4C and DPA4C-LR

Both variants use the plugin's `model.type: "dp_lr"` construction path with a
`dpa4c` descriptor. Omit `long_range` for a pure short-range DPA4C model; retain
it for DPA4C-LR. The built-in `standard` model does not consume this plugin's
`zbl` block. This example does not change that upstream registry.

[`input.json`](input.json) is the bulk-water input of
[example 01](../01-bulk) with one addition inside `model`:

```json
"zbl": {"r_inner": 0.5, "r_outer": 0.8}
```

Distances are in Angstrom. An empty `"zbl": {}` block enables the same
defaults: `cutoff: "fixed"`, `r_inner=0.5`, `r_outer=0.8`. Either radius can
be overridden independently; explicit values in existing inputs and
checkpoints are preserved. Require `0 < r_inner < r_outer <= descriptor.rcut`.
Omit the block to disable ZBL.

### Where to put the switch

Two conventions exist, and they answer different questions:

- **`cutoff: "fixed"`** (the default), one `r_inner`/`r_outer` for every
  pair, is DPA4's bridging convention and `0.5/0.8` are its defaults. DPA4
  pairs those radii with a *descriptor clamp* that blinds the network below
  `r_outer`, so ZBL alone governs there. This plugin has no clamp -- the
  network stays active -- and with radii this short the term is inert at
  ordinary bond lengths: for Si-C (bond 1.89 A), C-C (1.54) or O-H (0.97)
  nothing comes within 0.8 A, and the regime that wrecks a trajectory, a
  pair at 1.0-1.4 A the data never sampled, is left to the network. Water
  cannot go higher (O-H sits at 0.97, the data reaches 0.93); Si/diamond
  can take `r_outer` ~ 1.2 A, the floor of its data's C-C distances.
- **`cutoff: "covalent"`** is MACE's convention: per pair,
  `r_outer = covalent_scale * (R_i + R_j)` with ASE's covalent radii
  (Si 1.11, C 0.76, O 0.66, H 0.31 A) and `r_inner = inner_ratio * r_outer`
  (defaults 1.0 and 0.5). The wall reaches each bond and training absorbs
  what it adds there -- a few eV per bond with the standard screening, which
  MACE softens (its screening length is about half the ZBL one; this
  implementation keeps the standard value). Expect the baseline twin to be
  needed to see what it costs in accuracy.

```json
"zbl": {"cutoff": "covalent", "covalent_scale": 1.0, "inner_ratio": 0.5}
```

Whichever you choose, validate against close-contact data, normal bond
distances and the intended collision energies.

### The clamp, and why ZBL alone does not make a wall

`clamp: true` (the schema default for a new input) shows the descriptor
nothing below the switch: a pair closer than `r_outer` looks to the network
like one at `r_outer`, and the true length returns over `clamp_width` beyond
it (default: the pair's own `r_outer - r_inner`) through a C^3 septic remap. The charge head
reads the same descriptor, so it is frozen too. Below `r_outer` the learned
energy and the charges are constant in that pair's distance and ZBL alone
carries it inward.

Without the clamp the term is a prior and nothing more. Compressing one O-H
pair in a liquid-water frame, both the plain water model and its ZBL twin run
to -10^3 .. -10^5 eV below 0.75 A -- the DPA4C features and the charge head
extrapolate there with arbitrary sign -- and a few eV of ZBL do not register.
Freezing at `r_inner` instead, DPA4's choice, does not help this descriptor:
the network is still shown lengths in `[r_inner, r_outer]` it never saw, and
the same model gave -3600 eV at 0.8 A that way. Hence the freeze at `r_outer`.

Two consequences for choosing radii with the clamp on. `r_outer` should sit
at or below the data's closest distance for that pair, or the model is
trained on remapped geometry there (the covalent cutoff puts `r_outer` at the
bond, so the clamp blinds the network to bond compression: use fixed radii
with the clamp). And `r_outer + clamp_width` must stay below `descriptor.rcut`.
`tests/test_zbl.py::TestClamp` pins the frozen energy, the frozen charges and
the finite-difference forces through the remap.

A checkpoint written before the key existed is reproduced with the clamp off.
`DP_CUDA_INFER=2` is refused with the clamp: that fused pipeline never calls
the descriptor the clamp wraps.

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

The directed graph contributes half the pair energy to each destination,
and a ghost destination's half belongs to the rank that owns it. The term has
no parameters and no network between it and the edge vectors, so its forces
and virial are assembled analytically from the per-edge derivative -- no
second autograd pass -- through the same edge-to-node reduction DeePMD uses
for the learned energy (its fused CUDA kernel under `DP_CUDA_INFER>=1`). On
a CUDA device one Warp kernel produces the per-edge energy and derivative in
a single pass; on the CPU, and inside a trace (export, compiled training),
the same arithmetic runs as torch ops. Measured on a 33k-atom SiC slab,
compressed, one GB10: 54.9 ms/step without the term, 59.0 with it (either
cutoff -- the radius does not change the work), against 156 ms for the
autograd version this replaced.

Atom/pair exclusions and non-DPA4C descriptors are rejected when ZBL is enabled.
Dense-lower and compact canonical compressed export are rejected rather than
silently dropping ZBL, so a ZBL model cannot be exported for LAMMPS; LAMMPS's
own `pair_style zbl` under `hybrid/overlay` is not a substitute, since its
switching polynomial differs from this one. Ordinary graph inference remains
available. Existing
LR graph-export restrictions remain in force. Extending this implementation to
DPA4-style descriptor clamping or table blending requires a separate design.

`tests/test_zbl.py` and `tests/test_zbl_gpu.py` pin all of this; see
[the test map](../../docs/tests.md).
