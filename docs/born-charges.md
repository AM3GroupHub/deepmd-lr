# Born effective charges and MD in an external field

A trained `dp_lr` model predicts geometry-dependent charges, and the
derivative of those charges is what a Born effective charge tensor is made of.
Raw tensors need no extra training labels; comparison with DFPT can use the
[calibration below](#calibrating-the-scale-and-sign-against-dfpt-labels):

```python
calc = LongRangeCalculator(model, efield=[0.0, 0.0, 0.05])   # V/angstrom
atoms.calc = calc
atoms.get_forces()          # includes F_i = Z_i.T E (polarization axis contracted)
calc.results["born_charge"] # (natoms, 3, 3)
```

The tensor, following 10.1021/acs.jctc.6c00146, is

```
Z_iab = q_i d_ab + 1/2 sum_{j!=i} ( r_ji,a dq_i/dr_ji,b  -  r_ij,a dq_j/dr_ij,b )
```

The first axis is polarization and the second displacement, so the field
force is `F_ib = sum_a E_a Z_iab`, or `F = Z.T·E`. Its net is exactly `Q·E`,
zero for a neutral cell, because `sum_i Z_i = Q·I` holds identically.

### Electronic screening: `eps_inf`

The model's Coulomb term is `q_i q_j / (4π ε0 r_ij)`. In a medium with
high-frequency dielectric constant ε∞ the same energy reads
`q~_i q~_j / (4π ε0 ε∞ r_ij)`, so the physical, screened charges are
`q~ = sqrt(ε∞) q` and the Born tensors scale by the same factor. Pass
`eps_inf` to get them reported that way:

```python
calc = LongRangeCalculator(model, efield=[0, 0, 0.05], eps_inf=1.78)   # water
ev = DistributedEvaluator(model, atype, cell, devices=[...], efield=[0, 0, 0.05], eps_inf=1.78)
evaluate_partitioned(model, coord, atype, cell, compute_born=True, eps_inf=1.78)
born_charges(charge, raw, edge_vec, edge_index, edge_mask, eps_inf=1.78)
```

Scaled by `sqrt(eps_inf)`: charges (`results["charges"]` in ASE,
`results["charge"]` in the evaluator), `results["born_charge"]`,
hence the field force `F = Z~.T E` and anything built from `Z~`, such as the
IR dipole rate `Ṁ = Σ_i Z~_i v_i`. Not scaled: the energy, the field-free
forces, the stress, and the model itself (tables, checkpoint, training), since
this rescales the reported response without changing the fitted interaction.
The default `1.0` leaves it unscaled; non-positive or non-finite values
are refused. A compressed model is unaffected, as the scaling comes after the
tables.

### Calibrating the scale and sign against DFPT labels

Training does not supervise Born tensors. Fit a signed scalar to DFPT labels
after training, then assess the remaining error on held-out data:
`s = sum(Z_raw * Z_dft) / sum(Z_raw * Z_raw)`.

```bash
python scripts/calibrate_born.py model.pt valid_BEC.xyz test_BEC.xyz
```

The scale is fitted on the first file and the second is scored at it; atoms
carry the DFPT tensor as nine numbers under `BEC`, laid out
`[polarization, displacement]` like `results["born_charge"]`. For the water
model `s = 1.754`, so `eps_inf = 3.08` where DFPT's own electronic dielectric
constant is 1.84, at a test RMSE of 0.049 e. This fitted scale need not equal
the independently computed dielectric constant.

The script reports `s`, `eps_inf = s²`, the sign and errors; it does not modify
the checkpoint. For positive `s`, use `eps_inf=s*s` in later field runs.
**For negative `s`, `eps_inf` applies only the magnitude:** the APIs multiply
by the positive square root and do not flip the charge branch. The script's
reported errors use the full signed tensor `s * Z_raw`; using `eps_inf` alone
will not reproduce that calibration when `s < 0`.

### Read this before interpreting a field run

**`F = Z.T·E` is not the gradient of `−M·E`**, and no timestep makes it one.
For a charge model with a total-charge constraint, the exact dipole derivative
of an isolated system is

```
dM_a/dr_ib = q_i d_ab  -  sum_{e: src=i} (r_e (x) g_e)_ab  +  (centroid - r_i)_a S_ib
```

verified against finite differences to 5e-11. The last term is proportional to
the displacement from the centroid: unbounded, and meaningless in a periodic
cell where `M` itself is undefined. The formula above keeps the local,
translation-invariant content and discards it, which is why it works under PBC
at all; on a compact molecule the two differ by ~50%, and that gap is the
discarded term. So:

- **A constant-field NVE run will not conserve `E − M·E`.** There is no energy
  whose gradient this force is. Use a thermostat and look at structure and
  response, not at a conserved quantity.
- **No stress is reported under a field.** The field's contribution to the
  stress is a separate derivation, so the stress is withheld rather than the
  field-free value returned. For the same reason `compute_virial` cannot be
  combined with a field on the partitioned evaluator.

### Implementation notes

**One extra backward pass, not one per atom.** DPA4C has no message passing,
so an atom's charge depends only on edges where it is the centre, and a single
VJP yields every per-edge derivative at once.

**The derivative uses the raw charges, the `q_i d_ab` term the constrained
ones.** `project_to_total_charge` subtracts a global mean, so every projected
charge depends on every position: moving an atom 32.9 Å away (rcut 6.0) shifts
every distant charge by a uniform amount comparable to a genuine neighbour
derivative, and weighted by an unbounded `r_ji` those terms would dominate and
grow with system size. Splitting the roles keeps the sum rule exact and the
tensor local. The same split makes the partitioned assembly exact: the
projected term is assembled after the gather, the derivative half is strictly
local, and an atom's source-edge contributions from neighbours in other
partitions come back to it as ghost forces do. Verified against the serial
tensor to 5e-16 at 1, 2 and 4 partitions, with the net field force `Q·E` to
1e-15.

**On the GPU-direct list the Born pass is one kernel, and its backward is
shared with the charge response.** Both start from the same VJP of the raw
charges onto the edge vectors; because the descriptor is one-hop, the weighted
cotangent the charge-response force needs is that gradient times each edge's
destination weight, so `_charge_response_force` reuses it instead of a second
backward through the descriptor. The edge half, `1/2 (sum_in − sum_out) r ⊗ g`,
is summed along the destination and source CSR rows by a Warp kernel
(`deepmd_lr/nvalchemi/born_csr.py`): one thread per node, float64
accumulation, no `(E, 3, 3)` temporary, no atomics, deterministic. The
array-API `edge_born_response` remains the path for the uncompressed model and
the oracle the kernel is tested against. GB10, 169k water atoms, compressed,
one device: the long-range evaluation costs 104 ms, with Born tensors 120 ms
(138 ms before the kernel).

**Caveats.** It needs the shared-backbone path: under `DP_CUDA_INFER >= 2` the
model raises rather than return a wrong tensor, so unset that variable for
field simulations. And `Z` is only as meaningful as the charge head: if
`mean|q|` collapsed toward zero in training, the whole field response is
meaningless.

On the partitioned evaluator, as the tensor alone or as a field to run under:

```python
ev = DistributedEvaluator(model, atype, cell, devices=[...], compute_born=True)
ev(positions)
ev.results["born_charge"]          # (natoms, 3, 3)

# or: run under a uniform field, in V/angstrom
ev = DistributedEvaluator(model, atype, cell, devices=[...], efield=[0, 0, 0.05])
energy, forces = ev(positions)     # forces include F_ib = E_a Z_iab
ev.results["field_force"]          # the field term alone, (natoms, 3)
```

`efield` implies `compute_born`; the returned `forces` carry the response term
while `results["force"]` stays the field-free force. A model with no
`long_range` block has no charges to build `Z` from and is refused.
