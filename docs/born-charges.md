# Born effective charges and MD in an external field

A trained `dp_lr` model predicts geometry-dependent charges, and the derivative
of those charges is what a Born effective charge tensor is made of. No extra
training, no extra labels:

```python
calc = LongRangeCalculator(model, efield=[0.0, 0.0, 0.05])   # V/angstrom
atoms.calc = calc
atoms.get_forces()          # includes F_i = Z_i.T E (polarization axis contracted)
calc.results["born_charge"] # (natoms, 3, 3)
```

The tensor computed is

```
Z_iab = q_i d_ab + 1/2 sum_{j!=i} ( r_ji,a dq_i/dr_ji,b  -  r_ij,a dq_j/dr_ij,b )
```

following 10.1021/acs.jctc.6c00146.

### Read this before interpreting a field run

The first axis of `Z_iab` is polarization and the second is displacement,
so the field contraction is `F_ib = sum_a E_a Z_iab`, or `F = Z.T·E`.

**`F = Z.T·E` is not the gradient of `−M·E`.** That is not a bug and not an
approximation that gets better with a smaller timestep.

For a charge model with a total-charge constraint, the exact dipole derivative
of an isolated system is

```
dM_a/dr_ib = q_i d_ab  -  sum_{e: src=i} (r_e (x) g_e)_ab  +  (centroid - r_i)_a S_ib
```

verified against finite differences to **5e-11**. The final term is
proportional to the displacement from the centroid: unbounded, and meaningless
in a periodic cell where `M` itself is undefined because position is not
periodic. The formula above keeps the local, translation-invariant content and
discards that term — which is exactly why it works under PBC at all. On a
compact molecule the two differ by ~50%, and that gap *is* the discarded term.

- **A constant-field NVE run will not conserve `E − M·E`.** Do not read the
  drift as an integrator problem; there is no energy whose gradient this force
  is. Use a thermostat and look at structure and response, not at a conserved
  quantity.
- **No stress is reported under a field.** The field's contribution to the
  stress is a separate derivation, so `LongRangeCalculator` withholds the
  stress entirely rather than returning the field-free value, which would be
  quietly wrong.
- **Net force is exactly `Q·E`** — zero for a neutral cell, so nothing drifts.
  That follows from `sum_i Z_i = Q·I`, which holds identically.

### Implementation notes

**It costs one extra backward pass, not one per atom.** DPA4C has no message
passing, so an atom's charge depends only on edges where it is the centre. A
single VJP therefore yields every per-edge derivative at once. A naive Born
implementation is `O(natoms)` backward passes; this is `O(1)`.

**The derivative uses the raw charges, the `q_i d_ab` term uses the constrained
ones.** `project_to_total_charge` subtracts a global mean, so every charge
depends on every position — moving an atom 32.9 Å away (rcut 6.0) shifts every
distant charge by a uniform amount comparable to a genuine neighbour
derivative. Weighted by an unbounded `r_ji`, those terms would dominate and
grow with system size. Splitting the two roles keeps the sum rule exact while
keeping the tensor local.

**Caveats.** It needs the shared-backbone path: under `DP_CUDA_INFER >= 2` a
fused kernel bypasses it and the model **raises** rather than returning a wrong
tensor, so unset that variable for field simulations. And `Z` is only as
meaningful as the charge head — if `mean|q|` has collapsed toward zero during
training, the Born tensor and the whole field response are meaningless.

It is available on the partitioned evaluator too, either as the tensor alone
or as a field to run under:

```python
ev = DistributedEvaluator(model, atype, cell, devices=[...], compute_born=True)
ev(positions)
ev.results["born_charge"]          # (natoms, 3, 3)

# or: run under a uniform field, in V/angstrom
ev = DistributedEvaluator(model, atype, cell, devices=[...], efield=[0, 0, 0.05])
energy, forces = ev(positions)     # forces include F_ib = E_a Z_iab
ev.results["field_force"]          # the field term alone, (natoms, 3)
```

`efield` implies `compute_born`, and the returned `forces` carry the response
term while `results["force"]` stays the field-free force. Two deliberate
refusals: a model with no `long_range` block has no charges to build `Z` from,
and `compute_virial` cannot be combined with a field, because the field's
contribution to the stress is a separate derivation and reporting the
field-free virial under a field would be quietly wrong.

Verified against the serial path at 1, 2 and 4 partitions, and the net field
force comes out as `Q·E` to 1e-15 — zero for a neutral cell, so nothing
drifts.

The global projection is no obstacle there. It only enters the `q_i delta`
term, which is already projected by the time the partitions are assembled,
while the derivative half uses the raw charges and is strictly local. What the
partitioned assembly does need is a scatter-add: an atom's tensor sums over
edges where it is the *source* as well as the destination, and its source edges
point at neighbours that may live in another partition. Those contributions are
returned to their owners exactly as ghost forces are. Verified to agree with
the serial tensor to 5e-16 at 1, 2 and 4 partitions.

