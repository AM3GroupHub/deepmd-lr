# Atomic dipoles and quadrupoles

A point charge per atom is the first term of what an atom contributes to the
long-range field. `long_range.multipole_net` adds the next ones: every atom
gets a dipole, and with `lmax: 2` a traceless quadrupole, which interact with
each other and with the charges in the long-range energy.

```json
"descriptor": {"type": "dpa4", "rcut": 6.0, "n_blocks": 1, "lmax": 3},
"fitting_net": {"type": "dpa4_ener"},
"long_range": {
    "charge_net":    {"neuron": [64, 64]},
    "multipole_net": {"lmax": 2, "neuron": [64, 64]},
    "reciprocal":    {"alpha": 0.4, "mesh_dimensions": [20, 20, 20]}
}
```

Without the block the model is exactly what it was.

## The head

A dipole is a vector and a quadrupole a tensor of the atom's surroundings, so
neither can be fitted on the invariant features the energy and the charge
use. DPA4 already carries what is needed: its node state holds, per atom,
channels that rotate as vectors (`l = 1`) and as traceless tensors (`l = 2`).
The head mixes them,

```
mu_i = 1/C  sum_c a_c(D_i) X1_i,c          Q_i = 1/C  sum_c b_c(D_i) X2_i,c
```

with coefficients `a`, `b` from an ordinary fitting net on the invariant
features `D_i` (its keys are the charge head's). An invariant number times an
equivariant tensor is equivariant, so the moments turn with the frame by
construction.

- **DPA4 only.** The state with every degree is what DPA4 computes and its
  `call_graph` drops; the model keeps it. The last entry of the `l` schedule
  must be at least the head's `lmax`. DPA4C builds such moments too, but its
  compressed kernel never exposes them, and it is refused for now.
- With the default `so3_readout: "none"`, the last block's `l > 0` output
  does not reach the energy head: only this head trains it.
- The short-contact envelope (the charge shield) and `exclude_types` weigh
  the moments as they weigh the charges.

## The energy

The long-range term stays the smooth one (the network learns the short-range
rest): each atom is a Gaussian of the width that makes two charges interact
as `erf(alpha r)/r`, and now carries a dipole and a quadrupole too,

```
E = 2 pi k_e / V  sum_{k != 0} exp(-k^2 / 4 alpha^2) / k^2  |S(k)|^2  -  E_self
S(k) = sum_i ( q_i + i k.mu_i - 1/2 k.Q_i.k ) exp(i k.r_i)
```

The quadrupole is the traceless second moment; its energy in a potential is
`1/2 Q : grad grad phi`. Without moments this is the charge term, number for
number.

- **`method: "pme"` or `"ewald"`**, as for charges. The explicit sum is exact
  to its cutoff and costs atoms times k-vectors; PME is nvalchemiops'
  multipole PME on a mesh.
- **The mesh must be finer than for charges.** A dipole is spread with the
  derivative of the spline and a quadrupole with its second derivative, each
  an order less smooth. Measured on 96 atoms in a tilted 12.4 Å cell, alpha
  0.4, spline order 6, largest force error against the explicit sum:

  | mesh (spacing) | charges | dipoles | quadrupoles |
  |---|---|---|---|
  | 10 (1.24 Å, `1/(2 alpha)`) | 1.3e-3 | 4.3e-3 | 1.2e-3 eV/Å |
  | 16 (0.78 Å) | 9e-5 | 2e-4 | 1.9e-4 |
  | 20 (0.62 Å, `1/(4 alpha)`) | 2e-5 | 5e-5 | 1.1e-4 |
  | 32 (0.39 Å) | 1e-6 | 7e-6 | 2e-5 |

  The MD evaluator and the ASE calculator therefore give a multipole model
  at least `ceil(4 alpha L)` points per cell vector, twice the rule for
  charges. In training the mesh is the config's: write `mesh_dimensions`
  accordingly.
- **A defect in nvalchemiops 0.5.0 is worked around.** Its multipole PME
  builds its k-grid from the cell's inverse transposed the wrong way round:
  right in a cell whose matrix is symmetric (any orthorhombic one), 3e-3 eV
  off on 12 atoms in a tilted one, at any mesh. The grid is built here and
  handed to it; the tests compare PME with the explicit sum in a tilted cell.
- **Periodic in three directions only.** There is no slab correction that
  knows of atomic dipoles and no isolated multipole sum yet; both are refused.

The model reports `dipole` `(nf, nloc, 3)` in e Å and `quadrupole`
`(nf, nloc, 3, 3)` in e Å², beside `charge`; the ASE calculator as
`results["dipoles"]` and `results["quadrupoles"]`. `eps_inf` scales them as
it scales the charges.

## The Born tensor

With `P = sum_j q_j r_j + sum_j mu_j`, the dipoles add their own derivative
to the [Born tensor](born-charges.md):

```
Z_i  +=  sum_j d mu_j / d r_i
```

Three more backward passes; no position stands in it, so it is exact at any
depth of message passing and in a periodic cell, and it sums to zero over
the atoms (the acoustic sum rule stays the total charge). **Quadrupoles add
nothing to `Z`**: a uniform field couples to the dipole moment alone. They
act through the energy and the forces. The Born loss (`pref_bec`) and the
field force work unchanged.

## Training and MD

- The SR → LR → FT curriculum works as before: the graft starts the multipole
  head small, like the charge head, and `train_charge_head_only` leaves both
  heads trainable.
- Partitions, the merged backward and the MD evaluator carry the moments:
  nine numbers per atom cross a partition where one did.
- The virial of a model with tensor moments is symmetric, as it must be:
  turning the atoms at fixed moments and turning the moments are not
  separately symmetries, and the two contributions are combined before it is
  reported.
- Not yet: `charge_same_as` links, DPA4C, slabs, clusters.
