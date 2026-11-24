# DPA4 as the backbone

`dp_lr` also runs on DPA4, DeePMD-kit's message-passing descriptor, for
systems DPA4C does not describe well enough. Only the descriptor block
changes:

```json
{
  "model": {
    "type": "dp_lr",
    "type_map": ["O", "H"],
    "descriptor": { "type": "dpa4", "rcut": 6.0, "n_blocks": 3 },
    "fitting_net": { "type": "dpa4_ener", "neuron": [0] },
    "long_range": {
      "charge_net": { "neuron": [64, 64] },
      "reciprocal": { "alpha": 0.4, "mesh_dimensions": [16, 16, 16] }
    }
  }
}
```

`fitting_net.type` may be `ener` or `dpa4_ener`, the energy head DeePMD-kit
pairs with DPA4. The charge head reads the descriptor's invariant per-atom
features, as it does with DPA4C, and the same backbone pass serves both heads.

## What works

| | DPA4 |
|---|---|
| Training with `dp --pt-expt train`, restart, the `ener_lr` loss | yes |
| Energy, force, virial, charges; the charge filter; `charge_spin` | yes |
| ASE calculator | yes |
| MD and relaxation on one GPU (`DistributedEvaluator` with one device) | yes |
| `same_as` links | yes |
| Several GPUs (`DistributedEvaluator`, `evaluate_partitioned`) | yes, with a [deeper halo](#several-gpus) |
| Born effective charges, an external field | refused for now |
| Short contacts: `bridging_method` with a shield on the charges | yes, [below](#short-contacts) |
| The `zbl` block | refused: DPA4C only |
| Compression | not available: DeePMD-kit has no compressed DPA4 |
| A shared backbone across heads (`shared_dict`, `charge_same_as`) | not available: DeePMD-kit's DPA4 does not share parameters yet |
| Export of a long-range model | refused, as with DPA4C |

## Short contacts

DPA4 has DeePMD-kit's own short-contact treatment, and `dp_lr` takes its keys
as a `standard` model does:

```json
{
  "model": {
    "type": "dp_lr",
    "descriptor": { "type": "dpa4", "rcut": 6.0 },
    "fitting_net": { "type": "dpa4_ener" },
    "bridging_method": "ZBL",
    "bridging_r_inner": 0.5,
    "bridging_r_outer": 0.8,
    "long_range": { "charge_net": {}, "reciprocal": { "alpha": 0.4 } }
  }
}
```

- **The energy is DeePMD-kit's.** It adds the ZBL pair repulsion over every
  pair inside the cutoff and freezes the distances the descriptor sees below
  `bridging_r_inner` (in Å), so the network is fitted to what ZBL leaves.
  The model is built by DeePMD-kit's own builder and, without a `long_range`
  block, returns its numbers exactly. It needs `fitting_net.type: dpa4_ener`.
- **The charges are shielded here.** An atom's charge is weighted by the
  product over its neighbours of a C2 step that is 0 at `bridging_r_inner` and
  1 at `bridging_r_outer`. The weight enters the total-charge projection, so
  two atoms pushed together carry no charge and the others carry the total.
  Beyond `bridging_r_outer` nothing changes.
- `long_range.charge_shield: [r_lo, r_hi]` chooses other radii for the shield,
  and works without `bridging_method` too.

DeePMD-kit recommends setting `training.training_data.min_pair_dist` to
`bridging_r_inner`, so frames with closer pairs are skipped in training.

## How far an atom sees

DPA4C scans the edges into an atom once, so everything an atom carries
depends on its own neighbours. DPA4 passes messages: an atom's energy and
charge reach `n_blocks` neighbour shells, and one more with `use_env_seed`
(the default), which is 24 Å at the default three blocks and 6 Å cutoff.
`deepmd_lr.model.lr_model.receptive_hops(model)` returns that number of
shells; a test moves one atom of a chain and checks that the reach is exactly
that and no further.

## Several GPUs

A partition's halo is as many cutoffs deep as the descriptor reaches, and its
graph also holds the edges into the ghosts whose state an owned atom reads
(those within one cutoff less). An owned atom then sees in its partition
exactly what it sees in the whole system, with no exchange between partitions
during a step; energies, forces, the virial and charges equal the undivided
model's to rounding.

The price is the halo. With three blocks and a 6 Å cutoff it is 24 Å on each
face, against 6 Å for DPA4C: a slab-shaped partition 50 Å thick holds about
twice its own atoms as ghosts. Several GPUs pay off only when the partitions
are much thicker than the halo; fewer blocks or a shorter cutoff shrink it in
proportion. On one device there is no halo at all.

## The Born tensor is refused for now

Its pair formula assigns each edge's charge derivative to the two atoms of
that edge, which holds only when a charge depends on the edges into its own
atom. On DPA4 it would return wrong numbers without an error, so
`compute_born` and an external field raise `NotImplementedError`.

## Checks

`tests/test_dpa4.py`, with random weights in float64:

- the short-range force equals the energy gradient to 1e-10 eV/Å, and the
  long-range model's to the accuracy of the reciprocal sum (below 1e-6 eV/Å);
- the virial against a volume change;
- a training step with a force loss reaches the descriptor, both fitting nets
  and the charge head;
- the one-device evaluator equals the model to 1e-8 eV and 1e-8 eV/Å;
- partitions with a deep halo (1, 2 and 4, one to three blocks) equal the
  serial model to 1e-8, and a halo is reused while atoms move within the skin;
- with bridging: the model without a long-range term equals DeePMD-kit's
  bridged model, both atoms of a contact below the window carry exactly zero
  charge with the total conserved, and the force is the energy gradient
  through the window.
