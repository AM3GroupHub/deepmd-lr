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
| Several GPUs, or `evaluate_partitioned` with a halo | refused for now |
| Born effective charges, an external field | refused for now |
| The `zbl` block | refused: DPA4C only |
| Compression | not available: DeePMD-kit has no compressed DPA4 |
| A shared backbone across heads (`shared_dict`, `charge_same_as`) | not available: DeePMD-kit's DPA4 does not share parameters yet |
| Export of a long-range model | refused, as with DPA4C |

## Why some paths are refused

DPA4C scans the edges into an atom once, so everything an atom carries
depends on its own neighbours. DPA4 passes messages: an atom's energy and
charge reach `n_blocks` neighbour shells, and one more with `use_env_seed`
(the default), which is 24 Å at the default three blocks and 6 Å cutoff.
`deepmd_lr.model.lr_model.receptive_hops(model)` returns that number of
shells; a test moves one atom of a chain and checks that the reach is exactly
that and no further.

Two parts of this package were written for one shell and would return wrong
numbers on DPA4 without an error, so they raise `NotImplementedError` instead:

- **Partitions with a halo.** A partition's halo is one cutoff wide and holds
  only the edges into its own atoms. Atoms near a partition face would see an
  incomplete neighbourhood from the second block on.
- **The Born tensor.** Its pair formula assigns each edge's charge derivative
  to the two atoms of that edge, which holds only when a charge depends on the
  edges into its own atom.

On one device the system is not partitioned (its periodic images are in the
neighbour list itself), so energies, forces, virials and charges are exact for
any depth.

## Checks

`tests/test_dpa4.py`, with random weights in float64:

- the short-range force equals the energy gradient to 1e-10 eV/Å, and the
  long-range model's to the accuracy of the reciprocal sum (below 1e-6 eV/Å);
- the virial against a volume change;
- a training step with a force loss reaches the descriptor, both fitting nets
  and the charge head;
- the one-device evaluator equals the model to 1e-8 eV and 1e-8 eV/Å.
