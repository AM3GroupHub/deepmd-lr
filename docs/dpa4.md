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
| Staged training, SR → LR → FT ([training](training.md#plan-b-sr--lr--ft-staged)); freeze with `long_range.train_charge_head_only`, since `dpa4_ener` ignores `trainable: false` | yes |
| Energy, force, virial, charges; the charge filter; `charge_spin` | yes |
| ASE calculator | yes |
| MD and relaxation on one GPU (`DistributedEvaluator` with one device) | yes |
| `same_as` links | yes |
| Several GPUs (`DistributedEvaluator`, `evaluate_partitioned`) | for `n_blocks: 1` only, with a [deeper halo](#several-gpus) |
| Born effective charges, an external field | yes, [in the pair form](#the-born-tensor) |
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
that and no further. With `bridging_method` a neighbour's message is also
gated by that neighbour's own short contacts, which adds `bridging_r_outer`
(0.8 Å by default) to the reach, not another shell
(`deepmd_lr.model.lr_model.gate_margin`).

## Several GPUs

Several devices are supported for a DPA4 of **one interaction block**. With
more blocks the evaluator and `evaluate_partitioned` raise
`NotImplementedError`; run such a model on one device.

A partition's halo is as many cutoffs deep as the descriptor reaches (plus
`bridging_r_outer` with bridging), and its graph also holds the edges into
the ghosts whose state or gate an owned atom reads. An owned atom then sees in its partition
exactly what it sees in the whole system, with no exchange between partitions
during a step; energies, forces, the virial, charges and Born tensors equal
the undivided model's to rounding.

The halo is the reason for the limit. One block reaches two cutoffs with
`use_env_seed` (the default) and one without: a 12 Å or 6 Å halo at a 6 Å
cutoff. Three blocks would need 24 Å on every face, where a slab-shaped
partition 50 Å thick holds about twice its own atoms as ghosts. On one device
there is no halo at all, whatever the depth.

## The Born tensor

The tensor keeps the pair form it has on DPA4C,
`Z_i = q_i 1 + ½ Σ_j r_ji ⊗ (∂q/∂r_ji)`, summed with opposite sign at the
pair's other atom, from one backward pass of the charges through the edge
vectors. On DPA4C only the pair's own charges depend on `r_ji`. On DPA4 every
charge within reach does, and the derivative is the sum over all of them:
each pair vector's whole effect on the charges is shared by the pair's two
atoms. A third atom's response therefore enters with the pair's lever arm,
not with its own position. The sum of the tensors over the atoms is the total
charge exactly, as before, and the cost stays one backward pass.

Serial, partitioned and one-device evaluation return the same tensors; an
external field uses them as with DPA4C ([Born charges](born-charges.md)).

## Speed at inference

DPA4 has no compression, but DeePMD-kit ships Triton kernels for its SO(2)
blocks, switched on by an environment variable read when the model is built:

```bash
DP_TRITON_INFER=2 python run_md.py ...
```

They pass through the plugin unchanged. Measured on 1,536 atoms of water (one
block, 64 channels, `lmax` 3, GB10, GPU shared with other jobs):

| `DP_TRITON_INFER` | step | peak memory | forces against level 0 |
|---|---|---|---|
| 0 | 2.6–2.9 s | 10.1 GB | – |
| 1 | 1.9 s | 6.9 GB | 5e-6 eV/Å at most |
| 2 | 0.85–0.94 s | 6.4 GB | 5e-6 eV/Å at most |

The first call compiles the kernels (some 10 s more, once per shape).

Those steps took two backward passes through the descriptor: DeePMD-kit's
graph lower differentiates the short-range energy itself, and the
charge-response force needs the descriptor's derivative again once the
long-range term has given `dE/dq`. The MD evaluator now takes both in one
pass: each partition keeps its atomic energies attached and differentiates
`sum_i E_i + sum_j dE/dq_j raw_j` once (`evaluator.merge_backward`, on by
default). Same system, same shared GPU:

| `DP_TRITON_INFER` | two backwards | one |
|---|---|---|
| 0 | 2.84 s | 1.70 s |
| 2 | 0.89–0.99 s | 0.59–0.62 s |

The total force and virial are those of the two passes to round-off; the
charge response is no longer reported on its own
(`results["force_charge_response"]`), so set `evaluator.merge_backward =
False` to see it. Two passes remain where DeePMD-kit's lower is not the plain
autograd one (a compressed DPA4C, `DP_CUDA_INFER >= 2`, a `zbl` block) and
when Born tensors are computed.

`DP_AMP_INFER=1` (bf16 in the blocks) is not recommended: it moved the forces
of that model by 0.0017 eV/Å on average and 0.008 at most, a quarter of the
model's own error, for a smaller gain, and together with `DP_TRITON_INFER` it
fails inside DeePMD-kit's kernels (3.2.0).

## Checks

`tests/test_dpa4.py`, with random weights in float64:

- the short-range force equals the energy gradient to 1e-10 eV/Å, and the
  long-range model's to the accuracy of the reciprocal sum (below 1e-6 eV/Å);
- the virial against a volume change;
- a training step with a force loss reaches the descriptor, both fitting nets
  and the charge head;
- the one-device evaluator equals the model to 1e-8 eV and 1e-8 eV/Å;
- for one block, partitions with a deep halo (1, 2 and 4) equal the serial
  model to 1e-8, and a halo is reused while atoms move within the skin; two
  and three blocks are refused on several devices and run on one;
- the Born tensor equals the pair formula assembled from finite differences
  of the charges against every edge vector (1e-6), reduces to the one-hop
  formula for one block without a seed, and is the same on one device and,
  for one block, on 1, 2 and 4 partitions (1e-8);
- with bridging: the model without a long-range term equals DeePMD-kit's
  bridged model, both atoms of a contact below the window carry exactly zero
  charge with the total conserved, and the force is the energy gradient
  through the window.
