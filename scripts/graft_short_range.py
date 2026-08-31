#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graft a trained short-range checkpoint into a long-range model.

Why this is needed
------------------
``dp train --init-model`` loads the state dict strictly, so it refuses a
checkpoint that is missing the charge head's weights -- which is exactly the
situation when warming up on a short-range-only model and then switching the
long-range term on. This writes a checkpoint that *does* have every key the
joint model expects: the shared descriptor and energy-head weights come from
the warm-up run, and the charge head is zeroed so the long-range term starts
at exactly zero.

Zeroing matters. A freshly initialized charge head predicts random charges,
which inject a large spurious long-range energy and undo the warm start --
in testing it threw the force error from the warm-up value straight back up
to 0.79 eV/A on the first step. Zeroing the head's output layer makes the
grafted model *identical* to the warm-up model at step 0, so the long-range
term grows from nothing instead of fighting its way down.

Usage
-----
    python scripts/graft_short_range.py \\
        --short-range run_stage1/model.ckpt.pt \\
        --input       input_joint.json \\
        --out         init_joint.ckpt.pt

    dp --pt-expt train input_joint.json --init-model init_joint.ckpt
"""

import argparse
import copy
import json
import pathlib

import torch
from deepmd.pt_expt.model.get_model import get_model

PREFIX = "model.Default."


def _zero_charge_head(state: dict) -> int:
    """Zero the charge head's output layer so it predicts exactly zero.

    Only the last layer of each sub-network is zeroed, not the whole head:
    that is enough to make the output identically zero while leaving the
    hidden layers at a healthy random initialization, so the head still
    trains normally once gradients start flowing.
    """
    tag = "charge_fitting."
    layer_keys = [
        k for k in state if tag in k and ".layers." in k and k.endswith((".w", ".b"))
    ]
    depth: dict[str, int] = {}
    for key in layer_keys:
        net, _, rest = key.partition(".layers.")
        depth[net] = max(depth.get(net, -1), int(rest.split(".")[0]))

    zeroed = 0
    for key in layer_keys:
        net, _, rest = key.partition(".layers.")
        if int(rest.split(".")[0]) == depth[net]:
            state[key] = torch.zeros_like(state[key])
            zeroed += 1
    for key in state:
        if key.endswith(tag + "bias_atom_e"):
            state[key] = torch.zeros_like(state[key])
            zeroed += 1
    return zeroed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--short-range", required=True, help="warm-up checkpoint (.pt)")
    ap.add_argument("--input", required=True, help="the JOINT training input.json")
    ap.add_argument("--out", required=True, help="checkpoint to write")
    ap.add_argument(
        "--keep-charge-init",
        action="store_true",
        help="do NOT zero the charge head; the grafted model then starts with "
        "random charges and a large spurious long-range energy",
    )
    args = ap.parse_args()

    warm = torch.load(args.short_range, map_location="cpu", weights_only=False)
    warm_state = warm["model"]

    model_params = copy.deepcopy(json.load(open(args.input))["model"])
    joint = get_model(copy.deepcopy(model_params))
    fresh = {PREFIX + k: v for k, v in joint.state_dict().items()}

    copied, kept, mismatched = 0, 0, []
    for key, value in fresh.items():
        if key not in warm_state:
            kept += 1
            continue
        source = warm_state[key]
        if getattr(source, "shape", None) != getattr(value, "shape", None):
            mismatched.append(key)
            kept += 1
            continue
        fresh[key] = source.clone()
        copied += 1

    if mismatched:
        raise SystemExit(
            "Refusing to graft: these tensors exist in both checkpoints with "
            "different shapes, so the two runs do not share an architecture:\n  "
            + "\n  ".join(mismatched[:10])
        )
    if copied == 0:
        raise SystemExit(
            "Refusing to graft: no weights matched. Check that the warm-up run "
            "used the same descriptor and fitting net as the joint input."
        )

    zeroed = 0
    if not args.keep_charge_init:
        zeroed = _zero_charge_head(fresh)
        if zeroed == 0:
            raise SystemExit(
                "Refusing to graft: could not find the charge head's output "
                "layer to zero. Pass --keep-charge-init to graft anyway, but "
                "expect a jump in the loss when the long-range term switches on."
            )

    fresh["_extra_state"] = {"model_params": model_params}
    out = pathlib.Path(args.out)
    torch.save({"model": fresh, "optimizer": {}}, out)
    # `dp train --init-model` takes the prefix, i.e. the path without .pt
    print(f"wrote {out}")
    print(f"  {copied} tensors carried over from the warm-up run")
    if zeroed:
        print(f"  {zeroed} charge-head output tensors zeroed, so the grafted "
              "model starts identical to the warm-up model")
    else:
        print(f"  {kept} left at their fresh initialization (the charge head)")
    print(f"\nnow run:\n  dp --pt-expt train {args.input} "
          f"--init-model {str(out).removesuffix('.pt')}")


if __name__ == "__main__":
    main()
