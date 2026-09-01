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
the warm-up run, and the charge head's output layer is scaled down so the
long-range term starts small.

Scaling matters. A freshly initialized charge head predicts random charges,
which inject a large spurious long-range energy and undo the warm start --
in testing it threw the force error from the warm-up value straight back up
to 0.79 eV/A on the first step. Exact zero is not safe either: electrostatic
energy is quadratic in charge, so a zero charge head has exactly zero gradient
and can never leave that stationary point. A small nonzero output preserves
the warm start to leading order while keeping the charge head trainable.

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
DEFAULT_CHARGE_SCALE = 1.0e-2


def _scale_charge_head_output(state: dict, scale: float) -> int:
    """Scale the charge head's output layer while keeping it trainable.

    Only the last layer of each sub-network is scaled, not the whole head.
    Because that layer is affine, this scales every raw charge by the requested
    factor while leaving the hidden representation at a healthy initialization.
    The factor must be strictly positive: zero charge is a stationary point of
    the quadratic electrostatic energy and would freeze the head permanently.
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"charge output scale must be in (0, 1], got {scale}")

    tag = "charge_fitting."
    layer_keys = [
        k for k in state if tag in k and ".layers." in k and k.endswith((".w", ".b"))
    ]
    depth: dict[str, int] = {}
    for key in layer_keys:
        net, _, rest = key.partition(".layers.")
        depth[net] = max(depth.get(net, -1), int(rest.split(".")[0]))

    scaled = 0
    for key in layer_keys:
        net, _, rest = key.partition(".layers.")
        if int(rest.split(".")[0]) == depth[net]:
            state[key] = state[key] * scale
            scaled += 1
    for key in state:
        if key.endswith(tag + "bias_atom_e"):
            state[key] = state[key] * scale
            scaled += 1
    return scaled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--short-range", required=True, help="warm-up checkpoint (.pt)")
    ap.add_argument("--input", required=True, help="the JOINT training input.json")
    ap.add_argument("--out", required=True, help="checkpoint to write")
    ap.add_argument(
        "--keep-charge-init",
        action="store_true",
        help="do NOT scale the charge head; the grafted model then starts with "
        "random charges and a large spurious long-range energy",
    )
    ap.add_argument(
        "--charge-init-scale",
        type=float,
        default=DEFAULT_CHARGE_SCALE,
        help="scale applied to the fresh charge output layer (default: "
        f"{DEFAULT_CHARGE_SCALE:g}); must be in (0, 1]",
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

    scaled = 0
    if not args.keep_charge_init:
        try:
            scaled = _scale_charge_head_output(fresh, args.charge_init_scale)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if scaled == 0:
            raise SystemExit(
                "Refusing to graft: could not find the charge head's output "
                "layer to scale. Pass --keep-charge-init to graft anyway, but "
                "expect a jump in the loss when the long-range term switches on."
            )

    fresh["_extra_state"] = {"model_params": model_params}
    out = pathlib.Path(args.out)
    torch.save({"model": fresh, "optimizer": {}}, out)
    # `dp train --init-model` takes the prefix, i.e. the path without .pt
    print(f"wrote {out}")
    print(f"  {copied} tensors carried over from the warm-up run")
    if scaled:
        print(
            f"  {scaled} charge-head output tensors scaled by "
            f"{args.charge_init_scale:g}, so the long-range term starts small "
            "but remains trainable"
        )
    else:
        print(f"  {kept} left at their fresh initialization (the charge head)")
    print(
        f"\nnow run:\n  dp --pt-expt train {args.input} "
        f"--init-model {str(out).removesuffix('.pt')}"
    )


if __name__ == "__main__":
    main()
