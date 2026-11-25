# SPDX-License-Identifier: MIT
"""The SR -> LR -> FT curriculum: a short-range model, then the charge head
alone on that model frozen, then everything together.

The checks are on the weights each stage moves, through ``dp train`` itself,
because that is where a freeze can fail without a sign: a fitting net that
ignores its ``trainable`` key trains on quietly.
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model

from deepmd_lr.model.lr_model import backbone

from .test_joint_model import frame

REPO = Path(__file__).resolve().parents[1]
CHARGE = "charge_fitting."

DPA4C = {
    "descriptor": {
        "type": "dpa4c",
        "rcut": 4.0,
        "channels": 8,
        "lmax": 2,
        "n_radial": 8,
        "radial_modes": 2,
        "precision": "float32",
        "seed": 1,
    },
    "fitting_net": {"neuron": [16, 16], "precision": "float32", "seed": 1},
}
DPA4 = {
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        # No `sel`: training takes it from the neighbour statistics, and the
        # graft has to build the same model from an input without it.
        "channels": 16,
        "lmax": 2,
        "n_blocks": 1,
        "n_radial": 8,
        "precision": "float32",
        "seed": 1,
    },
    # The net DeePMD-kit pairs with DPA4, and the one that trains although
    # its `trainable` key is false.
    "fitting_net": {"type": "dpa4_ener", "neuron": [0], "seed": 1},
}
BRIDGING = {
    "bridging_method": "ZBL",
    "bridging_r_inner": 0.5,
    "bridging_r_outer": 0.8,
}
SHORT_RANGE = {
    "dpa4c": DPA4C,
    "dpa4": DPA4,
    "dpa4_bridged": {**DPA4, **BRIDGING},
}
LONG_RANGE = {
    "charge_net": {"neuron": [16, 16], "seed": 2},
    "reciprocal": {"alpha": 0.4, "mesh_dimensions": [12, 12, 12]},
}


def short_range(kind: str) -> dict:
    return {
        "type": "standard",
        "type_map": ["O", "H"],
        **copy.deepcopy(SHORT_RANGE[kind]),
    }


def long_range(kind: str, charge_head_only: bool) -> dict:
    block = copy.deepcopy(LONG_RANGE)
    if charge_head_only:
        block["train_charge_head_only"] = True
    return {**short_range(kind), "type": "dp_lr", "long_range": block}


@pytest.mark.parametrize("kind", list(SHORT_RANGE))
def test_only_the_charge_head_is_left_trainable(kind):
    def built(charge_head_only: bool):
        cfg = long_range(kind, charge_head_only)
        if cfg["descriptor"]["type"] == "dpa4":
            cfg["descriptor"]["sel"] = 40  # `dp train` would fill it in
        return get_model(cfg)

    model = built(charge_head_only=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(name.startswith(CHARGE) for name in trainable)
    assert not any(p.requires_grad for p in model.atomic_model.parameters())
    # Without the key the learned model trains.
    free = built(charge_head_only=False)
    assert all(p.requires_grad for p in backbone(free).parameters())


def write_dataset(root: Path) -> Path:
    rng = np.random.default_rng(0)
    data = root / "data"
    (data / "set.000").mkdir(parents=True)
    coord, atype, _ = frame(n_mol=2, box=8.0)
    n_frame, n_atom = 4, atype.shape[1]
    (data / "type.raw").write_text("\n".join(map(str, atype[0].tolist())) + "\n")
    (data / "type_map.raw").write_text("O\nH\n")
    coordinates = coord[0].cpu().numpy()[None] + 0.05 * rng.normal(
        size=(n_frame, n_atom, 3)
    )
    np.save(data / "set.000/coord.npy", coordinates.reshape(n_frame, -1))
    box = np.tile((np.eye(3) * 8.0).reshape(1, 9), (n_frame, 1))
    np.save(data / "set.000/box.npy", box)
    np.save(data / "set.000/energy.npy", rng.normal(size=n_frame))
    np.save(data / "set.000/force.npy", rng.normal(size=(n_frame, n_atom * 3)))
    return data


def changed(before: dict, after: dict) -> set[str]:
    assert before.keys() == after.keys()
    return {
        key
        for key, value in before.items()
        if torch.is_tensor(value) and not torch.equal(value, after[key])
    }


@pytest.mark.parametrize("kind", list(SHORT_RANGE))
def test_three_stages_through_the_cli(tmp_path, kind):
    data = write_dataset(tmp_path)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(REPO),
    }

    def run(*command, cwd):
        proc = subprocess.run(
            [sys.executable, *command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def weights(path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)["model"]

    def stage(name: str, model: dict, *flags: str) -> dict:
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        cfg = {
            "model": model,
            "learning_rate": {"start_lr": 1e-3, "stop_lr": 1e-4, "decay_steps": 1},
            "loss": {
                "type": "ener",
                "start_pref_e": 1.0,
                "limit_pref_e": 1.0,
                "start_pref_f": 1.0,
                "limit_pref_f": 1.0,
            },
            "training": {
                "training_data": {"systems": [str(data)], "batch_size": 1},
                "numb_steps": 3,
                "disp_freq": 1,
                "save_freq": 3,
                "seed": 1,
            },
        }
        (folder / "input.json").write_text(json.dumps(cfg))
        if name != "lr":
            run("-m", "deepmd", "--pt-expt", "train", "input.json", *flags, cwd=folder)
            return weights(folder / "model.ckpt.pt")
        run(
            str(REPO / "scripts" / "graft_short_range.py"),
            "--short-range",
            str(tmp_path / "sr" / "model.ckpt.pt"),
            "--input",
            "input.json",
            "--out",
            "init.ckpt.pt",
            cwd=folder,
        )
        run(
            "-m", "deepmd", "--pt-expt", "train", "input.json",
            "--init-model", "init.ckpt",
            cwd=folder,
        )  # fmt: skip
        return weights(folder / "model.ckpt.pt")

    sr = stage("sr", short_range(kind))
    lr = stage("lr", long_range(kind, charge_head_only=True))
    grafted = weights(tmp_path / "lr" / "init.ckpt.pt")
    # The graft carries every short-range weight over...
    assert all(torch.equal(grafted[key], value) for key, value in sr.items()
               if torch.is_tensor(value))  # fmt: skip
    # ...the LR stage moves the charge head and nothing else...
    moved = changed(grafted, lr)
    assert moved and all(CHARGE in key for key in moved)
    # ...and the FT stage, from the LR model, moves both.
    ft = stage(
        "ft",
        long_range(kind, charge_head_only=False),
        "--init-model",
        str(tmp_path / "lr" / "model.ckpt"),
    )
    moved = changed(lr, ft)
    assert any(CHARGE in key for key in moved)
    assert any(".descriptor." in key for key in moved)
    assert any(".fitting_net." in key for key in moved)
