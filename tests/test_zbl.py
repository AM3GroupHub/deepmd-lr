"""ZBL is a fixed prior: conservative E/F/virial, complete checkpoints and training."""

import copy
import math

import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.model.model import BaseModel
from deepmd.utils.argcheck import normalize

from deepmd_lr.zbl import ZBL
from tests.test_joint_model import DESCRIPTOR, FITTING


def config(lr=False):
    cfg = {
        "type": "dp_lr",
        "type_map": ["Si", "C"],
        "descriptor": copy.deepcopy(DESCRIPTOR),
        "fitting_net": copy.deepcopy(FITTING),
        "zbl": {"r_inner": 0.8, "r_outer": 1.5},
    }
    if lr:
        cfg["long_range"] = {
            "charge_net": {**copy.deepcopy(FITTING), "seed": 7},
            "reciprocal": {"method": "isolated", "alpha": 0.4},
        }
    return cfg


def dimer(r=1.1):
    return (
        torch.tensor([[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[0, 1]], dtype=torch.int64),
    )


def pair_reference(r):
    a = 0.46850 / (14**0.23 + 6**0.23)
    phi = sum(
        c * math.exp(-b * r / a)
        for c, b in [
            (0.18175, 3.19980),
            (0.50986, 0.94229),
            (0.28022, 0.40290),
            (0.02817, 0.20162),
        ]
    )
    t = min(1.0, max(0.0, (r - 0.8) / 0.7))
    return 14.3996454784255 * 14 * 6 / r * phi * (1 - 10 * t**3 + 15 * t**4 - 6 * t**5)


@pytest.mark.parametrize("lr", [False, True])
def test_fixed_parameters_and_serialization(lr):
    cfg = config(lr)
    model = get_model(cfg).eval()
    base = get_model({k: v for k, v in cfg.items() if k != "zbl"}).eval()
    assert [(n, p.shape) for n, p in model.named_parameters()] == [
        (n, p.shape) for n, p in base.named_parameters()
    ]
    assert list(model.zbl.parameters()) == []
    assert model.state_dict().keys() == base.state_dict().keys()
    restored = BaseModel.deserialize(model.serialize()).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    x, t = dimer()
    for key in ["energy", "force", "virial"]:
        torch.testing.assert_close(model(x, t)[key], restored(x, t)[key])


@pytest.mark.parametrize("lr", [False, True])
@pytest.mark.parametrize("r", [0.5, 0.8, 1.1, 1.5, 2.0])
def test_energy_and_unchanged_long_range(lr, r):
    cfg = config(lr)
    model = get_model(cfg).eval()
    base = get_model({k: v for k, v in cfg.items() if k != "zbl"}).eval()
    base.load_state_dict(model.state_dict(), strict=True)
    x, t = dimer(r)
    actual, original = model(x, t), base(x, t)
    assert (actual["energy"] - original["energy"]).item() == pytest.approx(
        pair_reference(r), abs=1e-10
    )
    if lr:
        for key in ["charge", "energy_long_range"]:
            torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize("lr", [False, True])
def test_force_and_virial_are_derivatives(lr):
    model = get_model(config(lr)).eval()
    x, t = dimer()
    out = model(x, t)
    h = 1e-5
    plus, minus = x.clone(), x.clone()
    plus[0, 1, 0] += h
    minus[0, 1, 0] -= h
    force = -(model(plus, t)["energy"] - model(minus, t)["energy"]) / (2 * h)
    torch.testing.assert_close(
        out["force"][0, 1, 0], force.squeeze(), rtol=2e-7, atol=1e-6
    )
    vir = -(model(x * (1 + h), t)["energy"] - model(x * (1 - h), t)["energy"]) / (2 * h)
    torch.testing.assert_close(out["virial"][0, 0], vir.squeeze(), rtol=2e-7, atol=1e-6)
    torch.testing.assert_close(
        out["force"].sum(1),
        torch.zeros((1, 3), dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("lr", [False, True])
def test_energy_force_loss_backward_and_update(lr):
    model = get_model(config(lr)).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x, t = dimer()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(3):
        optimizer.zero_grad()
        out = model(x, t)
        loss = (out["energy"] - 1).square().mean() + out["force"].square().mean()
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        optimizer.step()
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_normalized_training_configuration():
    cfg = {
        "model": config(),
        "learning_rate": {"start_lr": 1e-3},
        "loss": {"type": "ener"},
        "training": {"training_data": {"systems": ["./data"]}, "numb_steps": 3},
    }
    normalized = normalize(cfg)
    model = get_model(normalized["model"])
    assert model.zbl.serialize() == config()["zbl"]


@pytest.mark.parametrize("lr", [False, True])
@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, {"r_inner": 0.5, "r_outer": 0.8}),
        ({"r_inner": 0.4}, {"r_inner": 0.4, "r_outer": 0.8}),
        ({"r_outer": 1.0}, {"r_inner": 0.5, "r_outer": 1.0}),
        ({"r_inner": 0.8, "r_outer": 1.5}, {"r_inner": 0.8, "r_outer": 1.5}),
    ],
)
def test_default_radii_and_overrides(lr, settings, expected):
    cfg = config(lr)
    cfg["zbl"] = settings
    model = get_model(cfg)
    assert model.zbl.serialize() == expected
    restored = BaseModel.deserialize(model.serialize()).eval()
    x, t = dimer(0.6)
    for key in ["energy", "force", "virial"]:
        torch.testing.assert_close(model(x, t)[key], restored(x, t)[key])
    normalized = normalize(
        {
            "model": cfg,
            "learning_rate": {"start_lr": 1e-3},
            "loss": {"type": "ener"},
            "training": {"training_data": {"systems": ["./data"]}, "numb_steps": 3},
        }
    )
    assert get_model(normalized["model"]).zbl.serialize() == expected
    without_zbl = dict(cfg)
    without_zbl.pop("zbl")
    assert get_model(without_zbl).zbl is None


@pytest.mark.parametrize(
    "settings",
    [
        {"r_inner": 0, "r_outer": 1},
        {"r_inner": 1, "r_outer": 1},
        {"r_inner": True, "r_outer": 1},
        {"r_inner": float("nan"), "r_outer": 1},
        {"r_inner": 1, "r_outer": float("inf")},
        {"r_inner": 1, "r_outer": 7},
        {"r_inner": 0.8, "r_outer": 1.5, "trainable": True},
    ],
)
def test_invalid_configuration(settings):
    cfg = config()
    cfg["zbl"] = settings
    with pytest.raises(ValueError):
        get_model(cfg)


def test_masked_zero_edges_and_empty_graph():
    zbl = ZBL(["Si", "C"], config()["zbl"])
    for n in [0, 1]:
        edges = torch.zeros((2, n), dtype=torch.int64)
        vec = torch.zeros((n, 3), dtype=torch.float64, requires_grad=True)
        energy = zbl.atomic_energy(
            torch.tensor([0, 1]), edges, vec, torch.zeros(n, dtype=torch.bool)
        )
        assert energy.eq(0).all()
        (gradient,) = torch.autograd.grad(energy.sum(), vec)
        assert torch.isfinite(gradient).all()


def test_export_paths_cannot_silently_drop_zbl():
    model = get_model(config())
    with pytest.raises(NotImplementedError, match="ZBL"):
        model.forward_lower()
    with pytest.raises(NotImplementedError, match="ZBL"):
        model.forward_lower_canonical_graph()


@pytest.mark.parametrize("lr", [False, True])
def test_real_cli_training_checkpoint_restart(tmp_path, lr):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np

    from deepmd_lr.ase import load_model

    data = tmp_path / "data"
    (data / "set.000").mkdir(parents=True)
    (data / "type.raw").write_text("0\n1\n")
    (data / "type_map.raw").write_text("Si\nC\n")
    np.save(data / "set.000/box.npy", np.tile((np.eye(3) * 20).reshape(1, 9), (4, 1)))
    coordinates = np.array(
        [[[0, 0, 0], [r, 0, 0]] for r in [1.0, 1.1, 1.2, 1.3]], dtype=float
    )
    np.save(data / "set.000/coord.npy", coordinates.reshape(4, 6))
    np.save(data / "set.000/energy.npy", np.arange(4, dtype=float))
    np.save(data / "set.000/force.npy", np.zeros((4, 6)))
    cfg = {
        "model": config(lr),
        "learning_rate": {"start_lr": 1e-4, "stop_lr": 1e-5, "decay_steps": 1},
        "loss": {
            "type": "ener",
            "start_pref_e": 1.0,
            "limit_pref_e": 1.0,
            "start_pref_f": 1.0,
            "limit_pref_f": 1.0,
        },
        "training": {
            "training_data": {"systems": [str(data)], "batch_size": 1},
            "numb_steps": 2,
            "disp_freq": 1,
            "save_freq": 1,
            "seed": 1,
        },
    }
    if lr:
        cfg["loss"].update(type="ener_lr", start_pref_qn=100.0, limit_pref_qn=100.0)
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps(cfg))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    command = [sys.executable, "-m", "deepmd", "--pt-expt", "train", str(inp)]
    proc = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    checkpoint = tmp_path / "model.ckpt.pt"
    model = load_model(str(checkpoint), device="cpu")
    assert model.zbl.serialize() == cfg["model"]["zbl"]
    x, t = dimer()
    assert torch.isfinite(model(x, t)["force"]).all()
    cfg["training"]["numb_steps"] = 3
    inp.write_text(json.dumps(cfg))
    proc = subprocess.run(
        command + ["--restart", str(checkpoint)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ema = tmp_path / "model_ema.ckpt.pt"
    assert ema.exists()
    restored = load_model(str(ema), device="cpu")
    assert restored.zbl.serialize() == cfg["model"]["zbl"]
    assert torch.isfinite(restored(x, t)["energy"]).all()


def test_same_as_inherits_zbl_and_shares_weights():
    from tests.test_same_as import build_group, pseudo_head_config

    cfg = pseudo_head_config()
    cfg["model_dict"]["bulk"]["zbl"] = config()["zbl"]
    models = build_group(cfg)
    assert models["slab"].zbl.serialize() == models["bulk"].zbl.serialize()
    assert models["slab"].atomic_model is models["bulk"].atomic_model


def test_invalid_element_and_conflicting_table_rejected():
    cfg = config()
    cfg["type_map"] = ["X", "C"]
    with pytest.raises(ValueError, match="real element"):
        get_model(cfg)
    cfg = config()
    cfg["use_srtab"] = "unused.table"
    with pytest.raises(ValueError, match="choose one"):
        get_model(cfg)


def test_charge_filter_remains_independent_of_zbl():
    cfg = config(True)
    cfg["long_range"]["charge_filter"] = [0.1, 0.05]
    model = get_model(cfg).eval()
    plain = get_model({k: v for k, v in cfg.items() if k != "zbl"}).eval()
    plain.load_state_dict(model.state_dict(), strict=True)
    x, t = dimer()
    for key in ["charge", "energy_long_range", "charge_raw", "charge_projected"]:
        actual, reference = model(x, t), plain(x, t)
        if key in actual:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)


def test_switch_is_c2_at_both_boundaries():
    zbl = ZBL(["Si", "C"], config()["zbl"])
    types = torch.tensor([0, 1])
    edges = torch.tensor([[0, 1], [1, 0]])
    mask = torch.ones(2, dtype=torch.bool)

    def derivatives(r):
        distance = torch.tensor(r, dtype=torch.float64, requires_grad=True)
        vec = torch.stack([distance, -distance])[:, None] * torch.tensor(
            [[1.0, 0.0, 0.0]]
        )
        energy = zbl.atomic_energy(types, edges, vec, mask).sum()
        (grad,) = torch.autograd.grad(energy, distance, create_graph=True)
        (hess,) = torch.autograd.grad(grad, distance)
        return torch.stack([energy, grad, hess]).detach()

    for boundary in [0.8, 1.5]:
        torch.testing.assert_close(
            derivatives(boundary - 1e-8),
            derivatives(boundary + 1e-8),
            atol=2e-4,
            rtol=2e-6,
        )


def test_pure_dpa4c_graph_export_retains_zbl():
    from deepmd_lr.neighbors import build_neighbor_graph_fast

    model = get_model(config()).eval()
    x, t = dimer()
    from deepmd.dpmodel.utils.neighbor_graph import attach_edge_csr

    graph = attach_edge_csr(build_neighbor_graph_fast(x, t, None, model.get_rcut()), 2)
    counts = torch.tensor([2])
    args = (
        t.reshape(-1),
        counts,
        counts,
        graph.edge_index,
        graph.edge_vec,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_order,
        graph.source_row_ptr,
    )
    exported = model.forward_lower_graph_exportable(*args)
    output = exported(*args, None, None, None)
    reference = model(x, t)
    for key, public in [
        ("energy", "energy"),
        ("force", "force"),
        ("virial", "virial"),
    ]:
        torch.testing.assert_close(
            output[key].reshape(-1), reference[public].reshape(-1)
        )
