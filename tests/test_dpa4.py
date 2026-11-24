# SPDX-License-Identifier: MIT
"""The long-range model on DPA4, DeePMD-kit's message-passing descriptor.

DPA4 gives the model the same graph interface as DPA4C, so the long-range
term, the charge head and the charge-response force need nothing new on one
device. What does differ is locality: an atom's output reaches several
neighbour shells, and the paths that assume one shell -- the Born tensor's
pair formula, partitions with a halo of one cutoff -- must refuse rather than
return wrong numbers.
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
from deepmd.pt_expt.model.model import BaseModel
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.model.lr_model import (
    backbone,
    gate_margin,
    receptive_hops,
    require_one_hop,
)
from deepmd_lr.neighbors import build_neighbor_graph_fast

from .test_joint_model import DT, FITTING, LONG_RANGE, build, frame

RCUT = 4.0
DESCRIPTOR = {
    "type": "dpa4",
    "rcut": RCUT,
    "sel": 40,
    "channels": 16,
    "lmax": 2,
    "n_blocks": 2,
    "n_radial": 8,
    "precision": "float64",
    "use_amp": False,
    "seed": 1,
}


BRIDGING = {
    "bridging_method": "ZBL",
    "bridging_r_inner": 0.5,
    "bridging_r_outer": 0.8,
}


def build_dpa4(
    long_range: bool = True,
    fitting: str = "ener",
    bridging: bool = False,
    model_type: str = "dp_lr",
    **descriptor,
):
    """A small DPA4 model whose weights have all been moved off their
    initial values: DPA4 starts with a readout that returns zero, and a zero
    energy would pass every force test."""
    cfg = {
        "type": model_type,
        "type_map": ["O", "H"],
        "descriptor": {**copy.deepcopy(DESCRIPTOR), **descriptor},
        "fitting_net": {**copy.deepcopy(FITTING), "type": fitting},
    }
    if long_range:
        cfg["long_range"] = copy.deepcopy(LONG_RANGE)
    if bridging:
        # DeePMD-kit bridges a DPA4 descriptor with its own energy net only.
        cfg.update(BRIDGING)
        cfg["fitting_net"]["type"] = "dpa4_ener"
    return roughened(get_model(cfg))


def roughened(model):
    generator = torch.Generator(device="cpu").manual_seed(11)
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.is_floating_point():
                continue
            noise = torch.randn(parameter.shape, generator=generator, dtype=DT)
            parameter.add_(0.05 * noise.to(parameter))
    return model.eval()


def evaluate(model, coord, atype, cell):
    return model(coord, atype, box=cell)


def derivative(f, h: float = 5e-4) -> float:
    """Fourth-order central difference of ``f`` at zero. The random weights
    make a rough surface: the second-order stencil at a step small enough to
    resolve it would sit on the round-off of the energy."""
    return (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)


class TestBuild:
    def test_the_model_is_dp_lr_on_a_dpa4_descriptor(self):
        model = build_dpa4()
        assert type(model).__name__ == "LongRangeEnergyModel"
        assert type(model.atomic_model.descriptor).__name__ == "DescrptDPA4"
        assert model.has_long_range

    def test_the_fitting_net_deepmd_kit_pairs_with_dpa4_is_accepted(self):
        model = build_dpa4(fitting="dpa4_ener")
        coord, atype, cell = frame(n_mol=6, box=10.0)
        assert torch.isfinite(evaluate(model, coord, atype, cell)["energy"]).all()

    def test_another_fitting_type_is_still_refused(self):
        with pytest.raises(ValueError, match="energy model"):
            build_dpa4(fitting="dipole")

    def test_it_round_trips_through_serialization(self):
        model = build_dpa4()
        coord, atype, cell = frame(n_mol=6, box=10.0)
        again = BaseModel.deserialize(model.serialize()).to(DEVICE).eval()
        assert type(again) is type(model)
        a, b = evaluate(model, coord, atype, cell), evaluate(again, coord, atype, cell)
        for key in ("energy", "force", "charge"):
            assert torch.allclose(a[key], b[key], atol=1e-10), key


class TestLocality:
    """How far an atom's descriptor output reaches, which is what the halo
    of a partition and the Born tensor will have to cover."""

    def test_dpa4c_is_one_hop(self):
        assert receptive_hops(build()) == 1

    @pytest.mark.parametrize(
        ("n_blocks", "use_env_seed"), [(1, True), (2, True), (2, False), (3, True)]
    )
    def test_the_reach_is_the_blocks_and_the_seed(self, n_blocks, use_env_seed):
        """A chain with one neighbour per hop: moving the atom ``k`` links
        away changes the first atom's features up to ``k = hops`` and not at
        all beyond."""
        model = build_dpa4(
            long_range=False, n_blocks=n_blocks, use_env_seed=use_env_seed
        )
        hops = receptive_hops(model)
        assert hops == n_blocks + int(use_env_seed)
        n = hops + 3
        base = torch.zeros((1, n, 3), dtype=DT, device=DEVICE)
        base[0, :, 0] = 0.9 * RCUT * torch.arange(n, dtype=DT, device=DEVICE)
        atype = torch.zeros((1, n), dtype=torch.int64, device=DEVICE)

        def features(coord):
            graph = build_neighbor_graph_fast(coord, atype, None, RCUT)
            out = model.atomic_model.descriptor.call_graph(graph, atype.reshape(-1))
            return out[0][0].detach()

        reference = features(base)
        changed = []
        for k in range(1, n):
            moved = base.clone()
            moved[0, k, 1] += 0.2
            # Exactly zero beyond the reach; the signal that does arrive
            # fades by orders of magnitude per hop, so no threshold.
            changed.append(bool((features(moved) != reference).any()))
        assert changed == [k <= hops for k in range(1, n)]

    @pytest.mark.parametrize(
        ("n_blocks", "use_env_seed"), [(1, True), (1, False), (2, True)]
    )
    def test_bridging_adds_the_gate_margin_and_no_shell(self, n_blocks, use_env_seed):
        """A neighbour's message is gated by that neighbour's own short
        contacts. A chain ends ``k`` links away in an atom with a partner
        inside the bridging window on its far side: the partner is beyond
        every cutoff of the path, and is felt at the chain's start only with
        bridging, and only when its atom is the outermost one read."""
        options = {
            "long_range": False,
            "fitting": "dpa4_ener",
            "n_blocks": n_blocks,
            "use_env_seed": use_env_seed,
        }
        for bridging in (False, True):
            model = build_dpa4(bridging=bridging, **options)
            hops = receptive_hops(model)
            assert hops == n_blocks + int(use_env_seed)
            assert gate_margin(model) == (0.8 if bridging else 0.0)

            def felt(k, model=model):
                atype = torch.zeros((1, k + 2), dtype=torch.int64, device=DEVICE)
                descriptor = backbone(model).descriptor

                def features(length):
                    coord = torch.zeros((1, k + 2, 3), dtype=DT, device=DEVICE)
                    coord[0, : k + 1, 0] = (
                        0.9 * RCUT * torch.arange(k + 1, dtype=DT, device=DEVICE)
                    )
                    coord[0, k + 1, 0] = coord[0, k, 0] + length  # the partner
                    graph = build_neighbor_graph_fast(coord, atype, None, RCUT)
                    return descriptor.call_graph(graph, atype.reshape(-1))[0][
                        0
                    ].detach()

                return bool((features(0.65) != features(0.67)).any())

            assert felt(hops - 1), bridging  # within the cutoffs either way
            assert felt(hops) == bridging
            assert not felt(hops + 1), bridging


class TestSerial:
    """One device, the whole system: exact for any depth of message passing."""

    @pytest.mark.parametrize("long_range", [False, True])
    def test_forces_and_virial_are_the_energy_gradient(self, long_range):
        model = build_dpa4(long_range=long_range)
        coord, atype, cell = frame(n_mol=8, box=10.0)
        out = evaluate(model, coord, atype, cell)
        force = out["force"][0]
        assert float(force.abs().max()) > 1e-3  # a model that does something
        # The short-range force is the gradient to 1e-10 eV/A. The reciprocal
        # sum's force is not quite the gradient of its mesh energy: with these
        # weights fixed the gap moves between 6e-11 and 2e-7 eV/A with the
        # mesh, the spline order and alpha, so it is the sum's, not DPA4's.
        tolerance = 1e-6 if long_range else 1e-8
        for atom, axis in [(0, 0), (7, 1), (13, 2), (20, 0)]:

            def displaced(step, atom=atom, axis=axis):
                moved = coord.clone()
                moved[0, atom, axis] += step
                return float(evaluate(model, moved, atype, cell)["energy"].sum())

            assert -derivative(displaced) == pytest.approx(
                float(force[atom, axis]), abs=tolerance
            )

        # The virial against an isotropic strain: -dE/d ln V.
        def strained(step):
            scale = 1 + step
            energy = evaluate(model, coord * scale, atype, cell * scale)["energy"]
            return float(energy.sum())

        trace = out["virial"].reshape(3, 3).diagonal().sum()
        assert -derivative(strained) == pytest.approx(float(trace), abs=1e-6)

    def test_the_charges_carry_the_total_and_move_with_the_geometry(self):
        model = build_dpa4()
        coord, atype, cell = frame(n_mol=8, box=10.0)
        charge = evaluate(model, coord, atype, cell)["charge"]
        assert float(charge.sum()) == pytest.approx(0.0, abs=1e-10)
        assert float(charge.abs().max()) > 1e-4

    def test_a_training_step_reaches_every_part(self):
        """Energy and force loss: the force term needs the second derivative
        through the message passing and the charge response."""
        model = build_dpa4().train()
        coord, atype, cell = frame(n_mol=6, box=10.0)
        out = model(coord, atype, box=cell)
        (out["energy"].sum() + out["force"].square().sum()).backward()
        for name in ("descriptor", "fitting_net"):
            module = getattr(model.atomic_model, name)
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            assert any(float(g.abs().max()) > 0 for g in grads), name
        grads = [p.grad for p in model.charge_fitting.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)

    def test_the_frame_condition_reaches_the_charges(self):
        model = build_dpa4(add_chg_spin_ebd=True, default_chg_spin=[0.0, 1.0])
        coord, atype, cell = frame(n_mol=6, box=10.0)
        neutral = model(coord, atype, box=cell)["charge"]
        condition = torch.tensor([[1.0, 2.0]], dtype=DT, device=DEVICE)
        cation = model(coord, atype, box=cell, charge_spin=condition)["charge"]
        assert float(neutral.sum()) == pytest.approx(0.0, abs=1e-10)
        assert float(cation.sum()) == pytest.approx(1.0, abs=1e-10)


class TestInterfaces:
    """What sits on top of the model reaches DPA4 the way it reaches DPA4C."""

    def test_the_ase_calculator_is_the_model(self):
        from ase import Atoms

        from deepmd_lr.ase import LongRangeCalculator

        model = build_dpa4()
        coord, atype, cell = frame(n_mol=6, box=10.0)
        symbols = ["OH"[t] for t in atype[0].tolist()]
        atoms = Atoms(
            symbols, coord[0].cpu().numpy(), cell=cell[0].cpu().numpy(), pbc=True
        )
        atoms.calc = LongRangeCalculator(model)
        direct = evaluate(model, coord, atype, cell)
        assert atoms.get_potential_energy() == pytest.approx(
            float(direct["energy"]), abs=1e-9
        )
        assert np.allclose(
            atoms.get_forces(), direct["force"][0].detach().cpu().numpy(), atol=1e-9
        )

    def test_the_charge_filter_keeps_the_total_and_the_gradient(self):
        model = build_dpa4()
        coord, atype, cell = frame(n_mol=8, box=10.0)
        plain = evaluate(model, coord, atype, cell)["charge"]
        threshold = float(plain.abs().median())
        model.charge_filter = (threshold, 0.5 * threshold)
        out = evaluate(model, coord, atype, cell)
        assert int((out["charge"] == 0).sum()) > 0
        assert float(out["charge"].sum()) == pytest.approx(0.0, abs=1e-10)

        def displaced(step):
            moved = coord.clone()
            moved[0, 3, 1] += step
            return float(evaluate(model, moved, atype, cell)["energy"].sum())

        assert -derivative(displaced) == pytest.approx(
            float(out["force"][0, 3, 1]), abs=1e-6
        )

    def test_a_same_as_link_shares_the_dpa4_backbone(self):
        from .test_same_as import build_group, pseudo_head_config

        cfg = pseudo_head_config()
        cfg["shared_dict"]["des"] = copy.deepcopy(DESCRIPTOR)
        models = build_group(cfg)
        assert type(models["bulk"].atomic_model.descriptor).__name__ == "DescrptDPA4"
        assert models["slab"].atomic_model is models["bulk"].atomic_model
        assert models["slab"].charge_fitting is models["bulk"].charge_fitting


@pytest.mark.parametrize("bridging", [False, True])
def test_cli_training_checkpoint_and_restart(tmp_path, bridging):
    """``dp train`` with the long-range loss on a DPA4 model, the fitting net
    DeePMD-kit pairs with it at its automatic width, and a restart."""
    from deepmd_lr.ase import load_model

    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    (data / "set.000").mkdir(parents=True)
    coord, atype, cell = frame(n_mol=2, box=8.0)
    n_frame, n_atom = 4, atype.shape[1]
    (data / "type.raw").write_text("\n".join(map(str, atype[0].tolist())) + "\n")
    (data / "type_map.raw").write_text("O\nH\n")
    coordinates = coord[0].cpu().numpy()[None] + 0.05 * rng.normal(
        size=(n_frame, n_atom, 3)
    )
    np.save(data / "set.000/coord.npy", coordinates.reshape(n_frame, -1))
    np.save(data / "set.000/box.npy", np.tile((np.eye(3) * 8.0).reshape(1, 9), (4, 1)))
    np.save(data / "set.000/energy.npy", rng.normal(size=n_frame))
    np.save(data / "set.000/force.npy", rng.normal(size=(n_frame, n_atom * 3)))
    descriptor = {**DESCRIPTOR, "precision": "float32"}
    del descriptor["use_amp"]  # DeePMD-kit's default
    cfg = {
        "model": {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": descriptor,
            "fitting_net": {"type": "dpa4_ener", "neuron": [0], "seed": 1},
            "long_range": {
                "charge_net": {"neuron": [16, 16], "seed": 2},
                "reciprocal": {"alpha": 0.4, "mesh_dimensions": [12, 12, 12]},
            },
        },
        "learning_rate": {"start_lr": 1e-4, "stop_lr": 1e-5, "decay_steps": 1},
        "loss": {
            "type": "ener_lr",
            "start_pref_e": 1.0,
            "limit_pref_e": 1.0,
            "start_pref_f": 1.0,
            "limit_pref_f": 1.0,
            "start_pref_qn": 100.0,
            "limit_pref_qn": 100.0,
        },
        "training": {
            "training_data": {"systems": [str(data)], "batch_size": 1},
            "numb_steps": 2,
            "disp_freq": 1,
            "save_freq": 1,
            "seed": 1,
        },
    }
    if bridging:
        cfg["model"].update(BRIDGING)
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
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    checkpoint = tmp_path / "model.ckpt.pt"
    model = load_model(str(checkpoint), device="cpu")
    assert type(model).__name__ == "LongRangeEnergyModel"
    assert type(backbone(model).descriptor).__name__ == "DescrptDPA4"
    out = model(coord.cpu(), atype.cpu(), box=torch.eye(3, dtype=DT).unsqueeze(0) * 8.0)
    assert torch.isfinite(out["force"]).all() and "charge" in out
    assert (model.charge_shield is not None) == bridging
    if bridging:
        assert type(model.atomic_model).__name__ == "LinearEnergyAtomicModel"
    cfg["training"]["numb_steps"] = 3
    inp.write_text(json.dumps(cfg))
    proc = subprocess.run(
        [*command, "--restart", str(checkpoint)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def born_tensors(model, coord, atype, cell) -> np.ndarray:
    model.compute_born = True
    try:
        out = evaluate(model, coord, atype, cell)
    finally:
        model.compute_born = False
    return out["born_charge"][0].detach().cpu().numpy()


class TestBornTensor:
    """The pair formula on a descriptor that passes messages: each pair
    vector's effect on every charge within reach, shared by its two atoms."""

    @pytest.mark.parametrize("n_blocks", [1, 2])
    def test_it_is_the_pair_assembly_of_finite_differences(self, n_blocks):
        """An oracle that shares nothing with the implementation but the
        graph: every edge vector is moved on its own, the raw charges are
        recomputed, and the tensor is summed edge by edge."""
        from deepmd_lr.model.lr_model import _run_backbone

        model = build_dpa4(n_blocks=n_blocks)
        coord, atype, cell = frame(n_mol=2, box=6.5)
        flat = atype.reshape(-1)
        graph = build_neighbor_graph_fast(coord, atype, cell, RCUT)
        vectors = graph.edge_vec.detach().clone()

        def total_raw(edge_vec):
            moved = copy.copy(graph)
            moved.edge_vec = edge_vec
            features, rot_mat = _run_backbone(model, moved, flat, None)
            return float(model.raw_charges(features, rot_mat, flat).sum())

        h = 1e-4
        mask = graph.edge_mask.bool()
        gradient = torch.zeros_like(vectors)
        for edge in torch.nonzero(mask).reshape(-1).tolist():
            for axis in range(3):
                plus, minus = vectors.clone(), vectors.clone()
                plus[edge, axis] += h
                minus[edge, axis] -= h
                gradient[edge, axis] = (total_raw(plus) - total_raw(minus)) / (2 * h)

        charge = evaluate(model, coord, atype, cell)["charge"][0].detach()
        expected = charge[:, None, None] * torch.eye(3, dtype=DT, device=DEVICE)
        src, dst = graph.edge_index[0].long(), graph.edge_index[1].long()
        for edge in torch.nonzero(mask).reshape(-1).tolist():
            half = 0.5 * torch.outer(vectors[edge], gradient[edge])
            expected[dst[edge]] += half
            expected[src[edge]] -= half
        tensors = born_tensors(model, coord, atype, cell)
        assert np.abs(tensors).max() > 0.1
        assert np.abs(tensors - expected.cpu().numpy()).max() < 1e-6

    def test_one_block_without_a_seed_is_the_one_hop_formula(self):
        """A DPA4 that reaches one shell is a one-hop descriptor, and the
        tensor is DPA4C's: the pair's own two charges, from finite
        differences of each charge against each atom."""
        model = build_dpa4(n_blocks=1, use_env_seed=False)
        assert receptive_hops(model) == 1
        coord, atype, cell = frame(n_mol=2, box=20.0)
        n = coord.shape[1]
        h = 1e-5

        def raw(positions):
            graph = build_neighbor_graph_fast(positions, atype, cell, RCUT)
            from deepmd_lr.model.lr_model import _run_backbone

            flat = atype.reshape(-1)
            return model.raw_charges(*_run_backbone(model, graph, flat, None), flat)

        jacobian = torch.zeros((n, n, 3), dtype=DT, device=DEVICE)
        for atom in range(n):
            for axis in range(3):
                plus, minus = coord.clone(), coord.clone()
                plus[0, atom, axis] += h
                minus[0, atom, axis] -= h
                jacobian[:, atom, axis] = (raw(plus) - raw(minus)).detach() / (2 * h)
        charge = evaluate(model, coord, atype, cell)["charge"][0].detach()
        expected = charge[:, None, None] * torch.eye(3, dtype=DT, device=DEVICE)
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_ji = coord[0, j] - coord[0, i]
                    expected[i] += 0.5 * torch.outer(
                        r_ji, jacobian[i, j] + jacobian[j, i]
                    )
        tensors = born_tensors(model, coord, atype, cell)
        assert np.abs(tensors - expected.cpu().numpy()).max() < 1e-6


def crowded():
    """A box big enough to partition, with one O-H pair inside the bridging
    window so that the gate and the charge shield are at work."""
    coord, atype, cell = frame(n_mol=40, box=16.0)
    coord = coord.clone()
    coord[0, 1] = coord[0, 0] + torch.tensor([0.62, 0.0, 0.0], dtype=DT, device=DEVICE)
    return coord, atype, cell


def squeezed(r: float):
    """A water box with the first O-H bond set to ``r``."""
    coord, atype, cell = frame(n_mol=6, box=10.0)
    coord = coord.clone()
    coord[0, 1] = coord[0, 0] + torch.tensor([r, 0.0, 0.0], dtype=DT, device=DEVICE)
    return coord, atype, cell


class TestBridging:
    """Short contacts on DPA4: DeePMD-kit's own ZBL bridging for the energy,
    and a shield on the charges from here."""

    def test_without_a_long_range_term_it_is_deepmd_kits_bridged_model(self):
        ours = build_dpa4(long_range=False, bridging=True, fitting="dpa4_ener")
        theirs = build_dpa4(
            long_range=False, bridging=True, fitting="dpa4_ener", model_type="standard"
        )
        assert type(ours).__name__ == "LongRangeEnergyModel"
        assert type(theirs).__name__ != "LongRangeEnergyModel"
        assert ours.charge_shield is None
        for r in (1.0, 0.65, 0.4):
            a, b = evaluate(ours, *squeezed(r)), evaluate(theirs, *squeezed(r))
            for key in ("energy", "force", "virial"):
                assert torch.allclose(a[key], b[key], atol=1e-10), (r, key)

    def test_the_shield_takes_the_bridging_radii(self):
        model = build_dpa4(bridging=True)
        assert model.charge_shield.radii == (0.5, 0.8)
        assert model.has_charge_envelope and model.zbl is None
        assert not list(model.charge_shield.state_dict())

    @pytest.mark.parametrize("r", [0.499, 0.4])
    def test_atoms_in_contact_carry_no_charge(self, r):
        model = build_dpa4(bridging=True)
        charge = evaluate(model, *squeezed(r))["charge"][0]
        assert charge[:2].eq(0).all()
        assert float(charge[2:].abs().max()) > 1e-4
        assert float(charge.sum()) == pytest.approx(0.0, abs=1e-10)

    def test_beyond_the_window_the_shield_changes_nothing(self):
        model = build_dpa4(bridging=True)
        shielded = evaluate(model, *squeezed(0.9))
        model.set_charge_shield(None)
        plain = evaluate(model, *squeezed(0.9))
        for key in ("energy", "force", "charge"):
            assert torch.allclose(shielded[key], plain[key], atol=1e-10), key

    @pytest.mark.parametrize("r", [1.0, 0.7, 0.6, 0.45])
    def test_the_force_is_the_gradient_through_the_window(self, r):
        model = build_dpa4(bridging=True)
        coord, atype, cell = squeezed(r)
        force = evaluate(model, coord, atype, cell)["force"][0, 1, 0]

        def displaced(step):
            moved = coord.clone()
            moved[0, 1, 0] += step
            return float(evaluate(model, moved, atype, cell)["energy"].sum())

        assert -derivative(displaced, 2e-4) == pytest.approx(float(force), abs=2e-6)

    def test_the_shield_has_its_own_key(self):
        cfg_radii = [0.6, 1.0]
        model = build_dpa4(bridging=True)
        model.set_charge_shield(cfg_radii)
        assert evaluate(model, *squeezed(0.6))["charge"][0, :2].eq(0).all()
        again = BaseModel.deserialize(model.serialize()).to(DEVICE).eval()
        assert again.charge_shield.radii == (0.6, 1.0)
        assert type(again.atomic_model).__name__ == "LinearEnergyAtomicModel"
        a, b = evaluate(model, *squeezed(0.7)), evaluate(again, *squeezed(0.7))
        for key in ("energy", "force", "charge"):
            assert torch.allclose(a[key], b[key], atol=1e-10), key

    def test_the_config_key_overrides_the_bridging_radii(self):
        cfg = {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": copy.deepcopy(DESCRIPTOR),
            "fitting_net": {**copy.deepcopy(FITTING), "type": "dpa4_ener"},
            "long_range": {**copy.deepcopy(LONG_RANGE), "charge_shield": [0.7, 0.9]},
            **BRIDGING,
        }
        assert get_model(cfg).charge_shield.radii == (0.7, 0.9)
        del cfg["bridging_method"]  # a shield without bridging
        assert get_model(cfg).charge_shield.radii == (0.7, 0.9)

    @pytest.mark.parametrize(
        ("radii", "message"),
        [([0.8, 0.5], "r_lo < r_hi"), ([0.5, 9.0], "beyond the cutoff"), (0.5, "r_lo")],
    )
    def test_bad_radii_are_refused(self, radii, message):
        with pytest.raises(ValueError, match=message):
            build_dpa4().set_charge_shield(radii)

    def test_the_zbl_block_and_bridging_do_not_mix(self):
        with pytest.raises(ValueError, match="both handle short contacts"):
            get_model(
                {
                    "type": "dp_lr",
                    "type_map": ["O", "H"],
                    "descriptor": copy.deepcopy(DESCRIPTOR),
                    "fitting_net": copy.deepcopy(FITTING),
                    "zbl": {},
                    **BRIDGING,
                }
            )

    def test_the_schema_takes_the_keys(self):
        from deepmd.utils.argcheck import normalize

        cfg = {
            "model": {
                "type": "dp_lr",
                "type_map": ["O", "H"],
                "descriptor": {"type": "dpa4", "rcut": 4.0},
                "fitting_net": {"type": "dpa4_ener"},
                "long_range": {
                    **copy.deepcopy(LONG_RANGE),
                    "charge_shield": [0.5, 0.8],
                },
                **BRIDGING,
            },
            "learning_rate": {"start_lr": 1e-3},
            "loss": {"type": "ener_lr"},
            "training": {"training_data": {"systems": ["./data"]}, "numb_steps": 1},
        }
        model = normalize(cfg)["model"]
        assert model["bridging_method"] == "ZBL"
        assert model["long_range"]["charge_shield"] == [0.5, 0.8]


class TestRefusals:
    """What assumes one neighbour shell says so instead of running."""

    def test_the_helper_names_the_descriptor_and_its_reach(self):
        require_one_hop(build(), "anything")
        with pytest.raises(NotImplementedError, match="DescrptDPA4.*3 neighbour"):
            require_one_hop(build_dpa4(), "A halo of one cutoff")

    def test_compression(self):
        with pytest.raises(NotImplementedError, match="no compressed form"):
            build_dpa4().enable_compression()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine runs on GPU")
class TestEngine:
    @pytest.fixture
    def system(self):
        return frame(n_mol=16, box=14.0)

    @staticmethod
    def host(system):
        coord, atype, cell = system
        return (t[0].cpu().numpy() for t in (coord, atype, cell))

    @pytest.mark.parametrize("long_range", [False, True])
    def test_one_device_is_the_serial_model(self, system, long_range):
        from deepmd_lr.nvalchemi import DistributedEvaluator

        model = build_dpa4(long_range=long_range)
        coord, atype, cell = self.host(system)
        serial = evaluate(model, *system)
        evaluator = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        energy, force = evaluator(coord)
        assert energy == pytest.approx(float(serial["energy"]), abs=1e-8)
        assert (
            np.abs(
                force.cpu().numpy() - serial["force"][0].detach().cpu().numpy()
            ).max()
            < 1e-8
        )

    def test_one_device_carries_the_charge_shield(self):
        from deepmd_lr.nvalchemi import DistributedEvaluator

        model = build_dpa4(bridging=True)
        coord, atype, cell = squeezed(0.65)
        serial = evaluate(model, coord, atype, cell)
        evaluator = DistributedEvaluator(
            model,
            atype[0].cpu().numpy(),
            cell[0].cpu().numpy(),
            devices=[DEVICE],
            memory_headroom=None,
        )
        energy, force = evaluator(coord[0].cpu().numpy())
        assert energy == pytest.approx(float(serial["energy"]), abs=1e-8)
        assert (force - serial["force"][0].detach()).abs().max() < 1e-8
        charge = evaluator.results["charge"]
        assert torch.allclose(
            torch.as_tensor(charge, device=DEVICE), serial["charge"][0], atol=1e-10
        )

    @pytest.mark.parametrize(
        "options",
        [
            {"long_range": False},
            {},
            {"bridging": True},
            {"use_env_seed": False},
            {"bridging": True, "use_env_seed": False},
        ],
        ids=["short-range", "long-range", "bridged", "one-hop", "bridged-one-hop"],
    )
    def test_partitions_with_a_deep_halo_are_the_serial_model(self, options):
        """The halo is as many cutoffs deep as the descriptor reaches, with
        the edges into the ghosts an owned atom reads: no exchange between
        partitions, and the same numbers as the undivided system."""
        from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy

        model = build_dpa4(n_blocks=1, **options)
        system = crowded()
        coord, atype, cell = self.host(system)
        serial = evaluate(model, *system)
        for n in (1, 2, 4):
            out = to_numpy(
                evaluate_partitioned(
                    model,
                    coord,
                    atype,
                    cell,
                    n_partitions=n,
                    devices=[DEVICE] * n,
                    compute_virial=True,
                )
            )
            assert out["energy"] == pytest.approx(float(serial["energy"]), abs=1e-8)
            for key in ("force", "virial") + (("charge",) if "charge" in out else ()):
                reference = serial[key].detach().cpu().numpy().reshape(out[key].shape)
                assert np.abs(out[key] - reference).max() < 1e-8, (n, key)

    def test_the_halo_is_as_deep_as_the_reach(self):
        from deepmd_lr.nvalchemi import partition_system

        coord, _atype, cell = self.host(frame(n_mol=40, box=16.0))
        shallow = partition_system(coord, cell, RCUT, 2)
        deep = partition_system(coord, cell, RCUT, 2, hops=3)
        for one, three in zip(shallow, deep, strict=True):
            assert one.n_centre is None and one.n_search == one.n_owned
            assert three.n_owned == one.n_owned
            assert three.n_owned < three.n_search < three.n_node
            assert three.n_node > one.n_node
            # The centres are the owned atoms and every ghost within two
            # cutoffs of one; the ghosts after them are farther than that.
            nodes = three.coordinates(coord)
            distance = np.linalg.norm(
                nodes[three.n_owned :, None] - nodes[None, : three.n_owned], axis=-1
            ).min(axis=1)
            inner = three.n_search - three.n_owned
            assert distance[:inner].max() <= 2 * RCUT < distance[inner:].min()
            assert distance.max() <= 3 * RCUT

    def test_partitions_built_for_one_hop_are_refused(self):
        from deepmd_lr.nvalchemi import evaluate_partitioned, partition_system

        coord, atype, cell = self.host(frame(n_mol=40, box=16.0))
        with pytest.raises(NotImplementedError, match="halo of one cutoff"):
            evaluate_partitioned(
                build_dpa4(n_blocks=1),
                coord,
                atype,
                cell,
                partitions=partition_system(coord, cell, RCUT, 2),
                devices=[DEVICE] * 2,
            )

    def test_several_devices_reuse_a_halo_while_atoms_move(self):
        from deepmd_lr.nvalchemi import DistributedEvaluator

        model = build_dpa4(n_blocks=1)
        system = frame(n_mol=40, box=16.0)
        coord, atype, cell = self.host(system)
        evaluator = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE] * 2, memory_headroom=None, skin=1.0
        )
        evaluator(coord)
        rebuilds = evaluator.rebuild_count
        moved = coord + np.random.default_rng(3).uniform(-0.2, 0.2, coord.shape)
        energy, force = evaluator(moved)
        assert evaluator.rebuild_count == rebuilds
        moved_t = torch.as_tensor(moved, dtype=DT, device=DEVICE).unsqueeze(0)
        serial = evaluate(model, moved_t, system[1], system[2])
        assert energy == pytest.approx(float(serial["energy"]), abs=1e-8)
        assert (force - serial["force"][0].detach()).abs().max() < 1e-8

    @pytest.mark.parametrize("n_blocks", [2, 3])
    def test_more_than_one_block_runs_on_one_device_only(self, n_blocks):
        """A halo of three or four cutoffs on every face is not offered."""
        from deepmd_lr.nvalchemi import (
            DistributedEvaluator,
            evaluate_partitioned,
            partition_system,
        )

        model = build_dpa4(n_blocks=n_blocks)
        system = frame(n_mol=40, box=16.0)
        coord, atype, cell = self.host(system)
        with pytest.raises(NotImplementedError, match="n_blocks = 1 only"):
            DistributedEvaluator(
                model, atype, cell, devices=[DEVICE] * 2, memory_headroom=None
            )
        for options in (
            {"n_partitions": 2},
            {"partitions": partition_system(coord, cell, RCUT, 2, hops=n_blocks + 1)},
        ):
            with pytest.raises(NotImplementedError, match="n_blocks = 1 only"):
                evaluate_partitioned(
                    model, coord, atype, cell, devices=[DEVICE] * 2, **options
                )
        # One device: no halo, and the serial numbers with the Born tensors.
        evaluator = DistributedEvaluator(
            model,
            atype,
            cell,
            devices=[DEVICE],
            memory_headroom=None,
            compute_born=True,
        )
        energy, force = evaluator(coord)
        serial = evaluate(model, *system)
        assert energy == pytest.approx(float(serial["energy"]), abs=1e-8)
        assert (force - serial["force"][0].detach()).abs().max() < 1e-8
        tensors = torch.as_tensor(evaluator.results["born_charge"]).cpu().numpy()
        assert np.abs(tensors - born_tensors(model, *system)).max() < 1e-8

    @pytest.mark.parametrize(
        "options",
        [{}, {"bridging": True}, {"use_env_seed": False}],
        ids=["long-range", "bridged", "one-hop"],
    )
    def test_born_tensors_agree_on_every_route(self, options):
        from deepmd_lr.nvalchemi import (
            DistributedEvaluator,
            evaluate_partitioned,
            to_numpy,
        )

        model = build_dpa4(n_blocks=1, **options)
        system = crowded()
        coord, atype, cell = self.host(system)
        reference = born_tensors(model, *system)
        assert np.abs(reference).max() > 0.1
        # A neutral frame: the tensors sum to zero whatever the derivative.
        assert np.abs(reference.sum(axis=0)).max() < 1e-10
        for n in (1, 2, 4):
            out = to_numpy(
                evaluate_partitioned(
                    model,
                    coord,
                    atype,
                    cell,
                    n_partitions=n,
                    devices=[DEVICE] * n,
                    compute_born=True,
                )
            )
            assert np.abs(out["born_charge"] - reference).max() < 1e-8, n
        evaluator = DistributedEvaluator(
            model,
            atype,
            cell,
            devices=[DEVICE],
            memory_headroom=None,
            compute_born=True,
        )
        evaluator(coord)
        one_device = torch.as_tensor(evaluator.results["born_charge"]).cpu().numpy()
        assert np.abs(one_device - reference).max() < 1e-8
