"""Short contacts: ZBL comes in and the network goes out, across one window.

What is pinned:
- **The settings**: one ``window`` in covalent-radius sums, required, checked;
  the keys of the first implementation are refused by name.
- **The pair term**: the ZBL formula times the C2 switch, per pair of
  elements, zero beyond the window, nothing for a type that is no element or
  an edge that is masked.
- **The envelope**: per atom, the product over its neighbours of the same
  step: exactly 1 beyond the window, exactly 0 below it, C2 between, with
  derivatives that vanish wherever the value is flat.
- **The model**: no parameter and no ``state_dict`` entry, so a trained model
  can be given the term; untouched wherever no pair is inside the window; a
  pair below it is its energy offsets plus ZBL, with no charge; force, virial
  and Born tensors are derivatives through the envelope; config, checkpoint
  and ``same_as`` carry it; interfaces that would drop it refuse.
"""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.data import atomic_numbers, covalent_radii
from dargs.dargs import ArgumentKeyError
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.model.model import BaseModel
from deepmd.pt_expt.utils.env import DEVICE
from deepmd.utils.argcheck import normalize

from deepmd_lr.zbl import ZBL

from .test_joint_model import DESCRIPTOR, DT, FITTING, LONG_RANGE, frame

#: O-H window 0.679-0.873 A, O-O 0.924-1.188 A, H-H 0.434-0.558 A.
WINDOW = [0.7, 0.9]


def config(lr: bool = False, zbl: bool = True) -> dict:
    cfg = {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": copy.deepcopy(DESCRIPTOR),
        "fitting_net": copy.deepcopy(FITTING),
    }
    if zbl:
        cfg["zbl"] = {"window": list(WINDOW)}
    if lr:
        cfg["long_range"] = copy.deepcopy(LONG_RANGE)
    return cfg


def twins(lr: bool = False):
    """The same weights with and without the term."""
    model = get_model(config(lr)).eval()
    plain = get_model(config(lr, zbl=False)).eval()
    plain.load_state_dict(model.state_dict(), strict=True)
    return model, plain


def dimer(r: float = 0.8):
    """An O-H pair at distance ``r``, with no cell."""
    coord = torch.tensor([[[0.0, 0.0, 0.0], [r, 0.0, 0.0]]], dtype=DT, device=DEVICE)
    return coord, torch.tensor([[0, 1]], dtype=torch.int64, device=DEVICE)


def pair_sum(a: str, b: str) -> float:
    return float(covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]])


def step(r: float, a: str, b: str, window=WINDOW) -> float:
    """The window's step for one pair: 0 below, 1 above."""
    r_lo, r_hi = (x * pair_sum(a, b) for x in window)
    t = min(1.0, max(0.0, (r - r_lo) / (r_hi - r_lo)))
    return t**3 * (10 - 15 * t + 6 * t**2)


def contact_frame(r: float = 0.78):
    """The periodic water box of the other tests with one O-H pair at ``r``."""
    coord, atype, cell = frame()
    coord = coord.clone()
    coord[0, 1] = coord[0, 0] + torch.tensor([r, 0.0, 0.0], dtype=DT, device=DEVICE)
    return coord, atype, cell


def pair_energy(r: float, a: str, b: str, window=WINDOW) -> float:
    """The switched ZBL energy of one pair, by the formula."""
    zi, zj = atomic_numbers[a], atomic_numbers[b]
    screening = 0.46850 / (zi**0.23 + zj**0.23)
    phi = sum(
        c * math.exp(-b_k * r / screening)
        for c, b_k in [
            (0.18175, 3.19980),
            (0.50986, 0.94229),
            (0.28022, 0.40290),
            (0.02817, 0.20162),
        ]
    )
    r_lo, r_hi = (x * pair_sum(a, b) for x in window)
    t = min(1.0, max(0.0, (r - r_lo) / (r_hi - r_lo)))
    return 14.3996454784255 * zi * zj / r * phi * (1 - 10 * t**3 + 15 * t**4 - 6 * t**5)


def pair_graph(r: float, types=(0, 1)):
    """Both directed edges of one pair along x."""
    edges = torch.tensor([[0, 1], [1, 0]])
    vec = torch.tensor([[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=DT)
    return torch.tensor(types), edges, vec, torch.ones(2, dtype=torch.bool)


class TestSettings:
    def test_window_round_trip(self):
        zbl = ZBL(["O", "H"], {"window": [0.7, 0.9], "_comment": "ignored"})
        assert zbl.serialize() == {"window": [0.7, 0.9]}
        assert list(zbl.parameters()) == []
        assert zbl.state_dict() == {}
        assert zbl.max_outer == pytest.approx(0.9 * pair_sum("O", "O"))

    @pytest.mark.parametrize(
        "window",
        [
            [0.9, 0.7],
            [0.7, 0.7],
            [-0.1, 0.9],
            [0.7],
            [0.7, 0.8, 0.9],
            [True, 0.9],
            [0.7, float("nan")],
            [0.7, float("inf")],
            "0.7, 0.9",
            None,
        ],
    )
    def test_bad_windows_are_refused(self, window):
        with pytest.raises(ValueError, match="window"):
            ZBL(["O", "H"], {"window": window})

    def test_the_window_is_required(self):
        with pytest.raises(ValueError, match="window is required"):
            ZBL(["O", "H"], {})

    def test_retired_keys_are_named(self):
        with pytest.raises(ValueError, match="first implementation"):
            ZBL(["O", "H"], {"window": WINDOW, "clamp": True})
        with pytest.raises(ValueError, match="accepts only"):
            ZBL(["O", "H"], {"window": WINDOW, "trainable": True})

    def test_a_type_map_without_an_element_is_refused(self):
        with pytest.raises(ValueError, match="at least one element"):
            ZBL(["X", "Y0"], {"window": WINDOW})

    def test_the_schema_takes_the_block_and_requires_the_window(self):
        def training_input(model):
            return {
                "model": model,
                "learning_rate": {"start_lr": 1e-3},
                "loss": {"type": "ener"},
                "training": {"training_data": {"systems": ["./data"]}, "numb_steps": 3},
            }

        normalized = normalize(training_input(config()))
        assert get_model(normalized["model"]).zbl.serialize() == {"window": WINDOW}
        assert normalize(training_input(config(zbl=False)))["model"]["zbl"] is None
        missing = config()
        missing["zbl"] = {}
        with pytest.raises(ArgumentKeyError):
            normalize(training_input(missing))


class TestPairTerm:
    @pytest.mark.parametrize("pair", [("O", "H"), ("O", "O"), ("H", "H")])
    def test_energy_is_the_formula_for_every_pair(self, pair):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        types = tuple(["O", "H"].index(s) for s in pair)
        total = pair_sum(*pair)
        for x in (0.3, 0.7, 0.75, 0.8, 0.85):
            r = x * total
            got = zbl.atomic_energy(*pair_graph(r, types)).sum().item()
            assert got == pytest.approx(pair_energy(r, *pair), rel=1e-12, abs=1e-12)
        # Exactly nothing from the upper edge on.
        for x in (0.9, 0.9001, 1.5):
            assert zbl.atomic_energy(*pair_graph(x * total, types)).sum().item() == 0.0

    def test_each_atom_takes_half(self):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        energy = zbl.atomic_energy(*pair_graph(0.75)).reshape(-1)
        assert energy[0].item() == pytest.approx(energy[1].item(), rel=1e-14)
        assert energy.sum().item() == pytest.approx(pair_energy(0.75, "O", "H"))

    def test_the_edge_derivative_is_the_gradient(self):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        atype, edges, vec, mask = pair_graph(0.78)
        # Off the axis, so every component is exercised.
        vec = vec @ torch.tensor(
            [[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]], dtype=DT
        )
        vec.requires_grad_(True)
        energy, g = zbl.edge_terms(atype, edges, vec, mask)
        (auto,) = torch.autograd.grad(energy.sum(), vec)
        torch.testing.assert_close(g, auto, rtol=1e-12, atol=1e-12)

    def test_the_switch_is_c2_at_both_window_edges(self):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        atype, edges, _vec, mask = pair_graph(1.0)
        axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=DT)

        def derivatives(r):
            distance = torch.tensor(r, dtype=DT, requires_grad=True)
            vec = torch.stack([distance, -distance])[:, None] * axis
            energy = zbl.atomic_energy(atype, edges, vec, mask).sum()
            (grad,) = torch.autograd.grad(energy, distance, create_graph=True)
            (hess,) = torch.autograd.grad(grad, distance)
            return torch.stack([energy, grad, hess]).detach()

        # The third derivative jumps there (by 60 / width^3 times the pair
        # energy, 1e5 eV/A^3 on this window), so the two sides are compared
        # close enough for that slope not to show.
        for x in WINDOW:
            edge = x * pair_sum("O", "H")
            torch.testing.assert_close(
                derivatives(edge - 1e-10),
                derivatives(edge + 1e-10),
                atol=2e-4,
                rtol=2e-6,
            )

    def test_masked_edges_and_an_empty_graph_give_nothing(self):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        for n in (0, 1):
            edges = torch.zeros((2, n), dtype=torch.int64)
            vec = torch.zeros((n, 3), dtype=DT, requires_grad=True)
            energy, g = zbl.edge_terms(
                torch.tensor([0, 1]), edges, vec, torch.zeros(n, dtype=torch.bool)
            )
            assert energy.eq(0).all()
            assert torch.isfinite(g).all() and g.eq(0).all()

    def test_a_type_that_is_no_element_has_no_nucleus(self):
        # The repository's periodic-table type maps start with ASE's dummy "X".
        zbl = ZBL(["X", "O", "H"], {"window": WINDOW})
        assert zbl.numbers.tolist() == [0.0, 8.0, 1.0]
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        vec = torch.tensor(
            [[0.75, 0, 0], [-0.75, 0, 0], [0.75, 0, 0], [-0.75, 0, 0]], dtype=DT
        )
        energy = zbl.atomic_energy(
            torch.tensor([0, 1, 2]), edges, vec, torch.ones(4, dtype=torch.bool)
        ).reshape(-1)
        assert energy[0].item() == 0.0
        assert energy[1:].sum().item() == pytest.approx(pair_energy(0.75, "O", "H"))

    def test_a_padded_node_is_skipped(self):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        atype, edges, vec, mask = pair_graph(0.75, types=(0, -1))
        assert zbl.atomic_energy(atype, edges, vec, mask).eq(0).all()


class TestEnvelope:
    @pytest.mark.parametrize("pair", [("O", "H"), ("O", "O"), ("H", "H")])
    def test_one_neighbour_gives_the_step(self, pair):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        types = tuple(["O", "H"].index(s) for s in pair)
        total = pair_sum(*pair)
        for x in (0.75, 0.8, 0.85):
            factor = zbl.envelope(*pair_graph(x * total, types))
            assert 0.0 < factor[0].item() < 1.0
            assert factor.tolist() == pytest.approx([step(x * total, *pair)] * 2)
        # Exact at both ends, not merely close.
        for x in (0.2, 0.69, 0.7):
            assert zbl.envelope(*pair_graph(x * total, types)).tolist() == [0.0, 0.0]
        for x in (0.9, 0.95, 3.0):
            assert zbl.envelope(*pair_graph(x * total, types)).tolist() == [1.0, 1.0]

    def test_the_product_runs_over_the_neighbours(self):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        # An O with two H at different distances inside the window, the H
        # atoms far from one another.
        r1, r2 = 0.76, 0.82
        edges = torch.tensor([[1, 0, 2, 0], [0, 1, 0, 2]])
        vec = torch.tensor([[r1, 0, 0], [-r1, 0, 0], [0, r2, 0], [0, -r2, 0]], dtype=DT)
        factor = zbl.envelope(
            torch.tensor([0, 1, 1]), edges, vec, torch.ones(4, dtype=torch.bool)
        )
        s1, s2 = step(r1, "O", "H"), step(r2, "O", "H")
        assert factor.tolist() == pytest.approx([s1 * s2, s1, s2], rel=1e-13)

    def test_the_gradient_vanishes_where_the_value_is_flat(self):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        total = pair_sum("O", "H")
        for x, flat in [
            (0.3, True),
            (0.7, True),
            (0.8, False),
            (0.9, True),
            (2.0, True),
        ]:
            atype, edges, vec, mask = pair_graph(x * total)
            vec.requires_grad_(True)
            factor = zbl.envelope(atype, edges, vec, mask)
            (grad,) = torch.autograd.grad(factor.sum(), vec, create_graph=True)
            (hess,) = torch.autograd.grad(grad.square().sum(), vec)
            assert torch.isfinite(grad).all() and torch.isfinite(hess).all()
            assert bool(grad.eq(0).all()) is flat

    def test_c2_at_both_window_edges(self):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        atype, edges, _vec, mask = pair_graph(1.0)
        axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=DT)

        def derivatives(r):
            distance = torch.tensor(r, dtype=DT, requires_grad=True)
            vec = torch.stack([distance, -distance])[:, None] * axis
            value = zbl.envelope(atype, edges, vec, mask)[0]
            (grad,) = torch.autograd.grad(value, distance, create_graph=True)
            (hess,) = torch.autograd.grad(grad, distance)
            return torch.stack([value, grad, hess]).detach()

        for x in WINDOW:
            edge = x * pair_sum("O", "H")
            torch.testing.assert_close(
                derivatives(edge - 1e-10), derivatives(edge + 1e-10), atol=1e-5, rtol=0
            )

    def test_what_is_no_contact_counts_for_nothing(self):
        zbl = ZBL(["X", "O", "H"], {"window": WINDOW})
        ones = [1.0, 1.0]
        # A masked edge, zero length included.
        atype, edges, vec, mask = pair_graph(0.5, types=(1, 2))
        assert zbl.envelope(atype, edges, vec, ~mask).tolist() == ones
        assert zbl.envelope(atype, edges, vec * 0, ~mask).tolist() == ones
        # A type that is no element, and a padded node.
        for types in [(0, 2), (1, -1)]:
            assert zbl.envelope(*pair_graph(0.5, types=types)).tolist() == ones
        # No edge at all.
        empty = torch.zeros((2, 0), dtype=torch.int64)
        factor = zbl.envelope(
            torch.tensor([1, 2]),
            empty,
            torch.zeros((0, 3), dtype=DT),
            torch.zeros(0, dtype=torch.bool),
        )
        assert factor.tolist() == ones

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_the_factor_keeps_the_vectors_precision(self, dtype):
        zbl = ZBL(["O", "H"], {"window": WINDOW})
        atype, edges, vec, mask = pair_graph(0.8)
        assert zbl.envelope(atype, edges, vec.to(dtype), mask).dtype == dtype


class TestModel:
    @pytest.mark.parametrize("lr", [False, True])
    def test_no_weights_and_a_complete_round_trip(self, lr):
        model, plain = twins(lr)
        assert [(n, p.shape) for n, p in model.named_parameters()] == [
            (n, p.shape) for n, p in plain.named_parameters()
        ]
        assert model.state_dict().keys() == plain.state_dict().keys()
        restored = BaseModel.deserialize(model.serialize()).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        assert restored.zbl.serialize() == {"window": WINDOW}
        coord, atype = dimer()
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                model(coord, atype)[key], restored(coord, atype)[key]
            )

    @pytest.mark.parametrize("lr", [False, True])
    def test_a_built_model_can_be_given_the_term(self, lr):
        model, plain = twins(lr)
        coord, atype = dimer()
        assert plain.zbl is None
        plain.set_zbl({"window": WINDOW})
        # In the model's own mode: a module made after `.eval()` would
        # otherwise stay in training mode and miss the inference kernel.
        assert plain.zbl.training is False
        assert plain.train().zbl.training is True and plain.eval().zbl.training is False
        assert json.loads(plain.model_def_script)["zbl"] == {"window": WINDOW}
        # Two evaluations agree to rounding, not to the bit: the kernels sum
        # with atomics, in whatever order the threads arrive.
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                plain(coord, atype)[key],
                model(coord, atype)[key],
                rtol=1e-12,
                atol=1e-12,
            )
        reference = get_model(config(lr, zbl=False)).eval()
        reference.load_state_dict(model.state_dict(), strict=True)
        plain.set_zbl(None)
        assert plain.zbl is None and "zbl" not in json.loads(plain.model_def_script)
        torch.testing.assert_close(
            plain(coord, atype)["energy"],
            reference(coord, atype)["energy"],
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize("lr", [False, True])
    @pytest.mark.parametrize("r", [0.6, 0.75, 0.85])
    def test_force_and_virial_are_derivatives(self, lr, r):
        model = get_model(config(lr)).eval()
        coord, atype = dimer(r)
        out = model(coord, atype)
        h = 1e-5
        plus, minus = coord.clone(), coord.clone()
        plus[0, 1, 0] += h
        minus[0, 1, 0] -= h
        force = -(model(plus, atype)["energy"] - model(minus, atype)["energy"]) / (
            2 * h
        )
        torch.testing.assert_close(
            out["force"][0, 1, 0], force.squeeze(), rtol=2e-7, atol=1e-6
        )
        virial = -(
            model(coord * (1 + h), atype)["energy"]
            - model(coord * (1 - h), atype)["energy"]
        ) / (2 * h)
        torch.testing.assert_close(
            out["virial"][0, 0], virial.squeeze(), rtol=2e-7, atol=1e-6
        )
        torch.testing.assert_close(
            out["force"].sum(1),
            torch.zeros((1, 3), dtype=DT, device=DEVICE),
            atol=1e-10,
            rtol=0,
        )

    @pytest.mark.parametrize("lr", [False, True])
    def test_fitted_geometries_are_untouched(self, lr):
        """No pair inside the window: every factor is exactly 1, ZBL exactly
        0, and the model is the one that was trained."""
        model, plain = twins(lr)
        coord, atype, cell = frame()
        actual, reference = model(coord, atype, box=cell), plain(coord, atype, box=cell)
        keys = ["energy", "atom_energy", "force", "virial"]
        if lr:
            keys += ["charge", "energy_long_range"]
        for key in keys:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=1e-13)

    @pytest.mark.parametrize("lr", [False, True])
    @pytest.mark.parametrize("r", [0.3, 0.6])
    def test_below_the_window_a_pair_is_its_offsets_and_zbl(self, lr, r):
        model, plain = twins(lr)
        offsets = torch.tensor([-3.0, -1.5], dtype=DT, device=DEVICE)
        for m in (model, plain):
            m.atomic_model.out_bias[0, :, 0] = offsets
        coord, atype = dimer(r)
        out = model(coord, atype)
        assert out["energy"].item() == pytest.approx(
            float(offsets.sum()) + pair_energy(r, "O", "H"), abs=1e-10
        )
        # The network alone is somewhere else entirely.
        assert abs(plain(coord, atype)["energy"].item() - float(offsets.sum())) > 0.1
        if lr:
            assert out["charge"].eq(0).all()
            assert out["energy_long_range"].item() == 0.0

    @pytest.mark.parametrize("lr", [False, True])
    def test_force_and_virial_in_a_periodic_frame(self, lr):
        """One O-H pair inside the window among ordinary molecules: the
        envelope's derivative reaches the force through the fitted energy
        and, with a long-range term, through the charges."""
        model, plain = twins(lr)
        coord, atype, cell = contact_frame()
        out = model(coord, atype, box=cell)
        different = out["energy"] - plain(coord, atype, box=cell)["energy"]
        assert different.abs().item() > 1.0
        h = 1e-5
        for atom in (0, 1, 2):
            for axis in range(3):
                plus, minus = coord.clone(), coord.clone()
                plus[0, atom, axis] += h
                minus[0, atom, axis] -= h
                numeric = -(
                    float(model(plus, atype, box=cell)["energy"])
                    - float(model(minus, atype, box=cell)["energy"])
                ) / (2 * h)
                assert out["force"][0, atom, axis].item() == pytest.approx(
                    numeric, rel=2e-6, abs=2e-6
                )
        eye = torch.eye(3, dtype=DT, device=DEVICE)
        numeric = torch.zeros((3, 3), dtype=DT, device=DEVICE)
        for i in range(3):
            for j in range(3):
                delta = torch.zeros((3, 3), dtype=DT, device=DEVICE)
                delta[i, j] = h
                delta = 0.5 * (delta + delta.T)
                energies = [
                    float(
                        model(
                            coord @ (eye + sign * delta).T,
                            atype,
                            box=cell @ (eye + sign * delta).T,
                        )["energy"]
                    )
                    for sign in (1.0, -1.0)
                ]
                numeric[i, j] = -(energies[0] - energies[1]) / (2 * h)
        torch.testing.assert_close(
            out["virial"].reshape(3, 3), numeric, atol=2e-5, rtol=2e-5
        )

    def test_a_contact_has_no_raw_charge_and_the_total_is_kept(self):
        from .test_born import pieces

        model, plain = twins(lr=True)
        coord, atype, cell = contact_frame(0.6)
        out, raw, _cap = pieces(model, coord, atype, box=cell)
        _out, raw_plain, _cap = pieces(plain, coord, atype, box=cell)
        assert raw[:2].eq(0).all() and raw_plain[:2].abs().min() > 1e-4
        # Two models on a GPU agree to rounding, not to the bit.
        torch.testing.assert_close(raw[2:], raw_plain[2:], rtol=0, atol=1e-13)
        assert out["charge"].sum().abs().item() < 1e-12
        # The projection spreads the deficit evenly, the two atoms included.
        torch.testing.assert_close(
            out["charge"][0, :2], -raw.mean().expand(2), rtol=0, atol=1e-14
        )

    @pytest.mark.parametrize("r", [0.78, 0.6])
    def test_born_tensors_carry_the_envelopes_derivative(self, r):
        from .test_born import build, molecule, pairwise_born

        model = build(isolated=True)
        model.set_zbl({"window": WINDOW})
        coord, atype = molecule()
        coord = coord.clone()
        coord[0, 1] = coord[0, 0] + torch.tensor([r, 0.0, 0.0], dtype=DT, device=DEVICE)
        fast, slow = pairwise_born(model, coord, atype)
        assert np.abs(fast - slow).max() < 1e-7, np.abs(fast - slow).max()
        plain = build(isolated=True)
        plain.load_state_dict(model.state_dict(), strict=True)
        changed = np.abs(fast - pairwise_born(plain, coord, atype)[0]).max()
        assert changed > 1e-3, "the contact must change the tensors"

    @pytest.mark.parametrize("lr", [False, True])
    def test_training_steps_stay_finite(self, lr):
        model = get_model(config(lr)).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        coord, atype = dimer(0.78)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        for _ in range(3):
            optimizer.zero_grad()
            out = model(coord, atype)
            loss = (out["energy"] - 1).square().mean() + out["force"].square().mean()
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            optimizer.step()
        assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters())

    def test_same_as_inherits_the_term(self):
        from .test_same_as import build_group, pseudo_head_config

        cfg = pseudo_head_config()
        cfg["model_dict"]["bulk"]["zbl"] = {"window": WINDOW}
        models = build_group(cfg)
        assert models["slab"].zbl.serialize() == models["bulk"].zbl.serialize()
        assert models["slab"].atomic_model is models["bulk"].atomic_model

    def test_a_shared_charge_head_needs_one_window(self):
        from .test_charge_same_as import build, two_theory_config

        both = two_theory_config(extra_mace={"zbl": {"window": WINDOW}})
        both["model_dict"]["ttt_dft"]["zbl"] = {"window": WINDOW}
        models, _wrapper, _cfg = build(both)
        assert all(m.zbl.serialize() == {"window": WINDOW} for m in models.values())
        for extra in ({"zbl": {"window": WINDOW}}, {"zbl": {"window": [0.6, 0.9]}}):
            mixed = two_theory_config(extra_mace=extra)
            if extra["zbl"]["window"] != WINDOW:
                mixed["model_dict"]["ttt_dft"]["zbl"] = {"window": WINDOW}
            with pytest.raises(ValueError, match="`zbl` block differs"):
                build(mixed)

    def test_a_periodic_table_type_map_builds(self):
        cfg = config()
        cfg["type_map"] = ["X", "O", "H"]
        assert get_model(cfg).zbl.numbers.tolist() == [0.0, 8.0, 1.0]


class TestRefusals:
    def test_a_window_past_the_cutoff(self):
        cfg = config()
        cfg["zbl"] = {"window": [2.0, 5.0]}  # O-O would end at 6.6 A, rcut is 6
        with pytest.raises(ValueError, match="beyond descriptor.rcut"):
            get_model(cfg)

    def test_a_tabulated_pair_potential_as_well(self):
        cfg = config()
        cfg["use_srtab"] = "unused.table"
        with pytest.raises(ValueError, match="choose one"):
            get_model(cfg)

    @pytest.mark.parametrize(
        "exclusion", [{"atom_exclude_types": [1]}, {"pair_exclude_types": [[0, 1]]}]
    )
    def test_exclusions(self, exclusion):
        with pytest.raises(ValueError, match="exclusions"):
            get_model({**config(), **exclusion})

    def test_interfaces_that_would_drop_the_term(self):
        model = get_model(config())
        with pytest.raises(NotImplementedError, match="ZBL"):
            model.forward_lower()
        with pytest.raises(NotImplementedError, match="ZBL"):
            model.forward_lower_canonical_graph()
        with pytest.raises(NotImplementedError, match="ZBL"):
            model.enable_hessian()

    def test_the_fused_inference_level(self, monkeypatch):
        """Level 2's kernel never calls the atomic model the envelope wraps."""
        monkeypatch.setenv("DP_CUDA_INFER", "2")
        model = get_model(config()).eval()
        with pytest.raises(RuntimeError, match="DP_CUDA_INFER=2"):
            model(*dimer())
        plain = get_model(config(zbl=False)).eval()
        assert torch.isfinite(plain(*dimer())["energy"]).all()

    def test_the_hessian_first_then_the_term(self):
        model = get_model(config(zbl=False))
        model.enable_hessian()
        with pytest.raises(NotImplementedError, match="ZBL"):
            model.set_zbl({"window": WINDOW})


def test_the_graph_export_keeps_the_term():
    """``dp freeze`` of a short-range model traces the lower, torch path."""
    from deepmd.dpmodel.utils.neighbor_graph import attach_edge_csr

    from deepmd_lr.neighbors import build_neighbor_graph_fast

    model = get_model(config()).eval()
    coord, atype = dimer(0.78)
    graph = attach_edge_csr(
        build_neighbor_graph_fast(coord, atype, None, model.get_rcut()), 2
    )
    counts = torch.tensor([2], device=DEVICE)
    args = (
        atype.reshape(-1),
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
    reference = model(coord, atype)
    for key in ("energy", "force", "virial"):
        torch.testing.assert_close(output[key].reshape(-1), reference[key].reshape(-1))


@pytest.mark.parametrize("lr", [False, True])
def test_cli_training_checkpoint_and_restart(tmp_path, lr):
    """``dp train`` writes the block into the checkpoint and a restart keeps it."""
    from deepmd_lr.ase import load_model

    data = tmp_path / "data"
    (data / "set.000").mkdir(parents=True)
    (data / "type.raw").write_text("0\n1\n")
    (data / "type_map.raw").write_text("O\nH\n")
    np.save(data / "set.000/box.npy", np.tile((np.eye(3) * 20).reshape(1, 9), (4, 1)))
    coordinates = np.array(
        [[[0, 0, 0], [r, 0, 0]] for r in [0.8, 0.9, 1.0, 1.1]], dtype=float
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
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    checkpoint = tmp_path / "model.ckpt.pt"
    model = load_model(str(checkpoint), device="cpu")
    assert model.zbl.serialize() == {"window": WINDOW}
    coord, atype = (v.cpu() for v in dimer())
    box = torch.eye(3, dtype=DT).unsqueeze(0) * 20
    assert torch.isfinite(model(coord, atype, box=box)["force"]).all()
    cfg["training"]["numb_steps"] = 3
    inp.write_text(json.dumps(cfg))
    proc = subprocess.run(
        [*command, "--restart", str(checkpoint)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    restored = load_model(str(tmp_path / "model_ema.ckpt.pt"), device="cpu")
    assert restored.zbl.serialize() == {"window": WINDOW}
    assert torch.isfinite(restored(coord, atype, box=box)["energy"]).all()
