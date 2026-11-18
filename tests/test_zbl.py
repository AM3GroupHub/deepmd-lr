"""Short contacts: ZBL comes in and the network goes out, across one window.

What is pinned:
- **The settings**: a ``window`` in covalent-radius sums for every pair, by
  default placed below a bond, and ``pairs`` with a window of their own; a
  window that reaches into the training data is refused when training starts;
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
- **The routes**: an evaluation takes the envelope as a plain weight per atom
  and adds the rest from a list of contacts, whichever of DeePMD-kit's routes
  evaluates the network; the list built with torch ops (a trace, a CPU) is
  the arithmetic over every edge; the per-atom virial and inference level 2
  give the numbers of level 0.
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

from deepmd_lr.zbl import DEFAULT_WINDOW, ZBL, normalized_settings

from .test_joint_model import DESCRIPTOR, DT, FITTING, LONG_RANGE, frame

#: O-H window 0.679-0.873 A, O-O 0.924-1.188 A, H-H 0.434-0.558 A.
WINDOW = [0.7, 0.9]
#: The block of these tests: that window for every pair, written out, so
#: the numbers below do not move with the default.
BLOCK = {"window": WINDOW}


def config(lr: bool = False, zbl: bool = True) -> dict:
    cfg = {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": copy.deepcopy(DESCRIPTOR),
        "fitting_net": copy.deepcopy(FITTING),
    }
    if zbl:
        cfg["zbl"] = copy.deepcopy(BLOCK)
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
    def test_an_empty_block_places_the_default_below_every_bond(self):
        zbl = ZBL(["X", "O", "H"], {})
        x_lo, x_hi = DEFAULT_WINDOW
        assert zbl.serialize() == {"window": [x_lo, x_hi]}
        types = {"O": 1, "H": 2}
        for a, b in [("O", "H"), ("O", "O"), ("H", "H")]:
            i, j = types[a], types[b]
            assert zbl.r_lo[i, j].item() == pytest.approx(x_lo * pair_sum(a, b))
            assert zbl.r_hi[i, j].item() == pytest.approx(x_hi * pair_sum(a, b))
            assert zbl.window_of(a, b, ["X", "O", "H"]) == pytest.approx(
                (x_lo * pair_sum(a, b), x_hi * pair_sum(a, b))
            )
        # The type that is no element is in no window.
        assert zbl.r_hi[0].eq(0).all() and zbl.r_hi[:, 0].eq(0).all()
        assert list(zbl.parameters()) == [] and zbl.state_dict() == {}
        assert zbl.max_outer == pytest.approx(x_hi * pair_sum("O", "O"))

    def test_pairs_have_a_window_of_their_own(self):
        written = {"O-O": [1.5, 1.75], "H-H": [1.45, 2.0], "_comment": "ignored"}
        zbl = ZBL(["O", "H"], {"window": WINDOW, "pairs": written, "_comment": "x"})
        kept = {"H-H": [1.45, 2.0], "O-O": [1.5, 1.75]}
        assert zbl.serialize() == {"window": WINDOW, "pairs": kept}
        assert ZBL(["H", "O"], zbl.serialize()).serialize() == zbl.serialize()
        o, h = 0, 1
        for (i, j), pair, window in [
            ((o, h), ("O", "H"), WINDOW),
            ((o, o), ("O", "O"), [1.5, 1.75]),
            ((h, h), ("H", "H"), [1.45, 2.0]),
        ]:
            for table, x in zip((zbl.r_lo, zbl.r_hi), window, strict=True):
                assert table[i, j].item() == pytest.approx(x * pair_sum(*pair))
                assert table[j, i].item() == table[i, j].item()
        assert zbl.max_outer == pytest.approx(1.75 * pair_sum("O", "O"))

    def test_without_a_window_only_the_pairs_have_one(self):
        zbl = ZBL(["O", "H"], {"window": None, "pairs": {"O-H": WINDOW}})
        assert zbl.serialize() == {"window": None, "pairs": {"H-O": WINDOW}}
        o, h = 0, 1
        assert zbl.r_hi[h, h].item() == 0.0 and zbl.r_hi[o, o].item() == 0.0
        graph = pair_graph(0.3, types=(h, h))
        assert zbl.atomic_energy(*graph).eq(0).all()
        assert zbl.envelope(*graph).tolist() == [1.0, 1.0]
        with pytest.raises(ValueError, match="no window"):
            ZBL(["O", "H"], {"window": None})

    def test_the_kernels_reach_follows_the_types_present(self):
        """A periodic-table type map has windows out to 4 A between its
        largest atoms; a water box is searched to its own O-O window."""
        zbl = ZBL(["X", "H", "O", "Cs", "Fr"], {})
        assert zbl.max_outer == pytest.approx(DEFAULT_WINDOW[1] * pair_sum("Fr", "Fr"))
        water = torch.tensor([2, 1, 1, -1])
        assert zbl.reach(water).sqrt().item() == pytest.approx(
            DEFAULT_WINDOW[1] * pair_sum("O", "O")
        )

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
            {"O-H": [0.7, 0.9]},
        ],
    )
    def test_bad_windows_are_refused(self, window):
        with pytest.raises(ValueError, match="window"):
            ZBL(["O", "H"], {"window": window})
        with pytest.raises(ValueError, match="pairs"):
            ZBL(["O", "H"], {"pairs": {"O-H": window}})

    @pytest.mark.parametrize(
        ("pairs", "message"),
        [
            ([0.7, 0.9], "maps pairs of elements"),
            ({"OH": WINDOW}, "not a pair of elements"),
            ({"O-H-H": WINDOW}, "not a pair of elements"),
            ({"X-O": WINDOW}, "not a pair of elements"),
            ({"O-H": WINDOW, "H-O": WINDOW}, "twice"),
            ({"Si-O": WINDOW}, "type_map has no Si"),
        ],
    )
    def test_bad_pairs_are_refused(self, pairs, message):
        with pytest.raises(ValueError, match=message):
            ZBL(["X", "O", "H"], {"pairs": pairs})

    def test_retired_keys_are_named(self):
        # Shared-head comparison must validate the same block the model builds.
        for read in (normalized_settings, lambda block: ZBL(["O", "H"], block)):
            with pytest.raises(ValueError, match="first implementation"):
                read({"clamp": True})
            with pytest.raises(ValueError, match="accepts only"):
                read({"trainable": True})

    def test_a_type_map_without_an_element_is_refused(self):
        with pytest.raises(ValueError, match="at least one element"):
            ZBL(["X", "Y0"], {})

    def test_the_schema_fills_in_the_default(self):
        def training_input(model):
            return {
                "model": model,
                "learning_rate": {"start_lr": 1e-3},
                "loss": {"type": "ener"},
                "training": {"training_data": {"systems": ["./data"]}, "numb_steps": 3},
            }

        bare = config()
        bare["zbl"] = {}
        normalized = normalize(training_input(bare))["model"]["zbl"]
        assert normalized == {"window": list(DEFAULT_WINDOW), "pairs": {}}
        built = get_model({**bare, "zbl": normalized})
        assert built.zbl.serialize() == {"window": list(DEFAULT_WINDOW)}
        assert normalize(training_input(config(zbl=False)))["model"]["zbl"] is None
        unknown = config()
        unknown["zbl"] = {"clamp": True}
        with pytest.raises(ArgumentKeyError):
            normalize(training_input(unknown))


class TestTrainingData:
    """A window that covers fitted geometries has the model trained with part
    of itself switched off: refused when the statistics are taken, with the
    pairs and a window that fits."""

    @staticmethod
    def sampled(r: float):
        coord, atype, cell = contact_frame(r)
        return [
            {
                "coord": coord.reshape(1, -1).cpu().numpy(),
                "atype": atype.cpu().numpy(),
                "box": cell.reshape(1, 9).cpu().numpy(),
            }
        ]

    def test_data_above_every_window_passes(self):
        model = get_model({**config(), "zbl": {}})
        model.check_zbl_against(self.sampled(0.96))

    def test_a_contact_grazing_the_upper_edge_passes(self):
        model = get_model({**config(), "zbl": {}})
        edge = DEFAULT_WINDOW[1] * pair_sum("O", "H")
        model.check_zbl_against(self.sampled(edge - 1e-3))

    def test_data_inside_a_window_is_refused_with_a_window_that_fits(self):
        model = get_model({**config(), "zbl": {}})
        r = 0.78  # 0.80 radius sums, inside the default 0.60-0.85
        with pytest.raises(ValueError, match="reaches into the training data") as info:
            model.check_zbl_against(self.sampled(r))
        message = str(info.value)
        assert "H-O 0.80 (window 0.60-0.85)" in message
        fits = json.loads(message.split('"pairs": ')[1])
        assert fits == {"H-O": [0.55, 0.8]}
        fitted = get_model({**config(), "zbl": {"pairs": fits}})
        fitted.check_zbl_against(self.sampled(r))

    def test_a_model_without_the_term_is_not_checked(self):
        get_model(config(zbl=False)).check_zbl_against(self.sampled(0.3))


class TestPairTerm:
    @pytest.mark.parametrize("pair", [("O", "H"), ("O", "O"), ("H", "H")])
    def test_energy_is_the_formula_for_every_pair(self, pair):
        zbl = ZBL(["O", "H"], BLOCK)
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
        zbl = ZBL(["O", "H"], BLOCK)
        energy = zbl.atomic_energy(*pair_graph(0.75)).reshape(-1)
        assert energy[0].item() == pytest.approx(energy[1].item(), rel=1e-14)
        assert energy.sum().item() == pytest.approx(pair_energy(0.75, "O", "H"))

    def test_the_edge_derivative_is_the_gradient(self):
        zbl = ZBL(["O", "H"], BLOCK)
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
        zbl = ZBL(["O", "H"], BLOCK)
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
        zbl = ZBL(["O", "H"], BLOCK)
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
        zbl = ZBL(["X", "O", "H"], BLOCK)
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
        zbl = ZBL(["O", "H"], BLOCK)
        atype, edges, vec, mask = pair_graph(0.75, types=(0, -1))
        assert zbl.atomic_energy(atype, edges, vec, mask).eq(0).all()


class TestEnvelope:
    @pytest.mark.parametrize("pair", [("O", "H"), ("O", "O"), ("H", "H")])
    def test_one_neighbour_gives_the_step(self, pair):
        zbl = ZBL(["O", "H"], BLOCK)
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
        zbl = ZBL(["O", "H"], BLOCK)
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
        zbl = ZBL(["O", "H"], BLOCK)
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
        zbl = ZBL(["O", "H"], BLOCK)
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
        zbl = ZBL(["X", "O", "H"], BLOCK)
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
        zbl = ZBL(["O", "H"], BLOCK)
        atype, edges, vec, mask = pair_graph(0.8)
        assert zbl.envelope(atype, edges, vec.to(dtype), mask).dtype == dtype


def random_graph(n_node: int = 60, n_edge: int = 80, seed: int = 3):
    """Edges of every length around the windows, with what a real graph holds
    besides: masked zero-length edges, a padded node, ghost destinations."""
    rng = torch.Generator().manual_seed(seed)
    atype = torch.randint(0, 2, (n_node,), generator=rng)
    atype[-1] = -1
    edges = torch.randint(0, n_node, (2, n_edge), generator=rng)
    direction = torch.nn.functional.normalize(
        torch.randn(n_edge, 3, generator=rng, dtype=DT), dim=1
    )
    length = 0.4 + 0.8 * torch.rand(n_edge, generator=rng, dtype=DT)
    mask = torch.rand(n_edge, generator=rng) > 0.1
    vec = direction * length[:, None]
    vec[~mask] = 0.0
    owned = torch.arange(n_node) < n_node - 5
    return atype, edges, vec, mask, owned


class TestContactList:
    """What an evaluation runs on where no kernel can: the list of contacts
    built with torch ops, which a trace records and an export compiles."""

    def dense(self, zbl, atype, edges, vec, mask, owned, cotangent):
        """The same numbers as torch arithmetic over every edge."""
        src, dst = edges[0], edges[1]
        real = (atype >= 0) & (atype < 2)
        valid = mask & real[src] & real[dst]
        e_edge, g = zbl._edge_terms_torch(atype, src, dst, vec, valid & owned[dst])
        energy = torch.zeros(atype.shape[0], dtype=DT).index_add(0, dst, e_edge)
        leaf = vec.clone().requires_grad_(True)
        envelope = zbl.envelope(atype, edges, leaf, mask)
        (dw,) = torch.autograd.grad((envelope * cotangent).sum(), leaf)
        return energy, envelope.detach(), g, dw

    def scattered(self, found, shape, cotangent):
        from deepmd_lr.zbl import add_envelope_gradient

        zero = torch.zeros(shape, dtype=DT)
        g = zero.clone().index_add_(0, found.edge, found.g)
        return g, add_envelope_gradient(zero.clone(), found, cotangent)

    def test_the_list_is_the_arithmetic_over_every_edge(self):
        zbl = ZBL(["O", "H"], BLOCK).eval()
        atype, edges, vec, mask, owned = random_graph()
        cotangent = torch.randn(
            atype.shape[0], generator=torch.Generator().manual_seed(4), dtype=DT
        )
        energy, envelope, g, dw = self.dense(
            zbl, atype, edges, vec, mask, owned, cotangent
        )
        assert not zbl.explicit(vec)
        found = zbl.contacts(atype, edges, vec, mask, owned)
        assert envelope.eq(0).any() and envelope.eq(1).any()
        assert ((envelope > 0) & (envelope < 1)).any() and dw.abs().max() > 1.0
        torch.testing.assert_close(found.energy, energy, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(found.envelope, envelope, rtol=1e-12, atol=1e-13)
        g_list, dw_list = self.scattered(found, vec.shape, cotangent)
        torch.testing.assert_close(g_list, g, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(dw_list, dw, rtol=1e-10, atol=1e-10)
        # Exactly as long as the contacts: the valid edges inside the window
        # of their own pair.
        src, dst = edges[0], edges[1]
        real = (atype >= 0) & (atype < 2)
        ti = atype.clamp(0, 1)
        inside = vec.norm(dim=1) < zbl.r_hi[ti[src], ti[dst]]
        expected = torch.nonzero(mask & real[src] & real[dst] & inside)[:, 0]
        assert found.edge.tolist() == expected.tolist()
        assert 0 < expected.numel() < int(mask.sum())

    def test_the_compact_canonical_stream_gives_the_same_list(self):
        """A compressed export is traced on DeePMD-kit's compact graph:
        every edge once, grouped by destination, the destination not
        listed but read off the row offsets."""
        zbl = ZBL(["O", "H"], BLOCK).eval()
        atype, edges, vec, mask, owned = random_graph(seed=5)
        edges, vec = edges[:, mask], vec[mask]
        order = torch.argsort(edges[1], stable=True)
        edges, vec = edges[:, order], vec[order]
        row_ptr = torch.zeros(atype.shape[0] + 1, dtype=torch.int64)
        row_ptr[1:] = torch.cumsum(
            torch.bincount(edges[1], minlength=atype.shape[0]), 0
        )
        every = torch.ones(vec.shape[0], dtype=torch.bool)
        reference = zbl.contacts(atype, edges, vec, every, owned)
        # Two slots of storage beyond the last row, as a tiny graph has.
        source = torch.cat([edges[0], torch.zeros(2, dtype=torch.int64)])
        padded = torch.cat([vec, torch.zeros(2, 3, dtype=DT)])
        found = zbl.canonical_contacts(
            atype, source.to(torch.uint32), padded.float(), row_ptr, owned
        )
        assert found.dst.tolist() == reference.dst.tolist()
        assert found.edge.tolist() == reference.edge.tolist()
        torch.testing.assert_close(found.energy, reference.energy, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            found.envelope, reference.envelope, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(found.g, reference.g, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(found.dlog, reference.dlog, rtol=1e-4, atol=1e-4)

    def test_an_empty_graph_and_no_contact_give_an_empty_list(self):
        zbl = ZBL(["O", "H"], BLOCK).eval()
        atype = torch.tensor([0, 1])
        none = zbl.contacts(
            atype,
            torch.zeros((2, 0), dtype=torch.int64),
            torch.zeros((0, 3), dtype=DT),
            torch.zeros(0, dtype=torch.bool),
        )
        far = zbl.contacts(*pair_graph(3.0))
        for found in (none, far):
            assert found.edge.numel() == 0
            assert found.energy.tolist() == [0.0, 0.0]
            assert found.envelope.tolist() == [1.0, 1.0]


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
        assert restored.zbl.serialize() == BLOCK
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
        plain.set_zbl(BLOCK)
        # In the model's own mode: a module made after `.eval()` would
        # otherwise stay in training mode and miss the inference kernel.
        assert plain.zbl.training is False
        assert plain.train().zbl.training is True and plain.eval().zbl.training is False
        assert json.loads(plain.model_def_script)["zbl"] == BLOCK
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

    @pytest.mark.parametrize("r", [0.78, 0.6], ids=["in_the_window", "below_it"])
    def test_the_per_atom_virial_takes_the_same_term(self, r):
        """Each contact's virial goes to its source, as DeePMD-kit assigns
        an edge's; the per-atom virials add up to the frame's."""
        model, plain = twins()
        coord, atype, cell = contact_frame(r)
        reference = model(coord, atype, box=cell)
        output = model(coord, atype, box=cell, do_atomic_virial=True)
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                output[key], reference[key], rtol=1e-9, atol=1e-9
            )
        torch.testing.assert_close(
            output["atom_virial"].sum(dim=1), reference["virial"], rtol=1e-9, atol=1e-9
        )
        without = plain(coord, atype, box=cell, do_atomic_virial=True)
        assert (output["atom_virial"] - without["atom_virial"]).abs().max() > 0.1

    @pytest.mark.parametrize("lr", [False, True])
    def test_inference_level_two_falls_back_with_the_term(self, monkeypatch, lr):
        """DeePMD-kit's fused operators serve a compressed model only; any
        other takes the autograd lower at level 2 too, the term with it.
        (``dp freeze`` traces a graph export at that level.)"""
        model, _plain = twins(lr)
        coord, atype, cell = contact_frame(0.78)
        monkeypatch.setenv("DP_CUDA_INFER", "0")
        reference = model(coord, atype, box=cell)
        monkeypatch.setenv("DP_CUDA_INFER", "2")
        output = model(coord, atype, box=cell)
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                output[key], reference[key], rtol=1e-9, atol=1e-9
            )

    def test_an_atom_in_contact_carries_no_charge_and_the_total_is_kept(self):
        """The envelope is the projection's weight: whatever offset the
        head's raw output has, the two atoms in contact end at exactly zero
        and the others share the deficit."""
        from .test_born import pieces

        model, plain = twins(lr=True)
        # A raw offset like a model trained without the neutrality penalty.
        for m in (model, plain):
            m.charge_fitting.bias_atom_e[:] = -0.4
        coord, atype, cell = contact_frame(0.6)
        out, raw, _cap = pieces(model, coord, atype, box=cell)
        _out, raw_plain, _cap = pieces(plain, coord, atype, box=cell)
        torch.testing.assert_close(raw, raw_plain, rtol=0, atol=1e-13)
        assert raw.mean().item() < -0.3
        charge = out["charge"][0]
        assert charge[:2].eq(0).all()
        assert charge.sum().abs().item() < 1e-12
        shift = -raw[2:].mean()
        torch.testing.assert_close(charge[2:], raw[2:] + shift, rtol=0, atol=1e-13)
        torch.testing.assert_close(
            out["charge_projection_shift"], shift.reshape(1), rtol=0, atol=1e-13
        )
        # Half way through the window the two keep a share, smoothly.
        half = model(*contact_frame(0.78)[:2], box=cell)["charge"][0]
        assert (half[:2].abs() > 0).all() and half.sum().abs().item() < 1e-12

    @pytest.mark.parametrize("training", [False, True])
    @pytest.mark.parametrize("filtered", [False, True])
    def test_a_fully_closed_charged_frame_is_invalid(self, training, filtered):
        cfg = config(lr=True)
        if filtered:
            cfg["long_range"]["charge_filter"] = [100.0, 1.0]
        model = get_model(cfg).train(training)
        coord, atype = dimer(0.6)
        for total in (0.0, 1.0, -1.0):
            state = torch.tensor([[total, 0.0]], dtype=DT, device=DEVICE)
            out = model(coord, atype, charge_spin=state)
            if total == 0:
                assert out["charge"].eq(0).all()
                assert torch.isfinite(out["energy"]).all()
                assert torch.isfinite(out["force"]).all()
            else:
                assert torch.isnan(out["charge"]).all()
                assert not torch.isfinite(out["energy"]).all()

    @pytest.mark.parametrize("training", [False, True])
    @pytest.mark.parametrize("r", [0.78, 0.6])
    def test_filter_fallback_follows_the_envelope(self, training, r):
        cfg = config(lr=True)
        cfg["long_range"]["charge_filter"] = [100.0, 1.0]
        model = get_model(cfg).train(training)
        coord, atype, cell = contact_frame(r)
        state = torch.tensor([[1.0, 0.0]], dtype=DT, device=DEVICE)
        out = model(coord, atype, box=cell, charge_spin=state)
        weight = torch.ones_like(out["charge"])
        weight[:, :2] = step(r, "O", "H")
        torch.testing.assert_close(out["charge"], weight / weight.sum(-1, keepdim=True))
        # The fallback's envelope derivative belongs in the force as well.
        h = 1e-5
        plus, minus = coord.clone(), coord.clone()
        plus[0, 1, 0] += h
        minus[0, 1, 0] -= h
        ep = model(plus, atype, box=cell, charge_spin=state)["energy"]
        em = model(minus, atype, box=cell, charge_spin=state)["energy"]
        fd = -(ep - em).reshape(()) / (2 * h)
        torch.testing.assert_close(out["force"][0, 1, 0], fd, atol=2e-6, rtol=1e-6)
        # Omitting a neutral condition skips the fallback-weight work, with
        # the same outputs as an explicit zero target through the general path.
        neutral = model(coord, atype, box=cell)
        explicit = model(coord, atype, box=cell, charge_spin=torch.zeros_like(state))
        assert neutral["charge"].eq(0).all()
        for key in ("charge", "energy", "force", "virial"):
            torch.testing.assert_close(
                neutral[key], explicit[key], rtol=1e-10, atol=1e-10
            )

    @pytest.mark.parametrize("r", [0.78, 0.6])
    def test_born_tensors_carry_the_envelopes_derivative(self, r):
        from .test_born import build, molecule, pairwise_born

        model = build(isolated=True)
        model.set_zbl(BLOCK)
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
        cfg["model_dict"]["bulk"]["zbl"] = BLOCK
        models = build_group(cfg)
        assert models["slab"].zbl.serialize() == models["bulk"].zbl.serialize()
        assert models["slab"].atomic_model is models["bulk"].atomic_model

    def test_a_shared_charge_head_needs_one_window(self):
        from .test_charge_same_as import build, two_theory_config

        both = two_theory_config(extra_mace={"zbl": BLOCK})
        both["model_dict"]["ttt_dft"]["zbl"] = BLOCK
        models, _wrapper, _cfg = build(both)
        assert all(m.zbl.serialize() == BLOCK for m in models.values())
        other = {"window": WINDOW, "pairs": {"H-O": [0.6, 0.9]}}
        for extra in ({"zbl": BLOCK}, {"zbl": other}):
            mixed = two_theory_config(extra_mace=extra)
            if extra["zbl"] is other:
                mixed["model_dict"]["ttt_dft"]["zbl"] = BLOCK
            with pytest.raises(ValueError, match="`zbl` block differs"):
                build(mixed)
        # The same block written differently is the same block: the default
        # spelled out, a pair the other way round.
        spelled = {"window": list(DEFAULT_WINDOW), "pairs": {"O-H": WINDOW}}
        same = two_theory_config(extra_mace={"zbl": spelled})
        same["model_dict"]["ttt_dft"]["zbl"] = {"pairs": {"H-O": WINDOW}}
        build(same)

    def test_a_periodic_table_type_map_builds(self):
        cfg = config()
        cfg["type_map"] = ["X", "O", "H"]
        assert get_model(cfg).zbl.numbers.tolist() == [0.0, 8.0, 1.0]


def test_the_weight_goes_onto_an_ownership_mask_only():
    """The envelope rides into DeePMD-kit's fused operators on their boolean
    ownership mask; anything else in its place is a changed interface."""
    from deepmd_lr.model.lr_model import _weighted_ownership

    envelope = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    owned = torch.tensor([True, True, False])
    assert _weighted_ownership(owned, envelope).tolist() == [1.0, 0.5, 0.0]
    for other in (torch.ones(3), torch.ones(3, dtype=torch.int64), None):
        with pytest.raises(RuntimeError, match="interface has changed"):
            _weighted_ownership(other, envelope)


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
            model.enable_hessian()

    def test_the_hessian_first_then_the_term(self):
        model = get_model(config(zbl=False))
        model.enable_hessian()
        with pytest.raises(NotImplementedError, match="ZBL"):
            model.set_zbl(BLOCK)


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
        [[[0, 0, 0], [r, 0, 0]] for r in [0.9, 1.0, 1.1, 1.2]], dtype=float
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
    assert model.zbl.serialize() == BLOCK
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
    assert restored.zbl.serialize() == BLOCK
    assert torch.isfinite(restored(coord, atype, box=box)["energy"]).all()
