"""Born effective charge tensors, and the force they give under a field.

The load-bearing fact about `Z` is what it is *not*. It is not the dipole
derivative: for a charge model with a global total-charge constraint the exact
`dM/dr` carries a centroid-dependent term that is unbounded and undefined under
periodic boundaries, and this formula deliberately discards it. Measured, the
two differ by ~50% on a compact molecule and the gap *is* that term. So
`F = Z E` is a response, not the gradient of `-M.E`, and a field run conserves
no corresponding energy. `test_field_force_is_not_conservative` pins that down
so nobody later reads the discrepancy as a bug.

The acoustic sum rule is asserted here but proves nothing -- see its docstring.
"""

import copy

import numpy as np
import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.born import field_force

from .test_joint_model import DESCRIPTOR, FITTING, LONG_RANGE, frame

DT = torch.float64


def build(isolated: bool = False):
    cfg = {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": copy.deepcopy(DESCRIPTOR),
        "fitting_net": copy.deepcopy(FITTING),
        "long_range": {
            "charge_net": copy.deepcopy(LONG_RANGE["charge_net"]),
            "reciprocal": (
                {"method": "isolated", "alpha": 0.4}
                if isolated
                else copy.deepcopy(LONG_RANGE["reciprocal"])
            ),
        },
    }
    model = get_model(cfg)
    model.eval()
    model.compute_born = True
    return model


def molecule(n_mol: int = 2, seed: int = 0):
    """A compact isolated cluster -- no cell, so `M` is well defined."""
    rng = np.random.default_rng(seed)
    coords = []
    for _ in range(n_mol):
        o = rng.uniform(0.0, 3.0, size=3)
        coords += [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.93, 0.0]]
    coord = torch.tensor(np.array(coords), dtype=DT, device=DEVICE).unsqueeze(0)
    atype = torch.tensor([0, 1, 1] * n_mol, dtype=torch.int64, device=DEVICE).unsqueeze(
        0
    )
    return coord, atype


def pieces(model, coord, atype, box=None):
    """Z, the charges, and the raw charges, from one captured forward."""
    with model._capturing_backbone() as cap:
        out = model(coord, atype, box=box)
    raw = model.charge_fitting.call_graph(
        cap["features"], atype.reshape(-1), gr=cap["rot_mat"]
    )["charge"].reshape(-1)
    return out, raw, cap


class TestAgainstBruteForce:
    """The check that actually validates the assembly."""

    def test_matches_a_pairwise_assembly_from_finite_differences(self):
        """One VJP + two segment sums must equal the formula summed by hand.

        The segment-sum form is a rewriting of the pair sum that exploits the
        descriptor being one-hop. If that assumption were wrong, or an index
        swapped src for dst, this is what would catch it.
        """
        model = build(isolated=True)
        coord, atype = molecule()
        n = coord.shape[1]
        base = coord[0].cpu().numpy()

        out, raw, cap = pieces(model, coord, atype)
        fast = out["born_charge"][0].cpu().numpy()
        q = out["charge"][0].detach().cpu().numpy()

        def raw_charges(pos):
            c = torch.tensor(pos.reshape(1, -1, 3), dtype=DT, device=DEVICE)
            with model._capturing_backbone() as cp:
                model(c, atype, box=None)
            return (
                model.charge_fitting.call_graph(
                    cp["features"], atype.reshape(-1), gr=cp["rot_mat"]
                )["charge"]
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )

        h = 1e-6
        jac = np.zeros((n, n, 3))  # jac[i, k] = d raw_i / d r_k
        for k in range(n):
            for b in range(3):
                plus, minus = base.copy(), base.copy()
                plus[k, b] += h
                minus[k, b] -= h
                jac[:, k, b] = (raw_charges(plus) - raw_charges(minus)) / (2 * h)

        slow = np.zeros((n, 3, 3))
        for i in range(n):
            slow[i] = q[i] * np.eye(3)
            for j in range(n):
                if j == i:
                    continue
                r_ji = base[j] - base[i]
                slow[i] += 0.5 * (
                    np.outer(r_ji, jac[i, j]) - np.outer(-r_ji, jac[j, i])
                )
        assert np.abs(fast - slow).max() < 1e-7, np.abs(fast - slow).max()


class TestSymmetries:
    def test_translation_invariant(self):
        model = build(isolated=True)
        coord, atype = molecule()
        a = pieces(model, coord, atype)[0]["born_charge"]
        shift = torch.tensor([3.7, -1.2, 0.8], dtype=DT, device=DEVICE)
        b = pieces(model, coord + shift, atype)[0]["born_charge"]
        assert torch.allclose(a, b, atol=1e-12)

    def test_rotates_as_a_rank_two_tensor(self):
        """Z -> R Z R^T. A vector or a scalar would fail this."""
        model = build(isolated=True)
        coord, atype = molecule()
        z = pieces(model, coord, atype)[0]["born_charge"][0]
        theta = 0.7
        rot = torch.tensor(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=DT,
            device=DEVICE,
        )
        turned = pieces(model, coord @ rot.T, atype)[0]["born_charge"][0]
        assert torch.allclose(turned, rot @ z @ rot.T, atol=1e-10)

    def test_is_not_symmetric(self):
        """A Born tensor has no reason to be symmetric.

        Worth pinning: the obvious thing to reuse would be DeePMD-kit's
        `polarizability` fitting, which symmetrizes unconditionally. If a
        future refactor routed Z through that, this is what would object.
        """
        model = build(isolated=True)
        coord, atype = molecule()
        z = pieces(model, coord, atype)[0]["born_charge"]
        assert (z - z.transpose(-1, -2)).abs().max() > 1e-3

    def test_acoustic_sum_rule(self):
        """Sum_i Z_i = Q.I.

        This is a REGRESSION GUARD, NOT A VALIDATION. It follows from the
        `dst - src` bracket telescoping, so it holds to machine precision for
        any per-edge derivative whatsoever -- during development it passed for
        implementations that disagreed with each other by 100%. Only
        `TestAgainstBruteForce` checks the derivative itself.
        """
        for isolated in (True, False):
            model = build(isolated=isolated)
            if isolated:
                coord, atype = molecule()
                box = None
            else:
                coord, atype, box = frame(n_mol=3)
            out = pieces(model, coord, atype, box)[0]
            z = out["born_charge"][0]
            q = out["charge"].detach().sum()
            eye = torch.eye(3, dtype=DT, device=DEVICE)
            assert (z.sum(0) - q * eye).abs().max() < 1e-10


class TestField:
    def test_force_sums_to_total_charge_times_field(self):
        """A neutral system feels no net force, so it cannot drift."""
        model = build(isolated=True)
        coord, atype = molecule()
        out = pieces(model, coord, atype)[0]
        e = torch.tensor([0.01, -0.02, 0.05], dtype=DT, device=DEVICE)
        f = field_force(out["born_charge"], e)
        net = f.sum(dim=1)[0]
        expected = out["charge"].detach().sum() * e
        assert torch.allclose(net, expected, atol=1e-12)

    def test_field_force_is_not_conservative(self):
        """`F = Z E` is NOT `-d(-M.E)/dr`, and this records by how much.

        Not a defect -- `Z` deliberately omits the centroid-dependent part of
        `dM/dr`, which is the only part that is ill defined in a periodic cell.
        The test exists so that the discrepancy is a documented number rather
        than a surprise during a field MD run, and so that a future change
        making them agree does not pass unnoticed.
        """
        model = build(isolated=True)
        coord, atype = molecule()
        base = coord[0].cpu().numpy()
        n = base.shape[0]
        e = np.array([0.0, 0.0, 0.05])

        z = pieces(model, coord, atype)[0]["born_charge"][0].cpu().numpy()
        analytic = np.einsum("iab,b->ia", z, e)

        def minus_m_dot_e(pos):
            c = torch.tensor(pos.reshape(1, -1, 3), dtype=DT, device=DEVICE)
            q = model(c, atype, box=None)["charge"].detach().cpu().numpy()[0]
            return -float((q[:, None] * pos).sum(0) @ e)

        h = 1e-5
        numeric = np.zeros((n, 3))
        for i in range(n):
            for b in range(3):
                plus, minus = base.copy(), base.copy()
                plus[i, b] += h
                minus[i, b] -= h
                numeric[i, b] = -(minus_m_dot_e(plus) - minus_m_dot_e(minus)) / (2 * h)
        gap = np.abs(analytic - numeric).max()
        scale = np.abs(numeric).max()
        # They agree in structure but not in value. Assert the gap is real
        # rather than asserting a brittle exact number.
        assert gap > 1e-6 * scale, (
            "Z now matches dM/dr -- if that is intended, the module docstring "
            "and the field-MD caveat both need rewriting."
        )
        assert gap < 10.0 * max(scale, 1e-12)


class TestGuards:
    def test_off_by_default(self):
        """The extra VJP is not paid unless asked for."""
        model = build(isolated=True)
        model.compute_born = False
        coord, atype = molecule()
        assert "born_charge" not in model(coord, atype, box=None)

    def test_refuses_when_the_backbone_was_not_captured(self):
        """No capture means no edge graph, so no cheap Z. Refuse, never guess."""
        model = build(isolated=True)
        coord, atype = molecule()
        # The class dict, not getattr: `_usable` is a staticmethod, and
        # getattr unwraps it to a plain function, so restoring that would
        # re-bind it as an instance method and break every later test.
        original = type(model).__dict__["_usable"]
        try:
            type(model)._usable = staticmethod(lambda captured: None)
            with pytest.raises(NotImplementedError, match="shared backbone"):
                model(coord, atype, box=None)
        finally:
            type(model)._usable = original

    def test_raw_charges_are_still_optional(self):
        """`_charges_from_features` must keep its single-value contract."""
        model = build(isolated=True)
        coord, atype = molecule()
        with model._capturing_backbone() as cap:
            model(coord, atype, box=None)
        mask = torch.ones_like(atype, dtype=torch.bool)
        one = model._charges_from_features(
            cap["features"], cap["rot_mat"], atype, None, mask
        )
        assert isinstance(one, torch.Tensor)
        pair = model._charges_from_features(
            cap["features"], cap["rot_mat"], atype, None, mask, return_raw=True
        )
        assert isinstance(pair, tuple) and len(pair) == 2


class TestAtScale:
    def test_large_periodic_system(self):
        """576 atoms, past the neighbour-backend size threshold.

        `neighbors.py` switches builder above 512 atoms, and Z is assembled
        from whichever graph the energy path captured. A brute-force Jacobian
        is out of reach here (3456 forwards), so this checks the properties
        that would break on an indexing or shape error at scale.
        """
        model = build(isolated=False)
        coord, atype, box = frame(n_mol=192)
        assert coord.shape[1] == 576
        out = model(coord, atype, box=box)
        z = out["born_charge"]
        assert z.shape == (1, 576, 3, 3)
        assert torch.isfinite(z).all()
        eye = torch.eye(3, dtype=DT, device=DEVICE)
        q = out["charge"].detach().sum()
        assert (z[0].sum(0) - q * eye).abs().max() < 1e-8
        assert (z - z.transpose(-1, -2)).abs().max() > 1e-3
