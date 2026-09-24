"""Born effective charge tensors, and the force they give under a field.

The load-bearing fact about `Z` is what it is *not*. It is not the dipole
derivative: for a charge model with a global total-charge constraint the exact
`dM/dr` carries a centroid-dependent term that is unbounded and undefined under
periodic boundaries, and this formula deliberately discards it. Measured, the
two differ by ~50% on a compact molecule and the gap *is* that term. So
`F = Z.T E` is a response, not the gradient of `-M.E`, and a field run conserves
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
    def test_nonsymmetric_tensor_and_power_identity(self):
        z = torch.tensor(
            [[[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=torch.float64
        )
        e = torch.tensor([1.0, 0.0, 0.0], dtype=z.dtype)
        force = field_force(z, e)
        torch.testing.assert_close(force, z.new_tensor([[[0.0, 2.0, 0.0]]]))
        velocity = z.new_tensor([0.3, -0.7, 0.2])
        polarization_rate = z[0, 0] @ velocity
        torch.testing.assert_close(e @ polarization_rate, force[0, 0] @ velocity)

    def test_ase_field_force_uses_polarization_axis(self):
        from ase import Atoms

        from deepmd_lr.ase import LongRangeCalculator

        model = build(isolated=True)
        coord, atype = molecule()
        out = model(coord, atype, box=None)
        z = out["born_charge"][0].cpu().numpy()
        e = np.array([0.01, -0.02, 0.05])
        expected = out["force"][0].detach().cpu().numpy() + e @ z
        atoms = Atoms(["O", "H", "H"] * 2, positions=coord[0].cpu().numpy())
        atoms.calc = LongRangeCalculator(model, efield=e)
        np.testing.assert_allclose(atoms.get_forces(), expected, atol=1e-10)

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
        """`F = Z.T E` is NOT `-d(-M.E)/dr`, and this records by how much.

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
        analytic = np.einsum("iab,a->ib", z, e)

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


class TestDistributed:
    """Born charges assembled across spatial partitions.

    The interesting part is an asymmetry. `partition_graph` keeps only edges
    whose *destination* is owned, which is exactly right for the energy and the
    forces. But `Z_i` also sums over edges where atom i is the **source**, and
    those point at neighbours that may be ghosts -- so they live in the
    neighbour's partition, not in i's. No partition sees all of an atom's terms
    on its own; the scatter-add onto global ids is what completes the sum, the
    same way ghost forces are returned to their owners.

    Partition-count invariance is therefore the test that matters: a wrong
    ownership rule shows up as a mismatch between 1, 2 and 4 partitions while
    every local property still looks fine.
    """

    @staticmethod
    def _setup(n_mol=8):
        from deepmd.pt_expt.model.get_model import get_model  # noqa: PLC0415

        cfg = {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": copy.deepcopy(DESCRIPTOR),
            "fitting_net": copy.deepcopy(FITTING),
            "long_range": copy.deepcopy(LONG_RANGE),
        }
        model = get_model(cfg)
        model.eval()
        coord, atype, cell = frame(n_mol=n_mol)
        return (
            model,
            coord[0].cpu().numpy(),
            atype[0].cpu().numpy(),
            cell[0].cpu().numpy(),
            coord,
            atype,
            cell,
        )

    def test_matches_the_serial_tensor_at_every_partition_count(self):
        from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy  # noqa: PLC0415

        model, c, a, cl, coord, atype, cell = self._setup()
        model.compute_born = True
        serial = model(coord, atype, box=cell)["born_charge"][0].cpu().numpy()

        for n_part in (1, 2, 4):
            out = to_numpy(evaluate_partitioned(
                model,
                c,
                a,
                cl,
                n_partitions=n_part,
                devices=[DEVICE] * n_part,
                compute_born=True,
            ))
            got = out["born_charge"]
            assert got.shape == (len(c), 3, 3)
            assert np.abs(got - serial).max() < 1e-10, (
                f"{n_part} partitions disagree with the serial tensor by "
                f"{np.abs(got - serial).max():.3e}"
            )

    def test_off_by_default(self):
        from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy  # noqa: PLC0415

        model, c, a, cl, *_ = self._setup()
        out = to_numpy(evaluate_partitioned(
            model, c, a, cl, n_partitions=2, devices=[DEVICE] * 2
        ))
        assert "born_charge" not in out

    def test_exposed_on_the_evaluator(self):
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        model, c, a, cl, *_ = self._setup()
        ev = DistributedEvaluator(
            model, a, cl, devices=[DEVICE, DEVICE], compute_born=True
        )
        ev(torch.tensor(c, dtype=DT, device=DEVICE))
        z = ev.results["born_charge"]
        assert z.shape == (len(c), 3, 3)
        q = ev.results["charge"].sum()
        assert np.abs(z.sum(0) - q * np.eye(3)).max() < 1e-10

    def test_charge_response_force_still_works_afterwards(self):
        """Born takes its VJP over the same tape the response force needs.

        It runs first with `retain_graph=True`; if that were dropped, the
        response force would die with a freed-graph error rather than a wrong
        number -- but it would die only when Born was enabled, which is easy
        to miss.
        """
        from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy  # noqa: PLC0415

        model, c, a, cl, *_ = self._setup()
        plain = to_numpy(evaluate_partitioned(
            model, c, a, cl, n_partitions=2, devices=[DEVICE] * 2
        ))
        withz = to_numpy(evaluate_partitioned(
            model,
            c,
            a,
            cl,
            n_partitions=2,
            devices=[DEVICE] * 2,
            compute_born=True,
        ))
        assert (
            np.abs(
                withz["force_charge_response"] - plain["force_charge_response"]
            ).max()
            < 1e-12
        )
        assert np.abs(withz["force"] - plain["force"]).max() < 1e-12


class TestDistributedField:
    """A uniform external field on the partitioned path.

    The field force is built from ``Z``, and ``Z`` is the quantity that needs
    a scatter-add across partitions (see :class:`TestDistributed`). So the
    thing worth asserting is that the *force* -- not just the tensor -- comes
    out partition-invariant and equal to the serial answer.
    """

    _setup = staticmethod(TestDistributed._setup)
    FIELD = [0.03, -0.02, 0.05]

    def _evaluator(self, **kwargs):
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        model, c, a, cl, *_ = self._setup()
        return DistributedEvaluator(model, a, cl, **kwargs), c, model

    def test_matches_the_serial_field_force(self):
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        model, c, a, cl, coord, atype, cell = self._setup()
        model.compute_born = True
        out = model(coord, atype, box=cell)
        serial = (
            out["force"][0] + field_force(out["born_charge"], self.FIELD)[0]
        ).detach().cpu().numpy()

        for n_part in (1, 2, 4):
            ev = DistributedEvaluator(
                model, a, cl, devices=[DEVICE] * n_part, efield=self.FIELD
            )
            _, forces = ev(torch.tensor(c, dtype=DT, device=DEVICE))
            got = forces.detach().cpu().numpy()
            assert np.abs(got - serial).max() < 1e-10, (
                f"{n_part} partitions disagree with the serial field force by "
                f"{np.abs(got - serial).max():.3e}"
            )

    def test_net_force_is_total_charge_times_field(self):
        """`sum_i Z_i = Q I` identically, so a neutral cell feels no net force
        and cannot drift. This is the field analogue of the sum rule."""
        ev, c, _ = self._evaluator(devices=[DEVICE] * 2, efield=self.FIELD)
        _, forces = ev(torch.tensor(c, dtype=DT, device=DEVICE))
        q = float(ev.results["charge"].sum())
        expected = q * np.asarray(self.FIELD)
        net = ev.results["field_force"].sum(axis=0)
        assert np.abs(net - expected).max() < 1e-9

    def test_zero_field_leaves_the_forces_alone(self):
        """A zero field must be exactly a no-op, not merely small -- otherwise
        the field path is adding noise to every trajectory that enables it."""
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        model, c, a, cl, *_ = self._setup()
        pos = torch.tensor(c, dtype=DT, device=DEVICE)
        plain = DistributedEvaluator(model, a, cl, devices=[DEVICE] * 2)
        zero = DistributedEvaluator(
            model, a, cl, devices=[DEVICE] * 2, efield=[0.0, 0.0, 0.0]
        )
        _, f0 = plain(pos)
        base = f0.detach().cpu().numpy().copy()
        _, f1 = zero(pos)
        assert np.abs(f1.detach().cpu().numpy() - base).max() < 1e-12

    def test_field_implies_born(self):
        ev, _, _ = self._evaluator(devices=[DEVICE], efield=self.FIELD)
        assert ev.compute_born is True

    def test_field_without_a_charge_head_is_refused(self):
        from deepmd.pt_expt.model.get_model import get_model  # noqa: PLC0415

        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        model, c, a, cl, *_ = self._setup()
        short = get_model(
            {
                "type": "dp_lr",
                "type_map": ["O", "H"],
                "descriptor": copy.deepcopy(DESCRIPTOR),
                "fitting_net": copy.deepcopy(FITTING),
            }
        )
        with pytest.raises(ValueError, match="no `long_range` block"):
            DistributedEvaluator(short, a, cl, devices=[DEVICE], efield=self.FIELD)

    def test_virial_and_field_are_refused_together(self):
        """The field's contribution to the stress is a separate derivation, so
        the field-free virial would be quietly wrong rather than absent."""
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        model, c, a, cl, *_ = self._setup()
        with pytest.raises(ValueError, match="cannot be combined"):
            DistributedEvaluator(
                model, a, cl, devices=[DEVICE], efield=self.FIELD, compute_virial=True
            )
