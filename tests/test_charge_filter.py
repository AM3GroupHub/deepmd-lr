"""The simulation-only charge filter: small charges switched off, smoothly.

What is pinned:
- **The filter itself**: exactly zero below c_t - delta, the input above
  c_t + delta (bar the restoring share), the total charge kept for neutral and
  charged frames, padding untouched, and a frame with nothing above the
  threshold handled as the plain projection would.
- **Smoothness**: the weight is C^3, so forces and the Hessian are continuous.
- **Consistency**: with the filter on, the evaluator's forces are minus the
  gradient of its energy (finite differences), on one and on two partitions,
  and the serial model, the evaluator and the ASE calculator agree.
- **Off means off**, and the combinations it cannot honour are refused.
"""

import copy

import numpy as np
import pytest
import torch
from ase import Atoms
from deepmd.pt_expt.model.get_model import get_model

from deepmd_lr.ase import LongRangeCalculator
from deepmd_lr.fitting.charge import (
    charge_filter_weight,
    check_charge_filter,
    filter_charges,
)
from deepmd_lr.nvalchemi import DistributedEvaluator

from .test_joint_model import DESCRIPTOR, DEVICE, FITTING, LONG_RANGE, build, frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)


class TestFilter:
    def test_small_charges_are_exactly_zero_and_the_total_is_kept(self):
        q = torch.tensor([[1e-4, -6e-4, 2.6e-3, 4e-3, -1e-2, 0.0]], dtype=torch.float64)
        q[0, -1] = -q[0, :-1].sum()
        out = filter_charges(q, 0.0, (3.5e-3, 1.5e-3))
        assert out[0, 0] == 0 and out[0, 1] == 0
        assert abs(float(out.sum())) < 1e-15
        assert torch.all(torch.sign(out[0, 3:5]) == torch.sign(q[0, 3:5]))

    @pytest.mark.parametrize("total", [0.0, 1.0, -2.0])
    def test_a_charged_frame_keeps_its_total(self, total):
        rng = np.random.default_rng(0)
        q = torch.tensor(rng.normal(0, 2e-3, (3, 40)), dtype=torch.float64)
        # The total sits on a few large charges, as in a charged frame.
        q[:, :4] += (total - q.sum(-1, keepdim=True)) / 4 + torch.tensor(
            [0.3, -0.3, 0.2, -0.2]
        )
        out = filter_charges(q, total, (3e-3, 1e-3))
        torch.testing.assert_close(
            out.sum(-1), torch.full((3,), total, dtype=torch.float64)
        )
        small = q.abs() <= 2e-3
        assert small.any() and not out[small].any()

    def test_padding_takes_no_part(self):
        q = torch.tensor([[0.01, -0.01, 0.5]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        out = filter_charges(q, 0.0, (3e-3, 1e-3), mask=mask)
        assert out[0, 2] == 0 and abs(float(out.sum())) < 1e-15

    def test_nothing_above_threshold_spreads_the_total_uniformly(self):
        q = torch.full((1, 4), 0.25, dtype=torch.float64) * 1e-3
        out = filter_charges(q, 1e-3, (3e-3, 1e-3))
        torch.testing.assert_close(
            out, torch.full((1, 4), 0.25e-3, dtype=torch.float64)
        )

    def test_the_weight_is_c3(self):
        """Value and first three derivatives continuous at both ends of the
        transition: zero at the lower end, (1, 0, 0, 0) at the upper."""
        c_t, delta = 3e-3, 1e-3
        for x0, want in ((c_t - delta, 0.0), (c_t + delta, 1.0)):
            x = torch.tensor([x0], dtype=torch.float64, requires_grad=True)
            y = charge_filter_weight(x, c_t, delta)
            derivs = [float(y)]
            g = y
            for _ in range(3):
                (g,) = torch.autograd.grad(g.sum(), x, create_graph=True)
                derivs.append(float(g) * (2 * delta) ** (len(derivs)))  # in units of u
            assert derivs[0] == pytest.approx(want, abs=1e-12)
            assert max(abs(d) for d in derivs[1:]) < 1e-9

    @pytest.mark.parametrize(
        "bad", [(0.0, 0.0), (1e-3, 2e-3), (1e-3, -1e-4), (float("nan"), 1e-4), "x"]
    )
    def test_bad_settings_are_refused(self, bad):
        with pytest.raises(ValueError):
            check_charge_filter(bad)


def system(n_mol=8, box=12.0):
    coord, atype, cell = frame(n_mol=n_mol, box=box)
    return coord[0].cpu().numpy(), atype[0].cpu().numpy(), cell[0].cpu().numpy()


def threshold_for(model, coord, atype, cell):
    """(c_t, delta) at the fixture's median |q|, so some atoms go and some stay."""
    ev = DistributedEvaluator(
        model, atype, cell, devices=[DEVICE], memory_headroom=None
    )
    ev(coord)
    q = np.abs(np.asarray(ev.results["charge"]))
    c_t = float(np.median(q))
    return c_t, 0.5 * c_t


class TestEvaluator:
    @pytest.mark.parametrize("n", [1, 2])
    def test_forces_are_the_gradient_of_the_filtered_energy(self, n):
        model = build()
        coord, atype, cell = system()
        filt = threshold_for(model, coord, atype, cell)
        ev = DistributedEvaluator(
            model,
            atype,
            cell,
            devices=[DEVICE] * n,
            memory_headroom=None,
            charge_filter=filt,
        )
        _, force = ev(coord)
        force = force.cpu().numpy().copy()
        q = np.asarray(ev.results["charge"])
        assert (q == 0).any() and (q != 0).any()
        h = 1e-5
        for i in (0, 5, 13):
            for a in range(3):
                plus, minus = coord.copy(), coord.copy()
                plus[i, a] += h
                minus[i, a] -= h
                fd = -(ev(plus)[0] - ev(minus)[0]) / (2 * h)
                assert force[i, a] == pytest.approx(fd, abs=2e-6)

    def test_serial_partitioned_and_ase_agree(self):
        model = build()
        coord, atype, cell = system()
        filt = threshold_for(model, coord, atype, cell)
        ev = DistributedEvaluator(
            model,
            atype,
            cell,
            devices=[DEVICE] * 2,
            memory_headroom=None,
            charge_filter=filt,
        )
        e_ev, f_ev = ev(coord)
        q_ev = np.asarray(ev.results["charge"])
        type_map = list(model.get_type_map())
        atoms = Atoms(
            [type_map[t] for t in atype], positions=coord, cell=cell, pbc=True
        )
        atoms.calc = LongRangeCalculator(model, charge_filter=filt)
        assert atoms.get_potential_energy() == pytest.approx(e_ev, abs=1e-8)
        np.testing.assert_allclose(atoms.get_forces(), f_ev.cpu().numpy(), atol=1e-8)
        np.testing.assert_allclose(atoms.calc.results["charges"], q_ev, atol=1e-10)

    def test_off_changes_nothing(self):
        model = build()
        coord, atype, cell = system()
        plain = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        off = DistributedEvaluator(
            model,
            atype,
            cell,
            devices=[DEVICE],
            memory_headroom=None,
            charge_filter=None,
        )
        # Not bit-equal: the gather and the reciprocal sum use atomics, so two
        # identical evaluations differ in the last bit from run to run.
        assert plain(coord)[0] == pytest.approx(off(coord)[0], abs=1e-9)

    def test_born_and_fields_are_refused(self):
        model = build()
        coord, atype, cell = system()
        with pytest.raises(ValueError, match="Born"):
            DistributedEvaluator(
                model,
                atype,
                cell,
                devices=[DEVICE],
                memory_headroom=None,
                compute_born=True,
                charge_filter=(1e-3, 5e-4),
            )
        with pytest.raises(ValueError, match="Born"):
            DistributedEvaluator(
                model,
                atype,
                cell,
                devices=[DEVICE],
                memory_headroom=None,
                efield=[0.1, 0, 0],
                charge_filter=(1e-3, 5e-4),
            )

    def test_a_model_without_charges_is_refused(self):
        coord, atype, cell = system()
        with pytest.raises(ValueError, match="long_range"):
            DistributedEvaluator(
                build(long_range=False),
                atype,
                cell,
                devices=[DEVICE],
                memory_headroom=None,
                charge_filter=(1e-3, 5e-4),
            )


def test_a_runtime_override_is_refused_in_training():
    """A simulation-time override in training would train one model and
    record another; a filter meant for training goes in the config."""
    model = build()
    model.train()
    model.charge_filter = (1e-3, 5e-4)
    coord, atype, cell = frame()
    with pytest.raises(RuntimeError, match="long_range.charge_filter"):
        model(coord, atype, box=cell)


def build_trained(filt):
    """The fixture model with `long_range.charge_filter` in its config."""
    cfg = {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": copy.deepcopy(DESCRIPTOR),
        "fitting_net": copy.deepcopy(FITTING),
        "long_range": {**copy.deepcopy(LONG_RANGE), "charge_filter": list(filt)},
    }
    model = get_model(cfg)
    model.eval()
    return model


class TestPartOfTheModel:
    """`long_range.charge_filter`: applied in training and simulation alike,
    saved with the model, and the default of every simulation path."""

    def filt(self):
        model = build()
        coord, atype, cell = system()
        return threshold_for(model, coord, atype, cell)

    def test_the_config_sets_it_and_a_round_trip_keeps_it(self):
        filt = self.filt()
        model = build_trained(filt)
        assert model.trained_charge_filter == pytest.approx(filt)
        again = type(model).deserialize(model.serialize())
        assert again.trained_charge_filter == pytest.approx(filt)

    def test_simulation_uses_it_by_default_and_false_switches_it_off(self):
        filt = self.filt()
        coord, atype, cell = system()
        trained = build_trained(filt)
        plain = build()
        trained.load_state_dict(plain.state_dict())

        def run(model, **kw):
            ev = DistributedEvaluator(
                model, atype, cell, devices=[DEVICE], memory_headroom=None, **kw
            )
            e, _ = ev(coord)
            return e, np.asarray(ev.results["charge"])

        e_default, q_default = run(trained)
        e_explicit, q_explicit = run(plain, charge_filter=filt)
        assert e_default == pytest.approx(e_explicit, abs=1e-10)
        np.testing.assert_allclose(q_default, q_explicit, atol=1e-12)
        assert (q_default == 0).any()
        e_off, q_off = run(trained, charge_filter=False)
        e_none, _ = run(plain)
        assert e_off == pytest.approx(e_none, abs=1e-10)
        assert not (q_off == 0).any()

    def test_the_ase_calculator_uses_it_by_default(self):
        filt = self.filt()
        coord, atype, cell = system()
        trained = build_trained(filt)
        trained.load_state_dict(build().state_dict())
        ev = DistributedEvaluator(
            trained, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        e_ev, _ = ev(coord)
        type_map = list(trained.get_type_map())
        atoms = Atoms(
            [type_map[t] for t in atype], positions=coord, cell=cell, pbc=True
        )
        atoms.calc = LongRangeCalculator(trained)
        assert atoms.get_potential_energy() == pytest.approx(e_ev, abs=1e-8)
        assert (atoms.calc.results["charges"] == 0).any()

    def test_it_trains(self):
        """The filter is in the graph a training step differentiates twice
        (forces in the loss): a backward through it reaches the charge head."""
        filt = self.filt()
        model = build_trained(filt)
        model.train()
        coord, atype, cell = frame(n_mol=8, box=12.0)
        out = model(coord, atype, box=cell)
        q = out["charge"].detach()
        assert (q == 0).any() and (q != 0).any()
        (out["force"] ** 2).sum().backward()
        grads = [
            p.grad for p in model.charge_fitting.parameters() if p.grad is not None
        ]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0
