# SPDX-License-Identifier: LGPL-3.0-or-later
"""The MD state container and its thermodynamic report.

Most of what `Thermo` does is delegate to nvalchemiops, so these tests check
the seams rather than the kernels: the conventions we depend on (ndof, units),
the aliasing invariants that would fail *silently* if broken, and the cell
sync that has no natural backstop.
"""

import numpy as np
import pytest
import torch
import warp as wp
from ase.units import kB

from deepmd_lr.nvalchemi import DistributedEvaluator, Thermo

from .test_joint_model import DEVICE, build, frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)


def setup(n_mol=8, **kwargs):
    model = build()
    coord, atype, cell = frame(n_mol=n_mol)
    ev = DistributedEvaluator(
        model, atype[0].cpu().numpy(), cell[0].cpu().numpy(),
        devices=[DEVICE], **kwargs,
    )
    return ev, coord[0].cpu().numpy()


def make_thermo(ev, positions, temperature=300.0):
    """Thermo no longer has a setpoint: velocities are drawn explicitly."""
    th = Thermo(ev, positions)
    th.maxwell_boltzmann(temperature)
    return th


class TestMeasurements:
    def test_kinetic_energy_matches_half_sum_m_v2(self):
        ev, pos = setup()
        th = make_thermo(ev, pos)
        v = th.velocities.numpy()
        m = th.masses.numpy()
        assert th.kinetic_energy == pytest.approx(
            0.5 * float((m[:, None] * v**2).sum()), rel=1e-10
        )

    def test_temperature_uses_3N_minus_3(self):
        """nvalchemiops assumes the COM motion is gone. If that ever changed
        to 3N the reported temperature would shift by 3/(3N-3) with nothing
        failing, so the convention is pinned here rather than trusted."""
        ev, pos = setup()
        th = make_thermo(ev, pos)
        ke = th.kinetic_energy
        n = th.n_atoms
        assert th.temperature == pytest.approx(2 * ke / ((3 * n - 3) * kB), rel=1e-10)
        assert th.temperature != pytest.approx(2 * ke / (3 * n * kB), rel=1e-6)

    def test_total_energy_is_the_sum(self):
        ev, pos = setup()
        th = make_thermo(ev, pos)
        th.evaluate()
        assert th.total_energy == pytest.approx(
            th.potential_energy + th.kinetic_energy, rel=1e-12
        )

    def test_pressure_matches_K_plus_W_over_V(self):
        """Assembled independently here: the kinetic tensor by hand, the
        virial from the evaluator, divided by the volume."""
        ev, pos = setup(compute_virial=True)
        th = make_thermo(ev, pos)
        th.evaluate()
        v, m = th.velocities.numpy(), th.masses.numpy()
        kinetic = np.einsum("i,ia,ib->ab", m, v, v)
        expected = np.trace(kinetic + np.asarray(ev.virial)) / (3 * ev.volume)
        assert th.pressure == pytest.approx(expected, rel=1e-8)

    def test_reading_a_measurement_leaves_the_kernel_inputs_alone(self):
        """Logging reads these getters, and the NPT thermostat and barostat
        read `ke_w` and `press_w`. A getter that refreshed those buffers let
        turning logging on change the trajectory: the next barostat half-step
        saw the logger's pressure instead of the integrator's."""
        ev, pos = setup(compute_virial=True)
        th = make_thermo(ev, pos)
        th.evaluate()
        th.update_kinetic_energy()
        th.update_pressure()
        ke_w, press_w = th.ke_w.numpy().copy(), th.press_w.numpy().copy()
        ke = th.kinetic_energy
        wp.to_torch(th.velocities).mul_(1.1)            # as a velocity kick does
        assert th.kinetic_energy == pytest.approx(1.21 * ke, rel=1e-10)
        _ = th.temperature, th.pressure, th.total_energy
        np.testing.assert_array_equal(th.ke_w.numpy(), ke_w)
        np.testing.assert_array_equal(th.press_w.numpy(), press_w)
        th.update_kinetic_energy()                       # the explicit refresh
        assert float(th.ke_w.numpy()[0]) == pytest.approx(1.21 * ke, rel=1e-10)

    def test_pressure_without_the_virial_is_refused(self):
        ev, pos = setup()
        th = make_thermo(ev, pos)
        th.evaluate()
        with pytest.raises(ValueError, match="compute_virial"):
            _ = th.pressure


class TestAliasing:
    """The failures here are silent ones: a run that looks healthy and is not."""

    def test_kernels_move_the_positions_thermo_evaluates(self):
        ev, pos = setup()
        th = make_thermo(ev, pos)
        before = th.positions.numpy().copy()
        wp.launch(
            kernel=_shift, dim=th.n_atoms, inputs=[th.positions],
            device=th.device,
        )
        after = th.positions.numpy()
        assert np.abs(after - before - 0.1).max() < 1e-12
        # and the evaluator must see the moved atoms, not the originals
        e_moved = th.evaluate()
        th2 = make_thermo(ev, after)
        assert e_moved == pytest.approx(th2.evaluate(), rel=1e-10)

    def test_forces_view_survives_repeated_evaluation(self):
        """`Thermo` binds the forces view once. That is only sound because the
        evaluator copies into its preallocated buffer; if it ever rebinds, the
        view goes stale and every kernel reads frozen forces."""
        ev, pos = setup()
        th = make_thermo(ev, pos)
        th.evaluate()
        first = th.forces.numpy().copy()
        # A *uniform* shift would not do: forces are translation invariant, so
        # they would be unchanged and the test would pass for the wrong reason.
        rng = np.random.default_rng(0)
        th._x += torch.tensor(
            rng.normal(0, 0.05, th._x.shape), dtype=torch.float64, device=th.device
        )
        th.evaluate()
        assert np.abs(th.forces.numpy() - first).max() > 1e-6, (
            "the held view did not observe the new forces"
        )

    def test_evaluator_does_not_reallocate_its_forces(self):
        ev, pos = setup()
        th = make_thermo(ev, pos)
        ptr = ev.forces.data_ptr()
        for _ in range(3):
            th.evaluate()
        assert ev.forces.data_ptr() == ptr

    def test_non_contiguous_positions_are_refused(self):
        """`.contiguous()` would return a COPY and every kernel write would
        land somewhere the caller never reads -- refuse instead of 'fixing'."""
        ev, pos = setup()
        th = make_thermo(ev, pos)
        wide = torch.zeros((th.n_atoms, 6), dtype=torch.float64, device=th.device)
        with pytest.raises(ValueError, match="contiguous"):
            th._vec(wide[:, :3])


class TestCell:
    def test_set_cell_invalidates_the_partitioning(self):
        """`_needs_rebuild` watches atom displacement only and cannot see the
        cell move, so nothing else would notice a stale halo."""
        ev, pos = setup(compute_virial=True)
        th = make_thermo(ev, pos)
        th.evaluate()
        assert ev._partitions is not None
        th.set_cell(np.asarray(ev.cell) * 1.01)
        assert ev._partitions is None

    def test_set_cell_syncs_volume_and_evaluator(self):
        ev, pos = setup(compute_virial=True)
        th = make_thermo(ev, pos)
        v0 = th.volume
        th.set_cell(np.asarray(ev.cell) * 1.01)
        assert th.volume == pytest.approx(v0 * 1.01**3, rel=1e-9)
        assert np.allclose(th.cell.numpy()[0], ev.cell)


class TestSetup:
    def test_maxwell_boltzmann_reaches_the_target_without_drift(self):
        ev, pos = setup(n_mol=64)
        th = make_thermo(ev, pos)
        assert th.temperature == pytest.approx(300.0, rel=0.25)
        v, m = th.velocities.numpy(), th.masses.numpy()
        assert np.abs((m[:, None] * v).sum(axis=0)).max() < 1e-10

    def test_atoms_snapshot_round_trips(self):
        ev, pos = setup()
        th = make_thermo(ev, pos)
        a = th.atoms()
        assert len(a) == th.n_atoms
        assert np.allclose(a.get_positions(), th.positions.numpy())
        assert np.allclose(a.get_velocities(), th.velocities.numpy())
        assert np.allclose(np.asarray(a.get_cell()), ev.cell)
        assert np.allclose(a.get_masses(), th.masses.numpy())


@wp.kernel
def _shift(x: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    x[i] = x[i] + wp.vec3d(0.1, 0.1, 0.1)


class TestStateGuards:
    """What lets several MD objects share one evaluator, one after another."""

    def test_fresh_tells_whose_forces_are_in_the_buffer(self):
        ev, pos = setup()
        a, b = make_thermo(ev, pos), make_thermo(ev, pos + 0.01)
        assert not a.fresh
        a.evaluate()
        assert a.fresh and not b.fresh
        b.evaluate()
        assert b.fresh and not a.fresh

    def test_cell_is_current_until_someone_else_moves_it(self):
        ev, pos = setup(compute_virial=True)
        a, b = make_thermo(ev, pos), make_thermo(ev, pos)
        assert a.cell_is_current
        b.set_cell(np.asarray(ev.cell) * 1.01)
        assert b.cell_is_current and not a.cell_is_current

    def test_set_state_copies_into_the_bound_buffers(self):
        ev, pos = setup()
        th = make_thermo(ev, pos)
        views = (th.positions.ptr, th.velocities.ptr)
        th.set_state(positions=pos + 0.5, velocities=np.zeros_like(pos))
        assert (th.positions.ptr, th.velocities.ptr) == views
        np.testing.assert_allclose(th.positions.numpy(), pos + 0.5)
        assert th.kinetic_energy == 0.0

    def test_remove_com_motion(self):
        ev, pos = setup()
        th = Thermo(ev, pos, velocities=np.ones_like(pos))
        th.remove_com_motion()
        v, m = th.velocities.numpy(), th.masses.numpy()
        assert np.abs((m[:, None] * v).sum(axis=0)).max() < 1e-10

    def test_the_keywords_that_moved_are_refused_by_name(self):
        ev, pos = setup()
        with pytest.raises(TypeError, match="timestep"):
            Thermo(ev, pos, timestep=0.5)
