"""The ASE-style MD classes: deepmd_lr.nvalchemi.md and .optimize.

The float64 fixture model runs on the GPU. What is pinned:
- **The ownership split.** The MD object builds its own Thermo and owns time
  and the ensemble. One evaluator is shared, by construction, and safely
  across objects that run one after another.
- **Faithful integrators.** Each class reproduces a hand-written kernel loop to
  1e-10. These loops are what `examples/06-md_nvalchemi` did by hand before
  the classes existed.
- **Run control.** run/irun/attach/stop, the log, the trajectory and
  checkpoints.
- **Physics.** Each class keeps its promise: NVE and MTK conservation, the
  Nose-Hoover eta/2 quirk, and 3N - 3 degrees of freedom.
- **FIRE's refusals.**
"""

import json
import math

import ase.io
import numpy as np
import pytest
import torch
import warp as wp
from ase import Atoms, units
from ase.constraints import FixAtoms
from nvalchemiops.dynamics.integrators import (
    langevin_baoab_finalize,
    langevin_baoab_half_step,
    nhc_compute_chain_energy,
    nhc_compute_masses,
    nhc_position_update,
    nhc_thermostat_chain_update,
    nhc_velocity_half_step,
    npt_barostat_half_step,
    npt_cell_update,
    npt_position_update,
    npt_thermostat_half_step,
    npt_velocity_half_step,
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)

from deepmd_lr.nvalchemi import (
    FIRE,
    DistributedEvaluator,
    IsotropicMTKNPT,
    Langevin,
    MDInstabilityError,
    NoseHooverChainNVT,
    RelaxationError,
    Thermo,
    VelocityVerlet,
    maxwell_boltzmann,
)
from deepmd_lr.nvalchemi import optimize as optimize_module

from .test_joint_model import DEVICE, build, frame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the MD kernels need a CUDA device"
)

DT = 0.25 * units.fs
T0 = 300.0


@pytest.fixture(scope="module")
def model():
    m = build(long_range=True)
    m.eval()
    return m


def make_atoms(model, seed=0, temperature=T0):
    coord, atype, cell = frame(n_mol=16, box=14.0, seed=seed)
    type_map = list(model.get_type_map())
    atoms = Atoms(
        [type_map[t] for t in atype[0].tolist()],
        positions=coord[0].cpu().numpy(),
        cell=cell[0].cpu().numpy(),
        pbc=True,
    )
    maxwell_boltzmann(atoms, temperature_K=temperature, seed=seed + 1)
    return atoms


def make_ev(model, atoms, n=1, **kwargs):
    type_map = list(model.get_type_map())
    atype = np.array([type_map.index(s) for s in atoms.get_chemical_symbols()])
    return DistributedEvaluator(
        model,
        atype,
        atoms.cell.array,
        devices=[DEVICE] * n,
        memory_headroom=None,
        **kwargs,
    )


# --- the kernel loops the classes replace (copied from the examples) ----------


def _thermo(ev, atoms):
    th = Thermo(
        ev,
        atoms.get_positions(),
        velocities=atoms.get_velocities(),
        masses=atoms.get_masses(),
    )
    th.remove_com_motion()
    th.evaluate()
    return th


def loop_vv(ev, atoms, steps):
    th = _thermo(ev, atoms)
    dt = th.scalar(DT)
    for _ in range(steps):
        velocity_verlet_position_update(
            th.positions, th.velocities, th.forces, th.masses, dt
        )
        th.evaluate()
        velocity_verlet_velocity_finalize(th.velocities, th.forces, th.masses, dt)
    return th


def loop_langevin(ev, atoms, steps, friction, seed):
    th = _thermo(ev, atoms)
    dt, kT, fr = th.scalar(DT), th.scalar(units.kB * T0), th.scalar(friction)
    for step in range(steps):
        langevin_baoab_half_step(
            th.positions, th.velocities, th.forces, th.masses, dt, kT, fr, seed + step
        )
        th.evaluate()
        langevin_baoab_finalize(th.velocities, th.forces, th.masses, dt)
    return th


def loop_nhc(ev, atoms, steps, tdamp, chain=3):
    th = _thermo(ev, atoms)
    dt, kT = th.scalar(DT), th.scalar(units.kB * T0)
    ndof = 3 * th.n_atoms - 3
    eta, eta_dot, mass = th.zeros(chain), th.zeros(chain), th.zeros(chain)
    nhc_compute_masses(
        wp.array(np.array([ndof]), dtype=wp.int32, device=th.device),
        kT,
        th.scalar(tdamp),
        chain,
        mass,
    )
    ke2, scale, step_scale, dt_chain = (
        th.zeros(1),
        th.scalar(1.0),
        th.zeros(1),
        th.zeros(1),
    )

    def half():
        scale.fill_(1.0)
        nhc_thermostat_chain_update(
            th.velocities,
            th.masses,
            eta,
            eta_dot,
            mass,
            kT,
            dt,
            th.scalar(float(ndof)),
            ke2,
            scale,
            step_scale,
            dt_chain,
            nloops=3,
        )

    for _ in range(steps):
        half()
        nhc_velocity_half_step(th.velocities, th.forces, th.masses, dt)
        nhc_position_update(th.positions, th.velocities, dt)
        th.evaluate()
        nhc_velocity_half_step(th.velocities, th.forces, th.masses, dt)
        half()
    return th


def loop_npt(ev, atoms, steps, pressure, tdamp, pdamp, chain=3):
    th = _thermo(ev, atoms)
    dt, kT = th.scalar(DT), th.scalar(units.kB * T0)
    ndof = 3 * th.n_atoms - 3
    eta, eta_dot = th.zeros((1, chain)), th.zeros((1, chain))
    q = np.full((1, chain), units.kB * T0 * tdamp**2)
    q[0, 0] *= ndof
    th_mass = wp.array(q, dtype=wp.float64, device=th.device)
    cell_v = th.zeros(1, dtype=wp.mat33d)
    cell_mass = th.scalar(3 * th.n_atoms * units.kB * T0 * pdamp**2)
    target = th.scalar(pressure)
    count = wp.array([th.n_atoms - 1], dtype=wp.int32, device=th.device)
    for _ in range(steps):
        th.update_kinetic_energy()
        th.update_pressure()
        npt_thermostat_half_step(eta, eta_dot, th.ke_w, kT, th_mass, count, chain, dt)
        npt_barostat_half_step(
            cell_v, th.press_w, target, th.volume_w, cell_mass, th.ke_w, count, dt
        )
        npt_velocity_half_step(
            th.velocities,
            th.masses,
            th.forces,
            cell_v,
            eta_dots=eta_dot,
            num_atoms=count,
            dt=dt,
        )
        npt_position_update(th.positions, th.velocities, th.cell, cell_v, dt)
        npt_cell_update(th.cell, cell_v, dt)
        th.set_cell()
        th.evaluate()
        npt_velocity_half_step(
            th.velocities,
            th.masses,
            th.forces,
            cell_v,
            eta_dots=eta_dot,
            num_atoms=count,
            dt=dt,
        )
        th.update_kinetic_energy()
        th.update_pressure()
        npt_barostat_half_step(
            cell_v, th.press_w, target, th.volume_w, cell_mass, th.ke_w, count, dt
        )
        npt_thermostat_half_step(eta, eta_dot, th.ke_w, kT, th_mass, count, chain, dt)
    return th


FRICTION = 0.01 / units.fs
TDAMP, PDAMP, PRESSURE = 100 * units.fs, 1000 * units.fs, 1e-4 * units.GPa

CASES = {
    "vv": (
        lambda ev, a, **kw: VelocityVerlet(ev, a, DT, **kw),
        lambda ev, a, n: loop_vv(ev, a, n),
        {},
    ),
    "langevin": (
        lambda ev, a, **kw: Langevin(
            ev, a, DT, T0, FRICTION, fixcm=False, seed=7, **kw
        ),
        lambda ev, a, n: loop_langevin(ev, a, n, FRICTION, 7),
        {},
    ),
    "nhc": (
        lambda ev, a, **kw: NoseHooverChainNVT(ev, a, DT, T0, TDAMP, **kw),
        lambda ev, a, n: loop_nhc(ev, a, n, TDAMP),
        {},
    ),
    "npt": (
        lambda ev, a, **kw: IsotropicMTKNPT(
            ev, a, DT, T0, PRESSURE, TDAMP, PDAMP, **kw
        ),
        lambda ev, a, n: loop_npt(ev, a, n, PRESSURE, TDAMP, PDAMP),
        {"compute_virial": True},
    ),
}


# --- the ownership split and the shared evaluator ----------------------------


class TestOwnership:
    def test_the_evaluator_is_shared_and_read_only(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms)
        md = VelocityVerlet(ev, atoms, DT)
        assert md.evaluator is md.thermo.evaluator is ev
        with pytest.raises(AttributeError):
            md.evaluator = make_ev(model, atoms)

    def test_an_atoms_or_a_path_is_refused_as_the_evaluator(self, model):
        atoms = make_atoms(model)
        with pytest.raises(TypeError, match="not ase.md"):
            Langevin(atoms, atoms, DT, T0, FRICTION)
        with pytest.raises(TypeError, match="build_evaluator"):
            VelocityVerlet("model.ckpt.pt", atoms, DT)

    def test_thermo_refuses_the_keywords_that_moved(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms)
        with pytest.raises(TypeError, match="timestep"):
            Thermo(ev, atoms.get_positions(), timestep=0.5)
        with pytest.raises(TypeError, match="integrator"):
            Thermo(ev, atoms.get_positions(), temperature=300.0)

    def test_one_evaluator_carries_a_whole_workflow(self, model):
        """FIRE, then Langevin, then NPT on one evaluator: no second model
        copy, no rebuild at the hand-overs, and the same trajectory as
        separate evaluators give."""

        def workflow(evaluators):
            atoms = make_atoms(model)
            FIRE(evaluators[0], atoms, logfile=None, check=False).run(
                fmax=1.0, steps=30
            )
            maxwell_boltzmann(atoms, temperature_K=T0, seed=3)
            Langevin(evaluators[1], atoms, DT, T0, FRICTION, seed=5).run(5)
            IsotropicMTKNPT(evaluators[2], atoms, DT, T0, PRESSURE, TDAMP, PDAMP).run(5)
            return atoms

        atoms = make_atoms(model)
        shared = make_ev(model, atoms, compute_virial=True)
        replicas = shared._replicas
        got = workflow([shared] * 3)
        assert shared._replicas is replicas
        want = workflow([make_ev(model, atoms, compute_virial=True) for _ in range(3)])
        np.testing.assert_allclose(got.positions, want.positions, atol=1e-10)
        np.testing.assert_allclose(got.cell.array, want.cell.array, atol=1e-10)

    def test_the_hand_over_does_not_rebuild(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms)
        VelocityVerlet(ev, atoms, DT).run(3)
        rebuilds = ev.rebuild_count
        VelocityVerlet(ev, atoms, DT).run(0)  # evaluates where the first stopped
        assert ev.rebuild_count == rebuilds

    def test_a_stale_object_reevaluates_before_its_first_kick(self, model):
        """Another object evaluating in between leaves its own forces in the
        shared buffer; resuming must not kick with them."""
        atoms = make_atoms(model)
        straight = VelocityVerlet(make_ev(model, atoms.copy()), atoms.copy(), DT)
        straight.run(4)
        straight.run(4)

        ev = make_ev(model, atoms)
        a = VelocityVerlet(ev, atoms.copy(), DT)
        a.run(4)
        other = atoms.copy()
        other.rattle(0.05, seed=9)
        VelocityVerlet(ev, other, DT).run(2)
        assert not a.thermo.fresh
        a.run(4)
        np.testing.assert_allclose(
            a.thermo.positions.numpy(), straight.thermo.positions.numpy(), atol=1e-10
        )

    def test_a_stale_cell_is_refused(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms, compute_virial=True)
        nvt = NoseHooverChainNVT(ev, atoms.copy(), DT, T0, TDAMP)
        nvt.run(2)
        IsotropicMTKNPT(ev, atoms.copy(), DT, T0, PRESSURE, TDAMP, PDAMP).run(3)
        with pytest.raises(RuntimeError, match="cell has changed"):
            nvt.run(1)

    def test_edits_to_atoms_are_honoured_between_runs(self, model):
        atoms = make_atoms(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        md.run(2)
        atoms.positions[0] += 0.01
        atoms.set_momenta(np.zeros((len(atoms), 3)))
        md.run(0)
        np.testing.assert_array_equal(md.thermo.positions.numpy(), atoms.positions)
        assert md.thermo.kinetic_energy == 0.0
        assert md.thermo.fresh

    def test_an_observer_can_edit_atoms(self, model):
        atoms = make_atoms(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)

        def freeze():
            atoms.set_momenta(np.zeros((len(atoms), 3)))

        md.attach(freeze, 3)
        md.run(3)
        assert md.thermo.kinetic_energy == 0.0

    def test_wrong_order_and_constraints_are_refused(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms)
        swapped = atoms.copy()
        swapped.positions[[0, 1]] = swapped.positions[[1, 0]]
        swapped.numbers[[0, 1]] = swapped.numbers[[1, 0]]
        with pytest.raises(ValueError, match="atom order"):
            VelocityVerlet(ev, swapped, DT)
        fixed = atoms.copy()
        fixed.set_constraint(FixAtoms([0]))
        with pytest.raises(ValueError, match="constraints"):
            VelocityVerlet(ev, fixed, DT)

    def test_isotope_masses_reach_the_kinetic_energy(self, model):
        atoms = make_atoms(model)
        heavy = atoms.get_masses()
        heavy[atoms.numbers == 1] = 2.014
        atoms.set_masses(heavy)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        v = md.thermo.velocities.numpy()
        assert md.thermo.kinetic_energy == pytest.approx(
            0.5 * float((heavy[:, None] * v**2).sum()), rel=1e-10
        )


# --- each class reproduces its kernel loop ------------------------------------


class TestIntegratorsMatchTheirKernelLoops:
    @pytest.mark.parametrize("case", list(CASES))
    def test_ten_steps(self, model, case):
        make, loop, ev_kw = CASES[case]
        atoms = make_atoms(model)
        md = make(make_ev(model, atoms, **ev_kw), atoms.copy())
        md.run(10)
        ref = loop(make_ev(model, atoms, **ev_kw), atoms.copy(), 10)
        np.testing.assert_allclose(
            md.thermo.positions.numpy(), ref.positions.numpy(), atol=1e-10
        )
        np.testing.assert_allclose(
            md.thermo.velocities.numpy(), ref.velocities.numpy(), atol=1e-10
        )


# --- running, logging, output -------------------------------------------------


class TestRunSemantics:
    def test_run_twice_continues_without_duplicates(self, model, tmp_path):
        atoms = make_atoms(model)
        md = Langevin(
            make_ev(model, atoms),
            atoms,
            DT,
            T0,
            FRICTION,
            seed=1,
            logfile=tmp_path / "md.log",
            loginterval=5,
            trajectory=tmp_path / "md.extxyz",
        )
        md.run(5)
        md.run(5)
        md.close()
        assert md.nsteps == 10
        assert np.loadtxt(tmp_path / "md.log")[:, 0].tolist() == [0, 5, 10]
        frames = ase.io.read(tmp_path / "md.extxyz", ":")
        assert [f.info["step"] for f in frames] == [0, 5, 10]
        assert md.get_time() == pytest.approx(10 * DT)

    def test_observers_fire_at_absolute_multiples(self, model):
        atoms = make_atoms(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        seen = []
        md.attach(lambda: seen.append(md.nsteps), 3, sync_atoms=False)
        md.run(5)
        md.run(5)
        assert seen == [0, 3, 6, 9]

    def test_irun_yields_every_step(self, model):
        atoms = make_atoms(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        assert list(md.irun(4)) == [1, 2, 3, 4]

    def test_stop_ends_the_run_and_writes_the_checkpoint(self, model, tmp_path):
        atoms = make_atoms(model)
        md = VelocityVerlet(
            make_ev(model, atoms), atoms, DT, checkpoint=tmp_path / "state.npz"
        )
        md.attach(lambda: md.stop() if md.nsteps == 4 else None, 1, sync_atoms=False)
        md.run(100)
        assert md.nsteps == 4
        with np.load(tmp_path / "state.npz") as z:
            assert json.loads(str(z["meta"]))["nsteps"] == 4

    def test_the_log_carries_the_ensemble_and_extra_columns(self, model, tmp_path):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms, compute_virial=True)
        md = IsotropicMTKNPT(
            ev,
            atoms,
            DT,
            T0,
            PRESSURE,
            TDAMP,
            PDAMP,
            logfile=tmp_path / "npt.log",
            loginterval=2,
            extra_columns={"n2": lambda m: 2.0 * m.nsteps},
        )
        md.run(4)
        md.close()
        header = (tmp_path / "npt.log").read_text().splitlines()[0].split()
        assert (
            header[-6:] == ["Econs[eV]", "P[GPa]", "V[A^3]", "n2", "ms/step"][-6:]
            or "n2" in header
        )
        rows = np.loadtxt(tmp_path / "npt.log")
        assert rows[
            :, header.index("n2") - 1 if header[0] == "#" else header.index("n2")
        ].tolist() == [0.0, 4.0, 8.0]

    def test_a_value_wider_than_its_column_keeps_the_log_readable(self, tmp_path):
        """A blow-up's last row carries a temperature of ~1e23 K. Without a
        separator it ran into the time column and loadtxt refused the file."""
        from deepmd_lr.nvalchemi.md import _Logger

        path = tmp_path / "md.log"
        log = _Logger(path, [("step", 6, "d"), ("T[K]", 10, ".2f")], append=False)
        log.write({"step": 10, "T[K]": 300.0})
        log.write({"step": 20, "T[K]": 1.5e23})
        log.close()
        assert path.read_text().splitlines()[1] == "    10    300.00"
        np.testing.assert_allclose(np.loadtxt(path), [[10, 300.0], [20, 1.5e23]])

    def test_throughput_excludes_the_first_interval(self, model):
        atoms = make_atoms(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, 0.5 * units.fs)
        assert math.isnan(md.seconds_per_step)
        md._clock = [(0, 0.0), (10, 5.0), (20, 6.0), (30, 7.0)]  # warm-up: 5 s
        assert md.seconds_per_step == pytest.approx(0.1)
        assert md.ns_per_day == pytest.approx(0.5e-6 * 86400 / 0.1)
        assert "ms/step" in md.performance() and "ns/day" in md.performance()

    def test_drift_is_mev_per_atom_per_ps(self, model):
        atoms = make_atoms(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        n = md.thermo.n_atoms
        md._rows = [
            {"time[ps]": t, "Etot[eV]": 2.0 * n * t / 1000.0} for t in (0.0, 0.5, 1.0)
        ]
        assert md.drift("Etot[eV]") == pytest.approx(2.0)

    def test_a_hot_run_raises(self, model):
        atoms = make_atoms(model)
        md = VelocityVerlet(
            make_ev(model, atoms), atoms, DT, logfile=None, max_temperature=1.0
        )
        with pytest.raises(MDInstabilityError, match="max_temperature"):
            md.run(5)


# --- checkpoints ----------------------------------------------------------------


class TestCheckpoints:
    @pytest.mark.parametrize("case", list(CASES))
    def test_restoring_continues_exactly(self, model, tmp_path, case):
        make, _loop, ev_kw = CASES[case]
        atoms = make_atoms(model)
        straight = make(make_ev(model, atoms, **ev_kw), atoms.copy())
        straight.run(10)

        first = make(
            make_ev(model, atoms, **ev_kw),
            atoms.copy(),
            checkpoint=tmp_path / "state.npz",
        )
        first.run(5)
        first.save_checkpoint()
        again = type(first).from_checkpoint(
            make_ev(model, atoms, **ev_kw), atoms.copy(), tmp_path / "state.npz"
        )
        assert again.nsteps == 5
        again.run(5)
        np.testing.assert_allclose(
            again.thermo.positions.numpy(),
            straight.thermo.positions.numpy(),
            atol=1e-10,
        )

    def test_a_mismatch_is_refused(self, model, tmp_path):
        atoms = make_atoms(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms.copy(), DT)
        md.run(2)
        path = md.save_checkpoint(tmp_path / "state.npz")
        with pytest.raises(ValueError, match="written by VelocityVerlet"):
            Langevin.from_checkpoint(make_ev(model, atoms), atoms.copy(), path)
        with pytest.raises(ValueError, match="different system or evaluator"):
            VelocityVerlet.from_checkpoint(
                make_ev(model, atoms, eps_inf=2.0), atoms.copy(), path
            )
        with pytest.raises(TypeError, match="output settings"):
            VelocityVerlet.from_checkpoint(
                make_ev(model, atoms), atoms.copy(), path, timestep=1.0
            )

    def test_a_hard_kill_is_trimmed_back(self, model, tmp_path):
        atoms = make_atoms(model)
        log = tmp_path / "md.log"
        md = VelocityVerlet(
            make_ev(model, atoms),
            atoms.copy(),
            DT,
            logfile=log,
            loginterval=2,
            checkpoint=tmp_path / "state.npz",
        )
        md.run(4)
        md.save_checkpoint()
        md.close()
        with open(log, "a") as fh:
            fh.write("   6   junk written after the checkpoint\n")
        again = VelocityVerlet.from_checkpoint(
            make_ev(model, atoms), atoms.copy(), tmp_path / "state.npz"
        )
        again.run(2)
        again.close()
        assert np.loadtxt(log)[:, 0].tolist() == [0, 2, 4, 6]


# --- physics ---------------------------------------------------------------------


def _fluctuation(md, steps, every):
    values = []
    md.attach(lambda: values.append(md.conserved_energy()), every, sync_atoms=False)
    md.run(steps)
    values = np.array(values)
    t = np.arange(len(values))
    return float(np.std(values - np.polyval(np.polyfit(t, values, 1), t)))


class TestPhysics:
    def test_nve_fluctuation_scales_as_dt_squared(self, model):
        atoms = make_atoms(model)
        big = _fluctuation(
            VelocityVerlet(make_ev(model, atoms), atoms.copy(), 0.2 * units.fs), 100, 5
        )
        small = _fluctuation(
            VelocityVerlet(make_ev(model, atoms), atoms.copy(), 0.1 * units.fs), 200, 10
        )
        assert 3.0 < big / small < 5.0

    def test_npt_conserved_quantity_scales_as_dt_squared(self, model):
        """The MTK structure -- step order, masses, the H' formula -- is
        second order. The thermostat is left idle on purpose: nvalchemiops'
        NPT velocity kernel applies the drag -(alpha eps_dot + eta_dot) v
        explicitly inside the kick, which is first order, and this fixture
        heats fast enough to keep a 100 fs thermostat busy (the ratio is then
        ~1.8). The NVT chain applies its scaling exactly and is O(dt^2) at any
        coupling (test_nhc_needs_eta_over_two runs it at 20 fs)."""
        atoms = make_atoms(model)

        def npt(dt):
            return IsotropicMTKNPT(
                make_ev(model, atoms, compute_virial=True),
                atoms.copy(),
                dt,
                T0,
                PRESSURE,
                1e6 * units.fs,
                PDAMP,
            )

        big = _fluctuation(npt(0.2 * units.fs), 100, 5)
        small = _fluctuation(npt(0.1 * units.fs), 200, 10)
        assert 3.0 < big / small < 5.0

    def test_nhc_needs_eta_over_two(self, model):
        """The kernel advances eta twice as fast as eta_dot implies. With eta/2
        the extended energy is conserved; with the raw eta it drifts."""
        atoms = make_atoms(model)
        md = NoseHooverChainNVT(make_ev(model, atoms), atoms, DT, T0, 20 * units.fs)
        half, raw = [], []
        ke, pe = md.thermo.zeros(1), md.thermo.zeros(1)

        def record():
            half.append(md.conserved_energy())
            nhc_compute_chain_energy(
                md.eta, md.eta_dot, md.eta_mass, md.kT, md._ndof, ke, pe
            )
            raw.append(
                md.thermo.total_energy + float(ke.numpy()[0]) + float(pe.numpy()[0])
            )

        md.attach(record, 5, sync_atoms=False)
        md.run(200)
        assert np.ptp(raw) > 5 * np.ptp(half)

    def test_nhc_follows_verlet_at_a_huge_tdamp(self, model):
        atoms = make_atoms(model)
        nhc = NoseHooverChainNVT(
            make_ev(model, atoms), atoms.copy(), DT, T0, 1e9 * units.fs
        )
        vv = VelocityVerlet(make_ev(model, atoms), atoms.copy(), DT)
        nhc.run(20)
        vv.run(20)
        np.testing.assert_allclose(
            nhc.thermo.positions.numpy(), vv.thermo.positions.numpy(), atol=1e-9
        )

    def test_npt_refuses_no_virial_and_a_short_chain(self, model):
        atoms = make_atoms(model)
        with pytest.raises(ValueError, match="virial"):
            IsotropicMTKNPT(
                make_ev(model, atoms), atoms, DT, T0, PRESSURE, TDAMP, PDAMP
            )
        with pytest.raises(ValueError, match="tchain"):
            IsotropicMTKNPT(
                make_ev(model, atoms, compute_virial=True),
                atoms,
                DT,
                T0,
                PRESSURE,
                TDAMP,
                PDAMP,
                tchain=1,
            )

    def test_one_evaluator_switches_the_virial_on_for_npt(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms)
        Langevin(ev, atoms, DT, T0, FRICTION, seed=2).run(2)
        ev.set_compute_virial(True)
        IsotropicMTKNPT(ev, atoms, DT, T0, PRESSURE, TDAMP, PDAMP).run(2)

    def test_langevin_fixcm_keeps_the_centre_of_mass_at_rest(self, model):
        atoms = make_atoms(model)
        md = Langevin(make_ev(model, atoms), atoms, DT, T0, 1.0 / units.fs, seed=4)
        md.run(10)
        v, m = md.thermo.velocities.numpy(), atoms.get_masses()
        assert np.abs((m[:, None] * v).sum(axis=0)).max() < 1e-12

    def test_logging_does_not_change_an_npt_trajectory(self, model, tmp_path):
        atoms = make_atoms(model)
        runs = []
        for log in (None, tmp_path / "every-step.log"):
            md = IsotropicMTKNPT(
                make_ev(model, atoms, compute_virial=True),
                atoms.copy(),
                DT,
                T0,
                PRESSURE,
                TDAMP,
                PDAMP,
                logfile=log,
                loginterval=1,
            )
            md.run(20)
            runs.append((md.thermo.positions.numpy(), md.cell_v.numpy()))
        np.testing.assert_allclose(runs[1][0], runs[0][0], atol=1e-10)
        np.testing.assert_allclose(runs[1][1], runs[0][1], atol=1e-14)

    @pytest.mark.parametrize("case", ["nhc", "npt"])
    def test_the_thermostat_rests_at_the_reported_temperature(self, model, case):
        """One count of the kinetic degrees of freedom, 3N - 3, for the
        thermostat's target and for the temperature it is judged by."""
        make, _loop, ev_kw = CASES[case]
        atoms = make_atoms(model)
        md = make(make_ev(model, atoms, **ev_kw), atoms)
        wp.to_torch(md.thermo.velocities).mul_(math.sqrt(T0 / md.thermo.temperature))
        assert md.thermo.temperature == pytest.approx(T0, rel=1e-12)
        md.thermo.evaluate()
        if case == "npt":
            md.thermo.update_kinetic_energy()
            md._thermostat()
        else:
            md._chain()
        assert abs(md.eta_dot.numpy().reshape(-1)[0]) < 1e-12


# --- FIRE ---------------------------------------------------------------------------


class _Perturbed:
    """A real evaluator, with what it returns perturbed after some calls."""

    def __init__(self, ev, energy=lambda n: 0.0, force_scale=lambda n: 1.0):
        self._ev, self._energy, self._scale, self.n = ev, energy, force_scale, 0

    def __getattr__(self, name):
        return getattr(self._ev, name)

    def __call__(self, positions):
        energy, forces = self._ev(positions)
        self.n += 1
        if self._scale(self.n) != 1.0:
            forces.mul_(self._scale(self.n))
        return energy + self._energy(self.n), forces


class TestFIRE:
    def test_it_relaxes_and_writes_back(self, model):
        atoms = make_atoms(model)
        cell = atoms.cell.array.copy()
        opt = FIRE(make_ev(model, atoms), atoms, logfile=None, check=False)
        assert opt.run(fmax=0.5, steps=500)
        assert opt.nsteps > 0
        assert np.linalg.norm(atoms.get_forces(), axis=1).max() < 0.5
        assert np.abs(atoms.get_momenta()).max() == 0.0
        np.testing.assert_array_equal(atoms.cell.array, cell)

    def test_divergence_is_fatal_even_without_checks(self, model):
        atoms = make_atoms(model)
        ev = _Perturbed(
            make_ev(model, atoms), force_scale=lambda n: 1e4 if n > 3 else 1.0
        )
        with pytest.raises(RelaxationError, match="diverged"):
            FIRE(ev, atoms, logfile=None, check=False).run(fmax=1e-6, steps=100)

    def test_the_trajectory_ends_where_the_run_did(self, model, tmp_path):
        """A frame at every log row and one at the end, never twice for the
        same step; the end of a divergence is where it went wrong."""
        atoms = make_atoms(model)
        path = tmp_path / "fire.extxyz"
        opt = FIRE(
            make_ev(model, atoms),
            atoms.copy(),
            logfile=None,
            loginterval=5,
            trajectory=path,
            check=False,
        )
        opt.run(fmax=1e-9, steps=10)
        opt.close()
        assert [f.info["step"] for f in ase.io.iread(path)] == [0, 5, 10]

        ev = _Perturbed(
            make_ev(model, atoms), force_scale=lambda n: 1e4 if n > 3 else 1.0
        )
        path = tmp_path / "diverged.extxyz"
        opt = FIRE(ev, atoms.copy(), logfile=None, trajectory=path, check=False)
        with pytest.raises(RelaxationError, match="diverged"):
            opt.run(fmax=1e-6, steps=100)
        opt.close()
        last = list(ase.io.iread(path))[-1]
        assert last.info["step"] == opt.nsteps > 0
        assert np.abs(last.get_forces()).max() > 1e3

    def test_non_convergence_is_refused(self, model):
        atoms = make_atoms(model)
        with pytest.raises(RelaxationError, match="not converged"):
            FIRE(make_ev(model, atoms), atoms, logfile=None).run(fmax=1e-9, steps=3)

    def test_an_energy_rise_is_refused_only_without_a_field(self, model, monkeypatch):
        """The fixture's random weights legitimately distort its molecules, so
        the bond check is pinned here to isolate the energy check."""
        monkeypatch.setattr(optimize_module, "_covalent_bonds", lambda a: set())
        atoms = make_atoms(model)

        def rising(ev):
            return _Perturbed(ev, energy=lambda n: 100.0 if n > 1 else 0.0)

        with pytest.raises(RelaxationError, match="energy rose"):
            FIRE(rising(make_ev(model, atoms)), atoms.copy(), logfile=None).run(
                fmax=0.5, steps=500
            )
        field = rising(make_ev(model, atoms, efield=[0.01, 0.0, 0.0]))
        assert FIRE(field, atoms.copy(), logfile=None).run(fmax=0.5, steps=500)

    def test_a_topology_change_is_refused(self, model, monkeypatch):
        atoms = make_atoms(model)
        sets = iter([{(0, 1)}, {(0, 2)}])
        monkeypatch.setattr(optimize_module, "_covalent_bonds", lambda a: next(sets))
        with pytest.raises(RelaxationError, match="topology"):
            FIRE(make_ev(model, atoms), atoms, logfile=None).run(fmax=1.0, steps=300)


class TestDipoleCurrent:
    def test_the_last_index_of_z_meets_the_velocity(self, model):
        atoms = make_atoms(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        md.run(0)
        rng = np.random.default_rng(0)
        z = rng.normal(size=(len(atoms), 3, 3))
        md.evaluator.results = {"born_charge": z}
        v = md.thermo.velocities.numpy()
        np.testing.assert_allclose(md.dipole_current(), np.einsum("iab,ib->a", z, v))
        assert not np.allclose(md.dipole_current(), np.einsum("iba,ib->a", z, v))
