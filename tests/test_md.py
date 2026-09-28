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
- **Barostats.** NPT and NPH in every cell mode: the kernels get the cell
  transposed, the enthalpy is conserved as dt^2, and each mode moves only
  what it may.
- **FIRE and FIRE2.** Each drives its own kernel, under ASE's names, and
  both refuse a bad relaxation. With `relax_cell` the cell force is the
  enthalpy gradient, and the cell reaches the target pressure.
- **Fixed atoms.** ASE's FixAtoms holds its atoms exactly, in every class;
  temperatures and the thermostats count three degrees of freedom per mobile
  atom, as ASE does; and NVE matches ASE's own VelocityVerlet with the same
  constraint.
"""

import json
import math
import os

import ase.io
import numpy as np
import pytest
import torch
import warp as wp
from ase import Atoms, units
from ase.constraints import FixAtoms, FixBondLengths, FixCartesian, FixCom
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from ase.md.verlet import VelocityVerlet as AseVelocityVerlet
from nvalchemiops.dynamics.integrators import (
    langevin_baoab_finalize,
    langevin_baoab_half_step,
    nhc_compute_chain_energy,
    nhc_compute_masses,
    nhc_position_update,
    nhc_thermostat_chain_update,
    nhc_velocity_half_step,
    nph_barostat_half_step,
    nph_cell_update,
    nph_position_update,
    nph_velocity_half_step,
    npt_barostat_half_step,
    npt_cell_update,
    npt_position_update,
    npt_thermostat_half_step,
    npt_velocity_half_step,
    velocity_rescale,
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)
from nvalchemiops.dynamics.integrators.npt import vec9d

from deepmd_lr.ase import LongRangeCalculator
from deepmd_lr.nvalchemi import (
    FIRE,
    FIRE2,
    MTKNPH,
    MTKNPT,
    DistributedEvaluator,
    Langevin,
    MDInstabilityError,
    NoseHooverChainNVT,
    RelaxationError,
    Thermo,
    VelocityRescaling,
    VelocityVerlet,
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
    thermalize_momenta(atoms, temperature_K=temperature,
                       rng=np.random.default_rng(seed + 1))
    Stationary(atoms)
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


def loop_rescale(ev, atoms, steps, temperature, tolerance, interval):
    th = _thermo(ev, atoms)
    dt, scale = th.scalar(DT), th.scalar(1.0)
    for k in range(1, steps + 1):
        velocity_verlet_position_update(
            th.positions, th.velocities, th.forces, th.masses, dt
        )
        th.evaluate()
        velocity_verlet_velocity_finalize(th.velocities, th.forces, th.masses, dt)
        if k % interval == 0 and abs(th.temperature - temperature) > tolerance:
            scale.fill_(math.sqrt(temperature / th.temperature))
            velocity_rescale(th.velocities, scale)
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


#: The kernels' name for each cell mode, and the dtype of its target.
KERNEL_MODES = {
    "isotropic": "isotropic",
    "orthorhombic": "anisotropic",
    "full": "triclinic",
}


def _target(th, pressure, mode):
    if mode == "isotropic":
        return th.scalar(pressure)
    if mode == "orthorhombic":
        return wp.array([[pressure] * 3], dtype=wp.vec3d, device=th.device)
    return wp.array([(pressure * np.eye(3)).reshape(9)], dtype=vec9d, device=th.device)


def _column_cell(th):
    """The kernels read a cell's columns as its lattice vectors; ASE's are
    its rows, so they get the transpose."""
    return wp.array(
        np.asarray(th.evaluator.cell).T.reshape(1, 3, 3).copy(),
        dtype=wp.mat33d,
        device=th.device,
    )


def loop_npt(ev, atoms, steps, pressure, tdamp, pdamp, chain=3, mode="isotropic"):
    th = _thermo(ev, atoms)
    dt, kT = th.scalar(DT), th.scalar(units.kB * T0)
    ndof = 3 * th.n_atoms - 3
    eta, eta_dot = th.zeros((1, chain)), th.zeros((1, chain))
    q = np.full((1, chain), units.kB * T0 * tdamp**2)
    q[0, 0] *= ndof
    th_mass = wp.array(q, dtype=wp.float64, device=th.device)
    cell_v = th.zeros(1, dtype=wp.mat33d)
    cell_mass = th.scalar(3 * th.n_atoms * units.kB * T0 * pdamp**2)
    target = _target(th, pressure, mode)
    count = wp.array([th.n_atoms - 1], dtype=wp.int32, device=th.device)
    h = _column_cell(th)
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
            mode=KERNEL_MODES[mode],
        )
        npt_position_update(th.positions, th.velocities, h, cell_v, dt)
        npt_cell_update(h, cell_v, dt)
        th.set_cell(h.numpy()[0].T)
        th.evaluate()
        npt_velocity_half_step(
            th.velocities,
            th.masses,
            th.forces,
            cell_v,
            eta_dots=eta_dot,
            num_atoms=count,
            dt=dt,
            mode=KERNEL_MODES[mode],
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


def loop_nph(ev, atoms, steps, pressure, pdamp, mode):
    """NPH by hand: nvalchemiops' ``run_nph_step`` order, with the kinetic
    energy and pressure refreshed before each barostat half-step."""
    th = _thermo(ev, atoms)
    dt = th.scalar(DT)
    cell_v = th.zeros(1, dtype=wp.mat33d)
    cell_mass = th.scalar(3 * th.n_atoms * units.kB * T0 * pdamp**2)
    target = _target(th, pressure, mode)
    count = wp.array([th.n_atoms - 1], dtype=wp.int32, device=th.device)
    h = _column_cell(th)

    def barostat():
        th.update_kinetic_energy()
        th.update_pressure()
        nph_barostat_half_step(
            cell_v, th.press_w, target, th.volume_w, cell_mass, th.ke_w, count, dt
        )

    def kick():
        nph_velocity_half_step(
            th.velocities,
            th.masses,
            th.forces,
            cell_v,
            num_atoms=count,
            dt=dt,
            mode=KERNEL_MODES[mode],
        )

    for _ in range(steps):
        barostat()
        kick()
        nph_position_update(th.positions, th.velocities, h, cell_v, dt)
        nph_cell_update(h, cell_v, dt)
        th.set_cell(h.numpy()[0].T)
        th.evaluate()
        kick()
        barostat()
    return th


CASES = {
    "vv": (
        lambda ev, a, **kw: VelocityVerlet(ev, a, DT, **kw),
        lambda ev, a, n: loop_vv(ev, a, n),
        {},
    ),
    "rescale": (
        lambda ev, a, **kw: VelocityRescaling(
            ev, a, DT, T0, interval=2, tolerance=5.0, **kw
        ),
        lambda ev, a, n: loop_rescale(ev, a, n, T0, 5.0, 2),
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
        lambda ev, a, **kw: MTKNPT(
            ev, a, DT, T0, PRESSURE, TDAMP, PDAMP, cell="isotropic", **kw
        ),
        lambda ev, a, n: loop_npt(ev, a, n, PRESSURE, TDAMP, PDAMP),
        {"compute_virial": True},
    ),
    "npt_full": (
        lambda ev, a, **kw: MTKNPT(ev, a, DT, T0, PRESSURE, TDAMP, PDAMP, **kw),
        lambda ev, a, n: loop_npt(ev, a, n, PRESSURE, TDAMP, PDAMP, mode="full"),
        {"compute_virial": True},
    ),
    "nph": (
        lambda ev, a, **kw: MTKNPH(
            ev, a, DT, PRESSURE, PDAMP, cell="isotropic", temperature_K=T0, **kw
        ),
        lambda ev, a, n: loop_nph(ev, a, n, PRESSURE, PDAMP, "isotropic"),
        {"compute_virial": True},
    ),
    "nph_orthorhombic": (
        lambda ev, a, **kw: MTKNPH(
            ev, a, DT, PRESSURE, PDAMP, cell="orthorhombic", temperature_K=T0, **kw
        ),
        lambda ev, a, n: loop_nph(ev, a, n, PRESSURE, PDAMP, "orthorhombic"),
        {"compute_virial": True},
    ),
    "nph_full": (
        lambda ev, a, **kw: MTKNPH(ev, a, DT, PRESSURE, PDAMP, temperature_K=T0, **kw),
        lambda ev, a, n: loop_nph(ev, a, n, PRESSURE, PDAMP, "full"),
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
        """FIRE2, then Langevin, then NPT on one evaluator: no second model
        copy, no rebuild at the hand-overs, and the same trajectory as
        separate evaluators give."""

        def workflow(evaluators):
            atoms = make_atoms(model)
            FIRE2(evaluators[0], atoms, logfile=None, check=False).run(
                fmax=1.0, steps=30
            )
            thermalize_momenta(atoms, temperature_K=T0, rng=np.random.default_rng(3))
            Stationary(atoms)
            Langevin(evaluators[1], atoms, DT, T0, FRICTION, seed=5).run(5)
            MTKNPT(
                evaluators[2], atoms, DT, T0, PRESSURE, TDAMP, PDAMP, cell="isotropic"
            ).run(5)
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
        MTKNPT(
            ev, atoms.copy(), DT, T0, PRESSURE, TDAMP, PDAMP, cell="isotropic"
        ).run(3)
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

    def test_wrong_order_and_other_constraints_are_refused(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms)
        swapped = atoms.copy()
        swapped.positions[[0, 1]] = swapped.positions[[1, 0]]
        swapped.numbers[[0, 1]] = swapped.numbers[[1, 0]]
        with pytest.raises(ValueError, match="atom order"):
            VelocityVerlet(ev, swapped, DT)
        for constraint in (
            FixCom(),
            FixBondLengths([(0, 1)]),
            FixCartesian(0, mask=(False, False, True)),
        ):
            held = atoms.copy()
            held.set_constraint([FixAtoms([3]), constraint])
            with pytest.raises(ValueError, match="FixAtoms alone"):
                VelocityVerlet(ev, held, DT)

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
        md = MTKNPT(
            ev,
            atoms,
            DT,
            T0,
            PRESSURE,
            TDAMP,
            PDAMP,
            cell="isotropic",
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
            return MTKNPT(
                make_ev(model, atoms, compute_virial=True),
                atoms.copy(),
                dt,
                T0,
                PRESSURE,
                1e6 * units.fs,
                PDAMP,
                cell="isotropic",
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
            MTKNPT(
                make_ev(model, atoms), atoms, DT, T0, PRESSURE, TDAMP, PDAMP,
                cell="isotropic",
            )
        with pytest.raises(ValueError, match="tchain"):
            MTKNPT(
                make_ev(model, atoms, compute_virial=True),
                atoms,
                DT,
                T0,
                PRESSURE,
                TDAMP,
                PDAMP,
                cell="isotropic",
                tchain=1,
            )

    def test_one_evaluator_switches_the_virial_on_for_npt(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms)
        Langevin(ev, atoms, DT, T0, FRICTION, seed=2).run(2)
        ev.set_compute_virial(True)
        MTKNPT(ev, atoms, DT, T0, PRESSURE, TDAMP, PDAMP, cell="isotropic").run(2)

    def test_velocity_rescaling_holds_its_window(self, model):
        """A hot start is pulled into the window at the first step it may be,
        and is inside it at every step that may rescale; between those, with
        an interval, it may stray."""
        for interval in (1, 5):
            atoms = make_atoms(model, temperature=2 * T0)
            md = VelocityRescaling(
                make_ev(model, atoms), atoms, DT, T0, interval=interval, tolerance=10.0
            )
            seen = []
            md.attach(
                lambda md=md, seen=seen: seen.append(
                    (md.nsteps, md.thermo.temperature)
                ),
                1,
                sync_atoms=False,
            )
            md.run(20)
            assert md.rescales >= 1
            for step, t in seen:  # at every step it may rescale on
                if step and step % interval == 0:
                    assert abs(t - T0) <= 10.0 + 1e-9
            assert abs(dict(seen)[interval] - T0) < 1e-9  # the first rescale
            assert abs(dict(seen)[0] - T0) > 10.0  # observers see step 0 too

    def test_langevin_fixcm_keeps_the_centre_of_mass_at_rest(self, model):
        atoms = make_atoms(model)
        md = Langevin(make_ev(model, atoms), atoms, DT, T0, 1.0 / units.fs, seed=4)
        md.run(10)
        v, m = md.thermo.velocities.numpy(), atoms.get_masses()
        assert np.abs((m[:, None] * v).sum(axis=0)).max() < 1e-12

    @pytest.mark.parametrize("case", ["npt", "nph_full"])
    def test_logging_does_not_change_a_barostat_trajectory(self, model, tmp_path, case):
        make, _loop, ev_kw = CASES[case]
        atoms = make_atoms(model)
        runs = []
        for log in (None, tmp_path / "every-step.log"):
            md = make(
                make_ev(model, atoms, **ev_kw), atoms.copy(), logfile=log, loginterval=1
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


def _sheared(atoms):
    h = atoms.cell.array.copy()
    h[1, 0], h[2, 0], h[2, 1] = 1.1, -0.7, 0.9
    atoms.set_cell(h, scale_atoms=True)
    return atoms


class TestBarostats:
    def test_the_kernels_get_the_cell_transposed(self, model):
        """nvalchemiops' cell update is h + dt eps_dot h, reading the columns
        of h as the lattice vectors. On ASE's rows that has to become
        H (1 + dt eps_dot); applied to the rows as they are, it would give
        (1 + dt eps_dot) H, another cell on a sheared one."""
        atoms = _sheared(make_atoms(model))
        md = MTKNPH(
            make_ev(model, atoms, compute_virial=True),
            atoms,
            DT,
            PRESSURE,
            PDAMP,
            temperature_K=T0,
        )
        e = np.array([[1.0, 0.4, -0.2], [0.4, -0.5, 0.3], [-0.2, 0.3, 0.8]]) * 1e-2
        md.cell_v.assign(e.reshape(1, 3, 3))
        h0 = np.array(md.evaluator.cell)
        x0, v0 = md.thermo.positions.numpy(), md.thermo.velocities.numpy()
        md._drift()
        one = np.eye(3) + DT * e
        np.testing.assert_allclose(md.evaluator.cell, h0 @ one, atol=1e-12)
        assert np.abs(one @ h0 - h0 @ one).max() > 1e-4
        np.testing.assert_allclose(
            md.thermo.positions.numpy(), x0 + DT * (v0 + x0 @ e), atol=1e-12
        )

    @pytest.mark.parametrize(
        "ensemble, cell",
        [
            ("nph", "isotropic"),
            ("nph", "orthorhombic"),
            ("nph", "full"),
            ("npt", "orthorhombic"),
            ("npt", "full"),
        ],
    )
    def test_the_conserved_quantity_scales_as_dt_squared(self, model, ensemble, cell):
        """NPH has no thermostat, so none of NPT's first-order drag; NPT's
        thermostat is left idle, as in test_npt_conserved_quantity_scales_as_
        dt_squared. nvalchemiops moves the cell by a first-order Euler step,
        whose error grows with eps_dot^2; a 1 ps pdamp keeps it below the
        dt^2 part (at 200 fs the ratio falls to ~2.5)."""
        atoms = make_atoms(model)

        def md(dt):
            ev = make_ev(model, atoms, compute_virial=True)
            if ensemble == "nph":
                return MTKNPH(
                    ev, atoms.copy(), dt, PRESSURE, PDAMP, cell=cell, temperature_K=T0
                )
            return MTKNPT(
                ev, atoms.copy(), dt, T0, PRESSURE, 1e6 * units.fs, PDAMP, cell=cell
            )

        big = _fluctuation(md(0.2 * units.fs), 100, 5)
        small = _fluctuation(md(0.1 * units.fs), 200, 10)
        assert 3.0 < big / small < 5.0

    def test_each_mode_moves_only_what_it_may(self, model):
        for cell in ("isotropic", "orthorhombic", "full"):
            atoms = make_atoms(model)
            h0 = atoms.cell.array.copy()
            MTKNPH(
                make_ev(model, atoms, compute_virial=True),
                atoms,
                DT,
                PRESSURE,
                200 * units.fs,
                cell=cell,
                temperature_K=T0,
            ).run(40)
            h = atoms.cell.array
            ratios = np.diag(h) / np.diag(h0)
            off = h - np.diag(np.diag(h))
            assert np.all(np.abs(ratios - 1.0) > 1e-6)
            if cell == "isotropic":
                assert np.ptp(ratios) < 1e-12 and not off.any()
            elif cell == "orthorhombic":
                assert np.ptp(ratios) > 1e-6 and not off.any()
            else:
                strain = np.linalg.solve(h0, h)
                assert np.abs(off).max() > 1e-6
                # no rotation: the deformation stays symmetric
                assert np.abs(strain - strain.T).max() < 1e-3 * np.abs(strain - 1).max()

    def test_nph_takes_its_barostat_mass_from_the_starting_temperature(self, model):
        atoms = make_atoms(model)
        md = MTKNPH(
            make_ev(model, atoms, compute_virial=True), atoms, DT, PRESSURE, PDAMP
        )
        assert md._barostat_T == pytest.approx(md.thermo.temperature, rel=1e-12)
        assert md._params()["temperature_K"] == md._barostat_T  # restored as is
        still = make_atoms(model)
        still.set_momenta(np.zeros((len(still), 3)))
        ev = make_ev(model, still, compute_virial=True)
        with pytest.raises(ValueError, match="temperature"):
            MTKNPH(ev, still, DT, PRESSURE, PDAMP)
        MTKNPH(ev, still, DT, PRESSURE, PDAMP, temperature_K=T0).run(2)

    def test_refusals(self, model):
        atoms = make_atoms(model)
        with pytest.raises(ValueError, match="virial"):
            MTKNPH(make_ev(model, atoms), atoms, DT, PRESSURE, PDAMP, temperature_K=T0)
        ev = make_ev(model, atoms, compute_virial=True)
        with pytest.raises(ValueError, match="cell must be"):
            MTKNPT(ev, atoms, DT, T0, PRESSURE, TDAMP, PDAMP, cell="cubic")
        with pytest.raises(ValueError, match="pdamp"):
            MTKNPH(ev, atoms, DT, PRESSURE, 0.0, temperature_K=T0)
        sheared = _sheared(make_atoms(model))
        with pytest.raises(ValueError, match="sheared"):
            MTKNPH(
                make_ev(model, sheared, compute_virial=True),
                sheared,
                DT,
                PRESSURE,
                PDAMP,
                cell="orthorhombic",
                temperature_K=T0,
            )


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


OPTIMIZERS = pytest.mark.parametrize("relax", [FIRE, FIRE2], ids=["FIRE", "FIRE2"])


class TestFIRE:
    @OPTIMIZERS
    def test_it_relaxes_and_writes_back(self, model, relax):
        atoms = make_atoms(model)
        cell = atoms.cell.array.copy()
        opt = relax(make_ev(model, atoms), atoms, logfile=None, check=False)
        assert opt.run(fmax=0.5, steps=500)
        assert opt.nsteps > 0
        assert np.linalg.norm(atoms.get_forces(), axis=1).max() < 0.5
        assert np.abs(atoms.get_momenta()).max() == 0.0
        np.testing.assert_array_equal(atoms.cell.array, cell)

    @OPTIMIZERS
    def test_divergence_is_fatal_even_without_checks(self, model, relax):
        atoms = make_atoms(model)
        ev = _Perturbed(
            make_ev(model, atoms), force_scale=lambda n: 1e4 if n > 3 else 1.0
        )
        with pytest.raises(RelaxationError, match="diverged"):
            relax(ev, atoms, logfile=None, check=False).run(fmax=1e-6, steps=100)

    @OPTIMIZERS
    def test_the_trajectory_ends_where_the_run_did(self, model, relax, tmp_path):
        """A frame at every log row and one at the end, never twice for the
        same step; the end of a divergence is where it went wrong."""
        atoms = make_atoms(model)
        path = tmp_path / "fire.extxyz"
        opt = relax(
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
        opt = relax(ev, atoms.copy(), logfile=None, trajectory=path, check=False)
        with pytest.raises(RelaxationError, match="diverged"):
            opt.run(fmax=1e-6, steps=100)
        opt.close()
        last = list(ase.io.iread(path))[-1]
        assert last.info["step"] == opt.nsteps > 0
        assert np.abs(last.get_forces()).max() > 1e3

    @OPTIMIZERS
    def test_non_convergence_is_refused(self, model, relax):
        atoms = make_atoms(model)
        with pytest.raises(RelaxationError, match="not converged"):
            relax(make_ev(model, atoms), atoms, logfile=None).run(fmax=1e-9, steps=3)

    @OPTIMIZERS
    def test_an_energy_rise_is_refused_only_without_a_field(
        self, model, relax, monkeypatch
    ):
        """The fixture's random weights legitimately distort its molecules, so
        the bond check is pinned here to isolate the energy check."""
        monkeypatch.setattr(optimize_module, "_covalent_bonds", lambda a: set())
        atoms = make_atoms(model)

        def rising(ev):
            return _Perturbed(ev, energy=lambda n: 100.0 if n > 1 else 0.0)

        with pytest.raises(RelaxationError, match="energy rose"):
            relax(rising(make_ev(model, atoms)), atoms.copy(), logfile=None).run(
                fmax=0.5, steps=500
            )
        field = rising(make_ev(model, atoms, efield=[0.01, 0.0, 0.0]))
        assert relax(field, atoms.copy(), logfile=None).run(fmax=0.5, steps=500)

    @OPTIMIZERS
    def test_a_topology_change_is_refused(self, model, relax, monkeypatch):
        atoms = make_atoms(model)
        sets = iter([{(0, 1)}, {(0, 2)}])
        monkeypatch.setattr(optimize_module, "_covalent_bonds", lambda a: next(sets))
        with pytest.raises(RelaxationError, match="topology"):
            relax(make_ev(model, atoms), atoms, logfile=None).run(fmax=1.0, steps=300)

    @OPTIMIZERS
    def test_bad_parameters_are_refused(self, model, relax):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms)
        with pytest.raises(ValueError, match="dtmin <= dt <= dtmax"):
            relax(ev, atoms, dt=2.0, dtmax=1.0, logfile=None)
        with pytest.raises(ValueError, match="finc > 1"):
            relax(ev, atoms, finc=0.9, logfile=None)
        with pytest.raises(ValueError, match="maxstep"):
            relax(ev, atoms, maxstep=0.0, logfile=None)


#: Distinct non-default values, so that a parameter reaching the wrong kernel
#: argument changes the trajectory.
ASE_NAMES = {
    "dt": 0.05,
    "dtmax": 0.3,
    "dtmin": 0.03,
    "Nmin": 2,
    "finc": 1.3,
    "fdec": 0.4,
    "astart": 0.2,
    "fa": 0.9,
    "maxstep": 0.03,
}


def loop_fire(ev, atoms, steps, p):
    """Classic FIRE by hand, in the kernel's own argument names; unit masses."""
    from nvalchemiops.dynamics.optimizers import fire_step

    th = Thermo(ev, atoms.get_positions())
    s, n = th.scalar, th.n_atoms
    vel = th.zeros(n, dtype=wp.vec3d)
    alpha, dt = s(p["astart"]), s(p["dt"])
    n_pos, uphill = th.zeros(1, dtype=wp.int32), th.zeros(1, dtype=wp.int32)
    dts = []
    th.evaluate()
    for _ in range(steps):
        fire_step(
            positions=th.positions,
            velocities=vel,
            forces=th.forces,
            masses=wp.array(np.ones(n), dtype=wp.float64, device=th.device),
            alpha=alpha,
            dt=dt,
            alpha_start=s(p["astart"]),
            f_alpha=s(p["fa"]),
            dt_min=s(p["dtmin"]),
            dt_max=s(p["dtmax"]),
            maxstep=s(p["maxstep"]),
            n_steps_positive=n_pos,
            n_min=s(p["Nmin"], dtype=wp.int32),
            f_dec=s(p["fdec"]),
            f_inc=s(p["finc"]),
            uphill_flag=uphill,
            vf=th.zeros(1),
            vv=th.zeros(1),
            ff=th.zeros(1),
        )
        th.evaluate()
        dts.append(float(dt.numpy()[0]))
    return th.positions.numpy(), dts


def loop_fire2(ev, atoms, steps, p):
    """FIRE2 by hand, in the kernel's own argument names."""
    from nvalchemiops.dynamics.optimizers import fire2_step

    th = Thermo(ev, atoms.get_positions())
    n = th.n_atoms
    vel = th.zeros(n, dtype=wp.vec3d)
    alpha, dt = th.scalar(p["astart"]), th.scalar(p["dt"])
    n_inc = th.zeros(1, dtype=wp.int32)
    dts = []
    th.evaluate()
    for _ in range(steps):
        fire2_step(
            th.positions,
            vel,
            th.forces,
            th.zeros(n, dtype=wp.int32),
            alpha,
            dt,
            n_inc,
            *[th.zeros(1) for _ in range(4)],
            delaystep=p["Nmin"],
            dtgrow=p["finc"],
            dtshrink=p["fdec"],
            alphashrink=p["fa"],
            alpha0=p["astart"],
            tmax=p["dtmax"],
            tmin=p["dtmin"],
            maxstep=p["maxstep"],
        )
        th.evaluate()
        dts.append(float(dt.numpy()[0]))
    return th.positions.numpy(), dts


class TestFIREKernels:
    @pytest.mark.parametrize(
        "relax, loop", [(FIRE, loop_fire), (FIRE2, loop_fire2)], ids=["FIRE", "FIRE2"]
    )
    def test_ase_names_reach_the_kernel(self, model, relax, loop):
        """The class under ASE's names matches its kernel driven by hand under
        the kernel's names. The timestep both grows and shrinks on the way,
        so the growth and the uphill parameters are both exercised. (Classic
        FIRE starts at rest, so its first step is uphill, as in ASE, and the
        shrink lands on dtmin.)"""
        atoms = make_atoms(model)
        want, dts = loop(make_ev(model, atoms), atoms, 40, ASE_NAMES)
        dts = [ASE_NAMES["dt"], *dts]
        assert max(dts) > ASE_NAMES["dt"] and min(np.diff(dts)) < 0
        opt = relax(
            make_ev(model, atoms), atoms, logfile=None, check=False, **ASE_NAMES
        )
        opt.run(fmax=1e-12, steps=40)
        assert opt.nsteps == 40
        np.testing.assert_allclose(opt.thermo.positions.numpy(), want, atol=1e-10)
        assert float(opt._dt.numpy()[0]) == pytest.approx(dts[-1], rel=1e-12)


class _CellOnly:
    """A real evaluator with the atomic forces held at zero and a spring on
    each cell length. The fixture has no stable volume -- its pressure rises
    with volume -- so the springs supply the stiffness. With no forces of
    their own the atoms move only with the cell, which pins that ride
    exactly: the positions stay the fractional coordinates times the cell."""

    def __init__(self, ev, k, h_ref):
        self._ev, self._k, self._h_ref = ev, k, np.diag(h_ref).copy()
        self._v_ref = abs(np.linalg.det(h_ref))

    def __getattr__(self, name):
        return getattr(self._ev, name)

    def _stretch(self):
        return np.diag(self._ev.cell) / self._h_ref - 1.0

    def __call__(self, positions):
        energy, forces = self._ev(positions)
        forces.zero_()
        spring = 0.5 * self._k * self._v_ref * float(np.sum(self._stretch() ** 2))
        return energy + spring, forces

    @property
    def virial(self):
        u = self._stretch()
        return self._ev.virial - np.diag(self._k * self._v_ref * u * (1.0 + u))


def _cell_only(model, strain):
    atoms = make_atoms(model)
    h_ref = atoms.cell.array.copy()
    atoms.set_cell(h_ref @ strain, scale_atoms=True)
    ev = _CellOnly(make_ev(model, atoms, compute_virial=True), 5.0 * units.GPa, h_ref)
    return atoms, ev


STRAINS = {"isotropic": np.eye(3) * 1.03, "orthorhombic": np.diag([1.03, 0.98, 1.0])}


class TestCellRelaxation:
    def test_the_cell_force_is_the_enthalpy_gradient(self, model):
        """Finite differences of E + P V over the nine cell entries, at fixed
        fractional coordinates, on a sheared cell. nvalchemiops'
        stress_to_cell_force is written for column cells and misses the
        shear entries of ASE's row cell by far more."""
        from nvalchemiops.dynamics.utils import stress_to_cell_force

        atoms = make_atoms(model)
        h = atoms.cell.array.copy()
        h[1, 0], h[2, 0], h[2, 1] = 1.1, -0.7, 0.9
        atoms.set_cell(h, scale_atoms=True)
        frac = atoms.get_scaled_positions(wrap=False)
        ev = make_ev(model, atoms, compute_virial=True)
        p = 0.3 * units.GPa
        opt = FIRE2(ev, atoms, relax_cell="full", pressure_au=p, logfile=None)
        opt.thermo.evaluate()
        got = opt._cell_force(h)
        sigma = ev.stress + p * np.eye(3)
        device = opt.thermo.device
        cell_force = wp.zeros(1, dtype=wp.mat33d, device=device)
        stress_to_cell_force(
            wp.array(sigma.reshape(1, 3, 3), dtype=wp.mat33d, device=device),
            wp.array(h.reshape(1, 3, 3), dtype=wp.mat33d, device=device),
            wp.array([ev.volume], dtype=wp.float64, device=device),
            cell_force,
            keep_aligned=False,
        )
        column_cells = cell_force.numpy()[0]

        def enthalpy(cell):
            ev.set_cell(cell)
            return ev(frac @ cell)[0] + p * abs(np.linalg.det(cell))

        d, fd = 1e-4, np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                step = np.zeros((3, 3))
                step[i, j] = d
                fd[i, j] = -(enthalpy(h + step) - enthalpy(h - step)) / (2 * d)
        error = np.abs(got - fd).max()
        assert error < 1e-6 * np.abs(fd).max()
        assert np.abs(column_cells - fd).max() > 1e3 * error

    @OPTIMIZERS
    @pytest.mark.parametrize("mode", list(STRAINS))
    def test_the_cell_relaxes_to_the_target_pressure(self, model, relax, mode):
        atoms, ev = _cell_only(model, STRAINS[mode])
        h0, frac = atoms.cell.array.copy(), atoms.get_scaled_positions(wrap=False)
        opt = relax(
            ev,
            atoms,
            relax_cell=mode,
            pressure_au=0.05 * units.GPa,
            logfile=None,
            check=False,
        )
        assert opt.run(fmax=0.05, steps=3000, smax=1e-4 * units.GPa)
        assert opt.stress_residual() < 1e-4 * units.GPa
        h = atoms.cell.array
        np.testing.assert_array_equal(h, ev.cell)
        np.testing.assert_allclose(atoms.positions, frac @ h, atol=1e-10)
        assert np.all(h == np.diag(np.diag(h)))  # no angle moved
        ratios = np.diag(h) / np.diag(h0)
        assert (
            np.ptp(ratios) > 1e-3 if mode == "orthorhombic" else np.ptp(ratios) < 1e-12
        )
        assert np.all(np.abs(ratios - 1.0) > 1e-3)  # it did move

    def test_both_algorithms_find_the_same_cell(self, model):
        cells = []
        for relax in (FIRE, FIRE2):
            atoms, ev = _cell_only(model, STRAINS["orthorhombic"])
            relax(ev, atoms, relax_cell="orthorhombic", logfile=None, check=False).run(
                fmax=0.05, steps=3000, smax=1e-5 * units.GPa
            )
            cells.append(atoms.cell.array)
        np.testing.assert_allclose(cells[0], cells[1], rtol=1e-5)

    def test_refusals(self, model):
        atoms = make_atoms(model)
        with pytest.raises(ValueError, match="virial"):
            FIRE2(make_ev(model, atoms), atoms, relax_cell="isotropic", logfile=None)
        ev = make_ev(model, atoms, compute_virial=True)
        with pytest.raises(ValueError, match="relax_cell must be"):
            FIRE2(ev, atoms, relax_cell="cubic", logfile=None)
        with pytest.raises(ValueError, match="pressure_au"):
            FIRE2(ev, atoms, pressure_au=1e-3, logfile=None)
        with pytest.raises(ValueError, match="fixed"):
            FIRE2(ev, atoms, logfile=None).stress_residual()
        sheared = atoms.copy()
        h = sheared.cell.array.copy()
        h[1, 0] = 1.0
        sheared.set_cell(h, scale_atoms=True)
        with pytest.raises(ValueError, match="diagonal"):
            FIRE2(
                make_ev(model, sheared, compute_virial=True),
                sheared,
                relax_cell="orthorhombic",
                logfile=None,
            )

    def test_the_log_and_the_trajectory_follow_the_cell(self, model, tmp_path):
        atoms, ev = _cell_only(model, STRAINS["isotropic"])
        opt = FIRE2(
            ev,
            atoms,
            relax_cell="isotropic",
            logfile=tmp_path / "relax.log",
            loginterval=10,
            trajectory=tmp_path / "relax.extxyz",
            check=False,
        )
        opt.run(fmax=0.05, steps=40, smax=1e-9)
        opt.close()
        header = (tmp_path / "relax.log").read_text().splitlines()[0].split()
        assert header[-3:] == ["P[GPa]", "dS[GPa]", "V[A^3]"]
        rows = np.loadtxt(tmp_path / "relax.log")
        frames = list(ase.io.iread(tmp_path / "relax.extxyz"))
        np.testing.assert_allclose(frames[-1].cell.array, ev.cell)
        assert frames[0].get_volume() > frames[-1].get_volume()
        np.testing.assert_allclose(  # the log prints three decimals
            rows[:, -1], [f.get_volume() for f in frames], atol=5e-4
        )

    def test_an_older_md_object_refuses_the_relaxed_cell(self, model):
        atoms = make_atoms(model)
        ev = make_ev(model, atoms, compute_virial=True)
        md = Langevin(ev, atoms.copy(), DT, T0, FRICTION, seed=1)
        FIRE2(ev, atoms, relax_cell="isotropic", logfile=None, check=False).run(
            fmax=1e-9, steps=5
        )
        with pytest.raises(RuntimeError, match="cell has changed"):
            md.run(1)
        Langevin(ev, atoms, DT, T0, FRICTION, seed=1).run(2)  # atoms carries it


@pytest.mark.skipif(
    "DEEPMD_LR_WATER_MODEL" not in os.environ,
    reason="needs a trained water checkpoint in DEEPMD_LR_WATER_MODEL",
)
@pytest.mark.parametrize("mode", ["isotropic", "orthorhombic", "full"])
def test_water_relaxes_its_cell(mode):
    """A water box stretched by 4% comes back to zero stress in every mode."""
    import pathlib

    from deepmd_lr.nvalchemi import build_evaluator, read_structure

    root = pathlib.Path(__file__).resolve().parents[1]
    atoms = read_structure(root / "examples" / "06-md_nvalchemi" / "water_192.xyz")
    strain = np.diag([1.04, 0.96, 1.0]) if mode == "orthorhombic" else 1.04 * np.eye(3)
    atoms.set_cell(atoms.cell.array @ strain, scale_atoms=True)
    ev = build_evaluator(
        os.environ["DEEPMD_LR_WATER_MODEL"],
        atoms,
        compute_virial=True,
        compress=os.environ.get("DEEPMD_LR_WATER_COMPRESS") == "1",
    )
    opt = FIRE2(ev, atoms, relax_cell=mode, logfile=None)
    assert opt.run(fmax=0.05, steps=3000)
    assert opt.stress_residual() < 0.01 * units.GPa


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


SLAB = (True, True, False)


def make_slab(model, vacuum=20.0, seed=0):
    """The fixture's molecules as a slab: z opened, with `vacuum` above the
    layer, which is centred."""
    atoms = make_atoms(model, seed=seed)
    pos = atoms.get_positions()
    pos[:, 2] += vacuum / 2 - pos[:, 2].min()
    cell = atoms.cell.array.copy()
    cell[2, 2] = np.ptp(pos[:, 2]) + vacuum
    atoms.set_positions(pos)
    atoms.set_cell(cell, scale_atoms=False)
    atoms.pbc = SLAB
    return atoms


class TestSlabs:
    def test_nvt_runs_on_a_slab_and_keeps_it_in_the_cell(self, model):
        atoms = make_slab(model)
        md = Langevin(make_ev(model, atoms, pbc=SLAB), atoms, DT, T0, FRICTION, seed=1)
        md.run(20)
        assert np.isfinite(md.thermo.energy)
        z = md.thermo.positions.numpy()[:, 2]
        assert 0.0 < z.min() and z.max() < atoms.cell[2, 2]

    def test_atoms_must_carry_the_evaluators_pbc(self, model):
        atoms = make_slab(model)
        ev = make_ev(model, atoms, pbc=SLAB)
        atoms.pbc = True
        with pytest.raises(ValueError, match="evaluator was built for"):
            VelocityVerlet(ev, atoms, DT)

    def test_barostats_are_refused_on_a_slab(self, model):
        atoms = make_slab(model)
        ev = make_ev(model, atoms, pbc=SLAB, compute_virial=True)
        with pytest.raises(ValueError, match="open axis"):
            MTKNPT(ev, atoms, DT, T0, PRESSURE, TDAMP, PDAMP, cell="isotropic")
        with pytest.raises(ValueError, match="open axis"):
            MTKNPH(ev, atoms, DT, PRESSURE, PDAMP, temperature_K=T0)

    def test_fire_relaxes_a_slabs_positions_but_not_its_cell(self, model):
        atoms = make_slab(model)
        ev = make_ev(model, atoms, pbc=SLAB, compute_virial=True)
        with pytest.raises(ValueError, match="relax the positions only"):
            FIRE2(ev, atoms, relax_cell="isotropic", logfile=None)
        FIRE2(ev, atoms, logfile=None, check=False).run(fmax=1.0, steps=5)

    def test_frames_carry_the_slabs_pbc(self, model, tmp_path):
        """Read back as periodic, the last frame would restart as bulk: the
        3D sum, and no correction."""
        atoms = make_slab(model)
        ev = make_ev(model, atoms, pbc=SLAB)
        md = VelocityVerlet(ev, atoms, DT, trajectory=tmp_path / "md.extxyz")
        md.run(2)
        md.close()
        opt = FIRE2(ev, atoms, logfile=None, trajectory=tmp_path / "fire.extxyz",
                    check=False)
        opt.run(fmax=1.0, steps=2)
        opt.close()
        for name in ("md.extxyz", "fire.extxyz"):
            for written in ase.io.read(tmp_path / name, ":"):
                assert written.pbc.tolist() == list(SLAB)
        assert md.thermo.atoms().pbc.tolist() == list(SLAB)


# --- fixed atoms ------------------------------------------------------------------

FIXED = np.arange(12)  # the first four molecules


def make_fixed(model, **kwargs):
    """The fixture, with FixAtoms attached after the momenta were drawn, so
    the fixed atoms start moving and the classes have to stop them."""
    atoms = make_atoms(model, **kwargs)
    atoms.set_constraint(FixAtoms(indices=FIXED))
    return atoms


def mobile(atoms):
    return np.setdiff1d(np.arange(len(atoms)), FIXED)


class TestFixedAtoms:
    @pytest.mark.parametrize(
        "make",
        [
            CASES["vv"][0],
            CASES["rescale"][0],
            CASES["langevin"][0],
            lambda ev, a: Langevin(ev, a, DT, T0, FRICTION, seed=7),  # fixcm
            CASES["nhc"][0],
        ],
        ids=["vv", "rescale", "langevin", "langevin_fixcm", "nhc"],
    )
    def test_they_stay_put_while_the_rest_move(self, model, make):
        """Exactly. No kick or drift moves them, and the noise Langevin's
        kernel gives every atom is taken away before the forces are computed."""
        atoms = make_fixed(model)
        x0 = atoms.get_positions()
        md = make(make_ev(model, atoms), atoms)
        md.run(20)
        x, v = md.thermo.positions.numpy(), md.thermo.velocities.numpy()
        np.testing.assert_array_equal(x[FIXED], x0[FIXED])
        assert not v[FIXED].any()
        assert np.abs(x - x0)[mobile(atoms)].max() > 1e-3

    def test_the_temperature_counts_three_per_mobile_atom(self, model):
        """As ASE's Atoms.get_temperature does. The fixed atoms carry neither
        kinetic energy nor degrees of freedom, and none is taken off for the
        centre of mass: with them holding the rest in place, the mobile atoms'
        momentum is not conserved."""
        atoms = make_fixed(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        md.run(5)
        th = md.thermo
        assert th.ndof == 3 * len(mobile(atoms))
        assert th.temperature == pytest.approx(atoms.get_temperature(), rel=1e-12)
        assert th.temperature == pytest.approx(
            2 * th.kinetic_energy / (th.ndof * units.kB), rel=1e-12
        )

    def test_nose_hoover_targets_the_same_count(self, model):
        """At the target temperature, as reported, the chain rests: its target
        and its mass use the degrees of freedom the temperature counts."""
        atoms = make_fixed(model)
        md = NoseHooverChainNVT(make_ev(model, atoms), atoms, DT, T0, TDAMP)
        th = md.thermo
        wp.to_torch(th.velocities).mul_(math.sqrt(T0 / th.temperature))
        th.evaluate()
        md._chain()
        assert abs(md.eta_dot.numpy()[0]) < 1e-12
        assert md.eta_mass.numpy()[0] == pytest.approx(
            th.ndof * units.kB * T0 * TDAMP**2, rel=1e-12
        )

    def test_velocity_verlet_matches_ase_with_the_same_constraint(self, model):
        """An independent reference: ASE's own VelocityVerlet, with ASE
        applying FixAtoms, on the ASE calculator."""
        atoms = make_fixed(model)
        ref = atoms.copy()
        ref.calc = LongRangeCalculator(model)
        AseVelocityVerlet(ref, timestep=DT).run(20)
        VelocityVerlet(make_ev(model, atoms), atoms, DT).run(20)
        np.testing.assert_allclose(atoms.positions, ref.positions, atol=1e-10)
        np.testing.assert_allclose(
            atoms.get_velocities(), ref.get_velocities(), atol=1e-10
        )

    def test_output_keeps_the_model_forces_and_the_constraint(self, model, tmp_path):
        """The kernels see zero force on the fixed atoms; atoms.calc and the
        trajectory keep what the model puts on them, and ASE zeroes it again
        on get_forces(), as with any calculator."""
        atoms = make_fixed(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT,
                            trajectory=tmp_path / "md.extxyz")
        md.run(3)
        md.close()
        raw = atoms.get_forces(apply_constraint=False)
        assert np.abs(raw[FIXED]).max() > 1e-3
        assert not atoms.get_forces()[FIXED].any()
        assert not md.thermo.forces.numpy()[FIXED].any()
        last = ase.io.read(tmp_path / "md.extxyz", -1)
        assert [type(c).__name__ for c in last.constraints] == ["FixAtoms"]
        np.testing.assert_array_equal(last.constraints[0].index, FIXED)
        np.testing.assert_allclose(
            last.get_forces(apply_constraint=False), raw, atol=1e-6
        )

    @pytest.mark.parametrize("relax", [FIRE, FIRE2])
    def test_a_relaxation_moves_only_the_rest(self, model, relax):
        atoms = make_fixed(model)
        x0 = atoms.get_positions()
        opt = relax(make_ev(model, atoms), atoms, logfile=None, check=False)
        assert opt.run(fmax=0.5, steps=300)
        np.testing.assert_array_equal(atoms.positions[FIXED], x0[FIXED])
        free = atoms.get_forces()[mobile(atoms)]
        assert opt.fmax() == pytest.approx(
            np.linalg.norm(free, axis=1).max(), rel=1e-12
        )

    def test_what_would_strain_them_is_refused(self, model):
        atoms = make_fixed(model)
        ev = make_ev(model, atoms, compute_virial=True)
        for make in (
            lambda: MTKNPT(ev, atoms, DT, T0, PRESSURE, TDAMP, PDAMP),
            lambda: MTKNPH(ev, atoms, DT, PRESSURE, PDAMP, temperature_K=T0),
            lambda: FIRE2(ev, atoms, relax_cell="isotropic", logfile=None),
        ):
            with pytest.raises(ValueError, match="fixed atoms would not stay"):
                make()

    def test_edits_move_where_they_are_held(self, model):
        """ASE's set_positions leaves them where they are. Writing the array
        directly moves them, and the new place is where they are held."""
        atoms = make_fixed(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        md.run(2)
        held = atoms.positions[FIXED].copy()
        atoms.set_positions(atoms.positions + 0.01)
        md.run(1)
        np.testing.assert_array_equal(md.thermo.positions.numpy()[FIXED], held)
        atoms.positions[FIXED] += 0.01
        moved = atoms.positions[FIXED].copy()
        md.run(3)
        np.testing.assert_array_equal(md.thermo.positions.numpy()[FIXED], moved)

    def test_a_changed_constraint_is_refused_between_runs(self, model):
        atoms = make_fixed(model)
        md = VelocityVerlet(make_ev(model, atoms), atoms, DT)
        md.run(2)
        atoms.set_constraint(FixAtoms(indices=FIXED[:3]))
        with pytest.raises(ValueError, match="FixAtoms changed"):
            md.run(1)

    def test_a_checkpoint_carries_them(self, model, tmp_path):
        atoms = make_fixed(model)
        straight = NoseHooverChainNVT(make_ev(model, atoms), atoms.copy(), DT, T0,
                                      TDAMP)
        straight.run(10)
        first = NoseHooverChainNVT(make_ev(model, atoms), atoms.copy(), DT, T0, TDAMP)
        first.run(5)
        path = first.save_checkpoint(tmp_path / "state.npz")
        again = NoseHooverChainNVT.from_checkpoint(
            make_ev(model, atoms), atoms.copy(), path
        )
        again.run(5)
        np.testing.assert_allclose(
            again.thermo.positions.numpy(), straight.thermo.positions.numpy(),
            atol=1e-10,
        )
        loose = atoms.copy()
        loose.set_constraint()
        with pytest.raises(ValueError, match="fixed_sha1"):
            NoseHooverChainNVT.from_checkpoint(make_ev(model, atoms), loose, path)

    def test_a_slab_held_by_its_bottom_layer(self, model):
        """What they are for: the bottom of a slab held as the bulk below it
        would hold it."""
        atoms = make_slab(model)
        z = atoms.positions[:, 2]
        bottom = np.flatnonzero(z < z.min() + 3.0)
        atoms.set_constraint(FixAtoms(indices=bottom))
        x0 = atoms.get_positions()
        md = Langevin(make_ev(model, atoms, pbc=SLAB), atoms, DT, T0, FRICTION,
                      seed=1)
        md.run(20)
        np.testing.assert_array_equal(atoms.positions[bottom], x0[bottom])
        assert md.thermo.ndof == 3 * (len(atoms) - len(bottom))
