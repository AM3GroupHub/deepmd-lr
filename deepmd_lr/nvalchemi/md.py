# SPDX-License-Identifier: MIT
"""ASE-style molecular dynamics on the GPU, driven by a shared evaluator.

.. code-block:: python

    from ase import units
    from ase.md.velocitydistribution import Stationary, thermalize_momenta
    from deepmd_lr.nvalchemi import Langevin, build_evaluator

    ev = build_evaluator("model.ckpt.pt", atoms, devices=["cuda:0", "cuda:1"])
    thermalize_momenta(atoms, temperature_K=300)
    Stationary(atoms)
    md = Langevin(ev, atoms, 0.5 * units.fs, temperature_K=300,
                  friction=0.01 / units.fs, logfile="md.log", loginterval=100)
    md.run(10_000)

**Who owns what.**
- **The evaluator** is built once by the caller and shared: model replicas on
  every device, the partitioning, the forces buffer. Several MD objects (and a
  :class:`~deepmd_lr.nvalchemi.optimize.FIRE` or
  :class:`~deepmd_lr.nvalchemi.optimize.FIRE2` relaxation) may use it one after
  another -- a relaxation, an NVT stage, an NPT stage -- without copying the
  model again.
- **Each MD object** builds its own :class:`~deepmd_lr.nvalchemi.thermo.Thermo`
  (the device state) around that evaluator, and :attr:`MolecularDynamics.evaluator`
  simply returns ``self.thermo.evaluator``, so the two can never diverge.
- **The MD object** owns everything about time and the ensemble: the timestep,
  temperature and pressure targets, thermostat/barostat state, the step count,
  the log, the trajectory, observers and checkpoints.

**Units** are ASE's, exactly: ``timestep=0.5 * units.fs``, ``friction`` in
1/(ASE time), ``tdamp``/``pdamp`` in ASE time, ``pressure_au`` in eV/A^3,
``temperature_K`` in kelvin. These are the kernels' native units.

**Sharing, made safe.**
- A stale object re-evaluates before its first kick: the forces buffer is
  shared, and ``Thermo.fresh`` tells whether it still holds this object's own
  forces.
- An object whose cell the evaluator has moved on from (after an NPT stage)
  refuses to step.
- Two objects stepping the same evaluator at the same time is not supported.

**atoms** is kept in step with the device state, both ways:
- **Device to atoms:** positions (unwrapped), momenta, cell and a
  ``SinglePointCalculator`` go back at log, trajectory and observer intervals,
  and at the end of every run (even one that raised).
- **Atoms to device:** edits made to ``atoms`` between runs or by an observer
  are picked up. A different atom order or ``pbc`` is refused at
  construction.

**Fixed atoms.** ASE's ``FixAtoms`` on ``atoms`` holds those atoms where they
start, at rest, in every class here; any other constraint is refused. They
carry no degrees of freedom: temperatures, and the thermostats' targets, count
three per mobile atom, as ``atoms.get_temperature()`` does, and no
centre-of-mass motion is removed, since the fixed atoms already hold the
system in place. The trajectory and ``atoms.calc`` keep the force the model
puts on them; ``atoms.get_forces()`` zeroes it, as ASE does. The barostats,
which strain every position with the cell, refuse them.

**Nothing here adds a host synchronisation per step** beyond the evaluator's
own; everything else happens at intervals.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import math
import os
import pathlib
import sys
import time
import types
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from deepmd_lr.nvalchemi.thermo import Thermo, _fixed_atoms

__all__ = [
    "Langevin",
    "MDInstabilityError",
    "MTKNPH",
    "MTKNPT",
    "MolecularDynamics",
    "NoseHooverChainNVT",
    "VelocityRescaling",
    "VelocityVerlet",
]

#: Bumped whenever the checkpoint layout changes.
CHECKPOINT_VERSION = 2


class MDInstabilityError(RuntimeError):
    """The trajectory left physical territory (non-finite energy, or a
    temperature above ``max_temperature``)."""


@functools.cache
def _kernels() -> types.SimpleNamespace:
    """nvalchemiops' integrator kernels, imported on first use.

    Lazily, like `Thermo.update_pressure`: an upstream break in that module
    should not take out `import deepmd_lr.nvalchemi`.
    """
    from nvalchemiops.dynamics import integrators as k

    return types.SimpleNamespace(
        **{
            name: getattr(k, name)
            for name in (
                "velocity_verlet_position_update",
                "velocity_verlet_velocity_finalize",
                "langevin_baoab_half_step",
                "langevin_baoab_finalize",
                "nhc_compute_masses",
                "nhc_thermostat_chain_update",
                "nhc_velocity_half_step",
                "nhc_position_update",
                "nhc_compute_chain_energy",
                "npt_thermostat_half_step",
                "npt_barostat_half_step",
                "npt_velocity_half_step",
                "npt_position_update",
                "npt_cell_update",
                "nph_barostat_half_step",
                "nph_velocity_half_step",
                "velocity_rescale",
            )
        },
        vec9d=k.npt.vec9d,
    )


# ----------------------------------------------------------------------------
# Shared by the MD classes and the relaxations
# ----------------------------------------------------------------------------


def _check_evaluator(evaluator: Any) -> None:
    """Refuse what is obviously not an evaluator, with the likely mix-up named."""
    if isinstance(evaluator, Atoms):
        raise TypeError(
            "the first argument is an Atoms object: these are the GPU classes, "
            "which take a DistributedEvaluator first (build_evaluator(model, "
            "atoms, ...)), not ase.md.*"
        )
    if isinstance(evaluator, (str, os.PathLike)):
        raise TypeError(
            "the first argument is a path: build the evaluator once with "
            "build_evaluator(model, atoms, ...) and pass it in -- MD classes "
            "never build one, so several can share it"
        )
    for name in ("forces", "cell", "atype", "calls", "set_cell"):
        if not hasattr(evaluator, name):
            raise TypeError(
                f"expected a DistributedEvaluator, got {type(evaluator).__name__}"
            )


def _check_atoms(evaluator: Any, atoms: Atoms) -> None:
    """``atoms`` must describe the system the evaluator was built for."""
    if not isinstance(atoms, Atoms):
        raise TypeError(f"expected an ase.Atoms, got {type(atoms).__name__}")
    if len(atoms) != evaluator.n_atoms:
        raise ValueError(
            f"atoms has {len(atoms)} atoms, the evaluator {evaluator.n_atoms}"
        )
    type_map = list(evaluator.model.get_type_map())
    expected = [type_map[i] for i in np.asarray(evaluator.atype)]
    if atoms.get_chemical_symbols() != expected:
        first = next(
            i
            for i, (a, b) in enumerate(
                zip(atoms.get_chemical_symbols(), expected, strict=True)
            )
            if a != b
        )
        raise ValueError(
            f"atoms is not in the evaluator's atom order (first mismatch at "
            f"index {first}: {atoms[first].symbol} vs {expected[first]}); "
            "pass the Atoms the evaluator was built from"
        )
    _fixed_atoms(atoms)
    expected_pbc = tuple(getattr(evaluator, "pbc", (True, True, True)))
    if tuple(bool(p) for p in atoms.pbc) != expected_pbc:
        raise ValueError(
            f"atoms.pbc is {atoms.pbc.tolist()} but the evaluator was built for "
            f"{list(expected_pbc)}; build it from these atoms (build_evaluator "
            "reads their pbc). Molecules and clusters run through the ASE "
            "calculator, deepmd_lr.ase.LongRangeCalculator"
        )
    if not np.allclose(atoms.cell.array, evaluator.cell, rtol=0, atol=1e-9):
        raise ValueError(
            "atoms.cell differs from the evaluator's cell. After an NPT stage "
            "pass that stage's atoms (it carries the new cell), or call "
            "evaluator.set_cell(atoms.cell) deliberately"
        )


class _Logger:
    """The thermo table: one ``#`` header line, then whitespace-separated
    rows. ``numpy.loadtxt`` reads it directly. ``'-'`` writes to stdout."""

    def __init__(self, target: Any, columns: list, append: bool):
        self.columns = columns  # [(name, width, spec)]
        self.path = None
        header = "".join(f"{name:>{width}}" for name, width, _ in columns)
        header = "#" + header[1:]
        fresh = True
        if target == "-":
            self.fh, self._owned = sys.stdout, False
        elif isinstance(target, (str, os.PathLike)):
            self.path = pathlib.Path(target)
            fresh = not (append and self.path.exists() and self.path.stat().st_size)
            self.fh, self._owned = open(self.path, "a" if append else "w"), True
        else:
            self.fh, self._owned = target, False
        if fresh:
            self.fh.write(header + "\n")
            self.fh.flush()

    def write(self, row: Mapping[str, float]) -> None:
        fields = [f"{row[name]:>{width}{spec}}" for name, width, spec in self.columns]
        # A value wider than its column (a blow-up's temperature) would run
        # into its neighbour and cost loadtxt the whole file.
        line = "".join(f if f[:1].isspace() else " " + f for f in fields)
        self.fh.write(line + "\n")
        self.fh.flush()

    def tell(self) -> int | None:
        return self.fh.tell() if self.path is not None else None

    def close(self) -> None:
        if self._owned:
            self.fh.close()


class _Trajectory:
    """Frames to ASE's ``.traj``, or to extxyz through one handle for any
    other suffix. An extxyz frame is flushed as it is written, so a
    checkpoint can record the byte offset to trim back to."""

    def __init__(self, path: Any, append: bool):
        self.path = pathlib.Path(path)
        self.kind = "traj" if self.path.suffix == ".traj" else "extxyz"
        mode = "a" if append else "w"
        if self.kind == "traj":
            from ase.io.trajectory import Trajectory

            self.writer = Trajectory(str(self.path), mode)
        else:
            self.fh = open(self.path, mode)

    def write(self, frame: Atoms) -> None:
        if self.kind == "traj":
            self.writer.write(frame)
        else:
            import ase.io

            ase.io.write(self.fh, frame, format="extxyz")
            self.fh.flush()

    def close(self) -> None:
        (self.writer if self.kind == "traj" else self.fh).close()


def _sync_atoms(atoms: Atoms, thermo: Thermo, *, results: bool = True) -> dict:
    """Copy the device state into ``atoms``; return what was written.

    The returned snapshot is what edits are detected against. It holds the
    momenta as written rather than recomputed, because velocity -> momentum
    -> velocity is not bit-exact, and an unconditional re-upload would break
    exact continuation.
    """
    ev = thermo.evaluator
    atoms.set_cell(np.asarray(ev.cell), scale_atoms=False)
    atoms.set_positions(thermo.positions.numpy(), apply_constraint=False)
    atoms.set_velocities(thermo.velocities.numpy())
    if results and math.isfinite(thermo.energy):
        extra = {}
        res = ev.results if hasattr(ev, "results") else {}
        charge = res.get("charge") if res else None
        if charge is not None:
            extra["charges"] = np.asarray(charge).reshape(-1).copy()
        # The model's forces, the fixed atoms' included: ASE applies the
        # constraint itself on get_forces(), as with any calculator.
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=thermo.energy,
            forces=thermo.get_forces(apply_constraint=False),
            **extra,
        )
        born = res.get("born_charge") if res else None
        if born is not None:
            atoms.arrays["born_charges"] = (
                np.asarray(born).reshape(len(atoms), 9).copy()
            )
    return {
        "positions": atoms.get_positions().copy(),
        "momenta": atoms.get_momenta().copy(),
        "cell": atoms.cell.array.copy(),
    }


def _fingerprint(evaluator: Any, masses: np.ndarray, fixed: Any = ()) -> dict:
    """What a checkpoint must share with the evaluator it is restored onto."""
    type_map = list(evaluator.model.get_type_map())
    symbols = [type_map[i] for i in np.asarray(evaluator.atype)]
    fingerprint = {
        "n_atoms": int(evaluator.n_atoms),
        "symbols_sha1": hashlib.sha1(",".join(symbols).encode()).hexdigest(),
        "masses_sha1": hashlib.sha1(
            np.round(np.asarray(masses, float), 8).tobytes()
        ).hexdigest(),
        "type_map": type_map,
        "rcut": float(evaluator.model.get_rcut()),
        "efield": None
        if evaluator.efield is None
        else [float(x) for x in evaluator.efield],
        "eps_inf": float(evaluator.eps_inf),
        "charge_spin": (
            None
            if evaluator.charge_spin is None
            else [float(x) for x in np.asarray(evaluator.charge_spin).reshape(-1)]
        ),
        "compute_virial": bool(evaluator.compute_virial),
    }
    pbc = [bool(p) for p in getattr(evaluator, "pbc", (True, True, True))]
    if not all(pbc):
        # The boundary is physics, as the field is: a slab sums its images
        # differently. Recorded for a slab only, so checkpoints from before
        # slabs ran here still restore.
        fingerprint["pbc"] = pbc
    if len(fixed):
        # Likewise only when there are any.
        fingerprint["fixed_sha1"] = hashlib.sha1(
            np.asarray(fixed, dtype=np.int64).tobytes()
        ).hexdigest()
    return fingerprint


# ----------------------------------------------------------------------------
# The base class
# ----------------------------------------------------------------------------


class MolecularDynamics:
    """The step loop, logging, trajectory, observers and checkpoints.

    Subclasses implement :meth:`step` -- the kernels around exactly one
    ``self.thermo.evaluate()`` -- plus their parameters (:meth:`_params`) and
    state (:meth:`_state`, :meth:`_load_state`).

    Parameters
    ----------
    evaluator
        A :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator`, shared
        as it is.
    atoms
        The system, in the evaluator's atom order and cell. Positions,
        velocities (from momenta), masses and any ``FixAtoms`` are read from
        it; it is written back as the run goes.
    timestep
        In ASE time units: ``0.5 * units.fs``.
    logfile, loginterval
        The thermo table (``'-'`` for stdout, a path, or None) and its period
        in steps.
    extra_columns
        ``{name: fn(md) -> float}``, appended to every log row.
    trajectory, traj_interval, trajectory_properties, append_trajectory
        Frames of unwrapped positions: ``.traj`` through ``ase.io.Trajectory``,
        anything else as extxyz. The interval defaults to ``loginterval``.
        Properties: ``energy``, ``forces``, ``velocities``, ``charges``,
        ``born_charges``.
    checkpoint, checkpoint_interval
        An atomic npz written every ``checkpoint_interval`` steps, on
        :meth:`stop`, and read back by :meth:`from_checkpoint`.
    max_temperature
        Raise :class:`MDInstabilityError` when a logged temperature exceeds it
        (K). None disables the check.
    """

    #: Log columns this class adds after Etot, e.g. ``("Econs[eV]", ...)``.
    _EXTRA_BUILTIN: tuple = ()

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        timestep: float,
        *,
        logfile: Any = None,
        loginterval: int = 1,
        extra_columns: Mapping[str, Callable[[MolecularDynamics], float]] | None = None,
        trajectory: Any = None,
        traj_interval: int | None = None,
        trajectory_properties: tuple = ("energy", "forces"),
        append_trajectory: bool = False,
        checkpoint: Any = None,
        checkpoint_interval: int | None = None,
        max_temperature: float | None = None,
        _restoring: bool = False,
    ) -> None:
        _check_evaluator(evaluator)
        _check_atoms(evaluator, atoms)
        self.atoms = atoms
        self.thermo = Thermo.from_atoms(evaluator, atoms)
        self._masses = np.array(atoms.get_masses(), dtype=np.float64)
        self.timestep = float(timestep)
        if not self.timestep > 0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        if self.timestep > 4 * units.fs:
            warnings.warn(
                f"timestep is {self.timestep / units.fs:.2f} fs -- unusually "
                "large. The unit is ASE time: did you mean 0.5 * units.fs?",
                stacklevel=2,
            )
        self.dt = self.thermo.scalar(self.timestep)
        self.nsteps = 0
        self._time = 0.0
        self.max_temperature = max_temperature
        self._observers: list = []
        self._stopping = False
        self._started = _restoring
        self._clock: list = []
        self._rows: list = []
        if loginterval < 1:
            raise ValueError("loginterval must be at least 1")
        self.loginterval = int(loginterval)
        self._extra = dict(extra_columns or {})
        self._logger = (
            None
            if logfile is None
            else _Logger(logfile, self._columns(), append=_restoring)
        )
        self._traj = None
        self.traj_interval = int(traj_interval or loginterval)
        self.trajectory_properties = tuple(trajectory_properties)
        if trajectory is not None:
            self._traj = _Trajectory(trajectory, append_trajectory or _restoring)
        self.checkpoint = None if checkpoint is None else pathlib.Path(checkpoint)
        self.checkpoint_interval = checkpoint_interval
        self._io = {
            "logfile": (
                str(logfile) if isinstance(logfile, (str, os.PathLike)) else None
            ),
            "loginterval": self.loginterval,
            "trajectory": None if trajectory is None else str(trajectory),
            "traj_interval": self.traj_interval,
            "trajectory_properties": list(self.trajectory_properties),
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "checkpoint_interval": checkpoint_interval,
            "max_temperature": max_temperature,
        }
        if not _restoring:
            self._remove_com_drift()
        self._snapshot = _sync_atoms(atoms, self.thermo, results=False)

    # --- the shared evaluator ----------------------------------------------

    @property
    def evaluator(self) -> Any:
        """The evaluator this object and its Thermo share. Read-only: the
        only reference is the Thermo's, so the two cannot diverge."""
        return self.thermo.evaluator

    @property
    def results(self) -> Any:
        """The evaluator's results from the last evaluation."""
        return self.evaluator.results

    # --- what subclasses provide -------------------------------------------

    def step(self) -> None:
        """Advance one step: the kernels around exactly one evaluate."""
        raise NotImplementedError

    def conserved_energy(self) -> float | None:
        """The ensemble's conserved quantity in eV, where it has one."""
        return None

    def _params(self) -> dict:
        """Constructor parameters beyond the base class, for checkpoints."""
        return {}

    def _state(self) -> dict:
        """Integrator arrays for checkpoints."""
        return {}

    def _load_state(self, arrays: Mapping[str, np.ndarray]) -> None:
        pass

    # --- running -----------------------------------------------------------

    def run(self, steps: int) -> None:
        """Run ``steps`` more steps, as ASE's ``run`` does."""
        for _ in self.irun(steps):
            pass

    def irun(self, steps: int) -> Iterator[int]:
        """Generator form of :meth:`run`; yields the step count after each step."""
        steps = int(steps)
        try:
            self._require_current_cell()
            self._pull_atoms_edits()
            if not self.thermo.fresh:
                self.thermo.evaluate()
                self._require_finite()
            if not self._started:
                self._started = True
                self._at_intervals(first=True)
            for _ in range(steps):
                if self._stopping:
                    break
                self._require_current_cell()
                self.step()
                self.nsteps += 1
                self._time += self.timestep
                self._require_finite()
                self._at_intervals()
                if (
                    self.checkpoint is not None
                    and self.checkpoint_interval
                    and self.nsteps % self.checkpoint_interval == 0
                ):
                    self.save_checkpoint()
                yield self.nsteps
            if self._stopping:
                self._stopping = False
                if self.checkpoint is not None:
                    self.save_checkpoint()
        finally:
            if math.isfinite(self.thermo.energy):
                self._snapshot = _sync_atoms(self.atoms, self.thermo)
            for out in (self._logger, self._traj):
                if out is not None and hasattr(out, "fh"):
                    out.fh.flush()

    def stop(self) -> None:
        """End the run at the end of the current step, writing the checkpoint
        if one is configured. Call it from an observer, or a signal handler."""
        self._stopping = True

    def attach(
        self,
        fn: Callable,
        interval: int = 1,
        *args: Any,
        sync_atoms: bool = True,
        **kwargs: Any,
    ) -> None:
        """Call ``fn(*args, **kwargs)`` every ``interval`` steps, at absolute
        multiples of the step count, as ASE does. With ``sync_atoms`` the
        device state is copied into ``atoms`` first, and edits the observer
        makes to ``atoms`` are picked up after; switch it off for a cheap
        every-step observer that does not read ``atoms``."""
        if int(interval) < 1:
            raise ValueError("interval must be at least 1")
        self._observers.append((fn, int(interval), args, kwargs, bool(sync_atoms)))

    def get_time(self) -> float:
        """Elapsed simulation time, in ASE time units."""
        return self._time

    # --- per-interval work --------------------------------------------------

    def _at_intervals(self, first: bool = False) -> None:
        n = self.nsteps
        log_due = self._logger is not None and (first or n % self.loginterval == 0)
        traj_due = self._traj is not None and (first or n % self.traj_interval == 0)
        due = [o for o in self._observers if first or n % o[1] == 0]
        if log_due or traj_due or any(o[4] for o in due):
            self._snapshot = _sync_atoms(self.atoms, self.thermo)
        if traj_due:
            self._write_frame()
        if log_due:
            self._log_row()
        elif self.max_temperature is not None and (first or n % self.loginterval == 0):
            self._check_temperature(self.thermo.temperature)
        for fn, _, args, kwargs, _sync in due:
            fn(*args, **kwargs)
        if any(o[4] for o in due):
            self._pull_atoms_edits()

    def _columns(self) -> list:
        cols = [
            ("step", 10, "d"),
            ("time[ps]", 12, ".4f"),
            ("T[K]", 10, ".2f"),
            ("Epot[eV]", 18, ".6f"),
            ("Ekin[eV]", 16, ".6f"),
            ("Etot[eV]", 18, ".6f"),
        ]
        cols += [(name, 18, ".6f") for name in self._EXTRA_BUILTIN]
        cols += [(name, max(14, len(name) + 2), ".6g") for name in self._extra]
        return cols + [("ms/step", 10, ".1f")]

    def _row(self) -> dict:
        th = self.thermo
        ekin = th.kinetic_energy
        row = {
            "step": self.nsteps,
            "time[ps]": self._time / (1000 * units.fs),
            "T[K]": th.temperature,
            "Epot[eV]": th.energy,
            "Ekin[eV]": ekin,
            "Etot[eV]": th.energy + ekin,
        }
        row.update(self._builtin_extras())
        for name, fn in self._extra.items():
            row[name] = float(fn(self))
        return row

    def _builtin_extras(self) -> dict:
        return {}

    def _log_row(self) -> None:
        now = time.perf_counter()
        row = self._row()
        if self._clock:
            s_prev, t_prev = self._clock[-1]
            row["ms/step"] = (
                (now - t_prev) / (self.nsteps - s_prev) * 1e3
                if self.nsteps > s_prev
                else float("nan")
            )
        else:
            row["ms/step"] = float("nan")
        self._clock.append((self.nsteps, now))
        self._rows.append(row)
        self._logger.write(row)
        self._check_temperature(row["T[K]"])

    def _check_temperature(self, temperature: float) -> None:
        if self.max_temperature is not None and not temperature <= self.max_temperature:
            raise MDInstabilityError(
                f"temperature {temperature:.0f} K exceeds max_temperature "
                f"{self.max_temperature:g} K at step {self.nsteps}"
            )

    def _require_finite(self) -> None:
        if not math.isfinite(self.thermo.energy):
            raise MDInstabilityError(f"non-finite energy at step {self.nsteps}")

    def _require_current_cell(self) -> None:
        if not self.thermo.cell_is_current:
            raise RuntimeError(
                "the evaluator's cell has changed since this object last set it "
                "(another object ran NPT on the shared evaluator). Build a new "
                "MD object from that stage's atoms instead of resuming this one."
            )

    def _remove_com_drift(self) -> None:
        if len(self.thermo.fixed):
            return  # they hold the system in place; Thermo has them at rest
        v, m = self.thermo.velocities.numpy(), self._masses
        p = (m[:, None] * v).sum(axis=0)
        ke = 0.5 * (m[:, None] * v**2).sum()
        ke_com = 0.5 * float(p @ p) / m.sum()
        if ke > 0 and ke_com > 1e-6 * ke:
            warnings.warn(
                f"removed a centre-of-mass drift carrying {ke_com / ke:.2%} of "
                "the kinetic energy; every temperature here counts 3N - 3 "
                "degrees of freedom, which assumes the centre of mass at rest",
                stacklevel=3,
            )
        self.thermo.remove_com_motion()

    def _pull_atoms_edits(self) -> None:
        """Upload what was changed on ``atoms`` since the last sync."""
        snap, atoms = self._snapshot, self.atoms
        if not np.array_equal(atoms.cell.array, snap["cell"]):
            raise ValueError(
                "atoms.cell was edited since the last sync; the cell changes only "
                "through NPT (or evaluator.set_cell, then a new MD object)"
            )
        if not np.array_equal(atoms.get_masses(), self._masses):
            raise ValueError("atoms masses were changed after construction")
        if not np.array_equal(_fixed_atoms(atoms), self.thermo.fixed):
            raise ValueError(
                "atoms' FixAtoms changed after construction, and with it the "
                "degrees of freedom the temperature counts; build a new MD "
                "object for the new constraint"
            )
        moved = not np.array_equal(atoms.get_positions(), snap["positions"])
        kicked = not np.array_equal(atoms.get_momenta(), snap["momenta"])
        if not (moved or kicked):
            return
        self.thermo.set_state(
            positions=atoms.get_positions() if moved else None,
            velocities=atoms.get_velocities() if kicked else None,
        )
        if moved:
            self.thermo.evaluate()
            self._require_finite()
        self._snapshot = _sync_atoms(atoms, self.thermo, results=moved)

    # --- trajectory ----------------------------------------------------------

    def _write_frame(self) -> None:
        props = self.trajectory_properties
        frame = Atoms(
            self.atoms.get_chemical_symbols(),
            positions=self.atoms.get_positions(),
            cell=self.atoms.cell,
            pbc=self.atoms.pbc,
            masses=self._masses,
        )
        if len(self.thermo.fixed):
            frame.set_constraint(FixAtoms(indices=self.thermo.fixed))
        if "velocities" in props:
            frame.set_momenta(self.atoms.get_momenta())
        results = {}
        if "energy" in props:
            results["energy"] = self.thermo.energy
        if "forces" in props:
            results["forces"] = self.thermo.get_forces(apply_constraint=False)
        res = self.evaluator.results
        if "charges" in props and res.get("charge") is not None:
            results["charges"] = np.asarray(res["charge"]).reshape(-1).copy()
        if "born_charges" in props and res.get("born_charge") is not None:
            frame.arrays["born_charges"] = (
                np.asarray(res["born_charge"]).reshape(-1, 9).copy()
            )
        if results:
            frame.calc = SinglePointCalculator(frame, **results)
        frame.info["step"] = self.nsteps
        frame.info["time_ps"] = self._time / (1000 * units.fs)
        self._traj.write(frame)

    def close(self) -> None:
        """Close the log and trajectory files."""
        if self._logger is not None:
            self._logger.close()
        if self._traj is not None:
            self._traj.close()

    # --- diagnostics ----------------------------------------------------------

    @property
    def rows(self) -> list:
        """The log rows of this object's runs so far, as dicts (column -> value)."""
        return list(self._rows)

    @property
    def seconds_per_step(self) -> float:
        """Wall-clock seconds per step over the logged history, first interval
        (kernel compilation, the memory probe) excluded."""
        pts = self._clock[1:] if len(self._clock) > 2 else self._clock
        if len(pts) < 2 or pts[-1][0] == pts[0][0]:
            return float("nan")
        return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])

    @property
    def ns_per_day(self) -> float:
        """Throughput in simulated ns per day at the current timestep."""
        return (self.timestep / (1e6 * units.fs)) * 86400.0 / self.seconds_per_step

    def performance(self) -> str:
        """One line of throughput, for the end of a run."""
        return (
            f"{self.thermo.n_atoms} atoms, dt {self.timestep / units.fs:g} fs: "
            f"{self.seconds_per_step * 1e3:.1f} ms/step -> {self.ns_per_day:.3f} ns/day"
        )

    def drift(self, column: str = "Etot[eV]", per_atom: bool = True) -> float:
        """Least-squares drift of a logged column, in meV (per atom) per ps."""
        if len(self._rows) < 2:
            return float("nan")
        t = np.array([r["time[ps]"] for r in self._rows])
        y = np.array([r[column] for r in self._rows]) * 1e3
        if per_atom:
            y /= self.thermo.n_atoms
        return float(np.polyfit(t, y, 1)[0])

    def dipole_current(self) -> np.ndarray:
        """J_a = sum_i sum_b Z_iab v_ib at the current step, in e A/(ASE time).

        The last index of Z (atom, polarisation, displacement) is contracted
        with the velocity. Needs Born charges: build the evaluator with
        ``compute_born=True`` or an ``efield``.
        """
        born = self.evaluator.results.get("born_charge")
        if born is None:
            raise ValueError(
                "no Born charges: build the evaluator with compute_born=True"
            )
        return np.einsum("iab,ib->a", np.asarray(born), self.thermo.velocities.numpy())

    # --- checkpoints --------------------------------------------------------

    def save_checkpoint(self, path: Any = None) -> pathlib.Path:
        """Write an atomic npz: the state, the parameters, and the fingerprint
        of the evaluator it belongs to."""
        path = pathlib.Path(path or self.checkpoint)
        meta = {
            "format": "deepmd_lr.nvalchemi.md checkpoint",
            "version": CHECKPOINT_VERSION,
            "class": type(self).__name__,
            "timestep": self.timestep,
            "params": self._params(),
            "io": self._io,
            "nsteps": self.nsteps,
            "time": self._time,
            "fingerprint": _fingerprint(
                self.evaluator, self._masses, self.thermo.fixed
            ),
            "log_bytes": None if self._logger is None else self._logger.tell(),
            "traj_bytes": (
                None
                if self._traj is None or self._traj.kind != "extxyz"
                else self._traj.fh.tell()
            ),
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        arrays = {
            "positions": self.thermo.positions.numpy(),
            "velocities": self.thermo.velocities.numpy(),
            "cell": np.asarray(self.evaluator.cell, dtype=np.float64),
            **self._state(),
        }
        tmp = path.with_name("." + path.stem + ".tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, path)
        return path

    @classmethod
    def from_checkpoint(
        cls, evaluator: Any, atoms: Atoms, path: Any, **io_overrides: Any
    ) -> MolecularDynamics:
        """Continue a run exactly where :meth:`save_checkpoint` left it.

        The checkpoint must come from this class, on this system (atom
        order, masses, fixed atoms) and on an evaluator with the same physics
        (cutoff, boundary, field, eps_inf, charge/spin, virial); a mismatch is
        refused before anything changes. The state is written into ``atoms``,
        the evaluator's cell set, and the log and trajectory are trimmed back
        to the checkpoint after a hard kill. Only output settings may be
        overridden; ``extra_columns`` has to be passed again.
        """
        _check_evaluator(evaluator)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        if meta.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path}: checkpoint version {meta.get('version')}, "
                f"this code reads {CHECKPOINT_VERSION}"
            )
        if meta["class"] != cls.__name__:
            raise ValueError(
                f"{path} was written by {meta['class']}, not {cls.__name__}"
            )
        allowed = {
            "logfile",
            "loginterval",
            "extra_columns",
            "trajectory",
            "traj_interval",
            "trajectory_properties",
            "checkpoint",
            "checkpoint_interval",
            "max_temperature",
        }
        if set(io_overrides) - allowed:
            raise TypeError(
                f"only output settings can be overridden on restore: "
                f"{sorted(set(io_overrides) - allowed)}"
            )
        # Checked before anything is written, so a refusal leaves atoms and
        # the evaluator as they were.
        want = _fingerprint(evaluator, atoms.get_masses(), _fixed_atoms(atoms))
        if meta["fingerprint"] != want:
            had = meta["fingerprint"]
            diff = {
                k: (had.get(k), want.get(k))
                for k in sorted(set(had) | set(want))
                if had.get(k) != want.get(k)
            }
            raise ValueError(
                f"{path} belongs to a different system or evaluator: "
                f"(checkpoint, now) = {diff}"
            )
        atoms.set_cell(arrays["cell"], scale_atoms=False)
        atoms.set_positions(arrays["positions"], apply_constraint=False)
        atoms.set_velocities(arrays["velocities"])
        if not np.array_equal(np.asarray(evaluator.cell), arrays["cell"]):
            evaluator.set_cell(arrays["cell"])
        io = {k: v for k, v in meta["io"].items() if v is not None and k in allowed}
        io.update(io_overrides)
        for key, size_key in (("logfile", "log_bytes"), ("trajectory", "traj_bytes")):
            target, size = io.get(key), meta.get(size_key)
            if isinstance(target, (str, os.PathLike)) and size is not None:
                p = pathlib.Path(target)
                if p.exists() and p.stat().st_size > size:
                    os.truncate(p, size)
        md = cls(
            evaluator, atoms, meta["timestep"], **meta["params"], **io, _restoring=True
        )
        md.nsteps, md._time = int(meta["nsteps"]), float(meta["time"])
        md._load_state(arrays)
        return md


# ----------------------------------------------------------------------------
# Integrators. Each copies a tested loop, and every kernel gets the FULL dt
# (the kernels apply their own factors of 1/2).
# ----------------------------------------------------------------------------


class VelocityVerlet(MolecularDynamics):
    """NVE. The total energy is conserved: the check for a new model, system
    or timestep. Its fluctuation falls ~4x when the timestep is halved."""

    def step(self) -> None:
        k, th = _kernels(), self.thermo
        k.velocity_verlet_position_update(
            th.positions, th.velocities, th.forces, th.masses, self.dt
        )
        th.evaluate()
        k.velocity_verlet_velocity_finalize(
            th.velocities, th.forces, th.masses, self.dt
        )

    def conserved_energy(self) -> float:
        return self.thermo.total_energy


class VelocityRescaling(VelocityVerlet):
    """Velocity rescaling, nvalchemiops' simplest thermostat: velocity Verlet,
    and every ``interval`` steps the velocities are scaled by
    sqrt(T_target / T) once the temperature has left the window
    ``temperature_K +- tolerance``, as LAMMPS' fix temp/rescale does.

    It samples no ensemble and disturbs the dynamics at every rescale: use it
    to drag a structure to temperature, then hand it to :class:`Langevin` or
    :class:`NoseHooverChainNVT`. :attr:`rescales` counts the rescales.
    """

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        *,
        interval: int = 1,
        tolerance: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not temperature_K > 0:
            raise ValueError("temperature_K must be positive")
        if int(interval) < 1 or not tolerance >= 0:
            raise ValueError("interval must be at least 1 and tolerance >= 0")
        super().__init__(evaluator, atoms, timestep, **kwargs)
        self.temperature_K = float(temperature_K)
        self.interval, self.tolerance = int(interval), float(tolerance)
        self.rescales = 0
        self._scale = self.thermo.scalar(1.0)

    def set_temperature(self, temperature_K: float) -> None:
        """Change the target temperature in place."""
        self.temperature_K = float(temperature_K)

    def step(self) -> None:
        super().step()
        if (self.nsteps + 1) % self.interval:  # the step being taken
            return
        t = self.thermo.temperature  # a host read, every interval only
        if t > 0 and abs(t - self.temperature_K) > self.tolerance:
            self._scale.fill_(math.sqrt(self.temperature_K / t))
            _kernels().velocity_rescale(self.thermo.velocities, self._scale)
            self.rescales += 1

    def conserved_energy(self) -> None:
        return None

    def _params(self) -> dict:
        return {
            "temperature_K": self.temperature_K,
            "interval": self.interval,
            "tolerance": self.tolerance,
        }


class Langevin(MolecularDynamics):
    """NVT, BAOAB Langevin.

    The kernel's noise is stateless, seeded by ``(seed + nsteps) mod 2^31``,
    so a restart replays exactly the noise an uninterrupted run would draw.
    ``fixcm`` (ASE's default) removes the centre-of-mass motion the noise adds
    each step, which makes ``3N - 3`` degrees of freedom exact.

    With fixed atoms ``fixcm`` does nothing: they hold the system in place,
    and every mobile atom is thermostatted in all three directions. The
    kernel gives the fixed atoms noise too, which ``Thermo.evaluate`` takes
    away again before the forces are computed.
    """

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        friction: float,
        *,
        fixcm: bool = True,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not temperature_K >= 0 or not friction > 0:
            raise ValueError("temperature_K must be >= 0 and friction > 0")
        super().__init__(evaluator, atoms, timestep, **kwargs)
        self.temperature_K = float(temperature_K)
        self.kT = self.thermo.scalar(units.kB * self.temperature_K)
        self.friction = float(friction)
        self._friction = self.thermo.scalar(self.friction)
        self.fixcm = bool(fixcm)
        self._fix_com = self.fixcm and not len(self.thermo.fixed)
        self.seed = (
            int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little")
        )
        self.seed %= 2**31

    def set_temperature(self, temperature_K: float) -> None:
        """Change the target temperature in place."""
        self.temperature_K = float(temperature_K)
        self.kT.fill_(units.kB * self.temperature_K)

    def step(self) -> None:
        k, th = _kernels(), self.thermo
        k.langevin_baoab_half_step(
            th.positions,
            th.velocities,
            th.forces,
            th.masses,
            self.dt,
            self.kT,
            self._friction,
            (self.seed + self.nsteps) % 2**31,
        )
        th.evaluate()
        k.langevin_baoab_finalize(th.velocities, th.forces, th.masses, self.dt)
        if self._fix_com:
            th.remove_com_motion()

    def _params(self) -> dict:
        return {
            "temperature_K": self.temperature_K,
            "friction": self.friction,
            "fixcm": self.fixcm,
            "seed": self.seed,
        }


class NoseHooverChainNVT(MolecularDynamics):
    """NVT, Nose-Hoover chain (Martyna-Tuckerman splitting); deterministic.

    Conventions verified on water (bend ~1640, stretch ~3400 cm-1):
    - The full dt goes to every call. Passing dt/2 to the kick halves every
      force.
    - ndof = 3N - 3, or three per mobile atom with fixed atoms
      (``Thermo.ndof``).
    - ``total_scale`` is reset to 1 before each chain update.
    - :meth:`conserved_energy` uses **eta/2**: each chain update scales the
      velocities by a half step but moves eta by a full ``eta_dot * dt``, so
      eta runs twice as fast as eta_dot implies. eta never feeds back into the
      dynamics, so the trajectory is right.
    """

    #: Yoshida-Suzuki stages per chain update; ASE's ``tloop=1`` is one such sweep.
    _NLOOPS = 3
    _EXTRA_BUILTIN = ("Econs[eV]",)

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        tdamp: float,
        *,
        tchain: int = 3,
        tloop: int = 1,
        **kwargs: Any,
    ) -> None:
        if not temperature_K > 0 or not tdamp > 0:
            raise ValueError("temperature_K and tdamp must be positive")
        if not 1 <= int(tchain) <= 8:
            raise ValueError("tchain must be between 1 and 8")
        if int(tloop) != 1:
            raise ValueError(
                "only tloop=1 (one 3-stage Yoshida-Suzuki sweep) is supported"
            )
        super().__init__(evaluator, atoms, timestep, **kwargs)
        import warp as wp

        th, k = self.thermo, _kernels()
        self.temperature_K, self.tdamp = float(temperature_K), float(tdamp)
        self.tchain, self.tloop = int(tchain), 1
        self.kT = th.scalar(units.kB * self.temperature_K)
        ndof = th.ndof
        self._ndof = th.scalar(float(ndof))  # float64
        ndof_i32 = wp.array(np.array([ndof]), dtype=wp.int32, device=th.device)
        self.eta, self.eta_dot = th.zeros(self.tchain), th.zeros(self.tchain)
        self.eta_mass = th.zeros(self.tchain)
        k.nhc_compute_masses(
            ndof_i32, self.kT, th.scalar(self.tdamp), self.tchain, self.eta_mass
        )
        self._ke2, self._step_scale, self._dt_chain = (
            th.zeros(1),
            th.zeros(1),
            th.zeros(1),
        )
        self._total_scale = th.scalar(1.0)
        self._ke_chain, self._pe_chain = th.zeros(1), th.zeros(1)

    def _chain(self) -> None:
        th = self.thermo
        self._total_scale.fill_(1.0)
        _kernels().nhc_thermostat_chain_update(
            th.velocities,
            th.masses,
            self.eta,
            self.eta_dot,
            self.eta_mass,
            self.kT,
            self.dt,
            self._ndof,
            self._ke2,
            self._total_scale,
            self._step_scale,
            self._dt_chain,
            nloops=self._NLOOPS,
        )

    def step(self) -> None:
        k, th = _kernels(), self.thermo
        self._chain()
        k.nhc_velocity_half_step(th.velocities, th.forces, th.masses, self.dt)
        k.nhc_position_update(th.positions, th.velocities, self.dt)
        th.evaluate()
        k.nhc_velocity_half_step(th.velocities, th.forces, th.masses, self.dt)
        self._chain()

    def conserved_energy(self) -> float:
        import warp as wp

        th = self.thermo
        half_eta = wp.array(0.5 * self.eta.numpy(), dtype=wp.float64, device=th.device)
        _kernels().nhc_compute_chain_energy(
            half_eta,
            self.eta_dot,
            self.eta_mass,
            self.kT,
            self._ndof,
            self._ke_chain,
            self._pe_chain,
        )
        return (
            th.total_energy
            + float(self._ke_chain.numpy()[0])
            + float(self._pe_chain.numpy()[0])
        )

    def _builtin_extras(self) -> dict:
        return {"Econs[eV]": self.conserved_energy()}

    def _params(self) -> dict:
        return {
            "temperature_K": self.temperature_K,
            "tdamp": self.tdamp,
            "tchain": self.tchain,
            "tloop": self.tloop,
        }

    def _state(self) -> dict:
        return {"eta": self.eta.numpy(), "eta_dot": self.eta_dot.numpy()}

    def _load_state(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.eta.assign(arrays["eta"])
        self.eta_dot.assign(arrays["eta_dot"])


#: What a barostat lets the cell do, and nvalchemiops' name for each mode.
_BAROSTAT_MODES = {
    "isotropic": "isotropic",
    "orthorhombic": "anisotropic",
    "full": "triclinic",
}


class _MTKBarostat(MolecularDynamics):
    """The Martyna-Tobias-Klein barostat that NPT and NPH share.

    - **Fresh inputs.** The kinetic energy and pressure are refreshed right
      before each barostat half-step, after the second kick too, as LAMMPS'
      fix nh does. (nvalchemiops' ``run_nph_step`` reuses the pressure it
      measured before that kick.)
    - **Degrees of freedom.** The kernels compute N_f = 3 * (the count they
      are given). The centre of mass is at rest and stays there, so they get
      N - 1, which gives 3N - 3, matching the masses and the temperature.
    - **The cell.** ``cell`` picks what fluctuates: ``"isotropic"`` scales the
      cell uniformly, ``"orthorhombic"`` moves the three lengths of a diagonal
      cell independently, ``"full"`` moves all nine entries. The strain rate is
      driven by the pressure tensor, which is symmetric, so the cell never
      rotates. After each move ``Thermo.set_cell`` pushes the cell to the
      evaluator, whose partitions follow it and rebuild only once the cell
      or the atoms have moved enough.
    - **Conventions.** nvalchemiops' cell update, h + dt eps_dot h, reads the
      columns of h as the lattice vectors, where ASE's cell has them as rows,
      so the kernels get the transpose. For an isotropic eps_dot the two
      agree; for any other they would part on a sheared cell.
    - **Pressure** is hydrostatic: ``pressure_au`` (eV/A^3) on every axis.

    The conserved quantity is H' = Epot + Ekin + P V + W |eps_dot|^2 / 2, plus
    the thermostat's energy under NPT, with W = 3N kT pdamp^2.

    Needs the virial: build the evaluator with ``compute_virial=True``, or
    call ``evaluator.set_compute_virial(True)`` first.
    """

    _EXTRA_BUILTIN = ("Econs[eV]", "P[GPa]", "V[A^3]")

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        timestep: float,
        pressure_au: float,
        pdamp: float,
        *,
        cell: str,
        barostat_temperature_K: float | None,
        **kwargs: Any,
    ) -> None:
        _check_evaluator(evaluator)
        name = type(self).__name__
        if not all(getattr(evaluator, "pbc", (True, True, True))):
            raise ValueError(
                f"{name} moves the cell, and a slab's open axis is not a "
                "periodic length a pressure can act on; run the slab in NVE or "
                "NVT (the in-plane pressure is not controlled)"
            )
        if len(_fixed_atoms(atoms)):
            raise ValueError(
                f"{name} strains every position with the cell, so fixed atoms "
                "would not stay fixed; hold them in NVT or NVE, at a cell "
                "reached without them"
            )
        if not evaluator.compute_virial:
            raise ValueError(
                f"{name} needs the virial: build the evaluator with "
                "compute_virial=True or call evaluator.set_compute_virial(True)"
            )
        if cell not in _BAROSTAT_MODES:
            raise ValueError(
                f"cell must be one of {', '.join(_BAROSTAT_MODES)}; got {cell!r}"
            )
        h = np.asarray(evaluator.cell, dtype=np.float64)
        if cell == "orthorhombic" and np.any(h != np.diag(np.diag(h))):
            raise ValueError(
                "cell='orthorhombic' moves the three lengths of a diagonal cell, "
                "and this cell is sheared; use 'full'"
            )
        if not pdamp > 0:
            raise ValueError("pdamp must be positive")
        super().__init__(evaluator, atoms, timestep, **kwargs)
        import warp as wp

        th = self.thermo
        self.cell_mode = cell
        self.pressure_au, self.pdamp = float(pressure_au), float(pdamp)
        t = (
            th.temperature
            if barostat_temperature_K is None
            else float(barostat_temperature_K)
        )
        if not t > 0:
            raise ValueError(
                f"{name} needs a temperature for the barostat mass: give atoms "
                "velocities (ASE's thermalize_momenta) or pass temperature_K"
            )
        self._barostat_T = t
        self._w = 3 * th.n_atoms * units.kB * t * self.pdamp**2  # (N_f + 3) kT pdamp^2
        self.cell_mass = th.scalar(self._w)
        self.cell_v = th.zeros(1, dtype=wp.mat33d)
        # N - 1, so the kernels' N_f = 3 * count is 3N - 3.
        self.dof_count = wp.array([th.n_atoms - 1], dtype=wp.int32, device=th.device)
        p = self.pressure_au
        if cell == "isotropic":
            self.target_p = th.scalar(p)
        elif cell == "orthorhombic":
            self.target_p = wp.array([[p, p, p]], dtype=wp.vec3d, device=th.device)
        else:
            self.target_p = wp.array(
                [(p * np.eye(3)).reshape(9)], dtype=_kernels().vec9d, device=th.device
            )
        self._velocity_mode = _BAROSTAT_MODES[cell]
        # The kernels' cell: ASE's, transposed (lattice vectors as columns).
        self._h = wp.array(
            h.T.reshape(1, 3, 3).copy(), dtype=wp.mat33d, device=th.device
        )

    def _refresh(self) -> None:
        self.thermo.update_kinetic_energy()
        self.thermo.update_pressure()

    def _drift(self) -> None:
        """Positions, then the cell, then the evaluator."""
        k, th = _kernels(), self.thermo
        k.npt_position_update(
            th.positions, th.velocities, self._h, self.cell_v, self.dt
        )
        k.npt_cell_update(self._h, self.cell_v, self.dt)
        import warp as wp

        # The kernels hold the cell transposed; hand it over on the device.
        th.set_cell_device(wp.to_torch(self._h).reshape(3, 3).T)

    def _barostat_energy(self) -> float:
        """P V + W |eps_dot|^2 / 2."""
        eps = self.cell_v.numpy()[0]
        return self.pressure_au * self.thermo.volume + 0.5 * self._w * float(
            (eps**2).sum()
        )

    def _builtin_extras(self) -> dict:
        return {
            "Econs[eV]": self.conserved_energy(),
            "P[GPa]": self.thermo.pressure / units.GPa,
            "V[A^3]": self.thermo.volume,
        }

    def _params(self) -> dict:
        return {
            "pressure_au": self.pressure_au,
            "pdamp": self.pdamp,
            "cell": self.cell_mode,
        }

    def _state(self) -> dict:
        return {"cell_v": self.cell_v.numpy()}

    def _load_state(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.cell_v.assign(arrays["cell_v"])


class MTKNPT(_MTKBarostat):
    """NPT: the MTK barostat with a Nose-Hoover chain thermostat, as ASE's
    ``MTKNPT``. By default the whole cell fluctuates (``cell="full"``);
    ``cell="isotropic"`` scales it uniformly and ``"orthorhombic"`` moves the
    three lengths of a diagonal cell.

    The step is a palindrome: thermostat, barostat, kick | drift | kick,
    barostat, thermostat. The barostat itself is not thermostatted.

    **How well H' is conserved.** nvalchemiops' velocity kernel applies the
    drag -(alpha eps_dot + eta_dot) v explicitly inside the kick, which is
    first order. While the thermostat works hard (far from equilibrium, strong
    coupling), the error in :meth:`conserved_energy` is O(dt). Near
    equilibrium or with weak coupling the O(dt^2) part dominates: equilibrated
    water at tdamp 100 fs gave 0.0218 -> 0.0054 meV/atom rms from dt 0.5 to
    0.25 fs. :class:`NoseHooverChainNVT` applies its scaling exactly and is
    O(dt^2) at any coupling; :class:`MTKNPH`, with no thermostat, has no such
    term.
    """

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        pressure_au: float,
        tdamp: float,
        pdamp: float,
        *,
        cell: str = "full",
        tchain: int = 3,
        tloop: int = 1,
        **kwargs: Any,
    ) -> None:
        if int(tchain) < 2:
            raise ValueError("tchain must be at least 2 (the NPT kernels index [1])")
        if int(tloop) != 1:
            raise ValueError("only tloop=1 is supported")
        if not temperature_K > 0 or not tdamp > 0:
            raise ValueError("temperature_K and tdamp must be positive")
        super().__init__(
            evaluator,
            atoms,
            timestep,
            pressure_au,
            pdamp,
            cell=cell,
            barostat_temperature_K=temperature_K,
            **kwargs,
        )
        import warp as wp

        th = self.thermo
        self.temperature_K = float(temperature_K)
        self.tdamp = float(tdamp)
        self.tchain, self.tloop = int(tchain), 1
        kT = units.kB * self.temperature_K
        self.kT = th.scalar(kT)
        self._nf = 3 * th.n_atoms - 3
        self.eta, self.eta_dot = th.zeros((1, self.tchain)), th.zeros((1, self.tchain))
        q = np.full((1, self.tchain), kT * self.tdamp**2)
        q[0, 0] *= self._nf  # Q_0 = N_f kT tdamp^2
        self._q = q[0].copy()
        self.th_mass = wp.array(q, dtype=wp.float64, device=th.device)

    def _thermostat(self) -> None:
        th = self.thermo
        _kernels().npt_thermostat_half_step(
            self.eta,
            self.eta_dot,
            th.ke_w,
            self.kT,
            self.th_mass,
            self.dof_count,
            self.tchain,
            self.dt,
        )

    def _barostat(self) -> None:
        th = self.thermo
        _kernels().npt_barostat_half_step(
            self.cell_v,
            th.press_w,
            self.target_p,
            th.volume_w,
            self.cell_mass,
            th.ke_w,
            self.dof_count,
            self.dt,
        )

    def _kick(self) -> None:
        th = self.thermo
        _kernels().npt_velocity_half_step(
            th.velocities,
            th.masses,
            th.forces,
            self.cell_v,
            eta_dots=self.eta_dot,
            num_atoms=self.dof_count,
            dt=self.dt,
            mode=self._velocity_mode,
        )

    def step(self) -> None:
        self._refresh()
        self._thermostat()
        self._barostat()
        self._kick()
        self._drift()
        self.thermo.evaluate()
        self._kick()
        self._refresh()  # after the kick
        self._barostat()
        self._thermostat()

    def conserved_energy(self) -> float:
        """MTK H' = Epot + Ekin + P V + W|eps_dot|^2/2 + sum Q eta_dot^2/2
        + N_f kT eta_1 + kT sum_{k>1} eta_k."""
        kT = units.kB * self.temperature_K
        eta, eta_dot = self.eta.numpy()[0], self.eta_dot.numpy()[0]
        chain = (
            0.5 * float((self._q * eta_dot**2).sum())
            + self._nf * kT * eta[0]
            + kT * float(eta[1:].sum())
        )
        return self.thermo.total_energy + self._barostat_energy() + chain

    def _params(self) -> dict:
        return {
            **super()._params(),
            "temperature_K": self.temperature_K,
            "tdamp": self.tdamp,
            "tchain": self.tchain,
            "tloop": self.tloop,
        }

    def _state(self) -> dict:
        return {
            **super()._state(),
            "eta": self.eta.numpy(),
            "eta_dot": self.eta_dot.numpy(),
        }

    def _load_state(self, arrays: Mapping[str, np.ndarray]) -> None:
        super()._load_state(arrays)
        self.eta.assign(arrays["eta"])
        self.eta_dot.assign(arrays["eta_dot"])


class MTKNPH(_MTKBarostat):
    """NPH: the MTK barostat alone, with no thermostat. The enthalpy
    H' = Epot + Ekin + P V + W |eps_dot|^2 / 2 is conserved; the temperature
    goes where the dynamics take it.

    The step is barostat, kick | drift | kick, barostat. By default the whole
    cell fluctuates (``cell="full"``); ``cell`` takes the same modes as
    :class:`MTKNPT`.

    ``temperature_K`` sets only the barostat mass, W = 3N kT pdamp^2; it
    defaults to the atoms' temperature at the start, as LAMMPS' fix nph does.
    """

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        timestep: float,
        pressure_au: float,
        pdamp: float,
        *,
        cell: str = "full",
        temperature_K: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            evaluator,
            atoms,
            timestep,
            pressure_au,
            pdamp,
            cell=cell,
            barostat_temperature_K=temperature_K,
            **kwargs,
        )

    def _barostat(self) -> None:
        th = self.thermo
        _kernels().nph_barostat_half_step(
            self.cell_v,
            th.press_w,
            self.target_p,
            th.volume_w,
            self.cell_mass,
            th.ke_w,
            self.dof_count,
            self.dt,
        )

    def _kick(self) -> None:
        th = self.thermo
        _kernels().nph_velocity_half_step(
            th.velocities,
            th.masses,
            th.forces,
            self.cell_v,
            num_atoms=self.dof_count,
            dt=self.dt,
            mode=self._velocity_mode,
        )

    def step(self) -> None:
        self._refresh()
        self._barostat()
        self._kick()
        self._drift()
        self.thermo.evaluate()
        self._kick()
        self._refresh()  # after the kick
        self._barostat()

    def conserved_energy(self) -> float:
        """MTK H' = Epot + Ekin + P V + W |eps_dot|^2 / 2."""
        return self.thermo.total_energy + self._barostat_energy()

    def _params(self) -> dict:
        return {**super()._params(), "temperature_K": self._barostat_T}
