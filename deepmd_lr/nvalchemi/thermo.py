# SPDX-License-Identifier: LGPL-3.0-or-later
"""The simulation state an MD loop carries, and the thermodynamics it reports.

:class:`Thermo` **owns** the state -- positions, velocities, masses, forces,
cell and timestep -- and exposes each as a warp array that nvalchemiops'
integrator kernels mutate in place. It also reports, the way LAMMPS' ``thermo``
and ASE's ``MDLogger`` do. The name follows LAMMPS; read it as "the object the
thermo output is about".

nvalchemiops supplies half-step kernels and nothing else: no state container,
no logger, no potential energy, no kelvin conversion, and every scratch array
is caller-allocated. This class is that missing half. It deliberately does not
own the *ensemble* -- thermostat chains and barostat masses stay with whoever
is integrating, which is what lets one class serve NVE, NVT and NPT alike.

Units are the model's throughout -- angstrom, eV, amu, ASE time units -- so no
conversion factor appears in any loop. Two deliberate exceptions, both because
they are what a human types or reads:

* ``temperature`` is in **kelvin**, as a measurement and as the constructor's
  setpoint. The eV-valued sibling is :attr:`kT`, which is a warp array with a
  different name, type and role, so the two cannot be confused at a call site.
* :meth:`log` prints pressure in GPa, and the column is named ``press_GPa``.
  That is the only unit conversion in the class, and it happens where a human
  reads the number.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import warp as wp
from ase.units import GPa, fs, kB
from nvalchemiops.dynamics.utils import (
    compute_kinetic_energy,
    compute_temperature,
    initialize_velocities,
)

__all__ = ["Thermo"]

#: What :meth:`Thermo.log` knows how to compute, with the unit it prints in.
BUILTIN_COLUMNS = {
    "step": "",
    "time": "fs",
    "temp": "K",
    "epot": "eV",
    "ekin": "eV",
    "etot": "eV",
    "press": "GPa",
    "vol": "A^3",
}

DEFAULT_COLUMNS = ("step", "time", "temp", "epot", "ekin", "etot")


class Thermo:
    """MD state plus its thermodynamic report.

    Parameters
    ----------
    evaluator
        A :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator`. It is
        the source of everything derivable -- device, atom count, masses, cell,
        and the forces buffer -- so nothing is passed twice.
    positions
        ``(n, 3)`` in angstrom. Copied into a tensor this object owns.
    velocities
        ``(n, 3)``, or ``None`` to start at rest. Prefer
        :meth:`maxwell_boltzmann`.
    masses
        ``(n,)`` in amu, or ``None`` to take the evaluator's own table.
    timestep
        In **femtoseconds**; converted once, here.
    temperature
        Setpoint in **kelvin**, or ``None``. Builds :attr:`kT` so no caller ever
        has to construct one -- passing kelvin where a kernel wants ``k_B T``
        thermostats to ~11,600x the temperature intended, and the surest way to
        prevent that is to leave no reason to write the conversion.
    columns, every, file
        See :meth:`log`.

    Notes
    -----
    Every warp attribute is bound once here and never reassigned. Rebinding one
    would leave the kernels writing into the old buffer while every read saw
    the new one.
    """

    def __init__(
        self,
        evaluator: Any,
        positions: Any,
        *,
        velocities: Any = None,
        masses: Any = None,
        timestep: float = 0.5,
        temperature: float | None = None,
        columns: tuple[str, ...] = DEFAULT_COLUMNS,
        every: int = 1,
        file: Any = None,
    ) -> None:
        self.evaluator = evaluator
        self.device = str(evaluator.home)
        self.n_atoms = int(evaluator.n_atoms)

        unknown = [c for c in columns if c not in BUILTIN_COLUMNS]
        if unknown:
            raise ValueError(
                f"unknown thermo column(s) {unknown}; built-ins are "
                f"{sorted(BUILTIN_COLUMNS)}. Anything else is a keyword to "
                "log(), so it is computed by the caller."
            )
        if ("press" in columns or "vol" in columns) and not evaluator.compute_virial:
            # Caught here rather than at step 0, where it would surface as a
            # None halfway through a run that has already cost GPU time.
            raise ValueError(
                "columns ask for pressure or volume, which need the virial. "
                "Build the evaluator with compute_virial=True."
            )
        self.columns = tuple(columns)
        self.every = int(every)
        self.rows: list[tuple] = []
        #: Column order of `rows`, fixed by the first `log` call --
        #: built-ins followed by whatever extras that call passed.
        self.row_names: tuple[str, ...] = ()
        #: (step, wall clock) at each `log`, for the throughput properties.
        self._clock: list[tuple[int, float]] = []

        # --- state this object owns -------------------------------------
        # Allocated here, so contiguous by construction. That matters: warp's
        # view of a non-contiguous tensor is a silent COPY, and kernels would
        # then mutate a temporary while the real positions never moved.
        x = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
        if x.shape != (self.n_atoms, 3):
            raise ValueError(
                f"positions must be ({self.n_atoms}, 3), got {x.shape}"
            )
        self._x = torch.tensor(x, dtype=torch.float64, device=self.device)
        self._v = torch.zeros_like(self._x)
        if velocities is not None:
            self._v.copy_(torch.as_tensor(np.asarray(velocities, dtype=np.float64)))
        m = evaluator.masses() if masses is None else np.asarray(masses, float)
        self._m = torch.tensor(
            np.ascontiguousarray(m), dtype=torch.float64, device=self.device
        )

        self.positions = self._vec(self._x)
        self.velocities = self._vec(self._v)
        self.masses = wp.from_torch(self._m)
        # Bound ONCE to the evaluator's preallocated buffer. Valid for the life
        # of this object because `__call__` copies into it rather than
        # rebinding -- an invariant guarded in `evaluate` and pinned by a test.
        self.forces = self._vec(evaluator.forces)
        self._forces_ptr = evaluator.forces.data_ptr()

        self.cell = wp.array(
            np.asarray(evaluator.cell, dtype=np.float64)[None],
            dtype=wp.mat33d,
            device=self.device,
        )
        self.dt = self.scalar(timestep * fs)
        self.timestep = float(timestep)
        self.kT = None if temperature is None else self.scalar(kB * temperature)
        self.natoms = self.scalar(self.n_atoms, dtype=wp.int32)

        # --- scratch, allocated once ------------------------------------
        self._ke_w = self.zeros(1)
        self._temp_w = self.zeros(1)
        #: Volume as a device array, for kernels that take one.
        self.volume_w = self.scalar(evaluator.volume)
        self._momentum = self.zeros(1, dtype=wp.vec3d)
        self._total_mass = self.zeros(1)
        self._com = self.zeros(1, dtype=wp.vec3d)
        #: Pressure tensor as a device array (vec9d), built on first use
        #: by `update_pressure` -- see the lazy import there.
        self.press_w = None
        self._virial_w = None
        self._kin_w = None

        self.energy = float("nan")
        self._symbols = [
            evaluator.model.get_type_map()[i] for i in np.asarray(evaluator.atype)
        ]
        self._file = open(file, "w") if isinstance(file, (str, bytes)) else file
        self._header_written = False
        if temperature is not None and velocities is None:
            self.maxwell_boltzmann(temperature)

    # --- allocation helpers, bound to this object's device ----------------

    def scalar(self, value: float, dtype: Any = wp.float64) -> wp.array:
        """A length-1 warp array on this simulation's device."""
        return wp.array(np.full(1, value), dtype=dtype, device=self.device)

    def zeros(self, shape: Any = 1, dtype: Any = wp.float64) -> wp.array:
        """A zeroed warp array on this simulation's device.

        Thermostat and barostat state belongs to the caller, not here -- this
        is what a script uses to allocate its own chain variables without
        having to name the device every time.
        """
        return wp.zeros(shape, dtype=dtype, device=self.device)

    def _vec(self, t: torch.Tensor) -> wp.array:
        """A warp ``vec3d`` view of an ``(n, 3)`` float64 tensor.

        Refuses a non-contiguous tensor rather than quietly calling
        ``.contiguous()``: that returns a *copy*, and every kernel write would
        then land somewhere the caller never reads.
        """
        if not t.is_contiguous():
            raise ValueError("expected a contiguous (n, 3) tensor")
        return wp.from_torch(t, dtype=wp.vec3d)

    # --- the force call ----------------------------------------------------

    def evaluate(self) -> float:
        """Energy and forces at the current positions. Returns the energy."""
        self.energy, forces = self.evaluator(self._x)
        if forces.data_ptr() != self._forces_ptr:
            raise RuntimeError(
                "the evaluator reallocated its forces buffer, so this object's "
                "warp view is stale and every kernel would read frozen forces. "
                "DistributedEvaluator.__call__ must write into the existing "
                "tensor, not rebind it."
            )
        return self.energy

    def set_cell(self, cell: Any = None) -> None:
        """Push the cell to the evaluator and refresh the volume.

        Call this after a barostat has moved :attr:`cell`. It is the one seam
        where the device-side cell, the evaluator's host copy and the cached
        partitioning are brought back into agreement, in that order -- doing
        them in three places is how they drift apart.
        """
        if cell is not None:
            self.cell.assign(np.asarray(cell, dtype=np.float64).reshape(1, 3, 3))
        self.evaluator.set_cell(self.cell.numpy()[0])
        self.volume_w.fill_(self.evaluator.volume)

    # --- measurements ------------------------------------------------------

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy in eV."""
        compute_kinetic_energy(self.velocities, self.masses, self._ke_w)
        return float(self._ke_w.numpy()[0])

    @property
    def temperature(self) -> float:
        """Instantaneous temperature in kelvin.

        nvalchemiops divides by ``3N - 3``, i.e. it assumes the centre-of-mass
        motion was removed -- which :meth:`maxwell_boltzmann` does. Set
        velocities by hand and the report is biased by ``3/(3N-3)``.
        """
        compute_kinetic_energy(self.velocities, self.masses, self._ke_w)
        compute_temperature(self._ke_w, self._temp_w, self.natoms)
        return float(self._temp_w.numpy()[0]) / kB

    @property
    def potential_energy(self) -> float:
        """Potential energy in eV, from the last :meth:`evaluate`."""
        return self.energy

    @property
    def total_energy(self) -> float:
        """Potential plus kinetic, in eV. Conserved under NVE and nothing else."""
        return self.energy + self.kinetic_energy

    @property
    def volume(self) -> float:
        """Cell volume in angstrom^3."""
        return self.evaluator.volume

    @property
    def pressure(self) -> float:
        """Isotropic pressure in eV/angstrom^3.

        Native units, matching the evaluator's ``stress`` and what the barostat
        kernels consume -- a GPa-valued property would force a conversion back
        before feeding one, in the direction where an error is silent.
        """
        self.update_pressure()
        p = self.press_w.numpy()[0]
        return float(p[0] + p[4] + p[8]) / 3.0

    def update_pressure(self) -> None:
        """Refresh the pressure tensor from the current virial and velocities.

        Owns the mixed layouts nvalchemiops wants here -- flat ``vec9`` for the
        virial and pressure, ``mat33`` for the cell, and a 2-D ``(1, 9)``
        kinetic tensor -- so no caller has to know them.
        """
        # Imported lazily: `vec9d` lives in the one nvalchemiops module with a
        # packaging bug (`flat_virial_to_vec9` is exported but undefined), and
        # an upstream break should not take out `import deepmd_lr.nvalchemi`.
        from nvalchemiops.dynamics.integrators import (  # noqa: PLC0415
            compute_pressure_tensor,
            vec9d,
        )

        if self.evaluator.virial is None:
            raise ValueError(
                "no virial available; build the evaluator with "
                "compute_virial=True and call evaluate() first."
            )
        if self.press_w is None:
            self._virial_w = wp.zeros(1, dtype=vec9d, device=self.device)
            self.press_w = wp.zeros(1, dtype=vec9d, device=self.device)
            self._kin_w = wp.zeros((1, 9), dtype=wp.float64, device=self.device)
        self._virial_w.assign(
            np.asarray(self.evaluator.virial, dtype=np.float64).reshape(1, 9)
        )
        compute_pressure_tensor(
            self.velocities, self.masses, self._virial_w, self.cell,
            self._kin_w, self.press_w, self.volume_w,
        )

    def update_kinetic_energy(self) -> None:
        """Refresh :attr:`ke_w` on the device, with no copy to the host.

        For kernels that want the kinetic energy as an array -- the NPT
        thermostat and barostat both do -- so a loop need not round-trip a
        number to the host and straight back.
        """
        compute_kinetic_energy(self.velocities, self.masses, self._ke_w)

    @property
    def ke_w(self) -> wp.array:
        """Kinetic energy as a device array, for kernels that take one."""
        return self._ke_w

    # --- throughput --------------------------------------------------------

    @property
    def seconds_per_step(self) -> float:
        """Wall-clock seconds per MD step, over the logged history.

        The first logged interval is **dropped** when there is more than one,
        because it carries kernel compilation -- on this machine that is
        ~1 s against a ~30 ms step, and a least-squares fit over all points
        still comes out 2x high (measured). Excluding it is what the
        benchmarks in `experiments/` do, and it is the only thing that
        actually recovers the steady-state rate. The remaining points are then
        fitted by least squares.

        This measures the loop as it actually ran, `log` included. Logging
        every step means a device synchronisation every step, and that cost is
        real and counted here; a production run logging every 100th step is
        faster than this number suggests.
        """
        clock = self._clock[1:] if len(self._clock) > 2 else self._clock
        if len(clock) < 2:
            return float("nan")
        steps = np.array([c[0] for c in clock], dtype=float)
        wall = np.array([c[1] for c in clock], dtype=float)
        if steps.max() == steps.min():
            return float("nan")
        return float(np.polyfit(steps, wall, 1)[0])

    @property
    def ms_per_step(self) -> float:
        """Wall-clock milliseconds per MD step."""
        return self.seconds_per_step * 1000.0

    @property
    def ns_per_day(self) -> float:
        """Simulated nanoseconds per day of wall clock.

        Scales with the timestep, so it is a statement about this run's `dt`
        and not about the model alone.
        """
        seconds = self.seconds_per_step
        if not np.isfinite(seconds) or seconds <= 0:
            return float("nan")
        return self.timestep * 1e-6 * 86400.0 / seconds

    def performance(self) -> str:
        """One line of throughput, for the end of a run."""
        return (
            f"{self.n_atoms} atoms, dt {self.timestep} fs: "
            f"{self.ms_per_step:.1f} ms/step -> {self.ns_per_day:.3f} ns/day"
        )

    # --- setup -------------------------------------------------------------

    def maxwell_boltzmann(self, temperature: float | None = None,
                          seed: int = 0) -> None:
        """Draw velocities from Maxwell-Boltzmann, in kelvin, COM removed.

        Through nvalchemiops, so the draw happens on the device. Like ASE's
        equivalent it does not rescale to hit the target exactly -- the
        instantaneous temperature fluctuates about it.
        """
        if temperature is None and self.kT is None:
            raise ValueError("no temperature given, and none set at construction")
        kt = self.kT if temperature is None else self.scalar(kB * temperature)
        initialize_velocities(
            self.velocities, self.masses, kt,
            self._momentum, self._total_mass, self._com,
            random_seed=seed, remove_com=True,
        )

    def atoms(self) -> Any:
        """A snapshot as an ASE ``Atoms`` -- the way state leaves this object.

        One escape hatch rather than a parallel host API: an ``Atoms`` carries
        positions, cell, symbols and velocities together, which is what writing
        a trajectory frame actually needs. ASE's velocity units are these
        units, so nothing is converted.
        """
        from ase import Atoms  # noqa: PLC0415

        return Atoms(
            symbols=self._symbols,
            positions=self._x.detach().cpu().numpy(),
            cell=np.asarray(self.evaluator.cell),
            velocities=self._v.detach().cpu().numpy(),
            pbc=True,
        )

    # --- reporting ---------------------------------------------------------

    def _value(self, name: str, step: int) -> float:
        if name == "step":
            return step
        if name == "time":
            return step * self.timestep
        if name == "temp":
            return self.temperature
        if name == "epot":
            return self.energy
        if name == "ekin":
            return self.kinetic_energy
        if name == "etot":
            return self.total_energy
        if name == "press":
            return self.pressure / GPa
        if name == "vol":
            return self.volume
        raise KeyError(name)               # unreachable: validated in __init__

    def log(self, step: int, **extra: float) -> dict:
        """Record one row, print it every ``every`` steps, and return it.

        Built-in column names resolve first; anything else must arrive as a
        keyword, which is the line between what this object has the state to
        compute and what it does not. A Nose-Hoover invariant or a field
        alignment is the caller's to work out and pass in.

        Note this synchronises with the device to read the measurements. On a
        large system at ``every=1`` that serialises the pipeline -- log as
        rarely as the analysis allows.
        """
        names = (*self.columns, *extra)
        if not self.row_names:
            self.row_names = names
        elif names != self.row_names:
            raise ValueError(
                f"log() called with columns {names}, but the first call fixed "
                f"{self.row_names}. The rows are a table; keep them aligned."
            )
        if not self._header_written:
            def _head(n: str) -> str:
                unit = BUILTIN_COLUMNS.get(n)
                return f"{n + (f' [{unit}]' if unit else ''):>15s}"

            head = "".join(_head(n) for n in names)
            self._emit(head)
            self._header_written = True

        self._clock.append((step, time.perf_counter()))
        row = {n: self._value(n, step) for n in self.columns}
        row.update(extra)
        self.rows.append(tuple(row.values()))
        if step % self.every == 0:
            cells = "".join(
                f"{v:15d}" if isinstance(v, int) else f"{v:15.5f}"
                for v in row.values()
            )
            self._emit(cells)
        return row

    def _emit(self, line: str) -> None:
        print(line, flush=True)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()

    def drift(self, column: str = "etot", per_atom: bool = True) -> float:
        """Least-squares drift in a logged column, in meV/atom/ps.

        Works on anything logged, including a keyword extra -- a Nose-Hoover
        invariant is exactly the quantity worth fitting, and it is the
        caller's to compute.

        The reason this is here rather than in a caller: the unit chain --
        eV and fs in, meV per atom per ps out -- is easy to get wrong, and it
        is the headline number of an NVE conservation check.
        """
        names = self.row_names or self.columns
        if column not in names:
            raise KeyError(f"{column!r} is not a logged column: {names}")
        if "time" not in names:
            raise KeyError("drift needs a 'time' column")
        i, j = names.index(column), names.index("time")
        t = np.array([r[j] for r in self.rows]) / 1000.0        # ps
        y = np.array([r[i] for r in self.rows]) * 1000.0        # meV
        if per_atom:
            y = y / self.n_atoms
        if len(t) < 2:
            return float("nan")
        return float(np.polyfit(t, y, 1)[0])
