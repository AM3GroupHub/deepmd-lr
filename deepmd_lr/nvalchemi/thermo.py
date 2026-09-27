# SPDX-License-Identifier: LGPL-3.0-or-later
"""The state an MD run carries, and what can be measured from it.

:class:`Thermo` **owns** the state -- positions, velocities, masses, the forces
view and the cell -- and exposes each as a warp array that nvalchemiops'
integrator kernels mutate in place. It also measures: kinetic, potential and
total energy, temperature, volume and pressure. The name follows LAMMPS; read
it as "the object the thermo output is about".

It deliberately owns neither **time** nor the **ensemble**. The timestep, the
target temperature, thermostat and barostat parameters, the step count and the
log all belong to the integrator (:mod:`deepmd_lr.nvalchemi.md`), which builds
a Thermo in its own ``__init__`` around the evaluator it is given. A custom
kernel loop makes those scalars itself, e.g. ``dt = th.scalar(0.5 * units.fs)``
and ``kT = th.scalar(units.kB * 300.0)``.

The evaluator is **shared, never copied**: :attr:`evaluator` is the object the
caller built, so a relaxation, an NVT stage and an NPT stage can run one after
another on one set of device replicas, each with a Thermo of its own. Two
checks make that safe:

- :attr:`fresh` is False once another object has evaluated since this one
  did. All of them view the same forces buffer.
- :attr:`cell_is_current` is False once the evaluator's cell has moved on,
  as an NPT stage leaves it.

Units are the model's throughout -- angstrom, eV, amu, ASE time units. The one
exception is :attr:`temperature`, in kelvin, because that is what a human reads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp
from ase.units import kB
from nvalchemiops.dynamics.utils import (
    compute_kinetic_energy,
    compute_temperature,
    remove_com_motion,
)

__all__ = ["Thermo"]

#: Keywords the constructor used to take, and where each went.
_MOVED = {
    "timestep": "the integrator's `timestep` (deepmd_lr.nvalchemi.md)",
    "temperature": "the integrator's `temperature_K`, or ASE's thermalize_momenta()",
    "columns": "the integrator's `logfile`/`extra_columns`",
    "every": "the integrator's `loginterval`",
    "file": "the integrator's `logfile`",
}


class Thermo:
    """MD state and its measurements.

    Parameters
    ----------
    evaluator
        A :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator`, shared
        as it is -- never copied or wrapped. It is the source of everything
        derivable: device, atom count, default masses, cell, and the forces
        buffer.
    positions
        ``(n, 3)`` in angstrom, wrapped or not. Copied into a tensor this
        object owns.
    velocities
        ``(n, 3)`` in angstrom per ASE time unit (ASE's own convention), or
        ``None`` to start at rest.
    masses
        ``(n,)`` in amu, or ``None`` to take the evaluator's own table. The MD
        classes pass ``atoms.get_masses()``, so isotopes work.

    Notes
    -----
    Every warp attribute is bound once here and never reassigned. Rebinding one
    would leave the kernels writing into the old buffer while every read saw
    the new one. :meth:`set_state` copies new values into the bound buffers.
    """

    def __init__(
        self,
        evaluator: Any,
        positions: Any,
        *,
        velocities: Any = None,
        masses: Any = None,
        **moved: Any,
    ) -> None:
        if moved:
            where = "; ".join(
                f"{k} -> {_MOVED.get(k, 'nowhere')}" for k in sorted(moved)
            )
            raise TypeError(
                f"Thermo no longer takes {sorted(moved)}. The timestep, the "
                "target temperature and the log belong to the integrator, "
                f"which builds its own Thermo: {where}. In a custom kernel "
                "loop, make them yourself: dt = th.scalar(timestep), "
                "kT = th.scalar(kB * T)."
            )
        self.evaluator = evaluator
        self.device = str(evaluator.home)
        self.n_atoms = int(evaluator.n_atoms)

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
        #: The evaluator's call count after this object's last evaluation.
        self._seen_calls: int | None = None

        self.cell = wp.array(
            np.asarray(evaluator.cell, dtype=np.float64)[None],
            dtype=wp.mat33d,
            device=self.device,
        )
        #: Host copy of the cell this object last agreed on with the evaluator.
        self._cell_host = np.array(evaluator.cell, dtype=np.float64)
        self.natoms = self.scalar(self.n_atoms, dtype=wp.int32)

        # --- scratch, allocated once ------------------------------------
        #: Kinetic energy as a kernel input, refreshed by update_kinetic_energy.
        self._ke_w = self.zeros(1)
        #: The getters' own kinetic energy. Kernels consume `_ke_w`, and a
        #: getter writing it would let a log change the next step.
        self._ke_probe = self.zeros(1)
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
        self._press_probe = None
        self._kin_probe = None

        self.energy = float("nan")
        self._symbols = [
            evaluator.model.get_type_map()[i] for i in np.asarray(evaluator.atype)
        ]

    # --- allocation helpers, bound to this object's device ----------------

    def scalar(self, value: float, dtype: Any = wp.float64) -> wp.array:
        """A length-1 warp array on this simulation's device."""
        return wp.array(np.full(1, value), dtype=dtype, device=self.device)

    def zeros(self, shape: Any = 1, dtype: Any = wp.float64) -> wp.array:
        """A zeroed warp array on this simulation's device.

        Thermostat and barostat state belongs to the integrator, not here --
        this is how it allocates its chain variables without naming the device
        every time.
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
        self._seen_calls = self.evaluator.calls
        return self.energy

    @property
    def fresh(self) -> bool:
        """Whether :attr:`forces` still hold this object's own evaluation.

        False before the first :meth:`evaluate`, and after any other object
        sharing the evaluator has evaluated since -- the buffer is shared.
        """
        return self._seen_calls == self.evaluator.calls

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
        self._cell_host = np.array(self.evaluator.cell, dtype=np.float64)

    @property
    def cell_is_current(self) -> bool:
        """Whether the evaluator still has the cell this object last set.

        An NPT stage on the shared evaluator moves its cell; a Thermo built
        before that describes a box the evaluator no longer partitions.
        """
        return bool(np.array_equal(self._cell_host, self.evaluator.cell))

    def set_state(self, positions: Any = None, velocities: Any = None) -> None:
        """Copy new positions and/or velocities into the bound buffers.

        Never rebinds, so every warp view stays live. Used when ``atoms`` was
        edited between runs or by an observer.
        """
        for value, tensor in ((positions, self._x), (velocities, self._v)):
            if value is None:
                continue
            new = np.asarray(value, dtype=np.float64)
            if new.shape != (self.n_atoms, 3):
                raise ValueError(f"expected ({self.n_atoms}, 3), got {new.shape}")
            tensor.copy_(torch.as_tensor(new, device=self.device))

    # --- measurements ------------------------------------------------------
    # Every getter measures into scratch of its own and never writes a kernel
    # input (`ke_w`, `press_w`): reading one for a log row must not change the
    # next step. `update_kinetic_energy` and `update_pressure` are the writers.

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy in eV."""
        compute_kinetic_energy(self.velocities, self.masses, self._ke_probe)
        return float(self._ke_probe.numpy()[0])

    @property
    def temperature(self) -> float:
        """Instantaneous temperature in kelvin.

        nvalchemiops divides by ``3N - 3``, i.e. it assumes the centre-of-mass
        motion was removed -- which :meth:`remove_com_motion` and the MD
        classes do.
        """
        compute_kinetic_energy(self.velocities, self.masses, self._ke_probe)
        compute_temperature(self._ke_probe, self._temp_w, self.natoms)
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

        From the current velocities and the virial of the last evaluation,
        computed into scratch of its own, never into :attr:`press_w`. The
        barostat consumes ``press_w``, and when this getter refreshed it, a
        log row changed the pressure the next barostat half-step saw.
        """
        self._pressure_into(probe=True)
        p = self._press_probe.numpy()[0]
        return float(p[0] + p[4] + p[8]) / 3.0

    def update_pressure(self) -> None:
        """Refresh :attr:`press_w`, the barostat's input, from the current
        virial and velocities.

        Owns the mixed layouts nvalchemiops wants here -- flat ``vec9`` for the
        virial and pressure, ``mat33`` for the cell, and a 2-D ``(1, 9)``
        kinetic tensor -- so no caller has to know them.
        """
        self._pressure_into(probe=False)

    def _pressure_into(self, probe: bool) -> None:
        """The pressure tensor into the getter's scratch or into ``press_w``."""
        # Imported lazily: `vec9d` lives in the one nvalchemiops module with a
        # packaging bug (`flat_virial_to_vec9` is exported but undefined), and
        # an upstream break should not take out `import deepmd_lr.nvalchemi`.
        from nvalchemiops.dynamics.integrators import (
            compute_pressure_tensor,
            vec9d,
        )

        if self.evaluator.virial is None:
            raise ValueError(
                "no virial available; build the evaluator with "
                "compute_virial=True (or call set_compute_virial(True)) and "
                "evaluate first."
            )
        if self.press_w is None:
            self._virial_w = wp.zeros(1, dtype=vec9d, device=self.device)
            self.press_w = wp.zeros(1, dtype=vec9d, device=self.device)
            self._kin_w = wp.zeros((1, 9), dtype=wp.float64, device=self.device)
            self._press_probe = wp.zeros(1, dtype=vec9d, device=self.device)
            self._kin_probe = wp.zeros((1, 9), dtype=wp.float64, device=self.device)
        # Shared by both paths: it only ever holds the evaluator's virial, so
        # either path rewrites it with the same numbers.
        self._virial_w.assign(
            np.asarray(self.evaluator.virial, dtype=np.float64).reshape(1, 9)
        )
        kinetic, pressure = ((self._kin_probe, self._press_probe) if probe
                             else (self._kin_w, self.press_w))
        compute_pressure_tensor(
            self.velocities, self.masses, self._virial_w, self.cell,
            kinetic, pressure, self.volume_w,
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

    # --- setup -------------------------------------------------------------

    def remove_com_motion(self) -> None:
        """Subtract the centre-of-mass velocity, in place on the device.

        Every temperature and thermostat here counts ``3N - 3`` degrees of
        freedom, which is only right with the centre of mass at rest.
        """
        remove_com_motion(self.velocities, self.masses, self._momentum,
                          self._total_mass, self._com)

    def atoms(self) -> Any:
        """A snapshot as an ASE ``Atoms`` -- the way state leaves this object.

        Positions (unwrapped, as integrated), cell, masses and velocities.
        ASE's velocity units are these units, so nothing is converted.
        """
        from ase import Atoms

        return Atoms(
            symbols=self._symbols,
            positions=self._x.detach().cpu().numpy(),
            cell=np.asarray(self.evaluator.cell),
            masses=self._m.detach().cpu().numpy(),
            velocities=self._v.detach().cpu().numpy(),
            pbc=True,
        )
