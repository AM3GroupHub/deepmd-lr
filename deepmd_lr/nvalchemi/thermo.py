# SPDX-License-Identifier: MIT
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

**Fixed atoms** come from ASE's ``FixAtoms`` on the ``Atoms`` given to
:meth:`from_atoms`, with no argument of their own. They stay where they
started, at rest. :meth:`evaluate` puts them back before every force call, so
a kernel that moves every atom -- Langevin's noise does -- cannot carry them
off, and :attr:`forces` gives them zero force, so no kick moves them either.
They carry no kinetic energy and no degrees of freedom: :attr:`ndof` counts
three per mobile atom, as ASE's ``Atoms.get_temperature`` does.

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
    remove_com_motion,
)

from deepmd_lr.cell import det_3x3

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

    :meth:`from_atoms` builds one from an ``Atoms`` instead, its ``FixAtoms``
    included; that is how the MD classes build theirs.

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
        _fixed: Any = None,
        **moved: Any,
    ) -> None:
        if moved:
            _refuse_moved(moved)
        self.evaluator = evaluator
        self.device = str(evaluator.home)
        self.n_atoms = int(evaluator.n_atoms)
        self._own_state(positions, velocities, masses)
        self._hold_fixed(_fixed)
        self._bind_views()
        self._allocate_scratch()
        self.energy = float("nan")
        self._symbols = [
            evaluator.model.get_type_map()[i] for i in np.asarray(evaluator.atype)
        ]

    def _own_state(self, positions: Any, velocities: Any, masses: Any) -> None:
        # Allocated here, so contiguous by construction. That matters: warp's
        # view of a non-contiguous tensor is a silent COPY, and kernels would
        # then mutate a temporary while the real positions never moved.
        x = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
        if x.shape != (self.n_atoms, 3):
            raise ValueError(f"positions must be ({self.n_atoms}, 3), got {x.shape}")
        self._x = torch.tensor(x, dtype=torch.float64, device=self.device)
        self._v = torch.zeros_like(self._x)
        if velocities is not None:
            self._v.copy_(torch.as_tensor(np.asarray(velocities, dtype=np.float64)))
        m = self.evaluator.masses() if masses is None else np.asarray(masses, float)
        self._m = torch.tensor(
            np.ascontiguousarray(m), dtype=torch.float64, device=self.device
        )

    def _hold_fixed(self, fixed: Any) -> None:
        #: Indices of the atoms held in place, sorted; empty when none are.
        self.fixed = np.asarray([] if fixed is None else fixed, dtype=np.int64)
        #: The kinetic degrees of freedom every temperature here divides by.
        #: With the centre of mass at rest, 3N - 3. Fixed atoms hold the rest
        #: in place instead, so the mobile atoms' momentum is not conserved
        #: and each keeps all three.
        self.ndof = (
            3 * (self.n_atoms - len(self.fixed))
            if len(self.fixed)
            else 3 * self.n_atoms - 3
        )
        self._fixed_idx = None
        if len(self.fixed):
            self._fixed_idx = torch.as_tensor(self.fixed, device=self.device)
            #: Where the fixed atoms are held.
            self._anchor = self._x.index_select(0, self._fixed_idx)
            self._v.index_fill_(0, self._fixed_idx, 0.0)

    def _bind_views(self) -> None:
        evaluator = self.evaluator
        self.positions = self._vec(self._x)
        self.velocities = self._vec(self._v)
        self.masses = wp.from_torch(self._m)
        # Bound ONCE to the evaluator's preallocated buffer. Valid for the life
        # of this object because `__call__` copies into it rather than
        # rebinding -- an invariant guarded in `evaluate` and pinned by a test.
        # With fixed atoms the kernels read a buffer of this object's own
        # instead, the evaluator's with their rows zeroed; the evaluator's
        # stays the model's, which is what output records.
        if self._fixed_idx is None:
            self.forces = self._vec(evaluator.forces)
        else:
            self._f = torch.zeros_like(self._x)
            self.forces = self._vec(self._f)
        self._forces_ptr = evaluator.forces.data_ptr()
        #: The evaluator's call count after this object's last evaluation.
        self._seen_calls: int | None = None
        self.cell = wp.array(
            np.asarray(evaluator.cell, dtype=np.float64)[None],
            dtype=wp.mat33d,
            device=self.device,
        )
        #: The evaluator's cell version this object last set or agreed on.
        self._cell_version_seen = evaluator.cell_version
        self.natoms = self.scalar(self.n_atoms, dtype=wp.int32)

    def _allocate_scratch(self) -> None:
        #: Kinetic energy as a kernel input, refreshed by update_kinetic_energy.
        self._ke_w = self.zeros(1)
        #: The getters' own kinetic energy. Kernels consume `_ke_w`, and a
        #: getter writing it would let a log change the next step.
        self._ke_probe = self.zeros(1)
        #: Volume as a device array, for kernels that take one.
        self.volume_w = self.scalar(self.evaluator.volume)
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

    @classmethod
    def from_atoms(cls, evaluator: Any, atoms: Any) -> Thermo:
        """A Thermo for ``atoms``: its positions, velocities, masses, and the
        atoms its ``FixAtoms`` holds. The MD classes build theirs this way.

        ``FixAtoms`` is the one ASE constraint honoured; any other is refused,
        since the kernels would ignore it silently.
        """
        return cls(
            evaluator,
            atoms.get_positions(),
            velocities=atoms.get_velocities(),
            masses=atoms.get_masses(),
            _fixed=_fixed_atoms(atoms),
        )

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
        """Energy and forces at the current positions. Returns the energy.

        Fixed atoms are first put back where they are held, at rest, and
        leave with zero force in :attr:`forces`.
        """
        if self._fixed_idx is not None:
            self._x.index_copy_(0, self._fixed_idx, self._anchor)
            self._v.index_fill_(0, self._fixed_idx, 0.0)
        self.energy, forces = self.evaluator(self._x)
        if forces.data_ptr() != self._forces_ptr:
            raise RuntimeError(
                "the evaluator reallocated its forces buffer, so this object's "
                "warp view is stale and every kernel would read frozen forces. "
                "DistributedEvaluator.__call__ must write into the existing "
                "tensor, not rebind it."
            )
        if self._fixed_idx is not None:
            self._f.copy_(forces)
            self._f.index_fill_(0, self._fixed_idx, 0.0)
        self._seen_calls = self.evaluator.calls
        return self.energy

    def get_forces(self, apply_constraint: bool = True) -> np.ndarray:
        """The last evaluation's forces on the host, ``(n, 3)`` in eV/A.

        As ASE's ``Atoms.get_forces``: with ``apply_constraint`` the fixed
        atoms' rows are zero, as the kernels read them in :attr:`forces`.
        Without it they hold the force the model puts on those atoms, which
        is what output records; ASE zeroes them again on reading. Valid until
        another object evaluates on the shared evaluator.
        """
        if apply_constraint or self._fixed_idx is None:
            return self.forces.numpy().copy()
        return self.evaluator.forces.detach().cpu().numpy().copy()

    @property
    def fresh(self) -> bool:
        """Whether :attr:`forces` still hold this object's own evaluation.

        False before the first :meth:`evaluate`, after the evaluator discarded
        its results (``clear_results``, as a failed relaxation does), and after
        any other object sharing the evaluator has evaluated since -- the
        buffer is shared.
        """
        return self._seen_calls == self.evaluator.calls and bool(self.evaluator.results)

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
        self._cell_version_seen = self.evaluator.cell_version

    def set_cell_device(self, cell: torch.Tensor) -> None:
        """:meth:`set_cell` from a device tensor, reading nothing back.

        The barostats' seam: ``cell`` is the ``(3, 3)`` cell they integrate,
        lattice vectors as rows, on this device. It is copied into
        :attr:`cell`, handed to the evaluator (:meth:`DistributedEvaluator.
        set_cell_device`) and its volume computed into :attr:`volume_w`, all
        on the device, with no host round trip per step.
        """
        h = cell.detach().to(device=self.device, dtype=torch.float64).reshape(3, 3)
        wp.to_torch(self.cell).reshape(3, 3).copy_(h)
        self.evaluator.set_cell_device(h)
        wp.to_torch(self.volume_w).reshape(1).copy_(det_3x3(h).abs().reshape(1))
        self._cell_version_seen = self.evaluator.cell_version

    @property
    def cell_is_current(self) -> bool:
        """Whether the evaluator still has the cell this object last set.

        An NPT stage on the shared evaluator moves its cell; a Thermo built
        before that describes a box the evaluator no longer partitions. A
        version counter, not a comparison of host copies: under a barostat
        the host copy is refreshed only on demand.
        """
        return self._cell_version_seen == self.evaluator.cell_version

    def set_state(self, positions: Any = None, velocities: Any = None) -> None:
        """Copy new positions and/or velocities into the bound buffers.

        Never rebinds, so every warp view stays live. Used when ``atoms`` was
        edited between runs or by an observer. New positions also move where
        the fixed atoms are held; their velocities stay zero.
        """
        for value, tensor in ((positions, self._x), (velocities, self._v)):
            if value is None:
                continue
            new = np.asarray(value, dtype=np.float64)
            if new.shape != (self.n_atoms, 3):
                raise ValueError(f"expected ({self.n_atoms}, 3), got {new.shape}")
            tensor.copy_(torch.as_tensor(new, device=self.device))
        if self._fixed_idx is not None:
            if positions is not None:
                self._anchor.copy_(self._x.index_select(0, self._fixed_idx))
            self._v.index_fill_(0, self._fixed_idx, 0.0)

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
        """Instantaneous temperature in kelvin, ``2 KE / (ndof kB)``.

        :attr:`ndof` is ``3N - 3``, which assumes the centre-of-mass motion
        was removed -- :meth:`remove_com_motion` and the MD classes do that --
        or, with fixed atoms, three per mobile atom.
        """
        if self.ndof < 1:
            return 0.0
        return 2.0 * self.kinetic_energy / self.ndof / kB

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
        computed into scratch of its own, never into :attr:`press_w`: the
        barostat consumes ``press_w``, and a log row must not change the
        pressure the next barostat half-step sees.
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

        virial = getattr(self.evaluator, "virial_dev", None)
        if virial is None:
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
        # either path rewrites it with the same numbers. Device to device: a
        # re-upload of the host copy is a pageable copy that would stall every
        # barostat step.
        wp.to_torch(self._virial_w).reshape(9).copy_(
            virial.detach().reshape(9).to(torch.float64)
        )
        kinetic, pressure = (
            (self._kin_probe, self._press_probe)
            if probe
            else (self._kin_w, self.press_w)
        )
        compute_pressure_tensor(
            self.velocities,
            self.masses,
            self._virial_w,
            self.cell,
            kinetic,
            pressure,
            self.volume_w,
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
        freedom, which is only right with the centre of mass at rest. With
        fixed atoms it is refused: they hold the system in place, and
        subtracting a velocity from every atom would set them moving.
        """
        if self._fixed_idx is not None:
            raise ValueError(
                "fixed atoms hold the system in place, so there is no "
                "centre-of-mass motion to remove; removing it would set them "
                "moving"
            )
        remove_com_motion(
            self.velocities, self.masses, self._momentum, self._total_mass, self._com
        )

    def atoms(self) -> Any:
        """A snapshot as an ASE ``Atoms`` -- the way state leaves this object.

        Positions (unwrapped, as integrated), cell, the evaluator's ``pbc``,
        masses, velocities, and the fixed atoms as a ``FixAtoms``. ASE's
        velocity units are these units, so nothing is converted.
        """
        from ase import Atoms
        from ase.constraints import FixAtoms

        atoms = Atoms(
            symbols=self._symbols,
            positions=self._x.detach().cpu().numpy(),
            cell=np.asarray(self.evaluator.cell),
            masses=self._m.detach().cpu().numpy(),
            velocities=self._v.detach().cpu().numpy(),
            pbc=getattr(self.evaluator, "pbc", True),
        )
        if len(self.fixed):
            atoms.set_constraint(FixAtoms(indices=self.fixed))
        return atoms


def _refuse_moved(moved: dict) -> None:
    where = "; ".join(f"{k} -> {_MOVED.get(k, 'nowhere')}" for k in sorted(moved))
    raise TypeError(
        f"Thermo no longer takes {sorted(moved)}. The timestep, the "
        "target temperature and the log belong to the integrator, "
        f"which builds its own Thermo: {where}. In a custom kernel "
        "loop, make them yourself: dt = th.scalar(timestep), "
        "kT = th.scalar(kB * T)."
    )


def _fixed_atoms(atoms: Any) -> np.ndarray:
    """The atoms ``atoms``' ``FixAtoms`` constraints hold, as sorted indices.

    ``FixAtoms`` is the one ASE constraint the GPU classes honour, and any
    other is refused: the kernels would ignore it without a word.
    """
    from ase.constraints import FixAtoms

    other = sorted(
        {type(c).__name__ for c in atoms.constraints if not isinstance(c, FixAtoms)}
    )
    if other:
        raise ValueError(
            f"atoms carries {', '.join(other)}: of ASE's constraints the GPU "
            "classes honour FixAtoms alone, and the kernels would ignore the "
            "rest silently. Remove them (atoms.set_constraint()), or run "
            "through the ASE calculator, deepmd_lr.ase.LongRangeCalculator"
        )
    held = [np.asarray(c.get_indices(), dtype=np.int64) for c in atoms.constraints]
    if not held:
        return np.zeros(0, dtype=np.int64)
    # Numpy's indexing, as ASE's own FixAtoms applies it: a negative index
    # counts from the end, and one past the last atom is refused.
    idx = np.unique(np.arange(len(atoms))[np.concatenate(held)])
    if len(idx) == len(atoms):
        raise ValueError("every atom is fixed, so nothing would move")
    return idx
