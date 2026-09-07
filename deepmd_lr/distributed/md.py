# SPDX-License-Identifier: LGPL-3.0-or-later
"""Distributed molecular dynamics on nvalchemiops' integrators.

The model is evaluated over spatial partitions -- one per GPU -- and the
trajectory is advanced by the Warp integrator kernels in
``nvalchemiops.dynamics.integrators``. There is no ASE calculator in this
path: ASE's calculator interface hands back energy and forces for one
configuration at a time, which forces the trajectory state through NumPy on
the host every step, and ASE's integrators cannot use the GPU state the
partitioned evaluation already has.

nvalchemiops provides half-step kernels rather than a driver, so the loop is
composed here:

    velocity_verlet_position_update(...)   # r(t+dt), v(t+dt/2), in place
    forces = evaluate_partitioned(...)     # F at the new positions
    velocity_verlet_velocity_finalize(...) # v(t+dt)

The kernels take Warp arrays, which ``wp.from_torch`` provides as zero-copy
views of the torch tensors -- the kernels mutate the tensors in place, so
there is no per-step conversion of the trajectory state.

Units
-----
ASE's self-consistent set: angstrom, eV, amu, and time in
``A * sqrt(amu/eV)`` (``1 fs = 0.0982269...`` of it). Only two constants and
the mass table are borrowed, not the calculator machinery.

**Temperature is ``k_B T`` in eV, not Kelvin.** Every nvalchemiops dynamics
entry point documents its temperature argument as "k_B*T in energy units".
This is the opposite of the heavy ``nvalchemi-toolkit``, whose thermostats
took Kelvin -- passing kT there silently produced a system at rest. This
module's public API takes Kelvin and converts, so the mistake cannot be made
from outside; :func:`kelvin_to_energy` is the single place it happens.
"""

from typing import (
    Any,
)

import numpy as np
import torch

from deepmd_lr.distributed.driver import (
    device_replicas,
    evaluate_partitioned,
)
from deepmd_lr.distributed.partition import (
    partition_system,
)

__all__ = [
    "BOLTZMANN_EV_PER_K",
    "ENSEMBLES",
    "FS",
    "DistributedMD",
    "kelvin_to_energy",
]

#: eV/K. ASE's value, so energies stay comparable with the serial path.
BOLTZMANN_EV_PER_K = 8.617333262e-5

#: One femtosecond in ASE time units (``A * sqrt(amu/eV)``).
FS = 0.09822694788464063

ENSEMBLES = ("nve", "nvt-langevin")


def kelvin_to_energy(temperature_k: float) -> float:
    """Kelvin -> ``k_B T`` in eV, which is what the kernels want."""
    return float(temperature_k) * BOLTZMANN_EV_PER_K


class DistributedMD:
    """Run MD with the model evaluated across several devices.

    Parameters
    ----------
    model
        A ``dp_lr`` model, already on its home device.
    coord, atype, cell
        The starting frame: ``(natoms, 3)``, ``(natoms,)``, ``(3, 3)``.
    devices
        One device per partition. Repeats are allowed but only exercise the
        partitioning arithmetic -- see the note in
        ``examples/run_distributed.py``.
    ensemble
        ``"nve"`` (velocity Verlet) or ``"nvt-langevin"`` (BAOAB).
    timestep
        In femtoseconds.
    temperature
        In **Kelvin**; converted internally.
    friction
        Langevin friction, in 1/fs.
    skin
        Halo padding beyond ``rcut``. The partitioning is reused until an
        atom has moved far enough to invalidate it.
    repartition_every
        Force a rebuild every N steps regardless. ``None`` uses the skin
        criterion alone.
    """

    def __init__(
        self,
        model: Any,
        coord: np.ndarray,
        atype: np.ndarray,
        cell: np.ndarray,
        *,
        devices: list[Any],
        ensemble: str = "nve",
        timestep: float = 0.5,
        temperature: float = 300.0,
        friction: float = 0.01,
        skin: float = 1.0,
        repartition_every: int | None = None,
        masses: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        if ensemble not in ENSEMBLES:
            raise ValueError(
                f"unknown ensemble {ensemble!r}; expected one of {ENSEMBLES}"
            )
        if not devices:
            raise ValueError("DistributedMD needs at least one device")
        if skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")

        self.model = model
        self.devices = list(devices)
        self.ensemble = ensemble
        self.timestep = float(timestep)
        self.temperature = float(temperature)
        self.friction = float(friction)
        self.skin = float(skin)
        self.repartition_every = repartition_every
        self.seed = int(seed)
        self.step_count = 0
        self.rebuild_count = 0

        self.home = next(model.parameters()).device
        self.atype = np.asarray(atype, dtype=np.int64).reshape(-1)
        self.cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        n_atoms = len(self.atype)

        def tensor(array):
            return torch.as_tensor(
                array, dtype=torch.float64, device=self.home
            ).contiguous()

        self.positions = tensor(np.asarray(coord, np.float64).reshape(-1, 3))
        self.velocities = tensor(np.zeros((n_atoms, 3)))
        self.forces = tensor(np.zeros((n_atoms, 3)))
        self.masses = tensor(
            self._default_masses() if masses is None else np.asarray(masses)
        )

        # Per-system scalars the kernels take as arrays.
        self._dt = tensor(np.array([self.timestep * FS]))
        self._kT = tensor(np.array([kelvin_to_energy(self.temperature)]))
        self._friction = tensor(np.array([self.friction / FS]))

        # Built once for the life of the device list, not per step.
        self._replicas = device_replicas(model, self.devices)
        self._partitions = None
        self._reference_positions = None
        self._calls_since_rebuild = 0

        self.potential_energy = 0.0
        self._compute_forces()

    # -- setup ---------------------------------------------------------------

    def _default_masses(self) -> np.ndarray:
        """Standard atomic masses, from the model's own type map."""
        from ase.data import atomic_masses, atomic_numbers

        type_map = self.model.get_type_map()
        table = np.array(
            [atomic_masses[atomic_numbers[name]] for name in type_map]
        )
        return table[self.atype]

    def set_temperature(self, temperature_k: float, seed: int | None = None) -> None:
        """Draw velocities from Maxwell-Boltzmann and remove COM drift."""
        import warp as wp
        from nvalchemiops.dynamics.utils.thermostat_utils import (
            initialize_velocities,
        )

        n = self.velocities.shape[0]
        target = torch.as_tensor(
            [kelvin_to_energy(temperature_k)], dtype=torch.float64, device=self.home
        )
        scratch = [
            torch.zeros(3, dtype=torch.float64, device=self.home),  # momentum
            torch.zeros(1, dtype=torch.float64, device=self.home),  # total mass
            torch.zeros(3, dtype=torch.float64, device=self.home),  # com velocity
        ]
        initialize_velocities(
            wp.from_torch(self.velocities, dtype=wp.vec3d),
            wp.from_torch(self.masses),
            wp.from_torch(target),
            wp.from_torch(scratch[0], dtype=wp.vec3d),
            wp.from_torch(scratch[1]),
            wp.from_torch(scratch[2], dtype=wp.vec3d),
            random_seed=self.seed if seed is None else int(seed),
            remove_com=True,
            num_systems=1,
        )
        wp.synchronize()
        assert self.velocities.shape[0] == n

    # -- diagnostics ---------------------------------------------------------

    def kinetic_energy(self) -> float:
        return float(
            0.5 * (self.masses.unsqueeze(-1) * self.velocities**2).sum()
        )

    def temperature_k(self) -> float:
        """Instantaneous temperature in Kelvin, from 3N degrees of freedom."""
        n_atoms = self.velocities.shape[0]
        return (
            2.0
            * self.kinetic_energy()
            / (3.0 * n_atoms * BOLTZMANN_EV_PER_K)
        )

    def total_energy(self) -> float:
        return self.potential_energy + self.kinetic_energy()

    # -- the loop ------------------------------------------------------------

    def _needs_rebuild(self) -> bool:
        """Standard Verlet criterion on the cached halo.

        The halo covers ``rcut + skin``, and two atoms approaching each other
        can each contribute, so the pair separation shrinks by up to twice the
        largest displacement.
        """
        if self._partitions is None or self._reference_positions is None:
            return True
        if (
            self.repartition_every is not None
            and self._calls_since_rebuild >= self.repartition_every
        ):
            return True
        moved = float(
            (self.positions - self._reference_positions).norm(dim=1).max()
        )
        return 2.0 * moved > self.skin

    def _compute_forces(self) -> None:
        """Evaluate the model over the partitions and store the result."""
        coord = self.positions.detach().cpu().numpy()
        if self._needs_rebuild():
            self._partitions = partition_system(
                coord,
                self.cell,
                float(self.model.get_rcut()),
                len(self.devices),
                pbc=(True, True, True),
                skin=self.skin,
            )
            self._reference_positions = self.positions.clone()
            self._calls_since_rebuild = 0
            self.rebuild_count += 1
        self._calls_since_rebuild += 1

        out = evaluate_partitioned(
            self.model,
            coord,
            self.atype,
            self.cell,
            n_partitions=len(self.devices),
            devices=self.devices,
            partitions=self._partitions,
            replicas=self._replicas,
        )
        self.potential_energy = float(out["energy"])
        self.forces.copy_(
            torch.as_tensor(out["force"], dtype=torch.float64, device=self.home)
        )
        self.results = out

    def step(self) -> None:
        """Advance one timestep."""
        import warp as wp

        vec = lambda t: wp.from_torch(t, dtype=wp.vec3d)  # noqa: E731
        positions = vec(self.positions)
        velocities = vec(self.velocities)
        masses = wp.from_torch(self.masses)
        dt = wp.from_torch(self._dt)

        if self.ensemble == "nve":
            from nvalchemiops.dynamics.integrators import (
                velocity_verlet_position_update,
                velocity_verlet_velocity_finalize,
            )

            velocity_verlet_position_update(
                positions, velocities, vec(self.forces), masses, dt
            )
            self._compute_forces()
            velocity_verlet_velocity_finalize(
                velocities, vec(self.forces), masses, dt
            )
        else:
            from nvalchemiops.dynamics.integrators import (
                langevin_baoab_finalize,
                langevin_baoab_half_step,
            )

            langevin_baoab_half_step(
                positions,
                velocities,
                vec(self.forces),
                masses,
                dt,
                wp.from_torch(self._kT),
                wp.from_torch(self._friction),
                random_seed=self.seed + self.step_count,
            )
            self._compute_forces()
            langevin_baoab_finalize(
                velocities, vec(self.forces), masses, dt
            )
        wp.synchronize()
        self.step_count += 1

    def run(self, steps: int) -> None:
        for _ in range(int(steps)):
            self.step()
