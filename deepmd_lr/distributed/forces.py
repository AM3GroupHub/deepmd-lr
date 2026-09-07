# SPDX-License-Identifier: LGPL-3.0-or-later
"""Energy and forces from the model, evaluated across several devices.

This is the force driver an MD engine calls. It owns the partitioning, the
ghost halo, the per-device model replicas and the decision of when the halo
needs rebuilding -- and nothing else. It has no timestep, no ensemble, no
thermostat and no velocities: those belong to whoever is integrating.

nvalchemiops' integrators are half-step kernels with no driver of their own,
so the caller composes the loop and asks this for the force term:

    from nvalchemiops.dynamics.integrators import (
        velocity_verlet_position_update,
        velocity_verlet_velocity_finalize,
    )

    driver = DistributedForces(model, atype, cell, devices=["cuda:0", "cuda:1"])
    energy, forces = driver(positions)
    for _ in range(steps):
        velocity_verlet_position_update(r, v, f, m, dt)   # Warp views
        energy, forces = driver(positions)
        velocity_verlet_velocity_finalize(v, f, m, dt)

Forces come back as a float64 torch tensor on the model's device, which
``wp.from_torch`` turns into a zero-copy Warp view -- so the trajectory never
has to leave the GPU for the integrator's sake.

There is deliberately no ASE calculator here. ASE's interface returns energy
and forces for one configuration at a time and its integrators cannot use
on-device state, which is the whole point of this path.

Units are whatever the model uses -- angstrom and eV for a DeePMD-kit model.
Pair them with masses in amu and a timestep in ASE time units
(``A sqrt(amu/eV)``, ``1 fs = 0.0982269...``) and the set is self-consistent;
:data:`FS` is provided for that conversion and is the only unit constant here.
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
    "FS",
    "DistributedForces",
    "atomic_masses",
]

#: One femtosecond in ASE time units (``A * sqrt(amu/eV)``), for pairing a
#: timestep in fs with eV/angstrom forces and amu masses.
FS = 0.09822694788464063


def atomic_masses(model: Any, atype: np.ndarray) -> np.ndarray:
    """Standard atomic masses in amu, ordered like ``atype``.

    A convenience for setting up an integrator: the mapping from type index
    to element is the model's own ``type_map``, so the caller does not have
    to reconstruct it.
    """
    from ase.data import atomic_masses as table
    from ase.data import atomic_numbers

    type_map = model.get_type_map()
    masses = np.array([table[atomic_numbers[name]] for name in type_map])
    return masses[np.asarray(atype, dtype=np.int64).reshape(-1)]


class DistributedForces:
    """Evaluate a ``dp_lr`` model over spatial partitions, one per device.

    Parameters
    ----------
    model
        A ``dp_lr`` model, already on its home device.
    atype, cell
        Atom types ``(natoms,)`` and the cell ``(3, 3)``. Both fixed for the
        life of the driver; a changing cell means rebuilding it.
    devices
        One device per partition. Repeats are accepted and exercise the
        partitioning arithmetic, but only distinct devices test placement.
    skin
        Halo padding beyond ``rcut``. The partitioning is reused until an
        atom has moved far enough to invalidate it.
    repartition_every
        Force a rebuild every N calls regardless. ``None`` uses the skin
        criterion alone, which is exact rather than merely usually right.
    compute_virial
        Also assemble the virial, exposed as :attr:`virial` (eV) and
        :attr:`stress` (eV/angstrom^3, ASE sign convention) after each call.
        Off by default because it costs an extra strain leaf per partition
        and a second backward through the charge head.

    Notes
    -----
    The reciprocal sum is gathered onto one device rather than decomposed,
    because it does not survive spatial decomposition. The charge-response
    force stays partitioned: ``dq_j/dr_i`` is nonzero only within ``rcut``,
    so although the energy is long-ranged its charge derivative is short.
    """

    def __init__(
        self,
        model: Any,
        atype: np.ndarray,
        cell: np.ndarray,
        *,
        devices: list[Any],
        skin: float = 1.0,
        repartition_every: int | None = None,
        compute_virial: bool = False,
    ) -> None:
        if not devices:
            raise ValueError("DistributedForces needs at least one device")
        if skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")

        self.model = model
        self.devices = list(devices)
        self.skin = float(skin)
        self.repartition_every = repartition_every
        self.compute_virial = bool(compute_virial)
        self.home = next(model.parameters()).device
        self.atype = np.asarray(atype, dtype=np.int64).reshape(-1)
        self.cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)

        # One deepcopy per non-home device, held for the life of the driver
        # rather than rebuilt per call.
        self._replicas = device_replicas(model, self.devices)
        self._partitions = None
        self._reference_positions = None
        self._calls_since_rebuild = 0

        self.rebuild_count = 0
        self.results: dict[str, Any] = {}
        self.virial: np.ndarray | None = None
        self.forces = torch.zeros(
            (len(self.atype), 3), dtype=torch.float64, device=self.home
        )

    @property
    def n_atoms(self) -> int:
        return len(self.atype)

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.cell)))

    @property
    def stress(self) -> np.ndarray | None:
        """Stress in eV/angstrom^3, ``-virial / volume`` as ASE defines it.

        ``None`` until a call has run with ``compute_virial``.
        """
        if self.virial is None:
            return None
        return -self.virial / self.volume

    def masses(self) -> np.ndarray:
        """Standard masses for this system, in amu."""
        return atomic_masses(self.model, self.atype)

    def _needs_rebuild(self, positions: torch.Tensor) -> bool:
        """Standard Verlet criterion on the cached halo.

        The halo covers ``rcut + skin``, and two atoms approaching each other
        can each contribute, so the pair separation shrinks by up to twice
        the largest displacement. Rebuilding once that exceeds the skin keeps
        every real neighbour inside the halo.
        """
        if self._partitions is None or self._reference_positions is None:
            return True
        if (
            self.repartition_every is not None
            and self._calls_since_rebuild >= self.repartition_every
        ):
            return True
        moved = float((positions - self._reference_positions).norm(dim=1).max())
        return 2.0 * moved > self.skin

    def __call__(self, positions: Any) -> tuple[float, torch.Tensor]:
        """Energy in eV and forces in eV/angstrom at ``positions``.

        ``positions`` may be a torch tensor -- including the one an integrator
        is mutating in place -- or anything ``np.asarray`` accepts. The
        returned tensor is reused between calls, so copy it if you need to
        keep a previous step's forces.

        With ``compute_virial``, :attr:`virial` and :attr:`stress` are also
        refreshed. They are not returned, so the ``(energy, forces)``
        signature an integrator consumes stays the same either way.
        """
        if isinstance(positions, torch.Tensor):
            tracked = positions.detach().to(
                device=self.home, dtype=torch.float64
            )
        else:
            tracked = torch.as_tensor(
                np.asarray(positions, dtype=np.float64).reshape(-1, 3),
                device=self.home,
            )
        if tracked.shape != (self.n_atoms, 3):
            raise ValueError(
                f"expected positions of shape {(self.n_atoms, 3)}, "
                f"got {tuple(tracked.shape)}"
            )

        coord = tracked.cpu().numpy()
        if self._needs_rebuild(tracked):
            self._partitions = partition_system(
                coord,
                self.cell,
                float(self.model.get_rcut()),
                len(self.devices),
                pbc=(True, True, True),
                skin=self.skin,
            )
            self._reference_positions = tracked.clone()
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
            compute_virial=self.compute_virial,
        )
        self.results = out
        self.virial = out.get("virial")
        self.forces.copy_(
            torch.as_tensor(out["force"], dtype=torch.float64, device=self.home)
        )
        return float(out["energy"]), self.forces
