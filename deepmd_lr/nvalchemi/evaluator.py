# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate the model across several devices: energy, forces, virial.

This is the driver an MD engine calls. It owns the partitioning, the ghost
halo, the per-device model replicas and the decision of when the halo needs
rebuilding -- and nothing else. It has no timestep, no ensemble, no
thermostat and no velocities: those belong to whoever is integrating.

Energy and forces always; the virial and stress on request, since they cost
an extra strain leaf per partition and a second backward through the charge
head. The ``(energy, forces)`` return signature does not change either way,
so an integrator is unaffected by the choice.

nvalchemiops' integrators are half-step kernels with no driver of their own,
so the caller composes the loop and asks this for the force term:

    from nvalchemiops.dynamics.integrators import (
        velocity_verlet_position_update,
        velocity_verlet_velocity_finalize,
    )

    driver = DistributedEvaluator(model, atype, cell, devices=["cuda:0", "cuda:1"])
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
"""

from typing import (
    Any,
)

import numpy as np
import torch

from deepmd_lr.born import (
    field_force,
)
from deepmd_lr.nvalchemi.driver import (
    device_replicas,
    evaluate_partitioned,
)
from deepmd_lr.nvalchemi.partition import (
    partition_system,
)

__all__ = [
    "DistributedEvaluator",
    "atomic_masses",
]


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


class DistributedEvaluator:
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
    compute_born
        Also assemble Born effective charge tensors, available afterwards as
        ``results["born_charge"]`` with shape ``(natoms, 3, 3)``. The global
        total-charge projection is no obstacle: it enters only the
        ``q_i delta`` term, which arrives already projected, while the
        derivative half uses the raw charges and is strictly local.
    compute_virial
        Also assemble the virial, exposed as :attr:`virial` (eV) and
        :attr:`stress` (eV/angstrom^3, ASE sign convention) after each call.
        Off by default because it costs an extra strain leaf per partition
        and a second backward through the charge head.
    efield
        Uniform external field in V/angstrom, or ``None``. Adds the response
        force ``F_ib = E_a Z_iab`` to every returned force and implies
        ``compute_born``. The reported energy is left untouched: this force
        is NOT the gradient of a ``-M.E`` energy -- under periodic boundaries
        ``M`` is not even defined -- so a constant-field NVE run will not
        conserve ``E - M.E``, by construction rather than by error. Use a
        thermostat and read structure and response, not a conserved quantity.
        Rejected together with ``compute_virial``, because the field's
        contribution to the stress is a separate derivation and returning the
        field-free value would be quietly wrong.
    charge_spin
        Fixed (total charge, spin multiplicity) for this system. If omitted,
        use the model's configured default condition.

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
        compute_born: bool = False,
        efield: Any = None,
        charge_spin: torch.Tensor | tuple[float, float] | None = None,
    ) -> None:
        if not devices:
            raise ValueError("DistributedEvaluator needs at least one device")
        if skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")

        self.model = model
        self.devices = list(devices)
        self.skin = float(skin)
        self.repartition_every = repartition_every
        self.compute_virial = bool(compute_virial)
        #: Also assemble Born effective charge tensors, exposed as
        #: ``results['born_charge']`` ``(natoms, 3, 3)``. Costs one extra
        #: backward pass per partition; needed for MD in a field.
        self.compute_born = bool(compute_born)
        #: Uniform external field in V/angstrom, or None. See the class
        #: docstring: the force it adds is a local response, not the gradient
        #: of a field energy.
        self.efield = (
            None if efield is None else np.asarray(efield, dtype=np.float64).reshape(3)
        )
        if self.efield is not None:
            if not getattr(model, "has_long_range", False):
                raise ValueError(
                    "an external field needs Born effective charges, which "
                    "need the charge head -- this model has no `long_range` "
                    "block, so it predicts no charges."
                )
            if self.compute_virial:
                raise ValueError(
                    "compute_virial and efield cannot be combined: the "
                    "field's contribution to the stress is a separate "
                    "derivation, and reporting the field-free virial under a "
                    "field would be quietly wrong."
                )
            # The field force is built from Z, so Z has to be assembled.
            self.compute_born = True
        self.charge_spin = charge_spin
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
            compute_born=self.compute_born,
            charge_spin=self.charge_spin,
        )
        self.results = out
        self.virial = out.get("virial")
        self.forces.copy_(
            torch.as_tensor(out["force"], dtype=torch.float64, device=self.home)
        )
        if self.efield is not None:
            # Through `field_force` rather than an einsum written out again:
            # which index of Z contracts with E is a convention that has been
            # wrong here before, and one definition cannot drift from another.
            born = torch.as_tensor(
                out["born_charge"], dtype=torch.float64, device=self.home
            )
            response = field_force(born[None], self.efield)[0]
            self.forces += response
            # Kept separately so a caller can watch the field term without
            # re-deriving it, and so `out["force"]` stays the field-free force.
            self.results["field_force"] = response.cpu().numpy()
        return float(out["energy"]), self.forces
