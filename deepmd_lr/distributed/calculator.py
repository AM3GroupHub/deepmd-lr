# SPDX-License-Identifier: LGPL-3.0-or-later
"""An ASE calculator that evaluates the model over spatial partitions.

Same results as :class:`deepmd_lr.ase.LongRangeCalculator`, computed over
partitions so the descriptor -- the expensive part, 883 ms against the
long-range term's 73 ms at 6520 atoms -- can be spread over devices.

The interface follows DistMLIP: ``from_existing`` to wrap a calculator you
already have, ``enable_distributed_mode`` to hand it devices. One process,
several devices, tensors moved with ``.to()``; no ``torch.distributed``.
Passing one device repeatedly (``["cuda:0"] * 4``) is supported and is how the
partitioning is exercised on a single-GPU machine.
"""

from typing import (
    Any,
)

import numpy as np
from ase.calculators.calculator import (
    all_changes,
)

from deepmd_lr.ase import (
    LongRangeCalculator,
)
from deepmd_lr.distributed.driver import (
    evaluate_partitioned,
)
from deepmd_lr.distributed.partition import (
    partition_system,
)

__all__ = ["DistLongRangeCalculator"]

#: Default extra halo width, in angstrom. Ghosts are gathered within
#: ``rcut + skin`` so a partitioning stays usable for several MD steps.
DEFAULT_SKIN = 1.0


class DistLongRangeCalculator(LongRangeCalculator):
    """Partitioned evaluation of a ``dp_lr`` model.

    Parameters
    ----------
    model
        Checkpoint path or a constructed model, as for
        :class:`~deepmd_lr.ase.LongRangeCalculator`.
    devices
        One device per partition; repeats allowed. ``None`` keeps the model's
        own device and evaluates a single partition.
    skin
        Extra halo width. A partitioning is rebuilt only once atoms have moved
        far enough to threaten it, so this trades memory for rebuild frequency
        without ever changing the result.
    repartition_every
        Force a rebuild every N calls regardless of displacement. ``None``
        leaves it to the displacement test alone.

    Notes
    -----
    ``stress`` is not provided: the partitioned path assembles energy and
    forces, and the virial would need the same treatment. Leaving it out of
    ``implemented_properties`` makes ASE raise ``PropertyNotImplementedError``
    of its own accord, which is better than returning a wrong virial or
    duplicating the refusal here. Use
    :class:`deepmd_lr.ase.LongRangeCalculator` when the virial is needed.
    """

    implemented_properties = ["energy", "free_energy", "forces", "charges"]

    def __init__(
        self,
        model: str | Any,
        devices: list[Any] | None = None,
        skin: float = DEFAULT_SKIN,
        repartition_every: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        if skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")
        self.skin = float(skin)
        self.repartition_every = repartition_every
        self.devices = None
        self._partitions = None
        self._reference_positions = None
        self._calls_since_rebuild = 0
        self.rebuild_count = 0
        if devices is not None:
            self.enable_distributed_mode(devices)

    @classmethod
    def from_existing(cls, existing: LongRangeCalculator, **kwargs: Any):
        """Wrap an already-constructed calculator, reusing its loaded model."""
        return cls(existing.model, **kwargs)

    def enable_distributed_mode(self, devices: list[Any] | int):
        """Set the devices to spread partitions over.

        Accepts a device list, or an integer to use that many partitions on
        the model's current device -- which is how the partitioning logic is
        tested where only one GPU exists.
        """
        if isinstance(devices, int):
            devices = [self.device] * devices
        if not devices:
            raise ValueError("enable_distributed_mode needs at least one device")
        self.devices = list(devices)
        self._invalidate()
        return self

    def _invalidate(self) -> None:
        self._partitions = None
        self._reference_positions = None
        self._calls_since_rebuild = 0

    def _needs_rebuild(self, positions: np.ndarray) -> bool:
        """Has anything moved far enough to invalidate the cached halo?

        The halo covers ``rcut + skin``. Two atoms approaching each other can
        each contribute, so the pair separation can shrink by twice the
        largest displacement; rebuilding once that exceeds the skin keeps
        every real neighbour inside the halo. This is the standard Verlet
        criterion, and it makes reuse safe rather than merely likely.
        """
        if self._partitions is None or self._reference_positions is None:
            return True
        if (
            self.repartition_every is not None
            and self._calls_since_rebuild >= self.repartition_every
        ):
            return True
        moved = np.linalg.norm(positions - self._reference_positions, axis=1).max()
        return 2.0 * moved > self.skin

    def calculate(
        self,
        atoms: Any = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        # Deliberately not calling LongRangeCalculator.calculate: that is the
        # serial path. Go up to the ASE base for the bookkeeping only.
        super(LongRangeCalculator, self).calculate(
            atoms, properties or ["energy"], system_changes
        )
        atoms = self.atoms

        if not atoms.pbc.all():
            raise ValueError(
                "the long-range term needs a fully periodic cell, but this "
                f"Atoms object has pbc={atoms.pbc}."
            )

        coord = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        atype = self._atom_types(atoms).cpu().numpy()
        devices = self.devices if self.devices is not None else [self.device]

        if self._needs_rebuild(coord):
            self._partitions = partition_system(
                coord,
                cell,
                float(self.model.get_rcut()),
                len(devices),
                pbc=(True, True, True),
                skin=self.skin,
            )
            self._reference_positions = coord.copy()
            self._calls_since_rebuild = 0
            self.rebuild_count += 1
        self._calls_since_rebuild += 1

        out = evaluate_partitioned(
            self.model,
            coord,
            atype,
            cell,
            n_partitions=len(devices),
            devices=devices,
            partitions=self._partitions,
        )

        self.results["energy"] = out["energy"]
        self.results["free_energy"] = out["energy"]
        self.results["forces"] = out["force"]
        for key in ("charge", "energy_short_range", "energy_long_range"):
            if key in out:
                name = "charges" if key == "charge" else key
                self.results[name] = out[key]

    def set_positions_changed(self) -> None:
        """Drop the cached partitioning (e.g. after a cell change)."""
        self._invalidate()
