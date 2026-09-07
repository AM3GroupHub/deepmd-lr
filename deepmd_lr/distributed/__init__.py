# SPDX-License-Identifier: LGPL-3.0-or-later
"""Distributed evaluation and MD: spatial partitioning with a ghost halo.

MD in this path runs on nvalchemiops' integrator kernels via
:class:`DistributedMD`, not through an ASE calculator. The serial
:class:`deepmd_lr.ase.LongRangeCalculator` remains for single-device work,
where ASE's ecosystem is worth having.
"""

from deepmd_lr.distributed.driver import (
    device_replicas,
    evaluate_partitioned,
    partition_graph,
)
from deepmd_lr.distributed.md import (
    ENSEMBLES,
    DistributedMD,
    kelvin_to_energy,
)
from deepmd_lr.distributed.partition import (
    Partition,
    partition_system,
)

__all__ = [
    "ENSEMBLES",
    "DistributedMD",
    "Partition",
    "device_replicas",
    "evaluate_partitioned",
    "kelvin_to_energy",
    "partition_graph",
    "partition_system",
]
