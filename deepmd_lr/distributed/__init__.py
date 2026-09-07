# SPDX-License-Identifier: LGPL-3.0-or-later
"""Distributed evaluation: spatial partitioning with a ghost halo.

:class:`DistributedForces` is the force driver an MD engine calls -- energy
and forces across several devices, and nothing else. The integrator,
ensemble and thermostat belong to the caller;
``nvalchemiops.dynamics.integrators`` supplies the half-step kernels and
consumes the returned tensor through ``wp.from_torch`` without a copy.
"""

from deepmd_lr.distributed.driver import (
    device_replicas,
    evaluate_partitioned,
    partition_graph,
)
from deepmd_lr.distributed.forces import (
    FS,
    DistributedForces,
    atomic_masses,
)
from deepmd_lr.distributed.partition import (
    Partition,
    partition_system,
)

__all__ = [
    "FS",
    "DistributedForces",
    "Partition",
    "atomic_masses",
    "device_replicas",
    "evaluate_partitioned",
    "partition_graph",
    "partition_system",
]
