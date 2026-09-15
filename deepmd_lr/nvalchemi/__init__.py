# SPDX-License-Identifier: LGPL-3.0-or-later
"""Running MD with nvalchemiops: the evaluator it calls for forces.

:class:`DistributedEvaluator` is the driver an MD engine calls -- energy and
forces across several devices, plus the virial on request, and nothing else.
The integrator, ensemble and thermostat belong to the caller;
``nvalchemiops.dynamics.integrators`` supplies the half-step kernels and
consumes the returned tensor through ``wp.from_torch`` without a copy.

:class:`Thermo` is the other half an MD loop needs: it owns the state those
kernels mutate -- positions, velocities, masses, forces, cell, timestep -- and
reports the thermodynamics, which nvalchemiops does not do at all. Ensembles
stay with the caller; see ``examples/05-md_nvalchemi`` for a loop per ensemble.
"""

from deepmd_lr.nvalchemi.driver import (
    device_replicas,
    evaluate_partitioned,
    partition_graph,
)
from deepmd_lr.nvalchemi.evaluator import (
    DistributedEvaluator,
    atomic_masses,
)
from deepmd_lr.nvalchemi.partition import (
    Partition,
    partition_system,
)
from deepmd_lr.nvalchemi.setup import (
    build_evaluator,
    read_structure,
)
from deepmd_lr.nvalchemi.thermo import (
    Thermo,
)

__all__ = [
    "DistributedEvaluator",
    "Thermo",
    "Partition",
    "atomic_masses",
    "build_evaluator",
    "device_replicas",
    "evaluate_partitioned",
    "partition_graph",
    "partition_system",
    "read_structure",
]
