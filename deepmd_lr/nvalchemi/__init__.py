# SPDX-License-Identifier: LGPL-3.0-or-later
"""Running MD with nvalchemiops: the evaluator, the state, and the integrators.

- :class:`DistributedEvaluator` computes energy and forces across several
  devices, plus the virial and Born charges on request. Build it once, with
  :func:`build_evaluator`, and share it.
- The MD classes -- :class:`VelocityVerlet` (NVE); :class:`Langevin`,
  :class:`NoseHooverChainNVT` and :class:`VelocityRescaling` (NVT);
  :class:`MTKNPT` and :class:`IsotropicMTKNPT` (NPT); :class:`MTKNPH` and
  :class:`IsotropicMTKNPH` (NPH) -- and the relaxations :class:`FIRE` and
  :class:`FIRE2` take that evaluator and an ``ase.Atoms``, ASE-style:
  parameters in ``__init__``, then ``run(steps)``. Several may run one after
  another on one evaluator.
- Each builds a :class:`Thermo`: the device state the nvalchemiops kernels
  mutate (positions, velocities, masses, forces view, cell) and its
  measurements. A custom kernel loop can use a Thermo directly; see
  ``examples/06-md_nvalchemi``.
"""

from deepmd_lr.nvalchemi.driver import (
    device_replicas,
    evaluate_partitioned,
    partition_graph,
    to_numpy,
)
from deepmd_lr.nvalchemi.evaluator import (
    DistributedEvaluator,
    atomic_masses,
)
from deepmd_lr.nvalchemi.md import (
    MTKNPH,
    MTKNPT,
    IsotropicMTKNPH,
    IsotropicMTKNPT,
    Langevin,
    MDInstabilityError,
    MolecularDynamics,
    NoseHooverChainNVT,
    VelocityRescaling,
    VelocityVerlet,
)
from deepmd_lr.nvalchemi.optimize import (
    FIRE,
    FIRE2,
    RelaxationError,
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
    "FIRE",
    "FIRE2",
    "IsotropicMTKNPH",
    "IsotropicMTKNPT",
    "Langevin",
    "MDInstabilityError",
    "MTKNPH",
    "MTKNPT",
    "MolecularDynamics",
    "NoseHooverChainNVT",
    "Partition",
    "RelaxationError",
    "Thermo",
    "VelocityRescaling",
    "VelocityVerlet",
    "atomic_masses",
    "build_evaluator",
    "device_replicas",
    "evaluate_partitioned",
    "partition_graph",
    "partition_system",
    "read_structure",
    "to_numpy",
]
