# SPDX-License-Identifier: LGPL-3.0-or-later
"""Getting from a structure file and a checkpoint to something runnable."""

from __future__ import annotations

from typing import Any

import ase.io
import numpy as np

from deepmd_lr.ase import load_model
from deepmd_lr.nvalchemi.evaluator import DistributedEvaluator

__all__ = ["build_evaluator", "read_structure"]


def read_structure(path: Any, index: int = -1) -> Any:
    """Read a starting structure with ASE.

    Anything ``ase.io.read`` understands -- extxyz, POSCAR, CIF, an ASE
    trajectory. ``index`` picks the frame from a multi-frame file and defaults
    to the last, usually the most equilibrated.

    A cell is required, and its absence is refused here. These are periodic
    simulations: without one the descriptor would silently treat the system as
    a molecule in vacuum and the reciprocal-space term would refuse it much
    further along, where the cause is far less obvious.
    """
    atoms = ase.io.read(str(path), index=index)
    if isinstance(atoms, list):
        atoms = atoms[-1]
    volume = abs(float(np.linalg.det(np.asarray(atoms.get_cell()))))
    if volume < 1e-8:
        raise ValueError(
            f"{path} has no cell (volume {volume:.2e} A^3). Periodic MD needs "
            "one -- set it in the file, or pad a slab with vacuum."
        )
    if not atoms.pbc.all():
        atoms.pbc = True                     # a cell implies periodicity
    return atoms


def build_evaluator(model: Any, atoms: Any, devices: Any = None,
                    head: str | None = None, **kwargs: Any) -> DistributedEvaluator:
    """Load a ``dp_lr`` checkpoint and wrap it for this structure.

    One device is the normal case; pass several to partition across GPUs.
    ``kwargs`` forwards to :class:`DistributedEvaluator` -- ``compute_virial``
    for NPT, ``efield`` for a field run, and so on.

    Masses are deliberately not returned: the evaluator already derives them
    from the model's own ``type_map``, and a second table built from the
    file's symbols would be a second source of truth that agrees only by
    coincidence.
    """
    loaded = load_model(str(model), head=head) if isinstance(model, str) else model
    type_map = list(loaded.get_type_map())
    atype = np.array(
        [type_map.index(s) for s in atoms.get_chemical_symbols()], dtype=np.int64
    )
    if devices is None:
        devices = [str(next(loaded.parameters()).device)]
    return DistributedEvaluator(
        loaded, atype, np.asarray(atoms.get_cell()), devices=devices, **kwargs
    )
