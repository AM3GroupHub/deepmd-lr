# SPDX-License-Identifier: MIT
"""Getting from a structure file and a checkpoint to something runnable."""

from __future__ import annotations

from typing import Any

import ase.io
import numpy as np

from deepmd_lr.ase import load_model
from deepmd_lr.nvalchemi.evaluator import DistributedEvaluator

__all__ = ["DEFAULT_VACUUM", "build_evaluator", "read_structure"]


#: The gap, in angstrom, a slab's open axis gets when its file leaves less.
DEFAULT_VACUUM = 20.0


def read_structure(path: Any, index: int = -1, vacuum: float = DEFAULT_VACUUM) -> Any:
    """Read a starting structure with ASE.

    Anything ``ase.io.read`` understands -- extxyz, POSCAR, CIF, an ASE
    trajectory. ``index`` picks the frame from a multi-frame file and defaults
    to the last, usually the most equilibrated.

    The file's ``pbc`` is kept, and it decides the boundary. All three axes
    periodic is a bulk cell, which the file must carry. One open axis is a
    slab: the evaluator makes no images along it and adds the Yeh-Berkowitz
    correction to the long-range sum. That sum is still taken over the whole
    cell, so the open axis needs vacuum. If the file leaves less than
    ``vacuum`` angstrom of gap, or no cell vector along it at all, the axis is
    rebuilt perpendicular to the slab, as long as the slab plus ``vacuum``.
    Either way the slab is centred, moved as a whole along its normal, and a
    slab the file wrapped across the cell face is first made whole again by
    lattice vectors, so no atom starts near a face.

    Two or three open axes -- a wire, a molecule, a cluster -- are refused:
    those run through the ASE calculator.

    Atoms the file holds fixed (POSCAR's selective dynamics ``F F F``,
    extxyz's ``move_mask``) come back as ASE's ``FixAtoms``, which the MD
    classes honour. Centring moves them with the slab.
    """
    atoms = ase.io.read(str(path), index=index)
    if isinstance(atoms, list):
        atoms = atoms[-1]
    pbc = [bool(p) for p in atoms.pbc]
    if sum(pbc) < 2:
        raise ValueError(
            f"{path} has pbc={pbc}: partitioned MD runs a bulk cell or a slab "
            "(one open axis). A molecule or cluster runs through the ASE "
            "calculator, deepmd_lr.ase.LongRangeCalculator(model, "
            "long_range={'method': 'isolated'})."
        )
    if all(pbc):
        volume = abs(float(np.linalg.det(np.asarray(atoms.get_cell()))))
        if volume < 1e-8:
            raise ValueError(
                f"{path} has no cell (volume {volume:.2e} A^3). Periodic MD "
                "needs one; set it in the file."
            )
        return atoms
    if not vacuum > 0:
        raise ValueError(f"vacuum must be positive, got {vacuum}")
    _pad_slab(atoms, pbc.index(False), float(vacuum), path)
    return atoms


def _pad_slab(atoms: Any, axis: int, vacuum: float, path: Any) -> None:
    """Give a slab's open ``axis`` at least ``vacuum`` of gap, in place."""
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    a, b = (cell[j] for j in range(3) if j != axis)
    normal = np.cross(a, b)
    area = float(np.linalg.norm(normal))
    if area < 1e-8:
        raise ValueError(
            f"{path}: the two periodic cell vectors do not span a plane; a slab "
            "needs its in-plane cell in the file."
        )
    normal /= area
    if axis == 1:  # keep the cell right-handed, whichever axis is open
        normal = -normal
    pos = atoms.get_positions()
    depth = float(cell[axis] @ normal)
    height = pos @ normal
    if depth > 1e-8 and np.ptp(height) < depth:
        # The atoms fit in one cell length, so the file may have wrapped the
        # slab across a face. Cut the cell at its widest empty band instead:
        # each atom moves by a lattice vector, which changes nothing periodic.
        frac = pos @ np.linalg.inv(cell)
        f = np.sort(np.mod(frac[:, axis], 1.0))
        gaps = np.diff(np.append(f, f[0] + 1.0))
        start = f[(int(np.argmax(gaps)) + 1) % len(f)]
        shift = np.floor(frac[:, axis] - start)
        pos = pos - shift[:, None] * cell[axis]
        height = pos @ normal
    span = float(np.ptp(height))
    gap = depth - span
    length = depth if depth > 1e-8 and gap >= vacuum else span + vacuum
    cell[axis] = normal * length
    target = (length - span) / 2.0 - height.min()
    atoms.set_cell(cell, scale_atoms=False)
    # Past the constraints: FixAtoms would otherwise leave the fixed atoms
    # behind while the rest of the slab moves.
    atoms.set_positions(pos + target * normal[None, :], apply_constraint=False)


def build_evaluator(model: Any, atoms: Any, devices: Any = None,
                    head: str | None = None, long_range: dict | None = None,
                    **kwargs: Any) -> DistributedEvaluator:
    """Load a ``dp_lr`` checkpoint and wrap it for this structure.

    One device is the normal case; pass several to partition across GPUs.
    ``kwargs`` forwards to :class:`DistributedEvaluator` -- ``compute_virial``
    for NPT, ``efield`` for a field run, and so on.

    ``long_range`` replaces the reciprocal-space settings the checkpoint was
    trained with, for this evaluator only: the term holds no parameters, so
    the choice belongs to the simulation. The structure's ``pbc`` already
    picks the 3D or the slab sum, so this is rarely needed. A model passed in
    is never changed, and may serve several evaluators.

    ``compress=True`` (or a float table stride) rides ``kwargs`` into the
    evaluator and compresses the DPA4C backbone before the device replicas are
    made. It needs a deepmd-kit built from source with the CUDA operators and
    is refused otherwise; see ``LongRangeEnergyModel.enable_compression``.

    Masses are deliberately not returned: the evaluator already derives them
    from the model's own ``type_map``, and a second table built from the
    file's symbols would be a second source of truth that agrees only by
    coincidence.
    """
    loaded = load_model(model, head=head) if isinstance(model, str) else model
    type_map = list(loaded.get_type_map())
    atype = np.array(
        [type_map.index(s) for s in atoms.get_chemical_symbols()], dtype=np.int64
    )
    if devices is None:
        devices = [str(next(loaded.parameters()).device)]
    kwargs.setdefault("pbc", tuple(bool(p) for p in atoms.pbc))
    return DistributedEvaluator(
        loaded,
        atype,
        np.asarray(atoms.get_cell()),
        devices=devices,
        long_range=long_range,
        **kwargs,
    )
