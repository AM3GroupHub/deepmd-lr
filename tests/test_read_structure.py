"""``read_structure``: the file's pbc decides the boundary, and a slab gets vacuum.

A slab's open axis is kept if the file leaves at least ``vacuum`` of gap, and
otherwise rebuilt perpendicular to the slab, as long as the slab plus
``vacuum``. Either way the slab is centred, moved only along its normal, and a
slab the file wrapped across the face is first made whole by lattice vectors.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write

from deepmd_lr.nvalchemi import read_structure
from deepmd_lr.nvalchemi.setup import DEFAULT_VACUUM


def layer(z_low=2.0, z_high=8.0, box=10.0, n=24, seed=0) -> Atoms:
    rng = np.random.default_rng(seed)
    pos = np.column_stack(
        [rng.uniform(0, box, n), rng.uniform(0, box, n), rng.uniform(z_low, z_high, n)]
    )
    return Atoms(["O"] * n, positions=pos, cell=[box, box, box], pbc=True)


def roundtrip(atoms, tmp_path, **kwargs):
    path = tmp_path / "start.extxyz"
    write(path, atoms)
    return read_structure(path, **kwargs)


def centred(atoms):
    z = atoms.positions[:, 2]
    return z.min() == pytest.approx(atoms.cell[2, 2] - z.max(), abs=1e-9)


def test_a_bulk_cell_is_kept(tmp_path):
    atoms = layer()
    out = roundtrip(atoms, tmp_path)
    assert out.pbc.all()
    np.testing.assert_allclose(out.cell, atoms.cell)
    np.testing.assert_allclose(out.positions, atoms.positions)


def test_a_bulk_cell_must_be_in_the_file(tmp_path):
    atoms = layer()
    atoms.set_cell(np.zeros((3, 3)))
    atoms.pbc = True
    with pytest.raises(ValueError, match="has no cell"):
        roundtrip(atoms, tmp_path)


def test_a_slab_without_a_z_vector_is_padded_and_centred(tmp_path):
    atoms = layer()
    atoms.set_cell([10.0, 10.0, 0.0])
    atoms.pbc = [True, True, False]
    out = roundtrip(atoms, tmp_path)
    assert out.pbc.tolist() == [True, True, False]
    span = np.ptp(atoms.positions[:, 2])
    np.testing.assert_allclose(out.cell[2], [0.0, 0.0, span + DEFAULT_VACUUM])
    assert centred(out)
    # Moved along the normal only.
    np.testing.assert_allclose(out.positions[:, :2], atoms.positions[:, :2])


def test_a_wide_gap_is_kept_and_the_slab_centred(tmp_path):
    atoms = layer()
    atoms.set_cell([10.0, 10.0, 40.0])  # a 6 A layer in 40 A: 34 A of gap
    atoms.pbc = [True, True, False]
    out = roundtrip(atoms, tmp_path)
    assert out.cell[2, 2] == pytest.approx(40.0)
    assert centred(out)


def test_a_narrow_gap_is_padded_and_vacuum_sets_how_much(tmp_path):
    atoms = layer()
    atoms.set_cell([10.0, 10.0, 12.0])  # 6 A of gap, under the 20 A default
    atoms.pbc = [True, True, False]
    span = np.ptp(atoms.positions[:, 2])
    assert roundtrip(atoms, tmp_path).cell[2, 2] == pytest.approx(span + 20.0)
    out = roundtrip(atoms, tmp_path, vacuum=30.0)
    assert out.cell[2, 2] == pytest.approx(span + 30.0)
    assert centred(out)


def test_a_slab_wrapped_across_the_face_is_made_whole(tmp_path):
    """Written by a periodic code, the slab straddles z = 0: half of it sits
    at the top of the cell. Made whole by lattice vectors, it is one layer."""
    atoms = layer(z_low=-3.0, z_high=3.0)
    atoms.set_cell([10.0, 10.0, 10.0])
    whole = atoms.positions.copy()
    atoms.wrap()                         # the file's version: split at z = 0
    atoms.pbc = [True, True, False]
    out = roundtrip(atoms, tmp_path)
    z = out.positions[:, 2]
    # extxyz keeps 8 decimals, so the round trip holds to ~1e-8.
    assert np.ptp(z) == pytest.approx(np.ptp(whole[:, 2]), abs=1e-6)
    assert centred(out)
    # Each atom moved by whole cell lengths along z, then rigidly.
    np.testing.assert_allclose(np.diff(np.sort(z)), np.diff(np.sort(whole[:, 2])),
                               atol=1e-6)


def test_a_cluster_comes_back_as_read_and_a_wire_is_refused(tmp_path):
    atoms = layer()
    atoms.pbc = False
    out = roundtrip(atoms, tmp_path)
    np.testing.assert_allclose(out.get_positions(), atoms.get_positions(), atol=1e-6)
    np.testing.assert_allclose(out.cell.array, atoms.cell.array)
    atoms.pbc = [True, False, False]
    with pytest.raises(ValueError, match="wire"):
        roundtrip(atoms, tmp_path)


def test_fixed_atoms_come_along_and_move_with_the_slab(tmp_path):
    """extxyz's move_mask comes back as FixAtoms, and centring moves the
    fixed atoms with the rest: ASE's set_positions would leave them behind."""
    atoms = layer()
    atoms.set_cell([10.0, 10.0, 0.0])
    atoms.pbc = [True, True, False]
    atoms.set_constraint(FixAtoms(indices=[0, 5]))
    out = roundtrip(atoms, tmp_path)
    assert [type(c).__name__ for c in out.constraints] == ["FixAtoms"]
    np.testing.assert_array_equal(out.constraints[0].index, [0, 5])
    shift = out.positions - atoms.positions
    np.testing.assert_allclose(shift, np.broadcast_to(shift[1], shift.shape),
                               atol=1e-6)
    assert centred(out)
