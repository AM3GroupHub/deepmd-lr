"""The trajectory converter, and the vacuum padding a slab needs.

The padding is the part worth testing. It mutates geometry, and ASE's
`SinglePointCalculator` invalidates itself when geometry changes -- so a naive
implementation silently drops the energies and forces and writes a
coordinates-only dataset. That failure is quiet: the conversion succeeds, the
files look right, and training runs on nothing.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

_SPEC = importlib.util.spec_from_file_location(
    "xyz_to_deepmd",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "xyz_to_deepmd.py",
)
x2d = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(x2d)


def slab(n=12, cell_z=32.0, seed=0):
    """A toy slab: atoms in the lower part of a tall cell, z non-periodic."""
    rng = np.random.default_rng(seed)
    pos = np.column_stack(
        [rng.uniform(0, 10, n), rng.uniform(0, 10, n), rng.uniform(0, 20, n)]
    )
    a = Atoms("H" * n, positions=pos, cell=[10.0, 10.0, cell_z], pbc=[1, 1, 0])
    a.calc = SinglePointCalculator(a, energy=-123.456, forces=rng.normal(0, 1, (n, 3)))
    return a


class TestPadVacuum:
    def test_labels_survive_the_padding(self):
        """The bug this test exists for: moving atoms invalidates the
        calculator, so energies and forces vanish and the dataset trains on
        nothing -- with no error anywhere."""
        a = slab()
        e0, f0 = a.get_potential_energy(), a.get_forces().copy()
        (padded,) = x2d.pad_vacuum([a], 52.0)
        assert padded.get_potential_energy() == pytest.approx(e0)
        assert np.allclose(padded.get_forces(), f0)

    def test_cell_is_extended_and_slab_centred(self):
        a = slab()
        (padded,) = x2d.pad_vacuum([a], 52.0)
        z = padded.get_positions()[:, 2]
        assert padded.get_cell()[2, 2] == pytest.approx(52.0)
        below, above = z.min(), 52.0 - z.max()
        assert below == pytest.approx(above, abs=1e-9), "slab is not centred"

    def test_gap_exceeds_twice_the_cutoff(self):
        """The reason padding happens at all: below 2*rcut the descriptor
        sees across the vacuum into the next image."""
        a = slab()
        (padded,) = x2d.pad_vacuum([a], 52.0)
        z = padded.get_positions()[:, 2]
        gap = padded.get_cell()[2, 2] - (z.max() - z.min())
        assert gap > 2 * 6.0

    def test_in_plane_axes_untouched(self):
        a = slab()
        before = np.array(a.get_cell())
        (padded,) = x2d.pad_vacuum([a], 52.0)
        after = np.array(padded.get_cell())
        assert np.allclose(after[:2], before[:2])

    def test_refuses_to_shrink(self):
        with pytest.raises(SystemExit, match="not larger than"):
            x2d.pad_vacuum([slab(cell_z=60.0)], 52.0)

    def test_refuses_a_slab_that_does_not_fit(self):
        a = slab()
        a.set_positions(a.get_positions() * [1, 1, 10])
        with pytest.raises(SystemExit, match="does not fit"):
            x2d.pad_vacuum([a], 52.0)


class TestConsistencyCheck:
    def test_accepts_identical_frames(self):
        assert x2d.check_consistent([slab(), slab(seed=1)]) == ["H"] * 12

    def test_rejects_a_changing_atom_count(self):
        with pytest.raises(SystemExit, match="not a single system"):
            x2d.check_consistent([slab(n=12), slab(n=13)])

    def test_rejects_a_changing_species(self):
        a, b = slab(), slab()
        b.symbols = ["He"] + list(b.symbols[1:])
        with pytest.raises(SystemExit, match="differs at atom 0"):
            x2d.check_consistent([a, b])


def molecule(symbols, charge=0, spin=1, seed=0):
    """A non-periodic molecule with a charge/spin header, in a given order."""
    rng = np.random.default_rng(seed)
    a = Atoms(symbols, pbc=False)
    n = len(a)
    a.set_positions(rng.uniform(0, 4, (n, 3)))
    a.info.update(charge=charge, spin=spin)
    a.calc = SinglePointCalculator(a, energy=-10.0 * n, forces=rng.normal(0, 1, (n, 3)))
    return a


class TestGroupByComposition:
    """A mixed file has to become one system per composition, because a
    DeePMD system carries a single `type.raw`."""

    def test_splits_on_composition(self):
        frames = [molecule("H2O"), molecule("CH4"), molecule("OH2", seed=1)]
        buckets = x2d.group_by_composition(frames)
        assert [x2d.formula(k) for k in buckets] == ["H2O1", "C1H4"]

    def test_atom_order_does_not_make_a_second_system(self):
        """OMol25 stores one composition in more than one order. Keying on the
        sorted symbols folds those together rather than writing two systems
        for one molecule."""
        buckets = x2d.group_by_composition([molecule("H2O"), molecule("OHH")])
        assert len(buckets) == 1
        assert sum(len(v) for v in buckets.values()) == 2

    def test_mixed_order_inside_one_bucket_is_canonicalized(self):
        (group,) = x2d.group_by_composition([molecule("H2O"), molecule("OHH")]).values()
        assert x2d.check_consistent(group) == ["H", "H", "O"]

    def test_charge_and_spin_do_not_split(self):
        """They vary per frame and ride in charge_spin.npy -- the same
        molecule in two charge states is still one system."""
        frames = [molecule("H2O", charge=0, spin=1), molecule("H2O", charge=-1, spin=2)]
        assert len(x2d.group_by_composition(frames)) == 1

    def test_each_bucket_passes_the_consistency_check(self):
        frames = [molecule("H2O"), molecule("CH4"), molecule("OHH")]
        for group in x2d.group_by_composition(frames).values():
            x2d.check_consistent(group)

    def test_formula_is_a_stable_directory_name(self):
        assert x2d.formula(x2d.composition_key(molecule("OHH"))) == "H2O1"
        assert x2d.formula(x2d.composition_key(molecule("H2O"))) == "H2O1"


def test_sort_preserves_labels_and_atom_arrays():
    a = molecule("OHHO", charge=-1, spin=2)
    a.set_initial_charges([0.1, 0.2, 0.3, 0.4])
    a.set_tags([10, 20, 30, 40])
    forces = np.arange(12.0).reshape(4, 3)
    charges = np.arange(4.0)
    stress = np.arange(6.0)
    a.calc = SinglePointCalculator(
        a, energy=-7.0, forces=forces, charges=charges, stress=stress
    )
    order = [1, 2, 0, 3]
    result = x2d.sort_frame(a)
    assert result.info == a.info
    assert result.get_potential_energy() == -7.0
    np.testing.assert_array_equal(result.positions, a.positions[order])
    np.testing.assert_array_equal(result.get_forces(), forces[order])
    np.testing.assert_array_equal(result.get_charges(), charges[order])
    np.testing.assert_array_equal(result.get_stress(), stress)
    np.testing.assert_array_equal(result.get_tags(), a.get_tags()[order])
    np.testing.assert_array_equal(
        result.get_initial_charges(), a.get_initial_charges()[order]
    )
    assert a.get_chemical_symbols() == ["O", "H", "H", "O"]
    np.testing.assert_array_equal(a.get_forces(), forces)


def test_cli_mixed_order_conversion_and_safe_rerun(tmp_path, monkeypatch):
    frames = [molecule("OH2", charge=-1, spin=2), molecule("HOH", seed=1)]
    monkeypatch.setattr(x2d, "load_frames", lambda *args: frames)
    out = tmp_path / "data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "xyz_to_deepmd",
            "unused.xyz",
            "-o",
            str(out),
            "--split-by-composition",
            "--no-split",
        ],
    )
    x2d.main()
    system = out / "H2O1"
    # The default type_map is the whole periodic table, indexed so that a type
    # IS the atomic number: H is 1 and O is 8 here, and in every dataset.
    assert (system / "type.raw").read_text().split() == ["1", "1", "8"]
    saved_forces = np.load(system / "set.000/force.npy")
    for i, atoms in enumerate(frames):
        order = np.argsort(atoms.get_chemical_symbols(), kind="stable")
        np.testing.assert_array_equal(
            saved_forces[i], atoms.get_forces()[order].reshape(-1)
        )
    np.testing.assert_array_equal(
        np.load(system / "set.000/charge_spin.npy"), [[-1.0, 2.0], [0.0, 1.0]]
    )
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit, match="not an empty directory"):
        x2d.main()
    assert before == {
        p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()
    }


def test_writer_refuses_stale_boundary_and_charge_files(tmp_path):
    coord = np.zeros((1, 3))
    charged = (coord, np.zeros(1), coord, None, np.array([[1.0, 2.0]]))
    x2d.write_system(tmp_path, np.array([0]), charged, [0], ["H"], False)
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    neutral_periodic = (coord, None, None, np.eye(3).reshape(1, 9), None)
    with pytest.raises(SystemExit, match="not an empty directory"):
        x2d.write_system(tmp_path, np.array([0]), neutral_periodic, [0], ["H"], True)
    assert before == {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }


def test_split_collision_refused_before_writing(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train/type.raw").write_text("0\n")
    args = SimpleNamespace(
        vacuum=None, no_split=False, valid_fraction=0.5, seed=0, input="x"
    )
    with pytest.raises(SystemExit, match="not an empty directory"):
        x2d.convert_one([molecule("H2O"), molecule("H2O")], tmp_path, args)
    assert not (tmp_path / "valid").exists()


class TestPeriodicTableDefault:
    """The default `type_map` is the whole periodic table.

    A per-dataset map makes the same integer mean different elements in
    different directories, which is how a dataset becomes unusable with any
    model but the one it was converted for. The cost is a coupling: DeePMD-kit
    refuses an element that a system's map has and the model's does not
    (`deepmd/utils/data.py:107`), so the model must carry the full table too.
    """

    def test_a_type_index_is_the_atomic_number(self):
        """The point of the layout. Index 0 is ASE's dummy so that H lands on
        1 rather than 0 -- otherwise every index is one less than the number
        anyone actually uses, which is a subtraction waiting to be forgotten.

        Pinned against ASE rather than derived from it: `type.raw` is data on
        disk, and its indices must not shift if a dependency reorders a list.
        """
        from ase.data import atomic_numbers, chemical_symbols

        assert list(x2d.PERIODIC_TABLE) == chemical_symbols
        assert len(x2d.PERIODIC_TABLE) == 119
        assert x2d.PERIODIC_TABLE[0] == "X"
        for element in ("H", "C", "O", "Fe", "Og"):
            assert x2d.PERIODIC_TABLE.index(element) == atomic_numbers[element]

    def test_the_dummy_slot_resolves_through_ase(self):
        """"X", not "NULL": deepmd_lr's own `atomic_masses` builds a mass for
        every type_map entry through `ase.data`, so slot 0 has to be a name ASE
        knows or MD setup raises on a model it could otherwise run."""
        from ase.data import atomic_masses, atomic_numbers

        assert atomic_masses[atomic_numbers[x2d.PERIODIC_TABLE[0]]] is not None

    def test_a_dummy_atom_in_the_data_is_refused(self, tmp_path, monkeypatch):
        """Slot 0 exists to be unused. Now that "X" is in the table, a frame
        carrying one would otherwise be quietly assigned to it."""
        monkeypatch.setattr(x2d, "load_frames", lambda *a: [Atoms("XH")])
        monkeypatch.setattr(
            "sys.argv",
            ["xyz_to_deepmd", "unused.xyz", "-o", str(tmp_path / "d"), "--no-split"],
        )
        with pytest.raises(SystemExit, match="cannot be given a type"):
            x2d.main()

    def _convert(self, tmp_path, monkeypatch, *flags):
        frames = [molecule("HOH", seed=1)]
        monkeypatch.setattr(x2d, "load_frames", lambda *args: frames)
        out = tmp_path / "data"
        monkeypatch.setattr(
            "sys.argv",
            ["xyz_to_deepmd", "unused.xyz", "-o", str(out), "--no-split", *flags],
        )
        x2d.main()
        return out

    def test_full_is_the_default(self, tmp_path, monkeypatch):
        out = self._convert(tmp_path, monkeypatch)
        written = (out / "type_map.raw").read_text().split()
        assert written == list(x2d.PERIODIC_TABLE)
        # Atoms keep their file order (H O H), so the indices are the atomic
        # numbers in that order -- sorting happens only under
        # --split-by-composition.
        assert (out / "type.raw").read_text().split() == ["1", "8", "1"]

    def test_present_keeps_the_old_per_dataset_map(self, tmp_path, monkeypatch):
        out = self._convert(tmp_path, monkeypatch, "--type-map", "present")
        assert (out / "type_map.raw").read_text().split() == ["H", "O"]
        assert (out / "type.raw").read_text().split() == ["0", "1", "0"]

    def test_the_two_modes_describe_the_same_atoms(self, tmp_path, monkeypatch):
        """Different integers, same elements -- the point of the remap."""
        full = self._convert(tmp_path / "a", monkeypatch)
        present = self._convert(tmp_path / "b", monkeypatch, "--type-map", "present")
        named = []
        for system in (full, present):
            table = (system / "type_map.raw").read_text().split()
            named.append([table[int(i)] for i in
                          (system / "type.raw").read_text().split()])
        assert named[0] == named[1] == ["H", "O", "H"]
