"""The trajectory converter, and the vacuum padding a slab needs.

The padding is the part worth testing. It mutates geometry, and ASE's
`SinglePointCalculator` invalidates itself when geometry changes -- so a naive
implementation silently drops the energies and forces and writes a
coordinates-only dataset. That failure is quiet: the conversion succeeds, the
files look right, and training runs on nothing.
"""

import importlib.util
import pathlib

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

    def test_mixed_order_inside_one_bucket_is_still_refused(self):
        """Folding the orders together is not the same as fixing them: one
        `type.raw` describes a whole system, so the frames must be sorted
        before conversion. The refusal is what says so."""
        (group,) = x2d.group_by_composition([molecule("H2O"), molecule("OHH")]).values()
        with pytest.raises(SystemExit, match="differs at atom 0"):
            x2d.check_consistent(group)
        x2d.check_consistent([_sorted(a) for a in group])

    def test_charge_and_spin_do_not_split(self):
        """They vary per frame and ride in charge_spin.npy -- the same
        molecule in two charge states is still one system."""
        frames = [molecule("H2O", charge=0, spin=1), molecule("H2O", charge=-1, spin=2)]
        assert len(x2d.group_by_composition(frames)) == 1

    def test_each_bucket_passes_the_consistency_check(self):
        frames = [molecule("H2O"), molecule("CH4"), molecule("OHH")]
        for group in x2d.group_by_composition(frames).values():
            x2d.check_consistent([_sorted(a) for a in group])

    def test_formula_is_a_stable_directory_name(self):
        assert x2d.formula(x2d.composition_key(molecule("OHH"))) == "H2O1"
        assert x2d.formula(x2d.composition_key(molecule("H2O"))) == "H2O1"


def _sorted(atoms):
    order = np.argsort(atoms.get_atomic_numbers(), kind="stable")
    return atoms[order]
