"""Phase A: partitioned evaluation must reproduce the serial answer.

The claim is partition-count invariance: 1, 2, 4 and 8 partitions must all
agree with the undivided model, to the floor set by GPU nondeterminism. This
is precisely the test the LAMMPS domain decomposition failed -- splitting a
reciprocal sum across ranks gave 2.0% error at 2 ranks and 36.5% at 8 --
so here it has to pass exactly, not approximately.

Phase A covers the short-range model; Phase B adds the long-range term,
gathered onto one device.
"""

import copy
import math
import os

import numpy as np
import pytest
import torch
from ase import Atoms
from ase import units as ase_units
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.nvalchemi import (
    DistributedEvaluator,
    evaluate_partitioned,
    partition_graph,
    partition_system,
    to_numpy,
)
from deepmd_lr.nvalchemi.partition import default_skin

from .test_joint_model import DESCRIPTOR, FITTING, LONG_RANGE, build, frame, get_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)

RCUT = 6.0
PARTITION_COUNTS = [1, 2, 4, 8]


@pytest.fixture(scope="module")
def system():
    """A periodic box big enough that partitions are non-trivial."""
    coord, atype, cell = frame(n_mol=16, box=14.0)
    return (
        coord[0].cpu().numpy(),
        atype[0].cpu().numpy(),
        cell[0].cpu().numpy(),
        coord,
        atype,
        cell,
    )


class TestPartitioner:
    def test_owned_sets_tile_the_system(self, system):
        """Every atom owned exactly once -- no gaps, no duplicates."""
        coord, _, cell, *_ = system
        for n in PARTITION_COUNTS:
            owned = np.concatenate(
                [p.owned for p in partition_system(coord, cell, RCUT, n)]
            )
            assert np.array_equal(np.sort(owned), np.arange(len(coord)))

    def test_halo_contains_every_neighbour(self, system):
        """The halo must be complete: for each owned atom, every atom within
        rcut (under any periodic image) has to be present as a node. A halo
        one atom too narrow silently changes the forces."""
        coord, _, cell, *_ = system
        inverse = np.linalg.inv(cell)
        for part in partition_system(coord, cell, RCUT, 4):
            nodes = part.coordinates(coord)
            for atom in part.owned:
                delta = coord[atom] - coord
                frac = delta @ inverse
                delta = (frac - np.round(frac)) @ cell
                within = np.flatnonzero(np.linalg.norm(delta, axis=1) < RCUT)
                for neighbour in within:
                    separation = np.linalg.norm(
                        nodes - coord[atom], axis=1
                    )
                    present = np.any(
                        (separation < RCUT + 1e-6)
                        & (part.global_ids() == neighbour)
                    )
                    assert present, (
                        f"atom {neighbour} is within rcut of owned atom "
                        f"{atom} but is absent from the partition"
                    )

    def test_single_partition_still_has_periodic_ghosts(self, system):
        """One partition owns everything but is not a special case.

        The subgraph is built non-periodically, so periodic images still have
        to appear as ghosts. Short-circuiting this dropped every interaction
        across the boundary and cost 0.4 eV on a 48-atom box.
        """
        coord, _, cell, *_ = system
        (part,) = partition_system(coord, cell, RCUT, 1)
        assert part.n_owned == len(coord)
        assert len(part.ghost) > 0

    def test_every_edge_belongs_to_exactly_one_partition(self, system):
        """The invariant that makes the assembled sum exact.

        A subgraph keeps only edges whose destination it owns, so summing
        over partitions counts each edge once. Keeping ghost-to-ghost edges
        would double-count them.
        """
        coord, atype, cell, *_ = system
        seen = {}
        for n in PARTITION_COUNTS:
            total = 0
            for part in partition_system(coord, cell, RCUT, n):
                graph, _, _ = partition_graph(part, coord, atype, RCUT, DEVICE)
                total += int(graph.edge_mask.sum())
            seen[n] = total
        assert len(set(seen.values())) == 1, (
            f"edge count depends on partitioning: {seen}"
        )

    @pytest.mark.parametrize("n", [0, -1])
    def test_rejects_bad_partition_counts(self, system, n):
        coord, _, cell, *_ = system
        with pytest.raises(ValueError, match="n_partitions"):
            partition_system(coord, cell, RCUT, n)

    def test_rejects_more_partitions_than_atoms(self, system):
        coord, _, cell, *_ = system
        with pytest.raises(ValueError, match="only .* atoms"):
            partition_system(coord, cell, RCUT, len(coord) + 1)


class TestHaloMatchesABallQuery:
    """The rebuild's ghost set is exactly "every image within the halo of an
    owned atom" -- the definition, computed here the slow way as a reference.

    The rebuild used to compute it that way too, with a ball query returning a
    Python list of neighbours per owned atom; at 98,304 atoms that was 2.3 s of
    a serial host step. It now filters images through the owned atoms'
    fractional bounding box grown by the halo, then keeps those whose nearest
    owned atom is in range. Same set, same order, byte for byte.
    """

    @staticmethod
    def reference(coord, cell, rcut, parts, pbc, skin):
        from itertools import product

        from scipy.spatial import cKDTree

        mine = partition_system(coord, cell, rcut, parts, pbc, skin=skin)
        # The definition applies to the atoms folded into the cell: an image of
        # an atom is the same atom however far it has wandered, and the nodes
        # are then expressed back relative to the coordinates handed in.
        n = np.where(np.asarray(pbc), np.floor(coord @ np.linalg.inv(cell)), 0.0)
        fold = n @ cell
        coord = coord - fold
        halo = rcut + skin
        volume = abs(np.linalg.det(cell))
        ranges = []
        for axis in range(3):
            if not pbc[axis]:
                ranges.append([0])
                continue
            other = [cell[i] for i in range(3) if i != axis]
            n = int(np.ceil(halo / (volume / np.linalg.norm(np.cross(*other)))))
            ranges.append(list(range(-n, n + 1)))
        offsets = np.array(list(product(*ranges)), dtype=int) @ cell
        images = (coord[None] + offsets[:, None]).reshape(-1, 3)
        atom = np.tile(np.arange(len(coord)), len(offsets))
        shift = np.repeat(offsets, len(coord), axis=0)
        tree = cKDTree(images)
        for part in mine:
            near = tree.query_ball_point(coord[part.owned], r=halo)
            cand = np.unique(np.concatenate([np.asarray(x, dtype=int) for x in near]))
            self_copy = np.isin(atom[cand], part.owned) & (
                np.abs(shift[cand]).sum(axis=1) == 0
            )
            ghosts = cand[~self_copy]
            np.testing.assert_array_equal(part.ghost, atom[ghosts])
            np.testing.assert_array_equal(
                part.ghost_shift, shift[ghosts] - fold[atom[ghosts]]
            )
            np.testing.assert_array_equal(part.owned_shift, -fold[part.owned])

    @pytest.mark.parametrize("parts", [1, 2, 3])
    @pytest.mark.parametrize("skin", [0.0, 2.0])
    def test_periodic_box(self, system, parts, skin):
        coord, _atype, cell, *_ = system
        self.reference(np.asarray(coord), np.asarray(cell), RCUT, parts,
                       (True, True, True), skin)

    @pytest.mark.parametrize("parts", [1, 2, 3])
    def test_unwrapped_box(self, system, parts):
        """Atoms several cell lengths out, as an integrator leaves them."""
        coord, _atype, cell, *_ = system
        moved = TestUnwrappedCoordinates.shifted(np.asarray(coord), np.asarray(cell))
        self.reference(moved, np.asarray(cell), RCUT, parts, (True, True, True), 2.0)

    def test_unwrapped_drifted_coordinates(self, system):
        """MD positions leave the cell; the bounding box must follow them."""
        coord, _atype, cell, *_ = system
        rng = np.random.default_rng(0)
        drifted = np.asarray(coord) + rng.normal(0, 3.0, np.shape(coord)) + [30, -20, 5]
        self.reference(drifted, np.asarray(cell), RCUT, 2, (True, True, True), 1.0)

    def test_triclinic_cell(self, system):
        coord, _atype, cell, *_ = system
        cell = np.asarray(cell, dtype=float)
        tilted = cell.copy()
        tilted[1, 0] += 0.4 * cell[0, 0]
        tilted[2, 1] += 0.3 * cell[1, 1]
        coord = np.asarray(coord) @ np.linalg.inv(cell) @ tilted
        self.reference(coord, tilted, RCUT, 2, (True, True, True), 1.5)

    def test_non_periodic_axis(self, system):
        coord, _atype, cell, *_ = system
        self.reference(np.asarray(coord), np.asarray(cell), RCUT, 2,
                       (True, True, False), 1.0)


class TestPartitionCountInvariance:
    """The Phase A gate."""

    @staticmethod
    def _noise_floor(model, coord, atype, cell):
        """Evaluating one model twice already differs; measure by how much.

        Comparing against a fixed tolerance would either be too loose in
        float64 or fail spuriously in float32, where this floor is ~1e-5 eV.
        """
        a = model(coord, atype, box=cell)
        b = model(coord, atype, box=cell)
        return (
            abs(float(a["energy"]) - float(b["energy"])),
            float((a["force"] - b["force"]).abs().max()),
        )

    def test_energy_and_force_match_serial(self, system):
        coord, atype, cell, coord_t, atype_t, cell_t = system
        model = build(long_range=False)
        model.eval()

        serial = model(coord_t, atype_t, box=cell_t)
        energy_ref = float(serial["energy"])
        force_ref = serial["force"][0].detach().cpu().numpy()
        d_energy, d_force = self._noise_floor(model, coord_t, atype_t, cell_t)

        for n in PARTITION_COUNTS:
            out = to_numpy(evaluate_partitioned(
                model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
            ))
            assert abs(out["energy"] - energy_ref) <= max(d_energy * 10, 1e-9), (
                f"{n} partitions: energy off by "
                f"{abs(out['energy'] - energy_ref):.3e}"
            )
            assert np.abs(out["force"] - force_ref).max() <= max(
                d_force * 10, 1e-9
            ), f"{n} partitions: force off by " \
               f"{np.abs(out['force'] - force_ref).max():.3e}"

    def test_atom_energies_match_serial(self, system):
        """Per-atom, not just the total -- a total can hide compensating errors."""
        coord, atype, cell, coord_t, atype_t, cell_t = system
        model = build(long_range=False)
        model.eval()
        reference = (
            model(coord_t, atype_t, box=cell_t)["atom_energy"]
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
        )
        out = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        ))
        assert np.abs(out["atom_energy"] - reference).max() < 1e-8

    def test_rejects_device_count_mismatch(self, system):
        coord, atype, cell, *_ = system
        model = build(long_range=False)
        with pytest.raises(ValueError, match="one device per partition"):
            to_numpy(evaluate_partitioned(
                model, coord, atype, cell, n_partitions=4, devices=[DEVICE]
            ))


class TestSkinDefault:
    """The skin is a cost knob, and a constant is the wrong setting for it.

    Measured on two A40s at 98,304 atoms: skin 1.0 spent 76 % of every step
    rebuilding the decomposition and capped the two-GPU speed-up at 1.14x;
    skin 4.0 ran 3.4x faster and lifted the ceiling to 1.9x. On a couple of
    thousand atoms the trade reverses. So the default scales with geometry.
    """

    def test_auto_grows_with_the_cell_and_shrinks_with_partitions(self):
        small = np.diag([24.86] * 3)
        large = np.diag([99.43] * 3)
        assert default_skin(small, 6.0, 1) < default_skin(large, 6.0, 1)
        assert default_skin(large, 6.0, 4) < default_skin(large, 6.0, 1)

    def test_auto_matches_the_measured_optimum_at_98k_atoms(self):
        """The 99.43 A box is the 98,304-atom system the profile was run on."""
        assert default_skin(np.diag([99.43] * 3), 6.0, 1) == pytest.approx(4.3, abs=0.4)

    @pytest.fixture(scope="class")
    def model(self):
        m = build(long_range=False)
        m.eval()
        return m

    def test_halo_inflation_stays_within_the_tolerance(self):
        from deepmd_lr.nvalchemi.partition import decomposition_grid

        cell = np.diag([99.43] * 3)
        for parts in (1, 2, 4, 8):
            skin = default_skin(cell, 6.0, parts, tolerance=0.25)
            edges = np.array([99.43] * 3) / np.array(
                decomposition_grid(cell, parts, 6.0), dtype=float
            )
            base = edges + 12.0
            assert float(np.prod((base + 2 * skin) / base)) <= 1.25 + 1e-9

    def test_bracket_grows_for_an_anisotropic_cell(self):
        """Half the shortest edge is not a bracket, only a guess.

        With edges [100, 100, 1] even a skin of 0.5 stays far inside the
        tolerance, so a fixed upper bound would return that cap instead of the
        largest admissible skin.
        """
        cell = np.diag([100.0, 100.0, 1.0])
        skin = default_skin(cell, 6.0, 1)
        assert skin > 0.5 * 1.0
        base = np.array([100.0, 100.0, 1.0]) + 12.0
        assert float(np.prod((base + 2 * skin) / base)) == pytest.approx(1.25, abs=1e-6)

    def test_auto_follows_the_cell_through_set_cell(self, model, system):
        """A barostat replaces the cell; the rule reads the cell."""
        _coord, atype, cell, *_ = system
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        before = ev.skin
        ev.set_cell(cell * 2.0)
        assert ev.skin > before
        assert ev.skin == pytest.approx(default_skin(cell * 2.0, ev._rcut, 1))
        ev.set_cell(cell)
        assert ev.skin == pytest.approx(before)

    def test_an_explicit_skin_is_not_touched_by_set_cell(self, model, system):
        _coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE], skin=2.5)
        ev.set_cell(cell * 2.0)
        assert ev.skin == 2.5

    def test_rejects_an_unknown_string(self, model, system):
        _coord, atype, cell, *_ = system
        with pytest.raises(ValueError, match="number or 'auto'"):
            DistributedEvaluator(model, atype, cell, devices=[DEVICE], skin="wide")


class TestSkinIsFittedToMemory:
    """The halo is chosen from a measurement, not from a guess about bytes.

    The geometry rule knows how many nodes a skin costs. What a node costs in
    bytes belongs to the model and spans a factor of thirty: 755 kB per node
    for an uncompressed long-range model against 24.3 kB compressed, measured
    on this machine. So the first evaluation is a probe at the floor skin, and
    the skin the rest of the run uses is the largest one whose extrapolated
    footprint fits the budget -- never more than the geometry rule asks for.
    """

    @pytest.fixture(scope="class")
    def model(self):
        m = build(long_range=False)
        m.eval()
        return m

    @staticmethod
    def _pin_memory(monkeypatch, peak, free, reserved=0):
        """Fix the three device numbers the fit reads."""
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: peak)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: reserved)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (free, free))

    def test_the_probe_runs_at_the_floor_then_the_rule_decides(self, system, model):
        from deepmd_lr.nvalchemi.evaluator import MIN_AUTO_SKIN

        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE])
        assert ev.skin == pytest.approx(MIN_AUTO_SKIN)
        ev(coord)
        # This device has room to spare, so the geometry rule still decides.
        assert ev.skin == pytest.approx(default_skin(cell, ev._rcut, 1))
        assert ev.bytes_per_node is not None and ev.bytes_per_node > 0
        rebuilds = ev.rebuild_count
        ev(coord)
        assert ev.rebuild_count == rebuilds + 1, "the probe halo must be replaced"

    def test_a_budget_that_does_not_fit_caps_the_halo(self, system, model):
        from deepmd_lr.nvalchemi.evaluator import MIN_AUTO_SKIN

        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=1e-6
        )
        with pytest.warns(RuntimeWarning, match="does not fit"):
            ev(coord)
        assert ev.skin == pytest.approx(MIN_AUTO_SKIN)
        assert ev.skin < default_skin(cell, ev._rcut, 1)

    def test_the_fitted_skin_is_the_arithmetic_it_claims(
        self, system, model, monkeypatch
    ):
        """A budget of twice the probe's footprint buys twice the nodes."""
        from deepmd_lr.nvalchemi.evaluator import MIN_AUTO_SKIN
        from deepmd_lr.nvalchemi.partition import (
            halo_node_ratio,
            skin_for_node_ratio,
        )

        coord, atype, cell, *_ = system
        peak, free = 8 * 2**30, 20 * 2**30
        headroom = 0.8
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=headroom
        )
        self._pin_memory(monkeypatch, peak=peak, free=free)
        ev(coord)
        ratio = headroom * free / peak
        expected = min(
            skin_for_node_ratio(cell, ev._rcut, 1, ratio, skin_from=MIN_AUTO_SKIN),
            default_skin(cell, ev._rcut, 1),
        )
        assert ev.skin == pytest.approx(expected, rel=1e-6)
        # and the prediction that skin encodes: at most `ratio` times the
        # nodes the probe actually counted.
        got = halo_node_ratio(cell, ev._rcut, 1, ev.skin, MIN_AUTO_SKIN)
        assert got <= ratio + 1e-9

    def test_a_tight_budget_binds_before_the_geometry_rule(
        self, system, model, monkeypatch
    ):
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE])
        self._pin_memory(monkeypatch, peak=8 * 2**30, free=int(8.5 * 2**30))
        with pytest.warns(RuntimeWarning, match="does not fit"):
            ev(coord)
        assert ev.skin < default_skin(cell, ev._rcut, 1)

    def test_none_skips_the_probe_entirely(self, system, model):
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        assert ev.skin == pytest.approx(default_skin(cell, ev._rcut, 1))
        ev(coord)
        assert ev.bytes_per_node is None
        assert ev.skin == pytest.approx(default_skin(cell, ev._rcut, 1))

    def test_an_explicit_skin_is_never_probed(self, system, model):
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE], skin=2.5)
        assert ev.skin == 2.5
        ev(coord)
        assert ev.skin == 2.5
        assert ev.bytes_per_node is None

    def test_the_probe_changes_no_numbers(self, system, model):
        """A halo only has to cover the cutoff; its thickness is a cost knob."""
        coord, atype, cell, *_ = system
        plain = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        want_e, want_f = plain(coord)
        want_f = want_f.clone()
        probed = DistributedEvaluator(model, atype, cell, devices=[DEVICE])
        probed(coord)
        energy, forces = probed(coord)
        assert energy == pytest.approx(want_e, abs=1e-9)
        np.testing.assert_allclose(
            forces.cpu().numpy(), want_f.cpu().numpy(), atol=1e-10
        )

    def test_a_nonsensical_headroom_is_refused(self, system, model):
        _coord, atype, cell, *_ = system
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="memory_headroom"):
                DistributedEvaluator(
                    model, atype, cell, devices=[DEVICE], memory_headroom=bad
                )


class TestSkinBacksOffWhenMemoryRunsOut:
    """The halo is the one part of the footprint this class chooses.

    The geometry rule sizes it blind to bytes: the same 25 % of extra nodes
    whether a node costs a compressed descriptor's memory or an uncompressed
    one's several times that. At 98,304 atoms on eight devices that was the
    difference between running and an out-of-memory, where the old fixed 1 A
    skin fitted. Predicting the footprint means guessing bytes per node for a
    model this class never inspects; reacting to it does not.
    """

    @pytest.fixture(scope="class")
    def model(self):
        m = build(long_range=False)
        m.eval()
        return m

    @staticmethod
    def _fail_n_times(monkeypatch, failures, error=None):
        """Make the first ``failures`` evaluations report an out-of-memory."""
        from deepmd_lr.nvalchemi import evaluator as ev_mod

        real = ev_mod.evaluate_partitioned
        left = {"n": failures}

        def fake(*args, **kwargs):
            if left["n"]:
                left["n"] -= 1
                raise error or torch.cuda.OutOfMemoryError("CUDA out of memory")
            return real(*args, **kwargs)

        monkeypatch.setattr(ev_mod, "evaluate_partitioned", fake)
        return left

    def test_the_halo_shrinks_and_the_step_still_completes(
        self, system, model, monkeypatch
    ):
        coord, atype, cell, *_ = system
        reference = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        want_energy, want_forces = reference(coord)
        want_forces = want_forces.clone()

        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        before = ev.skin
        self._fail_n_times(monkeypatch, 1)
        with pytest.warns(RuntimeWarning, match="out of memory"):
            energy, forces = ev(coord)
        assert ev.skin == pytest.approx(0.5 * before)
        assert ev.skin_backoffs == 1
        assert energy == pytest.approx(want_energy, abs=1e-9)
        np.testing.assert_allclose(
            forces.cpu().numpy(), want_forces.cpu().numpy(), atol=1e-10
        )

    def test_it_keeps_halving_until_it_fits(self, system, model, monkeypatch):
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        before = ev.skin
        self._fail_n_times(monkeypatch, 3)
        with pytest.warns(RuntimeWarning):
            ev(coord)
        assert ev.skin == pytest.approx(before / 8)
        assert ev.skin_backoffs == 3

    def test_the_new_ceiling_survives_a_barostat(self, system, model, monkeypatch):
        """`set_cell` re-runs the geometry rule, which must not climb back."""
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        self._fail_n_times(monkeypatch, 1)
        with pytest.warns(RuntimeWarning):
            ev(coord)
        capped = ev.skin
        ev.set_cell(cell * 2.0)
        assert ev.skin == pytest.approx(capped)
        assert default_skin(cell * 2.0, ev._rcut, 1) > capped

    def test_an_explicit_skin_is_the_callers_and_is_not_touched(
        self, system, model, monkeypatch
    ):
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE], skin=2.5)
        self._fail_n_times(monkeypatch, 1)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            ev(coord)
        assert ev.skin == 2.5
        assert ev.skin_backoffs == 0

    def test_the_floor_ends_the_backoff_rather_than_looping(
        self, system, model, monkeypatch
    ):
        """Below the floor the halo is not what is filling the device."""
        from deepmd_lr.nvalchemi.evaluator import MIN_AUTO_SKIN

        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        self._fail_n_times(monkeypatch, 10**6)
        with (
            pytest.warns(RuntimeWarning, match="floor"),
            pytest.raises(torch.cuda.OutOfMemoryError),
        ):
            ev(coord)
        assert ev.skin == pytest.approx(MIN_AUTO_SKIN)

    def test_any_other_failure_is_not_swallowed(self, system, model, monkeypatch):
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        before = ev.skin
        self._fail_n_times(monkeypatch, 1, error=ValueError("not a memory problem"))
        with pytest.raises(ValueError, match="not a memory problem"):
            ev(coord)
        assert ev.skin == before
        assert ev.skin_backoffs == 0

    def test_a_plain_runtime_error_about_memory_counts(
        self, system, model, monkeypatch
    ):
        """warp and the allocator behind the search raise a bare RuntimeError."""
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], memory_headroom=None
        )
        self._fail_n_times(
            monkeypatch, 1, error=RuntimeError("Failed to allocate 4 GiB on cuda:0")
        )
        with pytest.warns(RuntimeWarning):
            ev(coord)
        assert ev.skin_backoffs == 1


class TestGatherIsOnTheDevice:
    """One assembly path, and it ends on the gather device.

    The per-partition results used to be pulled to the host and scatter-added
    with `np.add.at`, which synchronised every device once per partition per
    step. Now `index_add_` runs on the gather device and the result is handed
    back as tensors; the only download is the caller's explicit `to_numpy`.
    Pinned here because a well-meaning `.cpu()` anywhere in the gather would
    silently restore the sync while every other test still passed.
    """

    @pytest.fixture(scope="class")
    def model(self):
        m = build(long_range=True)
        m.eval()
        return m

    def test_results_are_tensors_on_the_gather_device(self, system, model):
        coord, atype, cell, *_ = system
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=2,
            devices=[DEVICE] * 2, compute_born=True, compute_virial=False,
        )
        expected = torch.device(DEVICE)
        for key in ("force", "atom_energy", "charge", "born_charge",
                    "force_long_range", "force_charge_response", "dE_dq"):
            value = out[key]
            assert isinstance(value, torch.Tensor), f"{key} is {type(value).__name__}"
            assert value.device.type == expected.type, f"{key} on {value.device}"
        assert out["energy"].ndim == 0
        assert out["force"].shape == (len(coord), 3)

    def test_to_numpy_converts_arrays_and_scalars(self, system, model):
        coord, atype, cell, *_ = system
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=2, devices=[DEVICE] * 2
        )
        host = to_numpy(out)
        assert isinstance(host["force"], np.ndarray)
        assert isinstance(host["energy"], float)
        assert host["energy"] == pytest.approx(float(out["energy"]))
        np.testing.assert_allclose(host["force"], out["force"].cpu().numpy())

    def test_the_gather_never_calls_numpy_scatter(self, system, model, monkeypatch):
        """`np.add.at` in the gather is the regression this class exists for."""
        import numpy as real_np

        def refuse(*args, **kwargs):
            raise AssertionError("the gather scattered on the host")

        monkeypatch.setattr(real_np.add, "at", refuse)
        coord, atype, cell, *_ = system
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=2,
            devices=[DEVICE] * 2, compute_born=True,
        )
        assert torch.isfinite(out["force"]).all()


class TestGatheredLongRange:
    """Phase B: the reciprocal term is gathered, not decomposed.

    Splitting a reciprocal sum spatially does not work -- measured at 2.0%
    error on 2 ranks and 36.5% on 8 in a LAMMPS domain decomposition. So it
    is evaluated once on
    one device over every atom. The charges feeding it still come from the
    partitioned descriptor, and the total-charge projection is applied after
    the gather because it constrains a sum over all atoms.
    """

    @pytest.fixture(scope="class")
    def joint(self, system):
        model = build(long_range=True)
        model.eval()
        coord, atype, cell, coord_t, atype_t, cell_t = system
        return model, model(coord_t, atype_t, box=cell_t)

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_charges_match_serial(self, system, joint, n):
        """Charges are local, so partitioning must not move them at all."""
        coord, atype, cell, *_ = system
        model, serial = joint
        reference = serial["charge"].reshape(-1).detach().cpu().numpy()
        out = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        ))
        assert np.abs(out["charge"] - reference).max() < 1e-12

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_total_charge_constraint_survives_partitioning(self, system, joint, n):
        """The constraint is global; a per-partition projection would break it."""
        coord, atype, cell, *_ = system
        model, _ = joint
        out = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        ))
        assert abs(out["charge"].sum()) < 1e-10

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_long_range_energy_matches_serial(self, system, joint, n):
        coord, atype, cell, *_ = system
        model, serial = joint
        reference = float(serial["energy_long_range"].reshape(-1)[0])
        out = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        ))
        assert abs(out["energy_long_range"] - reference) < 1e-10

    def test_direct_long_range_force_matches_serial(self, system, joint):
        """The piece Phase B is responsible for: -dE/dr at frozen charge."""
        coord, atype, cell, coord_t, _, cell_t = system
        model, serial = joint
        frozen = serial["charge"].detach()
        positions = coord_t.clone().requires_grad_(True)
        energy = model.long_range_energy(positions, frozen, cell_t).sum()
        reference = (
            -torch.autograd.grad(energy, positions)[0][0].cpu().numpy()
        )
        out = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        ))
        assert np.abs(out["force_long_range"] - reference).max() < 1e-12

    def test_charge_gradient_is_available_and_nonzero(self, system, joint):
        """dE/dq is what Phase C backpropagates; taking it out as a value is
        what keeps the autograd graph on a single device."""
        coord, atype, cell, *_ = system
        model, _ = joint
        out = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        ))
        assert out["dE_dq"].shape == (len(coord),)
        assert np.isfinite(out["dE_dq"]).all()
        assert np.abs(out["dE_dq"]).max() > 1e-6

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_full_force_matches_serial(self, system, joint, n):
        """Phase C closes the loop: force, charge response included.

        Before the charge-response term the gap to serial was 3.1e-2 eV/A --
        about 12x the direct long-range force it accompanies, so this is a
        correctness requirement rather than a refinement.
        """
        coord, atype, cell, *_ = system
        model, serial = joint
        reference = serial["force"][0].detach().cpu().numpy()
        out = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        ))
        assert np.abs(out["force"] - reference).max() < 1e-12

    def test_charge_response_dominates_the_direct_term(self, system, joint):
        """Guard against it being quietly dropped or scaled to nothing.

        It is not a small correction here: ~7e-3 eV/A rms against ~2.5e-3 for
        the direct long-range force. A regression that zeroed it would still
        leave energies right and forces plausible.
        """
        coord, atype, cell, *_ = system
        model, _ = joint
        out = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        ))
        response = np.sqrt((out["force_charge_response"] ** 2).mean())
        direct = np.sqrt((out["force_long_range"] ** 2).mean())
        assert response > direct

    @pytest.mark.parametrize("n", [1, 4])
    def test_assembled_force_is_the_gradient_of_the_assembled_energy(
        self, system, joint, n
    ):
        """Finite-difference the partitioned pipeline against itself.

        Stronger than agreeing with the serial model: it checks the assembled
        force really is -dE/dr of the assembled energy, so a charge-response
        term with the wrong sign or a missing ghost contribution shows up even
        if the serial reference were also wrong.
        """
        coord, atype, cell, *_ = system
        model, _ = joint

        def energy(positions):
            return to_numpy(evaluate_partitioned(
                model, positions, atype, cell,
                n_partitions=n, devices=[DEVICE] * n,
            ))["energy"]

        force = to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        ))["force"]

        step = 1e-5
        for atom in range(0, len(coord), 11):
            for axis in range(3):
                plus, minus = coord.copy(), coord.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                numeric = -(energy(plus) - energy(minus)) / (2 * step)
                assert abs(numeric - force[atom, axis]) < 1e-6


    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_assembled_virial_is_the_strain_derivative(self, system, joint, n):
        """Finite-difference the partitioned virial against itself.

        The companion to the force check above, and the same argument: the
        distributed virial is otherwise only validated transitively -- equal
        to the serial virial, which is equal to a finite difference. That
        chain would hide an error common to both. This differentiates the
        assembled energy directly.

        It is the most intricate piece of the partitioned path: three
        contributions summed, a strain leaf per partition, and the identity
        that a ghost's Cartesian offset deforms like a coordinate. A sign
        error in any of them lands here.
        """
        coord, atype, cell, *_ = system
        model, _ = joint

        def energy(positions, box):
            return to_numpy(evaluate_partitioned(
                model, positions, atype, box,
                n_partitions=n, devices=[DEVICE] * n,
            ))["energy"]

        analytic = to_numpy(evaluate_partitioned(
            model, coord, atype, cell,
            n_partitions=n, devices=[DEVICE] * n, compute_virial=True,
        ))["virial"]

        step = 1e-5
        numeric = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                delta = np.zeros((3, 3))
                delta[i, j] = step
                delta = 0.5 * (delta + delta.T)      # symmetric strain
                energies = []
                for sign in (1.0, -1.0):
                    strain = np.eye(3) + sign * delta
                    energies.append(
                        energy(coord @ strain.T, cell @ strain.T)
                    )
                numeric[i, j] = -(energies[0] - energies[1]) / (2 * step)

        worst = float(np.abs(analytic - numeric).max())
        assert worst < 2e-5 * max(abs(np.trace(numeric)), 1.0), (
            f"{n} partitions: max |diff| = {worst:.3e}, "
            f"analytic trace {np.trace(analytic):.6f} vs "
            f"numeric trace {np.trace(numeric):.6f}"
        )


class TestDistributedEvaluator:
    """The force driver: energy and forces across devices, nothing else.

    No timestep, ensemble or thermostat lives here -- those are the caller's.
    The last test checks the driver is actually usable by nvalchemiops'
    integrators, which is the contract that matters.
    """

    @pytest.fixture(scope="class")
    def model(self):
        return build(long_range=True).eval()

    @staticmethod
    def _driver(model, system, **kwargs):
        kwargs.setdefault("devices", [str(DEVICE)])
        return DistributedEvaluator(model, system[1], system[2], **kwargs)

    def test_matches_the_unpartitioned_model(self, model, system):
        coord, atype, cell, coord_t, atype_t, cell_t = system
        reference = model(coord_t, atype_t, box=cell_t)
        driver = self._driver(model, system)
        energy, forces = driver(coord)
        assert abs(energy - float(reference["energy"])) < 1e-8
        expected = reference["force"][0].detach().cpu().numpy()
        assert np.abs(forces.cpu().numpy() - expected).max() < 1e-8

    def test_accepts_a_torch_tensor_and_returns_one_on_the_model_device(
        self, model, system
    ):
        coord = system[0]
        driver = self._driver(model, system)
        tensor_in = torch.tensor(coord, dtype=torch.float64, device=DEVICE)
        _, forces = driver(tensor_in)
        assert forces.device == next(model.parameters()).device
        assert forces.dtype == torch.float64
        assert forces.shape == (driver.n_atoms, 3)

    def test_rejects_the_wrong_number_of_atoms(self, model, system):
        driver = self._driver(model, system)
        with pytest.raises(ValueError, match="expected positions of shape"):
            driver(system[0][:-1])

    def test_exposes_the_energy_split_and_charges(self, model, system):
        driver = self._driver(model, system)
        driver(system[0])
        for key in ("energy_short_range", "energy_long_range", "charge"):
            assert key in driver.results

    def test_masses_come_from_the_models_type_map(self, model, system):
        driver = self._driver(model, system)
        masses = driver.masses()
        assert masses.shape == (driver.n_atoms,)
        # The synthetic system is water: oxygen first, then two hydrogens.
        type_map = model.get_type_map()
        for index, mass in zip(driver.atype[:3], masses[:3], strict=True):
            expected = {"O": 15.999, "H": 1.008}[type_map[index]]
            assert abs(mass - expected) < 0.01

    def test_reused_halo_stays_exact_under_motion(self, model, system):
        """The point of the skin: positions move, the halo is reused, and the
        forces stay equal to a driver that has just rebuilt."""
        coord = system[0]
        driver = self._driver(model, system, skin=1.0)
        driver(coord)
        moved = coord + np.random.default_rng(0).normal(0, 0.05, coord.shape)
        _, reused = driver(moved)
        fresh = self._driver(model, system, skin=1.0)
        _, rebuilt = fresh(moved)
        assert np.abs(
            reused.cpu().numpy() - rebuilt.cpu().numpy()
        ).max() < 1e-8

    def test_forcing_a_rebuild_every_call_changes_nothing(self, model, system):
        """Caching must be an optimisation, not a physics knob."""
        coord = system[0]
        cached = self._driver(model, system, skin=5.0)
        eager = self._driver(model, system, skin=5.0, repartition_every=1)
        rng = np.random.default_rng(1)
        for _ in range(4):
            coord = coord + rng.normal(0, 0.02, coord.shape)
            _, a = cached(coord)
            _, b = eager(coord)
            assert np.abs(a.cpu().numpy() - b.cpu().numpy()).max() < 1e-8
        assert eager.rebuild_count > cached.rebuild_count

    def test_virial_matches_the_unpartitioned_model(self, model, system):
        """The partitioned virial must equal the serial one, which is itself
        finite-difference checked in `tests/test_joint_model.py`."""
        coord, atype, cell, coord_t, atype_t, cell_t = system
        reference = (
            model(coord_t, atype_t, box=cell_t)["virial"]
            .reshape(3, 3)
            .detach()
            .cpu()
            .numpy()
        )
        driver = self._driver(model, system, compute_virial=True)
        driver(coord)
        assert np.abs(driver.virial - reference).max() < 1e-10

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_virial_is_partition_count_invariant(self, model, system, n):
        """Each edge lives in exactly one partition, so the per-partition
        virials sum to the global one -- the ownership argument that makes
        the forces exact applies unchanged."""
        driver = self._driver(
            model, system, devices=[str(DEVICE)] * n, compute_virial=True
        )
        driver(system[0])
        single = self._driver(model, system, compute_virial=True)
        single(system[0])
        assert np.abs(driver.virial - single.virial).max() < 1e-10

    def test_virial_is_symmetric(self, model, system):
        """A rotation-invariant energy has a symmetric virial."""
        driver = self._driver(model, system, compute_virial=True)
        driver(system[0])
        assert np.abs(driver.virial - driver.virial.T).max() < 1e-10

    def test_virial_splits_into_its_three_contributions(self, model, system):
        """Short-range, reciprocal and charge-response, and they must add up.

        The charge-response piece is the one that could plausibly be dropped
        without anything looking wrong, so its presence is asserted.
        """
        driver = self._driver(model, system, compute_virial=True)
        driver(system[0])
        parts = [
            driver.results[key]
            for key in (
                "virial_short_range",
                "virial_long_range",
                "virial_charge_response",
            )
        ]
        assert np.abs(sum(parts) - driver.virial).max() < 1e-12
        assert np.abs(parts[2]).max() > 0.0, "charge-response virial is zero"

    def test_stress_is_minus_virial_over_volume(self, model, system):
        driver = self._driver(model, system, compute_virial=True)
        driver(system[0])
        expected = -driver.virial / abs(np.linalg.det(system[2]))
        assert np.abs(driver.stress - expected).max() < 1e-14

    def test_no_virial_unless_asked(self, model, system):
        """It costs a strain leaf per partition and a second backward, so it
        is opt-in; asking for `stress` without it gives None, not zeros."""
        driver = self._driver(model, system)
        driver(system[0])
        assert driver.virial is None
        assert driver.stress is None
        assert "virial" not in driver.results

    def test_rejects_an_empty_device_list(self, model, system):
        with pytest.raises(ValueError, match="at least one device"):
            self._driver(model, system, devices=[])

    def test_rejects_a_negative_skin(self, model, system):
        with pytest.raises(ValueError, match="skin must be non-negative"):
            self._driver(model, system, skin=-1.0)

    def test_drives_an_nvalchemiops_integrator(self, model, system):
        """The contract: the returned tensor feeds the Warp kernels directly.

        This is the whole reason the driver returns a device tensor rather
        than a NumPy array -- `wp.from_torch` is a zero-copy view, so the
        trajectory never leaves the GPU. Velocity Verlet is used because it
        conserves energy, which is what catches a units or sequencing
        mistake; the ensemble itself is not the driver's business.
        """
        import warp as wp
        from nvalchemiops.dynamics.integrators import (
            velocity_verlet_position_update,
            velocity_verlet_velocity_finalize,
        )

        driver = self._driver(model, system, skin=1.0)
        positions = torch.tensor(
            system[0], dtype=torch.float64, device=DEVICE
        ).contiguous()
        velocities = torch.zeros_like(positions)
        masses = torch.tensor(
            driver.masses(), dtype=torch.float64, device=DEVICE
        )
        # ASE's unit system: angstrom, eV, amu, time in A sqrt(amu/eV).
        dt = torch.tensor(
            [0.25 * ase_units.fs], dtype=torch.float64, device=DEVICE
        )

        energy, forces = driver(positions)
        vec = lambda t: wp.from_torch(t, dtype=wp.vec3d)  # noqa: E731
        kinetic = lambda: float(  # noqa: E731
            0.5 * (masses.unsqueeze(-1) * velocities**2).sum()
        )
        start = energy + kinetic()
        for _ in range(20):
            velocity_verlet_position_update(
                vec(positions), vec(velocities), vec(forces),
                wp.from_torch(masses), wp.from_torch(dt),
            )
            energy, forces = driver(positions)
            velocity_verlet_velocity_finalize(
                vec(velocities), vec(forces),
                wp.from_torch(masses), wp.from_torch(dt),
            )
            wp.synchronize()
        drift = abs(energy + kinetic() - start) / driver.n_atoms * 1000
        assert drift < 1.0, f"drift {drift:.4f} meV/atom over 20 steps"


class TestLongRangeBoundary:
    """The system's pbc decides how the long-range sum is taken, and only a
    term that cannot be adapted is refused. Before either, the partitioned
    path ran an ``isolated`` term on the periodic water box 1.1 eV off, and a
    slab term 0.03 eV off, and raised nothing."""

    def test_an_isolated_term_is_refused(self, system):
        coord, atype, cell, *_ = system
        model = build()
        model.set_long_range({"method": "isolated"})
        with pytest.raises(ValueError, match="ASE calculator"):
            DistributedEvaluator(model, atype, cell, devices=[DEVICE])
        with pytest.raises(ValueError, match="does not match the system's pbc"):
            evaluate_partitioned(model, coord, atype, cell, n_partitions=1)

    def test_a_slab_term_on_a_bulk_cell_takes_the_3d_sum(self, system):
        coord, atype, cell, *_ = system

        def slab_model():
            model = build()
            settings = model.long_range_energy.serialize()
            settings.update(slab_correction=True, pbc=[True, True, False])
            model.set_long_range(settings)
            return model

        ev = DistributedEvaluator(
            slab_model(), atype, cell, devices=[DEVICE], memory_headroom=None
        )
        assert ev.long_range_energy.slab_correction is False
        assert ev.model.long_range_energy.slab_correction is True  # left alone
        bulk = DistributedEvaluator(
            build(), atype, cell, devices=[DEVICE], memory_headroom=None
        )
        (e, f), (e_ref, f_ref) = ev(coord), bulk(coord)
        assert e == pytest.approx(e_ref, abs=1e-9)
        np.testing.assert_allclose(f.cpu().numpy(), f_ref.cpu().numpy(), atol=1e-9)
        # The functional API adapts nothing, so there the mismatch is refused.
        with pytest.raises(ValueError, match="does not match the system's pbc"):
            evaluate_partitioned(slab_model(), coord, atype, cell, n_partitions=1)


def slab(n_mol=16, box=14.0, vacuum=20.0):
    """The fixture's molecules as a slab: a periodic xy plane, an open z with
    `vacuum` of it above the layer, the layer centred."""
    coord, atype, _ = frame(n_mol=n_mol, box=box)
    c = coord[0].cpu().numpy().copy()
    c[:, 2] += vacuum / 2 - c[:, 2].min()
    height = np.ptp(c[:, 2]) + vacuum
    return c, atype[0].cpu().numpy(), np.diag([box, box, height])


SLAB_PBC = (True, True, False)


class TestSlab:
    """One open axis: no periodic images along it, and the slab form of the
    long-range sum (Yeh-Berkowitz), whatever the model was trained with."""

    def serial(self, coord, atype, cell):
        """The unpartitioned model on the same slab. Its descriptor sees the
        cell as 3D-periodic, which the 20 A gap, far above rcut, makes the
        same as an open z."""
        model = build()
        model.adapt_long_range(cell, SLAB_PBC)
        out = model(
            torch.tensor(coord, dtype=torch.float64, device=DEVICE)[None],
            torch.tensor(atype, device=DEVICE)[None],
            box=torch.tensor(cell, dtype=torch.float64, device=DEVICE)[None],
        )
        energy = float(out["energy"].reshape(-1)[0])
        return energy, out["force"][0].detach().cpu().numpy()

    @pytest.mark.parametrize("n_part", [1, 2, 4])
    def test_a_slab_matches_the_serial_model(self, n_part):
        coord, atype, cell = slab()
        e_ref, f_ref = self.serial(coord, atype, cell)
        ev = DistributedEvaluator(
            build(), atype, cell, devices=[DEVICE] * n_part, pbc=SLAB_PBC,
            memory_headroom=None,
        )
        e, f = ev(coord)
        assert float(e) == pytest.approx(e_ref, abs=1e-9)
        np.testing.assert_allclose(f.cpu().numpy(), f_ref, atol=1e-9)

    def test_pbc_switches_the_slab_sum_on(self):
        coord, atype, cell = slab()
        ev = DistributedEvaluator(
            build(), atype, cell, devices=[DEVICE], pbc=SLAB_PBC, memory_headroom=None
        )
        term = ev.long_range_energy
        assert term.slab_correction is True
        assert term.pbc == SLAB_PBC
        # The mesh covers the vacuum too: at least ceil(2 alpha L) along z.
        assert term.mesh_dimensions[2] >= math.ceil(0.8 * cell[2, 2])

    def test_two_partitions_cut_the_plane_not_the_slab(self):
        coord, atype, cell = slab(n_mol=32)
        parts = partition_system(coord, cell, 6.0, 2, pbc=SLAB_PBC, skin=1.0)
        for part in parts:
            # Each half spans the whole thickness: the cut is in-plane.
            assert np.ptp(coord[part.owned, 2]) > 0.8 * np.ptp(coord[:, 2])

    def test_an_atom_near_an_open_face_is_refused(self):
        coord, atype, cell = slab()
        coord[0, 2] = cell[2, 2] - 0.1        # 0.1 A under the top face
        ev = DistributedEvaluator(
            build(), atype, cell, devices=[DEVICE], pbc=SLAB_PBC, memory_headroom=None
        )
        with pytest.raises(ValueError, match="inside a face"):
            ev(coord)

    def test_two_open_axes_are_refused(self):
        coord, atype, cell = slab()
        with pytest.raises(ValueError, match="bulk cell and a slab"):
            DistributedEvaluator(
                build(), atype, cell, devices=[DEVICE], pbc=(True, False, False)
            )


class TestMeshFollowsTheCell:
    def test_a_supercell_gets_the_mesh_its_cell_needs(self):
        """A 2x2x2 supercell is the same periodic system, so it must give the
        unit cell's energy per atom and its forces. It does only because the
        mesh doubles with the cell: a stored mesh kept on the bigger cell
        under-resolves the sum."""

        def model(mesh):
            m = build()
            settings = m.long_range_energy.serialize()
            settings.update(mesh_dimensions=list(mesh), mesh_spacing=None)
            m.set_long_range(settings)
            return m

        coord, atype, cell = frame(n_mol=8, box=10.0)
        c, t, h = coord[0].cpu().numpy(), atype[0].cpu().numpy(), cell[0].cpu().numpy()
        unit = DistributedEvaluator(model((8, 8, 8)), t, h, devices=[DEVICE],
                                    memory_headroom=None)
        e1, f1 = unit(c)
        shifts = np.array(
            [[i, j, k] for i in range(2) for j in range(2) for k in range(2)]
        )
        c2 = np.concatenate([c + s @ h for s in shifts])
        t2, h2 = np.tile(t, len(shifts)), 2.0 * h
        big = DistributedEvaluator(model((8, 8, 8)), t2, h2, devices=[DEVICE],
                                   memory_headroom=None)
        assert big.long_range_energy.mesh_dimensions == (16, 16, 16)
        e2, f2 = big(c2)
        assert float(e2) / len(t2) == pytest.approx(float(e1) / len(t), abs=1e-10)
        np.testing.assert_allclose(
            f2.cpu().numpy(), np.tile(f1.cpu().numpy(), (len(shifts), 1)), atol=1e-9
        )


class TestSharedModel:
    """One model, several evaluators: each builds the term its own cell and
    boundary need, and leaves the model alone. They used to adapt the model
    itself. A slab evaluator then broke the bulk one built before it, and a
    small cell's evaluator silently took a supercell's mesh away."""

    def test_each_evaluator_keeps_its_own_term(self):
        from deepmd_lr.ase import LongRangeCalculator

        model = build()
        before = model.long_range_energy.serialize()
        coord, atype, cell = frame(n_mol=8, box=10.0)
        c, t, h = coord[0].cpu().numpy(), atype[0].cpu().numpy(), cell[0].cpu().numpy()
        shifts = np.array(
            [[i, j, k] for i in range(2) for j in range(2) for k in range(2)]
        )
        systems = {
            "supercell": (np.concatenate([c + s @ h for s in shifts]),
                          np.tile(t, len(shifts)), 2.0 * h, (True, True, True)),
            "unit": (c, t, h, (True, True, True)),
            "slab": (*slab(n_mol=8, box=10.0), SLAB_PBC),
        }
        evs, first = {}, {}
        for name, (x, types, box, pbc) in systems.items():
            evs[name] = DistributedEvaluator(model, types, box, devices=[DEVICE],
                                             pbc=pbc, memory_headroom=None)
            first[name] = float(evs[name](x)[0])
        assert model.long_range_energy.serialize() == before
        mesh = {n: ev.long_range_energy.mesh_dimensions for n, ev in evs.items()}
        assert mesh["supercell"] > mesh["unit"]  # the larger cell needs more
        assert evs["slab"].long_range_energy.slab_correction is True
        # An ASE calculator on the same model adapts the model itself, here
        # to the unit cell's bulk sum; no evaluator notices.
        atoms = Atoms(numbers=[8, 1, 1] * 8, positions=c, cell=h, pbc=True)
        atoms.calc = LongRangeCalculator(model)
        atoms.get_potential_energy()
        for name, (x, *_rest) in systems.items():
            assert float(evs[name](x)[0]) == pytest.approx(first[name], abs=1e-9)

    def test_long_range_settings_are_the_evaluators_alone(self):
        model = build()
        before = model.long_range_energy.serialize()
        coord, atype, cell = frame(n_mol=8, box=10.0)
        t, h = atype[0].cpu().numpy(), cell[0].cpu().numpy()
        ev = DistributedEvaluator(
            model, t, h, devices=[DEVICE], memory_headroom=None,
            long_range={"method": "pme", "mesh_dimensions": [20, 20, 20]},
        )
        assert ev.long_range_energy.mesh_dimensions == (20, 20, 20)
        assert model.long_range_energy.serialize() == before
        with pytest.raises(ValueError, match="alpha is fixed"):
            DistributedEvaluator(model, t, h, devices=[DEVICE],
                                 long_range={"method": "pme", "alpha": 0.1})


class TestBackboneCapture:
    """The charge head reuses the energy head's descriptor pass.

    `partition_charges` used to run the backbone a second time per partition,
    which measured 18-19% of the long-range step. The capture removes it. Two
    things have to hold: it must actually be taken, and it must not change any
    number -- the features come from a different autograd leaf, so the
    charge-response force and the Born tensor both had to be retargeted.
    """

    @staticmethod
    def _system(n_mol=8):
        model = build()
        coord, atype, cell = frame(n_mol=n_mol)
        return (model, coord[0].cpu().numpy(), atype[0].cpu().numpy(),
                cell[0].cpu().numpy())

    @staticmethod
    def _run(model, c, a, cl, devices, *, force_fallback=False, **kw):
        import deepmd_lr.nvalchemi.driver as DRV

        original = DRV._usable_capture
        if force_fallback:
            DRV._usable_capture = lambda captured: False
        try:
            ev = DistributedEvaluator(model, a, cl, devices=devices, **kw)
            energy, forces = ev(c)
            return (float(energy), forces.detach().cpu().numpy().copy(),
                    ev.results["charge"].copy(), ev.results, ev)
        finally:
            DRV._usable_capture = original

    def test_capture_is_taken_on_every_partition(self):
        import deepmd_lr.nvalchemi.driver as DRV

        model, c, a, cl = self._system()
        calls = {"n": 0}
        original = DRV.partition_charges

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        DRV.partition_charges = counting
        try:
            self._run(model, c, a, cl, [DEVICE] * 4)
        finally:
            DRV.partition_charges = original
        assert calls["n"] == 0, (
            f"{calls['n']} partitions fell back to a second descriptor pass"
        )

    @pytest.mark.parametrize("n_part", [1, 2, 4])
    def test_identical_to_recomputing_the_backbone(self, n_part):
        model, c, a, cl = self._system()
        e_cap, f_cap, q_cap, _, _ = self._run(model, c, a, cl, [DEVICE] * n_part)
        e_ref, f_ref, q_ref, _, _ = self._run(
            model, c, a, cl, [DEVICE] * n_part, force_fallback=True
        )
        assert abs(e_cap - e_ref) < 1e-10
        assert np.abs(f_cap - f_ref).max() < 1e-10
        assert np.abs(q_cap - q_ref).max() < 1e-10

    def test_born_is_identical_too(self):
        """Born differentiates `raw` w.r.t. the edge vectors, so it had to be
        retargeted onto the captured leaf."""
        model, c, a, cl = self._system()
        _, _, _, cap, _ = self._run(model, c, a, cl, [DEVICE] * 2, compute_born=True)
        _, _, _, ref, _ = self._run(
            model, c, a, cl, [DEVICE] * 2, compute_born=True, force_fallback=True
        )
        assert np.abs(cap["born_charge"] - ref["born_charge"]).max() < 1e-10

    def test_virial_is_identical_too(self):
        """The strain reaches `eps` only through the partition's own
        `edge_vec`, which is why the response force chains through it rather
        than scattering the captured gradient by hand."""
        model, c, a, cl = self._system()
        _, _, _, _, ev_cap = self._run(
            model, c, a, cl, [DEVICE] * 2, compute_virial=True
        )
        _, _, _, _, ev_ref = self._run(
            model, c, a, cl, [DEVICE] * 2, compute_virial=True, force_fallback=True
        )
        assert np.abs(ev_cap.virial - ev_ref.virial).max() < 1e-10


class TestUnwrappedCoordinates:
    """Integrators never wrap, so no result may depend on which periodic image
    an atom sits in. The halo used to be built from the raw coordinates with
    only the +-1 images, so atoms that had diffused out through opposite faces
    silently lost their neighbours: 1.1 eV/A on water at a spread of L + 12 A,
    and blow-ups after 5-20 ps. Every check compares with the same atoms
    wrapped into the cell.
    """

    KEYS = ("energy", "force", "charge", "born_charge", "virial")

    @pytest.fixture(scope="class")
    def model(self):
        m = build(long_range=True)
        m.eval()
        return m

    @staticmethod
    def shifted(coord, cell, n_moved=12, seed=0, reach=3):
        """Random atoms moved by random lattice vectors, up to `reach` cells."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coord), n_moved, replace=False)
        out = np.array(coord, dtype=np.float64, copy=True)
        out[idx] += rng.integers(-reach, reach + 1, size=(n_moved, 3)) @ cell
        return out

    def evaluate(self, model, coord, atype, cell, n):
        return to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n,
            compute_born=True, compute_virial=True,
        ))

    def assert_same(self, got, want):
        for key in self.KEYS:
            np.testing.assert_allclose(got[key], want[key], atol=1e-9, err_msg=key)

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_lattice_shifts_change_nothing(self, system, model, n):
        coord, atype, cell, *_ = system
        want = self.evaluate(model, coord, atype, cell, n)
        got = self.evaluate(model, self.shifted(coord, cell), atype, cell, n)
        self.assert_same(got, want)

    def test_the_evaluator_is_indifferent_to_images(self, system, model):
        coord, atype, cell, *_ = system
        results = []
        for positions in (coord, self.shifted(coord, cell)):
            ev = DistributedEvaluator(
                model, atype, cell, devices=[DEVICE] * 2, compute_born=True,
                compute_virial=True, memory_headroom=None,
            )
            energy, forces = ev(torch.as_tensor(positions, device=DEVICE))
            results.append((energy, forces.cpu().numpy().copy(), ev.virial.copy(),
                            np.asarray(ev.results["born_charge"]).copy()))
        (e0, f0, v0, z0), (e1, f1, v1, z1) = results
        assert e1 == pytest.approx(e0, abs=1e-9)
        np.testing.assert_allclose(f1, f0, atol=1e-9)
        np.testing.assert_allclose(v1, v0, atol=1e-9)
        np.testing.assert_allclose(z1, z0, atol=1e-9)

    @pytest.mark.parametrize("spread", [6.0, 8.0, 10.0, 12.0])
    def test_a_two_sided_spread(self, system, model, spread):
        """Atoms near x = 0 leave through +x and atoms near x = L through -x,
        so the unwrapped cloud spans L + spread along x."""
        coord, atype, cell, *_ = system
        length = cell[0, 0]
        moved = np.array(coord, copy=True)
        moved[coord[:, 0] < spread / 2, 0] += length
        moved[coord[:, 0] > length - spread / 2, 0] -= length
        assert np.ptp(moved[:, 0]) > length + spread - 2.0
        for n in (1, 2):
            self.assert_same(self.evaluate(model, moved, atype, cell, n),
                             self.evaluate(model, coord, atype, cell, n))

    def test_the_serial_model_is_indifferent_to_images(self, system, model):
        """The serial path (the ASE calculator, training) never had the bug:
        DeePMD-kit's own neighbour builder wraps. Pinned so an upgrade that
        stops doing so is caught here rather than in a trajectory."""
        coord, _atype, cell, coord_t, atype_t, cell_t = system
        moved = torch.as_tensor(self.shifted(coord, cell), dtype=coord_t.dtype,
                                device=coord_t.device).unsqueeze(0)
        want = model(coord_t, atype_t, box=cell_t)
        got = model(moved, atype_t, box=cell_t)
        for key in ("energy", "force", "charge", "virial"):
            torch.testing.assert_close(got[key], want[key], atol=1e-9, rtol=0,
                                       msg=key)

    def test_crossing_a_face_between_rebuilds(self, system, model):
        """An atom walks out through a face while the partitioning is reused:
        every step must match a fresh evaluation of the wrapped positions."""
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE], skin=3.0,
                                  memory_headroom=None)
        walk = np.array(coord, copy=True)
        i = int(np.argmax(walk[:, 0]))
        ev(torch.as_tensor(walk, device=DEVICE))
        rebuilds = ev.rebuild_count
        for _ in range(7):
            walk[i, 0] += 0.2
            energy, forces = ev(torch.as_tensor(walk, device=DEVICE))
            wrapped = walk.copy()
            wrapped[i] -= np.floor(wrapped[i] @ np.linalg.inv(cell)) @ cell
            want = self.evaluate(model, wrapped, atype, cell, 1)
            assert energy == pytest.approx(want["energy"], abs=1e-9)
            np.testing.assert_allclose(forces.cpu().numpy(), want["force"], atol=1e-9)
        assert walk[i, 0] > cell[0, 0], "the walk never crossed the face"
        assert ev.rebuild_count == rebuilds, "the walk was meant to reuse the halo"

    def test_a_lattice_jump_forces_a_rebuild(self, system, model):
        """The Verlet check uses raw displacement on purpose: a caller that
        wraps an atom moves it by a lattice vector, and the halo built around
        its old image must not be reused."""
        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE], skin=2.0,
                                  memory_headroom=None)
        e0, _ = ev(torch.as_tensor(coord, device=DEVICE))
        rebuilds = ev.rebuild_count
        jumped = np.array(coord, copy=True)
        jumped[0] += cell[0]
        e1, _ = ev(torch.as_tensor(jumped, device=DEVICE))
        assert ev.rebuild_count == rebuilds + 1
        assert e1 == pytest.approx(e0, abs=1e-9)

    @pytest.mark.skipif(
        not os.environ.get("DEEPMD_LR_WATER_MODEL"),
        reason="slow: set DEEPMD_LR_WATER_MODEL to a trained water checkpoint",
    )
    def test_a_long_water_run_keeps_exact_forces(self):
        """25 ps of Langevin water on two partitions, never wrapped. Every
        500 steps the forces must match a fresh evaluation of the wrapped
        positions. Water diffuses past L + 12 A within this time, the spread
        at which the old halo was off by 1.1 eV/A.

        Opt-in: DEEPMD_LR_WATER_MODEL (checkpoint), DEEPMD_LR_WATER_COMPRESS=1
        (needs the CUDA-operator build), DEEPMD_LR_WATER_STEPS (default 50000).
        """
        import pathlib

        from ase import units
        from ase.md.velocitydistribution import Stationary, thermalize_momenta

        from deepmd_lr.nvalchemi import (
            Langevin,
            build_evaluator,
            read_structure,
        )

        root = pathlib.Path(__file__).resolve().parents[1]
        atoms = read_structure(root / "examples" / "06-md_nvalchemi" / "water_192.xyz")
        compress = os.environ.get("DEEPMD_LR_WATER_COMPRESS") == "1"
        ev = build_evaluator(os.environ["DEEPMD_LR_WATER_MODEL"], atoms,
                             devices=[DEVICE] * 2, compress=compress)
        thermalize_momenta(atoms, temperature_K=300.0, rng=np.random.default_rng(1))
        Stationary(atoms)
        md = Langevin(ev, atoms, 0.5 * units.fs, 300.0, friction=0.05, seed=0)
        cell = np.asarray(ev.cell)
        inverse = np.linalg.inv(cell)
        steps = int(os.environ.get("DEEPMD_LR_WATER_STEPS", 50_000))
        seen = {"worst": 0.0, "spread": 0.0}

        def check():
            x = md.thermo.positions.numpy()
            assert np.isfinite(x).all(), f"blew up by step {md.nsteps}"
            seen["spread"] = max(seen["spread"],
                                 float((np.ptp(x, axis=0) - np.diag(cell)).max()))
            wrapped = x - np.floor(x @ inverse) @ cell
            fresh = to_numpy(evaluate_partitioned(ev.model, wrapped, ev.atype, cell,
                                                  devices=[DEVICE]))
            seen["worst"] = max(seen["worst"], float(
                np.abs(md.thermo.forces.numpy() - fresh["force"]).max()))

        md.attach(check, 500, sync_atoms=False)
        md.run(steps)
        worst, spread = seen["worst"], seen["spread"]
        print(f"\n  {steps} steps: widest spread L + {spread:.1f} A, "
              f"worst force difference {worst:.2e} eV/A")
        assert spread > 12.0, "the run never reached the regime it guards"
        assert worst <= 1e-5



class TestReusedNeighbourhoods:
    """Between calls the evaluator reuses the partitions -- across cell
    changes too -- and each partition's pair list. Neither may change an
    answer: every check compares with a fresh evaluation of the same positions
    and cell, with new partitions and a fresh neighbour search."""

    KEYS = ("energy", "force", "charge", "born_charge", "virial")

    @pytest.fixture(scope="class")
    def model(self):
        m = build(long_range=True)
        m.eval()
        return m

    @staticmethod
    def fresh(model, coord, atype, cell, n):
        return to_numpy(evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n,
            compute_born=True, compute_virial=True,
        ))

    @staticmethod
    def reused(ev, coord):
        energy, forces = ev(torch.as_tensor(coord, device=DEVICE))
        return {
            "energy": energy,
            "force": forces.cpu().numpy().copy(),
            "charge": np.asarray(ev.results["charge"]).copy(),
            "born_charge": np.asarray(ev.results["born_charge"]).copy(),
            "virial": ev.virial.copy(),
        }

    def assert_same(self, got, want):
        for key in self.KEYS:
            np.testing.assert_allclose(got[key], want[key], atol=1e-9, err_msg=key)

    @staticmethod
    def evaluator(model, atype, cell, n, **kwargs):
        return DistributedEvaluator(
            model, atype, cell, devices=[DEVICE] * n, compute_born=True,
            compute_virial=True, memory_headroom=None, **kwargs,
        )

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_each_kind_of_change_rebuilds_only_what_it_must(self, system, model, n):
        """A halo of 2 A, pair lists of 0.5 A: a small shear reuses both, a
        0.3 A step (past half the pair skin) rebuilds the pair lists only,
        and a 10 % strain repartitions."""
        coord, atype, cell, *_ = system
        ev = self.evaluator(model, atype, cell, n, skin=2.0, pair_skin=0.5)
        x, h = coord.copy(), cell.copy()
        self.assert_same(self.reused(ev, x), self.fresh(model, x, atype, h, n))
        counts = (ev.rebuild_count, ev.pair_rebuild_count)
        shear = np.eye(3)
        shear[1, 0] = shear[0, 1] = 1e-3
        h, x = h @ shear, x @ shear
        ev.set_cell(h)
        self.assert_same(self.reused(ev, x), self.fresh(model, x, atype, h, n))
        assert (ev.rebuild_count, ev.pair_rebuild_count) == counts
        x[3] += [0.3, 0.0, 0.0]
        self.assert_same(self.reused(ev, x), self.fresh(model, x, atype, h, n))
        assert (ev.rebuild_count, ev.pair_rebuild_count) == (counts[0], counts[1] + 1)
        h, x = h * 1.1, x * 1.1
        ev.set_cell(h)
        self.assert_same(self.reused(ev, x), self.fresh(model, x, atype, h, n))
        assert ev.rebuild_count == counts[0] + 1

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_a_walk_under_a_barostat_matches_fresh_evaluations(self, system, model, n):
        """Atoms walk and the cell strains a little every call, as under NPT,
        with the default skins. Every call matches a fresh evaluation, and the
        partitions survive the cell's changes."""
        coord, atype, cell, *_ = system
        ev = self.evaluator(model, atype, cell, n)
        rng = np.random.default_rng(1)
        x, h = coord.copy(), cell.copy()
        for step in range(10):
            if step:
                strain = np.eye(3) + 1e-3 * rng.normal(size=(3, 3))
                strain = 0.5 * (strain + strain.T)
                h, x = h @ strain, x @ strain + rng.normal(scale=0.01, size=x.shape)
                ev.set_cell(h)
            self.assert_same(self.reused(ev, x), self.fresh(model, x, atype, h, n))
        assert ev.rebuild_count < 5

    def test_the_cached_graph_is_the_fresh_one(self, system, model):
        """Same edges as a fresh search, and CSR views equal to what
        ``attach_edge_csr`` computes from the payload, without its sorts."""
        from deepmd.dpmodel.utils.neighbor_graph.csr import attach_edge_csr

        coord, atype, cell, *_ = system
        for part in partition_system(coord, cell, RCUT, 2, skin=1.0):
            fresh, _, _ = partition_graph(part, coord, atype, RCUT, DEVICE)
            cached, _, _ = partition_graph(
                part, coord, atype, RCUT, DEVICE, pair_skin=0.5, with_csr=True
            )
            assert part.pairs_on(DEVICE) is not None

            def edges(g):
                src, dst = (g.edge_index[k][g.edge_mask].tolist() for k in (0, 1))
                return set(zip(src, dst, strict=True))

            assert edges(cached) == edges(fresh)
            want = attach_edge_csr(cached, part.n_node)
            for name in ("destination_order", "destination_row_ptr",
                         "source_order", "source_row_ptr"):
                assert torch.equal(getattr(cached, name), getattr(want, name)), name
            part.drop_pairs()
            assert part.pairs_on(DEVICE) is None

    def test_no_pair_skin_searches_every_call(self, system, model):
        coord, atype, cell, *_ = system
        ev = self.evaluator(model, atype, cell, 2, pair_skin=0)
        want = self.fresh(model, coord, atype, cell, 2)
        self.assert_same(self.reused(ev, coord), want)
        assert ev.pair_skin == 0 and ev.pair_rebuild_count == 0
        assert all(p.pairs_on(DEVICE) is None for p in ev._partitions)


# --- charge_spin and exclude_types, serial against partitioned --------------


def conditioned_model(excluded=False):
    descriptor = copy.deepcopy(DESCRIPTOR)
    descriptor.update(add_chg_spin_ebd=True, default_chg_spin=[1.0, 2.0])
    long_range = copy.deepcopy(LONG_RANGE)
    long_range["charge_net"]["exclude_types"] = [1] if excluded else []
    return get_model(
        dict(
            type="dp_lr",
            type_map=["O", "H"],
            descriptor=descriptor,
            fitting_net=copy.deepcopy(FITTING),
            long_range=long_range,
        )
    ).eval()


@pytest.mark.parametrize("batched", [False, True])
def test_default_charge_matches_explicit_condition(batched):
    model = conditioned_model()
    coord, atype, cell = frame()
    if batched:
        coord, atype, cell = (
            x.repeat(2, *([1] * (x.ndim - 1))) for x in (coord, atype, cell)
        )
    implicit = model(coord, atype, box=cell)
    explicit = model(coord, atype, box=cell, charge_spin=coord.new_tensor([1.0, 2.0]))
    torch.testing.assert_close(
        implicit["charge"].sum(dim=1), torch.ones_like(implicit["charge"].sum(dim=1))
    )
    for key in ("charge", "energy", "force", "virial"):
        torch.testing.assert_close(implicit[key], explicit[key])


@pytest.mark.parametrize("excluded", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_partitioned_charge_contracts(excluded, explicit):
    model = conditioned_model(excluded)
    coord, atype, cell = frame()
    condition = (-1.0, 3.0) if explicit else None
    serial = model(coord, atype, box=cell, charge_spin=condition)
    positions = coord[0].cpu().numpy()
    types = atype[0].cpu().numpy()
    box = cell[0].cpu().numpy()
    result = to_numpy(evaluate_partitioned(
        model,
        positions,
        types,
        box,
        n_partitions=2,
        devices=[coord.device] * 2,
        compute_virial=True,
        charge_spin=condition,
    ))
    assert result["charge"].sum() == pytest.approx(-1.0 if explicit else 1.0, abs=1e-12)
    if excluded:
        np.testing.assert_array_equal(result["charge"][types == 1], 0.0)
        torch.testing.assert_close(
            serial["charge"].reshape(-1)[atype.reshape(-1) == 1], coord.new_zeros(8)
        )
    for key in ("charge", "energy", "force", "virial", "atom_energy"):
        reference = serial[key].detach().cpu().numpy().reshape(np.shape(result[key]))
        np.testing.assert_allclose(result[key], reference, atol=1e-8, rtol=1e-7)
    assert result["atom_energy"].sum() == pytest.approx(result["energy"], abs=1e-10)
    np.testing.assert_allclose(
        result["atom_energy_short_range"] + result["atom_energy_long_range"],
        result["atom_energy"],
        atol=1e-12,
    )
    evaluator = DistributedEvaluator(
        model,
        types,
        box,
        devices=[coord.device],
        charge_spin=condition,
    )
    energy, force = evaluator(coord[0])
    assert energy == pytest.approx(result["energy"], abs=1e-8)
    np.testing.assert_allclose(force.cpu().numpy(), result["force"], atol=1e-8)
