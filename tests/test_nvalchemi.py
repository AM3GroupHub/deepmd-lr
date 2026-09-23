"""Phase A: partitioned evaluation must reproduce the serial answer.

The claim is partition-count invariance: 1, 2, 4 and 8 partitions must all
agree with the undivided model, to the floor set by GPU nondeterminism. This
is precisely the test the LAMMPS domain decomposition failed -- splitting a
reciprocal sum across ranks gave 2.0% error at 2 ranks and 36.5% at 8
(see LAMMPS.md) -- so here it has to pass exactly, not approximately.

Phase A covers the short-range model; Phase B adds the long-range term,
gathered onto one device.
"""

import numpy as np
import pytest
import torch
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

from .test_joint_model import build, frame

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
        cell = np.diag([99.43] * 3)
        for parts in (1, 2, 4):
            skin = default_skin(cell, 6.0, parts, tolerance=0.25)
            edges = np.array([99.43] * 3)
            edges[0] /= parts
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
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        _coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE])
        before = ev.skin
        ev.set_cell(cell * 2.0)
        assert ev.skin > before
        assert ev.skin == pytest.approx(default_skin(cell * 2.0, ev._rcut, 1))
        ev.set_cell(cell)
        assert ev.skin == pytest.approx(before)

    def test_an_explicit_skin_is_not_touched_by_set_cell(self, model, system):
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        _coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE], skin=2.5)
        ev.set_cell(cell * 2.0)
        assert ev.skin == 2.5

    def test_evaluator_resolves_auto_and_reports_it(self, model, system):
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        coord, atype, cell, *_ = system
        ev = DistributedEvaluator(model, atype, cell, devices=[DEVICE])
        assert ev.skin == pytest.approx(default_skin(cell, float(model.get_rcut()), 1))
        explicit = DistributedEvaluator(
            model, atype, cell, devices=[DEVICE], skin=2.5
        )
        assert explicit.skin == 2.5

    def test_rejects_an_unknown_string(self, model, system):
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

        _coord, atype, cell, *_ = system
        with pytest.raises(ValueError, match="number or 'auto'"):
            DistributedEvaluator(model, atype, cell, devices=[DEVICE], skin="wide")


class TestThreadedLaunch:
    """Each device queues its own partitions, so one device's host syncs do
    not stall another's.

    The neighbour search makes eight blocking device-to-host reads per
    partition; from a single thread they serialise the devices whatever the
    CUDA streams do. Threading must not change a single number.
    """

    @pytest.fixture(scope="class")
    def model(self):
        m = build(long_range=True)
        m.eval()
        return m

    @pytest.mark.parametrize("parts", [2, 4])
    def test_threaded_equals_sequential(self, system, model, parts):
        coord, atype, cell, *_ = system
        devices = [DEVICE] * parts
        common = {"n_partitions": parts, "devices": devices, "compute_born": True}
        seq = to_numpy(
            evaluate_partitioned(
                model, coord, atype, cell, launch_threads=False, **common
            )
        )
        thr = to_numpy(
            evaluate_partitioned(
                model, coord, atype, cell, launch_threads=True, **common
            )
        )
        assert thr["energy"] == pytest.approx(seq["energy"], abs=1e-9)
        for key in ("force", "charge", "born_charge", "atom_energy"):
            np.testing.assert_allclose(thr[key], seq[key], atol=1e-10, err_msg=key)

    def test_the_launch_really_runs_off_the_main_thread(self, system, model):
        """`launch_threads=True` must enter the pool even on one device.

        Without this the suite on a single-GPU machine would exercise the old
        sequential loop while claiming to pin the threaded one: the automatic
        rule skips the pool when there is only one device, and every test here
        passes the same device twice.
        """
        import threading  # noqa: PLC0415

        from deepmd_lr.nvalchemi import driver as drv  # noqa: PLC0415

        coord, atype, cell, *_ = system
        seen: set[str] = set()
        original = drv.partition_graph

        def record(*args, **kwargs):
            seen.add(threading.current_thread().name)
            return original(*args, **kwargs)

        drv.partition_graph = record
        try:
            evaluate_partitioned(
                model, coord, atype, cell, n_partitions=2,
                devices=[DEVICE] * 2, launch_threads=False,
            )
            assert seen == {threading.main_thread().name}, seen
            seen.clear()
            evaluate_partitioned(
                model, coord, atype, cell, n_partitions=2,
                devices=[DEVICE] * 2, launch_threads=True,
            )
            assert seen and threading.main_thread().name not in seen, seen
        finally:
            drv.partition_graph = original

    def test_a_failing_partition_raises_on_the_calling_thread(self, system, model):
        """An exception in a worker must not be swallowed into a hang."""
        coord, atype, cell, *_ = system
        with pytest.raises(RuntimeError, match="partition exploded"):
            evaluate_partitioned(
                model,
                coord,
                atype,
                cell,
                n_partitions=2,
                devices=[DEVICE] * 2,
                launch_threads=True,
                charge_spin=_Exploding(),
            )


class _Exploding:
    """A charge_spin stand-in that raises inside the launch phase."""

    def __getitem__(self, item):
        raise RuntimeError("partition exploded")

    def __len__(self):
        raise RuntimeError("partition exploded")


class TestFirstPassIsSerialised:
    """The first pass over a device runs alone; every later pass does not.

    Backends under this phase build state on first use and are not thread-safe
    while they do it. Two were found by running the scaling matrix on V100s:
    torch's CUDA linear-algebra library, whose loader asserts it runs at most
    once, and warp's code generation, which handed a worker a half-built
    ``Adjoint``. The first is warmed by name before the pool; the second, and
    anything else lazy, is covered by making the first pass hold a lock.

    What must NOT be the fix: running the first evaluation on the main thread.
    Threading is a documented request, and a caller that asks for it on step
    one must get it.
    """

    @pytest.fixture(scope="class")
    def model(self):
        m = build(long_range=True)
        m.eval()
        return m

    @staticmethod
    def _counting_lock(monkeypatch):
        """Replace the module's warm-up lock with one that counts entries."""
        import threading  # noqa: PLC0415

        from deepmd_lr.nvalchemi import driver as drv  # noqa: PLC0415

        class Counter:
            def __init__(self):
                self.entries = 0
                self._lock = threading.Lock()

            def __enter__(self):
                self._lock.acquire()
                self.entries += 1
                return self

            def __exit__(self, *exc):
                self._lock.release()
                return False

        counter = Counter()
        monkeypatch.setattr(drv, "_WARM_UP_LOCK", counter)
        return counter

    def test_only_the_first_pass_over_a_device_takes_the_lock(
        self, system, model, monkeypatch
    ):
        """Serialising every step would throw the parallelism away."""
        coord, atype, cell, *_ = system
        fresh = build(long_range=True).eval()
        counter = self._counting_lock(monkeypatch)
        common = {"n_partitions": 2, "devices": [DEVICE] * 2, "launch_threads": True}
        evaluate_partitioned(fresh, coord, atype, cell, **common)
        assert counter.entries == 1, "the first pass must be serialised"
        evaluate_partitioned(fresh, coord, atype, cell, **common)
        evaluate_partitioned(fresh, coord, atype, cell, **common)
        assert counter.entries == 1, "later passes must run lock-free"

    def test_a_different_model_warms_again(self, system, model, monkeypatch):
        """A compressed model reaches kernels the uncompressed one never did,
        so the bookkeeping is per model, not a module-wide flag."""
        coord, atype, cell, *_ = system
        counter = self._counting_lock(monkeypatch)
        common = {"n_partitions": 2, "devices": [DEVICE] * 2, "launch_threads": True}
        for _ in range(2):
            model_n = build(long_range=True).eval()
            evaluate_partitioned(model_n, coord, atype, cell, **common)
        assert counter.entries == 2

    def test_the_first_pass_still_runs_off_the_main_thread(self, system, monkeypatch):
        """Serialised is not the same as sequential: the pool is still used."""
        import threading  # noqa: PLC0415

        from deepmd_lr.nvalchemi import driver as drv  # noqa: PLC0415

        coord, atype, cell, *_ = system
        seen: set[str] = set()
        original = drv.partition_graph

        def record(*args, **kwargs):
            seen.add(threading.current_thread().name)
            return original(*args, **kwargs)

        monkeypatch.setattr(drv, "partition_graph", record)
        evaluate_partitioned(
            build(long_range=True).eval(),
            coord, atype, cell,
            n_partitions=2, devices=[DEVICE] * 2, launch_threads=True,
        )
        assert seen and threading.main_thread().name not in seen, seen

    def test_the_serialised_pass_gives_the_same_numbers(self, system, model):
        """The lock reorders nothing: a warm model and a cold one agree."""
        coord, atype, cell, *_ = system
        common = {
            "n_partitions": 2,
            "devices": [DEVICE] * 2,
            "compute_born": True,
            "launch_threads": True,
        }
        cold_model = build(long_range=True).eval()
        cold = to_numpy(
            evaluate_partitioned(cold_model, coord, atype, cell, **common)
        )
        warm_model = build(long_range=True).eval()
        evaluate_partitioned(warm_model, coord, atype, cell, **common)
        warm = to_numpy(evaluate_partitioned(warm_model, coord, atype, cell, **common))
        for key in ("force", "charge", "born_charge", "atom_energy"):
            np.testing.assert_allclose(warm[key], cold[key], atol=1e-10, err_msg=key)


class TestLazyBackendsAreWarmed:
    """The paths that can be warmed by name are, before any pool exists."""

    def test_the_cuda_linalg_library_is_loaded_for_each_device(self, system):
        """`torch.linalg.inv` of the cell is what the neighbour search runs."""
        from deepmd_lr.nvalchemi import driver as drv  # noqa: PLC0415

        coord, atype, cell, *_ = system
        drv._LINALG_READY.discard(torch.device(DEVICE))
        evaluate_partitioned(
            build(long_range=True).eval(),
            coord, atype, cell,
            n_partitions=2, devices=[DEVICE] * 2,
        )
        assert torch.device(DEVICE) in drv._LINALG_READY

    def test_the_probe_runs_once_and_skips_a_host_device(self, monkeypatch):
        """Once loaded it is never touched again, and there is nothing to load
        for a device without CUDA."""
        from deepmd_lr.nvalchemi import driver as drv  # noqa: PLC0415

        calls = []
        monkeypatch.setattr(
            torch.linalg, "inv", lambda t: calls.append(t.device) or torch.eye(3)
        )
        drv._LINALG_READY.discard(torch.device(DEVICE))
        drv._ensure_cuda_linalg_loaded(DEVICE)
        drv._ensure_cuda_linalg_loaded(DEVICE)
        drv._ensure_cuda_linalg_loaded("cpu")
        assert len(calls) == 1


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

        def refuse(*args, **kwargs):  # noqa: ANN002, ANN003
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
    error on 2 ranks and 36.5% on 8 (LAMMPS.md). So it is evaluated once on
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
        from .test_joint_model import build, frame  # noqa: PLC0415

        model = build()
        coord, atype, cell = frame(n_mol=n_mol)
        return (model, coord[0].cpu().numpy(), atype[0].cpu().numpy(),
                cell[0].cpu().numpy())

    @staticmethod
    def _run(model, c, a, cl, devices, *, force_fallback=False, **kw):
        import deepmd_lr.nvalchemi.driver as DRV  # noqa: PLC0415
        from deepmd_lr.nvalchemi import DistributedEvaluator  # noqa: PLC0415

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
        import deepmd_lr.nvalchemi.driver as DRV  # noqa: PLC0415

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
