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
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.distributed import (
    DistributedForces,
    evaluate_partitioned,
    partition_graph,
    partition_system,
)

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
            out = evaluate_partitioned(
                model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
            )
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
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        )
        assert np.abs(out["atom_energy"] - reference).max() < 1e-8

    def test_rejects_device_count_mismatch(self, system):
        coord, atype, cell, *_ = system
        model = build(long_range=False)
        with pytest.raises(ValueError, match="one device per partition"):
            evaluate_partitioned(
                model, coord, atype, cell, n_partitions=4, devices=[DEVICE]
            )


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
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        )
        assert np.abs(out["charge"] - reference).max() < 1e-12

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_total_charge_constraint_survives_partitioning(self, system, joint, n):
        """The constraint is global; a per-partition projection would break it."""
        coord, atype, cell, *_ = system
        model, _ = joint
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        )
        assert abs(out["charge"].sum()) < 1e-10

    @pytest.mark.parametrize("n", PARTITION_COUNTS)
    def test_long_range_energy_matches_serial(self, system, joint, n):
        coord, atype, cell, *_ = system
        model, serial = joint
        reference = float(serial["energy_long_range"].reshape(-1)[0])
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        )
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
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        )
        assert np.abs(out["force_long_range"] - reference).max() < 1e-12

    def test_charge_gradient_is_available_and_nonzero(self, system, joint):
        """dE/dq is what Phase C backpropagates; taking it out as a value is
        what keeps the autograd graph on a single device."""
        coord, atype, cell, *_ = system
        model, _ = joint
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        )
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
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        )
        assert np.abs(out["force"] - reference).max() < 1e-12

    def test_charge_response_dominates_the_direct_term(self, system, joint):
        """Guard against it being quietly dropped or scaled to nothing.

        It is not a small correction here: ~7e-3 eV/A rms against ~2.5e-3 for
        the direct long-range force. A regression that zeroed it would still
        leave energies right and forces plausible.
        """
        coord, atype, cell, *_ = system
        model, _ = joint
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        )
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
            return evaluate_partitioned(
                model, positions, atype, cell,
                n_partitions=n, devices=[DEVICE] * n,
            )["energy"]

        force = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=n, devices=[DEVICE] * n
        )["force"]

        step = 1e-5
        for atom in range(0, len(coord), 11):
            for axis in range(3):
                plus, minus = coord.copy(), coord.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                numeric = -(energy(plus) - energy(minus)) / (2 * step)
                assert abs(numeric - force[atom, axis]) < 1e-6


class TestDistributedForces:
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
        return DistributedForces(model, system[1], system[2], **kwargs)

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

        from deepmd_lr.distributed import FS

        driver = self._driver(model, system, skin=1.0)
        positions = torch.tensor(
            system[0], dtype=torch.float64, device=DEVICE
        ).contiguous()
        velocities = torch.zeros_like(positions)
        masses = torch.tensor(
            driver.masses(), dtype=torch.float64, device=DEVICE
        )
        dt = torch.tensor([0.25 * FS], dtype=torch.float64, device=DEVICE)

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
