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
    DistributedMD,
    evaluate_partitioned,
    partition_graph,
    partition_system,
)
from deepmd_lr.distributed.md import (
    BOLTZMANN_EV_PER_K,
    kelvin_to_energy,
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


class TestDistributedMD:
    """MD on nvalchemiops' integrators, over a reused partitioning.

    There is no ASE calculator in this path. ASE's interface returns energy
    and forces for one configuration at a time, which drags the trajectory
    state through the host every step and cannot use the on-device state the
    partitioned evaluation already holds.
    """

    @staticmethod
    def _md(model, system, **kwargs):
        coord, atype, cell = system[0], system[1], system[2]
        kwargs.setdefault("devices", [str(DEVICE)])
        return DistributedMD(model, coord, atype, cell, **kwargs)

    @pytest.fixture(scope="class")
    def model(self):
        return build(long_range=True).eval()

    def test_forces_match_the_serial_calculator(self, model, system):
        """The driver must not change the physics -- only how it is scheduled."""
        coord, atype, cell, coord_t, atype_t, cell_t = system
        serial = model(coord_t, atype_t, box=cell_t)
        md = self._md(model, system)
        assert abs(md.potential_energy - float(serial["energy"])) < 1e-8
        reference = serial["force"][0].detach().cpu().numpy()
        assert np.abs(md.forces.cpu().numpy() - reference).max() < 1e-8

    def test_nose_hoover_chain_absorbs_energy(self, model, system):
        """The chain must actually couple, and its energy is real.

        `chain_energy` is checked against the MTK expression directly --
        `sum_k Q_k eta_dot_k^2 / 2 + N_dof kT eta_1 + kT sum_{k>1} eta_k` --
        because a thermostat that silently does nothing still produces a
        plausible-looking trajectory.
        """
        md = self._md(
            model, system, ensemble="nvt-nose-hoover", timestep=0.5,
            temperature=300.0, tdamp=50.0, seed=1,
        )
        md.set_temperature(300.0, seed=1)
        assert md.chain_energy() == 0.0
        md.run(20)
        assert md.chain_energy() > 0.0, "the chain never coupled"

        chain = md._chain
        eta = chain["eta"].cpu().numpy()
        eta_dot = chain["eta_dot"].cpu().numpy()
        q = chain["eta_mass"].cpu().numpy()
        kT = float(md._kT)
        ndof = float(chain["ndof"])
        expected = (
            0.5 * float((q * eta_dot**2).sum())
            + ndof * kT * float(eta[0])
            + kT * float(eta[1:].sum())
        )
        assert abs(md.chain_energy() - expected) < 1e-9 * max(1.0, abs(expected))

    def test_nose_hoover_masses_follow_the_coupling_time(self, model, system):
        """Q_1 = N_dof kT tau^2 and Q_k = kT tau^2, from the MTK paper."""
        md = self._md(
            model, system, ensemble="nvt-nose-hoover",
            temperature=300.0, tdamp=50.0, chain_length=3,
        )
        q = md._chain["eta_mass"].cpu().numpy()
        kT = float(md._kT)
        tau = float(md._chain["tau"])
        ndof = float(md._chain["ndof"])
        assert abs(q[0] - ndof * kT * tau**2) < 1e-9 * q[0]
        for later in q[1:]:
            assert abs(later - kT * tau**2) < 1e-9 * later

    def test_nose_hoover_drives_the_temperature_toward_the_target(
        self, model, system
    ):
        """Started hot, the chain has to take energy out."""
        md = self._md(
            model, system, ensemble="nvt-nose-hoover", timestep=0.5,
            temperature=100.0, tdamp=20.0, seed=2,
        )
        md.set_temperature(1500.0, seed=2)
        hot = md.temperature_k()
        md.run(60)
        assert md.temperature_k() < hot, (
            f"T went {hot:.0f} -> {md.temperature_k():.0f} K with a 100 K target"
        )
        assert md.chain_energy() > 0.0

    @pytest.mark.parametrize("ensemble", ["nve", "nvt-langevin", "nvt-nose-hoover"])
    def test_both_ensembles_advance_the_system(self, model, system, ensemble):
        md = self._md(model, system, ensemble=ensemble, timestep=0.5)
        md.set_temperature(300.0)
        before = md.positions.clone()
        md.run(5)
        assert md.step_count == 5
        assert float((md.positions - before).norm(dim=1).max()) > 0.0

    def test_nve_conserves_energy(self, model, system):
        """The integrator and the force must agree about what energy they are
        moving on. Getting the unit system wrong shows up here first."""
        md = self._md(model, system, ensemble="nve", timestep=0.25)
        md.set_temperature(300.0)
        start = md.total_energy()
        md.run(30)
        drift = abs(md.total_energy() - start) / md.positions.shape[0] * 1000
        assert drift < 1.0, f"drift {drift:.4f} meV/atom over 30 steps"

    def test_temperature_is_kelvin_at_the_boundary(self, model, system):
        """`k_B T` in eV is what the kernels take; Kelvin is what we take.

        nvalchemiops documents every dynamics temperature as "k_B*T in energy
        units", the opposite of the heavy `nvalchemi-toolkit`, whose
        thermostats took Kelvin. Confusing them is a factor of ~11600, so the
        conversion is pinned rather than trusted.
        """
        md = self._md(model, system)
        n_atoms = md.positions.shape[0]
        md.set_temperature(300.0, seed=3)
        exact = 1.5 * n_atoms * BOLTZMANN_EV_PER_K * 300.0
        # One draw fluctuates by ~sqrt(2/3N); the failure this guards against
        # is four orders of magnitude, not four percent.
        assert 0.5 < md.kinetic_energy() / exact < 2.0
        assert abs(kelvin_to_energy(300.0) - 0.0258520) < 1e-6

    def test_reused_partitioning_stays_correct_under_motion(self, model, system):
        """The point of the skin: atoms move, the halo is reused, and the
        forces stay exact until the Verlet criterion trips."""
        md = self._md(model, system, skin=1.0, ensemble="nve")
        md.set_temperature(300.0)
        md.run(10)
        # Re-evaluate from scratch at the current geometry.
        fresh = self._md(model, (md.positions.cpu().numpy(), system[1], system[2]))
        assert np.abs(md.forces.cpu().numpy() - fresh.forces.cpu().numpy()).max() < 1e-8

    def test_forcing_a_rebuild_every_step_changes_nothing(self, model, system):
        """Caching must be an optimisation, not a physics knob."""
        cached = self._md(model, system, skin=1.0, ensemble="nve")
        eager = self._md(model, system, skin=1.0, repartition_every=1, ensemble="nve")
        for md in (cached, eager):
            md.set_temperature(300.0, seed=7)
        cached.run(6)
        eager.run(6)
        assert eager.rebuild_count > cached.rebuild_count
        assert np.abs(
            cached.positions.cpu().numpy() - eager.positions.cpu().numpy()
        ).max() < 1e-10

    def test_caching_actually_avoids_rebuilds(self, model, system):
        md = self._md(model, system, skin=5.0, ensemble="nve")
        md.set_temperature(50.0)
        md.run(8)
        assert md.rebuild_count < 8

    def test_exposes_the_energy_split_and_charges(self, model, system):
        md = self._md(model, system)
        for key in ("energy_short_range", "energy_long_range", "charge"):
            assert key in md.results

    def test_rejects_an_unknown_ensemble(self, model, system):
        with pytest.raises(ValueError, match="unknown ensemble"):
            self._md(model, system, ensemble="npt")

    def test_rejects_an_empty_device_list(self, model, system):
        with pytest.raises(ValueError, match="at least one device"):
            self._md(model, system, devices=[])

    def test_rejects_a_negative_skin(self, model, system):
        with pytest.raises(ValueError, match="skin must be non-negative"):
            self._md(model, system, skin=-1.0)
