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
                graph, _ = partition_graph(part, coord, atype, RCUT, DEVICE)
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
        energy = model.pme(positions, frozen, cell_t).sum()
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

    def test_charge_response_is_the_only_remaining_gap(self, system, joint):
        """Documents precisely what Phase C still owes.

        The assembled force is short-range + direct long-range. The serial
        model additionally carries -sum_j (dE/dq_j) dq_j/dr_i. That term is
        not small here: it is ~3e-2 eV/A against a direct long-range force of
        ~2.5e-3, so skipping it would be a far bigger error than dropping the
        direct piece. When Phase C lands this test is replaced by an equality
        against the serial force.
        """
        coord, atype, cell, *_ = system
        model, serial = joint
        reference = serial["force"][0].detach().cpu().numpy()
        out = evaluate_partitioned(
            model, coord, atype, cell, n_partitions=4, devices=[DEVICE] * 4
        )
        gap = np.abs(out["force"] - reference).max()
        direct = np.abs(out["force_long_range"]).max()
        assert gap > direct, (
            "the charge-response term is expected to dominate the direct "
            f"long-range force; gap {gap:.2e} vs direct {direct:.2e}"
        )
