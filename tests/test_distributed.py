"""Phase A: partitioned evaluation must reproduce the serial answer.

The claim is partition-count invariance: 1, 2, 4 and 8 partitions must all
agree with the undivided model, to the floor set by GPU nondeterminism. This
is precisely the test the LAMMPS domain decomposition failed -- splitting a
reciprocal sum across ranks gave 2.0% error at 2 ranks and 36.5% at 8
(see LAMMPS.md) -- so here it has to pass exactly, not approximately.

Phase A covers the short-range model. The long-range term arrives in Phase B.
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
                within = np.flatnonzero(
                    (np.linalg.norm(delta, axis=1) < RCUT)
                )
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
