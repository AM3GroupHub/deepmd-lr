# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate a model partition by partition and assemble the global result.

Phase A scope: the short-range model only. The long-range term is added in
Phase B, where it is gathered onto one device rather than distributed --
profiling put it at 73 ms against 883 ms for the descriptor, so the descriptor
is the part worth splitting.

Devices are handled DistMLIP-style: one process, tensors moved with ``.to()``,
no ``torch.distributed``. Passing the same device several times
(``["cuda:0"] * 4``) is a supported configuration and is how the partitioning
logic is tested on a single-GPU machine.
"""

from typing import (
    Any,
)

import numpy as np
import torch
from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)

from deepmd_lr.distributed.partition import (
    Partition,
    partition_system,
)

__all__ = ["evaluate_partitioned", "partition_graph"]


def partition_graph(part: Partition, coord: np.ndarray, atype: np.ndarray,
                    rcut: float, device: Any, dtype=torch.float64):
    """Build one partition's neighbour graph, keeping only owned destinations.

    Built with ``box=None`` -- non-periodic. The halo already carries explicit
    periodic images, so applying the boundary again inside the partition would
    double-count it.

    Edges whose destination is a ghost are masked out. That is what makes the
    assembled sum exact: each edge survives in exactly one partition, the one
    owning its destination.
    """
    node_coord = torch.tensor(
        part.coordinates(coord)[None, ...], dtype=dtype, device=device
    )
    node_type = torch.tensor(
        part.types(atype)[None, ...], dtype=torch.int64, device=device
    )
    graph = build_neighbor_graph(node_coord, node_type, None, rcut)

    owned_dst = graph.edge_index[1] < part.n_owned
    graph.edge_mask = graph.edge_mask & owned_dst
    return graph, node_type.reshape(-1)


def evaluate_partitioned(
    model: Any,
    coord: np.ndarray,
    atype: np.ndarray,
    cell: np.ndarray,
    n_partitions: int = 1,
    devices: list[Any] | None = None,
    pbc=(True, True, True),
    dtype=torch.float64,
) -> dict[str, Any]:
    """Run ``model`` over spatial partitions and assemble energy and forces.

    Returns
    -------
    dict
        ``energy`` (float), ``force`` ``(N, 3)``, ``atom_energy`` ``(N,)``.
    """
    rcut = float(model.get_rcut())
    parts = partition_system(coord, cell, rcut, n_partitions, pbc)
    if devices is None:
        devices = [next(model.parameters()).device] * len(parts)
    if len(devices) != len(parts):
        raise ValueError(
            f"{len(devices)} devices for {len(parts)} partitions; pass one "
            "device per partition (repeats are allowed)"
        )

    n_atoms = len(coord)
    force = np.zeros((n_atoms, 3))
    atom_energy = np.zeros(n_atoms)
    energy = 0.0

    for part, device in zip(parts, devices, strict=True):
        graph, node_type = partition_graph(part, coord, atype, rcut, device, dtype)
        n_node = torch.tensor([part.n_node], dtype=torch.int64, device=device)
        n_local = torch.tensor([part.n_owned], dtype=torch.int64, device=device)

        out = model.forward_common_lower_graph(
            node_type,
            n_node,
            n_local,
            graph.edge_index,
            graph.edge_vec,
            graph.edge_mask,
        )

        ids = part.global_ids()
        # Energy: owned atoms only. Each atom is scored by exactly one owner.
        per_atom = out["energy"].reshape(-1).detach().cpu().numpy()
        atom_energy[part.owned] += per_atom[: part.n_owned]
        energy += float(per_atom[: part.n_owned].sum())

        # Force: every node, ghosts included. A ghost carries the reaction to
        # an owned atom's edge, and that belongs to the ghost's owner --
        # scatter-add, exactly as ghost-force reverse communication in a
        # domain-decomposed MD code.
        # `energy_derv_r` is already the force, not the energy gradient --
        # `_to_public_keys` maps it straight onto the public "force" key with
        # no sign change.
        node_force = out["energy_derv_r"].reshape(part.n_node, 3)
        np.add.at(force, ids, node_force.detach().cpu().numpy())

    return {"energy": energy, "force": force, "atom_energy": atom_energy}
