# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate a model partition by partition and assemble the global result.

The short-range half is split across partitions; the long-range half is
gathered onto one device rather than distributed. That is a measurement, not
a shortcut: at 6520 atoms the descriptor costs 883 ms and the reciprocal term
73 ms, so the descriptor is what is worth splitting. Gathering costs about
104 KB in and 130 KB out.

Two things are necessarily global and so happen after the gather:

- **The total-charge projection.** ``sum_i q_i = Q_total`` is a constraint
  over every atom, so it cannot be applied per partition. Partitions produce
  *raw* charges; the projection runs once on the assembled vector.
- **The reciprocal sum itself**, which is a structure factor over every atom.
  This is exactly what does not survive spatial decomposition -- splitting it
  across ranks gave 2.0% error at 2 ranks and 36.5% at 8 (see LAMMPS.md).

Devices are handled DistMLIP-style: one process, tensors moved with ``.to()``,
no ``torch.distributed``. Passing the same device several times
(``["cuda:0"] * 4``) is a supported configuration and is how the partitioning
logic is tested on a single-GPU machine.
"""

import copy
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
from deepmd_lr.fitting.charge import (
    project_to_total_charge,
)

__all__ = [
    "device_replicas",
    "evaluate_partitioned",
    "partition_charges",
    "partition_graph",
]


def device_replicas(model, devices) -> dict:
    """One model per distinct device.

    Moving only the *tensors* to each device is not enough: the weights have to
    be there too, or a partition on ``cuda:1`` meets parameters still on
    ``cuda:0``. Single-GPU testing hides this, because every "device" is the
    same one.

    The model that is already on the target device is used as-is rather than
    copied, so the common single-device case allocates nothing extra. Replicas
    are read-only here -- this is inference -- so there is no gradient to
    reduce across them.
    """
    home = next(model.parameters()).device
    replicas = {}
    for device in {torch.device(d) for d in devices}:
        replicas[device] = (
            model if device == home else copy.deepcopy(model).to(device).eval()
        )
    return replicas


def partition_charges(model, graph, node_type, n_owned: int):
    """Raw (unprojected) charges for one partition's owned atoms.

    Runs the shared descriptor and the charge head on the subgraph, mirroring
    ``LongRangeEnergyModel._predict_charges``. Only owned entries are kept:
    a ghost's neighbour shell is incomplete here, and its charge is produced
    correctly by whichever partition owns it.

    The projection is deliberately NOT applied -- it is a global constraint,
    handled once the partitions are assembled.
    """
    descriptor = model.atomic_model.descriptor
    charge_spin_kwargs = (
        {"charge_spin": None} if model.atomic_model.supports_charge_spin else {}
    )
    features, rot_mat = descriptor.call_graph(
        graph,
        node_type,
        type_embedding=descriptor.graph_type_embedding_table(),
        **charge_spin_kwargs,
    )
    raw = model.charge_fitting.call_graph(features, node_type, gr=rot_mat)["charge"]
    return raw.reshape(-1)[:n_owned]


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
    gather_device: Any | None = None,
    partitions: list[Partition] | None = None,
) -> dict[str, Any]:
    """Run ``model`` over spatial partitions and assemble the global result.

    Parameters
    ----------
    gather_device
        Where the long-range term is evaluated. Defaults to the first device.
    partitions
        Reuse a previously built partitioning instead of rebuilding. Valid as
        long as its halo still covers every neighbour -- see the skin logic in
        :class:`~deepmd_lr.distributed.calculator.DistLongRangeCalculator`.

    Returns
    -------
    dict
        ``energy``, ``force`` ``(N, 3)``, ``atom_energy`` ``(N,)``, and when the
        model carries a long-range term also ``charge`` ``(N,)``,
        ``energy_short_range``, ``energy_long_range``, and ``dE_dq`` ``(N,)``
        -- the last is what Phase C backpropagates through each partition's
        charge head.
    """
    rcut = float(model.get_rcut())
    parts = (
        partition_system(coord, cell, rcut, n_partitions, pbc)
        if partitions is None
        else partitions
    )
    if devices is None:
        devices = [next(model.parameters()).device] * len(parts)
    if len(devices) != len(parts):
        raise ValueError(
            f"{len(devices)} devices for {len(parts)} partitions; pass one "
            "device per partition (repeats are allowed)"
        )

    replicas = device_replicas(model, devices)
    long_range = getattr(model, "has_long_range", False)
    n_atoms = len(coord)
    force = np.zeros((n_atoms, 3))
    atom_energy = np.zeros(n_atoms)
    raw_charge = np.zeros(n_atoms)
    energy = 0.0

    for part, device in zip(parts, devices, strict=True):
        local = replicas[torch.device(device)]
        graph, node_type = partition_graph(part, coord, atype, rcut, device, dtype)
        n_node = torch.tensor([part.n_node], dtype=torch.int64, device=device)
        n_local = torch.tensor([part.n_owned], dtype=torch.int64, device=device)

        out = local.forward_common_lower_graph(
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

        if long_range:
            raw_charge[part.owned] = (
                partition_charges(local, graph, node_type, part.n_owned)
                .detach()
                .cpu()
                .numpy()
            )

    result = {"energy": energy, "force": force, "atom_energy": atom_energy}
    if not long_range:
        return result

    gather = gather_device if gather_device is not None else devices[0]
    result["energy_short_range"] = energy
    result.update(
        _long_range_global(
            replicas[torch.device(gather)],
            coord,
            cell,
            raw_charge,
            gather,
            dtype,
        )
    )
    result["energy"] = result["energy_short_range"] + result["energy_long_range"]
    result["force"] = result["force"] + result["force_long_range"]

    result["force_charge_response"] = _charge_response_force(
        replicas, parts, devices, coord, atype, result["dE_dq"], rcut, dtype
    )
    result["force"] = result["force"] + result["force_charge_response"]
    return result


def _charge_response_force(
    replicas, parts, devices, coord, atype, dE_dq, rcut, dtype
) -> np.ndarray:
    r"""The force from charges shifting as atoms move.

    Because the charges are predicted rather than fixed, the long-range force
    has a second term beyond the electrostatic one:

    .. math::
        F_i \supset -\sum_j \frac{\partial E}{\partial q_j}
                              \frac{\partial q_j}{\partial r_i}

    It is not a correction: measured here it is ~3e-2 eV/A against a direct
    long-range force of ~2.5e-3, an order of magnitude larger.

    **This is the term that keeps the scheme parallel.** ``dq_j/dr_i`` is
    nonzero only for atoms within ``rcut`` of j, so although the energy is
    long-ranged its charge derivative is *short*-ranged. Each partition
    backpropagates only its own owned atoms' cotangents through its own charge
    head, on its own device -- no autograd graph spans devices and no
    communication is needed beyond the halo already present.

    The cotangent passes through the global projection first. With
    ``q_i = q~_i + (Q - sum_k q~_k) / N``, the chain rule gives
    ``dE/dq~_j = dE/dq_j - mean(dE/dq)``: a single mean subtraction.
    """
    cotangent = dE_dq - dE_dq.mean()
    force = np.zeros((len(coord), 3))

    for part, device in zip(parts, devices, strict=True):
        node_coord = torch.tensor(
            part.coordinates(coord)[None, ...],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        node_type = torch.tensor(
            part.types(atype)[None, ...], dtype=torch.int64, device=device
        )
        graph = build_neighbor_graph(node_coord, node_type, None, rcut)
        graph.edge_mask = graph.edge_mask & (graph.edge_index[1] < part.n_owned)

        raw = partition_charges(
            replicas[torch.device(device)], graph, node_type.reshape(-1), part.n_owned
        )
        weights = torch.tensor(
            cotangent[part.owned], dtype=raw.dtype, device=device
        )
        (gradient,) = torch.autograd.grad((raw * weights).sum(), node_coord)
        np.add.at(
            force,
            part.global_ids(),
            -gradient.reshape(part.n_node, 3).detach().cpu().numpy(),
        )
    return force


def _long_range_global(model, coord, cell, raw_charge, device, dtype) -> dict:
    """The global half: project the charges, then evaluate the reciprocal sum.

    Returns the direct force ``-dE/dr`` at fixed charge, and ``dE/dq``. The
    remaining *charge-response* force, ``-sum_j (dE/dq_j) dq_j/dr_i``, is
    short-ranged and is added per partition in Phase C -- taking ``dE/dq`` out
    as a plain value here is what keeps the autograd graph on one device.
    """
    n_atoms = len(coord)
    coord_t = torch.tensor(
        coord[None, ...], dtype=dtype, device=device, requires_grad=True
    )
    cell_t = torch.tensor(cell.reshape(1, 3, 3), dtype=dtype, device=device)
    charge_t = torch.tensor(
        raw_charge[None, ...], dtype=dtype, device=device, requires_grad=True
    )

    charge = project_to_total_charge(charge_t, 0.0)
    atom_energy = model.pme(coord_t, charge, cell_t)
    total = atom_energy.sum()
    grad_coord, grad_charge = torch.autograd.grad(total, [coord_t, charge_t])

    return {
        "charge": charge.reshape(-1).detach().cpu().numpy(),
        "energy_long_range": float(total),
        "atom_energy_long_range": atom_energy.reshape(-1).detach().cpu().numpy(),
        "force_long_range": -grad_coord.reshape(n_atoms, 3).detach().cpu().numpy(),
        "dE_dq": grad_charge.reshape(-1).detach().cpu().numpy(),
    }
