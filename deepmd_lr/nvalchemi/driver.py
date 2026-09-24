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

import contextlib
import copy
from typing import (
    Any,
)

import numpy as np
import torch
from deepmd.dpmodel.utils.neighbor_graph import (
    segment_sum,
)
from deepmd.dpmodel.utils.neighbor_graph.csr import (
    attach_edge_csr,
)
from deepmd.dpmodel.utils.neighbor_graph.graph import (
    NeighborGraph,
)

from deepmd_lr.born import (
    screening_factor,
)
from deepmd_lr.fitting.charge import (
    charge_mask,
    project_to_total_charge,
    resolve_charge_spin,
)
from deepmd_lr.neighbors import (
    build_neighbor_graph_fast,
)
from deepmd_lr.nvalchemi.partition import (
    Partition,
    partition_system,
)

__all__ = [
    "device_replicas",
    "evaluate_partitioned",
    "partition_charges",
    "partition_graph",
]


def _current_device(device) -> contextlib.AbstractContextManager:
    """Make ``device`` the CUDA *current* device for the block.

    Tensor placement alone is not enough for the compressed DPA4C operator:
    measured on two H200s, a compressed model whose every buffer sits on
    ``cuda:1`` returns forces wrong by 5 eV/A when ``cuda:0`` is current, and
    the right ones (1e-5 of the uncompressed reference) once ``cuda:1`` is
    current -- the uncompressed path is indifferent. The operator evidently
    binds something (a stream, a workspace) to the current device rather than
    to its arguments' device. So every per-partition block runs with its own
    device current. Cheap: it is a thread-local switch, not a synchronisation.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.cuda.device(device)


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
        if device == home:
            replicas[device] = model
            continue
        with _current_device(device):
            replicas[device] = copy.deepcopy(model).to(device).eval()
    return replicas


def _usable_capture(captured) -> bool:
    """Whether the capture can carry the charge response.

    An ``edge_vec`` that does not require grad means the energy path did not
    route through the graph lower, so there is nothing to differentiate and
    the backbone must be run again -- the same guard the serial path applies.
    """
    return bool(captured) and captured["edge_vec"].requires_grad


def charges_from_capture(model, captured, node_type, n_owned: int):
    """Raw charges from already-computed descriptor features."""
    raw = model.charge_fitting.call_graph(
        captured["features"], node_type, gr=captured["rot_mat"]
    )["charge"]
    return raw.reshape(-1)[:n_owned]


def partition_charges(model, graph, node_type, n_owned: int, charge_spin=None):
    """Raw (unprojected) charges for one partition's owned atoms.

    Runs the shared descriptor and the charge head on the subgraph, mirroring
    ``LongRangeEnergyModel._predict_charges``. Only owned entries are kept:
    a ghost's neighbour shell is incomplete here, and its charge is produced
    correctly by whichever partition owns it.

    The projection is deliberately NOT applied -- it is a global constraint,
    handled once the partitions are assembled.
    """
    descriptor = model.atomic_model.descriptor
    if getattr(descriptor, "compress", False):
        # Compressed inference is defined only on the captured route: this
        # second pass would run the uncompressed math on a CSR-less graph
        # while the model still claims to be compressed.
        raise RuntimeError(
            "the fallback charge pass was reached on a compressed model; the "
            "shared-backbone capture should have been taken. Check "
            "DP_CUDA_INFER (it must be exactly 1)."
        )
    charge_spin_kwargs = (
        {"charge_spin": charge_spin} if model.atomic_model.supports_charge_spin else {}
    )
    features, rot_mat = descriptor.call_graph(
        graph,
        node_type,
        type_embedding=descriptor.graph_type_embedding_table(),
        **charge_spin_kwargs,
    )
    raw = model.charge_fitting.call_graph(features, node_type, gr=rot_mat)["charge"]
    return raw.reshape(-1)[:n_owned]


def partition_graph(
    part: Partition,
    coord: np.ndarray,
    atype: np.ndarray,
    rcut: float,
    device: Any,
    dtype=torch.float64,
    requires_grad: bool = False,
    strain: torch.Tensor | None = None,
):
    """Build one partition's neighbour graph, keeping only owned destinations.

    Built with ``box=None`` -- non-periodic. The halo already carries explicit
    periodic images, so applying the boundary again inside the partition would
    double-count it.

    Edges whose destination is a ghost are masked out. That is what makes the
    assembled sum exact: each edge survives in exactly one partition, the one
    owning its destination.

    With ``requires_grad``, the node coordinates are tracked and returned so a
    later backward can reach them. That lets the charge-response term reuse
    this graph instead of rebuilding it -- the search is the expensive part,
    and it is identical either way.

    With ``strain`` (a ``(3, 3)`` matrix, typically ``I + sym(eps)``), the
    node coordinates are deformed by it before the search. That is all the
    virial needs here: a ghost's Cartesian offset is a lattice-vector
    combination ``n A``, which deforms as ``n A (I+eps)^T`` -- the same
    transform as a coordinate -- so straining the assembled node set is
    exact, and the strain reaches ``edge_vec`` through the coordinates rather
    than through a box. That is why the partitioned virial does not need a
    cell-differentiable neighbour builder, unlike the serial path.

    Returns ``(graph, node_type_flat, node_coord)``. ``node_coord`` is the
    unstrained leaf, so a backward reaches both it and the strain.

    ``coord`` may be a device tensor or a numpy array; either way the node
    set is gathered on ``device`` with :meth:`Partition.views`, never
    concatenated on the host.
    """
    # One gather, on the device: owned nodes plus ghosts shifted into place,
    # taken from the global coordinate tensor with this partition's cached
    # index tensors. A numpy argument is uploaded once, here, so there is no
    # second implementation that assembles nodes on the host.
    coord_t = (
        coord
        if isinstance(coord, torch.Tensor)
        else torch.as_tensor(np.asarray(coord, dtype=np.float64))
    )
    v = part.views(device, dtype)
    c = coord_t.detach().to(device=device, dtype=dtype)
    nodes = torch.cat([c[v["owned"]], c[v["ghost"]] + v["shift"]])
    node_coord = nodes.unsqueeze(0).clone().requires_grad_(requires_grad)
    node_type = part.types_on(atype, device).unsqueeze(0)

    search_coord = (
        node_coord
        if strain is None
        else torch.bmm(node_coord, strain.transpose(-1, -2).unsqueeze(0))
    )

    # The search is the expensive half of a step -- a cell list plus eight
    # blocking device-to-host reads -- and it does not have to run every step.
    # The partition already carries a skin: every pair within `rcut` of an
    # owned atom is inside a list searched at `rcut + skin` until some atom has
    # moved `skin / 2`, which is precisely when the partition itself is
    # rebuilt. So search once at the padded cutoff, keep the integer topology
    # on the partition, and let each later step recompute only the geometry.
    cached = part.edges_on(device) if part.skin > 0.0 else None
    if cached is None:
        graph = build_neighbor_graph_fast(
            search_coord.detach(), node_type, None, rcut + max(part.skin, 0.0)
        )
        # Fold in what never changes between rebuilds: the search's own
        # padding, and the rule that a partition only owns edges whose centre
        # it owns. What is left to decide per step is the cutoff alone.
        keep = graph.edge_mask & (graph.edge_index[1] < part.n_owned)
        if part.skin > 0.0:
            part.cache_edges(device, graph.edge_index, keep, graph.n_node)
        edge_index, n_node = graph.edge_index, graph.n_node
    else:
        edge_index, keep, n_node = cached

    src, dst = edge_index[0], edge_index[1]
    flat = search_coord.reshape(-1, 3)
    # `r_src - r_dst`, the graph's own convention, and the only geometry the
    # model sees -- so the backward reaches `node_coord` (and the strain)
    # through this whether the topology was searched now or three steps ago.
    edge_vec = flat[src] - flat[dst]
    edge_mask = keep & (edge_vec.detach().square().sum(-1) <= rcut * rcut)
    # Padding is zero-length by construction upstream (`pad_and_guard_edges`
    # points dummies at node 0 with a zero vector), so an edge dropped by the
    # cutoff is zeroed rather than left long: the descriptor's tables are
    # built for `[0, rcut]` and a masked-but-oversized vector would be a new
    # thing to ask of them.
    edge_vec = edge_vec * edge_mask.unsqueeze(-1)
    graph = NeighborGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
    )
    return graph, node_type.reshape(-1), node_coord


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
    replicas: dict | None = None,
    compute_virial: bool = False,
    charge_spin: torch.Tensor | tuple[float, float] | None = None,
    compute_born: bool = False,
    eps_inf: float = 1.0,
) -> dict[str, Any]:
    """Run ``model`` over spatial partitions and assemble the global result.

    Everything is assembled on the gather device and **returned there**:
    per-partition forces, energies and charges are moved device-to-device and
    scatter-added with ``index_add_`` -- the ghost-force reverse communication
    of a domain-decomposed code, done as a device primitive rather than a
    serial host loop. There is no host-side assembly path; a caller that wants
    numpy arrays converts explicitly with :func:`to_numpy`, which makes the
    download a visible line rather than a hidden default.

    Parameters
    ----------
    gather_device
        Where the long-range term is evaluated. Defaults to the first device.
    partitions
        Reuse a previously built partitioning instead of rebuilding. Valid as
        long as its halo still covers every neighbour -- see the skin logic in
        :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator`.
    charge_spin
        Global (total charge, spin multiplicity), shared by every partition.
        If omitted, use the model's configured default condition.
    eps_inf
        High-frequency dielectric constant. The returned ``charge`` and
        ``born_charge`` are multiplied by ``sqrt(eps_inf)`` -- the screened
        charges ``q~`` for which the Coulomb energy reads
        ``q~_i q~_j / (4 pi eps_0 eps_inf r)``. Energies and forces are
        evaluated with the model's own charges and are unchanged. ``1.0`` is
        vacuum. See :func:`deepmd_lr.born.screening_factor`.
    Returns
    -------
    dict
        Device tensors on the gather device, scalars as 0-d tensors:
        ``energy``, ``force`` ``(N, 3)``, ``atom_energy`` ``(N,)``, and when the
        model carries a long-range term also ``charge`` ``(N,)``,
        ``energy_short_range``, ``energy_long_range``, and ``dE_dq`` ``(N,)``
        -- the last is what Phase C backpropagates through each partition's
        charge head.
    """
    rcut = float(model.get_rcut())
    coord_np = (
        None if isinstance(coord, torch.Tensor) else np.asarray(coord, dtype=np.float64)
    )
    parts = (
        partition_system(
            coord_np if coord_np is not None else coord.detach().cpu().numpy(),
            cell,
            rcut,
            n_partitions,
            pbc,
        )
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

    if replicas is None:
        replicas = device_replicas(model, devices)
    long_range = getattr(model, "has_long_range", False)
    home = torch.device(gather_device if gather_device is not None else devices[0])
    coord_t = (
        coord.detach().to(device=home, dtype=dtype)
        if isinstance(coord, torch.Tensor)
        else torch.as_tensor(coord_np, dtype=dtype, device=home)
    )
    n_atoms = coord_t.shape[0]
    force = torch.zeros((n_atoms, 3), dtype=dtype, device=home)
    atom_energy = torch.zeros(n_atoms, dtype=dtype, device=home)
    raw_charge = torch.zeros(n_atoms, dtype=dtype, device=home) if long_range else None

    # ---- launch phase -------------------------------------------------------
    # Queue every partition's work WITHOUT touching the host. A `.cpu()` here
    # would synchronise, and the partition on the next device could not start
    # until this one finished -- which is how two GPUs end up running strictly
    # one after the other. Kernel launches are asynchronous and each device has
    # its own stream, so deferring the transfers is what buys the overlap.
    #
    # The loop is single-threaded, and deliberately. Giving each device its own
    # thread did recover some of the overlap the neighbour search's blocking
    # host reads take away -- 1.31x to 1.88x of an ideal 2x, on two A40s at
    # 98,304 atoms -- but underneath it every lazily-initialised library became
    # a race: deepmd-kit's fake-op registration, torch's CUDA linear-algebra
    # loader, warp's kernel build. Four fixes in, a 4-GPU V100 run still lost
    # rows, and the short-range case at 3,456 atoms went from 101 ms a step to
    # 1612, with 12,288 atoms hanging past thirteen minutes. A modest constant
    # factor is not worth that: one thread, in partition order.
    launched: list = [None] * len(parts)
    virial = torch.zeros((3, 3), dtype=dtype, device=home) if compute_virial else None
    # One copy of the coordinates per device, not per partition: every
    # partition on a device reads the same full array and gathers its own rows
    # out of it, so four partitions sharing a GPU would otherwise upload the
    # same thing four times. Done here rather than lazily in the loop because
    # a cross-device copy runs on the SOURCE device's stream, behind whatever
    # is already queued there, and keeping all of them ahead of the forwards
    # is what stops one device's backlog from delaying another's first copy.
    coord_on = {}
    for device in devices:
        dev = torch.device(device)
        if dev not in coord_on:
            with _current_device(dev):
                coord_on[dev] = coord_t.to(device=dev, dtype=dtype)
    def launch_one(index, part, device):
        """One partition's forward, on its own device."""
        with _current_device(device):
            local = replicas[torch.device(device)]
            condition = resolve_charge_spin(
                local, charge_spin, 1, torch.empty((), dtype=dtype, device=device)
            )
            # One strain leaf per partition. They cannot be shared -- a leaf
            # lives on one device -- but the virial is a sum of contributions,
            # so per-device leaves and summed gradients are equivalent.
            eps = strain = None
            if compute_virial:
                eps = torch.zeros(
                    (3, 3), dtype=dtype, device=device, requires_grad=True
                )
                strain = torch.eye(3, dtype=dtype, device=device) + 0.5 * (
                    eps + eps.transpose(-1, -2)
                )
            # `requires_grad` only when the charge-response backward will need it.
            graph, node_type, node_coord = partition_graph(
                part,
                coord_on[torch.device(device)],
                atype,
                rcut,
                device,
                dtype,
                requires_grad=long_range,
                strain=strain,
            )
            n_node = torch.tensor([part.n_node], dtype=torch.int64, device=device)
            n_local = torch.tensor([part.n_owned], dtype=torch.int64, device=device)

            if getattr(local.atomic_model.descriptor, "compress", False):
                # The compressed kernel needs the destination CSR views, attached
                # AFTER the ownership mask: `build_edge_csr` routes masked edges
                # into a padding segment using the mask it is shown, so attaching
                # inside the builder would leave ghost-destination edges in owned
                # rows. `canonicalize=False` leaves `edge_vec` the very tensor the
                # search produced, so the charge-response backward below still
                # reaches `node_coord` through it.
                graph = attach_edge_csr(graph, part.n_node)
            # The model's own capture context: it records the backbone output the
            # charge head reuses, and it is what permits the graph lower -- any
            # caller outside one would be reporting the short-range energy alone.
            capture = (
                local._capturing_backbone()
                if long_range
                else contextlib.nullcontext({})
            )
            with capture as captured:
                out = local.forward_common_lower_graph(
                    node_type,
                    n_node,
                    n_local,
                    graph.edge_index,
                    graph.edge_vec,
                    graph.edge_mask,
                    destination_order=graph.destination_order,
                    destination_row_ptr=graph.destination_row_ptr,
                    source_order=graph.source_order,
                    source_row_ptr=graph.source_row_ptr,
                    destination_sorted=graph.destination_sorted,
                    charge_spin=condition,
                )
            # Charges come from the SAME descriptor pass the energy head just ran,
            # and the tensor is kept live so the charge-response force can
            # backpropagate through it later. Falling back to a second pass only
            # when the capture cannot carry the gradient.
            raw = cap_edge_vec = None
            if long_range:
                if _usable_capture(captured):
                    raw = charges_from_capture(local, captured, node_type, part.n_owned)
                    cap_edge_vec = captured["edge_vec"]
                else:
                    raw = partition_charges(
                        local, graph, node_type, part.n_owned, condition
                    )
            launched[index] = (
                part, device, out, raw, node_coord, eps, graph, cap_edge_vec
            )

    for index, (part, device) in enumerate(zip(parts, devices, strict=True)):
        launch_one(index, part, device)

    # ---- gather phase -------------------------------------------------------
    # Now synchronise. Every device has been given its work, so the transfers
    # below wait on kernels that have been running concurrently.
    # The transfers are device-to-device onto the gather device and the
    # scatter-adds are device primitives, so no partition waits for the host
    # and no GPU idles while a CPU loop runs. `.to()` between devices does not
    # synchronise the host; it orders on the streams.
    for part, _device, out, raw, _node_coord, _eps, _graph, _cap in launched:
        v = part.views(home, dtype)
        if compute_virial:
            # Each edge survives in exactly one partition, so the per-partition
            # virials sum to the global one exactly -- the same ownership
            # argument as the forces.
            virial += out["energy_derv_c_redu"].reshape(3, 3).detach().to(home, dtype)
        # Energy: owned atoms only. Each atom is scored by exactly one owner.
        per_atom = out["energy"].reshape(-1).detach().to(home, dtype)
        atom_energy.index_add_(0, v["owned"], per_atom[: part.n_owned])

        # Force: every node, ghosts included. A ghost carries the reaction to
        # an owned atom's edge, and that belongs to the ghost's owner --
        # scatter-add, exactly as ghost-force reverse communication in a
        # domain-decomposed MD code.
        # `energy_derv_r` is already the force, not the energy gradient --
        # `_to_public_keys` maps it straight onto the public "force" key with
        # no sign change.
        node_force = (
            out["energy_derv_r"].reshape(part.n_node, 3).detach().to(home, dtype)
        )
        force.index_add_(0, v["ids"], node_force)

        if long_range:
            raw_charge.index_copy_(0, v["owned"], raw.detach().to(home, dtype))

    energy = atom_energy.sum()
    result = {"energy": energy, "force": force, "atom_energy": atom_energy}
    if compute_virial:
        result["virial"] = virial.clone()
        result["virial_short_range"] = virial.clone()
    if not long_range:
        return result

    result["energy_short_range"] = energy
    result.update(
        _long_range_global(
            replicas[home],
            coord_t,
            cell,
            raw_charge,
            home,
            dtype,
            compute_virial=compute_virial,
            atype=atype,
            charge_spin=charge_spin,
        )
    )
    result["energy"] = result["energy_short_range"] + result["energy_long_range"]
    result["atom_energy_short_range"] = result["atom_energy"]
    result["atom_energy"] = result["atom_energy"] + result["atom_energy_long_range"]
    result["force"] = result["force"] + result["force_long_range"]

    if compute_born:
        # Before `_charge_response_force`, which backpropagates through the
        # same `raw` tensors without retaining the graph.
        result["born_charge"] = _born_charges_partitioned(
            launched, n_atoms, result["charge"], home, dtype, eps_inf=eps_inf
        )
    # The energy above used the model's own charges; only the report is
    # rescaled to screened charges, and after Born so its delta term saw the
    # unscaled values once and the whole tensor was scaled once.
    factor = screening_factor(eps_inf)
    if factor != 1.0:
        result["charge"] = result["charge"] * factor

    response_force, response_virial = _charge_response_force(
        launched, n_atoms, result["dE_dq"], home, dtype, compute_virial=compute_virial
    )
    result["force_charge_response"] = response_force
    result["force"] = result["force"] + response_force
    if compute_virial:
        result["virial"] = (
            result["virial"] + result["virial_long_range"] + response_virial
        )
        result["virial_charge_response"] = response_virial
    return result


def to_numpy(result: dict) -> dict:
    """Copy a :func:`evaluate_partitioned` result to the host.

    Arrays become numpy, 0-d tensors become Python floats. This is the only
    download in the pipeline and it is the caller's explicit choice: the
    driver assembles and returns on the device, so a step that never reads a
    result on the host never pays for one.
    """
    out = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            value = value.detach()
            out[key] = float(value) if value.ndim == 0 else value.cpu().numpy()
        else:
            out[key] = value
    return out


def _born_charges_partitioned(
    launched, n_atoms: int, charge, home, dtype, *, eps_inf: float = 1.0
) -> torch.Tensor:
    r"""Born effective charge tensors, assembled across partitions.

    The global total-charge projection is not an obstacle here, for the same
    reason it is not one for the energy: it touches only the
    :math:`q_i\delta_{\alpha\beta}` term, and that arrives already projected
    from ``_long_range_global``. The derivative half uses the **raw** charges,
    which are strictly local -- DPA4C is one-hop, so an atom's raw charge sees
    only edges where it is the centre -- and therefore partitions exactly.

    Each partition contributes both halves of the pair sum for every edge it
    owns::

        + 1/2 (r_e (x) g_e)  ->  dst(e)     always owned
        - 1/2 (r_e (x) g_e)  ->  src(e)     owned OR a ghost

    A directed edge survives in exactly one partition, the one owning its
    destination (``partition_graph`` masks the rest), so summing these over
    partitions reproduces the serial tensor with no double counting. The src
    half may land on a ghost, whose contribution belongs to whichever partition
    owns it -- that is what the scatter-add onto ``global_ids`` does, exactly
    as ghost forces are returned to their owners.

    Note the asymmetry this resolves: an owned atom's *destination* edges all
    live here, but its *source* edges point at neighbours that may be ghosts,
    and those edges live in the neighbour's partition. Neither partition sees
    all of atom i's terms alone; the scatter-add is what completes the sum.
    """
    born = torch.zeros((n_atoms, 3, 3), dtype=dtype, device=home)
    contributions = []
    # Launch every backward before transferring anything, for the same reason
    # the forward pass is staged.
    for part, _device, _out, raw, _node_coord, _eps, graph, cap in launched:
        with _current_device(raw.device):
            # One VJP per partition. `raw` already holds owned atoms only, and
            # every kept edge has an owned destination, so this is exactly
            # d raw_{dst(e)} / d edge_vec_e for the edges this partition owns.
            # With a captured backbone the features hang off the leaf
            # `forward_common_lower_graph` made, not off `graph.edge_vec`. The two
            # are numerically identical, so the outer product below is unaffected;
            # only the differentiation target changes.
            leaf = cap if cap is not None else graph.edge_vec
            (g,) = torch.autograd.grad(raw.sum(), leaf, retain_graph=True)
            # Defensive: `partition_graph` masked ghost-destination edges before
            # the descriptor ran, so those gradients are already identically zero.
            g = g * graph.edge_mask[:, None].to(g.dtype)
            outer = graph.edge_vec.detach()[:, :, None] * g[:, None, :]
            src = graph.edge_index[0].to(torch.int64)
            dst = graph.edge_index[1].to(torch.int64)
            node = 0.5 * (
                segment_sum(outer, dst, part.n_node)
                - segment_sum(outer, src, part.n_node)
            )
            contributions.append((part, node))

    for part, node in contributions:
        born.index_add_(
            0,
            part.views(home, dtype)["ids"],
            node.reshape(part.n_node, 3, 3).detach().to(home, dtype),
        )
    # The delta term, from the globally projected charges. Adding it here
    # rather than per partition is what makes the acoustic sum rule come out
    # as the system's actual total charge.
    eye = torch.eye(3, dtype=dtype, device=home)
    born = born + charge.reshape(-1, 1, 1).to(home, dtype) * eye
    # Screened tensor, Z~ = sqrt(eps_inf) Z -- same factor as the serial
    # `born_charges`, applied once to the assembled whole.
    return born * screening_factor(eps_inf)


def _charge_response_force(
    launched, n_atoms: int, dE_dq, home, dtype, compute_virial: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
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

    The incoming cotangent already passed through the global projection in
    ``_long_range_global``, including its exclusion mask. Do not project twice.

    The charges arrive already computed, still attached to the graph that
    produced them, so this is only the backward pass. Rebuilding the neighbour
    search and re-running the descriptor here -- as an earlier version did --
    was a third descriptor evaluation per partition for no additional
    information.
    """
    # dE_dq already differentiates through the global, possibly masked projection.
    cotangent = dE_dq
    force = torch.zeros((n_atoms, 3), dtype=dtype, device=home)
    virial = torch.zeros((3, 3), dtype=dtype, device=home) if compute_virial else None

    # Launch every backward first, then transfer, for the same reason the
    # forward pass is staged: a `.cpu()` between them serialises the devices.
    gradients = []
    for part, _device, _out, raw, node_coord, eps, graph, cap in launched:
        with _current_device(raw.device):
            weights = cotangent[part.views(home, dtype)["owned"]].to(
                device=raw.device, dtype=raw.dtype
            )
            targets = [node_coord] if eps is None else [node_coord, eps]
            if cap is None:
                grads = torch.autograd.grad((raw * weights).sum(), targets)
            else:
                # Two stages, because the captured features hang off the detached
                # leaf `forward_common_lower_graph` created, which has no path back
                # to `node_coord` (or to the strain). First the VJP onto that leaf,
                # then push it through the partition's own `edge_vec`, which IS a
                # function of both. The leaves are numerically identical, so the
                # composition is the exact chain rule, and routing through
                # `graph.edge_vec` keeps the strain derivative the virial needs.
                (g_e,) = torch.autograd.grad(
                    (raw * weights).sum(), cap, retain_graph=True
                )
                grads = torch.autograd.grad(
                    (graph.edge_vec * g_e.detach()).sum(), targets
                )
            gradients.append((part, grads[0], grads[1] if eps is not None else None))

    for part, gradient, grad_eps in gradients:
        force.index_add_(
            0,
            part.views(home, dtype)["ids"],
            -gradient.reshape(part.n_node, 3).detach().to(home, dtype),
        )
        if compute_virial:
            virial -= grad_eps.reshape(3, 3).detach().to(home, dtype)
    return force, virial


def _long_range_global(
    model,
    coord,
    cell,
    raw_charge,
    device,
    dtype,
    compute_virial: bool = False,
    *,
    atype,
    charge_spin=None,
) -> dict:
    """The global half: project the charges, then evaluate the reciprocal sum.

    Returns the direct force ``-dE/dr`` at fixed charge, and ``dE/dq``. The
    remaining *charge-response* force, ``-sum_j (dE/dq_j) dq_j/dr_i``, is
    short-ranged and is added per partition in Phase C -- taking ``dE/dq`` out
    as a plain value here is what keeps the autograd graph on one device.
    """
    n_atoms = coord.shape[0]
    coord_t = (
        torch.as_tensor(coord, dtype=dtype, device=device)
        .detach()
        .reshape(1, n_atoms, 3)
        .clone()
        .requires_grad_(True)
    )
    cell_t = torch.as_tensor(
        cell if isinstance(cell, torch.Tensor) else np.asarray(cell),
        dtype=dtype,
        device=device,
    ).reshape(1, 3, 3)
    charge_t = (
        torch.as_tensor(raw_charge, dtype=dtype, device=device)
        .detach()
        .reshape(1, n_atoms)
        .clone()
        .requires_grad_(True)
    )

    eps = None
    if compute_virial:
        # Strain coordinates and cell together; the reciprocal sum is
        # evaluated on one device, so this is the same construction the
        # serial model uses.
        eps = torch.zeros((3, 3), dtype=dtype, device=device, requires_grad=True)
        strain = torch.eye(3, dtype=dtype, device=device) + 0.5 * (
            eps + eps.transpose(-1, -2)
        )
        transform = strain.transpose(-1, -2).unsqueeze(0)
        coord_in = torch.bmm(coord_t, transform)
        cell_in = torch.bmm(cell_t, transform)
    else:
        coord_in, cell_in = coord_t, cell_t

    condition = resolve_charge_spin(model, charge_spin, 1, coord_t)
    total_charge = 0.0 if condition is None else condition[:, 0]
    types = torch.as_tensor(atype, dtype=torch.int64, device=device).reshape(1, -1)
    charge = project_to_total_charge(
        charge_t, total_charge, mask=charge_mask(model, types)
    )
    atom_energy = model.long_range_energy(coord_in, charge, cell_in)
    total = atom_energy.sum()
    targets = [coord_t, charge_t] if eps is None else [coord_t, charge_t, eps]
    grads = torch.autograd.grad(total, targets)
    grad_coord, grad_charge = grads[0], grads[1]

    return {
        "charge": charge.reshape(-1).detach(),
        "energy_long_range": total.detach(),
        "atom_energy_long_range": atom_energy.reshape(-1).detach(),
        "force_long_range": -grad_coord.reshape(n_atoms, 3).detach(),
        "dE_dq": grad_charge.reshape(-1).detach(),
        **(
            {"virial_long_range": -grads[2].reshape(3, 3).detach()}
            if eps is not None
            else {}
        ),
    }
