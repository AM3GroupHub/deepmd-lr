# SPDX-License-Identifier: MIT
"""Evaluate a model partition by partition and assemble the global result.

The short-range half is split across partitions; the long-range half is
gathered onto one device rather than distributed. That is a measurement: at
6520 atoms the descriptor costs 883 ms and the reciprocal term 73 ms, so the
descriptor is what is worth splitting, and the gather moves about 104 KB in
and 130 KB out.

Two things are necessarily global and so happen after the gather:

- **The total-charge projection.** ``sum_i q_i = Q_total`` is a constraint
  over every atom. Partitions produce *raw* charges; the projection runs once
  on the assembled vector.
- **The reciprocal sum itself**, a structure factor over every atom. Split
  across ranks in a LAMMPS domain decomposition it gave 2.0% error at 2 ranks
  and 36.5% at 8.

One call runs in phases: :func:`_launch` queues every partition's forward
with no host read, :func:`_gather` sums the owned atoms onto the first device,
:func:`_long_range_global` projects the charges and evaluates the sum there,
:func:`_born_charges_partitioned` (on request) and
:func:`_charge_response_force` take the backward passes per partition, and
:func:`_assembled` adds the terms up.

Devices are handled DistMLIP-style: one process, tensors moved with ``.to()``,
no ``torch.distributed``. Passing the same device several times
(``["cuda:0"] * 4``) is supported and is how the partitioning is tested on a
single-GPU machine.
"""

import contextlib
import copy
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    NamedTuple,
)

import numpy as np
import torch

from deepmd_lr.born import (
    edge_born_response,
    screening_factor,
)
from deepmd_lr.cell import pbc_flags
from deepmd_lr.fitting.charge import (
    charge_mask,
    check_charge_filter,
    filter_charges,
    project_to_total_charge,
    resolve_charge_spin,
)
from deepmd_lr.model.lr_model import (
    _is_compressed,
    _run_backbone,
    _symmetric_strain,
)
from deepmd_lr.nvalchemi.born_csr import (
    edge_born_response_csr,
)
from deepmd_lr.nvalchemi.nlist import (
    GpuNeighborList,
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
    "to_numpy",
]


class _Launched(NamedTuple):
    """What the launch phase keeps of one partition for the phases after it."""

    part: Partition
    out: dict
    raw: torch.Tensor | None
    node_coord: torch.Tensor
    eps: torch.Tensor | None
    graph: Any
    cap: torch.Tensor | None


@dataclass
class _Run:
    """One call of :func:`evaluate_partitioned`: the partitions on their
    devices and replicas, the coordinates on every device, and the sums on the
    first device, ``home``, where everything is assembled."""

    parts: list[Partition]
    devices: list[Any]
    replicas: dict
    atype: Any
    rcut: float
    dtype: torch.dtype
    home: torch.device
    long_range: bool
    compute_virial: bool
    coord: torch.Tensor  # (N, 3) on `home`
    coord_on: dict  # one copy per distinct device
    force: torch.Tensor
    atom_energy: torch.Tensor
    raw_charge: torch.Tensor | None
    virial: torch.Tensor | None

    @property
    def n_atoms(self) -> int:
        return self.coord.shape[0]


def _condition_on(model, charge_spin, device, dtype) -> torch.Tensor | None:
    """The frame condition on ``device``, resolved once rather than every step.

    Resolving it means uploading a Python pair and, for a compressed model,
    comparing it with the state folded into the tables -- each of which blocks
    the host until ``device`` has drained. The answer only depends on the
    requested pair, so it is kept on the replica under that pair. A tensor
    argument is resolved per call instead: it could be mutated in place, and
    then a cached copy would be quietly out of date.
    """
    if isinstance(charge_spin, torch.Tensor):
        return resolve_charge_spin(
            model, charge_spin, 1, torch.empty((), dtype=dtype, device=device)
        )
    requested = model.get_default_chg_spin() if charge_spin is None else charge_spin
    key = (
        torch.device(device),
        dtype,
        None if requested is None else tuple(float(x) for x in requested),
    )
    cache = model.__dict__.setdefault("_nvalchemi_conditions", {})
    if key not in cache:
        cache[key] = resolve_charge_spin(
            model, charge_spin, 1, torch.empty((), dtype=dtype, device=device)
        )
    return cache[key]


def _current_device(device) -> contextlib.AbstractContextManager:
    """Make ``device`` the CUDA *current* device for the block.

    Tensor placement alone is not enough for the compressed DPA4C operator:
    measured on two H200s, a compressed model whose every buffer sits on
    ``cuda:1`` returns forces wrong by 5 eV/A when ``cuda:0`` is current, and
    the right ones (1e-5 of the uncompressed reference) once ``cuda:1`` is
    current; the uncompressed path is indifferent. The operator evidently
    binds something (a stream, a workspace) to the current device rather than
    to its arguments' device. The switch is thread-local, not a
    synchronisation.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.cuda.device(device)


def device_replicas(model, devices) -> dict:
    """One model per distinct device.

    The weights have to be on each device, not just the tensors, or a
    partition on ``cuda:1`` meets parameters still on ``cuda:0`` (single-GPU
    testing hides this). The model already on the target device is used as is,
    so the single-device case allocates nothing extra. Replicas are read-only
    -- this is inference -- so there is no gradient to reduce across them.
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
    correctly by whichever partition owns it. The projection is not applied:
    it is a global constraint, handled once the partitions are assembled.
    """
    if _is_compressed(model):
        # Compressed inference is defined only on the captured route: this
        # second pass would run the uncompressed math on a CSR-less graph
        # while the model still claims to be compressed.
        raise RuntimeError(
            "the fallback charge pass was reached on a compressed model; the "
            "shared-backbone capture should have been taken. Check "
            "DP_CUDA_INFER (it must be exactly 1)."
        )
    features, rot_mat = _run_backbone(model, graph, node_type, charge_spin)
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
    with_csr: bool = False,
):
    """Build one partition's neighbour graph: every edge into an owned atom.

    The search is a :class:`~deepmd_lr.nvalchemi.nlist.GpuNeighborList`,
    cached on the partition and run at every call, with only the owned atoms
    as centres and the result compacted on the device with no host
    synchronisation. Searching only from owned atoms is what makes the
    assembled sum exact -- each edge survives in exactly one partition, the
    one owning its destination -- without an ownership mask. The whole system
    on one device (:func:`~deepmd_lr.nvalchemi.partition.whole_system`) is
    searched in the real cell with the real periodicity, the images carried as
    integer shifts; a partition's node set (owned atoms plus explicit ghost
    images) is searched non-periodically.

    With ``requires_grad`` the node coordinates are tracked, so the
    charge-response backward differentiates through this graph's ``edge_vec``
    instead of rebuilding it. With ``strain`` (a ``(3, 3)`` matrix, typically
    ``I + sym(eps)``) the finished vectors are deformed by it, which is all
    the virial needs: an edge vector is ``x_j + S H - x_i``, every term of
    which deforms alike, so no cell-differentiable search is required.

    ``coord`` may be a device tensor or a numpy array; either way the node set
    is gathered on ``device`` with :meth:`Partition.views`. ``with_csr`` asks
    for the destination and source CSR views the compressed kernel needs; a
    compressed model then takes the kernel's float32 vectors over the list's
    fixed-capacity buffers (its kernels walk the CSR rows and never see the
    tail), unless the charge response needs ``edge_vec`` differentiable. An
    uncompressed model gets vectors in its own precision on a compact edge
    axis, since its array-API descriptor processes every edge it is handed.

    Returns ``(graph, node_type_flat, node_coord)``; ``node_coord`` is the
    unstrained leaf, so a backward reaches both it and the strain.
    """
    coord_t = (
        coord
        if isinstance(coord, torch.Tensor)
        else torch.as_tensor(np.asarray(coord, dtype=np.float64))
    )
    v = part.views(device, dtype)
    c = coord_t.detach().to(device=device, dtype=dtype)
    if part.is_whole_system:
        # The nodes are the atoms, in order: no gather, no shift.
        nodes = c
    else:
        nodes = torch.cat(
            [c[v["owned"]] + v["owned_shift"], c[v["ghost"]] + v["shift"]]
        )
    node_coord = nodes.unsqueeze(0).clone().requires_grad_(requires_grad)
    node_type = part.types_on(atype, device).unsqueeze(0)

    nlist = part.neighbor_list_on(device)
    if nlist is None:
        # Float64 vectors when they will be differentiated (the charge
        # response) or the model runs uncompressed; the compressed
        # short-range model takes the float32 ones.
        keep64 = requires_grad or not with_csr
        if part.is_whole_system:
            # `v["cell"]` is the tensor `Partition.set_cell` keeps current. A
            # system with no periodic axis has no cell to search: the list
            # boxes the atoms itself (their bounding box plus a margin, fixed
            # at the first search; an atom that later leaves it is clamped
            # into an edge cell, which keeps the search complete).
            nlist = GpuNeighborList(
                part.n_node,
                rcut,
                device,
                cell=v["cell"] if any(part.periodic) else None,
                pbc=part.periodic,
                float64_vectors=keep64,
            )
        else:
            nlist = GpuNeighborList(
                part.n_node,
                rcut,
                device,
                cell=None,
                targets=torch.arange(
                    part.n_owned, dtype=torch.int32, device=torch.device(device)
                ),
                float64_vectors=keep64,
            )
        part.store_neighbor_list(device, nlist)
    nlist.search(nodes)
    if with_csr and not requires_grad:
        graph = nlist.graph()
    elif with_csr:
        graph = nlist.graph(node_coord, strain, dtype=torch.float32)
    else:
        graph = nlist.graph(node_coord, strain, dtype=dtype, compact=True)
        # Without the CSR views the uncompressed path assembles forces with
        # the array-API scatter.
        graph.destination_order = graph.source_order = None
        graph.destination_row_ptr = graph.source_row_ptr = None
        graph.destination_sorted = False
    return graph, node_type.reshape(-1), node_coord


def check_long_range_boundary(model: Any, pbc, term: Any = None) -> None:
    """Refuse a long-range term whose boundary is not the system's.

    ``term`` is the one about to be evaluated, by default the model's own.
    Unchecked, the term simply runs: an ``isolated`` term on a periodic cell
    sums no images, and a slab term corrects a cell that is periodic along z.
    On the 192-atom water box that was 1.1 eV and 0.03 eV off.
    """
    if not getattr(model, "has_long_range", False):
        return
    if term is None:
        term = model.long_range_energy
    expected = (
        tuple(bool(p) for p in term.pbc) if term.needs_cell else (False, False, False)
    )
    have = pbc_flags(pbc)
    if have != expected:
        method = getattr(term, "method", getattr(term, "METHOD", "?"))
        raise ValueError(
            f"the long-range term ('{method}', pbc {list(expected)}) does not "
            f"match the system's pbc {list(have)}. DistributedEvaluator builds "
            "the term the system needs; evaluate_partitioned takes one as "
            "long_range_energy=model.long_range_energy_for(cell, pbc)."
        )


def evaluate_partitioned(
    model: Any,
    coord: np.ndarray,
    atype: np.ndarray,
    cell: np.ndarray,
    n_partitions: int = 1,
    devices: list[Any] | None = None,
    pbc=(True, True, True),
    dtype=torch.float64,
    partitions: list[Partition] | None = None,
    replicas: dict | None = None,
    compute_virial: bool = False,
    charge_spin: torch.Tensor | tuple[float, float] | None = None,
    compute_born: bool = False,
    eps_inf: float = 1.0,
    long_range_energy: Any = None,
    charge_filter: tuple[float, float] | None = None,
    atype_device: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run ``model`` over spatial partitions and assemble the global result.

    Everything is assembled on the gather device (the first of ``devices``)
    and **returned there**: per-partition forces, energies and charges are
    moved device-to-device and scatter-added with ``index_add_`` -- the
    ghost-force reverse communication of a domain-decomposed code, done as a
    device primitive. A caller that wants numpy arrays converts explicitly
    with :func:`to_numpy`, which makes the download a visible line.

    Parameters
    ----------
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
    long_range_energy
        The long-range term to evaluate, by default ``model.long_range_energy``.
        :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator` passes
        the one it built for its cell and pbc, which leaves the model alone.
    charge_filter
        ``(c_t, delta)``: switch charges below about ``c_t`` off, smoothly,
        after the projection (:func:`~deepmd_lr.fitting.charge.filter_charges`).
        A simulation-only setting; refused with Born charges.
    atype_device
        ``atype`` already on the gather device, for the long-range term. A
        caller that evaluates every step keeps one copy rather than paying a
        pageable upload per call; ``cell`` may likewise be a device tensor.

    Returns
    -------
    dict
        Device tensors on the gather device, scalars as 0-d tensors:
        ``energy``, ``force`` ``(N, 3)``, ``atom_energy`` ``(N,)``, and when the
        model carries a long-range term also ``charge`` ``(N,)``,
        ``energy_short_range``, ``energy_long_range``, and ``dE_dq`` ``(N,)``
        -- the last is what :func:`_charge_response_force` backpropagates
        through each partition's charge head.
    """
    charge_filter = _validated(
        model, pbc, long_range_energy, charge_filter, compute_born
    )
    run = _prepared(
        model,
        coord,
        atype,
        cell,
        n_partitions,
        devices,
        pbc,
        dtype,
        partitions,
        replicas,
        compute_virial,
    )
    launched = _launch(run, charge_spin)
    _gather(run, launched)

    energy = run.atom_energy.sum()
    result = {"energy": energy, "force": run.force, "atom_energy": run.atom_energy}
    if compute_virial:
        result["virial"] = run.virial.clone()
        result["virial_short_range"] = run.virial.clone()
    if not run.long_range:
        return result

    result["energy_short_range"] = energy
    result.update(
        _long_range_global(
            run.replicas[run.home],
            run.coord,
            cell,
            run.raw_charge,
            run.home,
            dtype,
            compute_virial=compute_virial,
            atype=atype if atype_device is None else atype_device,
            charge_spin=charge_spin,
            term=long_range_energy,
            charge_filter=charge_filter,
        )
    )
    born_grads = None
    if compute_born:
        # Before the charge-response force, which backpropagates through the
        # same raw charges without retaining the graph.
        result["born_charge"], born_grads = _born_charges_partitioned(
            run, launched, result["charge"], eps_inf
        )
    response = _charge_response_force(run, launched, result["dE_dq"], born_grads)
    return _assembled(result, response, compute_virial, eps_inf)


def _validated(model, pbc, long_range_energy, charge_filter, compute_born):
    """The checked charge filter; a term whose boundary is not the system's,
    and a filter with Born charges, are refused."""
    check_long_range_boundary(model, pbc, long_range_energy)
    charge_filter = check_charge_filter(charge_filter)
    if charge_filter is not None and compute_born:
        # The Born tensor differentiates the raw charges; the filter would
        # need its own chain through them.
        raise NotImplementedError(
            "charge_filter cannot be combined with Born charges (compute_born or "
            "an efield) yet: the Born tensor is built from the unfiltered charges"
        )
    return charge_filter


def _prepared(
    model,
    coord,
    atype,
    cell,
    n_partitions: int,
    devices,
    pbc,
    dtype,
    partitions,
    replicas,
    compute_virial: bool,
) -> _Run:
    """The partitions (those given, or a decomposition of ``coord``) on their
    devices and replicas, the coordinates on every device, the zeroed sums."""
    rcut = float(model.get_rcut())
    coord_np = (
        None if isinstance(coord, torch.Tensor) else np.asarray(coord, dtype=np.float64)
    )
    parts = partitions
    if parts is None:
        host = coord.detach().cpu().numpy() if coord_np is None else coord_np
        parts = partition_system(host, cell, rcut, n_partitions, pbc)
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

    home = torch.device(devices[0])
    coord_t = (
        coord.detach().to(device=home, dtype=dtype)
        if isinstance(coord, torch.Tensor)
        else torch.as_tensor(coord_np, dtype=dtype, device=home)
    )
    n_atoms = coord_t.shape[0]
    force = torch.zeros((n_atoms, 3), dtype=dtype, device=home)
    atom_energy = torch.zeros(n_atoms, dtype=dtype, device=home)
    raw_charge = torch.zeros(n_atoms, dtype=dtype, device=home) if long_range else None
    virial = torch.zeros((3, 3), dtype=dtype, device=home) if compute_virial else None
    # One copy of the coordinates per device, not per partition: every
    # partition on a device gathers its rows out of the same array. All of
    # them are queued before any forward because a cross-device copy runs on
    # the SOURCE device's stream, behind whatever is already queued there.
    coord_on = {}
    for device in devices:
        dev = torch.device(device)
        if dev not in coord_on:
            with _current_device(dev):
                coord_on[dev] = coord_t.to(device=dev, dtype=dtype)
    return _Run(
        parts=parts,
        devices=devices,
        replicas=replicas,
        atype=atype,
        rcut=rcut,
        dtype=dtype,
        home=home,
        long_range=long_range,
        compute_virial=compute_virial,
        coord=coord_t,
        coord_on=coord_on,
        force=force,
        atom_energy=atom_energy,
        raw_charge=raw_charge,
        virial=virial,
    )


def _launch(run: _Run, charge_spin) -> list[_Launched]:
    """Every partition's forward, queued without touching the host.

    A host read here would synchronise, and the partition on the next device
    could not start until this one finished. Kernel launches are asynchronous
    and each device has its own stream, so a device keeps computing while the
    host moves on -- except inside the neighbour search, whose blocking host
    reads wait for their own device. The loop is deliberately single-threaded;
    ``docs/partitioned-md.md`` records why threads were tried and removed.
    """
    return [
        _launch_one(run, part, device, charge_spin)
        for part, device in zip(run.parts, run.devices, strict=True)
    ]


def _launch_one(run: _Run, part: Partition, device, charge_spin) -> _Launched:
    """One partition's forward on its own device. A function rather than a
    loop body so each partition's captured features are released when it
    returns, before the next one runs."""
    dtype = run.dtype
    with _current_device(device):
        local = run.replicas[torch.device(device)]
        condition = _condition_on(local, charge_spin, device, dtype)
        # One strain leaf per partition: a leaf lives on one device, and the
        # virial is a sum of contributions, so summed gradients are the same.
        eps = strain = None
        if run.compute_virial:
            eps, strain = _symmetric_strain(dtype, device)
        graph, node_type, node_coord = partition_graph(
            part,
            run.coord_on[torch.device(device)],
            run.atype,
            run.rcut,
            device,
            dtype,
            requires_grad=run.long_range,
            strain=strain,
            with_csr=_is_compressed(local),
        )
        n_node, n_local = part.counts_on(device)
        # The model's own capture context records the backbone output the
        # charge head reuses, and it is what permits the graph lower -- a
        # caller outside one would be reporting the short-range energy alone.
        capture = (
            local._capturing_backbone()
            if run.long_range
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
        # Charges from the same descriptor pass the energy head just ran, kept
        # attached for the charge-response backward; a second pass only when
        # the capture cannot carry the gradient.
        raw = cap_edge_vec = None
        if run.long_range:
            if _usable_capture(captured):
                raw = charges_from_capture(local, captured, node_type, part.n_owned)
                cap_edge_vec = captured["edge_vec"]
            else:
                raw = partition_charges(
                    local, graph, node_type, part.n_owned, condition
                )
        return _Launched(part, out, raw, node_coord, eps, graph, cap_edge_vec)


def _gather(run: _Run, launched: list[_Launched]) -> None:
    """Short-range energies, forces, virial and raw charges onto ``home``.

    The transfers are device-to-device and the scatter-adds device
    primitives; ``.to()`` between devices orders on the streams without
    synchronising the host. Each atom's energy and charge come from its one
    owner; forces from every node, ghosts included, since a ghost carries the
    reaction to an owned atom's edge and that belongs to the ghost's owner.
    Each edge survives in exactly one partition, so the virials sum exactly.
    """
    home, dtype = run.home, run.dtype
    for part, out, raw, *_ in launched:
        v = part.views(home, dtype)
        if run.compute_virial:
            run.virial += (
                out["energy_derv_c_redu"].reshape(3, 3).detach().to(home, dtype)
            )
        per_atom = out["energy"].reshape(-1).detach().to(home, dtype)
        # `energy_derv_r` is already the force, not the energy gradient.
        node_force = (
            out["energy_derv_r"].reshape(part.n_node, 3).detach().to(home, dtype)
        )
        if part.is_whole_system:
            # Nodes are the atoms in order: the sums have one term each.
            run.atom_energy.copy_(per_atom)
            run.force.copy_(node_force)
            if run.long_range:
                run.raw_charge.copy_(raw.detach().to(home, dtype))
            continue
        run.atom_energy.index_add_(0, v["owned"], per_atom[: part.n_owned])
        run.force.index_add_(0, v["ids"], node_force)
        if run.long_range:
            run.raw_charge.index_copy_(0, v["owned"], raw.detach().to(home, dtype))


def _assembled(
    result: dict, response: tuple, compute_virial: bool, eps_inf: float
) -> dict:
    """``result`` with the short-range, long-range and charge-response terms
    added up, and the reported charges screened."""
    response_force, response_virial = response
    result["energy"] = result["energy_short_range"] + result["energy_long_range"]
    result["atom_energy_short_range"] = result["atom_energy"]
    result["atom_energy"] = result["atom_energy"] + result["atom_energy_long_range"]
    result["force"] = result["force"] + result["force_long_range"]
    result["force_charge_response"] = response_force
    result["force"] = result["force"] + response_force
    if compute_virial:
        result["virial"] = (
            result["virial"] + result["virial_long_range"] + response_virial
        )
        result["virial_charge_response"] = response_virial
    # The energy used the model's own charges; only the report is screened,
    # after the Born tensor took the unscaled ones for its delta term.
    factor = screening_factor(eps_inf)
    if factor != 1.0:
        result["charge"] = result["charge"] * factor
    return result


def to_numpy(result: dict) -> dict:
    """Copy a :func:`evaluate_partitioned` result to the host.

    Arrays become numpy, 0-d tensors become Python floats. This is the only
    download in the pipeline and it is the caller's explicit choice, so a step
    that never reads a result on the host never pays for one.
    """
    return {key: _to_host(value) for key, value in result.items()}


def _to_host(value):
    """A tensor as numpy -- a 0-d one as a Python float; anything else as is."""
    if not isinstance(value, torch.Tensor):
        return value
    value = value.detach()
    return float(value) if value.ndim == 0 else value.cpu().numpy()


def _csr_views(graph) -> bool:
    return (
        graph.destination_row_ptr is not None
        and graph.source_order is not None
        and graph.source_row_ptr is not None
    )


def _born_charges_partitioned(
    run: _Run, launched: list[_Launched], charge, eps_inf: float
) -> tuple[torch.Tensor, list]:
    r"""Born effective charge tensors, assembled across partitions.

    The global projection touches only the :math:`q_i\delta_{\alpha\beta}`
    term, which arrives already projected (``charge``). The derivative half
    uses the **raw** charges, which are strictly local -- DPA4C is one-hop, so
    an atom's raw charge sees only edges where it is the centre -- and
    therefore partition exactly. Each partition contributes both halves of the
    pair sum for every edge it owns::

        + 1/2 (r_e (x) g_e)  ->  dst(e)     always owned
        - 1/2 (r_e (x) g_e)  ->  src(e)     owned OR a ghost

    A directed edge survives only in the partition owning its destination, so
    summing over partitions reproduces the serial tensor with no double
    counting; a src half on a ghost is scatter-added to its owner, as ghost
    forces are. Neither partition sees all of an atom's terms alone: its
    source edges may live in a neighbour's partition.

    With the GPU list's CSR views and float32 vectors the edge half is summed
    along the rows by :func:`~deepmd_lr.nvalchemi.born_csr.edge_born_response_csr`;
    otherwise by the array-API :func:`~deepmd_lr.born.edge_born_response`.

    Returns the screened tensor and, per partition, the unweighted cotangent
    ``d raw / d edge_vec`` (``None`` where no capture was used), which
    :func:`_charge_response_force` reuses instead of a second backward.
    """
    home, dtype = run.home, run.dtype
    born = torch.zeros((run.n_atoms, 3, 3), dtype=dtype, device=home)
    contributions = []
    unweighted = []
    # Every backward is launched before anything is transferred, as in the
    # forward pass.
    for part, _out, raw, _node_coord, _eps, graph, cap in launched:
        with _current_device(raw.device):
            # One VJP per partition: `raw` holds owned atoms only and every
            # kept edge has an owned destination, so this is exactly
            # d raw_{dst(e)} / d edge_vec_e. With a captured backbone the
            # features hang off the leaf `forward_common_lower_graph` made,
            # numerically identical to `graph.edge_vec`.
            leaf = cap if cap is not None else graph.edge_vec
            (g,) = torch.autograd.grad(raw.sum(), leaf, retain_graph=True)
            vec = graph.edge_vec.detach()
            if (
                _csr_views(graph)
                and g.dtype == torch.float32
                and vec.dtype == torch.float32
            ):
                node = edge_born_response_csr(
                    g,
                    vec,
                    graph.destination_row_ptr,
                    graph.source_order,
                    graph.source_row_ptr,
                    part.n_node,
                )
            else:
                # The mask is defensive: only owned-destination edges are in
                # the graph, and the tail's gradients are zero.
                node = edge_born_response(
                    g, vec, graph.edge_index, graph.edge_mask, part.n_node
                )
            contributions.append((part, node))
            unweighted.append(g if cap is not None else None)

    for part, node in contributions:
        born.index_add_(
            0,
            part.views(home, dtype)["ids"],
            node.reshape(part.n_node, 3, 3).detach().to(home, dtype),
        )
    # The delta term from the globally projected charges, so the acoustic sum
    # rule comes out as the system's actual total charge.
    eye = torch.eye(3, dtype=dtype, device=home)
    born = born + charge.reshape(-1, 1, 1).to(home, dtype) * eye
    return born * screening_factor(eps_inf), unweighted


def _charge_response_force(
    run: _Run,
    launched: list[_Launched],
    dE_dq,
    born_grads: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""The force (and virial) from charges shifting as atoms move.

    Because the charges are predicted rather than fixed, the long-range force
    has a second term beyond the electrostatic one:

    .. math::
        F_i \supset -\sum_j \frac{\partial E}{\partial q_j}
                              \frac{\partial q_j}{\partial r_i}

    It is not a correction: measured here it is ~3e-2 eV/A against a direct
    long-range force of ~2.5e-3.

    **This is the term that keeps the scheme parallel.** ``dq_j/dr_i`` is
    nonzero only for atoms within ``rcut`` of j, so although the energy is
    long-ranged its charge derivative is short-ranged. Each partition
    backpropagates its owned atoms' cotangents through its own charge head,
    on its own device, through the graph the launch phase kept: no autograd
    graph spans devices and nothing is communicated beyond the halo.
    ``dE_dq`` already passed through the global projection (and its exclusion
    mask) in :func:`_long_range_global`; it is not projected again.

    ``born_grads``, when the Born tensors were just computed, holds each
    partition's unweighted cotangent ``d raw / d edge_vec``. The descriptor
    is one-hop, so ``d (sum_j w_j raw_j) / d edge_vec_e`` is that gradient
    times ``w_{dst(e)}``, and the backward through the descriptor is not
    repeated.
    """
    home, dtype = run.home, run.dtype
    force = torch.zeros((run.n_atoms, 3), dtype=dtype, device=home)
    virial = (
        torch.zeros((3, 3), dtype=dtype, device=home) if run.compute_virial else None
    )
    # Every backward first, then the transfers, as in the forward pass.
    gradients = []
    for k, (part, _out, raw, node_coord, eps, graph, cap) in enumerate(launched):
        with _current_device(raw.device):
            weights = dE_dq[part.views(home, dtype)["owned"]].to(
                device=raw.device, dtype=raw.dtype
            )
            targets = [node_coord] if eps is None else [node_coord, eps]
            if cap is None:
                grads = torch.autograd.grad((raw * weights).sum(), targets)
            else:
                # Two stages: the captured features hang off the detached leaf
                # `forward_common_lower_graph` created, which has no path back
                # to `node_coord` or the strain. First the VJP onto that leaf,
                # then through the partition's own `edge_vec`, which is a
                # function of both; the leaves are numerically identical, so
                # this is the exact chain rule.
                g1 = born_grads[k] if born_grads is not None else None
                if g1 is not None:
                    w_node = torch.zeros(part.n_node, dtype=g1.dtype, device=g1.device)
                    w_node[: part.n_owned] = weights.to(g1.dtype)
                    g_e = g1 * w_node[graph.edge_index[1]][:, None]
                else:
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
        if run.compute_virial:
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
    term=None,
    charge_filter=None,
) -> dict:
    """The global half: project the charges, then evaluate the reciprocal sum.

    ``term`` is the long-range term to evaluate, by default the model's own.

    Returns the direct force ``-dE/dr`` at fixed charge, and ``dE/dq`` with
    respect to the raw charges. The remaining *charge-response* force,
    ``-sum_j (dE/dq_j) dq_j/dr_i``, is short-ranged and is added per partition
    by :func:`_charge_response_force`; taking ``dE/dq`` out as a plain value
    here is what keeps the autograd graph on one device.
    """
    n_atoms = coord.shape[0]
    coord_t = coord.detach().reshape(1, n_atoms, 3).clone().requires_grad_(True)
    cell_t = torch.as_tensor(
        cell if isinstance(cell, torch.Tensor) else np.asarray(cell),
        dtype=dtype,
        device=device,
    ).reshape(1, 3, 3)
    charge_t = raw_charge.detach().reshape(1, n_atoms).clone().requires_grad_(True)

    eps = None
    if compute_virial:
        # Coordinates and cell strained together, as the serial model does.
        eps, strain = _symmetric_strain(dtype, device)
        transform = strain.transpose(-1, -2).unsqueeze(0)
        coord_in = torch.bmm(coord_t, transform)
        cell_in = torch.bmm(cell_t, transform)
    else:
        coord_in, cell_in = coord_t, cell_t

    # Resolved once per model and device; the launch phase already validated it.
    condition = _condition_on(model, charge_spin, device, dtype)
    total_charge = 0.0 if condition is None else condition[:, 0]
    types = torch.as_tensor(atype, dtype=torch.int64, device=device).reshape(1, -1)
    mask = charge_mask(model, types)
    charge = project_to_total_charge(charge_t, total_charge, mask=mask)
    if charge_filter is not None:
        # Inside this graph, so dE/dq below already carries the filter to the
        # charge-response force.
        charge = filter_charges(charge, total_charge, charge_filter, mask=mask)
    if term is None:
        term = model.long_range_energy
    atom_energy = term(coord_in, charge, cell_in)
    total = atom_energy.sum()
    targets = [coord_t, charge_t] if eps is None else [coord_t, charge_t, eps]
    grads = torch.autograd.grad(total, targets)
    grad_coord, grad_charge = grads[0], grads[1]

    out = {
        "charge": charge.reshape(-1).detach(),
        "energy_long_range": total.detach(),
        "atom_energy_long_range": atom_energy.reshape(-1).detach(),
        "force_long_range": -grad_coord.reshape(n_atoms, 3).detach(),
        "dE_dq": grad_charge.reshape(-1).detach(),
    }
    if eps is not None:
        out["virial_long_range"] = -grads[2].reshape(3, 3).detach()
    return out
