# SPDX-License-Identifier: MIT
"""Evaluate the model across several devices: energy, forces, virial.

This is the driver an MD engine calls. It owns the partitioning, the ghost
halo, the per-device model replicas and the decision of when the halo needs
rebuilding -- and nothing else. It has no timestep, no ensemble, no
thermostat and no velocities: those belong to whoever is integrating.

Energy and forces always; the virial and stress on request, since they cost
an extra strain leaf per partition and an extra target in the charge-response
backward. The ``(energy, forces)`` return signature does not change either way,
so an integrator is unaffected by the choice.

nvalchemiops' integrators are half-step kernels with no driver of their own,
so the caller composes the loop and asks this for the force term:

    from nvalchemiops.dynamics.integrators import (
        velocity_verlet_position_update,
        velocity_verlet_velocity_finalize,
    )

    driver = DistributedEvaluator(model, atype, cell, devices=["cuda:0", "cuda:1"])
    energy, forces = driver(positions)
    for _ in range(steps):
        velocity_verlet_position_update(r, v, f, m, dt)   # Warp views
        energy, forces = driver(positions)
        velocity_verlet_velocity_finalize(v, f, m, dt)

Forces come back as a float64 torch tensor on the model's device, which
``wp.from_torch`` turns into a zero-copy Warp view -- so the trajectory never
has to leave the GPU for the integrator's sake.

There is deliberately no ASE calculator here. ASE's interface returns energy
and forces for one configuration at a time and its integrators cannot use
on-device state, which is the whole point of this path.
"""

import traceback
import warnings
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd_lr.born import (
    field_force,
    screening_factor,
)
from deepmd_lr.cell import inverse_3x3
from deepmd_lr.compress import (
    compress_model,
)
from deepmd_lr.fitting.charge import check_charge_filter
from deepmd_lr.nvalchemi.driver import (
    _to_host,
    check_long_range_boundary,
    device_replicas,
    evaluate_partitioned,
)
from deepmd_lr.nvalchemi.nlist import (
    STATUS_EDGE_OVERFLOW,
    STATUS_LEN,
    STATUS_MATRIX_OVERFLOW,
)
from deepmd_lr.nvalchemi.partition import (
    _perpendicular_widths,
    default_skin,
    partition_system,
    skin_for_node_ratio,
    whole_system,
)

__all__ = [
    "DistributedEvaluator",
    "atomic_masses",
]


#: Below this the halo is no longer what is using the memory, so backing off
#: further would trade real speed for nothing.
MIN_AUTO_SKIN = 0.1

#: Slot of the status word that the evaluator owns: a slab atom outside the
#: cell along an open axis (the list's own slots are the ``STATUS_*`` ones).
STATUS_OPEN_FACE = 2

#: How many times one call may widen the neighbour list and redo the step
#: before giving up: each widening is 1.25x, so this is a factor of about 3.
MAX_LIST_GROWTHS = 5


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc`` is a device that ran out of memory.

    Torch raises its own class; warp and the allocator behind the neighbour
    search raise a plain ``RuntimeError`` whose message is the only signal.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError | MemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    return "out of memory" in text or "failed to allocate" in text


def atomic_masses(model: Any, atype: np.ndarray) -> np.ndarray:
    """Standard atomic masses in amu, ordered like ``atype``.

    A convenience for setting up an integrator: the mapping from type index
    to element is the model's own ``type_map``, so the caller does not have
    to reconstruct it.
    """
    from ase.data import atomic_masses as table
    from ase.data import atomic_numbers

    type_map = model.get_type_map()
    masses = np.array([table[atomic_numbers[name]] for name in type_map])
    return masses[np.asarray(atype, dtype=np.int64).reshape(-1)]


class _LazyNumpy(dict):
    """A result dict whose tensors become numpy arrays on first access.

    The lazy twin of :func:`~deepmd_lr.nvalchemi.driver.to_numpy`: the driver
    returns device tensors, and this defers each download to the moment the
    entry is actually read.

    The evaluator assembles everything on the GPU; most callers read one or
    two entries per step, if any, so converting all of them eagerly would
    reintroduce the host traffic the device gather removed. 0-d tensors become
    Python floats, exactly as :func:`~deepmd_lr.nvalchemi.driver.to_numpy`
    converts them.
    """

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if isinstance(value, torch.Tensor):
            value = _to_host(value)
            dict.__setitem__(self, key, value)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return [(k, self[k]) for k in self]

    def values(self):
        return [self[k] for k in self]


_VIRIAL_UNDER_FIELD = (
    "compute_virial and efield cannot be combined: the field's contribution "
    "to the stress is a separate derivation, and reporting the field-free "
    "virial under a field would be quietly wrong."
)


_VIRIAL_ISOLATED = (
    "compute_virial needs a cell to strain, and a cluster (pbc all False) has "
    "none: its box is only the neighbour list's. Run it in NVE or NVT."
)


def _checked_pbc(pbc: Any) -> tuple[bool, bool, bool]:
    """A bulk cell, a slab or a cluster: the boundaries the engine handles."""
    flags = tuple(bool(p) for p in np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)))
    if sum(flags) == 1:
        raise ValueError(
            f"pbc={list(flags)}: one periodic axis (a wire) is not supported. The "
            "engine handles a bulk cell, a slab (one open axis) and a cluster "
            "(no periodic axis)."
        )
    return flags


def _isolated_dtype(model: Any, requested: Any) -> torch.dtype:
    """The dtype a cluster's all-pairs sum runs in: ``requested`` (a dtype or
    its name), else the charge head's precision. In float32 the sum costs
    8 + 37 ms per step at 27,000 atoms on a GB10, against 0.8 s in float64."""
    if requested is None:
        requested = getattr(model.charge_fitting, "precision", "float64")
    name = str(requested).removeprefix("torch.")
    if name not in ("float32", "float64"):
        raise ValueError(
            f"isolated_dtype must be float32 or float64, got {requested!r}"
        )
    return getattr(torch, name)


def _indexed_device(device) -> str:
    """``"cuda"`` -> ``"cuda:<current>"``; anything else unchanged."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return str(d)


class DistributedEvaluator:
    """Evaluate a ``dp_lr`` model over spatial partitions, one per device.

    Parameters
    ----------
    model
        A ``dp_lr`` model, already on its home device. It is never changed,
        so several evaluators, and ASE calculators, may share it.
    atype, cell
        Atom types ``(natoms,)`` and the cell ``(3, 3)``. The types are fixed
        for the life of the driver; a barostat changes the cell through
        :meth:`set_cell`.
    pbc
        Which cell axes are periodic; all three by default. One open axis is
        a slab: no periodic images along it, and the long-range term takes its
        slab form (the Yeh-Berkowitz correction). The term's PME mesh is set
        from ``cell`` as well, once, for the life of the driver. The term is
        this evaluator's own, :attr:`long_range_energy`
        (:meth:`~deepmd_lr.model.lr_model.LongRangeEnergyModel.long_range_energy_for`).
        A slab's atoms must stay inside the cell along the open axis; see
        :meth:`_check_open_axes`. No periodic axis is a cluster: one device, no
        virial, the isolated sum on the chunked kernel
        (:mod:`deepmd_lr.lr.isolated`), and ``cell`` is never read (a zero
        cell is fine; the neighbour list boxes the atoms itself). One periodic
        axis (a wire) is refused.
    long_range
        Reciprocal-space settings to adapt instead of the model's, as
        ``set_long_range`` takes them (alpha is the model's and cannot
        change). For this evaluator only. A cluster also takes
        ``isolated_dtype`` (``"float32"`` or ``"float64"``), the dtype of its
        all-pairs sum; the charge head's precision by default.
    devices
        One device per partition. Repeats are accepted and exercise the
        partitioning arithmetic, but only distinct devices test placement.
    skin
        Halo padding beyond ``rcut``. The partitioning is reused until an atom
        has moved far enough to invalidate it, so this trades ghost work
        against how often the (host-side, serial) rebuild runs. ``"auto"``,
        the default, sizes it from the cell, the cutoff and the partition
        count -- see :func:`~deepmd_lr.nvalchemi.partition.default_skin`, which
        documents why a constant is the wrong answer at both ends of the size
        range. The value actually used is :attr:`skin`. An ``"auto"`` skin is
        also fitted to the device, since the geometry rule cannot see how many
        bytes a node costs: the first step measures that cost (see
        ``memory_headroom``), and a device that runs out anyway halves and caps
        the skin (see :meth:`_evaluate`). An explicit value is never overridden.
    repartition_every
        Force a rebuild every N calls regardless. ``None`` uses the skin
        criterion alone, which is exact rather than merely usually right.
    compute_born
        Also assemble Born effective charge tensors, available afterwards as
        ``results["born_charge"]`` with shape ``(natoms, 3, 3)``. The global
        total-charge projection is no obstacle: it enters only the
        ``q_i delta`` term, which arrives already projected, while the
        derivative half uses the raw charges and is strictly local.
    compute_virial
        Also assemble the virial, exposed as :attr:`virial` (eV) and
        :attr:`stress` (eV/angstrom^3, ASE sign convention) after each call.
        Off by default because it costs an extra strain leaf per partition
        and an extra target in the charge-response backward.
    efield
        Uniform external field in V/angstrom, or ``None``. Adds the response
        force ``F_ib = E_a Z_iab`` to every returned force and implies
        ``compute_born``. The reported energy is left untouched: this force
        is NOT the gradient of a ``-M.E`` energy -- under periodic boundaries
        ``M`` is not even defined -- so a constant-field NVE run will not
        conserve ``E - M.E``, by construction rather than by error. Use a
        thermostat and read structure and response, not a conserved quantity.
        Rejected together with ``compute_virial``, because the field's
        contribution to the stress is a separate derivation and returning the
        field-free value would be quietly wrong.
    charge_spin
        Fixed (total charge, spin multiplicity) for this system. If omitted,
        use the model's configured default condition.
    memory_headroom
        Fraction of usable device memory the halo may be sized for, ``0.8`` by
        default. The first evaluation is a probe: it runs at the floor skin,
        measures what a node costs in bytes there, and the skin is then set to
        the largest value that fits the budget, never above what the geometry
        rule asks for. ``None`` skips the probe and the measurement, which is
        what a run that wants the geometry rule and nothing else should pass.
        The out-of-memory backoff stays either way, since a prediction can
        still be wrong when another process is sharing the device.
    compress
        ``True`` to compress the DPA4C backbone at DeePMD-kit's default table
        stride, or a float stride in angstrom. Done before the device replicas
        are made, with ``charge_spin`` folded into the tables. Needs a
        deepmd-kit built from source with the CUDA operators and is refused
        otherwise; see ``LongRangeEnergyModel.enable_compression``.

    eps_inf
        High-frequency dielectric constant of the medium, default ``1.0``
        (vacuum). The reported charges and Born tensors are multiplied by
        ``sqrt(eps_inf)``: the model's Coulomb energy ``q_i q_j / (4 pi eps_0 r)``
        is the screened ``q~_i q~_j / (4 pi eps_0 eps_inf r)`` with
        ``q~ = sqrt(eps_inf) q``. The field force, built from ``Z``, scales with
        it; energies and field-free forces do not change.

    charge_filter
        ``(c_t, delta)`` in e; ``False`` for none; None (the default) for the
        model's own ``long_range.charge_filter``, if it was trained with one.
        Charges with ``|q| <= c_t - delta`` are set exactly to zero, those
        above ``c_t + delta`` kept, with a C^3 polynomial between, and the
        total is restored on the atoms that keep charge
        (:func:`~deepmd_lr.fitting.charge.filter_charges`). It removes the
        small charge a model can leave on every bulk atom, which adds up over
        a thick slab. Applying it to a model trained without it changes the
        interface electrostatics that model learned. Refused with Born charges
        and fields.

    Notes
    -----
    The reciprocal sum is gathered onto one device rather than decomposed,
    because it does not survive spatial decomposition. The charge-response
    force stays partitioned: ``dq_j/dr_i`` is nonzero only within ``rcut``,
    so although the energy is long-ranged its charge derivative is short.
    """

    def __init__(
        self,
        model: Any,
        atype: np.ndarray,
        cell: np.ndarray,
        *,
        devices: list[Any],
        pbc: Any = (True, True, True),
        long_range: dict | None = None,
        skin: float | str = "auto",
        repartition_every: int | None = None,
        compute_virial: bool = False,
        compute_born: bool = False,
        efield: Any = None,
        charge_spin: torch.Tensor | tuple[float, float] | None = None,
        compress: bool | float = False,
        eps_inf: float = 1.0,
        memory_headroom: float | None = 0.8,
        charge_filter: Any = None,
    ) -> None:
        if not devices:
            raise ValueError("DistributedEvaluator needs at least one device")
        if isinstance(skin, str):
            if skin != "auto":
                raise ValueError(f"skin must be a number or 'auto', got {skin!r}")
        elif skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")
        if memory_headroom is not None and not 0.0 < memory_headroom <= 1.0:
            raise ValueError(
                f"memory_headroom must be in (0, 1], got {memory_headroom}"
            )

        self._cell_host = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        self._cell_host_stale = False
        #: Bumped by every cell change, host or device path, so that a
        #: :class:`~deepmd_lr.nvalchemi.thermo.Thermo` can tell whether the
        #: cell is still the one it set without comparing host copies.
        self.cell_version = 0
        self._skin_stale = False
        self.pbc = _checked_pbc(pbc)
        #: No periodic axis: the isolated sum, one device, no virial, and a
        #: cell nothing reads (the neighbour list boxes the atoms itself).
        self.isolated = not any(self.pbc)
        # The term holds no weights, so this evaluator builds its own: its pbc
        # picks the 3D, the slab or the isolated sum, its cell the PME mesh.
        # The mesh is fixed here for the life of the driver, so the small cell
        # changes of a barostat keep the energy smooth. The model is left
        # alone: another evaluator built from it, for another cell or
        # boundary, would otherwise change the term under this one.
        #: The long-range term this evaluator evaluates, or None for a model
        #: without one.
        self.long_range_energy = None
        long_range = None if long_range is None else dict(long_range)
        isolated_dtype = (
            None if long_range is None else long_range.pop("isolated_dtype", None)
        )
        if getattr(model, "has_long_range", False):
            self.long_range_energy = model.long_range_energy_for(
                self.cell, self.pbc, long_range or None
            )
        elif long_range is not None:
            raise ValueError("long_range= needs a model with a long-range block")
        if self.long_range_energy is not None:
            # The cell is validated here and in `set_cell`, on the host; the
            # term's own per-call check would read the device every step.
            self.long_range_energy.check_cell = False
            if self.isolated:
                # O(chunk N) memory and no host reads; a step needs only first
                # derivatives.
                self.long_range_energy.isolated_kernel = "chunked"
                self.long_range_energy.kernel_dtype = _isolated_dtype(
                    model, isolated_dtype
                )
        if isolated_dtype is not None and not self.isolated:
            raise ValueError(
                "isolated_dtype sets the dtype of a cluster's all-pairs sum; this "
                f"system (pbc={list(self.pbc)}) is periodic"
            )
        self._check_cell(self.cell)
        # Checked here as well as per call, to fail before the memory probe.
        check_long_range_boundary(model, self.pbc, self.long_range_energy)
        self.model = model
        # A bare "cuda" means the current device; spelled out here so the
        # list, the partition views and the model agree on one device object.
        self.devices = [_indexed_device(d) for d in devices]
        #: One device and the GPU-direct list: the node set is the system
        #: itself, periodic through the list's image shifts, and nothing is
        #: partitioned, probed or rebuilt
        #: (:func:`~deepmd_lr.nvalchemi.partition.whole_system`).
        self._whole_system = len(self.devices) == 1
        if self.isolated and not self._whole_system:
            raise ValueError(
                "a cluster (pbc all False) runs on one device: its long-range "
                "sum runs over every pair and does not partition"
            )
        #: Slab axes whose faces atoms must stay inside. A cluster has none:
        #: nothing is summed over images, and its list searches its own box.
        self._open_axes = (
            [] if self.isolated else [k for k in range(3) if not self.pbc[k]]
        )
        if (
            self._whole_system
            and not self.isolated
            and np.linalg.det(self.cell) <= 0.0
        ):
            raise ValueError(
                "the cell is left-handed or degenerate (det <= 0); nvalchemiops' "
                "cell list refuses it. Swap two lattice vectors."
            )
        self.repartition_every = repartition_every
        if compute_virial and self.isolated:
            raise ValueError(_VIRIAL_ISOLATED)
        self.compute_virial = bool(compute_virial)
        #: Also assemble Born effective charge tensors, exposed as
        #: ``results['born_charge']`` ``(natoms, 3, 3)``. Costs one extra
        #: backward pass per partition; needed for MD in a field.
        self.compute_born = bool(compute_born)
        #: Uniform external field in V/angstrom, or None. See the class
        #: docstring: the force it adds is a local response, not the gradient
        #: of a field energy.
        self.efield = (
            None if efield is None else np.asarray(efield, dtype=np.float64).reshape(3)
        )
        if self.efield is not None:
            if not getattr(model, "has_long_range", False):
                raise ValueError(
                    "an external field needs Born effective charges, which "
                    "need the charge head -- this model has no `long_range` "
                    "block, so it predicts no charges."
                )
            if self.compute_virial:
                raise ValueError(_VIRIAL_UNDER_FIELD)
            # The field force is built from Z, so Z has to be assembled.
            self.compute_born = True
        self.charge_spin = charge_spin
        #: ``(c_t, delta)`` or None; see the class docstring. Absent, the
        #: model's own (``long_range.charge_filter``); ``False``, none.
        self.charge_filter = (
            getattr(model, "trained_charge_filter", None)
            if charge_filter is None
            else check_charge_filter(charge_filter)
        )
        if self.charge_filter is not None:
            if not getattr(model, "has_long_range", False):
                raise ValueError("charge_filter needs a model with a long_range block")
            if self.compute_born:
                raise ValueError(
                    "charge_filter cannot be combined with Born charges or an "
                    "efield yet: the Born tensor is built from unfiltered charges"
                )
        #: High-frequency dielectric constant. ``results['charge']`` and
        #: ``results['born_charge']`` -- and therefore the field force -- are
        #: scaled by sqrt(eps_inf); energies and field-free forces are not.
        #: Validated here so a typo fails at construction, not mid-run.
        screening_factor(eps_inf)
        self.eps_inf = float(eps_inf)
        #: Leave ``results`` as device tensors instead of the lazy numpy view.
        self.home = next(model.parameters()).device
        self.atype = np.asarray(atype, dtype=np.int64).reshape(-1)
        #: The types on the home device, uploaded once for the long-range term.
        self._atype_dev = torch.as_tensor(self.atype, device=self.home)
        #: The cell and its inverse on the home device. A barostat replaces
        #: the cell every step; the copies go through pinned memory with
        #: `non_blocking`, a ring of four slots so a copy still in flight is
        #: never overwritten, instead of a pageable upload that stalls the host.
        self._cell_dev = torch.as_tensor(
            self.cell, dtype=torch.float64, device=self.home
        )
        self._cell_inv_dev = torch.as_tensor(
            self._cell_inverse(), dtype=torch.float64, device=self.home
        )
        self._reference_cell_dev: torch.Tensor | None = None
        self._pinned = (
            torch.empty((4, 2, 3, 3), dtype=torch.float64).pin_memory()
            if self.home.type == "cuda"
            else None
        )
        self._pin_slot = 0
        if self.long_range_energy is not None:
            # The k-vectors follow the cell, which only `set_cell` changes.
            self.long_range_energy.set_k_vectors(self._cell_dev)
        #: Whether the caller asked for the geometry rule. Kept because the
        #: rule reads the cell, and `set_cell` can replace it mid-run: under a
        #: barostat a contraction would otherwise keep a skin that no longer
        #: honours the halo bound, and an expansion would keep one too small
        #: to buy the rebuild reduction it was chosen for.
        self._skin_is_auto = isinstance(skin, str)
        self._rcut = float(model.get_rcut())
        #: A ceiling the geometry rule may not exceed, lowered every time a
        #: device runs out of memory. Infinite until that happens, so a run
        #: that fits pays nothing for this.
        self._skin_cap = float("inf")
        #: How often the halo has been given back to free memory.
        self.skin_backoffs = 0
        self.list_growths = 0
        #: Fraction of usable device memory the halo may be sized for, or
        #: ``None`` to skip the measurement entirely.
        self.memory_headroom = (
            None if memory_headroom is None else float(memory_headroom)
        )
        #: Bytes the footprint costs per node, filled in by the probe step.
        self.bytes_per_node: float | None = None
        self.skin = self._auto_skin() if self._skin_is_auto else float(skin)
        #: Whether the first evaluation is still the probe. It runs at the
        #: floor skin, which is the smallest halo there is, so that the step
        #: that measures the footprint is also the one least likely to fail.
        # No probe on one device with the GPU list: the node count is the
        # atom count whatever the skin, so there is no halo to size.
        self._probe_pending = (
            self._skin_is_auto
            and not self._whole_system
            and self.memory_headroom is not None
            and any(torch.device(d).type == "cuda" for d in self.devices)
        )
        if self._probe_pending:
            self.skin = min(self.skin, MIN_AUTO_SKIN)

        # Compress BEFORE the replicas are made, so the deep copies carry the
        # tables; the run's charge state is folded in here, once.
        compress_model(model, compress, charge_spin=charge_spin)
        # One deepcopy per non-home device, held for the life of the driver
        # rather than rebuilt per call.
        self._replicas = device_replicas(model, self.devices)
        self._partitions = None
        self._reference_positions = None
        self._calls_since_rebuild = 0
        #: The halo the current partitions were built with, and the cell and
        #: per-atom fold images they were built in (see :meth:`_needs_rebuild`).
        self._halo_skin = 0.0
        self._images = None
        self._reach = 0.0

        self.rebuild_count = 0
        #: Completed evaluations. Several MD objects may share this evaluator,
        #: one after another, and all of them view the same forces buffer; a
        #: Thermo compares this count with the one it saw at its own last
        #: evaluation to know whether the forces are still its own.
        self.calls = 0
        self.results: dict[str, Any] = {}
        self.virial: np.ndarray | None = None
        self._virial_dev: torch.Tensor | None = None
        self.forces = torch.zeros(
            (len(self.atype), 3), dtype=torch.float64, device=self.home
        )

    @property
    def n_atoms(self) -> int:
        return len(self.atype)

    @property
    def cell(self) -> np.ndarray:
        """The cell ``(3, 3)``, lattice vectors as rows: the host copy.

        Under a barostat the device leads (:meth:`set_cell_device`) and this
        copy is refreshed on demand, which costs the one read of the device
        a log row, a checkpoint or a repartition then makes; a step reads
        nothing.
        """
        if self._cell_host_stale:
            self._cell_host = self._cell_dev.detach().cpu().numpy().reshape(3, 3)
            self._cell_host_stale = False
        return self._cell_host

    @cell.setter
    def cell(self, value) -> None:
        self._cell_host = np.asarray(value, dtype=np.float64).reshape(3, 3)
        self._cell_host_stale = False

    @property
    def virial_dev(self) -> torch.Tensor | None:
        """The last virial as a ``(3, 3)`` device tensor (``None`` without
        ``compute_virial``): what a barostat's pressure kernel should read,
        instead of re-uploading :attr:`virial`."""
        return self._virial_dev

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.cell)))

    @property
    def stress(self) -> np.ndarray | None:
        """Stress in eV/angstrom^3, ``-virial / volume`` as ASE defines it.

        ``None`` until a call has run with ``compute_virial``.
        """
        if self.virial is None:
            return None
        return -self.virial / self.volume

    def masses(self) -> np.ndarray:
        """Standard masses for this system, in amu."""
        return atomic_masses(self.model, self.atype)

    def clear_results(self) -> None:
        """Discard a failed evaluation: the results, the virial and the force
        buffer, filled with NaN in place because other objects hold views of
        it that must never be rebound. :attr:`Thermo.fresh` reads False for
        every Thermo on this evaluator until the next evaluation."""
        self.results.clear()
        self.virial = None
        self._virial_dev = None
        self.forces.fill_(float("nan"))

    def set_compute_virial(self, flag: bool) -> None:
        """Switch the virial on or off for the evaluations that follow.

        One evaluator can then carry a relaxation and an NVT stage without
        paying for the virial, and an NPT stage with it, instead of a second
        evaluator copying the model onto every device again. Refused under a
        field, for the reason the constructor refuses the combination. The
        first-step memory probe did not price the virial; should switching it
        on outgrow a device, the out-of-memory backoff shrinks the skin.
        """
        flag = bool(flag)
        if flag and self.efield is not None:
            raise ValueError(_VIRIAL_UNDER_FIELD)
        if flag and self.isolated:
            raise ValueError(_VIRIAL_ISOLATED)
        self.compute_virial = flag
        if not flag:
            self.virial = None
            self._virial_dev = None

    def _auto_skin(self) -> float:
        """The geometry rule, never above a ceiling memory has forced on us.

        The rule sizes the halo from the cell, the cutoff and the partition
        count, because those are what decide how much ghost work a skin buys
        and how often the rebuild runs. What it cannot see is how many bytes a
        node costs: that is a property of the model -- an uncompressed
        descriptor holds several times what a compressed one does -- and of
        the step, so no geometric rule can predict it. Memory therefore enters
        as a cap set by measurement: the first-step probe
        (:meth:`_fit_skin_to_memory`), lowered again whenever a device runs out.
        """
        if self.isolated:
            # One device, the list searched afresh every step: no halo.
            return 0.0
        return min(
            default_skin(self.cell, self._rcut, len(self.devices)), self._skin_cap
        )

    def _cell_inverse(self) -> np.ndarray:
        """``inv(cell)`` on the host; the identity for a cluster, whose cell
        (possibly all zero) nothing reads."""
        if self.isolated:
            return np.eye(3)
        return np.linalg.inv(self.cell)

    def _check_cell(self, cell: np.ndarray) -> None:
        """Refuse a degenerate cell before anything stores it. A cluster's
        cell is never read."""
        if self.isolated:
            return
        det = np.linalg.det(cell)
        if not np.isfinite(det) or abs(det) <= 0:
            raise ValueError("the cell is degenerate (zero volume)")

    def set_cell(self, cell: Any) -> None:
        """Replace the cell, keeping the partitioning while it still holds.

        A barostat deforms the cell a little every step. The partitions keep
        their nodes -- the same atoms and lattice images -- and re-express the
        images' offsets in the new cell (:meth:`Partition.set_cell`); the
        Verlet bound in :meth:`_needs_rebuild` counts the cell's change as well
        as the atoms', so a small deformation costs no repartition and a large
        one rebuilds at the next call. Before this, every step of NPT
        repartitioned: 66 against 38 ms/step at 12,288 water atoms.

        An ``"auto"`` skin is recomputed for the next rebuild, since the rule
        is a function of the cell; the current halo keeps the skin it was
        built with. An explicitly chosen skin is the caller's and is left
        alone.
        """
        cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        self._check_cell(cell)
        self.cell = cell
        self._upload_cell()
        self.cell_version += 1
        if self.long_range_energy is not None:
            self.long_range_energy.set_k_vectors(self._cell_dev)
        if self._partitions is not None:
            for part, device in zip(self._partitions, self.devices, strict=True):
                part.set_cell(self.cell, device_cell=self._cell_dev)
                nlist = part.neighbor_list_on(device)
                if nlist is not None:
                    nlist.cell_changed(self.cell)
        if self._skin_is_auto:
            self.skin = self._auto_skin()
            self._skin_stale = False

    def set_cell_device(self, cell: torch.Tensor) -> None:
        """Replace the cell from a device tensor, reading nothing back.

        The barostats' path: the cell they integrate lives on the device and
        changes every step, and the old seam copied it to the host and back
        twice per step. This copies it device-to-device into the evaluator's
        cell, inverts it there, re-expresses the partitions' images and the
        long-range term's k-vectors on the device, and leaves the host copy
        stale: :attr:`cell` refreshes it when something on the host asks. An
        ``"auto"`` skin is recomputed at the next rebuild rather than here.
        The neighbour list refreshes its search radius on the device at the
        next search, using the new cell and bin counts.
        """
        h = cell.detach().to(device=self.home, dtype=torch.float64).reshape(3, 3)
        self._cell_dev.copy_(h)
        self._cell_inv_dev.copy_(inverse_3x3(self._cell_dev))
        self._cell_host_stale = True
        self.cell_version += 1
        self._skin_stale = self._skin_is_auto
        if self.long_range_energy is not None:
            self.long_range_energy.set_k_vectors(self._cell_dev)
        if self._partitions is not None:
            for part, device in zip(self._partitions, self.devices, strict=True):
                part.set_cell_device(self._cell_dev)
                nlist = part.neighbor_list_on(device)
                if nlist is not None:
                    nlist.cell_changed_device()

    def _upload_cell(self) -> None:
        """The host cell and its inverse onto the home device, without a stall."""
        inverse = self._cell_inverse()
        if self._pinned is None:
            self._cell_dev.copy_(torch.as_tensor(self.cell))
            self._cell_inv_dev.copy_(torch.as_tensor(inverse))
            return
        slot = self._pinned[self._pin_slot]
        self._pin_slot = (self._pin_slot + 1) % self._pinned.shape[0]
        slot[0].copy_(torch.from_numpy(self.cell))
        slot[1].copy_(torch.from_numpy(inverse))
        self._cell_dev.copy_(slot[0], non_blocking=True)
        self._cell_inv_dev.copy_(slot[1], non_blocking=True)

    def _needs_rebuild(self, positions: torch.Tensor) -> bool:
        """Standard Verlet criterion on the cached halo.

        The halo covers ``rcut + skin``, and two atoms approaching each other
        can each contribute, so the pair separation shrinks by up to twice
        the largest displacement. Rebuilding once that exceeds the skin keeps
        every real neighbour inside the halo.
        """
        if self._partitions is None or self._reference_positions is None:
            return True
        if (
            self.repartition_every is not None
            and self._calls_since_rebuild >= self.repartition_every
        ):
            return True
        if self._whole_system:
            # Searched afresh every step: nothing can go stale.
            return False
        # Raw displacement, not minimum image, on purpose. The halo was built
        # around each atom's folded position at the last rebuild; a caller that
        # wraps an atom moves it by a lattice vector, and a minimum-image check
        # would call that no motion and keep a halo built around the old image.
        # A node moves by df_i + S dH: folded atom motion plus its image's
        # share of the cell deformation. Include one extra lattice step for
        # images just outside the halo. The Frobenius norm bounds ||dH||_2
        # without a decomposition.
        moved = positions - self._reference_positions
        if self._reference_cell_dev is not None and self._images is not None:
            dh = self._cell_dev - self._reference_cell_dev
            moved = moved - self._images @ dh.to(moved.dtype)
            drift = moved.norm(dim=1).max() + self._reach * dh.norm()
        else:
            drift = moved.norm(dim=1).max()
        # The one host read the halo needs: a bool, not the positions.
        return bool(2.0 * drift > self._halo_skin)

    def _check_open_axes(self, coord: np.ndarray) -> None:
        """Keep every atom inside the cell along an open axis.

        Nothing folds an open axis, but the reciprocal sum is still taken over
        the whole cell, so an atom that crossed a face would be summed as its
        image at the opposite face, across the vacuum. An atom moves less than
        half the skin before the next rebuild, so being that far inside the
        faces now keeps it inside until then.
        """
        open_axes = [k for k in range(3) if not self.pbc[k]]
        if not open_axes:
            return
        frac = coord @ np.linalg.inv(self.cell)
        widths = _perpendicular_widths(self.cell)
        margin = 0.5 * self.skin
        for k in open_axes:
            depth = np.minimum(frac[:, k], 1.0 - frac[:, k]) * widths[k]
            i = int(np.argmin(depth))
            if depth[i] < margin:
                raise ValueError(
                    f"atom {i} is {depth[i]:.2f} A inside a face of the cell "
                    f"along the open axis {k}, and must stay {margin:.2f} A "
                    "inside (half the skin): past the face the long-range sum "
                    "would put it across the vacuum. Give the slab more vacuum, "
                    "e.g. read_structure(..., vacuum=30.0)."
                )

    def _rebuild(self, tracked: torch.Tensor) -> None:
        """Re-partition at the current skin and restart the displacement clock."""
        if self._skin_stale:
            # Deferred from `set_cell_device`: the rule reads the host cell,
            # and a rebuild reads the host anyway.
            self.skin = self._auto_skin()
            self._skin_stale = False
        if self._whole_system:
            # Nothing to partition and nothing to copy to the host: the one
            # node set is the system, built once; a forced "rebuild"
            # (`repartition_every`) only restarts the counters.
            if self._partitions is None:
                self._partitions = [whole_system(self.n_atoms, self.pbc, self.cell)]
            self._reference_positions = tracked
            self._halo_skin = float("inf")
            self._images, self._reach = None, 0.0
            self._calls_since_rebuild = 0
            self.rebuild_count += 1
            return
        # The only host copy of the coordinates: the spatial decomposition is
        # a numpy/scipy routine, and it runs only when the skin says so.
        host = tracked.cpu().numpy()
        self._check_open_axes(host)
        self._partitions = partition_system(
            host,
            self.cell,
            self._rcut,
            len(self.devices),
            pbc=self.pbc,
            skin=self.skin,
        )
        self._reference_positions = tracked.clone()
        self._reference_cell_dev = self._cell_dev.clone()
        self._halo_skin = self.skin
        # Every atom is owned once, and its owned node is its fold image undone.
        images = np.zeros((self.n_atoms, 3))
        for part in self._partitions:
            images[part.owned] = -part.owned_image
        self._images = torch.as_tensor(images, dtype=torch.float64, device=self.home)
        shifts = [
            part.ghost_image + images[part.ghost]
            for part in self._partitions
            if len(part.ghost)
        ]
        widest = max(
            (float(np.linalg.norm(s, axis=1).max()) for s in shifts), default=0.0
        )
        self._reach = widest + np.sqrt(3.0)
        self._calls_since_rebuild = 0
        self.rebuild_count += 1

    def _nodes_per_device(self) -> dict[torch.device, int]:
        """How many nodes each device is holding under the current halo."""
        nodes: dict[torch.device, int] = {}
        for part, device in zip(self._partitions, self.devices, strict=True):
            key = torch.device(device)
            nodes[key] = nodes.get(key, 0) + int(part.n_node)
        return nodes

    def _fit_skin_to_memory(self) -> None:
        """Choose the halo the device can actually hold, from one measurement.

        The geometry rule knows how many *nodes* a skin costs but not what a
        node costs in bytes, and that coefficient spans a factor of thirty:
        measured here, 755 kB per node for an uncompressed long-range model
        against 24.3 kB compressed. So it is measured rather than assumed. The
        probe step runs at the floor skin, the smallest halo there is; its
        peak allocation divided by its node count is the coefficient, and the
        footprint is linear in the node count to 0.3 % (uncompressed) and
        1.2 % (compressed) across sizes and skins, which is what makes the
        extrapolation to another skin sound.

        Density is what turns a skin into a node count, and it enters through
        the node count this step actually *counted* rather than through
        ``N/V``: for a slab, an interface or anything with a void, the counted
        occupancy is right where a uniform density is not.

        The budget is a fraction of what this process could hold on the device
        -- free memory plus what its own allocator is already caching -- so a
        co-tenant process is accounted for. The result is only ever a ceiling:
        the geometry rule still decides the skin whenever memory is not the
        binding constraint, which on a half-empty device it never is.
        """
        nodes = self._nodes_per_device()
        ratios = []
        for device, count in nodes.items():
            if device.type != "cuda" or count == 0:
                continue
            peak = float(torch.cuda.max_memory_allocated(device))
            if peak <= 0.0:
                continue
            free, _total = torch.cuda.mem_get_info(device)
            usable = float(free) + float(torch.cuda.memory_reserved(device))
            ratios.append(self.memory_headroom * usable / peak)
            self.bytes_per_node = max(self.bytes_per_node or 0.0, peak / count)
        if not ratios:
            return
        affordable = skin_for_node_ratio(
            self.cell,
            self._rcut,
            len(self.devices),
            min(ratios),
            skin_from=self.skin,
        )
        wanted = default_skin(self.cell, self._rcut, len(self.devices))
        if affordable < wanted:
            warnings.warn(
                f"the halo the geometry rule asks for ({wanted:.2f} A) does not "
                f"fit in {self.memory_headroom:.0%} of this device's memory at "
                f"{self.bytes_per_node / 1024:.0f} kB per node, so it is capped "
                f"at {max(affordable, MIN_AUTO_SKIN):.2f} A. A compressed model "
                f"or more partitions would buy the halo back.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._skin_cap = min(self._skin_cap, max(affordable, MIN_AUTO_SKIN))

    def _evaluate(self, tracked: torch.Tensor) -> dict[str, Any]:
        """One evaluation, giving the halo back if a device runs out of memory.

        The halo is the one part of the footprint this object chooses, and the
        geometry rule chooses it blind to bytes: it grants the same 25 % of
        extra nodes whether each node costs a compressed descriptor's memory
        or an uncompressed one's several times that. At 98,304 atoms on eight
        devices that was the difference between running and an out-of-memory,
        while the old fixed 1 A skin fitted.

        The first-step probe predicts the footprint from a measured cost per
        node; this is the net under it, and it needs no prediction at all:
        on an out-of-memory the skin is halved, the decomposition rebuilt and
        the step retried, and the new value becomes a ceiling the geometry
        rule may not climb back over -- including through ``set_cell`` under a
        barostat. A run with memory to spare never enters this path.

        An explicitly chosen skin is the caller's and is never overridden; so
        is a skin already down at :data:`MIN_AUTO_SKIN`, where the halo is no
        longer what is filling the device.

        Expect it to overshoot by one halving. Measured at 41,472 atoms with
        the allocator capped between the two footprints, the retry took two
        reductions (3.34 -> 1.67 -> 0.84 A) where a fresh process fits at
        1.67: the retry runs in an allocator that has already been fragmented
        by the failed step, and a collection before it does not change that.
        The cap is one-way, so a run pays this once and keeps the smaller
        halo; forces are unaffected (2e-6 eV/A against an unconstrained run,
        which is the nondeterminism floor).
        """
        # The retry runs after the `except` block on purpose: leaving it unbinds
        # `exc`, which frees the failed step's traceback -- and every tensor its
        # frames hold -- before `empty_cache()` is asked to hand memory back.
        while True:
            try:
                return evaluate_partitioned(
                    self.model,
                    tracked,
                    self.atype,
                    self._cell_dev,
                    n_partitions=len(self.devices),
                    devices=self.devices,
                    partitions=self._partitions,
                    replicas=self._replicas,
                    compute_virial=self.compute_virial,
                    compute_born=self.compute_born,
                    charge_spin=self.charge_spin,
                    eps_inf=self.eps_inf,
                    pbc=self.pbc,
                    long_range_energy=self.long_range_energy,
                    charge_filter=self.charge_filter,
                    atype_device=self._atype_dev,
                )
            except Exception as exc:
                if (
                    not _is_out_of_memory(exc)
                    or not self._skin_is_auto
                    or self._whole_system
                ):
                    # On one device there is no halo to give back.
                    raise
                if self.skin <= MIN_AUTO_SKIN:
                    warnings.warn(
                        f"out of memory with the halo already at its floor "
                        f"({self.skin:.2f} A) after {self.skin_backoffs} "
                        f"reduction(s): the halo is not what is filling the "
                        f"device. Fewer atoms per device, a compressed model or "
                        f"more partitions are the levers left.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    raise
                # The traceback holds every frame of the failed forward, and
                # every frame holds tensors. Dropping them is what makes the
                # memory the retry needs available again.
                traceback.clear_frames(exc.__traceback__)
                shrunk = max(0.5 * self.skin, MIN_AUTO_SKIN)
            self._skin_cap = shrunk
            self.skin = shrunk
            self.skin_backoffs += 1
            torch.cuda.empty_cache()
            self._rebuild(tracked)
            warnings.warn(
                f"a device ran out of memory, so the halo skin was reduced to "
                f"{shrunk:.2f} A and the step retried; the geometry rule will "
                f"not exceed that for the rest of this run. Pass an explicit "
                f"`skin=` to take the decision yourself.",
                RuntimeWarning,
                stacklevel=3,
            )

    def __call__(self, positions: Any) -> tuple[float, torch.Tensor]:
        """Energy in eV and forces in eV/angstrom at ``positions``.

        ``positions`` may be a torch tensor -- including the one an integrator
        is mutating in place -- or anything ``np.asarray`` accepts. The
        returned tensor is reused between calls, so copy it if you need to
        keep a previous step's forces.

        With ``compute_virial``, :attr:`virial` and :attr:`stress` are also
        refreshed. They are not returned, so the ``(energy, forces)``
        signature an integrator consumes stays the same either way.
        """
        if isinstance(positions, torch.Tensor):
            tracked = positions.detach().to(device=self.home, dtype=torch.float64)
        else:
            tracked = torch.as_tensor(
                np.asarray(positions, dtype=np.float64).reshape(-1, 3),
                device=self.home,
            )
        if tracked.shape != (self.n_atoms, 3):
            raise ValueError(
                f"expected positions of shape {(self.n_atoms, 3)}, "
                f"got {tuple(tracked.shape)}"
            )

        if self._probe_pending:
            # A clean peak for the probe: this object's own first step, not
            # whatever else has run in this process. The one side effect the
            # measurement has, and it happens once.
            for device in {torch.device(d) for d in self.devices}:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

        if self._needs_rebuild(tracked):
            self._rebuild(tracked)
        self._calls_since_rebuild += 1

        out = self._evaluate(tracked)
        # The one download of the step: energy, the device status word and,
        # when asked for, the virial, in a single small copy. A list that
        # overflowed a buffer truncated the graph, so that step is redone
        # after the buffer has grown; this happens on the first steps of a
        # run, if at all, and the growth is kept.
        for _ in range(MAX_LIST_GROWTHS):
            energy, flags, virial = self._readback(out, tracked)
            if not (flags[STATUS_MATRIX_OVERFLOW] or flags[STATUS_EDGE_OVERFLOW]):
                break
            self._grow_lists()
            warnings.warn(
                "a neighbour-list buffer was too small for this step; it was "
                "widened and the step repeated",
                RuntimeWarning,
                stacklevel=2,
            )
            out = self._evaluate(tracked)
        else:
            raise RuntimeError(
                f"the neighbour list overflowed {MAX_LIST_GROWTHS} times in one "
                "step; the system's density is not what the list was sized for"
            )
        if flags[STATUS_OPEN_FACE]:
            self._raise_open_face(tracked)

        if self._probe_pending:
            self._probe_pending = False
            self._fit_skin_to_memory()
            self.skin = self._auto_skin()
            # The probe ran on the floor skin; the halo the rest of the run
            # wants is a different one, so the decomposition is rebuilt on the
            # next call rather than kept.
            self._reference_positions = None
        # Results stay on the device; `_LazyNumpy` hands out numpy on access,
        # so a caller that reads `results["charge"]` pays for that copy and a
        # caller that reads nothing pays for none.
        self.results = _LazyNumpy(out)
        self.virial = virial
        self._virial_dev = out.get("virial")
        # INVARIANT: this writes INTO the preallocated buffer and never
        # rebinds `self.forces`, so a warp view taken once stays live for the
        # life of the object -- which is what lets `Thermo` bind it a single
        # time instead of rewrapping every step. Turning either this or the
        # `+=` below into an assignment would silently freeze the forces every
        # MD driver sees. `tests/test_thermo.py` pins the pointer.
        self.forces.copy_(out["force"].to(device=self.home, dtype=torch.float64))
        if self.efield is not None:
            # Through `field_force` rather than an einsum written out again:
            # which index of Z contracts with E is a convention that has been
            # wrong here before, and one definition cannot drift from another.
            born = out["born_charge"].to(device=self.home, dtype=torch.float64)
            response = field_force(born[None], self.efield)[0]
            self.forces += response
            # Kept separately so a caller can watch the field term without
            # re-deriving it, and so `out["force"]` stays the field-free force.
            # A tensor, like the rest: downloaded only if read.
            self.results["field_force"] = response
        self.calls += 1
        return energy, self.forces

    # ------------------------------------------------------------ the readback

    def _lists(self):
        """The GPU neighbour lists the partitions hold, with their devices."""
        if self._partitions is None:
            return
        for part, device in zip(self._partitions, self.devices, strict=True):
            nlist = part.neighbor_list_on(device)
            if nlist is not None:
                yield nlist, device

    @property
    def list_resizes(self) -> int:
        """How often a neighbour list's buffers were resized (host syncs)."""
        return sum(nlist.resizes for nlist, _ in self._lists())

    def _device_status(self, tracked: torch.Tensor) -> torch.Tensor:
        """The step's status word on the home device: the lists' flags OR-ed
        together, plus the evaluator's own open-face check."""
        flags = torch.zeros(STATUS_LEN, dtype=torch.int64, device=self.home)
        for nlist, _device in self._lists():
            flags |= nlist.status.to(self.home)
        if self._whole_system and self._open_axes:
            # Nothing folds an open axis, and the reciprocal sum is taken over
            # the whole cell: an atom past a face would be summed as its image
            # at the opposite face, across the vacuum. The halo path checks
            # this on the host at every rebuild; here it is checked on the
            # device at every step and read back with the energy.
            frac = tracked @ self._cell_inv_dev
            open_frac = frac[:, self._open_axes]
            outside = ((open_frac < 0.0) | (open_frac > 1.0)).any()
            flags[STATUS_OPEN_FACE : STATUS_OPEN_FACE + 1] |= outside.to(
                torch.int64
            ).reshape(1)
        return flags

    def _readback(self, out: dict, tracked: torch.Tensor):
        """Energy (float), status flags (list of int) and the virial (numpy or
        None), downloaded together in one copy."""
        pieces = [
            out["energy"].reshape(1).to(torch.float64),
            self._device_status(tracked).to(torch.float64),
        ]
        virial = out.get("virial")
        if virial is not None:
            pieces.append(virial.detach().reshape(9).to(torch.float64))
        host = torch.cat(pieces).cpu()
        flags = [int(x) for x in host[1 : 1 + STATUS_LEN]]
        vir = (
            None
            if virial is None
            else host[1 + STATUS_LEN :].reshape(3, 3).numpy().copy()
        )
        return float(host[0]), flags, vir

    def _grow_lists(self) -> None:
        for nlist, _device in self._lists():
            if bool(nlist.status[:2].any()):
                nlist.grow()
        self.list_growths += 1

    def _raise_open_face(self, tracked: torch.Tensor) -> None:
        """Name the atom that left the cell along an open axis (the step has
        already been found wrong, so a host copy costs nothing that matters)."""
        frac = tracked.detach().cpu().numpy() @ np.linalg.inv(self.cell)
        widths = _perpendicular_widths(self.cell)
        worst = None
        for k in self._open_axes:
            depth = np.minimum(frac[:, k], 1.0 - frac[:, k]) * widths[k]
            i = int(np.argmin(depth))
            if worst is None or depth[i] < worst[0]:
                worst = (float(depth[i]), i, k)
        depth, i, k = worst
        raise ValueError(
            f"atom {i} is {-depth:.2f} A outside the cell along the open axis "
            f"{k}: past the face the long-range sum would put it across the "
            "vacuum. Give the slab more vacuum, e.g. read_structure(..., "
            "vacuum=30.0)."
        )
