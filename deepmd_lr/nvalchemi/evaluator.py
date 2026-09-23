# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate the model across several devices: energy, forces, virial.

This is the driver an MD engine calls. It owns the partitioning, the ghost
halo, the per-device model replicas and the decision of when the halo needs
rebuilding -- and nothing else. It has no timestep, no ensemble, no
thermostat and no velocities: those belong to whoever is integrating.

Energy and forces always; the virial and stress on request, since they cost
an extra strain leaf per partition and a second backward through the charge
head. The ``(energy, forces)`` return signature does not change either way,
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
from deepmd_lr.compress import (
    compress_model,
)
from deepmd_lr.nvalchemi.driver import (
    device_replicas,
    evaluate_partitioned,
)
from deepmd_lr.nvalchemi.partition import (
    default_skin,
    partition_system,
)

__all__ = [
    "DistributedEvaluator",
    "atomic_masses",
]


#: Below this the halo is no longer what is using the memory, so backing off
#: further would trade real speed for nothing.
MIN_AUTO_SKIN = 0.1


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
    Python floats, matching the numpy contract of ``evaluate_partitioned``.
    """

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if isinstance(value, torch.Tensor):
            value = value.detach()
            value = float(value) if value.ndim == 0 else value.cpu().numpy()
            dict.__setitem__(self, key, value)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return [(k, self[k]) for k in self]

    def values(self):
        return [self[k] for k in self]


class DistributedEvaluator:
    """Evaluate a ``dp_lr`` model over spatial partitions, one per device.

    Parameters
    ----------
    model
        A ``dp_lr`` model, already on its home device.
    atype, cell
        Atom types ``(natoms,)`` and the cell ``(3, 3)``. Both fixed for the
        life of the driver; a changing cell means rebuilding it.
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
        range. The value actually used is :attr:`skin`, and an ``"auto"`` skin
        is halved and capped whenever a device runs out of memory, since the
        geometry rule cannot see how many bytes a node costs -- see
        :meth:`_evaluate`. An explicit value is never overridden.
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
        and a second backward through the charge head.
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
    launch_threads
        Run each device's partitions in its own thread while the forwards are
        queued. ``None`` (default) means on whenever more than one device is
        in use; see :func:`~deepmd_lr.nvalchemi.driver.evaluate_partitioned`.
    results_on_device
        Keep :attr:`results` as device tensors. By default they are wrapped so
        that each entry is converted to numpy on first access -- the forces
        the integrator consumes never leave the GPU either way.
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
        skin: float | str = "auto",
        repartition_every: int | None = None,
        compute_virial: bool = False,
        compute_born: bool = False,
        efield: Any = None,
        charge_spin: torch.Tensor | tuple[float, float] | None = None,
        compress: bool | float = False,
        eps_inf: float = 1.0,
        results_on_device: bool = False,
        launch_threads: bool | None = None,
    ) -> None:
        if not devices:
            raise ValueError("DistributedEvaluator needs at least one device")
        if isinstance(skin, str):
            if skin != "auto":
                raise ValueError(f"skin must be a number or 'auto', got {skin!r}")
        elif skin < 0:
            raise ValueError(f"skin must be non-negative, got {skin}")

        self.model = model
        self.devices = list(devices)
        self.repartition_every = repartition_every
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
                raise ValueError(
                    "compute_virial and efield cannot be combined: the "
                    "field's contribution to the stress is a separate "
                    "derivation, and reporting the field-free virial under a "
                    "field would be quietly wrong."
                )
            # The field force is built from Z, so Z has to be assembled.
            self.compute_born = True
        self.charge_spin = charge_spin
        #: High-frequency dielectric constant. ``results['charge']`` and
        #: ``results['born_charge']`` -- and therefore the field force -- are
        #: scaled by sqrt(eps_inf); energies and field-free forces are not.
        #: Validated here so a typo fails at construction, not mid-run.
        screening_factor(eps_inf)
        self.eps_inf = float(eps_inf)
        #: Leave ``results`` as device tensors instead of the lazy numpy view.
        self.results_on_device = bool(results_on_device)
        #: One thread per device in the launch phase; ``None`` means on
        #: whenever more than one device is used. See `evaluate_partitioned`.
        self.launch_threads = launch_threads
        self.home = next(model.parameters()).device
        self.atype = np.asarray(atype, dtype=np.int64).reshape(-1)
        self.cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
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
        self.skin = self._auto_skin() if self._skin_is_auto else float(skin)

        # Compress BEFORE the replicas are made, so the deep copies carry the
        # tables; the run's charge state is folded in here, once.
        compress_model(model, compress, charge_spin=charge_spin)
        # One deepcopy per non-home device, held for the life of the driver
        # rather than rebuilt per call.
        self._replicas = device_replicas(model, self.devices)
        self._partitions = None
        self._reference_positions = None
        self._calls_since_rebuild = 0

        self.rebuild_count = 0
        self.results: dict[str, Any] = {}
        self.virial: np.ndarray | None = None
        self.forces = torch.zeros(
            (len(self.atype), 3), dtype=torch.float64, device=self.home
        )

    @property
    def n_atoms(self) -> int:
        return len(self.atype)

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

    def _auto_skin(self) -> float:
        """The geometry rule, never above a ceiling memory has forced on us.

        The rule sizes the halo from the cell, the cutoff and the partition
        count, because those are what decide how much ghost work a skin buys
        and how often the rebuild runs. What it cannot see is how many bytes a
        node costs: that is a property of the model -- an uncompressed
        descriptor holds several times what a compressed one does -- and of
        the step, so no geometric rule can predict it. Memory therefore enters
        as a cap, set by measurement when a device actually runs out.
        """
        return min(
            default_skin(self.cell, self._rcut, len(self.devices)), self._skin_cap
        )

    def set_cell(self, cell: Any) -> None:
        """Replace the cell and invalidate the partitioning.

        A barostat moves the cell every step, and the partitioning was built
        against the old one. An ``"auto"`` skin is recomputed here too, since
        the rule is a function of the cell. Invalidation here is
        **correctness, not caching**:
        :meth:`_needs_rebuild` watches atom displacement only and cannot see a
        cell change, so nothing else would ever notice. A stale halo puts ghost
        images at the wrong offsets and the forces are quietly wrong -- there
        is no natural backstop for that, which is why this is unconditional
        rather than left to the caller's judgement.
        """
        self.cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        self._partitions = None
        self._reference_positions = None
        # The rule is a function of the cell, so a new cell means a new answer.
        # An explicitly chosen skin is the caller's and is left alone.
        if self._skin_is_auto:
            self.skin = self._auto_skin()

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
        moved = float((positions - self._reference_positions).norm(dim=1).max())
        return 2.0 * moved > self.skin

    def _rebuild(self, tracked: torch.Tensor) -> None:
        """Re-partition at the current skin and restart the displacement clock."""
        # The only host copy of the coordinates: the spatial decomposition is
        # a numpy/scipy routine, and it runs only when the skin says so.
        self._partitions = partition_system(
            tracked.cpu().numpy(),
            self.cell,
            float(self.model.get_rcut()),
            len(self.devices),
            pbc=(True, True, True),
            skin=self.skin,
        )
        self._reference_positions = tracked.clone()
        self._calls_since_rebuild = 0
        self.rebuild_count += 1

    def _evaluate(self, tracked: torch.Tensor) -> dict[str, Any]:
        """One evaluation, giving the halo back if a device runs out of memory.

        The halo is the one part of the footprint this object chooses, and the
        geometry rule chooses it blind to bytes: it grants the same 25 % of
        extra nodes whether each node costs a compressed descriptor's memory
        or an uncompressed one's several times that. At 98,304 atoms on eight
        devices that was the difference between running and an out-of-memory,
        while the old fixed 1 A skin fitted.

        Predicting the footprint instead would mean guessing bytes per node
        for a model this class never inspects. Reacting to it needs no guess:
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
        while True:
            try:
                return evaluate_partitioned(
                    self.model,
                    tracked,
                    self.atype,
                    self.cell,
                    n_partitions=len(self.devices),
                    devices=self.devices,
                    partitions=self._partitions,
                    replicas=self._replicas,
                    compute_virial=self.compute_virial,
                    compute_born=self.compute_born,
                    charge_spin=self.charge_spin,
                    eps_inf=self.eps_inf,
                    launch_threads=self.launch_threads,
                )
            except Exception as exc:
                if not _is_out_of_memory(exc) or not self._skin_is_auto:
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

        if self._needs_rebuild(tracked):
            self._rebuild(tracked)
        self._calls_since_rebuild += 1

        out = self._evaluate(tracked)
        # Results stay on the device; `_LazyNumpy` hands out numpy on access,
        # so a caller that reads `results["charge"]` pays for that copy and a
        # caller that reads nothing pays for none. `results_on_device` skips
        # the wrapper for callers that want the tensors themselves.
        self.results = out if self.results_on_device else _LazyNumpy(out)
        self.virial = (
            None if out.get("virial") is None else out["virial"].detach().cpu().numpy()
        )
        # INVARIANT: this writes INTO the preallocated buffer and never
        # rebinds `self.forces`, so a warp view taken once stays live for the
        # life of the object -- which is what lets `Thermo` bind it a single
        # time instead of rewrapping every step. Turning either this or the
        # `+=` below into an assignment would silently freeze the forces every
        # MD driver sees. `tests/test_nvalchemi.py` pins the pointer.
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
            self.results["field_force"] = (
                response if self.results_on_device else response.cpu().numpy()
            )
        return float(out["energy"]), self.forces
