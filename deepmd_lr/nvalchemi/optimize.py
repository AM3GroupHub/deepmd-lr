# SPDX-License-Identifier: LGPL-3.0-or-later
"""Relaxation on the GPU (FIRE, FIRE2, L-BFGS), on the evaluator the MD shares.

.. code-block:: python

    ev = build_evaluator("model.ckpt.pt", atoms, devices=["cuda:0"])
    FIRE2(ev, atoms).run(fmax=0.05, steps=2000)   # RelaxationError if it went wrong
    thermalize_momenta(atoms, temperature_K=300)  # ase.md.velocitydistribution
    Stationary(atoms)
    Langevin(ev, atoms, 0.5 * units.fs, temperature_K=300,
             friction=0.01 / units.fs).run(10_000)  # same evaluator, no second copy

Three algorithms, as in ASE, each on its nvalchemiops kernel:
- :class:`FIRE`, the original (Bitzek et al. 2006), with ASE's defaults;
- :class:`FIRE2`, FIRE 2.0 (Guenole et al. 2020), with nvalchemiops' more
  cautious defaults;
- :class:`LBFGS`, limited-memory BFGS with a ``maxstep`` trust region and no
  line search (nvalchemiops 0.5), under ASE's ``memory`` and ``maxstep``.

On a 192-water MD snapshot (``examples/06-md_nvalchemi/water_192.xyz``) FIRE
reached 0.05 eV/A in 370 steps, FIRE2 in 450 and LBFGS in 212; from a packed
box pre-relaxed with TIP3P, in 170, 160 and 229, LBFGS ending 1.2 eV lower.

A freshly packed box can hold tens of eV of clashes, and MD would release that
as heat in the first few hundred fs, pushing the model into regions it never
saw -- so relax first.

**The cell, optionally.** ``relax_cell`` chooses what moves besides the atoms:

- ``None`` (the default): nothing; the cell is fixed.
- ``"isotropic"``: the cell scales uniformly and keeps its shape. The choice for
  a liquid, which has no shear stiffness: ``"full"`` lets its cell shear.
- ``"orthorhombic"``: the three lengths of a diagonal cell move independently.
- ``"full"``: all nine entries move.

The target is ``pressure_au`` (eV/A^3, e.g. ``1e-4 * units.GPa``), and
``run(smax=...)`` says how close the stress must come to it. The evaluator must
compute the virial (``evaluator.set_compute_virial(True)``), and the new cell is
pushed to it every step; the partitions follow it, rebuilding only once the
cell or the atoms have moved enough.

The cell moves on nvalchemiops' own variable-cell kernels. FIRE2's coupled
coordinate-cell step and L-BFGS's coordinate-cell step relax the atoms and the
cell together; FIRE, which has no coupled kernel upstream, steps the extended
array its cell filter packs (``pack_positions_with_cell``) and the atoms then
ride the cell affinely, as the two coupled kernels make them. All three carry
six cell parameters, one triangle of the cell matrix, so the cell cannot
rotate while it relaxes. What this module adds around the kernels:

- **The cell force** FIRE and FIRE2 are given is -d(E + PV)/dH =
  -V H^-T (sigma + P) for the rows of ASE's cell, projected onto the mode's
  directions (:meth:`_Relaxation._cell_force`); since 0.5.0 nvalchemiops'
  ``stress_to_cell_force`` computes the same gradient (0.4 was written for
  column cells and was 0.70 eV/A off on a sheared water cell; a test pins the
  agreement). L-BFGS takes the stress itself, projected the same way.
- **The frame.** The kernels want the cell triangular: FIRE and FIRE2 the
  lower triangle of ASE's rows (ASE's standard form); L-BFGS, whose kernel
  holds the lattice vectors as columns, the upper. A cell that is not already
  in the form its kernel needs is rotated into it at the start of ``run``,
  positions with it, and at the end everything is rotated back into the
  orientation of the start: the first lattice vector keeps its direction and
  the second its plane, as ASE's standard form has it. Log rows and
  trajectory frames are written in that frame too. A diagonal cell is never
  rotated, and nothing is rotated for FIRE and FIRE2 when the cell already
  is ASE's standard form.
- **The cell's weight.** The cell force is not divided by the atom count
  (nvalchemiops' ``cell_force_scale`` is ``1/N`` for FIRE2), which makes the
  cell about as heavy as one atom; in the measurement that chose it (192 and
  12,288 atoms alike) FIRE2 took 390 steps from a 4% stretch and 1000-1570
  with the force divided by N. For L-BFGS the chart scale ``kappa`` is the
  cube root of the starting volume for the same reason.

**Divergence is always fatal.** A model without a short-range repulsive wall can
let a minimiser collapse atoms onto each other from a badly distorted start.
Forces growing tenfold, more than 2 eV/atom released, the cell volume changing
more than twofold, or anything non-finite stops the run with a
:class:`RelaxationError` naming the cause. Pre-relax such a box with a classical
force field instead.

With ``check=True`` (the default) it also refuses a relaxation that did not
converge, one that changed the covalent-bond topology between non-metal atoms,
and one whose energy (with the cell, enthalpy) rose. That check applies only
without a field: under a field the minimiser seeks zero total force, not
minimum energy.

**Fixed atoms** (ASE's ``FixAtoms`` on ``atoms``) stay where they are: the
kernel sees no force on them, and ``fmax`` is the largest force on the others,
as in ASE. A cell relaxation would strain them with the cell, so it refuses
them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.cell import Cell
from ase.constraints import FixAtoms

from deepmd_lr.nvalchemi.md import (
    _check_atoms,
    _check_evaluator,
    _Logger,
    _sync_atoms,
    _Trajectory,
)
from deepmd_lr.nvalchemi.thermo import Thermo, _fixed_atoms

__all__ = ["FIRE", "FIRE2", "LBFGS", "RelaxationError"]

#: Elements whose mutual covalent bonds the topology check guards. Metals are
#: left out on purpose: an ion's coordination shell may legitimately change.
NONMETALS = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
)


class RelaxationError(RuntimeError):
    """The relaxation diverged, did not converge, or broke the structure."""


def _covalent_bonds(atoms: Atoms) -> set:
    """Bonded pairs among non-metal atoms: ASE covalent radii x 1.2."""
    from ase.neighborlist import natural_cutoffs, neighbor_list

    folded = atoms.copy()
    folded.calc = None
    folded.wrap()
    i, j = neighbor_list("ij", folded, natural_cutoffs(folded, mult=1.2))
    sym = np.array(folded.get_chemical_symbols())
    ok = np.isin(sym, list(NONMETALS))
    keep = (i < j) & ok[i] & ok[j]
    return set(zip(i[keep].tolist(), j[keep].tolist(), strict=True))


def _check_fire(dt, dtmax, dtmin, Nmin, finc, fdec, astart, fa) -> None:
    problems = []
    if not 0 < dtmin <= dt <= dtmax:
        problems.append(f"0 < dtmin <= dt <= dtmax (got {dtmin}, {dt}, {dtmax})")
    if not int(Nmin) >= 0:
        problems.append(f"Nmin >= 0 (got {Nmin})")
    if not (finc > 1 and 0 < fdec < 1):
        problems.append(f"finc > 1 and 0 < fdec < 1 (got {finc}, {fdec})")
    if not (0 < astart < 1 and 0 < fa <= 1):
        problems.append(f"0 < astart < 1 and 0 < fa <= 1 (got {astart}, {fa})")
    if problems:
        raise ValueError("FIRE parameters need " + "; ".join(problems))


_CELL_MODES = ("isotropic", "orthorhombic", "full")


def _standard_form(cell: np.ndarray, form: str) -> tuple[np.ndarray, np.ndarray]:
    """ASE's standard form of a cell: ``(rcell, Q)`` with ``rcell @ Q == cell``,
    ``rcell`` lower- or upper-triangular and ``Q`` orthogonal."""
    rcell, q = Cell(np.asarray(cell, dtype=np.float64)).standard_form(form)
    # Copies: ASE's upper form comes out of reversed views, which torch refuses.
    return np.array(rcell, dtype=np.float64), np.array(q, dtype=np.float64)


def _is_identity(r: np.ndarray) -> bool:
    return bool(np.abs(r - np.eye(3)).max() < 1e-12)


class _Relaxation:
    """What the relaxations share: a Thermo of their own on the shared
    evaluator, the loop with its divergence guard, the checks, the log, the
    trajectory, observers, and the cell's frame. A subclass owns its kernel's
    buffers and moves the atoms, and with ``relax_cell`` the cell, once per
    :meth:`_move`.

    Parameters
    ----------
    evaluator
        The :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator` the MD
        stages will share; the relaxation builds its own ``Thermo`` on it.
    atoms
        The system, in the evaluator's atom order and cell. Written back at
        the end: relaxed positions (and cell), zero momenta, energy and forces.
    relax_cell
        ``None``, ``"isotropic"``, ``"orthorhombic"`` or ``"full"``: what the
        cell may do (see the module docstring).
    pressure_au
        The target pressure of a cell relaxation, in eV/A^3.
    logfile, loginterval
        ``'-'`` (stdout), a path, or None; one row every ``loginterval`` steps.
    trajectory
        A ``.traj`` or extxyz path, or None: a frame (positions, energy,
        forces) at every log row and at the end -- the end of a divergence
        included, which is where a failed relaxation went.
    maxstep
        Largest displacement of any atom per step, in angstrom.
    check
        Refuse a bad relaxation (see the module docstring). Divergence is
        refused either way.
    """

    #: How often the loop reads the largest force back: every step for a
    #: quasi-Newton method, whose restart kicks a converged structure by
    #: ``maxstep`` if it is stepped once more; every ten for FIRE.
    _check_every = 10
    #: The triangle of ASE's cell matrix the kernel's six cell parameters fill.
    _cell_form = "lower"

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        *,
        relax_cell: str | None = None,
        pressure_au: float = 0.0,
        logfile: Any = "-",
        loginterval: int = 50,
        trajectory: Any = None,
        maxstep: float = 0.1,
        check: bool = True,
    ) -> None:
        import warp as wp

        _check_evaluator(evaluator)
        _check_atoms(evaluator, atoms)
        if relax_cell not in (None, *_CELL_MODES):
            raise ValueError(
                f"relax_cell must be None or one of {', '.join(_CELL_MODES)}; "
                f"got {relax_cell!r}"
            )
        if relax_cell is not None and not all(
            getattr(evaluator, "pbc", (True, True, True))
        ):
            raise ValueError(
                "relax_cell moves the cell, and a slab's open axis is not a "
                "periodic length a stress can act on; relax the positions only"
            )
        fixed = _fixed_atoms(atoms)
        if relax_cell is not None and len(fixed):
            raise ValueError(
                "relax_cell strains every position with the cell, so fixed "
                "atoms would not stay fixed; relax the positions only"
            )
        self.relax_cell = relax_cell
        self.pressure_au = float(pressure_au)
        if relax_cell is None and self.pressure_au != 0.0:
            raise ValueError(
                "pressure_au is the target of a cell relaxation; pass relax_cell too"
            )
        if relax_cell is not None and not evaluator.compute_virial:
            raise ValueError(
                "a cell relaxation needs the virial: build the evaluator with "
                "compute_virial=True or call evaluator.set_compute_virial(True)"
            )
        cell = np.asarray(evaluator.cell)
        if relax_cell == "orthorhombic" and np.any(cell != np.diag(np.diag(cell))):
            raise ValueError(
                "relax_cell='orthorhombic' keeps every angle at 90 degrees, so it "
                "needs a diagonal cell; use 'full' for this one"
            )
        self.atoms = atoms
        self.thermo = Thermo.from_atoms(evaluator, atoms)
        # A relaxation starts, and hands atoms back, at rest.
        self.thermo.set_state(velocities=np.zeros((len(atoms), 3)))
        self.maxstep = float(maxstep)
        if not self.maxstep > 0:
            raise ValueError("maxstep must be positive")
        self.check = bool(check)
        self.loginterval = max(1, int(loginterval))
        self.nsteps = 0
        self.converged = False
        th = self.thermo
        self._x = wp.to_torch(th.positions)  # views of the Thermo's buffers
        self._f = wp.to_torch(th.forces)
        #: ``(Q, Q0)`` while a run is in the kernel's frame: the cell was
        #: rotated by ``Q`` (``cell @ Q.T``) into the triangle the kernel
        #: fills, and ``Q0`` is the start's lower standard form, the
        #: orientation everything comes back in.
        self._frame: tuple[np.ndarray, np.ndarray] | None = None
        self._observers: list = []
        self._last_recorded: int | None = None
        self._traj = None if trajectory is None else _Trajectory(trajectory, False)
        columns = [
            ("step", 10, "d"),
            ("Epot[eV]", 18, ".6f"),
            ("fmax[eV/A]", 14, ".5f"),
        ]
        if relax_cell is not None:
            columns += [
                ("P[GPa]", 12, ".5f"),
                ("dS[GPa]", 12, ".5f"),
                ("V[A^3]", 14, ".3f"),
            ]
        self._logger = (
            None if logfile is None else _Logger(logfile, columns, append=False)
        )

    @property
    def evaluator(self) -> Any:
        """The shared evaluator (read-only; the Thermo holds the reference)."""
        return self.thermo.evaluator

    def attach(
        self, fn: Callable, interval: int = 1, *args: Any, **kwargs: Any
    ) -> None:
        """Call ``fn(*args, **kwargs)`` every ``interval`` steps."""
        if int(interval) < 1:
            raise ValueError("interval must be at least 1")
        self._observers.append((fn, int(interval), args, kwargs))

    def fmax(self) -> float:
        """The largest force on any atom not fixed, eV/A (one scalar read)."""
        return float(torch.linalg.vector_norm(self._f, dim=1).max())

    def stress_residual(self) -> float:
        """How far the stress is from the target ``-pressure_au``, in eV/A^3,
        along the directions the cell may move: the mean for ``"isotropic"``,
        the diagonal for ``"orthorhombic"``, every component for ``"full"``."""
        if self.relax_cell is None:
            raise ValueError("the cell is fixed (relax_cell=None)")
        h = np.asarray(self.evaluator.cell, dtype=np.float64)
        d = -self._enthalpy_virial(h) / abs(np.linalg.det(h))
        if self.relax_cell == "isotropic":
            return abs(float(np.trace(d))) / 3.0
        if self.relax_cell == "orthorhombic":
            return float(np.abs(np.diag(d)).max())
        return float(np.abs(d).max())

    def _enthalpy_virial(self, h: np.ndarray) -> np.ndarray:
        """The virial less P V: -d(E + P V)/d(strain), symmetrised."""
        w = np.asarray(self.evaluator.virial, dtype=np.float64)
        return 0.5 * (w + w.T) - self.pressure_au * abs(np.linalg.det(h)) * np.eye(3)

    def _cell_force(self, h: np.ndarray) -> np.ndarray:
        """-d(E + P V)/dH at fixed fractional coordinates, for the rows of
        ASE's cell, along the mode's free directions."""
        s = self._enthalpy_virial(h)
        if self.relax_cell == "isotropic":
            return float(np.trace(s)) / float(np.sum(h * h)) * h
        g = np.linalg.inv(h).T @ s
        if self.relax_cell == "orthorhombic":
            return np.diag(np.diag(g))
        return g

    def _enthalpy(self) -> float:
        return self.thermo.energy + self.pressure_au * self.evaluator.volume

    def _reset(self) -> None:
        """Start a run. Here: into the kernel's frame, if the cell is not in
        the triangular form it needs; a subclass then resets what its kernel
        carries between steps."""
        self._frame = None
        if self.relax_cell is None:
            return
        cell = np.array(self.evaluator.cell, dtype=np.float64)
        rcell, q = _standard_form(cell, self._cell_form)
        self._frame = (q, _standard_form(cell, "lower")[1])
        if not _is_identity(q):
            self._rotate(q.T)
            self.thermo.set_cell(rcell)

    def _rotate(self, r: np.ndarray) -> None:
        """Positions ``-> positions @ r`` on the device."""
        r = torch.as_tensor(np.ascontiguousarray(r), device=self._x.device)
        self._x.copy_(self._x @ r)

    def _user_frame(self, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The cell as it is reported: out of the kernel's frame, in the
        orientation of the start (its lower standard form). Returns the cell
        and the rotation ``r`` that takes device positions and forces there
        (``x @ r``), the identity when no frame is in use."""
        if self._frame is None:
            return cell, np.eye(3)
        q, q0 = self._frame
        lower, q1 = _standard_form(cell @ q, "lower")
        return lower @ q0, q @ q1.T @ q0

    def _restore(self) -> None:
        """Leave the kernel's frame at the end of a run, re-evaluating once
        when that is a rotation, so forces and virial are reported in it."""
        if self._frame is None:
            return
        cell, r = self._user_frame(np.array(self.evaluator.cell, dtype=np.float64))
        self._frame = None
        if _is_identity(r):
            return
        self._rotate(r)
        self.thermo.set_cell(cell)
        self.thermo.evaluate()

    def _move(self) -> None:
        """One step of the atoms, and with ``relax_cell`` of the cell too."""
        raise NotImplementedError

    def _converged(self, f: float, fmax: float, smax: float) -> bool:
        return f < fmax and (self.relax_cell is None or self.stress_residual() < smax)

    def run(
        self,
        fmax: float = 0.05,
        steps: int = 100_000_000,
        smax: float = 0.01 * units.GPa,
    ) -> bool:
        """Relax until the largest force is below ``fmax`` (eV/A) and, with
        ``relax_cell``, the stress within ``smax`` (eV/A^3) of the target, or
        until ``steps`` have run. Returns whether it converged; raises
        :class:`RelaxationError` on divergence, and with ``check`` on the
        other failures."""
        for _ in self.irun(fmax, steps, smax):
            pass
        return self.converged

    def irun(
        self,
        fmax: float = 0.05,
        steps: int = 100_000_000,
        smax: float = 0.01 * units.GPa,
    ) -> Iterator[int]:
        """Generator form of :meth:`run`; yields the step count."""
        th = self.thermo
        if self.relax_cell is not None and not np.array_equal(
            self.atoms.cell.array, self.evaluator.cell
        ):
            th.set_cell(self.atoms.cell.array)  # atoms edited between runs
        th.set_state(positions=self.atoms.get_positions())
        self._reset()
        th.evaluate()
        n = th.n_atoms
        e0, f0, v0 = self._enthalpy(), self.fmax(), self.evaluator.volume
        bonds0 = _covalent_bonds(self.atoms) if self.check else None
        self._record(f0)
        diverged = None
        self.converged = self._converged(f0, fmax, smax)
        try:
            for _ in range(int(steps)):
                if self.converged:
                    break
                self._move()
                th.evaluate()
                self.nsteps += 1
                if not math.isfinite(th.energy):
                    diverged = "non-finite energy"
                    break
                if (
                    self.nsteps % self._check_every == 0
                    or self.nsteps % self.loginterval == 0
                ):
                    f = self.fmax()
                    released = (e0 - self._enthalpy()) / n
                    if (
                        not math.isfinite(f)
                        or f > max(10.0 * f0, 50.0)
                        or released > 2.0
                    ):
                        diverged = (
                            f"max|F| reached {f:.3g} eV/A and the energy "
                            f"fell {released:.3g} eV/atom"
                        )
                        break
                    grown = self.evaluator.volume / v0
                    if not 0.5 < grown < 2.0:
                        diverged = f"the cell volume changed {grown:.3g}-fold"
                        break
                    self.converged = self._converged(f, fmax, smax)
                    if self.nsteps % self.loginterval == 0:
                        self._record(f)
                for fn, interval, args, kwargs in self._observers:
                    if self.nsteps % interval == 0:
                        fn(*args, **kwargs)
                yield self.nsteps
        finally:
            if math.isfinite(th.energy):
                self._restore()
                _sync_atoms(self.atoms, th)
        f1 = self.fmax()
        if self._last_recorded != self.nsteps:
            self._record(f1)
        if diverged:
            why = (
                "the model holds no equilibrium near this density at this pressure"
                if diverged.startswith("the cell")
                else "the model lacks short-range repulsion somewhere near this "
                "structure and the minimiser fell in. Pre-relax the box with a "
                "classical force field, then start from that structure"
            )
            raise RelaxationError(
                f"relaxation diverged after {self.nsteps} steps ({diverged}): {why}"
            )
        if not self.check:
            return
        problems = []
        if not self.converged:
            stress = (
                ""
                if self.relax_cell is None
                else f"; stress {self.stress_residual() / units.GPa:.4f} GPa off "
                f"the target, against {smax / units.GPa:g}"
            )
            problems.append(
                f"not converged (max|F| {f1:.3f} eV/A against {fmax}{stress})"
            )
        e1 = self._enthalpy()
        if self.evaluator.efield is None and e1 > e0:
            what = "energy" if self.relax_cell is None else "enthalpy"
            problems.append(f"{what} rose ({e0:.3f} -> {e1:.3f} eV)")
        bonds1 = _covalent_bonds(self.atoms)
        if bonds1 != bonds0:
            changed = sorted(bonds0 ^ bonds1)
            problems.append(
                f"covalent topology changed at {len(changed)} pair(s), "
                f"e.g. atoms {changed[:3]}"
            )
        if problems:
            raise RelaxationError(
                "relaxation rejected: "
                + "; ".join(problems)
                + ". check=False accepts it anyway"
            )

    def _record(self, fmax: float) -> None:
        """A log row and a trajectory frame for the current step."""
        self._last_recorded = self.nsteps
        th = self.thermo
        cell, r = self._user_frame(np.array(self.evaluator.cell, dtype=np.float64))
        if self._logger is not None:
            row = {"step": self.nsteps, "Epot[eV]": th.energy, "fmax[eV/A]": fmax}
            if self.relax_cell is not None:
                v = self.evaluator.volume
                row["P[GPa]"] = np.trace(self.evaluator.virial) / (3 * v) / units.GPa
                row["dS[GPa]"] = self.stress_residual() / units.GPa
                row["V[A^3]"] = v
            self._logger.write(row)
        if self._traj is not None:
            frame = Atoms(
                self.atoms.get_chemical_symbols(),
                positions=th.positions.numpy() @ r,
                cell=cell,
                pbc=self.atoms.pbc,
            )
            if len(th.fixed):
                frame.set_constraint(FixAtoms(indices=th.fixed))
            frame.calc = SinglePointCalculator(
                frame,
                energy=th.energy,
                forces=th.get_forces(apply_constraint=False) @ r,
            )
            frame.info["step"] = self.nsteps
            self._traj.write(frame)

    def close(self) -> None:
        """Close the log and trajectory files."""
        if self._logger is not None:
            self._logger.close()
        if self._traj is not None:
            self._traj.close()


class FIRE(_Relaxation):
    """FIRE, the original algorithm (Bitzek et al., Phys. Rev. Lett. 97, 170201
    (2006)), through nvalchemiops' ``fire_step``.

    While the power P = F.v stays positive, each step mixes the velocity toward
    the force, kicks and moves; the moment P turns negative the velocity is
    zeroed and the timestep shrinks. ASE's names and defaults, and, as in ASE,
    unit masses: the timestep is a relaxation parameter, not a physical one.
    (With real masses, oxygen moved 16x slower and water took 2-6x as many
    steps.) Each atom's step is capped at ``maxstep`` on its own.

    With ``relax_cell`` the cell's six parameters join the atoms in the
    extended array nvalchemiops' cell filter packs, stepped by the same
    kernel, and the atoms then ride the cell affinely (the module docstring).
    The kernel caps the two cell entries separately, so an isotropic or
    orthorhombic step is projected back onto its family afterwards.

    Parameters
    ----------
    dt, dtmax, dtmin
        Initial, largest and smallest timestep.
    Nmin
        Steps of positive power before the timestep may grow.
    finc, fdec
        The timestep's growth factor (after ``Nmin``) and shrink factor (when
        P turns negative).
    astart, fa
        The mixing parameter at a (re)start, and its decay factor.

    ``evaluator``, ``atoms``, ``relax_cell``, ``pressure_au``, ``logfile``,
    ``loginterval``, ``trajectory``, ``maxstep`` and ``check`` are as for
    :class:`FIRE2`.
    """

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        *,
        dt: float = 0.1,
        dtmax: float = 1.0,
        dtmin: float = 0.01,
        Nmin: int = 5,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.1,
        fa: float = 0.99,
        **kwargs: Any,
    ) -> None:
        import warp as wp
        from nvalchemiops.dynamics.optimizers import fire_step

        _check_fire(dt, dtmax, dtmin, Nmin, finc, fdec, astart, fa)
        super().__init__(evaluator, atoms, **kwargs)
        th, n = self.thermo, self.thermo.n_atoms
        m = n + (2 if self.relax_cell else 0)  # the cell filter's two entries
        self._kernel = fire_step
        self._vel = th.zeros(m, dtype=wp.vec3d)
        self._batch = th.zeros(m, dtype=wp.int32)
        self._unit_masses = wp.array(np.ones(m), dtype=wp.float64, device=th.device)
        self._positions, self._forces = th.positions, th.forces
        if self.relax_cell is not None:
            self._positions = th.zeros(m, dtype=wp.vec3d)  # the extended arrays
            self._forces = th.zeros(m, dtype=wp.vec3d)
            self._cell_w = th.zeros(1, dtype=wp.mat33d)
            self._cell_force_w = th.zeros(1, dtype=wp.mat33d)
            self._x_old = torch.empty_like(self._x)
        self._alpha, self._dt = th.scalar(astart), th.scalar(dt)
        self._limits = [th.scalar(v) for v in (astart, fa, dtmin, dtmax, self.maxstep)]
        self._n_positive = th.zeros(1, dtype=wp.int32)
        self._n_min = th.scalar(int(Nmin), dtype=wp.int32)
        self._factors = [th.scalar(fdec), th.scalar(finc)]
        self._uphill = th.zeros(1, dtype=wp.int32)
        self._reductions = [th.zeros(1) for _ in range(3)]

    def _step(self) -> None:
        self._kernel(
            self._positions,
            self._vel,
            self._forces,
            self._unit_masses,
            self._alpha,
            self._dt,
            *self._limits,
            self._n_positive,
            self._n_min,
            *self._factors,
            self._uphill,
            *self._reductions,
            batch_idx=self._batch,
        )

    def _move(self) -> None:
        if self.relax_cell is None:
            self._step()
            return
        from nvalchemiops.dynamics.utils import (
            pack_forces_with_cell,
            pack_positions_with_cell,
            unpack_positions_with_cell,
        )

        th, n = self.thermo, self.thermo.n_atoms
        h = np.array(self.evaluator.cell, dtype=np.float64)
        self._cell_w.assign(h.reshape(1, 3, 3))
        self._cell_force_w.assign(np.tril(self._cell_force(h)).reshape(1, 3, 3))
        pack_positions_with_cell(th.positions, self._cell_w, extended=self._positions)
        pack_forces_with_cell(th.forces, self._cell_force_w, extended=self._forces)
        self._x_old.copy_(self._x)
        self._step()
        unpack_positions_with_cell(
            self._positions, positions=th.positions, cell=self._cell_w, num_atoms=n
        )
        h_new = self._cell_w.numpy()[0]
        # The kernel caps each of the two cell entries on its own, which would
        # take a uniform scaling off its ray: project the step back onto it.
        if self.relax_cell == "isotropic":
            h_new = float(np.sum(h_new * h) / np.sum(h * h)) * h
        elif self.relax_cell == "orthorhombic":
            h_new = np.diag(np.diag(h_new))
        # The atoms ride the cell affinely, on top of their own step.
        affine = torch.as_tensor(np.linalg.solve(h, h_new - h), device=self._x.device)
        self._x.add_(self._x_old @ affine)
        th.set_cell(h_new)


class FIRE2(_Relaxation):
    """FIRE 2.0 (Guenole et al., Comput. Mater. Sci. 175, 109584 (2020)),
    through nvalchemiops' ``fire2_step`` and, with ``relax_cell``, its coupled
    ``fire2_step_coord_cell``.

    FIRE with the changes of FIRE 2.0, as nvalchemiops implements them: the
    kick comes before the power is measured (semi-implicit Euler); a step on
    which the power turns negative is undone by half a step back, with the
    velocity zeroed; and the whole step is scaled so that no atom moves more
    than ``maxstep``, the timestep with it. Unit masses. The defaults are
    nvalchemiops', more cautious than ASE's FIRE2; the parameters keep ASE's
    names (``Nmin`` is the kernel's ``delaystep``, ``finc`` its ``dtgrow``,
    ``fdec`` its ``dtshrink``, ``astart`` its ``alpha0``, ``fa`` its
    ``alphashrink``, and ``dtmax``/``dtmin`` its ``tmax``/``tmin``).

    With ``relax_cell`` the coupled kernel mixes the six cell parameters with
    the atoms, deforms the cell and moves the atoms with it and by their own
    step, capping that whole Cartesian displacement at ``maxstep``.

    Parameters
    ----------
    evaluator
        The :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator` the MD
        stages will share; the relaxation builds its own ``Thermo`` on it.
    atoms
        The system, in the evaluator's atom order and cell. Written back at
        the end: relaxed positions (and cell), zero momenta, energy and forces.
    dt, dtmax, dtmin
        Initial, largest and smallest timestep.
    Nmin
        Steps of positive power before the timestep may grow.
    finc, fdec
        The timestep's growth factor (after ``Nmin``) and shrink factor (when
        the power turns negative).
    astart, fa
        The mixing parameter at a (re)start, and its decay factor.
    relax_cell
        ``None``, ``"isotropic"``, ``"orthorhombic"`` or ``"full"``: what the
        cell may do (see the module docstring).
    pressure_au
        The target pressure of a cell relaxation, in eV/A^3.
    logfile, loginterval
        ``'-'`` (stdout), a path, or None; one row every ``loginterval`` steps.
    trajectory
        A ``.traj`` or extxyz path, or None: a frame (positions, energy,
        forces) at every log row and at the end -- the end of a divergence
        included, which is where a failed relaxation went.
    maxstep
        Largest displacement of any atom per step, in angstrom.
    check
        Refuse a bad relaxation (see the module docstring). Divergence is
        refused either way.
    """

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        *,
        dt: float = 0.01,
        dtmax: float = 0.08,
        dtmin: float = 0.005,
        Nmin: int = 60,
        finc: float = 1.05,
        fdec: float = 0.75,
        astart: float = 0.09,
        fa: float = 0.985,
        **kwargs: Any,
    ) -> None:
        import warp as wp
        from nvalchemiops.dynamics.optimizers import fire2_step

        _check_fire(dt, dtmax, dtmin, Nmin, finc, fdec, astart, fa)
        super().__init__(evaluator, atoms, **kwargs)
        th, n = self.thermo, self.thermo.n_atoms
        self._kernel = fire2_step
        self._vel = th.zeros(n, dtype=wp.vec3d)
        self._batch = th.zeros(n, dtype=wp.int32)
        self._alpha, self._dt = th.scalar(astart), th.scalar(dt)
        self._n_inc = th.zeros(1, dtype=wp.int32)
        self._scratch = [th.zeros(1) for _ in range(4)]
        self._parameters = {
            "delaystep": int(Nmin),
            "dtgrow": float(finc),
            "dtshrink": float(fdec),
            "alphashrink": float(fa),
            "alpha0": float(astart),
            "tmax": float(dtmax),
            "tmin": float(dtmin),
        }
        if self.relax_cell is not None:
            dev = self._x.device
            zeros = lambda *shape, dtype=torch.float64: torch.zeros(  # noqa: E731
                shape, dtype=dtype, device=dev
            )
            self._cell_t = zeros(1, 3, 3)
            self._cell_vel_t = zeros(1, 3, 3)
            self._cell_force_t = zeros(1, 3, 3)
            # The coupled kernel's torch views of the warp state above, and
            # the packed layout it would otherwise allocate at every call.
            self._cell_args = (
                wp.to_torch(self._vel),
                wp.to_torch(self._batch),
                wp.to_torch(self._alpha),
                wp.to_torch(self._dt),
                wp.to_torch(self._n_inc),
            )
            self._cell_kwargs = {
                "atom_ptr": torch.tensor([0, n], dtype=torch.int32, device=dev),
                "ext_atom_ptr": torch.tensor([0, n + 2], dtype=torch.int32, device=dev),
                "ext_batch_idx": zeros(n + 2, dtype=torch.int32),
                "ext_velocities": zeros(n + 2, 3),
                "ext_forces": zeros(n + 2, 3),
                "vf": wp.to_torch(self._scratch[0]),
                "v_sumsq": wp.to_torch(self._scratch[1]),
                "f_sumsq": wp.to_torch(self._scratch[2]),
                "max_norm": wp.to_torch(self._scratch[3]),
                # Not divided by the atom count: see the module docstring.
                "cell_force_scale": 1.0 / n,
            }

    def _step(self) -> None:
        self._kernel(
            self.thermo.positions,
            self._vel,
            self.thermo.forces,
            self._batch,
            self._alpha,
            self._dt,
            self._n_inc,
            *self._scratch,
            maxstep=self.maxstep,
            **self._parameters,
        )

    def _move(self) -> None:
        if self.relax_cell is None:
            self._step()
            return
        from nvalchemiops.torch import fire2_step_coord_cell

        h = np.array(self.evaluator.cell, dtype=np.float64)
        vel, batch, alpha, dt, n_inc = self._cell_args
        self._cell_t[0].copy_(torch.as_tensor(h, device=self._x.device))
        self._cell_force_t[0].copy_(
            torch.as_tensor(np.tril(self._cell_force(h)), device=self._x.device)
        )
        fire2_step_coord_cell(
            self._x,
            vel,
            self._f,
            self._cell_t,
            self._cell_vel_t,
            self._cell_force_t,
            batch,
            alpha,
            dt,
            n_inc,
            maxstep=self.maxstep,
            **self._parameters,
            **self._cell_kwargs,
        )
        self.thermo.set_cell(self._cell_t[0].cpu().numpy())


class LBFGS(_Relaxation):
    """Limited-memory BFGS (Nocedal 1980), through nvalchemiops'
    ``lbfgs_step_coord`` and, with ``relax_cell``, its ``lbfgs_step_coord_cell``.

    A quasi-Newton step built from the last ``memory`` position and force
    differences, shortened so that no atom moves more than ``maxstep``. There
    is no line search and no energy is read, so the energy may rise on a step
    while the force converges; a direction that stops descending, or a pair
    with too little curvature, restarts the history from steepest descent, and
    the first step after a (re)start moves the most-forced atom by exactly
    ``maxstep``. On water it needed about half of FIRE's force evaluations
    from an MD snapshot and somewhat more from a packed box (module
    docstring). The history starts afresh at every ``run``.

    ASE's names: ``memory`` (ASE's default is 100; nvalchemiops' 6 is the
    default here, the usual range being 3 to 7, as the history costs
    ``48 * memory`` bytes per atom) and ``maxstep`` (0.2 A, as in ASE). ASE's
    ``damping`` and ``alpha`` have no counterpart in the kernel.

    With ``relax_cell`` the kernel relaxes the atoms and the cell together in
    ASE's ``UnitCellFilter`` chart, the deformation from the cell at the start
    of ``run``, driven by the stress; ``maxstep`` caps the Cartesian motion of
    the atoms, their ride with the cell included.

    Parameters
    ----------
    memory
        Curvature pairs kept.
    maxstep
        Largest displacement of any atom per step, in angstrom.

    ``evaluator``, ``atoms``, ``relax_cell``, ``pressure_au``, ``logfile``,
    ``loginterval``, ``trajectory`` and ``check`` are as for :class:`FIRE2`.
    """

    _check_every = 1
    _cell_form = "upper"  # lattice vectors as columns, lower-triangular

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        *,
        memory: int = 6,
        maxstep: float = 0.2,
        **kwargs: Any,
    ) -> None:
        if int(memory) < 1:
            raise ValueError(f"memory must be at least 1 (got {memory})")
        super().__init__(evaluator, atoms, maxstep=maxstep, **kwargs)
        self.memory = int(memory)
        dev, n = self._x.device, self.thermo.n_atoms
        self._batch_t = torch.zeros(n, dtype=torch.int32, device=dev)
        self._state = None
        self._cell_state = None
        if self.relax_cell is not None:
            self._atom_ptr_t = torch.tensor([0, n], dtype=torch.int32, device=dev)
            self._cell_t = torch.zeros((1, 3, 3), dtype=torch.float64, device=dev)
            self._stress_t = torch.zeros_like(self._cell_t)

    def _reset(self) -> None:
        from nvalchemiops.torch.lbfgs import (
            lbfgs_prepare_cell_state,
            lbfgs_prepare_state,
        )

        super()._reset()  # the frame first: the chart's reference is the cell now
        dev, n = self._x.device, self.thermo.n_atoms
        cell = self.relax_cell is not None
        self._state = lbfgs_prepare_state(
            n + (2 if cell else 0),
            1,
            dtype=torch.float64,
            history_size=self.memory,
            device=dev,
        )
        if cell:
            h = np.array(self.evaluator.cell, dtype=np.float64)
            self._cell_t[0].copy_(torch.as_tensor(h.T.copy(), device=dev))
            kappa = abs(np.linalg.det(h)) ** (1.0 / 3.0)
            self._cell_state = lbfgs_prepare_cell_state(
                self._atom_ptr_t,
                self._cell_t,
                cell_force_scale=kappa / n,
                dtype=torch.float64,
                device=dev,
            )

    def _stress(self, h: np.ndarray) -> np.ndarray:
        """The stress that drives the kernel's cell: tensile-positive, the
        target pressure folded in, along the mode's free directions."""
        s = -self._enthalpy_virial(h) / abs(np.linalg.det(h))
        if self.relax_cell == "isotropic":
            return np.trace(s) / 3.0 * np.eye(3)
        if self.relax_cell == "orthorhombic":
            return np.diag(np.diag(s))
        return s

    def _move(self) -> None:
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord, lbfgs_step_coord_cell

        if self.relax_cell is None:
            lbfgs_step_coord(
                self._x, self._f, self._state, self._batch_t, maxstep=self.maxstep
            )
            return
        h = np.array(self.evaluator.cell, dtype=np.float64)
        self._stress_t[0].copy_(torch.as_tensor(self._stress(h), device=self._x.device))
        lbfgs_step_coord_cell(
            self._x,
            self._cell_t,
            self._f,
            self._stress_t,
            self._state,
            self._cell_state,
            self._batch_t,
            maxstep=self.maxstep,
        )
        self.thermo.set_cell(self._cell_t[0].T.cpu().numpy())
