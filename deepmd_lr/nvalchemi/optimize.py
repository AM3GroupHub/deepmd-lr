# SPDX-License-Identifier: LGPL-3.0-or-later
"""Position relaxation on the GPU (FIRE), on the evaluator the MD shares.

.. code-block:: python

    ev = build_evaluator("model.ckpt.pt", atoms, devices=["cuda:0"])
    FIRE(ev, atoms).run(fmax=0.05, steps=2000)    # RelaxationError if it went wrong
    maxwell_boltzmann(atoms, temperature_K=300, seed=7)
    Langevin(ev, atoms, 0.5 * units.fs, temperature_K=300,
             friction=0.01 / units.fs).run(10_000)  # same evaluator, no second copy

FIRE moves positions only; the cell never changes. A freshly packed box can hold
tens of eV of clashes, and MD would release that as heat in the first few
hundred fs, pushing the model into regions it never saw -- so relax first.

**Divergence is always fatal.** A model without a short-range repulsive wall can
let a minimiser collapse atoms onto each other from a badly distorted start.
Forces growing tenfold, more than 2 eV/atom released, or anything non-finite
stops the run with a :class:`RelaxationError` naming the cause. Pre-relax such a
box with a classical force field instead.

With ``check=True`` (the default) it also refuses a relaxation that did not
converge, one that changed the covalent-bond topology between non-metal atoms,
and one whose energy rose. The energy check applies only without a field:
under a field FIRE seeks zero total force, not minimum energy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from deepmd_lr.nvalchemi.md import (
    _check_atoms,
    _check_evaluator,
    _Logger,
    _sync_atoms,
    _Trajectory,
)
from deepmd_lr.nvalchemi.thermo import Thermo

__all__ = ["FIRE", "RelaxationError"]

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


class FIRE:
    """FIRE2 on the positions, with the checks above.

    Parameters
    ----------
    evaluator
        The :class:`~deepmd_lr.nvalchemi.evaluator.DistributedEvaluator` the MD
        stages will share; FIRE builds its own ``Thermo`` on it.
    atoms
        The system, in the evaluator's atom order and cell. Written back at
        the end: relaxed positions, zero momenta, energy and forces.
    logfile, loginterval
        ``'-'`` (stdout), a path, or None; one row every ``loginterval`` steps.
    trajectory
        A ``.traj`` or extxyz path, or None: a frame (positions, energy,
        forces) at every log row and at the end -- the end of a divergence
        included, which is where a failed relaxation went.
    maxstep
        Largest displacement per step, in angstrom.
    check
        Refuse a bad relaxation (see the module docstring). Divergence is
        refused either way.
    """

    def __init__(
        self,
        evaluator: Any,
        atoms: Atoms,
        *,
        logfile: Any = "-",
        loginterval: int = 50,
        trajectory: Any = None,
        maxstep: float = 0.1,
        check: bool = True,
    ) -> None:
        import warp as wp

        _check_evaluator(evaluator)
        _check_atoms(evaluator, atoms)
        self.atoms = atoms
        self.thermo = Thermo(
            evaluator, atoms.get_positions(), masses=atoms.get_masses()
        )
        self.maxstep = float(maxstep)
        self.check = bool(check)
        self.loginterval = max(1, int(loginterval))
        self.nsteps = 0
        self.converged = False
        th, n = self.thermo, self.thermo.n_atoms
        self._vel = wp.zeros(n, dtype=wp.vec3d, device=th.device)
        self._batch = wp.zeros(n, dtype=wp.int32, device=th.device)
        self._alpha = th.scalar(0.09)
        self._fire_dt = th.scalar(0.01)
        self._n_inc = th.zeros(1, dtype=wp.int32)
        self._scratch = [th.zeros(1) for _ in range(4)]
        self._observers: list = []
        self._last_recorded: int | None = None
        self._traj = None if trajectory is None else _Trajectory(trajectory, False)
        self._logger = (
            None
            if logfile is None
            else _Logger(
                logfile,
                [("step", 10, "d"), ("Epot[eV]", 18, ".6f"), ("fmax[eV/A]", 14, ".5f")],
                append=False,
            )
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
        """The largest force on any atom, eV/A (a host read)."""
        return float(np.linalg.norm(self.thermo.forces.numpy(), axis=1).max())

    def run(self, fmax: float = 0.05, steps: int = 100_000_000) -> bool:
        """Relax until the largest force is below ``fmax`` (eV/A) or
        ``steps`` have run. Returns whether it converged; raises
        :class:`RelaxationError` on divergence, and with ``check`` on the
        other failures."""
        for _ in self.irun(fmax, steps):
            pass
        return self.converged

    def irun(self, fmax: float = 0.05, steps: int = 100_000_000) -> Iterator[int]:
        """Generator form of :meth:`run`; yields the step count."""
        from nvalchemiops.dynamics.optimizers import fire2_step

        th = self.thermo
        th.set_state(positions=self.atoms.get_positions())
        th.evaluate()
        n = th.n_atoms
        e0, f0 = th.energy, self.fmax()
        bonds0 = _covalent_bonds(self.atoms) if self.check else None
        self._record(f0)
        diverged = None
        self.converged = f0 < fmax
        try:
            for _ in range(int(steps)):
                if self.converged:
                    break
                fire2_step(
                    th.positions,
                    self._vel,
                    th.forces,
                    self._batch,
                    self._alpha,
                    self._fire_dt,
                    self._n_inc,
                    *self._scratch,
                    maxstep=self.maxstep,
                )
                th.evaluate()
                self.nsteps += 1
                if not math.isfinite(th.energy):
                    diverged = "non-finite energy"
                    break
                if self.nsteps % 10 == 0 or self.nsteps % self.loginterval == 0:
                    f = self.fmax()
                    if (
                        not math.isfinite(f)
                        or f > max(10.0 * f0, 50.0)
                        or (e0 - th.energy) / n > 2.0
                    ):
                        diverged = (
                            f"max|F| reached {f:.3g} eV/A and the energy "
                            f"fell {(e0 - th.energy) / n:.3g} eV/atom"
                        )
                        break
                    self.converged = f < fmax
                    if self.nsteps % self.loginterval == 0:
                        self._record(f)
                for fn, interval, args, kwargs in self._observers:
                    if self.nsteps % interval == 0:
                        fn(*args, **kwargs)
                yield self.nsteps
        finally:
            if math.isfinite(th.energy):
                _sync_atoms(self.atoms, th)
        f1 = self.fmax()
        if self._last_recorded != self.nsteps:
            self._record(f1)
        if diverged:
            raise RelaxationError(
                f"relaxation diverged after {self.nsteps} steps ({diverged}): the "
                "model lacks short-range repulsion somewhere near this structure "
                "and the minimiser fell in. Pre-relax the box with a classical "
                "force field, then start from that structure"
            )
        if not self.check:
            return
        problems = []
        if not self.converged:
            problems.append(f"not converged (max|F| {f1:.3f} >= {fmax} eV/A)")
        if self.evaluator.efield is None and th.energy > e0:
            problems.append(f"energy rose ({e0:.3f} -> {th.energy:.3f} eV)")
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
        if self._logger is not None:
            self._logger.write(
                {"step": self.nsteps, "Epot[eV]": th.energy, "fmax[eV/A]": fmax}
            )
        if self._traj is not None:
            frame = Atoms(
                self.atoms.get_chemical_symbols(),
                positions=th.positions.numpy(),
                cell=self.atoms.cell,
                pbc=True,
            )
            frame.calc = SinglePointCalculator(
                frame, energy=th.energy, forces=th.forces.numpy().copy()
            )
            frame.info["step"] = self.nsteps
            self._traj.write(frame)

    def close(self) -> None:
        """Close the log and trajectory files."""
        if self._logger is not None:
            self._logger.close()
        if self._traj is not None:
            self._traj.close()
