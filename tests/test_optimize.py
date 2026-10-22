"""Relaxation contracts on a small analytic CPU evaluator, without a model or a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from scipy.spatial.transform import Rotation

from deepmd_lr.nvalchemi import (
    FIRE,
    FIRE2,
    LBFGS,
    DistributedEvaluator,
    RelaxationError,
    Thermo,
)


class CellSpring:
    """E = k/2 |H H^T - G_ref|²; zero atomic forces at fixed cell."""

    clear_results = DistributedEvaluator.clear_results

    def __init__(self, atoms, *, fail_at=0, failure="raise"):
        self.cell = atoms.cell.array.copy()
        self.reference_metric = 0.9 * self.cell @ self.cell.T
        self.n_atoms = len(atoms)
        self.atype = np.zeros(self.n_atoms, dtype=int)
        self.model = SimpleNamespace(get_type_map=lambda: ["Ar"])
        self.home = "cpu"
        self.pbc = (True, True, True)
        self.compute_virial = True
        self.efield = None
        self.cell_version = self.calls = 0
        self.forces = torch.zeros((self.n_atoms, 3), dtype=torch.float64)
        self.virial = np.zeros((3, 3))
        self.results = {}
        self.fail_at, self.failure = fail_at, failure

    @property
    def volume(self):
        return abs(np.linalg.det(self.cell))

    def set_cell(self, cell):
        self.cell = np.array(cell, copy=True)
        self.cell_version += 1

    def __call__(self, positions):
        self.calls += 1
        if self.calls == self.fail_at and self.failure == "raise":
            raise RuntimeError("injected evaluation failure")
        d = self.cell @ self.cell.T - self.reference_metric
        k = 1e-4  # eV / A^4
        energy = 0.5 * k * np.sum(d * d)
        self.forces.zero_()
        self.virial = -2 * k * self.cell.T @ d @ self.cell
        if self.calls == self.fail_at:
            if self.failure == "virial":
                self.virial[0, 0] = np.nan
            elif self.failure == "force":
                self.forces[0, 0] = np.nan
            elif self.failure == "energy":
                energy = np.nan
        self.results = {"energy": energy, "force": self.forces}
        return float(energy), self.forces


class EvaluatorWrapper:
    """Read-only measurements and delegated buffers, like the GPU test wrappers."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __getattr__(self, name):
        return getattr(self.evaluator, name)

    def __call__(self, positions):
        return self.evaluator(positions)

    @property
    def virial(self):
        return self.evaluator.virial


def spring(*, rotated=False, **kwargs):
    cell = np.array([[5.0, 0.0, 0.0], [0.6, 6.0, 0.0], [-0.4, 0.7, 7.0]])
    if rotated:
        cell = cell @ Rotation.from_rotvec([0.3, -0.5, 0.8]).as_matrix()
    atoms = Atoms("Ar2", scaled_positions=[[0.2] * 3, [0.7] * 3], cell=cell, pbc=True)
    return atoms, CellSpring(atoms, **kwargs)


@pytest.mark.parametrize("relax", [FIRE, FIRE2, LBFGS])
@pytest.mark.parametrize("at", [1, 2, 3], ids=["first", "step", "restore"])
@pytest.mark.parametrize("failure", ["energy", "force", "virial"])
def test_nonfinite_evaluation_is_rejected(relax, at, failure):
    atoms, ev = spring(rotated=True, fail_at=at, failure=failure)
    orientation = atoms.cell.standard_form()[1]
    opt = relax(ev, atoms, relax_cell="full", logfile=None, check=False)
    with pytest.raises(RelaxationError, match="non-finite"):
        opt.run(steps=1, smax=1e-12)
    assert ev.calls == at
    assert atoms.calc is None and not opt.converged
    assert not ev.results and not opt.thermo.fresh
    np.testing.assert_allclose(atoms.cell.standard_form()[1], orientation, atol=1e-12)
    np.testing.assert_allclose(atoms.cell.array, ev.cell)
    np.testing.assert_allclose(atoms.positions, opt.thermo.positions.numpy())


@pytest.mark.parametrize("relax", [FIRE, FIRE2, LBFGS])
@pytest.mark.parametrize("wrapped", [False, True])
def test_failed_evaluation_discards_shared_results(relax, wrapped):
    atoms, ev = spring(fail_at=2)
    opt = relax(
        EvaluatorWrapper(ev) if wrapped else ev,
        atoms,
        relax_cell="full",
        logfile=None,
        check=False,
    )
    with pytest.raises(RuntimeError, match="injected"):
        opt.run(steps=1, smax=1e-12)
    assert atoms.calc is None and not ev.results
    assert not opt.thermo.fresh and np.isnan(opt.thermo.energy)
    ev.fail_at = 0
    opt.run(steps=1, smax=1e-12)
    assert atoms.calc is not None and ev.results and opt.thermo.fresh


@pytest.mark.parametrize("relax", [FIRE, FIRE2, LBFGS])
def test_closing_irun_checks_the_restored_evaluation(relax):
    atoms, ev = spring(rotated=True, fail_at=3, failure="force")
    opt = relax(ev, atoms, relax_cell="full", logfile=None, check=False)
    run = opt.irun(steps=2, smax=1e-12)
    next(run)
    with pytest.raises(RelaxationError, match="non-finite"):
        run.close()
    assert atoms.calc is None and not ev.results


def test_clearing_results_invalidates_every_thermo_using_the_buffer():
    atoms, ev = spring()
    th = Thermo.from_atoms(ev, atoms)
    th.evaluate()
    assert th.fresh
    pointer = ev.forces.data_ptr()
    ev.clear_results()
    assert not th.fresh and not ev.results and ev.virial is None
    assert ev.forces.data_ptr() == pointer and torch.isnan(ev.forces).all()
    th.evaluate()
    assert th.fresh and torch.isfinite(ev.forces).all()


@pytest.mark.parametrize("relax", [FIRE, FIRE2, LBFGS])
@pytest.mark.parametrize("mode", ["isotropic", "orthorhombic", "full"])
def test_cell_spring_converges_with_the_requested_shape(relax, mode):
    atoms, ev = spring(rotated=mode != "orthorhombic")
    if mode == "orthorhombic":
        atoms.set_cell(np.diag([5.0, 6.0, 7.0]), scale_atoms=True)
        ev = CellSpring(atoms)
    start = atoms.cell.array.copy()
    fractional = atoms.get_scaled_positions(wrap=False)
    opt = relax(ev, atoms, relax_cell=mode, logfile=None, check=False)
    assert opt.run(steps=1500, smax=1e-8)
    np.testing.assert_allclose(atoms.cell.array, np.sqrt(0.9) * start, atol=1e-3)
    np.testing.assert_allclose(
        atoms.positions, fractional @ atoms.cell.array, atol=1e-10
    )
    assert opt.thermo.fresh and atoms.calc is not None
