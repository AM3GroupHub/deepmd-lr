"""Deployment and zero-distance regressions from the multi-head review."""

import copy

import numpy as np
import pytest
import torch
from deepmd.pt_expt.train.wrapper import ModelWrapper

from deepmd_lr.ase import LongRangeCalculator, load_model
from deepmd_lr.lr.term import IsolatedLongRangeEnergy

from .test_ase_calculator import water_atoms
from .test_joint_model import build
from .test_multi_task import TestThreeBranchInterfaceBulkMolecule as ThreeBranch


@pytest.mark.parametrize("name", ["interface", "bulk", "molecule"])
def test_checkpoint_head_to_ase(tmp_path, name):
    fixture = ThreeBranch()
    cfg, _ = fixture.config()
    models = fixture.build()
    cfg["model"]["model_dict"][name]["model_branch_alias"] = ["selected"]
    state = ModelWrapper(models, model_params=cfg["model"]).state_dict()
    path = str(tmp_path / "multi.pt")
    torch.save({"model": state}, path)
    atoms = water_atoms(n_mol=2)
    if name == "interface":
        atoms.set_cell([10.0, 10.0, 30.0])  # a slab needs vacuum over it
    atoms.pbc = {"interface": [True, True, False], "bulk": True, "molecule": False}[
        name
    ]
    atoms.calc = LongRangeCalculator(models[name])
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = LongRangeCalculator(path, head="selected")
    assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-8)
    np.testing.assert_allclose(atoms.get_forces(), forces, atol=1e-8)
    with pytest.raises(ValueError, match="Select a checkpoint head"):
        load_model(path)
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        load_model(path, head="absent")
    broken = copy.deepcopy(state)
    key = next(k for k in broken if k.startswith(f"model.{name}.charge_fitting."))
    del broken[key]
    torch.save({"model": broken}, path)
    with pytest.raises(ValueError, match="weights do not match"):
        load_model(path, head=name)


@pytest.mark.parametrize(
    "name,pbc,message",
    [
        ("bulk", [True, False, False], "2 axes open"),
        ("molecule", [True, True, True], "does not match"),
    ],
)
def test_reject_boundary_mismatch(name, pbc, message):
    """What no periodic sum can take: a wire, and the isolated term in a cell."""
    model = ThreeBranch().build()[name]
    atoms = water_atoms(n_mol=1)
    atoms.pbc = pbc
    atoms.calc = LongRangeCalculator(model)
    with pytest.raises(ValueError, match=message):
        atoms.get_potential_energy()


@pytest.mark.parametrize("pbc", [[True, True, True], [False, True, True]])
def test_a_slab_branch_follows_the_frames_pbc(pbc):
    """The term holds no weights, so the frame's pbc picks the sum: the
    interface branch runs a bulk cell, or a slab open along x."""
    model = ThreeBranch().build()["interface"]
    atoms = water_atoms(n_mol=1)
    if not all(pbc):
        axis = pbc.index(False)
        cell = atoms.cell.array.copy()
        cell[axis, axis] += 20.0  # vacuum along the open axis
        atoms.set_cell(cell)
    atoms.pbc = pbc
    atoms.calc = LongRangeCalculator(model)
    assert np.isfinite(atoms.get_potential_energy())
    term = model.long_range_energy
    assert term.slab_correction is (not all(pbc))
    assert list(term.pbc) == pbc


def test_short_range_partial_pbc_is_refused():
    atoms = water_atoms(n_mol=1)
    atoms.pbc = [True, False, False]
    atoms.calc = LongRangeCalculator(build(long_range=False))
    with pytest.raises(ValueError, match="Partial pbc"):
        atoms.get_potential_energy()


@pytest.mark.parametrize("padded", [False, True])
def test_coincident_pair_derivatives(padded):
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    q = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, not padded]])
    term = IsolatedLongRangeEnergy()

    def fn(r, charges):
        return term(r, charges, mask=mask).sum()

    assert torch.autograd.gradcheck(fn, (coord, q))
    assert torch.autograd.gradgradcheck(fn, (coord, q))
    if padded:
        trimmed = term(coord[:, :2], q[:, :2]).sum()
        torch.testing.assert_close(fn(coord, q), trimmed)
        grad = torch.autograd.grad(fn(coord, q), coord)[0]
        torch.testing.assert_close(grad[:, 2], torch.zeros_like(grad[:, 2]))


def test_coincident_pair_has_analytic_limit():
    from deepmd.dpmodel.modifier.dipole_charge import ELECTROSTATIC_CONVERSION

    coord = torch.zeros((1, 2, 3), dtype=torch.float64)
    q = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    alpha = 0.4
    energy = IsolatedLongRangeEnergy(alpha=alpha)(coord, q).sum()
    assert float(energy) == pytest.approx(
        -2 * alpha / np.sqrt(np.pi) * ELECTROSTATIC_CONVERSION
    )
