"""Charge-state, exclusion and energy-accounting regression tests."""

import copy

import numpy as np
import pytest
import torch

from deepmd_lr.nvalchemi import DistributedEvaluator, evaluate_partitioned, to_numpy

from .test_joint_model import DESCRIPTOR, FITTING, LONG_RANGE, frame, get_model


def conditioned_model(excluded=False):
    descriptor = copy.deepcopy(DESCRIPTOR)
    descriptor.update(add_chg_spin_ebd=True, default_chg_spin=[1.0, 2.0])
    long_range = copy.deepcopy(LONG_RANGE)
    long_range["charge_net"]["exclude_types"] = [1] if excluded else []
    return get_model(
        dict(
            type="dp_lr",
            type_map=["O", "H"],
            descriptor=descriptor,
            fitting_net=copy.deepcopy(FITTING),
            long_range=long_range,
        )
    ).eval()


@pytest.mark.parametrize("batched", [False, True])
def test_default_charge_matches_explicit_condition(batched):
    model = conditioned_model()
    coord, atype, cell = frame()
    if batched:
        coord, atype, cell = (
            x.repeat(2, *([1] * (x.ndim - 1))) for x in (coord, atype, cell)
        )
    implicit = model(coord, atype, box=cell)
    explicit = model(coord, atype, box=cell, charge_spin=coord.new_tensor([1.0, 2.0]))
    torch.testing.assert_close(
        implicit["charge"].sum(dim=1), torch.ones_like(implicit["charge"].sum(dim=1))
    )
    for key in ("charge", "energy", "force", "virial"):
        torch.testing.assert_close(implicit[key], explicit[key])


@pytest.mark.parametrize("excluded", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_partitioned_charge_contracts(excluded, explicit):
    model = conditioned_model(excluded)
    coord, atype, cell = frame()
    condition = (-1.0, 3.0) if explicit else None
    serial = model(coord, atype, box=cell, charge_spin=condition)
    positions = coord[0].cpu().numpy()
    types = atype[0].cpu().numpy()
    box = cell[0].cpu().numpy()
    result = to_numpy(evaluate_partitioned(
        model,
        positions,
        types,
        box,
        n_partitions=2,
        devices=[coord.device] * 2,
        compute_virial=True,
        charge_spin=condition,
    ))
    assert result["charge"].sum() == pytest.approx(-1.0 if explicit else 1.0, abs=1e-12)
    if excluded:
        np.testing.assert_array_equal(result["charge"][types == 1], 0.0)
        torch.testing.assert_close(
            serial["charge"].reshape(-1)[atype.reshape(-1) == 1], coord.new_zeros(8)
        )
    for key in ("charge", "energy", "force", "virial", "atom_energy"):
        reference = serial[key].detach().cpu().numpy().reshape(np.shape(result[key]))
        np.testing.assert_allclose(result[key], reference, atol=1e-8, rtol=1e-7)
    assert result["atom_energy"].sum() == pytest.approx(result["energy"], abs=1e-10)
    np.testing.assert_allclose(
        result["atom_energy_short_range"] + result["atom_energy_long_range"],
        result["atom_energy"],
        atol=1e-12,
    )
    evaluator = DistributedEvaluator(
        model,
        types,
        box,
        devices=[coord.device],
        charge_spin=condition,
    )
    energy, force = evaluator(coord[0])
    assert energy == pytest.approx(result["energy"], abs=1e-8)
    np.testing.assert_allclose(force.cpu().numpy(), result["force"], atol=1e-8)
