"""The Born-charge calibration recovers a planted scale and sign."""

import numpy as np
import pytest
import torch

from scripts.calibrate_born import born_tensors, errors, fit_scale

from .test_ase_calculator import water_atoms
from .test_joint_model import build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)


def test_fit_scale_is_the_least_squares_factor():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 6, 3, 3))
    noise = rng.normal(size=z.shape)
    assert fit_scale(z, -1.7 * z) == pytest.approx(-1.7)
    expected = (z * (2.0 * z + noise)).sum() / (z * z).sum()
    assert fit_scale(z, 2.0 * z + noise) == pytest.approx(expected)


@pytest.mark.parametrize("factor", [1.75, -1.3])
def test_planted_scale_and_sign_are_recovered(factor):
    from deepmd_lr.ase import LongRangeCalculator

    calc = LongRangeCalculator(build(), efield=[0.0, 0.0, 0.0])
    frames = [water_atoms(seed=seed) for seed in range(3)]
    for atoms in frames:
        probe = atoms.copy()
        probe.calc = calc
        probe.get_forces()
        z = np.asarray(calc.results["born_charge"]).reshape(len(atoms), 9)
        atoms.arrays["BEC"] = factor * z

    z_raw, z_ref = born_tensors(calc, frames)
    s = fit_scale(z_raw, z_ref)
    assert s == pytest.approx(factor, rel=1e-6)
    mae, rmse = errors(s * z_raw, z_ref)
    assert rmse < 1e-6 and mae < 1e-6
