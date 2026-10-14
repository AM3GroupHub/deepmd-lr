"""The Born-charge calibration script recovers a known scale and sign."""

import numpy as np
import pytest
import torch

from scripts.calibrate_born import (
    born_statistics,
    calibrate,
    evaluate_frames,
    fit_scale,
    read_frames,
)

from .test_ase_calculator import water_atoms
from .test_joint_model import build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)


def _calc(model=None):
    from deepmd_lr.ase import LongRangeCalculator

    return LongRangeCalculator(model or build(), efield=[0.0, 0.0, 0.0])


def _labelled(calc, seeds, factor):
    """Frames whose 'BEC' labels are the model's own raw tensors times ``factor``."""
    frames = []
    for seed in seeds:
        atoms = water_atoms(seed=seed)
        probe = atoms.copy()
        probe.calc = calc
        probe.get_forces()
        z = np.asarray(calc.results["born_charge"]).reshape(len(atoms), 3, 3)
        atoms.arrays["BEC"] = (factor * z).reshape(len(atoms), 9)
        frames.append(atoms)
    return frames


def test_fit_scale_is_the_least_squares_factor():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 6, 3, 3))
    assert fit_scale(z, -1.7 * z) == pytest.approx(-1.7)
    # exact least squares, not a ratio of means
    noise = rng.normal(size=z.shape)
    expected = (z * (2.0 * z + noise)).sum() / (z * z).sum()
    assert fit_scale(z, 2.0 * z + noise) == pytest.approx(expected)
    with pytest.raises(ValueError, match="identically zero"):
        fit_scale(np.zeros_like(z), z)


@pytest.mark.parametrize("factor", [1.75, -1.3])
def test_calibration_recovers_scale_and_sign_and_scores_the_held_out_set(factor):
    calc = _calc()
    fit_frames = _labelled(calc, (0, 1, 2), factor)
    test_frames = _labelled(calc, (3, 4), factor)

    summary, data = calibrate(calc, fit_frames, test_frames)

    assert summary["scale"] == pytest.approx(factor, rel=1e-6)
    assert summary["sqrt_eps_inf"] == pytest.approx(abs(factor), rel=1e-6)
    assert summary["eps_inf"] == pytest.approx(factor * factor, rel=1e-6)
    assert summary["charge_sign"] == np.sign(factor)
    for name in ("fit", "test"):
        assert summary[name]["born"]["rmse_e"] < 1e-6
        assert set(summary[name]["born"]["isotropic"]) == {"O", "H"}
        np.testing.assert_allclose(
            data[name]["bec_pred"], data[name]["bec_ref"], atol=1e-6
        )
    # a given scale is applied as it is, so the errors then measure the mismatch
    given, _ = calibrate(calc, None, test_frames, scale=2.0 * factor)
    assert given["scale_source"] == "given"
    assert given["test"]["born"]["rmse_e"] > 1e-3


def test_born_statistics_split_diagonal_from_off_diagonal():
    symbols = np.array(["O", "H", "H"])
    z_ref = np.zeros((2, 3, 3, 3))
    z = z_ref.copy()
    z[..., 0, 0] = 0.3  # a diagonal error only
    s = born_statistics(z, z_ref, symbols)
    assert s["offdiag_mae_e"] == 0.0
    assert s["diag_mae_e"] == pytest.approx(0.1)
    assert s["isotropic"]["O"]["model"] == pytest.approx(0.1)
    assert s["isotropic"]["H"]["n_atoms"] == 2


def test_frames_without_labels_or_models_without_charges_are_refused(tmp_path):
    from ase.io import write

    plain = water_atoms(seed=0)
    write(tmp_path / "plain.xyz", plain)
    with pytest.raises(ValueError, match="no per-atom array 'BEC'"):
        read_frames(tmp_path / "plain.xyz")

    calc = _calc()
    frames = _labelled(calc, (0,), 1.0)
    write(tmp_path / "labelled.xyz", frames)
    assert len(read_frames(tmp_path / "labelled.xyz")) == 1

    # the calculator itself refuses a field on a model without charges
    with pytest.raises(ValueError, match="long_range"):
        evaluate_frames(_calc(build(long_range=False)), frames)
    from deepmd_lr.ase import LongRangeCalculator

    with pytest.raises(ValueError, match="efield"):
        evaluate_frames(LongRangeCalculator(build()), frames)
