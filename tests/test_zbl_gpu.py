"""CUDA compression and partition ownership retain the fixed ZBL contribution."""

import copy

import numpy as np
import pytest
import torch
from deepmd.kernels.cuda.dpa4c.graph_compress import op_available
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.compress import compress_model
from deepmd_lr.nvalchemi import DistributedEvaluator
from tests.test_joint_model import LONG_RANGE
from tests.test_zbl import config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("lr,level", [(False, 1), (False, 2), (True, 1)])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("clamp", [False, True])
def test_serial_compression_and_partition_parity(
    monkeypatch, lr, compressed, level, clamp
):
    monkeypatch.setenv("DP_CUDA_INFER", str(level))
    if compressed and not op_available():
        pytest.skip("compiled DPA4C compression operator required")
    if clamp and level == 2:
        pytest.skip("level 2 bypasses the descriptor; refused, tested below")
    cfg = config(lr)
    cfg["zbl"]["clamp"] = clamp
    cfg["descriptor"]["precision"] = "float32"
    cfg["fitting_net"]["precision"] = "float32"
    if lr:
        cfg["long_range"] = copy.deepcopy(LONG_RANGE)
        cfg["long_range"]["charge_net"]["precision"] = "float32"
    base = get_model(cfg).eval()
    model = get_model(cfg).eval()
    model.load_state_dict(base.state_dict(), strict=True)
    xyz = np.array([[0.2, 1, 1], [1.3, 1, 1], [6.5, 1, 1], [7.6, 1, 1]])
    types = np.array([0, 1, 0, 1], dtype=np.int64)
    box = np.eye(3) * 14
    x = torch.tensor(xyz, dtype=torch.float64, device=DEVICE)[None]
    t = torch.tensor(types, device=DEVICE)[None]
    b = torch.tensor(box, dtype=torch.float64, device=DEVICE)[None]
    reference = base(x, t, box=b)
    if compressed:
        compress_model(model, True)
    actual = model(x, t, box=b)
    for key in ["energy", "force", "virial"]:
        torch.testing.assert_close(actual[key], reference[key], atol=2e-4, rtol=2e-5)
    # Repeat one physical device to exercise ownership/halos, not multi-GPU speed.
    evaluator = DistributedEvaluator(
        model,
        types,
        box,
        devices=[DEVICE, DEVICE],
        compute_virial=True,
        memory_headroom=None,
    )
    evaluator(xyz)
    partitioned = evaluator.results
    for key in ["energy", "force", "virial"]:
        target = reference[key].detach().cpu().numpy().reshape(-1)
        np.testing.assert_allclose(
            np.asarray(partitioned[key]).reshape(-1), target, atol=2e-4, rtol=2e-5
        )


def test_level_2_is_refused_with_the_clamp(monkeypatch):
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    cfg = config(False)
    cfg["zbl"]["clamp"] = True
    model = get_model(cfg).eval()
    x = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]], dtype=torch.float64, device=DEVICE
    )
    t = torch.tensor([[0, 1]], device=DEVICE)
    with pytest.raises(RuntimeError, match="DP_CUDA_INFER=2"):
        model(x, t, box=torch.eye(3, dtype=torch.float64, device=DEVICE)[None] * 14)
