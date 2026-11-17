"""The short-contact term on the GPU: the kernel's list of contacts is the
torch arithmetic, and compressed and partitioned evaluation keep the term."""

import copy

import numpy as np
import pytest
import torch
from deepmd.kernels.cuda.dpa4c.graph_compress import op_available
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.compress import compress_model
from deepmd_lr.nvalchemi import DistributedEvaluator
from deepmd_lr.zbl import ZBL, add_envelope_gradient

from .test_joint_model import LONG_RANGE
from .test_zbl import BLOCK, config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def random_graph(dtype, n_node: int = 12, n_edge: int = 600, seed: int = 0):
    """Edges of every length around the windows, with what a real graph holds
    besides: masked zero-length edges, a padded node, ghost destinations."""
    rng = torch.Generator().manual_seed(seed)
    atype = torch.randint(0, 2, (n_node,), generator=rng)
    atype[-1] = -1
    edges = torch.randint(0, n_node, (2, n_edge), generator=rng)
    direction = torch.nn.functional.normalize(
        torch.randn(n_edge, 3, generator=rng, dtype=torch.float64), dim=1
    )
    length = 0.2 + 1.2 * torch.rand(n_edge, generator=rng, dtype=torch.float64)
    mask = torch.rand(n_edge, generator=rng) > 0.1
    vec = (direction * length[:, None]).to(dtype)
    vec[~mask] = 0.0
    owned = torch.arange(n_node) < n_node - 3
    return (t.to(DEVICE) for t in (atype, edges, vec, mask, owned))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_contact_list_is_the_torch_arithmetic(dtype):
    """What inference runs on: one kernel pass, the per-node ZBL energy and
    envelope, and per edge in contact the two derivatives."""
    zbl = ZBL(["O", "H"], BLOCK, device=DEVICE).eval()
    atype, edges, vec, mask, owned = random_graph(dtype, n_node=40, n_edge=120, seed=1)
    assert zbl.explicit(vec) and not zbl.train().explicit(vec)
    zbl.eval()
    found = zbl.contacts(atype, edges, vec, mask, owned)
    src, dst = edges[0], edges[1]
    real = (atype >= 0) & (atype < 2)
    valid = mask & real[src] & real[dst]
    e_edge, g_ref = zbl._edge_terms_torch(atype, src, dst, vec, valid & owned[dst])
    e_ref = torch.zeros_like(found.energy).index_add(0, dst, e_edge)
    leaf = vec.clone().requires_grad_(True)
    w_ref = zbl._envelope_torch(atype, src, dst, leaf, valid).double()
    cotangent = torch.randn(
        40, generator=torch.Generator().manual_seed(2), dtype=torch.float64
    ).to(DEVICE)
    (dw_ref,) = torch.autograd.grad((w_ref * cotangent).sum(), leaf)
    tight = 1e-12 if dtype == torch.float64 else 2e-5
    torch.testing.assert_close(found.energy, e_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(found.envelope, w_ref.detach(), rtol=tight, atol=tight)
    # The list holds every edge the dense arithmetic has something on.
    dense = torch.zeros(vec.shape, dtype=torch.float64, device=DEVICE)
    g = dense.clone().index_add_(0, found.edge, found.g)
    torch.testing.assert_close(g, g_ref.double(), rtol=10 * tight, atol=10 * tight)
    dw = add_envelope_gradient(dense.clone(), found, cotangent)
    assert dw_ref.abs().max() > 1.0
    torch.testing.assert_close(dw, dw_ref.double(), rtol=10 * tight, atol=10 * tight)
    torch.testing.assert_close(
        found.vec[found.g.abs().sum(1) > 0].norm(dim=1).max(),
        vec.double().norm(dim=1)[g_ref.abs().sum(1) > 0].max(),
    )
    # A list too short for the contacts says so in the energy.
    assert torch.isfinite(found.energy).all()
    short = zbl.contacts(atype, edges, vec, mask, owned, capacity=4)
    assert torch.isnan(short.energy).any()


@pytest.mark.parametrize("lr", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("r", [0.78, 0.6], ids=["in_the_window", "below_it"])
def test_serial_compressed_and_partitioned_agree(monkeypatch, lr, compressed, r):
    level = 1
    monkeypatch.setenv("DP_CUDA_INFER", str(level))
    if compressed and not op_available():
        pytest.skip("compiled DPA4C compression operator required")
    cfg = config(lr)
    cfg["descriptor"]["precision"] = "float32"
    cfg["fitting_net"]["precision"] = "float32"
    if lr:
        cfg["long_range"] = copy.deepcopy(LONG_RANGE)
        cfg["long_range"]["charge_net"]["precision"] = "float32"
    base = get_model(cfg).eval()
    model = get_model(cfg).eval()
    model.load_state_dict(base.state_dict(), strict=True)
    # Two O-H pairs at the same distance, a cutoff apart: half way through
    # the window, or below it, where the envelope is exactly zero.
    xyz = np.array([[0.2, 1, 1], [0.2 + r, 1, 1], [6.5, 1, 1], [6.5 + r, 1, 1]])
    types = np.array([0, 1, 0, 1], dtype=np.int64)
    box = np.eye(3) * 14
    coord = torch.tensor(xyz, dtype=torch.float64, device=DEVICE)[None]
    atype = torch.tensor(types, device=DEVICE)[None]
    cell = torch.tensor(box, dtype=torch.float64, device=DEVICE)[None]
    plain = get_model({k: v for k, v in cfg.items() if k != "zbl"}).eval()
    plain.load_state_dict(base.state_dict(), strict=True)
    monkeypatch.setenv("DP_CUDA_INFER", "0")
    keys = ["energy", "force", "virial"]
    if lr:
        base.compute_born = model.compute_born = True
        keys += ["charge", "born_charge"]
    reference = base(coord, atype, box=cell)
    difference = reference["energy"] - plain(coord, atype, box=cell)["energy"]
    assert difference.abs().item() > 1.0, "the pairs must be inside the window"
    monkeypatch.setenv("DP_CUDA_INFER", str(level))
    if compressed:
        compress_model(model, True)
    actual = model(coord, atype, box=cell)
    for key in keys:
        torch.testing.assert_close(actual[key], reference[key], atol=2e-4, rtol=2e-5)
    # One physical device twice: ownership and halos, not multi-GPU speed.
    evaluator = DistributedEvaluator(
        model,
        types,
        box,
        devices=[DEVICE, DEVICE],
        compute_virial=True,
        compute_born=lr,
        memory_headroom=None,
    )
    evaluator(xyz)
    for key in keys:
        target = reference[key].detach().cpu().numpy().reshape(-1)
        np.testing.assert_allclose(
            np.asarray(evaluator.results[key]).reshape(-1),
            target,
            atol=2e-4,
            rtol=2e-5,
        )


@pytest.mark.parametrize("compressed", [False, True])
def test_the_fused_level_is_refused(monkeypatch, compressed):
    if compressed and not op_available():
        pytest.skip("compiled DPA4C compression operator required")
    cfg = config()
    cfg["descriptor"]["precision"] = "float32"
    cfg["fitting_net"]["precision"] = "float32"
    model = get_model(cfg).eval()
    coord = torch.tensor(
        [[[0.2, 1, 1], [0.98, 1, 1]]], dtype=torch.float64, device=DEVICE
    )
    atype = torch.tensor([[0, 1]], device=DEVICE)
    cell = torch.eye(3, dtype=torch.float64, device=DEVICE)[None] * 14
    if compressed:
        monkeypatch.setenv("DP_CUDA_INFER", "1")
        compress_model(model, True)
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    with pytest.raises(RuntimeError, match="DP_CUDA_INFER=2"):
        model(coord, atype, box=cell)
    if not compressed:
        with pytest.raises(RuntimeError, match="DP_CUDA_INFER=2"):
            compress_model(model, True)


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("long_range", [False, True])
def test_the_md_step_reads_the_host_no_more_often(compress, long_range):
    """The kernels read nothing back: a step with the term makes the same
    number of downloads as one without (`tests/test_sync.py`)."""
    from .test_compress import build32
    from .test_sync import assert_downloads, count_syncs, system

    if compress and not op_available():
        pytest.skip("compiled DPA4C compression operator required")
    coord, atype, cell = system()
    model = build32(long_range=long_range)
    model.set_zbl(BLOCK)
    evaluator = DistributedEvaluator(
        model, atype, cell, devices=[DEVICE], memory_headroom=None, compress=compress
    )
    assert_downloads(count_syncs(evaluator, coord), 1 if compress else 2)
