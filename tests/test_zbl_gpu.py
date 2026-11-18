"""The short-contact term on the GPU: the kernel's list of contacts is the
torch arithmetic, and compressed and partitioned evaluation keep the term, by
whichever of DeePMD-kit's routes evaluates the network: the autograd lower,
the fused operators of ``DP_CUDA_INFER=2``, the compact canonical lower a
compressed export traces."""

import copy

import numpy as np
import pytest
import torch
from deepmd.dpmodel.utils.neighbor_graph import NeighborGraph, attach_edge_csr
from deepmd.kernels.cuda.dpa4c.graph_compress import op_available
from deepmd.kernels.utils import cuda_infer_level
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.canonical_graph import canonical_graph_from_neighbor_graph
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.compress import compress_model
from deepmd_lr.neighbors import build_neighbor_graph_fast
from deepmd_lr.nvalchemi import DistributedEvaluator
from deepmd_lr.zbl import ZBL, add_envelope_gradient

from .test_joint_model import LONG_RANGE
from .test_zbl import BLOCK, config, contact_frame, dimer, twins

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def fused_config(lr: bool = False) -> dict:
    """The test model in the form DeePMD-kit compresses and its fused
    operators serve: float32, and a fitting net without the per-layer
    timestep, which those operators do not implement."""
    cfg = config(lr)
    cfg["descriptor"]["precision"] = "float32"
    cfg["fitting_net"]["precision"] = "float32"
    cfg["fitting_net"]["resnet_dt"] = False
    return cfg


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
    w_ref = zbl.envelope(atype, edges, leaf, mask).double()
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


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("lr", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("r", [0.78, 0.6], ids=["in_the_window", "below_it"])
def test_serial_compressed_and_partitioned_agree(monkeypatch, lr, compressed, r, level):
    """Level 2 is DeePMD-kit's fused energy/force operators for a compressed
    short-range model, where the envelope rides in on the ownership mask; a
    long-range model shares its backbone with the charge head and stays at 1."""
    if lr and level == 2:
        pytest.skip("a long-range model is evaluated at level 1")
    monkeypatch.setenv("DP_CUDA_INFER", str(level))
    if compressed and not op_available():
        pytest.skip("compiled DPA4C compression operator required")
    cfg = fused_config(lr)
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
    served = fused_calls(model)
    actual = model(coord, atype, box=cell)
    for key in keys:
        torch.testing.assert_close(actual[key], reference[key], atol=2e-4, rtol=2e-5)
    # At level 2 DeePMD-kit asks the fused operators, and they serve the
    # compressed model; any other takes the autograd lower.
    assert served == ([compressed] if level == 2 else [])
    if not lr:
        # The per-atom virial is summed from the same list of contacts.
        atomic = model(coord, atype, box=cell, do_atomic_virial=True)
        for key in keys:
            torch.testing.assert_close(
                atomic[key], reference[key], atol=2e-4, rtol=2e-5
            )
        torch.testing.assert_close(
            atomic["atom_virial"].sum(dim=1).double(),
            reference["virial"],
            atol=2e-4,
            rtol=2e-5,
        )
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


def fused_calls(model) -> list:
    """Record, for every call of DeePMD-kit's fused energy/force operators on
    ``model`` from now on, whether they served it."""
    descriptor = model.atomic_model.descriptor
    original = descriptor.fused_energy_force_graph
    served: list = []

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        if cuda_infer_level() >= 2:
            served.append(out is not None)
        return out

    descriptor.fused_energy_force_graph = spy
    return served


def compact_inputs(model, coord, atype, cell):
    """One frame as the compact canonical lower takes it: every edge once,
    grouped by destination, float32."""
    n = atype.numel()
    graph = attach_edge_csr(
        build_neighbor_graph_fast(coord, atype, cell, model.get_rcut()),
        n,
        canonicalize=True,
    )
    counts = torch.tensor([n], device=DEVICE)
    compact = canonical_graph_from_neighbor_graph(
        NeighborGraph(
            n_node=counts,
            edge_index=graph.edge_index,
            edge_vec=graph.edge_vec.to(torch.float32),
            edge_mask=graph.edge_mask,
            n_local=counts,
            destination_order=graph.destination_order,
            destination_row_ptr=graph.destination_row_ptr,
            source_order=graph.source_order,
            source_row_ptr=graph.source_row_ptr,
            destination_sorted=True,
        )
    )
    return (
        atype.reshape(-1),
        compact.n_node,
        compact.n_local,
        compact.source,
        compact.edge_vec,
        compact.destination_row_ptr,
        compact.source_row_ptr,
        compact.source_order,
    )


@pytest.mark.parametrize("r", [0.78, 0.6], ids=["in_the_window", "below_it"])
def test_the_canonical_lower_keeps_the_term(monkeypatch, r):
    """What a compressed export traces and LAMMPS runs: DeePMD-kit's compact
    energy/force operator, weighed by the envelope through its ownership
    mask, with ZBL and the envelope's derivative assembled beside it."""
    if not op_available():
        pytest.skip("compiled DPA4C compression operator required")
    cfg = fused_config()
    model = get_model(cfg).eval()
    # Offsets, which the envelope must leave whole.
    with torch.no_grad():
        model.atomic_model.out_bias[0, :, 0] = torch.tensor([-3.0, -1.5])
    plain = get_model({k: v for k, v in cfg.items() if k != "zbl"}).eval()
    plain.load_state_dict(model.state_dict(), strict=True)
    xyz = [[0.2, 1, 1], [0.2 + r, 1, 1], [6.5, 1, 1], [6.5 + r, 1, 1], [3.3, 4, 4]]
    coord = torch.tensor([xyz], dtype=torch.float64, device=DEVICE)
    atype = torch.tensor([[0, 1, 0, 1, 0]], device=DEVICE)
    cell = torch.eye(3, dtype=torch.float64, device=DEVICE)[None] * 14
    monkeypatch.setenv("DP_CUDA_INFER", "0")
    reference = model(coord, atype, box=cell, do_atomic_virial=True)
    shift = reference["energy"] - plain(coord, atype, box=cell)["energy"]
    assert shift.abs().item() > 1.0, "the pairs must be inside the window"
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    compress_model(model, True)
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    inputs = compact_inputs(model, coord, atype, cell)

    def check(output):
        expected = {
            "energy": reference["energy"],
            "atom_energy": reference["atom_energy"],
            "force": reference["force"],
            "virial": reference["virial"],
            "atom_virial": reference["atom_virial"],
        }
        for key, value in expected.items():
            torch.testing.assert_close(
                output[key].double().reshape(-1),
                value.double().reshape(-1),
                atol=2e-4,
                rtol=2e-5,
            )
        # An atom the envelope switched off is still an atom of the frame.
        assert output["mask"].tolist() == [1] * atype.numel()

    check(model.forward_lower_canonical_graph(*inputs, do_atomic_virial=True))
    lean = model.forward_lower_canonical_graph(*inputs, do_atomic_virial=False)
    torch.testing.assert_close(
        lean["force"].double().reshape(-1),
        reference["force"].reshape(-1),
        atol=2e-4,
        rtol=2e-5,
    )
    # The trace an export compiles gives the same numbers.
    traced = model.forward_lower_canonical_graph_exportable(
        *inputs, do_atomic_virial=True
    )
    check(traced(*inputs))


def test_the_compiled_exports_keep_the_term(tmp_path, monkeypatch):
    """The files LAMMPS loads, made as ``dp freeze`` and ``dp compress`` make
    them and read back through DeePMD-kit's own loader: the graph lower of
    the model as trained, then the compact canonical lower of the compressed
    one, which ``dp compress`` builds from the frozen file.

    A trace runs at ``DP_CUDA_INFER=2`` and its arithmetic is compiled for
    the device, which is where a term can be lost or miscompiled without a
    trace of the lower noticing (``torch.searchsorted`` on a sliced sequence
    is, by torch 2.11's AOTInductor)."""
    from deepmd.infer import DeepPot
    from deepmd.pt_expt.entrypoints.compress import enable_compression as dp_compress
    from deepmd.pt_expt.utils.serialization import deserialize_to_file

    if not op_available():
        pytest.skip("compiled DPA4C compression operator required")
    model = get_model(fused_config()).eval()
    with torch.no_grad():
        model.atomic_model.out_bias[0, :, 0] = torch.tensor([-3.0, -1.5])
    plain = get_model({k: v for k, v in fused_config().items() if k != "zbl"}).eval()
    plain.load_state_dict(model.state_dict(), strict=True)
    # Two frames, as a trace is made on several. In each, one O-H pair in
    # its window and one below it, where the envelope is exactly zero, and
    # an atom in neither.
    xyz = [
        [[0.2, 1, 1], [0.98, 1, 1], [6.5, 1, 1], [7.1, 1, 1], [3.3, 4, 4]],
        [[0.2, 1, 1], [0.94, 1, 1], [6.5, 1, 1], [7.0, 1, 1], [3.3, 4, 5]],
    ]
    coord = torch.tensor(xyz, dtype=torch.float64, device=DEVICE)
    atype = torch.tensor([[0, 1, 0, 1, 0]] * 2, device=DEVICE)
    cell = torch.eye(3, dtype=torch.float64, device=DEVICE).repeat(2, 1, 1) * 14
    monkeypatch.setenv("DP_CUDA_INFER", "0")
    reference = model(coord, atype, box=cell, do_atomic_virial=True)
    shift = reference["energy"] - plain(coord, atype, box=cell)["energy"]
    assert shift.abs().min().item() > 1.0, "the pairs must be inside the window"
    assert (reference["energy"][0] - reference["energy"][1]).abs().item() > 0.1

    def check(path, kind):
        loaded = DeepPot(path)
        assert loaded.deep_eval.metadata["lower_input_kind"] == kind
        energy, force, virial, atom_energy, atom_virial = loaded.eval(
            coord.cpu().numpy(),
            cell.cpu().numpy(),
            atype[0].cpu().numpy(),
            atomic=True,
        )
        expected = {
            "energy": energy,
            "force": force,
            "virial": virial,
            "atom_energy": atom_energy,
            "atom_virial": atom_virial,
        }
        for key, value in expected.items():
            np.testing.assert_allclose(
                np.asarray(value).reshape(-1),
                reference[key].detach().cpu().numpy().reshape(-1),
                atol=2e-4,
                rtol=2e-5,
                err_msg=f"{kind}: {key}",
            )

    # `dp freeze`: the model as trained, on the graph lower.
    monkeypatch.delenv("DP_CUDA_INFER")
    frozen, compressed = str(tmp_path / "frozen.pt2"), str(tmp_path / "compressed.pt2")
    deserialize_to_file(frozen, {"model": model.serialize()}, lower_kind="auto")
    check(frozen, "graph")
    # `dp compress`: DeePMD-kit's own entry point, on that file. It
    # compresses the model it deserializes without leaving training mode.
    dp_compress(frozen, compressed)
    check(compressed, "dpa4c_canonical")


@pytest.mark.parametrize("partitions", [1, 2])
@pytest.mark.parametrize("r", [0.78, 0.6])
def test_charged_filter_fallback_agrees_across_partitions(partitions, r):
    model, _ = twins(lr=True)
    model.charge_filter = (100.0, 1.0)
    coord, atype, cell = contact_frame(r)
    state = torch.tensor([[1.0, 0.0]], dtype=coord.dtype, device=DEVICE)
    reference = model(coord, atype, box=cell, charge_spin=state)
    xyz, types, box = (x[0].cpu().numpy() for x in (coord, atype, cell))
    evaluator = DistributedEvaluator(
        model,
        types,
        box,
        devices=[DEVICE] * partitions,
        charge_spin=[1.0, 0.0],
        charge_filter=model.charge_filter,
        compute_virial=True,
        memory_headroom=None,
    )
    evaluator(xyz)
    for key in ("energy", "force", "virial", "charge"):
        np.testing.assert_allclose(
            np.asarray(evaluator.results[key]).reshape(-1),
            reference[key].detach().cpu().numpy().reshape(-1),
            atol=1e-8,
            rtol=1e-8,
        )
    assert np.asarray(evaluator.results["charge"]).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("filtered", [False, True])
def test_md_rejects_a_fully_closed_charged_frame(filtered):
    from ase import Atoms

    from deepmd_lr.nvalchemi import MDInstabilityError, VelocityVerlet

    model, _ = twins(lr=True)
    coord, atype = dimer(0.6)
    xyz, types = (x[0].cpu().numpy() for x in (coord, atype))
    atoms = Atoms("OH", positions=xyz, cell=np.eye(3) * 14, pbc=True)
    evaluator = DistributedEvaluator(
        model,
        types,
        atoms.cell.array,
        devices=[DEVICE, DEVICE],
        charge_spin=[1.0, 0.0],
        charge_filter=(100.0, 1.0) if filtered else None,
        memory_headroom=None,
    )
    md = VelocityVerlet(evaluator, atoms, timestep=0.1, logfile=None)
    with pytest.raises(MDInstabilityError, match="non-finite energy"):
        md.run(1)
    assert md.nsteps == 0


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize(
    ("long_range", "filtered"), [(False, False), (True, False), (True, True)]
)
def test_the_md_step_reads_the_host_no_more_often(compress, long_range, filtered):
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
        model,
        atype,
        cell,
        devices=[DEVICE],
        memory_headroom=None,
        compress=compress,
        charge_spin=[1.0, 0.0] if filtered else None,
        charge_filter=(100.0, 1.0) if filtered else None,
    )
    assert_downloads(count_syncs(evaluator, coord), 1 if compress else 2)
