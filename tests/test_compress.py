"""Compressed DPA4C inference for ``dp_lr``.

Two halves. What runs on any install pins the *refusals*: the whole danger of
DeePMD-kit's compression is that its fast branch falls through to the
uncompressed math without a word whenever a precondition fails, so a model
that "is compressed" and runs at the old speed must be impossible to build
here. What needs the compiled operator ``deepmd::dpa4c_graph_compress`` --
absent from the PyPI wheel -- pins that the branch actually fires and that
its numbers agree with the uncompressed model.

Set ``DEEPMD_LR_REQUIRE_COMPRESS=1`` in an environment that is supposed to
have the operator: the kernel tests then fail instead of skipping, which is
how a source build proves itself.
"""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from deepmd.kernels.cuda.dpa4c.graph_compress import op_available
from deepmd.pt_expt.descriptor.dpa4c import DescrptDPA4C
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr import compress as compress_module
from deepmd_lr.ase import LongRangeCalculator, load_model
from deepmd_lr.compress import compress_model
from deepmd_lr.fitting.charge import resolve_charge_spin
from deepmd_lr.model import lr_model
from deepmd_lr.neighbors import build_neighbor_graph_fast
from deepmd_lr.nvalchemi import build_evaluator
from deepmd_lr.nvalchemi import driver as DRV
from deepmd_lr.nvalchemi.evaluator import DistributedEvaluator

from .test_joint_model import DESCRIPTOR, FITTING, LONG_RANGE, build, frame

CUDA = torch.cuda.is_available()
OP = op_available()
REQUIRE = os.environ.get("DEEPMD_LR_REQUIRE_COMPRESS") == "1"
if REQUIRE and not (CUDA and OP):
    raise pytest.UsageError(
        "DEEPMD_LR_REQUIRE_COMPRESS=1 requires a usable CUDA device and "
        f"dpa4c_graph_compress operator (CUDA={CUDA}, operator={OP})"
    )
needs_kernel = pytest.mark.skipif(
    not (CUDA and OP), reason="compressed dispatch needs CUDA and its compiled operator"
)


def cfg32(long_range: bool = True, chg_ebd: bool = False) -> dict:
    """The joint-model fixture in float32 -- the one precision compression takes."""
    descriptor = {**copy.deepcopy(DESCRIPTOR), "precision": "float32"}
    if chg_ebd:
        descriptor.update(add_chg_spin_ebd=True, default_chg_spin=[0.0, 1.0])
    cfg = {
        "type": "dp_lr",
        "type_map": ["O", "H"],
        "descriptor": descriptor,
        "fitting_net": {**copy.deepcopy(FITTING), "precision": "float32"},
    }
    if long_range:
        lr = copy.deepcopy(LONG_RANGE)
        lr["charge_net"]["precision"] = "float32"
        cfg["long_range"] = lr
    return cfg


def build32(**kwargs):
    model = get_model(cfg32(**kwargs))
    model.eval()
    return model


def plain_energy_model():
    """A standard DeePMD-kit energy model: no `type`, so not `dp_lr` at all."""
    cfg = cfg32(long_range=False)
    cfg.pop("type")
    model = get_model(cfg)
    model.eval()
    return model


def water_atoms(n_mol: int = 4, box: float = 10.0):
    coord, atype, cell = frame(n_mol=n_mol, box=box)
    symbols = ["O" if t == 0 else "H" for t in atype.reshape(-1).tolist()]
    return Atoms(
        symbols, positions=coord[0].cpu().numpy(), cell=cell[0].cpu().numpy(), pbc=True
    )


def lower_args(model, coord, atype, cell):
    """What DeepPot hands the graph lower: a flat graph and no cell."""
    graph = build_neighbor_graph_fast(
        coord, atype, cell, model.atomic_model.descriptor.get_rcut()
    )
    nloc = atype.shape[1]
    n = torch.tensor([nloc], dtype=torch.int64, device=coord.device)
    return (atype.reshape(-1), n, n, graph.edge_index, graph.edge_vec, graph.edge_mask)


# --------------------------------------------------------------------------
# Runs on the wheel: the seam, the refusals, the guard holes.
# --------------------------------------------------------------------------


class TestSeam:
    def test_get_geo_compress_is_patched_and_idempotent(self):
        patched = DescrptDPA4C.get_geo_compress
        assert getattr(patched, "_deepmd_lr_patched", False)
        compress_module.install()
        assert DescrptDPA4C.get_geo_compress is patched

    def test_uncompressed_descriptor_still_reports_false(self):
        """No behaviour change for every model that exists today."""
        assert build32().atomic_model.descriptor.get_geo_compress() is False

    def test_tables_build_and_the_flag_flips(self):
        """Upstream's table build is pure PyTorch and works without the op;
        it is the DISPATCH that needs the kernel. After it, the descriptor
        asks for CSR views -- the one thing the patch exists to do."""
        descriptor = build32().atomic_model.descriptor
        descriptor.enable_compression(0.0, table_stride_1=0.01)
        assert descriptor.compress is True
        assert descriptor.get_geo_compress() is True
        tables = [k for k in descriptor.state_dict() if k.startswith("compress_")]
        assert len(tables) == 13

    def test_float64_is_refused_upstream(self):
        with pytest.raises(ValueError, match="float32"):
            build().atomic_model.descriptor.enable_compression(0.0)

    def test_charge_state_wrappers_are_patched_and_idempotent(self):
        patched = (DescrptDPA4C.enable_compression, DescrptDPA4C.apply_charge_state)
        assert all(getattr(f, "_deepmd_lr_patched", False) for f in patched)
        compress_module.install()
        assert (
            DescrptDPA4C.enable_compression,
            DescrptDPA4C.apply_charge_state,
        ) == patched

    @pytest.mark.parametrize("route", ["descriptor", "upstream_model"])
    def test_every_route_records_the_folded_state(self, route):
        """Upstream folds `default_chg_spin`; before the record lived on the
        descriptor, only the plugin's own `enable_compression` armed the
        check, and a model compressed any other way took any runtime
        condition while the kernel read the default."""
        from deepmd.pt_expt.model.ener_model import EnergyModel

        model = build32(chg_ebd=True)
        descriptor = model.atomic_model.descriptor
        if route == "descriptor":
            descriptor.enable_compression(0.0, table_stride_1=0.01)
        else:
            EnergyModel.enable_compression(model)
        assert model._compressed_charge_spin == (0.0, 1.0)
        ref = torch.empty((), dtype=torch.float64, device=DEVICE)
        with pytest.raises(ValueError, match="folded"):
            resolve_charge_spin(model, (1.0, 2.0), 1, ref)
        descriptor.apply_charge_state([1.0, 2.0])
        assert model._compressed_charge_spin == (1.0, 2.0)
        assert resolve_charge_spin(model, (1.0, 2.0), 1, ref).tolist() == [[1.0, 2.0]]
        with pytest.raises(ValueError, match="folded"):
            resolve_charge_spin(model, (0.0, 1.0), 1, ref)

    def test_no_charge_conditioning_records_nothing(self):
        model = build32()
        model.atomic_model.descriptor.enable_compression(0.0, table_stride_1=0.01)
        assert model._compressed_charge_spin is None


class TestRefusedWithoutTheKernel:
    """Every door into compression names the missing operator and the fix."""

    @pytest.fixture(autouse=True)
    def missing_kernel(self, monkeypatch):
        from deepmd_lr.model import lr_model

        monkeypatch.setattr(lr_model, "_kernel_available", lambda: False)

    def test_the_model_method(self):
        with pytest.raises(RuntimeError, match="dpa4c_graph_compress"):
            build32().enable_compression()

    def test_the_calculator(self):
        with pytest.raises(RuntimeError, match="dpa4c_graph_compress"):
            LongRangeCalculator(build32(), compress=True)

    def test_the_evaluator(self):
        model = build32()
        atoms = water_atoms()
        atype = np.array([0 if s == "O" else 1 for s in atoms.get_chemical_symbols()])
        with pytest.raises(RuntimeError, match="dpa4c_graph_compress"):
            DistributedEvaluator(
                model, atype, atoms.get_cell(), devices=[str(DEVICE)], compress=True
            )
        with pytest.raises(RuntimeError, match="dpa4c_graph_compress"):
            build_evaluator(model, atoms, compress=True)

    def test_tables_without_the_kernel_cannot_be_evaluated(self):
        """The silent-fallback hole, closed: a descriptor that carries tables
        (e.g. one arriving through `deserialize`) is refused at forward time,
        not run through the uncompressed math."""
        model = build32()
        model.atomic_model.descriptor.enable_compression(0.0, table_stride_1=0.01)
        coord, atype, cell = frame()
        with pytest.raises(RuntimeError, match="dpa4c_graph_compress"):
            model(coord, atype, box=cell)


class TestCompressModel:
    def test_false_is_a_no_op(self):
        model = build32()
        assert compress_model(model, False) is model
        assert model.atomic_model.descriptor.compress is False

    def test_a_plain_energy_model_is_refused(self):
        """Nothing would guard it against the fall-through."""
        with pytest.raises(TypeError, match="dp_lr"):
            compress_model(plain_energy_model(), True)


class TestFallbackTripwires:
    """A compressed model may only take the captured route."""

    def test_serial_fallback_charge_pass(self):
        model = build32()
        model.atomic_model.descriptor.enable_compression(0.0, table_stride_1=0.01)
        coord, atype, cell = frame()
        with pytest.raises(RuntimeError, match="fallback charge pass"):
            model._predict_charges(coord, atype, cell, None, None)

    def test_partitioned_fallback_charge_pass(self):
        model = build32()
        model.atomic_model.descriptor.enable_compression(0.0, table_stride_1=0.01)
        coord, atype, cell = frame()
        with pytest.raises(RuntimeError, match="fallback charge pass"):
            DRV.partition_charges(model, None, atype.reshape(-1), atype.shape[1])


class TestNotPersisted:
    def test_state_dict_with_tables_is_rejected_at_load(self, tmp_path):
        """Tables are rebuilt from the weights, never restored -- and the
        strict re-check says so instead of silently dropping them."""
        import json

        model = build32()
        model.atomic_model.descriptor.enable_compression(0.0, table_stride_1=0.01)
        cfg = json.loads(model.model_def_script)
        state = {f"model.Default.{k}": v for k, v in model.state_dict().items()}
        state["_extra_state"] = {"model_params": cfg}
        path = tmp_path / "model.ckpt.pt"
        torch.save({"model": state}, path)
        with pytest.raises(ValueError, match="compress_"):
            load_model(str(path), device="cpu")


class TestGuardHoles:
    """The graph lower returns the short-range energy alone. Outside the
    model's own capture context that is a silent wrong answer -- which is
    exactly what DeepPot, `dp test` and compiled training would get."""

    def test_direct_graph_lower_is_refused_for_long_range(self):
        model = build()
        coord, atype, cell = frame()
        with pytest.raises(NotImplementedError, match="graph-lower interface"):
            model.forward_common_lower_graph(*lower_args(model, coord, atype, cell))

    def test_graph_lower_is_allowed_inside_the_capture(self):
        """The partitioned driver's route: it adds the long-range term itself."""
        model = build()
        coord, atype, cell = frame()
        with model._capturing_backbone() as captured:
            out = model.forward_common_lower_graph(
                *lower_args(model, coord, atype, cell)
            )
        assert "energy" in out
        assert captured and captured["edge_vec"].requires_grad

    def test_short_range_model_keeps_the_full_surface(self):
        model = build(long_range=False)
        coord, atype, cell = frame()
        out = model.forward_common_lower_graph(*lower_args(model, coord, atype, cell))
        assert "energy" in out

    def test_canonical_lower_is_refused_for_long_range(self):
        with pytest.raises(NotImplementedError, match="Canonical"):
            build().forward_lower_canonical_graph(*[None] * 8, do_atomic_virial=False)

    def test_canonical_lower_reaches_the_parent_without_long_range(self):
        """Whatever the parent says about dummy arguments, it is not our refusal."""
        with pytest.raises(Exception) as excinfo:
            build(long_range=False).forward_lower_canonical_graph(
                *[None] * 8, do_atomic_virial=False
            )
        assert "long_range block" not in str(excinfo.value)

    def test_capture_depth_is_balanced(self):
        model = build()
        assert model._capture_depth == 0
        with model._capturing_backbone():
            assert model._capture_depth == 1
            with model._capturing_backbone():
                assert model._capture_depth == 2
            assert model._capture_depth == 1
        assert model._capture_depth == 0


class TestChargeState:
    def test_a_condition_differing_from_the_folded_state_is_refused(self):
        model = build32(chg_ebd=True)
        model._compressed_charge_spin = (0.0, 1.0)
        ref = torch.empty((), dtype=torch.float64, device=DEVICE)
        with pytest.raises(ValueError, match="folded"):
            resolve_charge_spin(model, (1.0, 2.0), 1, ref)
        assert resolve_charge_spin(model, (0.0, 1.0), 1, ref).tolist() == [[0.0, 1.0]]
        # None resolves to the default, which is the folded state here.
        assert resolve_charge_spin(model, None, 1, ref).tolist() == [[0.0, 1.0]]

    def test_no_folded_state_means_no_check(self):
        model = build32(chg_ebd=True)
        ref = torch.empty((), dtype=torch.float64, device=DEVICE)
        assert resolve_charge_spin(model, (1.0, 2.0), 1, ref).tolist() == [[1.0, 2.0]]


# --------------------------------------------------------------------------
# Needs the compiled operator (a source-built deepmd-kit).
# --------------------------------------------------------------------------


def twins(**kwargs):
    """Two models with identical weights; the second gets compressed."""
    a = build32(**kwargs)
    b = build32(**kwargs)
    b.load_state_dict(a.state_dict())
    return a, b


@needs_kernel
class TestKernel:
    @pytest.fixture(autouse=True)
    def level_one(self, monkeypatch):
        monkeypatch.setenv("DP_CUDA_INFER", "1")

    def test_the_op_actually_dispatches_once_per_forward(self, monkeypatch):
        """Without this every equivalence test could pass via the fallback."""
        import deepmd.kernels.cuda.dpa4c.graph_compress as gc

        calls = []
        original = gc.dpa4c_graph_compress

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(gc, "dpa4c_graph_compress", counting)
        model = build32()
        model.enable_compression()
        coord, atype, cell = frame()
        model(coord, atype, box=cell)
        assert len(calls) == 1

    def test_unset_level_becomes_one_and_two_is_refused(self, monkeypatch):
        monkeypatch.delenv("DP_CUDA_INFER", raising=False)
        model = build32()
        model.enable_compression()
        assert os.environ["DP_CUDA_INFER"] == "1"
        monkeypatch.setenv("DP_CUDA_INFER", "2")
        with pytest.raises(RuntimeError, match="DP_CUDA_INFER=2"):
            build32().enable_compression()

    def test_level_changed_after_compressing_is_caught_at_forward(self, monkeypatch):
        model = build32()
        model.enable_compression()
        coord, atype, cell = frame()
        monkeypatch.setenv("DP_CUDA_INFER", "0")
        with pytest.raises(RuntimeError, match="DP_CUDA_INFER"):
            model(coord, atype, box=cell)

    def test_training_mode_is_refused(self):
        model = build32()
        model.train()
        with pytest.raises(RuntimeError, match="eval"):
            model.enable_compression()

    @pytest.mark.parametrize("stride", [0.01, 0.002])
    def test_compressed_matches_uncompressed(self, stride):
        """Same weights, both at level 1; the gap is the tabulation error."""
        plain, packed = twins()
        packed.enable_compression(table_stride_1=stride)
        plain.compute_born = packed.compute_born = True
        coord, atype, cell = frame(n_mol=8, box=12.0)
        a = plain(coord, atype, box=cell)
        b = packed(coord, atype, box=cell)
        ceilings = {
            "energy": 1e-3,
            "force": 2e-3,
            "virial": 2e-2,
            "charge": 1e-4,
            "born_charge": 1e-3,
        }
        report = {}
        for key, ceiling in ceilings.items():
            gap = float((a[key] - b[key]).abs().max())
            report[key] = gap
            assert gap <= ceiling, (
                f"{key}: {gap:.3e} > {ceiling:.1e} at stride {stride}"
            )
        print(
            f"stride {stride}: " + ", ".join(f"{k} {v:.2e}" for k, v in report.items())
        )

    def test_backbone_runs_once_and_the_fallback_never_fires(self, monkeypatch):
        model = build32()
        model.enable_compression()
        descriptor = model.atomic_model.descriptor
        calls = []
        original = type(descriptor).call_graph

        def counting(self_, *args, **kwargs):
            calls.append(1)
            return original(self_, *args, **kwargs)

        monkeypatch.setattr(type(descriptor), "call_graph", counting)
        fallback = []
        monkeypatch.setattr(
            type(model),
            "_predict_charges",
            lambda *a, **k: fallback.append(1) or pytest.fail("fallback taken"),
        )
        coord, atype, cell = frame()
        model(coord, atype, box=cell)
        assert len(calls) == 1
        assert not fallback

    def test_charge_state_is_folded(self):
        plain, packed = twins(chg_ebd=True)
        packed.enable_compression(charge_spin=(1.0, 2.0))
        coord, atype, cell = frame()
        condition = torch.tensor([[1.0, 2.0]], dtype=torch.float64, device=DEVICE)
        a = plain(coord, atype, box=cell, charge_spin=condition)
        b = packed(coord, atype, box=cell, charge_spin=condition)
        assert float((a["energy"] - b["energy"]).abs()) <= 1e-3
        with pytest.raises(ValueError, match="folded"):
            packed(
                coord,
                atype,
                box=cell,
                charge_spin=torch.tensor(
                    [[0.0, 1.0]], dtype=torch.float64, device=DEVICE
                ),
            )

    @pytest.mark.parametrize("long_range", [True, False])
    def test_upstream_compression_refuses_another_charge_state(self, long_range):
        """The case measured on the water SR baseline: upstream's method,
        a forward at [1, 2], and the default state's energy back without a
        word (18 meV on 1,536 atoms). Now refused; the folded state runs and
        matches the uncompressed model."""
        from deepmd.pt_expt.model.ener_model import EnergyModel

        plain, packed = twins(chg_ebd=True, long_range=long_range)
        EnergyModel.enable_compression(packed)
        coord, atype, cell = frame()

        def cond(q, m):
            return torch.tensor([[q, m]], dtype=torch.float64, device=DEVICE)

        with pytest.raises(ValueError, match="folded"):
            packed(coord, atype, box=cell, charge_spin=cond(1.0, 2.0))
        a = plain(coord, atype, box=cell, charge_spin=cond(0.0, 1.0))
        b = packed(coord, atype, box=cell, charge_spin=cond(0.0, 1.0))
        assert float((a["energy"] - b["energy"]).abs()) <= 1e-3

    @pytest.mark.parametrize("n_part", [1, 2, 4])
    def test_partitioned_equals_serial_compressed(self, n_part):
        model = build32()
        model.enable_compression()
        model.compute_born = True
        coord, atype, cell = frame(n_mol=8, box=12.0)
        serial = model(coord, atype, box=cell)
        parted = DRV.to_numpy(DRV.evaluate_partitioned(
            model,
            coord[0].cpu().numpy(),
            atype[0].cpu().numpy(),
            cell[0].cpu().numpy(),
            n_partitions=n_part,
            compute_born=True,
        ))
        assert abs(parted["energy"] - float(serial["energy"])) < 1e-6
        np.testing.assert_allclose(
            parted["force"], serial["force"][0].detach().cpu().numpy(), atol=1e-6
        )
        np.testing.assert_allclose(
            parted["born_charge"],
            serial["born_charge"][0].detach().cpu().numpy(),
            atol=1e-6,
        )

    @pytest.mark.parametrize("n_part", [1, 2, 4])
    def test_the_gpu_list_feeds_the_compressed_kernel(self, n_part):
        """Each partition's GPU list hands the compressed kernel CSR views made
        on the device, without a sort; repeated calls on the same partitions
        give the same answer as the first."""
        from deepmd_lr.nvalchemi.partition import partition_system

        model = build32()
        model.enable_compression()
        model.compute_born = True
        coord, atype, cell = frame(n_mol=8, box=12.0)
        c, a, h = (t[0].cpu().numpy() for t in (coord, atype, cell))
        parts = partition_system(c, h, float(model.get_rcut()), n_part, skin=1.0)

        def run():
            return DRV.to_numpy(DRV.evaluate_partitioned(
                model, c, a, h, n_partitions=n_part, compute_born=True,
                partitions=parts,
            ))

        fresh = run()
        for _ in range(2):
            cached = run()
            assert abs(cached["energy"] - fresh["energy"]) < 1e-6
            for key in ("force", "born_charge"):
                np.testing.assert_allclose(cached[key], fresh[key], atol=1e-6)
        assert all(part.neighbor_list_on(DEVICE) is not None for part in parts)

    def test_a_replica_carries_the_tables(self):
        model = build32()
        model.enable_compression()
        replica = copy.deepcopy(model).to(DEVICE).eval()
        d = replica.atomic_model.descriptor
        assert d.compress and d.get_geo_compress()
        assert (
            d._compression_scalars == model.atomic_model.descriptor._compression_scalars
        )
        coord, atype, cell = frame()
        assert (
            float(
                (
                    model(coord, atype, box=cell)["energy"]
                    - replica(coord, atype, box=cell)["energy"]
                ).abs()
            )
            < 1e-6
        )

    def test_calculator_with_a_field_runs_compressed(self):
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(
            build32(), efield=[0.1, 0.0, 0.0], compress=True
        )
        forces = atoms.get_forces()
        assert np.isfinite(forces).all()
        assert np.isfinite(atoms.calc.results["born_charge"]).all()


# --- compress_model's switch and stride contracts ------------------------------

@pytest.mark.parametrize(
    "value", [0, 0.0, -0.01, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_numeric_stride_is_rejected(value):
    with pytest.raises(ValueError, match="positive and finite"):
        compress_model(None, value)


@pytest.mark.parametrize("value", [None, "0.01", [], {}])
def test_non_numeric_stride_is_rejected(value):
    with pytest.raises(TypeError, match="bool or"):
        compress_model(None, value)


def test_false_true_and_positive_stride(monkeypatch):
    model = build32()
    calls = []
    monkeypatch.setattr(
        model, "enable_compression", lambda **kwargs: calls.append(kwargs)
    )
    assert compress_model(model, False) is model
    assert calls == []
    compress_model(model, True)
    compress_model(model, 0.002)
    assert calls == [
        {"charge_spin": None},
        {"table_stride_1": 0.002, "charge_spin": None},
    ]


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_direct_model_api_rejects_bad_stride_before_mutation(monkeypatch, value):
    monkeypatch.setenv("DP_CUDA_INFER", "0")
    model = build32()
    with pytest.raises(ValueError, match="positive and finite"):
        model.enable_compression(table_stride_1=value)
    assert not model.atomic_model.descriptor.compress
    assert os.environ["DP_CUDA_INFER"] == "0"


def test_level_two_is_rejected_for_a_long_range_model(monkeypatch):
    monkeypatch.setattr(lr_model, "_kernel_available", lambda: True)
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    model = build32(long_range=True)
    with pytest.raises(RuntimeError, match="DP_CUDA_INFER=2"):
        model.enable_compression()
    assert not model.atomic_model.descriptor.compress


def test_level_two_compresses_a_short_range_model(monkeypatch):
    # No charge head shares the backbone, so the fused energy/force path is allowed.
    monkeypatch.setattr(lr_model, "_kernel_available", lambda: True)
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    model = build32(long_range=False)
    model.enable_compression()
    assert model.atomic_model.descriptor.compress


# --- a failed compression leaves the model as it was ---------------------------

@pytest.fixture
def available_operator(monkeypatch):
    # Table construction uses PyTorch; CUDA dispatch is outside these tests.
    monkeypatch.setattr(lr_model, "_kernel_available", lambda: True)


@pytest.mark.parametrize("level", [None, "0", "1"])
@pytest.mark.parametrize("condition", [[1, 2, 3], [float("nan"), 1], [0.5, 1]])
def test_invalid_condition_leaves_state_unchanged(
    available_operator, monkeypatch, level, condition
):
    if level is None:
        monkeypatch.delenv("DP_CUDA_INFER", raising=False)
    else:
        monkeypatch.setenv("DP_CUDA_INFER", level)
    model = build32(chg_ebd=True)
    descriptor = model.atomic_model.descriptor
    before = descriptor.state_dict()
    with pytest.raises(ValueError):
        model.enable_compression(charge_spin=condition)
    assert not descriptor.compress
    assert getattr(model, "_compressed_charge_spin", None) is None
    assert os.environ.get("DP_CUDA_INFER") == level
    assert descriptor.state_dict().keys() == before.keys()
    for key, value in before.items():
        torch.testing.assert_close(descriptor.state_dict()[key], value)
    model.enable_compression(charge_spin=[0, 1])
    assert descriptor.compress
    assert model._compressed_charge_spin == (0, 1)


@pytest.mark.parametrize("method", ["enable_compression", "apply_charge_state"])
def test_partial_failure_restores_buffers_and_allows_retry(
    available_operator, monkeypatch, method
):
    monkeypatch.delenv("DP_CUDA_INFER", raising=False)
    model = build32(chg_ebd=True)
    descriptor = model.atomic_model.descriptor
    original = getattr(type(descriptor), method)
    buffers = descriptor._buffers.copy()
    had_scalars = hasattr(descriptor, "_compression_scalars")
    scalars = getattr(descriptor, "_compression_scalars", None)

    def fail_after_mutation(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected post-mutation failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(descriptor), method, fail_after_mutation)
        with pytest.raises(RuntimeError, match="injected"):
            model.enable_compression(charge_spin=[0, 1])
    assert model.atomic_model.descriptor is descriptor
    assert not descriptor.compress
    assert descriptor._buffers.keys() == buffers.keys()
    assert all(descriptor._buffers[key] is value for key, value in buffers.items())
    assert hasattr(descriptor, "_compression_scalars") == had_scalars
    assert getattr(descriptor, "_compression_scalars", None) == scalars
    assert "DP_CUDA_INFER" not in os.environ
    model.enable_compression(charge_spin=[0, 1])
    assert descriptor.compress
    with pytest.raises(ValueError, match="already"):
        model.enable_compression(charge_spin=[0, 1])
    assert descriptor.compress
    assert model._compressed_charge_spin == (0, 1)


# --- DEEPMD_LR_REQUIRE_COMPRESS=1 refuses to run without the prerequisites -----

@pytest.mark.parametrize(
    "cuda,operator", [(False, True), (True, False), (False, False)]
)
def test_required_mode_fails_collection_without_prerequisites(cuda, operator):
    # Import the backend before simulating availability. No kernels are executed.
    code = f"""
import deepmd.pt_expt
import torch
import deepmd.kernels.cuda.dpa4c.graph_compress as gc
import pytest
torch.cuda.is_available = lambda: {cuda!r}
gc.op_available = lambda: {operator!r}
raise SystemExit(pytest.main([
    '-p', 'no:cacheprovider', '--collect-only', '-q', 'tests/test_compress.py'
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "DEEPMD_LR_REQUIRE_COMPRESS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "requires a usable CUDA device" in result.stdout + result.stderr
