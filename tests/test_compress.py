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
