# SPDX-License-Identifier: MIT
"""A multipole head on a compressed DPA4C: its moments from the radial table.

What is pinned:
- **The table is the descriptor**: the moments it gives are the ones the
  uncompressed descriptor reduces, and so are their gradients.
- **It holds what it has met**: a new type extends it, another frame
  condition rebuilds it.
- **The Triton kernel is the torch route**, values and gradients.
- **The compressed model is the uncompressed one**: energy, forces, charges,
  moments and Born tensors, serial and partitioned, by either route.
"""

import copy
import types

import numpy as np
import pytest
import torch
from deepmd.kernels.cuda.dpa4c.graph_compress import op_available
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.fitting.moment_table import MomentTable
from deepmd_lr.kernels import moments as kernel

from .test_dpa4 import evaluate, roughened
from .test_joint_model import DESCRIPTOR, frame
from .test_multipole_model import TestBornTensor, config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)
needs_operator = pytest.mark.skipif(
    not op_available(), reason="compression needs DeePMD-kit's compiled operator"
)
needs_triton = pytest.mark.skipif(not kernel.available(), reason="needs Triton")
CONDITION = (0.0, 1.0)


def build(lmax: int = 2, precision: str = "float64", **descriptor):
    cfg = config(lmax)
    cfg["descriptor"] = {
        **copy.deepcopy(DESCRIPTOR),
        "precision": precision,
        "add_chg_spin_ebd": True,
        "default_chg_spin": list(CONDITION),
        **descriptor,
    }
    for block in (cfg["fitting_net"], *cfg["long_range"].values()):
        if "precision" in block:
            block["precision"] = precision
    return roughened(get_model(cfg))


def captured(model, system, dtype=None):
    """The graph of one forward and the moments the descriptor reduced on
    it."""
    coord, atype, cell = system
    with model._capturing_backbone() as seen:
        model(coord, atype, box=cell)
    edge_vec = seen["edge_vec"].detach().to(dtype or seen["edge_vec"].dtype)
    graph = types.SimpleNamespace(
        edge_vec=edge_vec.clone().requires_grad_(True),
        edge_index=seen["edge_index"],
        edge_mask=seen["edge_mask"],
    )
    return graph, atype.reshape(-1), seen["equivariant"]


def weights(like: torch.Tensor) -> torch.Tensor:
    rng = torch.Generator().manual_seed(3)
    return torch.randn(like.shape, generator=rng, dtype=torch.float64).to(like)


class TestTable:
    @pytest.mark.parametrize("lmax", [1, 2])
    def test_it_gives_the_descriptors_moments(self, lmax):
        model = build(lmax)
        descriptor = model.atomic_model.descriptor
        graph, atype, reference = captured(model, frame(n_mol=16, box=14.0))
        table = MomentTable(descriptor, lmax)
        assert not table.fused(graph.edge_vec)  # float64: the torch route
        moments = table(graph, atype, CONDITION)
        start = descriptor.degree_offsets[1]
        stop = descriptor.degree_offsets[lmax + 1]
        assert moments.shape == (atype.shape[0], stop)
        assert float(reference[:, start:stop].detach().abs().max()) > 0.1
        torch.testing.assert_close(
            moments[:, start:], reference[:, start:stop], atol=1e-8, rtol=0
        )
        assert float(moments[:, :start].abs().max()) == 0.0

    def test_its_gradient_is_the_descriptors(self):
        model = build(2)
        descriptor = model.atomic_model.descriptor
        coord, atype, cell = frame(n_mol=16, box=14.0)
        start, stop = descriptor.degree_offsets[1], descriptor.degree_offsets[3]
        with model._capturing_backbone() as seen:
            model(coord, atype, box=cell)
        reference = seen["equivariant"][:, start:stop]
        weight = weights(reference)
        (expected,) = torch.autograd.grad((reference * weight).sum(), seen["edge_vec"])
        graph = types.SimpleNamespace(
            edge_vec=seen["edge_vec"].detach().clone().requires_grad_(True),
            edge_index=seen["edge_index"],
            edge_mask=seen["edge_mask"],
        )
        moments = MomentTable(descriptor, 2)(graph, atype.reshape(-1), CONDITION)
        (gradient,) = torch.autograd.grad(
            (moments[:, start:] * weight).sum(), graph.edge_vec
        )
        assert float(expected.abs().max()) > 0.1
        torch.testing.assert_close(gradient, expected, atol=1e-6, rtol=0)

    def test_it_grows_with_the_types_it_meets(self):
        model = build(2)
        descriptor = model.atomic_model.descriptor
        graph, atype, reference = captured(model, frame(n_mol=6, box=10.0))
        table = MomentTable(descriptor, 2)
        only_h = atype == 1
        hydrogen = types.SimpleNamespace(
            edge_vec=graph.edge_vec,
            edge_index=graph.edge_index,
            edge_mask=graph.edge_mask
            & only_h[graph.edge_index[0]]
            & only_h[graph.edge_index[1]],
        )
        table(hydrogen, torch.ones_like(atype), CONDITION)
        rows = table.knots + 1
        assert table.table.shape[0] == rows  # one ordered pair
        moments = table(graph, atype, CONDITION)
        assert table.table.shape[0] == 4 * rows
        start = descriptor.degree_offsets[1]
        torch.testing.assert_close(
            moments[:, start:], reference[:, start:], atol=1e-8, rtol=0
        )
        held = table.table
        table(graph, atype, CONDITION)
        assert table.table is held  # nothing new: nothing rebuilt

    def test_another_frame_condition_rebuilds_it(self):
        model = build(2)
        descriptor = model.atomic_model.descriptor
        graph, atype, _ = captured(model, frame(n_mol=6, box=10.0))
        table = MomentTable(descriptor, 2)
        neutral = table(graph, atype, CONDITION)
        charged = table(graph, atype, (1.0, 2.0))
        assert float((charged - neutral).abs().max()) > 1e-6
        again = table(graph, atype, CONDITION)
        torch.testing.assert_close(again, neutral, atol=1e-12, rtol=0)


@needs_triton
class TestKernel:
    @pytest.mark.parametrize("lmax", [1, 2])
    def test_it_is_the_torch_route(self, lmax, monkeypatch):
        model = build(lmax, "float32")
        descriptor = model.atomic_model.descriptor
        graph, atype, _ = captured(
            model, frame(n_mol=16, box=14.0), dtype=torch.float32
        )
        table = MomentTable(descriptor, lmax)
        results = []
        for switch in ("1", "0"):
            monkeypatch.setenv("DP_LR_TRITON", switch)
            assert table.fused(graph.edge_vec) == (switch == "1")
            moments = table(graph, atype, CONDITION)
            (gradient,) = torch.autograd.grad(
                (moments * weights(moments)).sum(), graph.edge_vec
            )
            results.append((moments.detach(), gradient))
        (fused, fused_gradient), (plain, plain_gradient) = results
        assert float(plain.abs().max()) > 0.1
        assert float(plain_gradient.abs().max()) > 0.1
        torch.testing.assert_close(fused, plain, atol=2e-6, rtol=0)
        torch.testing.assert_close(fused_gradient, plain_gradient, atol=5e-6, rtol=0)

    def test_a_second_derivative_is_refused(self):
        model = build(2, "float32")
        graph, atype, _ = captured(model, frame(n_mol=6, box=10.0), dtype=torch.float32)
        table = MomentTable(model.atomic_model.descriptor, 2)
        moments = table(graph, atype, CONDITION)
        (gradient,) = torch.autograd.grad(
            moments.square().sum(), graph.edge_vec, create_graph=True
        )
        with pytest.raises(RuntimeError):
            torch.autograd.grad(gradient.square().sum(), graph.edge_vec)

    def test_training_takes_the_torch_route(self):
        model = build(2, "float32")
        descriptor = model.atomic_model.descriptor
        graph, _, _ = captured(model, frame(n_mol=6, box=10.0), dtype=torch.float32)
        table = MomentTable(descriptor, 2)
        assert table.fused(graph.edge_vec)
        descriptor.train()
        assert not table.fused(graph.edge_vec)


@needs_operator
class TestCompressedModel:
    @pytest.mark.parametrize("triton", ["1", "0"])
    @pytest.mark.parametrize("lmax", [1, 2])
    def test_it_is_the_uncompressed_model(self, lmax, triton, monkeypatch):
        monkeypatch.setenv("DP_LR_TRITON", triton)
        model = build(lmax, "float32")
        system = frame(n_mol=16, box=14.0)
        before = evaluate(model, *system)
        born = TestBornTensor.born(model, *system)
        model.enable_compression()
        after = evaluate(model, *system)
        assert model._moment_table is not None
        assert model._moment_table.fused(system[0].float()) == (
            triton == "1" and kernel.available()
        )
        keys = ["charge", "dipole"] + ["quadrupole"] * (lmax == 2)
        for key in keys:
            torch.testing.assert_close(after[key], before[key], atol=2e-6, rtol=0)
        assert float(after["energy"]) == pytest.approx(
            float(before["energy"]), abs=2e-5
        )
        torch.testing.assert_close(after["force"], before["force"], atol=1e-4, rtol=0)
        torch.testing.assert_close(
            TestBornTensor.born(model, *system), born, atol=1e-4, rtol=0
        )

    def test_partitions_are_the_serial_model(self):
        from deepmd_lr.nvalchemi import evaluate_partitioned, to_numpy

        model = build(2, "float32")
        model.enable_compression()
        system = frame(n_mol=16, box=14.0)
        coord, atype, cell = (t[0].cpu().numpy() for t in system)
        serial = evaluate(model, *system)
        born = TestBornTensor.born(model, *system).cpu().numpy()
        for n in (1, 2, 4):
            out = to_numpy(
                evaluate_partitioned(
                    model, coord, atype, cell, n_partitions=n,
                    devices=[DEVICE] * n, compute_virial=True, compute_born=True,
                )
            )  # fmt: skip
            assert out["energy"] == pytest.approx(float(serial["energy"]), abs=2e-5)
            for key in ("force", "virial", "charge", "dipole", "quadrupole"):
                reference = serial[key].detach().cpu().numpy().reshape(out[key].shape)
                assert np.abs(out[key] - reference).max() < 1e-4, (n, key)
            assert np.abs(out["born_charge"] - born).max() < 1e-4, n
