"""Phase 2: the charge head and the total-charge projection.

Two things are being pinned here. The projection must enforce
sum(q) == Q_total exactly while staying differentiable, because the charges
are trained *through* the PME energy. And the charges themselves must be
physically well-behaved: invariant under rotation and translation of the
frame, and equivariant under permutation of the atoms.

Only `deepmd` and `deepmd_lr.fitting` are imported for the model-facing
pieces; the registration tests import `deepmd` alone so they fail if the
entry point breaks.
"""

import numpy as np
import pytest
import torch
from deepmd.dpmodel.utils.neighbor_graph import build_neighbor_graph
from deepmd.pt_expt.descriptor import BaseDescriptor
from deepmd.pt_expt.fitting import BaseFitting
from deepmd.pt_expt.utils.env import DEVICE
from deepmd.utils.argcheck import fitting_variant_type_args

from deepmd_lr.fitting.charge import project_to_total_charge

RCUT = 6.0
TYPE_MAP = ["O", "H"]


def make_descriptor(seed: int = 42):
    """A small DPA4C descriptor, built through the same registry deepmd uses."""
    return BaseDescriptor.get_class_by_type("dpa4c")(
        rcut=RCUT,
        channels=16,
        lmax=2,
        basis_type="bessel",
        n_radial=8,
        radial_modes=2,
        precision="float64",
        seed=seed,
        ntypes=len(TYPE_MAP),
        type_map=list(TYPE_MAP),
    )


def make_charge_head(descriptor, seed: int = 42):
    return BaseFitting.get_class_by_type("charge")(
        ntypes=len(TYPE_MAP),
        dim_descrpt=descriptor.get_dim_out(),
        neuron=[32, 32],
        activation_function="silu",
        precision="float64",
        mixed_types=descriptor.mixed_types(),
        type_map=list(TYPE_MAP),
        seed=seed,
    )


def water_box(seed: int = 0, box: float = 8.0, n_mol: int = 4):
    """A periodic water box: (1, nloc, 3) coords, (1, nloc) types, (1, 3, 3) cell."""
    rng = np.random.default_rng(seed)
    coords = []
    for _ in range(n_mol):
        o = rng.uniform(0.0, box, size=3)
        coords.append(o)
        coords.append(o + np.array([0.96, 0.0, 0.0]))
        coords.append(o + np.array([-0.24, 0.93, 0.0]))
    coord = torch.tensor(np.array(coords), dtype=torch.float64, device=DEVICE)
    atype = torch.tensor([0, 1, 1] * n_mol, dtype=torch.int64, device=DEVICE)
    cell = torch.eye(3, dtype=torch.float64, device=DEVICE) * box
    return coord.unsqueeze(0), atype.unsqueeze(0), cell.unsqueeze(0)


def raw_charges(descriptor, head, coord, atype, cell):
    """Run descriptor + charge head on the graph route, returning (nf, nloc)."""
    graph = build_neighbor_graph(coord, atype, cell, RCUT)
    atype_flat = atype.reshape(-1)
    gg, rot_mat = descriptor.call_graph(
        graph,
        atype_flat,
        type_embedding=descriptor.graph_type_embedding_table(),
    )
    out = head.call_graph(gg, atype_flat, gr=rot_mat)
    return out["charge"].reshape(1, -1)


class TestProjection:
    """The constraint must hold exactly, and hold up under autograd."""

    @pytest.mark.parametrize("total", [0.0, 1.0, -2.0, 0.5])
    def test_sum_matches_target(self, total):
        raw = torch.randn(3, 7, dtype=torch.float64)
        q = project_to_total_charge(raw, total)
        assert torch.allclose(
            q.sum(-1), torch.full((3,), total, dtype=torch.float64), atol=1e-12
        )

    def test_per_frame_targets(self):
        """Each frame gets its own total, not a batch-wide one."""
        raw = torch.randn(3, 7, dtype=torch.float64)
        totals = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)
        q = project_to_total_charge(raw, totals)
        assert torch.allclose(q.sum(-1), totals, atol=1e-12)

    def test_mask_excludes_padding(self):
        """Padding atoms take no share of the deficit and come back as zero."""
        raw = torch.randn(2, 6, dtype=torch.float64)
        mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
        )
        q = project_to_total_charge(raw, 1.0, mask=mask)
        assert torch.allclose(q[~mask], torch.zeros(5, dtype=torch.float64))
        assert torch.allclose(
            q.sum(-1), torch.ones(2, dtype=torch.float64), atol=1e-12
        )

    def test_shift_is_uniform(self):
        """The correction is a constant shift, so charge *differences* survive."""
        raw = torch.randn(1, 5, dtype=torch.float64)
        q = project_to_total_charge(raw, 3.0)
        shift = q - raw
        assert torch.allclose(shift, shift[0, 0].expand_as(shift), atol=1e-12)

    def test_permutation_equivariant(self):
        raw = torch.randn(1, 6, dtype=torch.float64)
        perm = torch.randperm(6)
        a = project_to_total_charge(raw, 1.0)[:, perm]
        b = project_to_total_charge(raw[:, perm], 1.0)
        assert torch.allclose(a, b, atol=1e-14)

    def test_is_differentiable(self):
        """Gradients must flow: the charges are trained through the PME energy."""
        raw = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda x: project_to_total_charge(x, 1.0), (raw,)
        )

    def test_no_nan_for_fully_masked_frame(self):
        """An all-padding frame must not poison the batch's gradients."""
        raw = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
        q = project_to_total_charge(raw, 1.0, mask=mask)
        assert torch.isfinite(q).all()
        q.sum().backward()
        assert torch.isfinite(raw.grad).all()


class TestRegistration:
    def test_fitting_type_is_registered(self):
        assert "charge" in BaseFitting.get_plugins()

    def test_fitting_schema_accepts_the_type(self):
        assert "charge" in fitting_variant_type_args().choice_dict


class TestChargeHead:
    def test_output_def(self):
        d = make_descriptor()
        head = make_charge_head(d)
        out = head.output_def()["charge"]
        assert out.shape == [1]
        assert out.reducible

    def test_produces_one_charge_per_atom(self):
        d = make_descriptor()
        head = make_charge_head(d)
        coord, atype, cell = water_box()
        q = raw_charges(d, head, coord, atype, cell)
        assert q.shape == (1, coord.shape[1])
        assert torch.isfinite(q).all()

    def test_serialization_round_trip(self):
        d = make_descriptor()
        head = make_charge_head(d)
        data = head.serialize()
        assert data["type"] == "charge"
        restored = BaseFitting.deserialize(data)
        assert type(restored).__name__ == "ChargeFittingNet"

        coord, atype, cell = water_box()
        before = raw_charges(d, head, coord, atype, cell)
        after = raw_charges(d, restored.to(DEVICE), coord, atype, cell)
        assert torch.allclose(before, after, atol=1e-12)


class TestPhysicalSymmetries:
    """Charges come from an invariant descriptor; verify that end to end.

    These run the real DPA4C descriptor rather than a synthetic feature
    vector, so they would catch a head that accidentally consumed an
    equivariant quantity (e.g. `gr`) alongside the invariant descriptor.
    """

    def _charges(self, coord, atype, cell, seed=7):
        d = make_descriptor(seed)
        head = make_charge_head(d, seed)
        return project_to_total_charge(raw_charges(d, head, coord, atype, cell), 0.0)

    def test_charges_are_not_degenerate(self):
        """Guard against the invariance tests passing vacuously.

        A head that returned a constant would satisfy every symmetry test in
        this class. Requiring real spread across atoms means those tests are
        actually constraining something.
        """
        coord, atype, cell = water_box()
        q = self._charges(coord, atype, cell)
        assert float(q.std()) > 1e-3, "charge head output is effectively constant"

    def test_charges_respond_to_geometry(self):
        """The whole point of a learned head: charges must see the environment.

        Fixed per-element charges would pass every other test here.
        """
        coord, atype, cell = water_box()
        moved = coord.clone()
        moved[0, 0, 0] = moved[0, 0, 0] + 0.35
        a = self._charges(coord, atype, cell)
        b = self._charges(moved, atype, cell)
        assert float((a - b).abs().max()) > 1e-4

    def test_translation_invariant(self):
        coord, atype, cell = water_box()
        shift = torch.tensor([1.3, -0.7, 2.1], dtype=torch.float64, device=DEVICE)
        a = self._charges(coord, atype, cell)
        b = self._charges(coord + shift, atype, cell)
        assert torch.allclose(a, b, atol=1e-10)

    def test_rotation_invariant(self):
        """Rotate coordinates and cell together; charges are scalars."""
        coord, atype, cell = water_box()
        theta = 0.7
        c, s = np.cos(theta), np.sin(theta)
        rot = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
            device=DEVICE,
        )
        a = self._charges(coord, atype, cell)
        b = self._charges(coord @ rot.T, atype, cell @ rot.T)
        assert torch.allclose(a, b, atol=1e-10)

    def test_permutation_equivariant(self):
        """Relabelling atoms permutes the charges, it does not change them."""
        coord, atype, cell = water_box()
        perm = torch.randperm(coord.shape[1], device=DEVICE)
        a = self._charges(coord, atype, cell)[:, perm]
        b = self._charges(coord[:, perm], atype[:, perm], cell)
        assert torch.allclose(a, b, atol=1e-10)

    def test_periodic_image_invariant(self):
        """Wrapping an atom by a lattice vector must not change any charge."""
        coord, atype, cell = water_box()
        moved = coord.clone()
        moved[0, 0] = moved[0, 0] + torch.diagonal(cell[0])
        a = self._charges(coord, atype, cell)
        b = self._charges(moved, atype, cell)
        assert torch.allclose(a, b, atol=1e-10)
