"""Phase 1: the plugin registers, builds, and changes nothing numerically.

The point of these tests is separation of concerns. Phase 1 adds no physics,
so `dp_lr` must be indistinguishable from the built-in standard model. Once
that is pinned, any numerical difference seen in a later phase can only come
from the long-range term being added, not from the plumbing.
"""

import copy
import json

import pytest
import torch

# Only `deepmd` is imported here, never `deepmd_lr`: these tests must fail if
# the entry point stops working, not silently pass because the test imported
# the plugin itself.
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.model.model import BaseModel
from deepmd.pt_expt.utils.env import DEVICE
from deepmd.utils.argcheck import model_args, normalize

BASE_MODEL_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 6.0,
        "channels": 16,
        "lmax": 2,
        "basis_type": "bessel",
        "n_radial": 8,
        "radial_modes": 2,
        "precision": "float64",
        "seed": 42,
    },
    "fitting_net": {
        "neuron": [32, 32],
        "resnet_dt": False,
        "activation_function": "silu",
        "precision": "float64",
        "seed": 42,
    },
}


def build(model_type: str | None):
    """Build a model from BASE_MODEL_CONFIG, optionally with an explicit type."""
    cfg = copy.deepcopy(BASE_MODEL_CONFIG)
    if model_type is not None:
        cfg["type"] = model_type
    return get_model(cfg)


def water_frame(dtype=torch.float64, seed: int = 0):
    """A small periodic 3-water box: coordinates, types, and cell.

    Everything is placed on ``deepmd.pt_expt.utils.env.DEVICE``. The pt_expt
    backend picks its own device at import and builds models there, so a test
    that forces CPU tensors into a CUDA-resident model fails on device
    mismatch rather than on anything this plugin does.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    box = 6.0
    coord = (torch.rand((9, 3), generator=g, dtype=dtype) * box).to(DEVICE)
    atype = torch.tensor([0, 1, 1] * 3, dtype=torch.int64, device=DEVICE)
    cell = (torch.eye(3, dtype=dtype) * box).to(DEVICE)
    return coord.unsqueeze(0), atype.unsqueeze(0), cell.unsqueeze(0)


class TestRegistration:
    """The entry point must wire the type into both registries DeePMD-kit reads."""

    def test_model_type_is_registered(self):
        assert "dp_lr" in BaseModel.get_plugins()

    def test_registered_class_is_ours(self):
        cls = BaseModel.get_class_by_type("dp_lr")
        assert cls.__module__ == "deepmd_lr.model.lr_model"
        assert cls.__name__ == "LongRangeEnergyModel"

    def test_input_schema_accepts_the_type(self):
        """Registration in the model registry alone is not enough: an input
        file is validated against the dargs schema first, so the type must
        appear there too or `dp train` rejects it before building anything."""
        choices = model_args().sub_variants["type"].choice_dict
        assert "dp_lr" in choices

    def test_full_input_normalizes(self):
        """Exercise the real normalization path a training run would take."""
        data = {
            "model": {"type": "dp_lr", **copy.deepcopy(BASE_MODEL_CONFIG)},
            "learning_rate": {"type": "exp", "start_lr": 1e-3, "stop_lr": 1e-5},
            "loss": {"type": "ener"},
            "training": {
                "training_data": {"systems": ["./data"], "batch_size": 1},
                "numb_steps": 1,
                "seed": 42,
            },
        }
        normalized = normalize(data)
        assert normalized["model"]["type"] == "dp_lr"


class TestConstruction:
    def test_get_model_returns_our_class(self):
        model = build("dp_lr")
        assert type(model).__name__ == "LongRangeEnergyModel"

    def test_rejects_non_energy_fitting(self):
        """A long-range electrostatic term is only defined for energies."""
        cfg = copy.deepcopy(BASE_MODEL_CONFIG)
        cfg["type"] = "dp_lr"
        cfg["fitting_net"] = {"type": "dipole", "neuron": [16], "seed": 42}
        with pytest.raises(ValueError, match="energy model"):
            get_model(cfg)

    def test_model_def_script_is_recorded(self):
        """Checkpoint metadata must round-trip the type that built the model."""
        model = build("dp_lr")
        assert json.loads(model.model_def_script)["type"] == "dp_lr"


class TestEquivalenceToStandardModel:
    """Phase 1 adds no physics, so dp_lr must be numerically a no-op."""

    def test_energy_and_force_match_standard_model(self):
        """dp_lr must be indistinguishable from the built-in standard model.

        The comparison is against the GPU's own nondeterminism floor rather
        than against bitwise equality. Segment/scatter reductions in the
        descriptor sum in a nondeterministic order, so evaluating one model
        twice on identical inputs already differs by ~1e-15; demanding
        bitwise equality would fail for reasons that have nothing to do with
        this plugin. Measuring that floor in-test and requiring the
        cross-model difference to sit at or below it is the stronger claim
        anyway: it says the two models differ by no more than one model
        differs from itself.
        """
        lr = build("dp_lr")
        std = build(None)
        # Same config and seed, but two constructions are not guaranteed to
        # draw identical initial weights; copy them so this isolates model
        # wiring rather than the initializer.
        std.load_state_dict(lr.state_dict())
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                lr.state_dict().values(), std.state_dict().values(), strict=True
            )
        )

        lr.eval()
        std.eval()
        coord, atype, cell = water_frame()
        out_lr = lr(coord, atype, box=cell)
        out_std = std(coord, atype, box=cell)
        out_lr_again = lr(coord, atype, box=cell)

        assert set(out_lr) == set(out_std)
        for key in ("energy", "atom_energy", "force", "virial"):
            noise_floor = (out_lr[key] - out_lr_again[key]).abs().max()
            cross = (out_lr[key] - out_std[key]).abs().max()
            tol = max(float(noise_floor), 1e-12)
            assert float(cross) <= tol, (
                f"{key}: dp_lr and standard differ by {float(cross):.3e}, "
                f"above the {tol:.3e} nondeterminism floor"
            )

    def test_backward_is_wired(self):
        """Gradients must reach the parameters -- a silently detached graph
        would still pass the forward-equivalence test above."""
        model = build("dp_lr")
        coord, atype, cell = water_frame()
        model(coord, atype, box=cell)["energy"].sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads, "model exposes no trainable parameters"
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                   for g in grads)


class TestSerializationRoundTrip:
    def test_round_trip_returns_our_class(self):
        """Without the type stamp, a checkpoint would deserialize back to a
        plain EnergyModel and silently drop the long-range term in Phase 4."""
        model = build("dp_lr")
        data = model.serialize()
        assert data["type"] == "dp_lr"
        restored = BaseModel.deserialize(data)
        assert type(restored).__name__ == "LongRangeEnergyModel"

    def test_round_trip_preserves_predictions(self):
        """Weights and buffers must survive the round trip.

        Compared against the same nondeterminism floor as the equivalence
        test rather than bitwise: two evaluations of one model already differ
        at ~1e-15 because of nondeterministic reduction order on GPU.
        """
        model = build("dp_lr")
        model.eval()
        coord, atype, cell = water_frame()
        before = model(coord, atype, box=cell)["energy"]
        before_again = model(coord, atype, box=cell)["energy"]
        restored = BaseModel.deserialize(model.serialize())
        restored.eval()
        after = restored(coord, atype, box=cell)["energy"]

        noise_floor = (before - before_again).abs().max()
        tol = max(float(noise_floor), 1e-12)
        assert float((before - after).abs().max()) <= tol
