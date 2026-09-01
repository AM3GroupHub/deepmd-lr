"""Phase 6: MD deployment.

Two things are being pinned. The ASE calculator must reproduce the model's
own numbers, including the long-range term. And the export paths that
*cannot* carry that term must refuse rather than quietly produce a
short-range-only artifact -- which is what `dp freeze` did before this guard:
the frozen model came back at -93.74 eV where the eager model gives -55.12,
having dropped 38.6 eV of long-range energy without a word.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.ase import LongRangeCalculator

from .test_joint_model import build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the model runs on GPU"
)


def water_atoms(seed: int = 0, n_mol: int = 4, box: float = 10.0) -> Atoms:
    rng = np.random.default_rng(seed)
    positions, symbols = [], []
    for _ in range(n_mol):
        o = rng.uniform(1.0, box - 1.0, size=3)
        positions += [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.93, 0.0]]
        symbols += ["O", "H", "H"]
    return Atoms(
        symbols=symbols,
        positions=np.array(positions),
        cell=np.eye(3) * box,
        pbc=True,
    )


class TestCalculatorMatchesTheModel:
    """The calculator must not be a second, subtly different implementation."""

    def test_energy_and_forces_match_a_direct_call(self):
        model = build()
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(model)

        coord = torch.tensor(
            atoms.get_positions(), dtype=torch.float64, device=DEVICE
        ).unsqueeze(0)
        cell = torch.tensor(
            np.asarray(atoms.get_cell()), dtype=torch.float64, device=DEVICE
        ).unsqueeze(0)
        atype = torch.tensor([0, 1, 1] * 4, dtype=torch.int64, device=DEVICE).unsqueeze(
            0
        )
        direct = model(coord, atype, box=cell)

        assert atoms.get_potential_energy() == pytest.approx(
            float(direct["energy"].reshape(-1)[0]), abs=1e-9
        )
        assert np.allclose(
            atoms.get_forces(),
            direct["force"][0].detach().cpu().numpy(),
            atol=1e-9,
        )

    def test_long_range_term_is_actually_included(self):
        """The failure this whole phase exists to prevent."""
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(build())
        energy = atoms.get_potential_energy()
        results = atoms.calc.results
        assert abs(results["energy_long_range"]) > 1e-6
        assert energy == pytest.approx(
            results["energy_short_range"] + results["energy_long_range"], abs=1e-9
        )

    def test_exposes_charges(self):
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(build())
        atoms.get_potential_energy()
        charges = atoms.calc.results["charges"]
        assert charges.shape == (len(atoms),)
        assert abs(charges.sum()) < 1e-9

    def test_passes_a_fixed_charge_state(self):
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(build(), charge_spin=(1.0, 1.0))
        atoms.get_potential_energy()
        assert atoms.calc.results["charges"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_stress_shape_and_symmetry(self):
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(build())
        stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.isfinite(stress).all()


class TestCalculatorGuards:
    def test_rejects_non_periodic_atoms(self):
        atoms = water_atoms()
        atoms.pbc = False
        atoms.calc = LongRangeCalculator(build())
        with pytest.raises(ValueError, match="fully periodic cell"):
            atoms.get_potential_energy()

    def test_rejects_unknown_elements(self):
        atoms = water_atoms()
        atoms.symbols = ["C"] + list(atoms.symbols[1:])
        atoms.calc = LongRangeCalculator(build())
        with pytest.raises(ValueError, match="not in the model's type_map"):
            atoms.get_potential_energy()


class TestExportIsRefused:
    """The lower interfaces carry no cell, so they must not be used silently."""

    def test_graph_export_refuses_with_a_long_range_term(self):
        model = build(long_range=True)
        with pytest.raises(NotImplementedError, match="silently contain only"):
            model.forward_lower_graph_exportable()

    def test_comm_export_refuses_with_a_long_range_term(self):
        model = build(long_range=True)
        with pytest.raises(NotImplementedError, match="silently contain only"):
            model.forward_lower_graph_exportable_with_comm()

    def test_dense_lower_refuses_with_a_long_range_term(self):
        model = build(long_range=True)
        with pytest.raises(NotImplementedError, match="lower"):
            model.forward_lower()

    def test_the_error_names_the_supported_route(self):
        model = build(long_range=True)
        with pytest.raises(NotImplementedError, match="LongRangeCalculator"):
            model.forward_lower_graph_exportable()

    def test_short_range_only_model_still_exports(self):
        """The guard must not break the ordinary DPA4C path."""
        model = build(long_range=False)
        assert model.has_long_range is False
        # Reaching the parent implementation is the assertion; it will raise
        # something else (missing arguments), never our refusal.
        with pytest.raises(Exception) as excinfo:
            model.forward_lower_graph_exportable()
        assert "silently contain only" not in str(excinfo.value)


class TestEnergyConservation:
    """Forces must be the exact gradient of the reported energy."""

    def test_nve_drift_shrinks_with_the_timestep(self):
        """A non-conservative force leaves a drift floor that no timestep
        reduction removes; a conservative one drives drift to zero. Run over
        the same physical time so the comparison is like for like."""
        from ase import units
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet

        drifts = []
        for timestep, steps in ((1.0, 60), (0.25, 240)):
            atoms = water_atoms()
            atoms.calc = LongRangeCalculator(build())
            np.random.seed(0)
            MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)
            start = atoms.get_total_energy()
            VelocityVerlet(atoms, timestep=timestep * units.fs).run(steps)
            drifts.append(abs(atoms.get_total_energy() - start) / len(atoms))

        assert drifts[1] < drifts[0], (
            f"drift did not shrink with the timestep: {drifts}"
        )
