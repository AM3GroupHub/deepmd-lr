"""Phase 6: MD deployment.

Two things are being pinned. The ASE calculator must reproduce the model's
own numbers, including the long-range term. And the export paths that
*cannot* carry that term must refuse rather than quietly produce a
short-range-only artifact -- which is what `dp freeze` did before this guard:
the frozen model came back at -93.74 eV where the eager model gives -55.12,
having dropped 38.6 eV of long-range energy without a word.
"""

import copy

import numpy as np
import pytest
import torch
from ase import Atoms
from deepmd.pt_expt.train.wrapper import ModelWrapper
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.ase import LongRangeCalculator, load_model

from .test_joint_model import build
from .test_multi_task import TestThreeBranchInterfaceBulkMolecule as ThreeBranch

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
        """Voigt shape, finite values. The Voigt form has no room for an
        antisymmetric part, so symmetry itself is pinned on the virial, by
        `test_virial_is_symmetric` in test_joint_model and test_nvalchemi."""
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(build())
        stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.isfinite(stress).all()


class TestCalculatorGuards:
    def test_non_periodic_atoms_take_the_isolated_sum(self):
        """No periodic axis: a periodic model's term becomes the isolated sum
        at its alpha, the same as asking for method 'isolated' outright."""
        atoms = water_atoms()
        atoms.pbc = False
        model = build()
        atoms.calc = LongRangeCalculator(model)
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        assert model.long_range_energy.method == "isolated"
        atoms.calc = LongRangeCalculator(
            build(), long_range={"method": "isolated", "isolated_kernel": "dense"}
        )
        assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-10)
        np.testing.assert_allclose(atoms.get_forces(), forces, atol=1e-10)

    def test_accepts_non_periodic_atoms_for_a_short_range_model(self):
        """The guard used to fire regardless of whether there was a
        long-range term at all, which refused a perfectly ordinary
        short-range model on a molecule."""
        atoms = water_atoms()
        atoms.pbc = False
        atoms.calc = LongRangeCalculator(build(long_range=False))
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()

    def test_accepts_non_periodic_atoms_for_the_isolated_term(self):
        from .test_isolated_term import isolated_model

        atoms = water_atoms()
        atoms.pbc = False
        atoms.calc = LongRangeCalculator(isolated_model())
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()

    def test_accepts_a_slab_with_one_open_axis(self):
        """pbc=[True, True, False] is exactly the configuration
        `slab_correction` exists for; the guard used to reject it."""
        atoms = water_atoms()
        atoms.set_cell([10.0, 10.0, 30.0])  # a real slab: 20 A of vacuum over it
        atoms.pbc = [True, True, False]
        from deepmd_lr.lr.term import LongRangeEnergy

        model = build()
        settings = model.long_range_energy.serialize()
        settings.update(slab_correction=True, pbc=[True, True, False])
        model.long_range_energy = LongRangeEnergy(**settings)
        atoms.calc = LongRangeCalculator(model)
        assert np.isfinite(atoms.get_potential_energy())

    def test_each_frames_pbc_picks_its_sum(self):
        """One calculator, three frames: the slab sum where pbc opens z, the
        3D sum where it does not, matching calculators fixed to each."""
        from deepmd_lr.lr.term import LongRangeEnergy

        bulk = water_atoms()
        slab = water_atoms()
        slab.set_cell([10.0, 10.0, 30.0])
        slab.pbc = [True, True, False]

        def fixed(atoms, slab_mode):
            model = build()
            settings = model.long_range_energy.serialize()
            if slab_mode:
                settings.update(slab_correction=True, pbc=[True, True, False])
            settings.update(mesh_dimensions=[15, 15, 24 if slab_mode else 15])
            model.long_range_energy = LongRangeEnergy(**settings)
            a = atoms.copy()
            a.calc = LongRangeCalculator(model)
            return a.get_potential_energy()

        calc = LongRangeCalculator(build())
        for atoms, slab_mode in ((bulk, False), (slab, True), (bulk, False)):
            a = atoms.copy()
            a.calc = calc
            energy = a.get_potential_energy()
            assert calc.model.long_range_energy.slab_correction is slab_mode
            assert energy == pytest.approx(fixed(atoms, slab_mode), abs=1e-9)

    def test_each_frame_gets_the_mesh_its_cell_needs(self):
        calc = LongRangeCalculator(build())
        for box, mesh in ((10.0, (15, 15, 15)), (40.0, (32, 32, 32))):
            atoms = water_atoms(box=box)
            atoms.calc = calc
            atoms.get_potential_energy()
            assert calc.model.long_range_energy.mesh_dimensions == mesh

    def test_a_slab_with_less_vacuum_than_the_cutoff_is_refused(self):
        """The calculator hands the model the whole cell, so across a thin
        gap the descriptor would see the next image's other face."""
        atoms = water_atoms()
        atoms.pbc = [True, True, False]  # a 10 A cell with atoms up to its faces
        atoms.calc = LongRangeCalculator(build())
        with pytest.raises(ValueError, match="less than the cutoff"):
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
        from ase.md.velocitydistribution import thermalize_momenta
        from ase.md.verlet import VelocityVerlet

        drifts = []
        for timestep, steps in ((1.0, 60), (0.25, 240)):
            atoms = water_atoms()
            atoms.calc = LongRangeCalculator(build())
            np.random.seed(0)
            thermalize_momenta(atoms, temperature_K=300.0)
            start = atoms.get_total_energy()
            VelocityVerlet(atoms, timestep=timestep * units.fs).run(steps)
            drifts.append(abs(atoms.get_total_energy() - start) / len(atoms))

        assert drifts[1] < drifts[0], (
            f"drift did not shrink with the timestep: {drifts}"
        )


class TestPlainEnergyModel:
    """A model trained without `type: dp_lr` at all.

    This is not the same as a `dp_lr` model with no `long_range` block: that
    one still carries `has_long_range = False`. Delete the `type` key and
    DeePMD-kit builds a plain `EnergyModel`, which has no such attribute --
    and that is exactly what the README's baseline-twin recipe produces
    ("delete `type: dp_lr` and the `long_range` block, change nothing else").

    The calculator used to read `model.has_long_range` unguarded and died with
    `AttributeError` on it, so the documented way to build a baseline could
    not be loaded back. The distributed driver had always used `getattr`.
    """

    @staticmethod
    def _standard_model():
        import copy

        from deepmd.pt_expt.model.get_model import get_model

        from .test_joint_model import DESCRIPTOR, FITTING

        model = get_model(
            {
                "type_map": ["O", "H"],
                "descriptor": copy.deepcopy(DESCRIPTOR),
                "fitting_net": copy.deepcopy(FITTING),
            }
        )
        assert not hasattr(model, "has_long_range"), (
            "fixture no longer reproduces the case: this model does carry the "
            "attribute, so it cannot regress the AttributeError"
        )
        return model.to(DEVICE).eval()

    def test_runs_a_model_that_has_no_long_range_attribute(self):
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(self._standard_model())
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()

    def test_reports_no_long_range_energy(self):
        atoms = water_atoms()
        atoms.calc = LongRangeCalculator(self._standard_model())
        atoms.get_potential_energy()
        assert "energy_long_range" not in atoms.calc.results
        assert "charges" not in atoms.calc.results

    def test_a_field_is_refused_rather_than_crashing(self):
        with pytest.raises(ValueError, match="no `long_range`"):
            LongRangeCalculator(self._standard_model(), efield=[0.0, 0.0, 0.1])


# --- multi-head checkpoints through the calculator ---------------------------

@pytest.mark.parametrize("name", ["interface", "bulk", "molecule"])
def test_checkpoint_head_to_ase(tmp_path, name):
    fixture = ThreeBranch()
    cfg, _ = fixture.config()
    models = fixture.build()
    cfg["model"]["model_dict"][name]["model_branch_alias"] = ["selected"]
    state = ModelWrapper(models, model_params=cfg["model"]).state_dict()
    path = str(tmp_path / "multi.pt")
    torch.save({"model": state}, path)
    atoms = water_atoms(n_mol=2)
    if name == "interface":
        atoms.set_cell([10.0, 10.0, 30.0])  # a slab needs vacuum over it
    atoms.pbc = {"interface": [True, True, False], "bulk": True, "molecule": False}[
        name
    ]
    atoms.calc = LongRangeCalculator(models[name])
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = LongRangeCalculator(path, head="selected")
    assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-8)
    np.testing.assert_allclose(atoms.get_forces(), forces, atol=1e-8)
    with pytest.raises(ValueError, match="Select a checkpoint head"):
        load_model(path)
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        load_model(path, head="absent")
    broken = copy.deepcopy(state)
    key = next(k for k in broken if k.startswith(f"model.{name}.charge_fitting."))
    del broken[key]
    torch.save({"model": broken}, path)
    with pytest.raises(ValueError, match="weights do not match"):
        load_model(path, head=name)


@pytest.mark.parametrize(
    "name,pbc,message",
    [
        ("bulk", [True, False, False], "wire"),
        ("molecule", [True, True, True], "sums no periodic images"),
    ],
)
def test_reject_boundary_mismatch(name, pbc, message):
    """What no periodic sum can take: a wire, and the isolated term in a cell."""
    model = ThreeBranch().build()[name]
    atoms = water_atoms(n_mol=1)
    atoms.pbc = pbc
    atoms.calc = LongRangeCalculator(model)
    with pytest.raises(ValueError, match=message):
        atoms.get_potential_energy()


@pytest.mark.parametrize("pbc", [[True, True, True], [False, True, True]])
def test_a_slab_branch_follows_the_frames_pbc(pbc):
    """The term holds no weights, so the frame's pbc picks the sum: the
    interface branch runs a bulk cell, or a slab open along x."""
    model = ThreeBranch().build()["interface"]
    atoms = water_atoms(n_mol=1)
    if not all(pbc):
        axis = pbc.index(False)
        cell = atoms.cell.array.copy()
        cell[axis, axis] += 20.0  # vacuum along the open axis
        atoms.set_cell(cell)
    atoms.pbc = pbc
    atoms.calc = LongRangeCalculator(model)
    assert np.isfinite(atoms.get_potential_energy())
    term = model.long_range_energy
    assert term.slab_correction is (not all(pbc))
    assert list(term.pbc) == pbc


def test_short_range_partial_pbc_is_refused():
    atoms = water_atoms(n_mol=1)
    atoms.pbc = [True, False, False]
    atoms.calc = LongRangeCalculator(build(long_range=False))
    with pytest.raises(ValueError, match="Partial pbc"):
        atoms.get_potential_energy()
