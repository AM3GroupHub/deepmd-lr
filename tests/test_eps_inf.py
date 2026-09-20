"""Electronic screening: reported charges and Born tensors carry sqrt(eps_inf).

The model's Coulomb energy is ``q_i q_j / (4 pi eps_0 r)``. In a medium with
high-frequency dielectric constant ``eps_inf`` the same energy is
``q~_i q~_j / (4 pi eps_0 eps_inf r)``, so the physical charges are
``q~ = sqrt(eps_inf) q`` and the Born tensors, derivatives of those charges,
scale alike. The invariant every test here pins: energies and field-free
forces are the model's own and do not move; charges, Born tensors and the
field force built from them scale by exactly ``sqrt(eps_inf)``.
"""

import copy
import math

import numpy as np
import pytest
import torch
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils.env import DEVICE

from deepmd_lr.born import born_charges, screening_factor

from .test_born import build, molecule, pieces
from .test_joint_model import DESCRIPTOR, FITTING, frame

DT = torch.float64
EPS = 4.0  # sqrt is exactly 2, so the assertions below are exact ratios


class TestScreeningFactor:
    def test_vacuum_is_one(self):
        assert screening_factor(1.0) == 1.0
        assert screening_factor(1) == 1.0

    def test_is_the_square_root(self):
        assert screening_factor(EPS) == 2.0
        assert screening_factor(1.78) == pytest.approx(math.sqrt(1.78))

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
    def test_refuses_non_positive_or_non_finite(self, bad):
        with pytest.raises(ValueError, match="positive and finite"):
            screening_factor(bad)

    @pytest.mark.parametrize("bad", ["4", None, [4.0], True])
    def test_refuses_non_numbers(self, bad):
        with pytest.raises(TypeError):
            screening_factor(bad)


class TestBornChargesFunction:
    def test_scales_the_whole_tensor(self):
        model = build(isolated=True)
        coord, atype = molecule()
        out, raw, cap = pieces(model, coord, atype)
        args = (
            out["charge"].detach(),
            raw,
            cap["edge_vec"],
            cap["edge_index"],
            cap["edge_mask"],
        )
        plain = born_charges(*args)
        screened = born_charges(*args, eps_inf=EPS)
        torch.testing.assert_close(screened, 2.0 * plain)


class TestModel:
    def test_energy_and_forces_do_not_move_but_charges_and_born_do(self):
        model = build()
        coord, atype, cell = frame(n_mol=8)
        vacuum = model(coord, atype, box=cell)
        model.eps_inf = EPS
        medium = model(coord, atype, box=cell)

        torch.testing.assert_close(medium["energy"], vacuum["energy"])
        torch.testing.assert_close(medium["force"], vacuum["force"])
        torch.testing.assert_close(medium["virial"], vacuum["virial"])
        torch.testing.assert_close(medium["charge"], 2.0 * vacuum["charge"])
        torch.testing.assert_close(medium["born_charge"], 2.0 * vacuum["born_charge"])

    def test_default_is_vacuum(self):
        assert build().eps_inf == 1.0

    def test_acoustic_sum_rule_holds_for_the_screened_total(self):
        model = build()
        model.eps_inf = EPS
        coord, atype, cell = frame(n_mol=8)
        out = model(coord, atype, box=cell)
        z = out["born_charge"][0]
        total = out["charge"].sum()
        eye = torch.eye(3, dtype=DT, device=DEVICE)
        assert torch.allclose(z.sum(0), total * eye, atol=1e-10)


class TestCalculator:
    def _atoms_and_model(self):
        from ase import Atoms

        model = build()
        coord, atype, cell = frame(n_mol=8)
        numbers = [8 if t == 0 else 1 for t in atype[0].cpu().numpy()]
        atoms = Atoms(
            numbers=numbers,
            positions=coord[0].cpu().numpy(),
            cell=cell[0].cpu().numpy(),
            pbc=True,
        )
        return model, atoms

    def test_charges_born_and_field_force_scale(self):
        from deepmd_lr.ase import LongRangeCalculator

        model, atoms = self._atoms_and_model()
        e = np.array([0.0, 0.0, 0.05])
        vac = atoms.copy()
        vac.calc = LongRangeCalculator(model, efield=e)
        f_vac = vac.get_forces()
        q_vac = vac.calc.results["charges"]
        z_vac = vac.calc.results["born_charge"]

        med = atoms.copy()
        med.calc = LongRangeCalculator(model, efield=e, eps_inf=EPS)
        f_med = med.get_forces()

        np.testing.assert_allclose(med.calc.results["charges"], 2.0 * q_vac)
        np.testing.assert_allclose(med.calc.results["born_charge"], 2.0 * z_vac)
        # The force difference is exactly one more copy of the vacuum field
        # term: F_med = F_0 + 2 Z.T E, F_vac = F_0 + Z.T E.
        np.testing.assert_allclose(
            f_med - f_vac, np.einsum("iab,a->ib", z_vac, e), atol=1e-10
        )
        assert med.get_potential_energy() == pytest.approx(vac.get_potential_energy())

    def test_without_a_field_only_the_reported_charges_change(self):
        from deepmd_lr.ase import LongRangeCalculator

        model, atoms = self._atoms_and_model()
        vac = atoms.copy()
        vac.calc = LongRangeCalculator(model)
        med = atoms.copy()
        med.calc = LongRangeCalculator(model, eps_inf=EPS)
        np.testing.assert_allclose(med.get_forces(), vac.get_forces())
        np.testing.assert_allclose(med.get_stress(), vac.get_stress())
        np.testing.assert_allclose(
            med.calc.results["charges"], 2.0 * vac.calc.results["charges"]
        )

    def test_invalid_value_is_refused_at_construction(self):
        from deepmd_lr.ase import LongRangeCalculator

        model, _atoms = self._atoms_and_model()
        with pytest.raises(ValueError, match="positive and finite"):
            LongRangeCalculator(model, eps_inf=0.0)

    def test_short_range_model_accepts_the_argument(self):
        """A model without charges has nothing to scale, but must not choke."""
        from deepmd_lr.ase import LongRangeCalculator

        cfg = {
            "type": "dp_lr",
            "type_map": ["O", "H"],
            "descriptor": copy.deepcopy(DESCRIPTOR),
            "fitting_net": copy.deepcopy(FITTING),
        }
        model = get_model(cfg).eval()
        _model, atoms = self._atoms_and_model()
        atoms.calc = LongRangeCalculator(model, eps_inf=EPS)
        assert np.isfinite(atoms.get_potential_energy())
        assert "charges" not in atoms.calc.results


class TestPartitioned:
    def _setup(self):
        model = build()
        coord, atype, cell = frame(n_mol=8)
        return (
            model,
            coord[0].cpu().numpy(),
            atype[0].cpu().numpy(),
            cell[0].cpu().numpy(),
            coord,
            atype,
            cell,
        )

    def test_evaluate_partitioned_matches_the_serial_screened_tensor(self):
        from deepmd_lr.nvalchemi import evaluate_partitioned

        model, c, a, cl, coord, atype, cell = self._setup()
        model.eps_inf = EPS
        serial = model(coord, atype, box=cell)
        model.eps_inf = 1.0
        for n_part in (1, 2, 4):
            out = evaluate_partitioned(
                model,
                c,
                a,
                cl,
                n_partitions=n_part,
                devices=[DEVICE] * n_part,
                compute_born=True,
                eps_inf=EPS,
            )
            np.testing.assert_allclose(
                out["born_charge"],
                serial["born_charge"][0].detach().cpu().numpy(),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                out["charge"], serial["charge"][0].detach().cpu().numpy(), atol=1e-10
            )
            np.testing.assert_allclose(
                out["force"], serial["force"][0].detach().cpu().numpy(), atol=1e-8
            )
            assert out["energy"] == pytest.approx(float(serial["energy"][0]))

    def test_evaluator_field_force_scales(self):
        from deepmd_lr.nvalchemi import DistributedEvaluator

        model, c, a, cl, *_ = self._setup()
        e = [0.0, 0.0, 0.05]
        vac = DistributedEvaluator(model, a, cl, devices=[DEVICE, DEVICE], efield=e)
        e_vac, f_vac = vac(torch.tensor(c, dtype=DT, device=DEVICE))
        f_vac = f_vac.clone()
        field_vac = vac.results["field_force"].copy()

        med = DistributedEvaluator(
            model, a, cl, devices=[DEVICE, DEVICE], efield=e, eps_inf=EPS
        )
        e_med, f_med = med(torch.tensor(c, dtype=DT, device=DEVICE))

        assert e_med == pytest.approx(e_vac)
        np.testing.assert_allclose(med.results["field_force"], 2.0 * field_vac)
        np.testing.assert_allclose((f_med - f_vac).cpu().numpy(), field_vac, atol=1e-10)
        np.testing.assert_allclose(med.results["charge"], 2.0 * vac.results["charge"])

    def test_evaluator_refuses_an_invalid_value(self):
        from deepmd_lr.nvalchemi import DistributedEvaluator

        model, _c, a, cl, *_ = self._setup()
        with pytest.raises(ValueError, match="positive and finite"):
            DistributedEvaluator(model, a, cl, devices=[DEVICE], eps_inf=-2.0)
