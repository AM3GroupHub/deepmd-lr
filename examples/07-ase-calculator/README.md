# 7 — The ASE calculator

`deepmd_lr.ase.LongRangeCalculator` is an ordinary ASE calculator for a
`dp_lr` model, long-range term included. Anything that takes an ASE calculator
takes it: single points, ASE's optimizers and MD integrators, NEB, phonons.

```python
from deepmd_lr.ase import LongRangeCalculator

atoms.calc = LongRangeCalculator("model.ckpt.pt")
atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress(), atoms.get_charges()
atoms.calc.results["energy_short_range"], atoms.calc.results["energy_long_range"]
```

[`ase_calculator.py`](ase_calculator.py) runs a single point, the Born
effective charges, a FIRE relaxation and a Langevin NVT run, on the water box
from [example 06](../06-md_nvalchemi/water_192.xyz):

```bash
python ase_calculator.py --model ../../path/to/model.ckpt.pt
```

## Arguments worth knowing

| argument | what it does |
|---|---|
| `compress=True` | the compressed DPA4C kernel: the same test-set accuracy, an order of magnitude faster on large systems. Needs deepmd-kit built with its CUDA operators ([compression](../../docs/compression.md)) |
| `efield=[Ex, Ey, Ez]` | a uniform field in V/Å: adds `F = Zᵀ E` to the forces and reports `results["born_charge"]`. A zero field gives the Born charges and moves nothing. Stress is withheld under a field |
| `eps_inf=` | reports charges and Born tensors screened by `sqrt(eps_inf)`; energies and field-free forces do not change ([Born charges](../../docs/born-charges.md)) |
| `charge_spin=(Q, 2S+1)` | total charge and spin multiplicity, for a charged or open-shell system |
| `long_range={...}` | reciprocal-space settings at load, e.g. `{"method": "pme", "slab_correction": True, "pbc": [True, True, False]}` for a slab |

## When to use it

The calculator evaluates the whole system on one device, and ASE moves the
atoms on the host between steps. That suits small systems and anything built
on ASE. For long MD, large systems or several GPUs, use the classes in
[example 06](../06-md_nvalchemi): the same model, stepped without leaving the
GPU.

A long-range model runs only through this package's own paths, this
calculator and the nvalchemi evaluator. `dp freeze`, `DeepPot`, `dp test` and
LAMMPS are refused for it, since none of them hands the cell to the long-range
term ([why](../../docs/md.md)).
