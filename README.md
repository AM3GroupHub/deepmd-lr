# deepmd-lr

Long-range electrostatics for **DPA4C**, computed with NVIDIA's PME
implementation (`nvalchemi-toolkit-ops`) and trained jointly with the
short-range model.

This is a **plugin package, not a fork** of DeePMD-kit. It registers itself
through the `deepmd.pt_expt` entry point that DeePMD-kit already exposes, so
it works against a stock `pip install deepmd-kit==3.2.0`.

See [PLAN.md](PLAN.md) for the design, the phase breakdown, and the
environment setup.

## Status

| Phase | | |
|---|---|---|
| 0 | Environment + PME validation | **done** |
| 1 | Plugin skeleton (`dp_lr` model type, no long-range term yet) | **done** |
| 2 | Charge head (`charge` fitting type + total-charge projection) | **done** |
| 3 | PME term (`PMEEnergy`, eV units, guards) | **done** |
| 4 | Joint model (`long_range` block, conservative forces) | **done** |
| 5 | Training + validation (benchmark + control) | **done** |
| 6 | MD via nvalchemi/ASE | not started |

## Install

Follow the environment section of [PLAN.md](PLAN.md) first, then:

```bash
pip install -e . --no-deps
```

`--no-deps` is deliberate: the dependency pins are recorded in
`pyproject.toml` for documentation, but the environment is built explicitly
(a CUDA-13 aarch64 `torch` from the PyTorch index, which pip should not try
to re-resolve).

## Usage

Select the model type in the input file:

```json
{
  "model": {
    "type": "dp_lr",
    "type_map": ["O", "H"],
    "descriptor": { "type": "dpa4c", "rcut": 6.0 },
    "fitting_net": { "neuron": [192, 192, 192] }
  }
}
```

Add a `long_range` block to switch the electrostatics on:

```json
{
  "model": {
    "type": "dp_lr",
    "type_map": ["O", "H"],
    "descriptor": { "type": "dpa4c", "rcut": 6.0 },
    "fitting_net": { "neuron": [192, 192, 192] },
    "long_range": {
      "charge_net": { "neuron": [64, 64] },
      "pme": { "rcut": 5.0, "alpha": 0.35, "mesh_dimensions": [16, 16, 16] }
    }
  }
}
```

```bash
dp --pt-expt train input.json
```

Without the `long_range` block the model is exactly the built-in standard
model. Prefer `mesh_dimensions` over `mesh_spacing` whenever the cell can
change — see PLAN.md, Phase 4.

## Tests

```bash
pytest tests/ -q
```

- `tests/test_pme_physics.py` — PME against the NaCl Madelung constant,
  `gradcheck`/`gradgradcheck`, and a guard for the `spline_order >= 7`
  silent-failure mode (see PLAN.md, Phase 0).
- `tests/test_plugin_registration.py` — entry-point registration, input-schema
  acceptance, equivalence to the standard model, and serialization round-trip.
- `tests/test_charge_head.py` — the total-charge projection (exactness,
  masking, differentiability) and the charge head's physical symmetries
  through a real DPA4C descriptor.
- `tests/test_pme_term.py` — the eV unit conversion (against analytic
  Coulomb and Madelung), boundary-condition guards, batching, masking, and
  gradients including the double backward that force training needs.
- `tests/test_joint_model.py` — the combined model: force and virial against
  finite differences of the total energy, energy decomposition, both calling
  conventions, and serialization of the long-range term.

The tests import only `deepmd`, never `deepmd_lr`, so they fail if the entry
point stops working rather than silently passing.

## Results

`experiments/RESULTS.md` has the Phase 5 write-up: a controlled periodic
benchmark where the joint model beats the strictly local baseline by 2.6x on
force RMSE and breaks through the floor a 6 A cutoff imposes, a control
showing the gain disappears when there is no long-range physics to capture,
and the charge head recovering the true charge magnitudes (up to an exact
global-sign degeneracy).

## Example

`examples/water_lr/input.json` is a runnable energy-training config. It expects
the DeePMD-kit water dataset in `examples/water_lr/data` — copy it from a
DeePMD-kit checkout (`examples/water/data`); it is not vendored here.
