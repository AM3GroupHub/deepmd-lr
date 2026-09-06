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
| 3 | Long-range term (`LongRangeEnergy`, reciprocal-only, eV) | **done** |
| 4 | Joint model (`long_range` block, conservative forces) | **done** |
| 5 | Training + validation (benchmark + control) | **done** |
| 6 | MD deployment (ASE calculator, export guard) | **done** |

## Training

**[TRAINING.md](TRAINING.md)** is the practical guide: choosing `alpha` and
the mesh (with measured tables), an annotated input file, the joint and
staged training schedules, what to monitor, and troubleshooting.

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
      "pme": { "alpha": 0.4, "mesh_dimensions": [16, 16, 16] }
    }
  }
}
```

```bash
dp --pt-expt train input.json
```

Without the `long_range` block the model is exactly the built-in standard
model.

Two settings matter. `alpha` divides the labour: the network handles the
screened part `erfc(alpha r) / r` and this term supplies the rest, so `alpha`
must be large enough that the network's share dies inside the descriptor
cutoff (`alpha = 0.4` leaves `erfc(2.4) = 7e-4` at 6 A), and it must match
between training and MD. Prefer `mesh_dimensions` over `mesh_spacing`
whenever the cell can change — see PLAN.md, Phases 3 and 4.

Only the long-range (reciprocal-space) half of the Ewald sum is computed
here; the short-range half is already covered by the descriptor.

The sum is evaluated by particle-mesh Ewald by default. Set
`"method": "ewald"` for an explicit k-vector sum — exact, with no mesh
interpolation error, which makes it the way to check a PME mesh setting. It
is much slower (114 ms vs 3 ms at 2048 atoms), so it is a validation tool
rather than a production one.

## Running MD

```python
from deepmd_lr.ase import LongRangeCalculator
atoms.calc = LongRangeCalculator("run_joint/model.ckpt.pt")
```

`examples/lifsi/md_minimal.py` is a ten-line distributed MD run.

For larger systems, `DistLongRangeCalculator` evaluates the same model over
spatial partitions — the descriptor is split, the reciprocal sum is gathered
onto one device. Identical results, verified partition-count invariant. See
TRAINING.md §10.

`dp freeze` is **refused** for a model with a `long_range` block: the exported
graph lower receives edge vectors and no cell, so the artifact would silently
contain only the short-range energy. See TRAINING.md §9.

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
- `tests/test_lr_term.py` — the eV unit conversion (against analytic
  Coulomb and Madelung), boundary-condition guards, batching, masking, and
  gradients including the double backward that force training needs.
- `tests/test_joint_model.py` — the combined model: force and virial against
  finite differences of the total energy, energy decomposition, both calling
  conventions, and serialization of the long-range term.

The tests import only `deepmd`, never `deepmd_lr`, so they fail if the entry
point stops working rather than silently passing.

## Results

`experiments/RESULTS.md` has the Phase 5 write-up: a controlled periodic
benchmark where the joint model beats the strictly local baseline by 4.9x on
force RMSE and breaks through the floor a 6 A cutoff imposes, a control
showing the gain disappears when there is no long-range physics to capture,
and the charge head recovering the true charge magnitudes (up to an exact
global-sign degeneracy).

## Example

`examples/lifsi/` is the worked example: a LiFSI/EC-DMC electrolyte against a
lithium anode (6520 atoms, a 2D-periodic slab). It covers building the dataset
from a UMA trajectory, the staged training schedule, and MD with ASE's
ensembles — serial or over spatial partitions. See that directory's README.
Datasets are not vendored.
