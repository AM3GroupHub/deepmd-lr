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
| 2 | Charge head | not started |
| 3 | PME term | not started |
| 4 | Joint model | not started |
| 5 | Training + validation | not started |
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

```bash
dp --pt-expt train input.json
```

As of Phase 1 this is numerically identical to the built-in standard model —
the long-range term is added in Phase 4. That is deliberate: it separates
"the plugin is wired in correctly" from "the physics is right", so any
numerical change later can only come from the physics.

## Tests

```bash
pytest tests/ -q
```

- `tests/test_pme_physics.py` — PME against the NaCl Madelung constant,
  `gradcheck`/`gradgradcheck`, and a guard for the `spline_order >= 7`
  silent-failure mode (see PLAN.md, Phase 0).
- `tests/test_plugin_registration.py` — entry-point registration, input-schema
  acceptance, equivalence to the standard model, and serialization round-trip.

The tests import only `deepmd`, never `deepmd_lr`, so they fail if the entry
point stops working rather than silently passing.

## Example

`examples/water_lr/input.json` is a runnable energy-training config. It expects
the DeePMD-kit water dataset in `examples/water_lr/data` — copy it from a
DeePMD-kit checkout (`examples/water/data`); it is not vendored here.
