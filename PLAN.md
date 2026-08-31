# Long-Range Electrostatics for DPA4C via nvalchemi PME

## Context

DPA4C is a strictly local descriptor: an atom's energy depends only on neighbours within
`rcut`, so the model cannot represent Coulomb interactions beyond the cutoff. We want to add
an explicit long-range electrostatic term computed by NVIDIA's PME implementation
(`nvalchemi-toolkit-ops`), trained jointly with the short-range DPA4C energy.

The central question was whether this requires forking deepmd-kit. **It does not.** Verified
against the released `deepmd-kit==3.2.0` aarch64 wheel:

- `deepmd/pt_expt/__init__.py` calls `load_entry_point("deepmd.pt_expt")` — a third-party pip
  package can register itself into the pt_expt backend at import time.
- `deepmd/dpmodel/model/model_factory.py:430` ends model-type dispatch with
  `base_model.get_class_by_type(model_type).get_model(data)` — a plugin-registered
  `model.type` is reachable from a stock install with no patch.
- `deepmd/utils/argcheck.py` exposes `descrpt_args_plugin`, `fitting_args_plugin`,
  `model_args_plugin`, `loss_args_plugin` — the JSON-schema side is extensible too. (Schema
  rigidity is normally what forces a fork; here it doesn't.)

The one real architectural constraint: `NeighborGraph`
(`deepmd/dpmodel/utils/neighbor_graph/graph.py:36`) states *"Geometry enters the model ONLY
through `edge_vec`"*, and neither `forward_common_atomic` nor `call_common_lower` receives the
cell. PME needs absolute coordinates + cell, so the long-range term **must attach at the
top-level model**, where `call_common(coord, atype, box, ...)`
(`deepmd/dpmodel/model/make_model.py:278`) does have the box. That is why this plan registers a
new *model type* rather than a new *atomic model*.

**Decisions taken** (confirmed with the user):
- Charges: environment-dependent charge head on the DPA4C descriptor, projected to satisfy the
  total-charge constraint.
- Training: joint end-to-end, one loss over `E_sr + E_lr`.
- MD target: nvalchemi / ASE (both drive the top-level forward, which has the box). LAMMPS is
  explicitly out of scope — its pair-style path cannot see the cell.

## Environment

Verified on this machine:

| Property | Value |
|---|---|
| Arch | `aarch64` (GB10 / DGX Spark class) |
| GPU | NVIDIA GB10, compute capability **12.1** (sm_121) |
| CUDA toolkit | 13.0 |
| glibc | 2.39 |
| RAM | 119 GB unified |
| `dpmd` env | conda, Python **3.11.16**, currently empty |

Constraints this imposes:
- `deepmd-kit==3.2.0` pins `torch==2.11.0.*`; its wheel is built against torch 2.11.0.
- sm_121 requires a CUDA 12.8+/13 build → use the **cu130** PyTorch index.
- `torch-2.11.0+cu130-cp311-cp311-manylinux_2_28_aarch64.whl` **exists** — confirmed.
- The PyPI deepmd-kit aarch64 wheel is `DP_VARIANT=cpu` (its `run_config.ini` confirms; no CUDA
  in `libdeepmd_op_pt.so`). This is fine: training runs in **eager PyTorch** on GPU, and the
  compiled CUDA mega-kernels are opt-in via `DP_CUDA_INFER` and inference-only
  (`deepmd/kernels/utils.py:cuda_infer_level`). No source build needed.

### Install into `dpmd`

```bash
# 1. PyTorch 2.11.0 + CUDA 13 (aarch64) — version pinned by deepmd-kit 3.2.0
conda run -n dpmd pip install --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.11.0+cu130"

# 2. deepmd-kit (prebuilt aarch64 wheel; core deps only, avoids pulling mpich)
conda run -n dpmd pip install "deepmd-kit==3.2.0"

# 3. deepmd's torch-path extras, explicitly
conda run -n dpmd pip install "e3nn>=0.5.9" "vesin[torch]"

# 4. nvalchemi PME engine (pure-python wheel + aarch64 warp)
conda run -n dpmd pip install "nvalchemi-toolkit-ops>=0.4.1" "warp-lang>=1.13.0"

# 5. data / MD / dev tooling
conda run -n dpmd pip install "dpdata>=0.2.7" "ase>=3.23.0" \
    pytest pytest-cov ruff pre-commit

# 6. optional: nvalchemi MD driver (for the ASE/nvalchemi MD path later)
conda run -n dpmd pip install nvalchemi-toolkit
```

Wheel availability confirmed for all of the above on `cp311` / `linux_aarch64`:
`nvalchemi_toolkit_ops-0.4.1-py3-none-any.whl` (pure python),
`warp_lang-1.17.0-py3-none-manylinux_2_34_aarch64.whl` (glibc 2.39 ≥ 2.34 ✓),
`deepmd_kit-3.2.0-py37-none-manylinux_2_28_aarch64.whl`.

**Smoke test before writing any code:**

```bash
conda run -n dpmd python -c "
import torch, warp, deepmd
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
from nvalchemiops.torch.interactions.electrostatics.pme import particle_mesh_ewald
print('pme ok')
from deepmd.pt_expt.descriptor.dpa4c import *   # dpa4c reachable
print('dpa4c ok')
"
```

The riskiest line is warp compiling for sm_121; if it fails, `warp.init()` will say so and the
fix is a newer `warp-lang`, not a change to this design.

## Architecture

Register a new top-level model type `dp_lr` in our own package. It owns a DPA4C descriptor, two
heads, and the PME term:

```
                 ┌─> ener fitting ──────> E_sr,i
descriptor(dpa4c)┤
                 └─> charge fitting ────> q~_i ──project──> q_i   (Σq_i = Q_total)
                                                             │
                            coord, box ──────────────────────┤
                                                             v
                                       particle_mesh_ewald(...) -> E_lr,i

E_total = Σ_i (E_sr,i + E_lr,i);  F = -dE/dr, virial from dE/dcell  (autograd)
```

Sharing one descriptor between both heads is the point: charges become environment-dependent for
free, and the marginal cost over plain DPA4C is one extra fitting network.

### Why these specific reuse points

| Need | Reuse | Path |
|---|---|---|
| Model registration | `BaseModel.register("dp_lr")` + `get_model` classmethod | `deepmd/dpmodel/model/base_model.py:284` |
| Backend model class | subclass `EnergyModel` | `deepmd/pt_expt/model/ener_model.py:36` |
| Charge head | subclass `InvarFitting` (`dim_out=1`) | `deepmd/dpmodel/fitting/invar_fitting.py:32` |
| Total charge per frame | existing `charge_spin` (nf × 2) tensor | `deepmd/utils/charge_state.py` |
| JSON schema | `model_args_plugin` / `fitting_args_plugin` | `deepmd/utils/argcheck.py:3389`, `:2767` |
| Loss | stock `ener` loss — output keys unchanged | no new code |
| PME | `particle_mesh_ewald(...)` | `nvalchemiops.torch.interactions.electrostatics.pme` |

Note the PyTorch bindings live at `nvalchemiops.torch.interactions.electrostatics`; the
`nvalchemiops/interactions/electrostatics` path is the framework-agnostic Warp layer and its
torch re-exports there are deprecated.

`particle_mesh_ewald` gives us exactly what the deepmd contract wants:
- `energy_reduction="atom"` → per-atom energies, matching the atomic-energy contract.
- `compute_charge_gradients=True` → dE/dq, to backprop into the charge head.
- `_pme_convolve_double_backward` exists → double backward works, which is required to train
  against **force** labels through an autograd-derived force.

## Repository layout

```
deepmd-lr/
├── pyproject.toml            # entry point: [project.entry-points."deepmd.pt_expt"]
├── README.md
├── PLAN.md                   # this document
├── deepmd_lr/
│   ├── __init__.py
│   ├── register.py           # entry-point target; imports the modules below
│   ├── fitting/charge.py     # ChargeFitting(InvarFitting), registered "charge"
│   ├── model/lr_model.py     # LongRangeEnergyModel(EnergyModel), registered "dp_lr"
│   ├── pme/term.py           # thin wrapper over particle_mesh_ewald + unit conversion
│   └── argcheck.py           # model_args_plugin / fitting_args_plugin registrations
├── tests/
└── examples/water_lr/input.json
```

`pyproject.toml` entry point:

```toml
[project.entry-points."deepmd.pt_expt"]
deepmd_lr = "deepmd_lr.register:register"
```

## Implementation phases

### Phase 0 — environment + baseline (do this first, it de-risks everything)
1. Install per the block above; run the smoke test.
2. Train stock DPA4C on `examples/water/dpa4c/input.json` from the deepmd-kit source tree for a
   few hundred steps. This proves the toolchain before any of our code exists, and becomes the
   accuracy baseline the long-range model must beat.
3. Independently exercise `particle_mesh_ewald` on a known system (e.g. NaCl lattice) and check
   the energy against the Madelung constant. Confirm forces and the charge gradient agree with
   `torch.autograd.gradcheck` in float64.

### Phase 1 — plugin skeleton
Create the package, the entry point, and a `register()` that registers a `dp_lr` model type
which simply *delegates to plain DPA4C* with no long-range term. Verify
`dp --pt-expt train` accepts `"type": "dp_lr"` and reproduces Phase 0 numbers exactly. This
isolates "does the plugin mechanism work" from "is the physics right".

### Phase 2 — charge head
`ChargeFitting(InvarFitting)` with `dim_out=1`, registered as fitting type `charge`, plus its
`fitting_args_plugin` entry. Apply the total-charge constraint by projection:

```
q_i = q~_i - (Σ_j q~_j - Q_total) / N
```

so `Σq_i = Q_total` exactly and the projection stays differentiable. `Q_total` comes from the
existing `charge_spin` tensor. Test: charges sum to the target, are permutation-equivariant, and
are invariant to rotation and translation of the frame.

### Phase 3 — PME term
`deepmd_lr/pme/term.py` wraps `particle_mesh_ewald`. Responsibilities:
- Unit conversion into deepmd's eV / Å convention (nvalchemi takes charges in elementary charge
  units; verify the Coulomb prefactor explicitly against a two-charge analytic case).
- Parameter selection: `accuracy`, `mesh_spacing`, `spline_order`; expose in the input JSON.
- Non-periodic frames: `box is None` must raise a clear error, matching how
  `deepmd/pt_expt/modifier/dipole_charge.py` handles it.

### Phase 4 — joint model
`LongRangeEnergyModel(EnergyModel)`, registered `dp_lr`. Override `forward` to:
1. call the parent for `E_sr` per atom,
2. run the charge head on the same descriptor output,
3. call the PME term with `coord`, `q`, `box`,
4. sum per-atom energies before the reduction/derivative machinery, so force and virial come out
   of the existing autograd path with no bespoke code.

Getting the summation *upstream* of the derivative machinery is the key detail — it means
`_to_public_keys` (`ener_model.py:57`) produces correct `force`/`virial` for free.

### Phase 5 — training + validation
Train jointly on a system where long-range matters and the baseline is known to struggle —
bulk water with an applied field, or an ionic solution. Compare against the Phase 0 DPA4C
baseline on energy/force RMSE, and specifically on properties the cutoff should ruin: dielectric
response, ion–ion PMF beyond `rcut`.

### Phase 6 — MD via nvalchemi/ASE
Wire the trained model into the nvalchemi driver, following
`examples/water/dpa4/nvalchemi/relax.py` and `doc/third-party/nvalchemi.md` in the deepmd-kit
tree. Confirm energy conservation in NVE — this is the sharpest test that forces are the exact
gradient of the energy, PME term included.

## Verification

```bash
# unit
conda run -n dpmd pytest tests/ -v

# physics: PME against analytic Madelung; gradcheck in float64
conda run -n dpmd pytest tests/test_pme_physics.py -v

# plugin wiring: is the type registered from a stock deepmd?
conda run -n dpmd python -c "
from deepmd.pt_expt.model.model import BaseModel
print('dp_lr' in BaseModel.get_plugins())"

# end-to-end
conda run -n dpmd dp --pt-expt train examples/water_lr/input.json
conda run -n dpmd dp --pt-expt test -m model.ckpt.pt -s /path/to/validation

# conservation
conda run -n dpmd python examples/water_lr/nve_drift.py   # NVE energy drift over 10 ps
```

Gate for each phase: Phase 1 must reproduce the Phase 0 baseline *bit-for-bit*; Phase 4 must
reduce to Phase 1 when charges are forced to zero. Both are cheap regression tests worth
keeping.

## Risks

| Risk | Mitigation |
|---|---|
| warp-lang cannot JIT for sm_121 | Caught in the Phase 0 smoke test; fix is a newer `warp-lang`. Blocks everything, so test first. |
| Coulomb prefactor / unit mismatch | Phase 0 Madelung check against an analytic value before any integration. |
| Joint training unstable from random charge init | Fall back to the staged scheme: fit charges first, then unfreeze. Design already supports it. |
| Double-counting: DPA4C absorbs short-range Coulomb | Expected and acceptable — PME real-space is screened within `rcut`. Watch for charge drift toward unphysical magnitudes; regularize charges if so. |
| deepmd-kit 3.2.0 lags master | The plugin hooks are all present in 3.2.0 (verified). Pin 3.2.0; revisit only if a needed DPA4C fix is master-only. |
