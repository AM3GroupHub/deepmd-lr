# 8 — Interfacial thermal conductance by NEMD

Two Langevin baths on one slab, and the heat they exchange through the
interface between them. This is the protocol behind the paper's thermal
boundary conductance numbers, written on the GPU MD classes of
[example 06](../06-md_nvalchemi): the integrator is a subclass of
`MolecularDynamics` defined in [`nemd.py`](nemd.py), the way
[`07_custom_berendsen.py`](../06-md_nvalchemi/07_custom_berendsen.py)
defines a thermostat of its own, and the fixed layers are ASE's `FixAtoms`.

```
vacuum | fixed_bottom | HOT bath | free: material A | material B | COLD bath | fixed_top | vacuum
                                              ^ interface, z
```

In steady state the power the hot bath injects equals the power the cold
bath removes, and all of it crosses the **single** interface, so the heat
flux is `J = P / A` with no factor ½ (a periodic two-interface cell would
need one). The temperature profile T(z) jumps by ΔT at the interface, and
the interfacial (Kapitza) conductance is `G = J / ΔT`.

```bash
python nemd.py --model ../../path/to/model.ckpt.pt --structure interface.extxyz --out run1
python analyze.py --out run1
```

## The structure

A slab open along z (`pbc T T F`); `read_structure` pads the vacuum to at
least 20 Å and centres the slab. The groups come from one of two places:

- **A per-atom `group` array** in the extxyz, with the labels
  `fixed_bottom`, `hot`, `free`, `cold`, `fixed_top` (the file the paper's
  runs used). They must lie in that order along z (by median height, so
  molecules may straddle a boundary); `hot`, `free` and `cold` must be
  non-empty, and the fixed groups may be.
- **Thicknesses**, when the file has no such array: `--fixed-thickness`
  ångström at each face are held, and the next `--bath-thickness` ångström
  inside each are the baths. Everything between is free.

Atoms the file already holds (POSCAR selective dynamics, extxyz `move_mask`)
stay fixed as well. The interface plane is midway between the highest free
atom of the lower species and the lowest free atom of the upper one; with
more than two species in the free region pass `--interface-z` (in the
coordinates after centring; the script prints the group extents).

## The protocol

One evaluator carries all three stages:

| stage | what runs | output |
|---|---|---|
| equilibration, `--equil-ps` | `Langevin` at `--t0` on every mobile atom | `equil.log` |
| steady, `--steady-ps` | `RegionLangevin`, baths at `--t-hot` / `--t-cold`; the gradient builds up | `thermo.log`, `profile.csv` from t = 0 |
| production, `--run-ps` | the same object; its heat account is reset to zero first | the same files, `trajectory.extxyz`, `checkpoint.npz` |

Resetting the account after the steady stage keeps the baths' transient
and the build-up of the gradient out of the fitted power. Time in the logs
counts from the moment the baths switch on; `analyze.py` uses the rows
after `--steady-ps`.

**`RegionLangevin`.** Each step is O(dt/2) on every bath, velocity Verlet
(dt), O(dt/2) again, where the O-step is `v ← c1 v + c2 √(kT/m) ξ` with
`c1 = exp(−friction dt/2)` and `c2 = √(1 − c1²)`. Atoms outside the baths
are NVE. The kinetic energy every O-step adds is charged to its bath on the
device, so `md.heat` is exact to rounding, costs no host synchronisation
per step, and `Etot − ΣQ` fluctuates like NVE's `Etot` (checked on the
test water box: 2.9 meV rms against 1.1 meV for NVE at the same timestep).
The noise is drawn from a generator reseeded with `seed + step` every
step, so a resumed run replays exactly the noise an uninterrupted one would
have drawn.

Three things the class does **not** do, on purpose:

- **No centre-of-mass correction during the run.** The fixed layers hold
  the slab; a running correction would remove kinetic energy no bath is
  charged for. `v_com[A/fs]` in the log is the diagnostic. Without fixed
  layers (`--fixed-thickness 0`) the base class removes the drift once at
  the start and never again, and a drifting slab smears T(z) in the lab
  frame: fix the layers.
- **No conserved quantity.** Judge a run by the two bath powers agreeing.
- **No fixed atom in a bath.** `Thermo` zeroes a fixed atom's velocity
  before every force call, so the energy a bath gave it would vanish from
  the account; the class refuses the overlap.

With `--t-hot` = `--t-cold` = `--t0` the run is the equilibrium control: the
long-time average of each bath's power is zero, but the cumulative Q of one
finite trajectory random-walks and is not expected to stay near zero.

## Units

`--tau` is the baths' relaxation time in fs and becomes `friction = 1 / (tau
* units.fs)`, in inverse ASE time units as `Langevin` takes it ([example
06](../06-md_nvalchemi/README.md#units-and-the-one-that-bites) explains the
unit). Q is in eV and t in ps, so the fitted power is in eV/ps and, with A
in Å², `J` in eV ps⁻¹ Å⁻²; `1 eV ps⁻¹ Å⁻² = 1.602 × 10¹³ W m⁻²`, which
puts G in W m⁻² K⁻¹ (printed in MW m⁻² K⁻¹).

## The outputs

| file | what |
|---|---|
| `thermo.log` | every `--log-every` steps: `T[K]` of the mobile atoms, `Epot`, `Ekin`, `Etot`, then `T_hot[K] Q_hot[eV] T_cold[K] Q_cold[eV]`, `T_left[K]`/`T_right[K]` (the free region on each side of the interface), `v_com[A/fs]`, `maxF[eV/A]`, and `q_sum`/`q_absmax` for a model with a charge head; `numpy.loadtxt` reads it |
| `profile.csv` | one row per z-bin (`--bin`) per block (`--block-ps`): the block's mean kinetic temperature of the mobile atoms in that bin, and the number of samples (every `--profile-every` steps) |
| `trajectory.extxyz` | every `--traj-every` steps, with energy, forces, velocities, the charges, and the `FixAtoms` as `move_mask` |
| `checkpoint.npz` | every `--checkpoint-ps`, atomically, plus one right after the heat reset |
| `run_meta.json` | the settings, group sizes and extents, the interface position and area, the final heat, throughput |
| `analysis.json`, `nemd.png` | what `analyze.py` fitted |

`--checkpoint-ps` must be a multiple of `--block-ps`, and `--steady-ps` as
well, so every checkpoint falls on a block boundary with the profile
accumulators empty. `--resume` restores the state and the heat account,
trims `thermo.log`, `trajectory.extxyz` and `profile.csv` back to the
checkpoint, and continues; pass the same arguments (the log is appended
without a new header, so the columns must not change).

## Reading the result

`analyze.py` fits a line to `Q_hot(t)` and to `−Q_cold(t)` over production:
the two slopes are the powers, and their mismatch, printed as a percentage,
is the steady-state check. The profile is averaged over production with the
sample counts as weights; a line is fitted to T(z) on each side of the
interface over the free region, leaving `--margin` ångström out on either
side of the plane, and the two lines extrapolated to the plane give ΔT.
Errors come from block slopes (`--nblocks`) and the fit covariances.

```
production 50.2 - 250.0 ps, 1999 rows; area 261.02 A^2; interface z 47.31 A
  P_hot  +0.0468 +- 0.0016 eV/ps
  P_cold +0.0451 +- 0.0021 eV/ps   (mismatch 3.7 % of the mean: the steady-state check)
  J      1.760e-04 eV/(ps A^2) = 2.820 GW/m^2
  ...
  dT     15.32 +- 0.61 K
  G      184.1 +- 9.8 MW m^-2 K^-1
```

(The numbers above are the shape of the output, not a result; the paper's
system and model are documented with the data release.) Things that go
wrong, and what they look like:

- **The two powers disagree by more than a few per cent.** Not steady yet:
  lengthen `--steady-ps`, or drop the start of production with `--skip-ps`.
- **`T_hot` or `T_cold` off their targets.** A bath needs a few `--tau` to
  settle; with a very stiff material and a short `--tau` the bath fights
  the phonons. 100 fs is a reasonable start.
- **T(z) is not linear on a side.** The fit region is too close to a bath
  or to the interface; raise `--margin`, or use thicker free layers. The
  per-side `kappa` printed is finite-size limited and is only a check that
  the two sides are sane.
- **`v_com` grows.** The slab is drifting: it has no fixed layers, or too
  thin ones.
- **`maxF` spikes or the run stops with `MDInstabilityError`.** The model is
  outside its training data; the run refuses to continue past five times
  `--t-hot`.

## Several GPUs

`--gpus N` partitions the slab over the first N visible GPUs
(`build_evaluator(..., devices=[...])`), as in example 06. Whether a second
card pays depends on the atom count and on compression; see
[partitioned MD](../../docs/partitioned-md.md#limits).

## What changed from the paper's script

`experiments/nemd/nemd.py` is the driver the paper's runs used, on an older
version of this package. This example keeps its physics (the symmetric
O–VV–O splitting, the single-interface flux, the reset after the steady
stage, no running COM correction) and drops what the package now does:
the hand-frozen layers are `FixAtoms`, the logs, trajectory, checkpoint and
resume are the MD classes', and the noise comes from a reseeded torch
generator rather than a saved RNG state. Its `trailing` splitting option
is gone, so a trajectory here is not step-for-step the old one at the
same seed; the steady-state powers and profiles are what to compare.
