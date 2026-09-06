# LiFSI / EC-DMC electrode–electrolyte interface

The worked example: training a long-range model on a real interface, and
running MD with it — serially or over spatial partitions.

## Quick start: distributed MD

If you already have a trained model, [`md_minimal.py`](md_minimal.py) is the
whole thing — the API is three lines:

```python
from deepmd_lr.distributed import DistLongRangeCalculator

atoms.calc = DistLongRangeCalculator("run_joint/model.ckpt.pt")
atoms.calc.enable_distributed_mode(4)          # or ["cuda:0", "cuda:1", ...]
VelocityVerlet(atoms, timestep=0.5 * units.fs).run(1000)
```

```
$ python md_minimal.py
6520 atoms, 4 partitions
  step      E_pot [eV]      E_tot [eV]    T [K]
     0     -34073.2102     -33818.1876    302.6
    40     -34040.5033     -33818.0056    264.0
   100     -34057.5840     -33818.0551    284.2
```

An integer means "that many partitions on the current device" — identical
arithmetic to that many GPUs, which is how the partitioning is exercised
without multi-GPU hardware.

`md.py` is the same thing with a CLI and a choice of `nve`, `nvt-langevin` or
`nvt-nose-hoover`:

```bash
python md.py --ensemble nvt-langevin --temperature 300 --partitions 4
```

The rest of this README is how to get to a trained model from the raw
trajectory.

## The system

A UMA trajectory of LiFSI salt in EC/DMC solvent against a lithium anode.
6520 atoms per frame, `pbc="T T F"` — a **slab**, 28.08 × 28.08 × 106.33 Å.

| region | contents |
|---|---|
| z 0–30 Å | ~1026 Li, nearest-neighbour 2.71 Å → metallic electrode |
| z 30–105 Å | ~38 Li⁺, 40 FSI⁻ (N=40, S=80, F=80), EC/DMC solvent |

N/S/F pin the salt at exactly 40 FSI⁻, so only 40 of the 1064 Li are
counter-ions; the rest is electrode.

> Charges on *metallic* Li are a fitting device, not observables — a metal
> screens fields and delocalises charge. The electrolyte-side numbers are the
> ones with a physical reading. `analyze_frame.py` reports the two regions
> separately.

## 1. Build the dataset

The trajectory is not vendored (3.3 GB). Point the script at your copy:

```bash
python prepare_dataset.py --traj LiFSI_EC_DMC_1M_T300.xyz \
    --frames 300 --skip 100 --vacuum 20 --out data
```

Three things happen, two of them mandatory:

- **300 frames sampled evenly** across the run (not a block — consecutive MD
  frames are correlated), after discarding 100 for equilibration.
- **x and y wrapped** back into the cell; the raw coordinates run outside it.
- **z padded with 20 Å of vacuum.** At the raw 1.99 Å gap, calling the cell
  3D-periodic would put 427 atoms at the top face within the 6 Å descriptor
  cutoff of 256 at the bottom — an interaction the UMA labels (`pbc="T T F"`)
  never contained. The gap must exceed twice the cutoff.

Padding makes the cell periodic; it does **not** make the physics 3D. The slab
still interacts with its images across the vacuum, so training uses the
Yeh–Berkowitz correction (`slab_correction: true`, `pbc: [true, true, false]`).
Without it the energy has not converged even at 160 Å of vacuum.

## 2. Train

Measured cost at 6520 atoms: 883 ms/step short-range against 4465 ms joint.
The long-range term is not the expense — profiled alone it is 73 ms; the gap
is the second descriptor pass and its double backward. So the budget goes
mostly where it is cheap:

```bash
# 1. short-range warm-up (~0.9 s/step)
mkdir -p run_stage1 && cd run_stage1 && ln -s ../data data
dp --pt-expt train ../input_stage1.json --skip-neighbor-stat && cd ..

# 2. graft into the joint architecture, charge head scaled down
python ../../scripts/graft_short_range.py \
    --short-range run_stage1/model.ckpt.pt \
    --input input_joint.json --out run_joint/init_joint.ckpt.pt

# 3. joint fine-tune
mkdir -p run_joint && cd run_joint && ln -s ../data data
dp --pt-expt train ../input_joint.json --init-model init_joint.ckpt \
    --skip-neighbor-stat
```

`--skip-neighbor-stat` is **required**: the statistics pass is an O(N²) scan
that tries to allocate 6 TB at this size, and DPA4C is graph-native with no
`sel` for it to compute.

Also train `input_baseline.json` (same budget, no `long_range`) and
`input_control.json` (same two-stage schedule, no `long_range`). Without the
*control* you cannot separate the long-range term from the schedule — on this
system the apparent 1.11× force gain over the baseline turned out to be
entirely the schedule. See `../../experiments/RESULTS_LiFSI.md` if you have it.

## 3. Run MD

```bash
# constant energy, whole cell on one device
python md.py --model run_joint/model.ckpt.pt --ensemble nve --steps 200

# Langevin thermostat at 300 K, model evaluated over 4 partitions
python md.py --ensemble nvt-langevin --temperature 300 --partitions 4
```

The integrators are ASE's `VelocityVerlet`, `Langevin` and
`NoseHooverChainNVT`, driven through `DistLongRangeCalculator`.

### Distributed evaluation

`--partitions N` splits the descriptor over N partitions; the reciprocal sum
is gathered onto one device because it does not survive spatial decomposition
(2.0% error at 2 ranks, 36.5% at 8 — `../../LAMMPS.md`). At 73 ms against the
descriptor's 883 ms that is an Amdahl floor, not a bottleneck.

Results are identical to the serial path — verified here:

```
1 partition  step 0: -34073.2104 eV   step 10: -34050.4355 eV
4 partitions step 0: -34073.2102 eV   step 10: -34050.4355 eV
```

On a single-GPU machine `--partitions 4` places all four on that GPU. The
arithmetic is identical to four devices; only placement differs, which is how
the partitioning is exercised without multi-GPU hardware. With more GPUs:

```bash
python md.py --devices cuda:0 cuda:1 cuda:2 cuda:3
```

`DistLongRangeCalculator` is documented in `../../TRAINING.md` §10.

## 4. Inspect the model

```bash
python evaluate.py --models baseline=run_baseline/model.ckpt.pt \
                            joint=run_joint/model.ckpt.pt
python analyze_frame.py --model run_joint/model.ckpt.pt --frame 0
```

`analyze_frame.py` prints the energy split, the learned charges by element and
by region, and their profile along the slab normal — where an electric double
layer would appear.
