# 07 — Production MD on one or more GPUs

`run_md.py` is a ready-to-use MD driver, where the scripts in
[06](../06-md_nvalchemi) each teach a single ensemble. One command runs NVE,
NVT (Langevin or Nose-Hoover chain) or isotropic NPT for any 3D-periodic
system, spread over as many GPUs as you give it. It also handles:

- restarts across walltime-limited jobs;
- measuring how many GPUs are worth using;
- an optional pre-relaxation;
- a trajectory, a thermo log and the final structure.

The full manual is the script's docstring, and `python run_md.py --help`
prints it together with every option. It is written so that an AI agent can
read it and adapt the run to a new system: see *ADAPTING TO A NEW SYSTEM* and
*FOR AI AGENTS* there.

```bash
# smoke test
python run_md.py --model model.ckpt.pt --structure start.extxyz --out runs/smoke --steps 200

# how many GPUs pay off here? (times 1, 2, 4 GPUs and exits)
python run_md.py --model model.ckpt.pt --structure start.extxyz --out runs/bench --benchmark --gpus 4

# production inside a 24 h job; resubmit with --restart until it reaches --steps
python run_md.py --model model.ckpt.pt --structure start.extxyz --out runs/nvt300 \
    --gpus 2 --temperature 300 --steps 200000 --max-hours 23.5
```

Run it from anywhere; it depends only on the installed `deepmd_lr`. The
bundled [`water_192.xyz`](../06-md_nvalchemi/water_192.xyz) is a convenient
first structure.

## What was checked

All checks used the 192-atom water box and a trained long-range water model on
one GB10 GPU. "Two partitions" means two partitions on that one GPU: that
checks the decomposition, not placement on separate cards. Placement on two
distinct A40s is covered by the library's own multi-GPU validation.

| check | result |
|---|---|
| NVE, dt 0.5 → 0.25 fs, 0.5 ps each | Etot fluctuation 0.0185 → 0.0051 meV/atom (3.6x, O(dt²)); drift ≈ 0.01 meV/atom/ps at both |
| compressed vs uncompressed (`--compress`) | step-0 energy within 1.4e-6 eV and forces within 7.6e-6 eV/Å; identical NVE drift and fluctuation; 11 vs ~37 ms/step |
| two partitions vs one | energy within 1.5e-6 eV and forces within 1.8e-6 eV/Å at step 0; 6e-6 after 20 steps |
| unwrapped input: 60 atoms moved by up to 3 lattice vectors | same energies and forces within 1e-6. Without the script's fold the evaluator was off by 699 eV and 98 eV/Å (see *Known limitations*) |
| restart = uninterrupted run (Langevin, Nose-Hoover, NPT) | positions agree to 7e-7 Å, NPT cell to 3e-10 Å; logs match |
| SIGTERM mid-run; SIGKILL after a checkpoint | clean stop, checkpoint, exit 0; after the kill, `--restart` trims the log and trajectory back to the checkpoint, with no duplicate steps |
| Nose-Hoover NVT, 1 ps, 300 K | ⟨T⟩ 307 K over the second half; Econs drift −0.034 meV/atom/ps |
| NPT: logging every step vs never, 200 steps | identical within GPU noise (7.8e-7 Å). Before the fix they were 1.0e-4 Å apart: reading the pressure for the log rewrote the barostat's input |
| NPT, MTK conserved quantity H′, dt 0.5 → 0.25 fs | rms 0.0218 → 0.0054 meV/atom (O(dt²)); thermostat and reported temperature both count 3N − 3 degrees of freedom |
| FIRE pre-relaxation | from an MD snapshot: 380 steps to 0.038 eV/Å, bonds intact. A badly distorted box made the minimiser collapse atoms, since the model has no short-range wall there; the script aborts with that diagnosis |
| blow-up (dt 8 fs) | stopped at the next log row with exit code 2 and `blowup.extxyz` written |

`tests/test_run_md.py` pins the essentials on the test fixture model.

## Known limitations

- **Periodic systems only.** The partitioned evaluator assumes periodicity in
  all three directions, so slabs and molecules in vacuum are not supported by
  this script.
- **Unwrapped coordinates.** The evaluator currently drops neighbours once
  unwrapped atoms spread more than about one cutoff beyond the cell, and MD
  never wraps. The script therefore hands the evaluator folded positions
  (the `Folder` class) while keeping the trajectory unwrapped. It is a
  workaround until the library folds coordinates itself.
- **NPT repartitions every step**, so it is slower than NVT. Use it to reach
  the density, then produce in NVT.
