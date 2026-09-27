#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Multi-GPU molecular dynamics with a deepmd-lr model, for any periodic system.

WHAT IT DOES
    Runs NVE, NVT (Langevin or Nose-Hoover chain) or isotropic NPT (MTK) for a
    3D-periodic system. The atoms are split over one or more GPUs by
    deepmd_lr.nvalchemi.DistributedEvaluator, and the equations of motion are
    integrated on the GPU with nvalchemiops kernels.

    Output: a thermo log, a trajectory, restart checkpoints, the final structure
    and a record of the settings. The script can also measure how many GPUs are
    worth using before a long run.

REQUIREMENTS
    - deepmd-lr with its MD dependencies: nvalchemi-toolkit-ops, warp-lang, ase.
    - At least one CUDA GPU.
    - A checkpoint trained with deepmd-lr ("type": "dp_lr"). A plain DeePMD-kit
      pt_expt energy model also works, as short-range only.
    - --compress also needs deepmd-kit built from source with the CUDA
      operators (docs/compression.md).

QUICK START
    # 1. Smoke test: 200 steps of NVT at 300 K on one GPU.
    python run_md.py --model model.ckpt.pt --structure start.extxyz \
        --out runs/smoke --steps 200 --log-every 20

    # 2. How many GPUs pay off for this system? Times 1, 2, 4, ... GPUs, then exits.
    python run_md.py --model model.ckpt.pt --structure start.extxyz \
        --out runs/bench --benchmark --gpus 4

    # 3. Production: 100 ps of NVT on 2 GPUs inside a 24 h job.
    python run_md.py --model model.ckpt.pt --structure start.extxyz \
        --out runs/nvt300 --gpus 2 --ensemble nvt --temperature 300 \
        --steps 200000 --max-hours 23.5
    #    If the job stops early (wall time, a signal), repeat the same command
    #    with --restart until the log reaches step 200000.

    # 4. NPT at 1 bar, continuing from the end of the NVT run.
    python run_md.py --model model.ckpt.pt --structure runs/nvt300/final.extxyz \
        --keep-velocities --out runs/npt300 --gpus 2 --ensemble npt \
        --temperature 300 --pressure 1e-4 --steps 200000

CHOOSING THE NUMBER OF GPUS
    More GPUs help only when the GPUs themselves are the bottleneck. Measured on
    two A40s:

        case                                    1 GPU     2 GPUs
        compressed long-range, 12,288 atoms     20.4 ms   24.8 ms
        compressed long-range, 41,472 atoms     48.7 ms   47.6 ms
        uncompressed short-range, 12,288 atoms  222 ms    141 ms

    - A compressed model below about 100,000 atoms runs fastest on one GPU,
      because each step is limited by host work.
    - An uncompressed model gains from a second GPU from about ten thousand
      atoms on.
    - Add GPUs when one GPU runs out of memory: memory per GPU falls with the
      number of atoms per GPU.
    - Always measure your own system with --benchmark. Its numbers leave out
      the occasional halo rebuild, so treat them as a lower bound.

ENSEMBLES (--ensemble)
    nve  Velocity Verlet. The total energy is conserved, which makes this the
         check for a new model, system or timestep: run a few ps and read the
         summary. The rms fluctuation of Etot should fall about 4x when the
         timestep is halved, and the drift should be tiny: about 0.01
         meV/atom/ps for the bundled water box at either 0.5 or 0.25 fs.
    nvt  --thermostat langevin (the default) uses BAOAB with friction 1/tau_t.
         It is robust and stochastic: use it for equilibration and structure.
         --thermostat nhc uses a Nose-Hoover chain. It is deterministic: use it
         for dynamics (diffusion, spectra). The log gains Econs, the extended
         energy that should stay flat.
    npt  Isotropic MTK barostat with a Nose-Hoover chain thermostat. It needs
         the virial, which the script requests. The cell moves every step and
         that forces a repartition every step, so NPT is slower than NVT;
         equilibrate the density with it, then produce in NVT.

    Typical values:
    - timestep: 0.5 fs with hydrogen; 1-2 fs without light atoms.
    - tau_t: 100 fs.
    - tau_p: 1000 fs.

UNITS
    - Command line: fs, K, GPa (1 bar = 1e-4 GPa), V/A, A.
    - Files: eV, A, K, GPa, and ps for the time column of the log.

OUTPUT FILES (in --out)
    md.log          The thermo table: a '#' header, then one row every
                    --log-every steps. Columns: step, time[ps], T[K],
                    Epot/Ekin/Etot[eV]; Econs[eV] for nhc; P[GPa] and V[A^3]
                    for npt; ms/step. Read it with numpy.loadtxt.
    traj.extxyz     One frame every --traj-every steps, with energy, step and
                    time_ps in the comment line. Positions are UNWRAPPED, which
                    diffusion analysis needs; --traj-wrapped wraps them.
                    --traj-format traj writes ASE's binary format instead.
    checkpoint.npz  The full state (positions, velocities, cell, thermostat and
                    barostat variables, step, seed). Written every
                    --checkpoint-every steps, at the end, and on a clean stop.
    final.extxyz    The last structure with velocities. It is a valid
                    --structure for the next run; add --keep-velocities.
    run.json        Settings, versions, devices and timings, one entry per run
                    or restart segment.
    blowup.extxyz   Written only when the run became unstable.

RESTARTS AND JOB CHAINS
    --steps is the TOTAL length of the run. --restart continues the run in
    --out from checkpoint.npz up to --steps, appending to the log and the
    trajectory without duplicating frames. With the same --steps, a restart
    produces the same trajectory an uninterrupted run would, to rounding.

    The physics settings must match the checkpoint, otherwise the restart is
    refused. These are ensemble, thermostat, dt, temperature, tau_t, pressure,
    tau_p, chain length, field, eps_inf and charge/spin. The number of GPUs,
    output intervals and --steps may change.

    To change the physics, start a NEW run from --structure final.extxyz with
    --keep-velocities.

    The run stops cleanly and writes a checkpoint when:
    - --max-hours is reached;
    - it receives SIGTERM, SIGUSR1 or SIGINT (a second SIGINT quits at once).
    Schedulers send SIGTERM before killing a job, so this usually happens on
    its own. After a hard kill, --restart resumes from the last periodic
    checkpoint and trims the log and trajectory back to it.

ADAPTING TO A NEW SYSTEM (checklist)
    1. Every element in the structure must be in the model's type_map. The
       script refuses otherwise and names the missing elements. A model whose
       type_map is the whole periodic table passes this check for any element,
       so also make sure its training data contained every element of your
       system.
    2. The structure needs a cell and is treated as periodic along all three
       axes. Slabs and molecules in vacuum are not supported by this script.
    3. Pick the timestep (see Typical values above).
    4. For a freshly built or packed box, add --relax-steps 2000. The relaxation
       moves positions only (cell fixed). It refuses to continue if it did not
       converge or broke a covalent bond between non-metal atoms. It aborts if
       it diverges: a model with no short-range repulsive wall can let the
       minimiser collapse atoms onto each other from a badly distorted start.
       Pre-relax such a box with a classical force field instead; for water,
       docs/md.md ("Relaxing a packed box first") has a rigid-TIP3P recipe.
       To relax the cell as well, toward --pressure, add --relax-cell:
       isotropic keeps the shape (the choice for a liquid, which has no shear
       stiffness), orthorhombic moves the three lengths of a diagonal cell,
       full moves all nine entries (crystals). It converges when the stress is
       within --relax-smax GPa of the target as well.
    5. For a net charge or a spin state the model was trained to condition on,
       pass --charge-spin Q MULTIPLICITY. Otherwise the model's default is used.
    6. For a multi-head checkpoint, pass --head BRANCH.
    7. Run nve for a few ps and check the drift before production.
    8. Run --benchmark to choose --gpus, and add --compress if the build has
       the CUDA operators.

FIELD RUNS (--efield EX EY EZ, V/A; NVT only)
    The model's Born charges Z give the force F = Z^T E. There is no conserved
    energy under a field, so a thermostat is mandatory and must be strong. On
    bulk water at 2 V/A, friction 0.01 (tau_t about 1000 fs) ran away and
    friction 0.5 (tau_t about 20 fs) was stable.

    The charge head is trained without charge labels, so its sign is not
    guaranteed. The mean charge per element is printed at the start: check
    that it is physical (O negative in water). A flipped sign reverses every
    field force.

KNOWN LIMITATIONS
    - Positions stay unwrapped (diffusion needs them); deepmd-lr folds them into
      the cell at every halo rebuild, so it does not matter how far atoms drift.
    - NPT repartitions every step, and a field cannot be combined with NPT.
    - There is no conserved quantity for Langevin, NPT or field runs.

EXIT CODES
    0  finished, or stopped cleanly with a checkpoint (wall time or signal)
    1  bad input, with a message saying what to change
    2  the run became unstable (non-finite energy, or T above --max-temperature)

FOR AI AGENTS
    Recommended order:
    1. `python run_md.py --help` for every option.
    2. A smoke run (QUICK START 1): it must finish and the temperature must hover
       near the target.
    3. An nve run of a few ps: check the printed drift.
    4. --benchmark, to choose --gpus.
    5. Production with --max-hours, resubmitting with --restart until done.

    Code map:
    - The MD itself is deepmd_lr.nvalchemi: VelocityVerlet, Langevin,
      NoseHooverChainNVT, IsotropicMTKNPT, FIRE2, maxwell_boltzmann. This script
      turns the command line into those objects (build_md, output_settings)
      and adds what a job needs around them.
    - run_md(): a new run or a --restart; pre_relax(): the FIRE2 stage before
      the MD, with the cell under --relax-cell; benchmark(): the GPU-count
      timing.
    - New features belong in run_md(), after the MD object is built, e.g. as
      an observer: md.attach(fn, interval).
"""

from __future__ import annotations

import argparse
import collections
import datetime
import gc
import json
import math
import os
import pathlib
import platform
import signal
import subprocess
import sys
import time
from importlib import metadata

import ase.io
import numpy as np
import torch
from ase import units

from deepmd_lr.ase import load_model
from deepmd_lr.nvalchemi import (
    FIRE2,
    IsotropicMTKNPT,
    Langevin,
    MDInstabilityError,
    NoseHooverChainNVT,
    RelaxationError,
    VelocityVerlet,
    build_evaluator,
    maxwell_boltzmann,
    read_structure,
)

#: Settings a restart must share with the checkpoint it continues.
PHYSICS_KEYS = ("ensemble", "thermostat", "dt", "temperature", "tau_t",
                "pressure", "tau_p", "chain_length", "efield", "eps_inf",
                "charge_spin")


class InputError(Exception):
    """A problem with the input that the user has to fix (exit code 1)."""


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


class _Formatter(argparse.RawDescriptionHelpFormatter,
                 argparse.ArgumentDefaultsHelpFormatter):
    pass


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)

    g = ap.add_argument_group("system")
    g.add_argument("--model", required=True,
                   help="training checkpoint, e.g. model.ckpt.pt")
    g.add_argument("--head", default=None,
                   help="branch name, for a multi-head checkpoint")
    g.add_argument("--structure", required=True,
                   help="starting structure in any format ase.io.read "
                        "understands (extxyz, POSCAR, CIF, .traj); needs a cell")
    g.add_argument("--index", type=int, default=-1,
                   help="frame to take from a multi-frame file")
    g.add_argument("--repeat", type=int, nargs=3, default=[1, 1, 1],
                   metavar=("NX", "NY", "NZ"), help="replicate the cell")
    g.add_argument("--charge-spin", type=float, nargs=2, default=None,
                   metavar=("Q", "MULT"),
                   help="total charge (e) and spin multiplicity, for models "
                        "trained with that condition; default: the model's own")

    g = ap.add_argument_group("hardware")
    g.add_argument("--gpus", type=int, default=1,
                   help="number of GPUs, cuda:0 .. cuda:N-1 as numbered by "
                        "CUDA_VISIBLE_DEVICES")
    g.add_argument("--devices", default=None,
                   help="explicit comma-separated devices, e.g. cuda:0,cuda:2; "
                        "overrides --gpus. A repeated device runs several "
                        "partitions on one GPU, which is only useful for tests")
    g.add_argument("--compress", nargs="?", const=True, default=False,
                   type=float, metavar="STRIDE",
                   help="compress the DPA4C descriptor (much faster; needs "
                        "deepmd-kit built with the CUDA operators); optional "
                        "table stride in A")
    g.add_argument("--skin", default="auto",
                   help="halo skin in A, or 'auto' (sized from the cell, the "
                        "cutoff, the GPU count and the GPU memory)")
    g.add_argument("--memory-headroom", type=float, default=0.8,
                   help="fraction of free GPU memory the auto skin may plan for")

    g = ap.add_argument_group("ensemble")
    g.add_argument("--ensemble", choices=("nve", "nvt", "npt"), default="nvt",
                   help="see ENSEMBLES above")
    g.add_argument("--thermostat", choices=("langevin", "nhc"), default="langevin",
                   help="NVT thermostat")
    g.add_argument("--temperature", type=float, default=300.0,
                   help="K; thermostat target, and the temperature of the "
                        "initial velocities")
    g.add_argument("--tau-t", type=float, default=100.0,
                   help="thermostat time constant, fs (Langevin friction = "
                        "1/tau_t)")
    g.add_argument("--pressure", type=float, default=1e-4,
                   help="target pressure of NPT and of --relax-cell, GPa "
                        "(1 bar = 1e-4 GPa)")
    g.add_argument("--tau-p", type=float, default=1000.0,
                   help="NPT barostat time constant, fs")
    g.add_argument("--chain-length", type=int, default=3,
                   help="Nose-Hoover chain length (nhc and npt)")
    g.add_argument("--efield", type=float, nargs=3, default=None,
                   metavar=("EX", "EY", "EZ"),
                   help="uniform electric field, V/A (NVT only; see FIELD RUNS)")
    g.add_argument("--eps-inf", type=float, default=1.0,
                   help="high-frequency dielectric constant; scales charges, "
                        "Born charges and the field force by sqrt(eps_inf)")

    g = ap.add_argument_group("run control")
    g.add_argument("--steps", type=int, default=1000,
                   help="TOTAL number of MD steps; --restart continues up to it")
    g.add_argument("--dt", type=float, default=0.5, help="timestep, fs")
    g.add_argument("--seed", type=int, default=None,
                   help="seed for the initial velocities and the Langevin "
                        "noise; default: random, recorded in run.json")
    g.add_argument("--keep-velocities", action="store_true",
                   help="take the initial velocities from the structure file "
                        "(e.g. a previous final.extxyz) instead of drawing "
                        "them; any centre-of-mass drift is removed")
    g.add_argument("--relax-steps", type=int, default=0,
                   help="FIRE2 relaxation before MD, of the positions (and with "
                        "--relax-cell the cell); 0 = off. Recommended for "
                        "freshly built boxes")
    g.add_argument("--relax-fmax", type=float, default=0.05,
                   help="FIRE2 convergence criterion, eV/A")
    g.add_argument("--relax-cell", choices=("isotropic", "orthorhombic", "full"),
                   default=None,
                   help="also relax the cell, toward --pressure: isotropic "
                        "keeps the shape (liquids), orthorhombic moves the "
                        "three lengths of a diagonal cell, full all nine "
                        "entries (crystals)")
    g.add_argument("--relax-smax", type=float, default=0.01,
                   help="with --relax-cell, how close the stress must come to "
                        "--pressure, GPa")
    g.add_argument("--no-relax-checks", action="store_true",
                   help="continue even if the relaxation did not converge or "
                        "changed covalent bonds")
    g.add_argument("--max-temperature", type=float, default=None,
                   help="stop as unstable above this temperature, K; default "
                        "max(5 T, 1500); 0 disables")
    g.add_argument("--max-hours", type=float, default=0.0,
                   help="wall-time budget: checkpoint and exit cleanly after "
                        "this many hours (set it below the job limit); 0 = none")
    g.add_argument("--restart", action="store_true",
                   help="continue the run in --out from its checkpoint.npz")
    g.add_argument("--overwrite", action="store_true",
                   help="start a new run in an --out that already holds one")
    g.add_argument("--benchmark", action="store_true",
                   help="time 1, 2, 4, ... GPUs up to --gpus on this system "
                        "with these settings, print a table and exit")
    g.add_argument("--benchmark-steps", type=int, default=50,
                   help="timed steps per GPU count, after a warm-up")

    g = ap.add_argument_group("output")
    g.add_argument("--out", default="md_run", help="output directory")
    g.add_argument("--log-every", type=int, default=100,
                   help="steps between log rows (each costs a GPU sync)")
    g.add_argument("--traj-every", type=int, default=1000,
                   help="steps between trajectory frames; 0 = no trajectory")
    g.add_argument("--traj-format", choices=("extxyz", "traj"), default="extxyz")
    g.add_argument("--traj-wrapped", action="store_true",
                   help="wrap trajectory positions into the cell (default: "
                        "unwrapped, as diffusion analysis needs)")
    g.add_argument("--traj-velocities", action="store_true",
                   help="also store velocities in the trajectory")
    g.add_argument("--traj-forces", action="store_true",
                   help="also store forces in the trajectory")
    g.add_argument("--traj-charges", action="store_true",
                   help="also store per-atom charges (long-range models only)")
    g.add_argument("--checkpoint-every", type=int, default=10000,
                   help="steps between checkpoints")
    a = ap.parse_args(argv)
    check_args(a)
    return a


def check_args(a: argparse.Namespace) -> None:
    """Refuse combinations that cannot work, before any GPU time is spent."""
    if a.efield is not None and a.ensemble != "nvt":
        raise InputError(
            "--efield needs --ensemble nvt: the evaluator refuses a field with "
            "the virial NPT needs, and under a field NVE conserves nothing")
    if a.ensemble == "npt" and a.chain_length < 2:
        raise InputError("--ensemble npt needs --chain-length >= 2")
    if a.relax_cell is not None:
        if a.relax_steps < 1:
            raise InputError("--relax-cell relaxes during --relax-steps; set that "
                             "as well")
        if a.efield is not None:
            raise InputError("--relax-cell needs the virial, which the evaluator "
                             "refuses under --efield")
        if a.relax_smax <= 0:
            raise InputError("--relax-smax must be positive")
    if not 1 <= a.chain_length <= 8:
        raise InputError("--chain-length must be between 1 and 8")
    for name in ("steps", "log_every", "checkpoint_every", "gpus",
                 "benchmark_steps"):
        if getattr(a, name) < 1:
            raise InputError(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("dt", "tau_t", "tau_p", "temperature"):
        if getattr(a, name) <= 0:
            raise InputError(f"--{name.replace('_', '-')} must be positive")
    if a.restart and a.benchmark:
        raise InputError("--restart and --benchmark do not go together")
    if a.restart and a.overwrite:
        raise InputError("--restart continues a run; --overwrite replaces it. "
                         "Pick one.")
    if a.dt > 4.0:
        print(f"warning: --dt {a.dt} fs is unusually large (the unit is fs)",
              flush=True)


def resolve_devices(a: argparse.Namespace) -> list[str]:
    n_visible = torch.cuda.device_count()
    if n_visible == 0:
        raise InputError("no CUDA device is visible; this script needs a GPU")
    if a.devices:
        devices = [d.strip() for d in a.devices.split(",") if d.strip()]
        for d in devices:
            try:
                index = torch.device(d).index
            except RuntimeError:
                index = None
            if not d.startswith("cuda") or index is None or index >= n_visible:
                raise InputError(f"device {d!r} is not one of the {n_visible} "
                                 "visible CUDA devices (use cuda:<index>)")
        return devices
    if a.gpus > n_visible:
        raise InputError(f"--gpus {a.gpus}, but only {n_visible} GPU(s) are "
                         "visible (check CUDA_VISIBLE_DEVICES / the job request)")
    return [f"cuda:{i}" for i in range(a.gpus)]


# --------------------------------------------------------------------------
# System, model and evaluator
# --------------------------------------------------------------------------


def load_system(a: argparse.Namespace, device: str):
    """Read the structure and the model, and check that they fit together."""
    try:
        atoms = read_structure(a.structure, a.index)
    except (OSError, ValueError, IndexError) as exc:
        raise InputError(f"cannot use --structure {a.structure}: {exc}") from exc
    if list(a.repeat) != [1, 1, 1]:
        atoms = atoms.repeat(tuple(a.repeat))
    try:
        model = load_model(a.model, device=device, head=a.head)
    except (OSError, KeyError, ValueError) as exc:
        raise InputError(f"cannot load --model {a.model}: {exc}") from exc
    type_map = list(model.get_type_map())
    missing = sorted(set(atoms.get_chemical_symbols()) - set(type_map))
    if missing:
        raise InputError(f"element(s) {missing} are not in the model's type_map "
                         f"{type_map}; this model cannot describe the system")
    has_lr = bool(getattr(model, "has_long_range", False))
    if not has_lr and (a.efield is not None or a.traj_charges):
        raise InputError("--efield and --traj-charges need a model with a "
                         "long-range (charge) head; this one is short-range only")
    return atoms, model


def make_evaluator(model, atoms, devices, a: argparse.Namespace):
    kwargs = {
        "devices": devices,
        "skin": "auto" if a.skin == "auto" else float(a.skin),
        "memory_headroom": a.memory_headroom,
        "compute_virial": a.ensemble == "npt",   # the barostat needs the virial
        "eps_inf": a.eps_inf,
    }
    if a.efield is not None:
        kwargs["efield"] = list(a.efield)
    if a.charge_spin is not None:
        kwargs["charge_spin"] = tuple(a.charge_spin)
    if a.compress:
        kwargs["compress"] = a.compress         # True, or a table stride
    try:
        return build_evaluator(model, atoms, **kwargs)
    except (ValueError, NotImplementedError) as exc:
        raise InputError(f"cannot build the evaluator: {exc}") from exc


def md_class(a: argparse.Namespace):
    if a.ensemble == "nve":
        return VelocityVerlet
    if a.ensemble == "npt":
        return IsotropicMTKNPT
    return Langevin if a.thermostat == "langevin" else NoseHooverChainNVT


def max_temperature(a: argparse.Namespace) -> float | None:
    if a.max_temperature is None:
        return max(5.0 * a.temperature, 1500.0)
    return a.max_temperature or None                     # 0 disables


def output_settings(a: argparse.Namespace, out: pathlib.Path) -> dict:
    """Where the MD object writes; a restart passes the same settings back."""
    props = ["energy"]
    props += ["forces"] if a.traj_forces else []
    props += ["velocities"] if a.traj_velocities else []
    props += ["charges"] if a.traj_charges else []
    return {
        "logfile": out / "md.log",
        "loginterval": a.log_every,
        "trajectory": out / f"traj.{a.traj_format}" if a.traj_every > 0 else None,
        "traj_interval": a.traj_every if a.traj_every > 0 else None,
        "trajectory_properties": tuple(props),
        "traj_wrapped": a.traj_wrapped,
        "checkpoint": out / "checkpoint.npz",
        "checkpoint_interval": a.checkpoint_every,
        "max_temperature": max_temperature(a),
        "echo_log": True,
    }


def build_md(a: argparse.Namespace, ev, atoms, seed: int, io: dict):
    """The command line's units (fs, K, GPa) turned into ASE's."""
    dt, cls = a.dt * units.fs, md_class(a)
    if cls is VelocityVerlet:
        return cls(ev, atoms, dt, **io)
    if cls is Langevin:
        return cls(ev, atoms, dt, a.temperature, 1.0 / (a.tau_t * units.fs),
                   seed=seed, **io)
    if cls is NoseHooverChainNVT:
        return cls(ev, atoms, dt, a.temperature, a.tau_t * units.fs,
                   tchain=a.chain_length, **io)
    return cls(ev, atoms, dt, a.temperature, a.pressure * units.GPa,
               a.tau_t * units.fs, a.tau_p * units.fs, tchain=a.chain_length, **io)


def physics_of(a: argparse.Namespace) -> dict:
    """The settings a restart must keep, minus those this ensemble ignores."""
    d = {k: getattr(a, k) for k in PHYSICS_KEYS}
    if a.ensemble != "nvt":
        d.pop("thermostat")
    if a.ensemble != "npt":
        d.pop("pressure")
        d.pop("tau_p")
    if a.ensemble == "nve":
        for k in ("temperature", "tau_t", "chain_length"):
            d.pop(k)
    if a.ensemble == "nvt" and a.thermostat == "langevin":
        d.pop("chain_length")
    return d


def checkpoint_meta(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    if meta.get("format", "").startswith("deepmd-lr examples/07"):
        raise InputError(
            f"{path} was written by an older run_md.py (before the MD classes); "
            "it cannot be continued. Start a new run from <out>/final.extxyz or "
            "the last trajectory frame with --keep-velocities")
    return meta


def refuse_changed_physics(meta: dict, a: argparse.Namespace) -> None:
    old = meta.get("metadata", {}).get("cli_physics")
    if old is None:
        raise InputError("the checkpoint was not written by run_md.py; restore it "
                         "with the MD class's from_checkpoint instead")
    now = physics_of(a)
    diff = [f"{k}: {old.get(k)!r} -> {v!r}" for k, v in now.items() if old.get(k) != v]
    diff += [f"{k}: {v!r} -> (unset)" for k, v in old.items() if k not in now]
    if diff:
        raise InputError(
            "a restart must keep the physics of the run it continues; changed: "
            + "; ".join(diff) + ". To change them, start a new run from "
            "<out>/final.extxyz (or the last trajectory frame) with "
            "--keep-velocities and a new --out")


def environment(devices) -> dict:
    versions = {}
    for dist in ("deepmd-lr", "deepmd-kit", "torch", "nvalchemi-toolkit-ops",
                 "warp-lang", "ase", "numpy"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    try:
        import deepmd_lr

        commit = subprocess.run(
            ["git", "-C", str(pathlib.Path(deepmd_lr.__file__).parent),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "versions": versions,
        "deepmd_lr_commit": commit,
        "gpus": [torch.cuda.get_device_name(torch.device(d)) for d in devices],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def report_charges(ev, symbols, a) -> None:
    """Mean charge per element: the sanity check a field run needs."""
    if not getattr(ev.model, "has_long_range", False):
        return
    q = np.asarray(ev.results.get("charge")).reshape(-1)
    sym = np.array(symbols)
    parts = [f"{el} {q[sym == el].mean():+.3f}" for el in sorted(set(symbols))]
    print("  mean charge per element (e): " + ", ".join(parts), flush=True)
    if a.efield is not None:
        print("  field run: check these signs are physical (e.g. O negative in "
              "water); a flipped sign reverses every field force", flush=True)


class StopFlag:
    """SIGTERM/SIGUSR1/SIGINT ask the MD object to stop at the end of the
    current step, so the checkpoint is never taken halfway through a step. A
    second SIGINT quits immediately."""

    SIGNALS = (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)

    def __init__(self):
        self.reason = None
        self.md = None

    def __call__(self, signum, frame):
        if self.reason is not None and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.reason = signal.Signals(signum).name
        print(f"\n  {self.reason} received: stopping after this step "
              "(send SIGINT again to quit at once)", flush=True)
        if self.md is not None:
            self.md.stop()

    def install(self) -> None:
        self._previous = {sig: signal.signal(sig, self) for sig in self.SIGNALS}

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


# --------------------------------------------------------------------------
# The two modes
# --------------------------------------------------------------------------


def run_md(a: argparse.Namespace) -> int:
    out = pathlib.Path(a.out)
    ckpt_path, log_path = out / "checkpoint.npz", out / "md.log"
    if a.restart and not ckpt_path.exists():
        raise InputError(f"--restart: {ckpt_path} does not exist")
    if not a.restart and not a.overwrite and (log_path.exists() or ckpt_path.exists()):
        raise InputError(f"{out} already holds a run. Continue it with --restart, "
                         "replace it with --overwrite, or choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    if a.overwrite:
        # An old checkpoint left behind would be picked up by a later --restart
        # of this run if it died before writing its own.
        for name in ("checkpoint.npz", "final.extxyz", "blowup.extxyz",
                     "traj.extxyz", "traj.traj"):
            (out / name).unlink(missing_ok=True)

    devices = resolve_devices(a)
    atoms, model = load_system(a, devices[0])
    symbols = atoms.get_chemical_symbols()
    n = len(atoms)
    meta = None
    if a.restart:
        meta = checkpoint_meta(ckpt_path)
        refuse_changed_physics(meta, a)
        if meta["nsteps"] >= a.steps:
            print(f"{out}: already at step {meta['nsteps']} >= --steps {a.steps}; "
                  "nothing to do (raise --steps to extend the run)")
            return 0
    seed = a.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little") % 2**31

    counts = sorted(collections.Counter(symbols).items())
    comp = " ".join(f"{el} {c}" for el, c in counts)
    cellpar = atoms.cell.cellpar()
    print(f"deepmd-lr MD | {md_class(a).__name__} | T {a.temperature:g} K | "
          f"dt {a.dt:g} fs | steps {meta['nsteps'] if meta else 0} -> {a.steps}",
          flush=True)
    print(f"  system   {n} atoms ({comp}); cell {cellpar[0]:.3f} {cellpar[1]:.3f} "
          f"{cellpar[2]:.3f} A, {cellpar[3]:.1f} {cellpar[4]:.1f} {cellpar[5]:.1f} deg",
          flush=True)
    print(f"  model    {a.model} (long-range: "
          f"{'yes' if getattr(model, 'has_long_range', False) else 'no'}, "
          f"rcut {model.get_rcut():g} A, compress: {a.compress or 'no'})", flush=True)
    print(f"  devices  {', '.join(devices)} ({n // len(devices)} atoms per partition)",
          flush=True)
    if len(set(devices)) > 1 and n < 100_000:
        print("  tip: below ~100k atoms more GPUs are not always faster "
              "(especially with --compress); --benchmark measures it", flush=True)

    ev = make_evaluator(model, atoms, devices, a)
    io = output_settings(a, out)
    try:
        if a.restart:
            md = md_class(a).from_checkpoint(ev, atoms, ckpt_path, **io)
            if a.seed is not None and getattr(md, "seed", a.seed) != a.seed:
                print(f"note: --seed ignored on restart; the run keeps seed {md.seed}")
        else:
            kept = atoms.get_velocities() if a.keep_velocities else None
            if kept is not None and not np.any(kept):
                print("note: the structure file has no velocities; drawing them at "
                      f"{a.temperature} K instead", flush=True)
                kept = None
            if a.relax_steps > 0:
                pre_relax(a, ev, atoms)
            if kept is not None:
                atoms.set_velocities(kept)
            else:
                maxwell_boltzmann(atoms, a.temperature, seed=seed)
            md = build_md(a, ev, atoms, seed, io)
            md.metadata["cli_physics"] = physics_of(a)
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    seed = getattr(md, "seed", seed)
    print(f"  ensemble {type(md).__name__}; seed {seed}", flush=True)

    t0 = time.perf_counter()
    md.thermo.evaluate()
    print(f"  first force call {time.perf_counter() - t0:.1f} s (kernel "
          f"compilation, memory probe); skin {ev.skin:.2f} A", flush=True)
    report_charges(ev, symbols, a)

    run_json = out / "run.json"
    record = (json.loads(run_json.read_text())
              if a.restart and run_json.exists() else {"segments": []})
    segment = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "arguments": vars(a),
        "devices": devices,
        "seed": seed,
        "start_step": md.nsteps,
        "n_atoms": n,
        "composition": dict(collections.Counter(symbols)),
        "environment": environment(devices),
        "skin_A": ev.skin,
    }
    record["segments"].append(segment)
    run_json.write_text(json.dumps(record, indent=2, default=str))

    stop = StopFlag()
    stop.md = md
    stop.install()
    deadline = time.monotonic() + a.max_hours * 3600 if a.max_hours > 0 else None

    def wall_time():
        if deadline and time.monotonic() > deadline and stop.reason is None:
            stop.reason = "wall-time budget"
            md.stop()

    md.attach(wall_time, 1, sync_atoms=False)
    try:
        md.run(a.steps - md.nsteps)
    except MDInstabilityError as exc:
        ase.io.write(out / "blowup.extxyz", md.atoms)
        print(f"\n*** UNSTABLE: {exc}\n"
              f"    last configuration: {out / 'blowup.extxyz'}\n"
              "    Usual causes: timestep too large, an unrelaxed start "
              "(--relax-steps), a thermostat too weak for a field, or a "
              "structure outside the model's training data.", flush=True)
        segment.update(status=f"unstable: {exc}", end_step=md.nsteps)
        run_json.write_text(json.dumps(record, indent=2, default=str))
        return 2
    finally:
        stop.restore()
    if md.nsteps >= a.steps:
        md.save_checkpoint()
        status = "finished"
    else:
        status = (f"stopped at step {md.nsteps} ({stop.reason}); "
                  "continue with --restart")
    md.close()
    ase.io.write(out / "final.extxyz", md.atoms)

    summary = summarize(md, ev, a, n, devices)
    print(f"\n{status}\n" + "\n".join(f"  {k}: {v}" for k, v in summary.items()),
          flush=True)
    segment.update(status=status, end_step=md.nsteps, summary=summary)
    run_json.write_text(json.dumps(record, indent=2, default=str))
    return 0


def _cell_text(atoms) -> str:
    c = atoms.cell.cellpar()
    return (" ".join(f"{x:.3f}" for x in c[:3]) + " A, "
            + " ".join(f"{x:.1f}" for x in c[3:]) + " deg")


def pre_relax(a: argparse.Namespace, ev, atoms) -> None:
    """FIRE2 before the MD: the positions, and with --relax-cell the cell.

    The virial is switched on for this stage only; an NVT or NVE run after it
    does not pay for it.
    """
    cell = a.relax_cell
    before, v0 = _cell_text(atoms), atoms.get_volume()
    if cell is not None:
        ev.set_compute_virial(True)
    FIRE2(ev, atoms, relax_cell=cell,
          pressure_au=a.pressure * units.GPa if cell is not None else 0.0,
          logfile="-", loginterval=max(1, a.relax_steps // 10),
          check=not a.no_relax_checks).run(fmax=a.relax_fmax, steps=a.relax_steps,
                                           smax=a.relax_smax * units.GPa)
    if cell is not None:
        ev.set_compute_virial(a.ensemble == "npt")
        print(f"  cell ({cell}): {before} -> {_cell_text(atoms)}; "
              f"volume {100 * (atoms.get_volume() / v0 - 1):+.2f} %", flush=True)


def summarize(md, ev, a, n, devices) -> dict:
    s = {}
    if math.isfinite(md.seconds_per_step):
        s["speed"] = (f"{md.seconds_per_step * 1e3:.1f} ms/step = "
                      f"{md.ns_per_day:.3f} ns/day ({n} atoms on {len(devices)} "
                      "device(s))")
    s["halo"] = (f"skin {ev.skin:.2f} A, {ev.rebuild_count} rebuilds, "
                 f"{ev.skin_backoffs} out-of-memory backoffs")
    rows = md.rows
    if len(rows) >= 2:
        half = rows[len(rows) // 2:]
        s["mean T (second half)"] = f"{np.mean([r['T[K]'] for r in half]):.1f} K"
        if a.ensemble == "npt":
            p_mean = np.mean([r["P[GPa]"] for r in half])
            v_mean = np.mean([r["V[A^3]"] for r in half])
            s["mean P (second half)"] = f"{p_mean:+.5f} GPa"
            s["mean V (second half)"] = f"{v_mean:.2f} A^3"
        key = "Etot[eV]" if a.ensemble == "nve" else (
            "Econs[eV]" if "Econs[eV]" in rows[0] else None)
        if key is not None:
            span = rows[-1]["time[ps]"] - rows[0]["time[ps]"]
            if span > 0:
                # A fit over a short stretch mostly measures the fluctuation:
                # 4 rows over 0.15 ps read +0.43 where 0.5 ps gave +0.0095.
                short = span < 0.5 or len(rows) < 20
                s[f"{key.split('[')[0]} drift"] = (
                    f"{md.drift(key):+.4f} meV/atom/ps over {span:.3g} ps, "
                    f"{len(rows)} log rows"
                    + ("; too short to judge: use >= 0.5 ps and >= 20 rows"
                       if short else ""))
            if a.ensemble == "nve":
                e = np.array([r[key] for r in rows]) * 1e3 / n
                s["Etot fluctuation"] = (f"{np.std(e):.4f} meV/atom rms (falls "
                                         "~4x when dt is halved)")
    return s


def _time_devices(a: argparse.Namespace, devices: list[str]):
    """ms/step for one device list, or NaN. Everything it builds is local, so
    it is all freed on return, before the next device count is built."""
    atoms, model = load_system(a, devices[0])
    ev = make_evaluator(model, atoms, devices, a)
    maxwell_boltzmann(atoms, a.temperature, seed=1)
    md = build_md(a, ev, atoms, seed=1, io={})
    md.thermo.evaluate()
    md.run(max(10, a.benchmark_steps // 5))       # compilation, memory probe
    for d in set(devices):
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    md.run(a.benchmark_steps)
    for d in set(devices):
        torch.cuda.synchronize(d)
    ms = (time.perf_counter() - t0) / a.benchmark_steps * 1e3
    return (ms if math.isfinite(md.thermo.energy) else float("nan")), ev.skin


def benchmark(a: argparse.Namespace) -> int:
    """Time the same MD step on 1, 2, 4, ... devices and print a table."""
    devices_all = resolve_devices(a)
    counts = sorted({c for c in (1, 2, 4, 8, 16) if c <= len(devices_all)}
                    | {len(devices_all)})
    print(f"benchmark: {a.benchmark_steps} timed steps per device count, "
          f"ensemble {a.ensemble}, compress {a.compress or 'no'}", flush=True)
    results = []
    for count in counts:
        try:
            ms, skin = _time_devices(a, devices_all[:count])
            note = "" if math.isfinite(ms) else "  (UNSTABLE: non-finite energy)"
        except torch.cuda.OutOfMemoryError:
            ms, skin, note = float("nan"), float("nan"), "  (out of memory)"
        except MDInstabilityError as exc:
            ms, skin, note = float("nan"), float("nan"), f"  (UNSTABLE: {exc})"
        gc.collect()
        torch.cuda.empty_cache()
        results.append((count, ms, skin))
        print(f"  {count} device(s): {ms:8.1f} ms/step{note}", flush=True)

    good = [r for r in results if math.isfinite(r[1])]
    if not good:
        print("no device count completed; see the messages above")
        return 2
    base = next((r[1] for r in good if r[0] == 1), None)
    print(f"\n{'devices':>8} {'ms/step':>9} {'ns/day':>8} {'speed-up':>9} "
          f"{'skin A':>7}")
    for count, ms, skin in results:
        if not math.isfinite(ms):
            print(f"{count:>8} {'failed':>9}")
            continue
        speedup = f"{base / ms:.2f}x" if base else "-"
        print(f"{count:>8} {ms:9.1f} {a.dt * 86.4 / ms:8.3f} {speedup:>9} "
              f"{skin:7.2f}")
    best = min(good, key=lambda r: r[1])
    print(f"\nfastest: {best[0]} device(s). Rebuilds are rare in a short "
          "benchmark, so a long run is a little slower than this.")
    return 0


def main(argv=None) -> int:
    try:
        a = parse_args(argv)
        return benchmark(a) if a.benchmark else run_md(a)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    except RelaxationError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
