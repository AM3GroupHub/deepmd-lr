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
       Pre-relax such a box with a classical force field instead.
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
    - VelocityVerlet, Langevin, NoseHooverChain, IsotropicMTK: one step each.
    - relax(): the FIRE pre-relaxation.
    - Logger, TrajectoryWriter, save_checkpoint() and load_checkpoint():
      output and restarts.
    - run_md() and benchmark(): the two top-level modes.
    - New features belong in run_md(), after the evaluator is built, the same
      way the existing options are handled.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import gc
import hashlib
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
import warp as wp
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from nvalchemiops.dynamics.integrators import (
    langevin_baoab_finalize,
    langevin_baoab_half_step,
    nhc_compute_chain_energy,
    nhc_compute_masses,
    nhc_position_update,
    nhc_thermostat_chain_update,
    nhc_velocity_half_step,
    npt_barostat_half_step,
    npt_cell_update,
    npt_position_update,
    npt_thermostat_half_step,
    npt_velocity_half_step,
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)
from nvalchemiops.dynamics.optimizers import fire2_step

from deepmd_lr.ase import load_model
from deepmd_lr.nvalchemi import Thermo, build_evaluator, read_structure

#: Bumped whenever the checkpoint layout changes.
CHECKPOINT_VERSION = 1
#: Settings a restart must share with the checkpoint it continues.
PHYSICS_KEYS = ("ensemble", "thermostat", "dt", "temperature", "tau_t",
                "pressure", "tau_p", "chain_length", "efield", "eps_inf",
                "charge_spin")
#: Elements whose mutual covalent bonds the relaxation check guards. Metals are
#: left out on purpose: an ion's coordination shell may legitimately change.
NONMETALS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}


class InputError(Exception):
    """A problem with the input that the user has to fix (exit code 1)."""


class Unstable(Exception):
    """The simulation left physical territory (exit code 2)."""


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
                   help="NPT target pressure, GPa (1 bar = 1e-4 GPa)")
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
                   help="FIRE relaxation of positions (cell fixed) before MD; "
                        "0 = off. Recommended for freshly built boxes")
    g.add_argument("--relax-fmax", type=float, default=0.05,
                   help="FIRE convergence criterion, eV/A")
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


def fingerprint(model, symbols) -> dict:
    """What identifies the system a checkpoint belongs to."""
    return {
        "n_atoms": len(symbols),
        "symbols_sha1": hashlib.sha1(",".join(symbols).encode()).hexdigest(),
        "type_map": list(model.get_type_map()),
        "rcut": float(model.get_rcut()),
    }


# --------------------------------------------------------------------------
# Integrators: each step is the kernels around exactly one force evaluation.
# Every kernel takes the FULL timestep and applies its own factor 1/2.
# --------------------------------------------------------------------------


class Integrator:
    name = "?"

    def __init__(self, th: Thermo, evaluate):
        self.th = th
        self.evaluate = evaluate

    def prepare(self) -> None:
        """Called once after the first force evaluation."""

    def step(self, index: int) -> None:
        """Advance from step `index` to `index + 1`."""
        raise NotImplementedError

    def conserved(self) -> float | None:
        """A conserved extended energy, where the ensemble has one."""
        return None

    def state(self) -> dict:
        """Integrator variables for the checkpoint."""
        return {}

    def load_state(self, arrays: dict) -> None:
        pass


class VelocityVerlet(Integrator):
    name = "NVE, velocity Verlet"

    def step(self, index):
        th = self.th
        velocity_verlet_position_update(th.positions, th.velocities, th.forces,
                                        th.masses, th.dt)
        self.evaluate()
        velocity_verlet_velocity_finalize(th.velocities, th.forces, th.masses,
                                          th.dt)


class Langevin(Integrator):
    """BAOAB Langevin. The noise seed is `seed + step`, so a restart repeats
    exactly the noise an uninterrupted run would have drawn."""

    def __init__(self, th, evaluate, tau_t_fs, seed):
        super().__init__(th, evaluate)
        self.name = f"NVT, Langevin BAOAB, tau_t {tau_t_fs:g} fs"
        self.friction = th.scalar(1.0 / (tau_t_fs * units.fs))
        self.seed = int(seed)

    def step(self, index):
        th = self.th
        langevin_baoab_half_step(th.positions, th.velocities, th.forces,
                                 th.masses, th.dt, th.kT, self.friction,
                                 (self.seed + index) % 2**31)
        self.evaluate()
        langevin_baoab_finalize(th.velocities, th.forces, th.masses, th.dt)


class NoseHooverChain(Integrator):
    """Nose-Hoover chain NVT (Martyna-Tuckerman splitting).

    Conventions verified on water (bend ~1640, stretch ~3400 cm-1):
    - The full dt goes to every kernel. Passing dt/2 to the velocity kick
      halves every force.
    - ndof = 3N - 3, since the centre of mass is fixed.
    - total_scale is reset to 1 before each chain update.
    - The conserved energy uses eta/2: the kernel advances eta at twice the
      rate of eta_dot. eta never feeds back into the dynamics, so the
      trajectory itself is right.
    """

    NLOOPS = 3          # Yoshida-Suzuki stages per chain update

    def __init__(self, th, evaluate, tau_t_fs, chain_length):
        super().__init__(th, evaluate)
        self.name = (f"NVT, Nose-Hoover chain, tau_t {tau_t_fs:g} fs, "
                     f"chain {chain_length}")
        ndof = 3 * th.n_atoms - 3
        self.ndof = th.scalar(float(ndof))                         # float64
        ndof_i32 = wp.array(np.array([ndof]), dtype=wp.int32, device=th.device)
        self.L = chain_length
        self.eta = th.zeros(self.L)
        self.eta_dot = th.zeros(self.L)
        self.eta_mass = th.zeros(self.L)
        nhc_compute_masses(ndof_i32, th.kT, th.scalar(tau_t_fs * units.fs),
                           self.L, self.eta_mass)
        self.ke2 = th.zeros(1)
        self.total_scale = th.scalar(1.0)
        self.step_scale = th.zeros(1)
        self.dt_chain = th.zeros(1)
        self.ke_chain = th.zeros(1)
        self.pe_chain = th.zeros(1)

    def _chain(self):
        th = self.th
        self.total_scale.fill_(1.0)
        nhc_thermostat_chain_update(
            th.velocities, th.masses, self.eta, self.eta_dot, self.eta_mass,
            th.kT, th.dt, self.ndof, self.ke2, self.total_scale,
            self.step_scale, self.dt_chain, nloops=self.NLOOPS)

    def step(self, index):
        th = self.th
        self._chain()
        nhc_velocity_half_step(th.velocities, th.forces, th.masses, th.dt)
        nhc_position_update(th.positions, th.velocities, th.dt)
        self.evaluate()
        nhc_velocity_half_step(th.velocities, th.forces, th.masses, th.dt)
        self._chain()

    def conserved(self):
        th = self.th
        half_eta = wp.array(0.5 * self.eta.numpy(), dtype=wp.float64,
                            device=th.device)
        nhc_compute_chain_energy(half_eta, self.eta_dot, self.eta_mass, th.kT,
                                 self.ndof, self.ke_chain, self.pe_chain)
        return (th.total_energy + float(self.ke_chain.numpy()[0])
                + float(self.pe_chain.numpy()[0]))

    def state(self):
        return {"nhc_eta": self.eta.numpy(), "nhc_eta_dot": self.eta_dot.numpy()}

    def load_state(self, arrays):
        self.eta.assign(arrays["nhc_eta"])
        self.eta_dot.assign(arrays["nhc_eta_dot"])


class IsotropicMTK(Integrator):
    """Isotropic MTK NPT with a Nose-Hoover chain thermostat.

    The step is the palindrome thermostat, barostat, kick | drift | kick,
    barostat, thermostat. Three points are not in the kernel signatures:

    - Fresh inputs. The kinetic energy and pressure are refreshed right before
      each barostat/thermostat half-step, as LAMMPS' fix nh does. The barostat
      must see the current state, never a value cached earlier in the step. An
      earlier version reused the pressure from before the last kick, and
      because the Thermo.pressure getter rewrote that cache, logging changed
      the trajectory. With the refresh, the MTK conserved quantity is kept to
      O(dt^2) as before: 0.0218 -> 0.0054 meV/atom rms from dt 0.5 to 0.25 fs.
    - Degrees of freedom. The kernels compute N_f = 3 * (the count they are
      given). The centre of mass is at rest and stays there (forces sum to
      zero, and both drags scale velocities uniformly), so the moving degrees
      of freedom are 3N - 3: the kernels get N - 1. That matches the masses
      below and Thermo.temperature. Given N, they held the system 3N/(3N-3)
      hot, which is 0.5 % at 192 atoms.
    - The cell. After the barostat moves the cell, `th.set_cell()` pushes it
      to the evaluator, which invalidates the partitioning. That is correct
      and costs a repartition every step. Skipping it would leave ghost images
      at stale offsets, giving quietly wrong forces.
    """

    def __init__(self, th, evaluate, pressure_gpa, tau_t_fs, tau_p_fs,
                 chain_length):
        super().__init__(th, evaluate)
        self.name = (f"NPT, isotropic MTK, P {pressure_gpa:g} GPa, "
                     f"tau_t {tau_t_fs:g} fs, tau_p {tau_p_fs:g} fs, "
                     f"chain {chain_length}")
        self.L = chain_length
        kT = float(th.kT.numpy()[0])
        ndof = 3 * th.n_atoms - 3
        self.eta = th.zeros((1, self.L))
        self.eta_dot = th.zeros((1, self.L))
        q = np.full((1, self.L), kT * (tau_t_fs * units.fs) ** 2)
        q[0, 0] *= ndof                          # Q_0 = ndof kT tau^2
        self.th_mass = wp.array(q, dtype=wp.float64, device=th.device)
        self.cell_v = th.zeros(1, dtype=wp.mat33d)
        # W = (ndof + 3) kT tau_p^2
        self.cell_mass = th.scalar((ndof + 3) * kT * (tau_p_fs * units.fs) ** 2)
        self.target_p = th.scalar(pressure_gpa * units.GPa)
        # N - 1, so that the kernels' N_f = 3 * count is 3N - 3.
        self.dof_count = wp.array([th.n_atoms - 1], dtype=wp.int32,
                                  device=th.device)

    def _refresh(self):
        """Kinetic energy and pressure of the current state, for the kernels."""
        self.th.update_kinetic_energy()
        self.th.update_pressure()

    def _thermostat(self):
        th = self.th
        npt_thermostat_half_step(self.eta, self.eta_dot, th.ke_w, th.kT,
                                 self.th_mass, self.dof_count, self.L, th.dt)

    def _barostat(self):
        th = self.th
        npt_barostat_half_step(self.cell_v, th.press_w, self.target_p,
                               th.volume_w, self.cell_mass, th.ke_w,
                               self.dof_count, th.dt)

    def _kick(self):
        th = self.th
        npt_velocity_half_step(th.velocities, th.masses, th.forces, self.cell_v,
                               eta_dots=self.eta_dot, num_atoms=self.dof_count,
                               dt=th.dt)

    def step(self, index):
        th = self.th
        self._refresh()
        self._thermostat()
        self._barostat()
        self._kick()
        npt_position_update(th.positions, th.velocities, th.cell, self.cell_v,
                            th.dt)
        npt_cell_update(th.cell, self.cell_v, th.dt)
        th.set_cell()                                # evaluator + volume
        self.evaluate()
        self._kick()
        self._refresh()                              # after the kick
        self._barostat()
        self._thermostat()

    def state(self):
        return {"npt_eta": self.eta.numpy(), "npt_eta_dot": self.eta_dot.numpy(),
                "npt_cell_v": self.cell_v.numpy()}

    def load_state(self, arrays):
        self.eta.assign(arrays["npt_eta"])
        self.eta_dot.assign(arrays["npt_eta_dot"])
        self.cell_v.assign(arrays["npt_cell_v"])


def build_integrator(a, th, seed) -> Integrator:
    if a.ensemble == "nve":
        return VelocityVerlet(th, th.evaluate)
    if a.ensemble == "nvt" and a.thermostat == "langevin":
        return Langevin(th, th.evaluate, a.tau_t, seed)
    if a.ensemble == "nvt":
        return NoseHooverChain(th, th.evaluate, a.tau_t, a.chain_length)
    return IsotropicMTK(th, th.evaluate, a.pressure, a.tau_t,
                        a.tau_p, a.chain_length)


# --------------------------------------------------------------------------
# Relaxation
# --------------------------------------------------------------------------


def covalent_bonds(symbols, positions, cell) -> set:
    """Bonded pairs among non-metal atoms (ASE covalent radii x 1.2)."""
    from ase.neighborlist import natural_cutoffs, neighbor_list

    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
    atoms.wrap()
    i, j = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.2))
    sym = np.array(symbols)
    keep = (i < j) & np.isin(sym[i], list(NONMETALS)) & np.isin(sym[j], list(NONMETALS))
    return set(zip(i[keep].tolist(), j[keep].tolist(), strict=True))


def relax(th: Thermo, a: argparse.Namespace, symbols) -> None:
    """FIRE on the positions only; the cell never changes.

    A freshly packed box can hold tens of eV of clashes. Released by MD, that
    energy heats the system in the first few hundred fs and can push the model
    into regions it never saw.

    Aborts, whatever the flags, when it diverges: non-finite values, forces
    growing tenfold, or more than 2 eV/atom released. Those mean atoms are
    collapsing where the model has no repulsion. Refuses to go on (unless
    --no-relax-checks) when:
    - it did not converge;
    - the energy rose (checked only without a field, since under a field FIRE
      seeks zero total force, not minimum energy);
    - a covalent bond between non-metal atoms broke or formed.
    """
    n, dev = th.n_atoms, th.device
    vel = wp.zeros(n, dtype=wp.vec3d, device=dev)
    batch = wp.zeros(n, dtype=wp.int32, device=dev)
    alpha = wp.array([0.09], dtype=wp.float64, device=dev)
    fire_dt = wp.array([0.01], dtype=wp.float64, device=dev)
    n_inc = wp.zeros(1, dtype=wp.int32, device=dev)
    scratch = [wp.zeros(1, dtype=wp.float64, device=dev) for _ in range(4)]

    def fmax() -> float:
        return float(np.linalg.norm(th.forces.numpy(), axis=1).max())

    cell = np.asarray(th.evaluator.cell)
    bonds0 = covalent_bonds(symbols, th.positions.numpy(), cell)
    e0, f0 = th.energy, fmax()
    k, diverged = 0, None
    for k in range(1, a.relax_steps + 1):
        fire2_step(th.positions, vel, th.forces, batch, alpha, fire_dt, n_inc,
                   *scratch)
        th.evaluate()
        if not math.isfinite(th.energy):
            break
        if k % 10 == 0:
            f = fmax()
            if f < a.relax_fmax:
                break
            # A relaxation lowers the forces. Forces growing tenfold, or more
            # than 2 eV/atom released, means atoms are collapsing into a region
            # where the model has no repulsion.
            if f > max(10.0 * f0, 50.0) or (e0 - th.energy) / n > 2.0:
                diverged = (f"after {k} steps max|F| reached {f:.3g} eV/A and "
                            f"the energy fell {(e0 - th.energy) / n:.3g} eV/atom")
                break
    f1 = fmax()
    print(f"  FIRE relaxation: {k} steps, E {e0:.3f} -> {th.energy:.3f} eV "
          f"({(th.energy - e0) / n * 1e3:+.2f} meV/atom), "
          f"max|F| {f0:.3f} -> {f1:.3f} eV/A", flush=True)

    if diverged or not (math.isfinite(th.energy) and math.isfinite(f1)):
        # Never overridable: there is nothing physical to continue from.
        raise Unstable(
            "relaxation diverged (" + (diverged or "non-finite energy or forces")
            + "): the model lacks short-range repulsion somewhere near this "
            "structure and the minimiser fell in. Pre-relax the box with a "
            "classical force field, then start from that structure")
    problems = []
    if f1 >= a.relax_fmax:
        problems.append(f"not converged (max|F| {f1:.3f} >= {a.relax_fmax} "
                        "eV/A; raise --relax-steps)")
    if a.efield is None and th.energy > e0:
        problems.append(f"energy rose ({e0:.3f} -> {th.energy:.3f} eV)")
    bonds1 = covalent_bonds(symbols, th.positions.numpy(), cell)
    if bonds1 != bonds0:
        broken, formed = sorted(bonds0 - bonds1), sorted(bonds1 - bonds0)
        problems.append(
            f"covalent topology changed: {len(broken)} bond(s) broken, "
            f"{len(formed)} formed, e.g. atoms {(broken or formed)[:3]}")
    if problems and not a.no_relax_checks:
        raise Unstable("relaxation rejected: " + "; ".join(problems) + ". "
                       "A heavily clashing start is better pre-relaxed with a "
                       "classical force field; --no-relax-checks overrides.")
    for p in problems:
        print(f"  warning: {p}", flush=True)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


class Logger:
    """The thermo table: one '#' header line, then whitespace-separated rows,
    mirrored to stdout. numpy.loadtxt reads the file directly."""

    def __init__(self, path: pathlib.Path, columns, append: bool):
        fresh = not (append and path.exists() and path.stat().st_size > 0)
        self.fh = open(path, "a" if append else "w")
        self.columns = columns                  # [(name, width, format spec)]
        header = "".join(f"{name:>{width}}" for name, width, _ in columns)
        header = "#" + header[1:]
        if fresh:
            self.fh.write(header + "\n")
            self.fh.flush()
        print(header, flush=True)

    def write(self, row: dict) -> None:
        line = "".join(f"{row[name]:>{width}{spec}}"
                       for name, width, spec in self.columns)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def tell(self) -> int:
        return self.fh.tell()

    def close(self) -> None:
        self.fh.close()


def log_columns(a: argparse.Namespace):
    cols = [("step", 10, "d"), ("time[ps]", 12, ".4f"), ("T[K]", 10, ".2f"),
            ("Epot[eV]", 18, ".6f"), ("Ekin[eV]", 16, ".6f"),
            ("Etot[eV]", 18, ".6f")]
    if a.ensemble == "nvt" and a.thermostat == "nhc":
        cols.append(("Econs[eV]", 18, ".6f"))
    if a.ensemble == "npt":
        cols += [("P[GPa]", 12, ".5f"), ("V[A^3]", 14, ".3f")]
    cols.append(("ms/step", 10, ".1f"))
    return cols


class TrajectoryWriter:
    """Appends frames as extxyz (one open handle) or ASE .traj."""

    def __init__(self, path: pathlib.Path, a: argparse.Namespace, symbols,
                 append: bool):
        self.a = a
        self.symbols = symbols
        self.path = path
        if a.traj_format == "traj":
            self.traj, self.fh = Trajectory(str(path), "a" if append else "w"), None
        else:
            self.traj, self.fh = None, open(path, "a" if append else "w")

    def write(self, th: Thermo, step: int) -> None:
        a, ev = self.a, th.evaluator
        atoms = Atoms(self.symbols, positions=th.positions.numpy(),
                      cell=np.asarray(ev.cell), pbc=True)
        if a.traj_wrapped:
            atoms.wrap()
        if a.traj_velocities:
            atoms.set_velocities(th.velocities.numpy())
        results = {"energy": th.energy}
        if a.traj_forces:
            results["forces"] = th.forces.numpy().copy()
        if a.traj_charges:
            results["charges"] = np.asarray(ev.results["charge"]).reshape(-1).copy()
        atoms.calc = SinglePointCalculator(atoms, **results)
        atoms.info["step"] = int(step)
        atoms.info["time_ps"] = step * a.dt / 1000.0
        if self.fh is not None:
            ase.io.write(self.fh, atoms, format="extxyz")
            self.fh.flush()
        else:
            self.traj.write(atoms)

    def tell(self) -> int | None:
        """Bytes written so far (extxyz only), recorded in each checkpoint."""
        return self.fh.tell() if self.fh is not None else None

    def close(self) -> None:
        (self.fh or self.traj).close()


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


def save_checkpoint(path: pathlib.Path, th: Thermo, integ: Integrator,
                    step: int, seed: int, a, fp, logger, traj) -> None:
    """Atomic: written to a temporary file, then renamed over the old one."""
    meta = {
        "format": "deepmd-lr examples/07-multi-gpu-md/run_md.py checkpoint",
        "version": CHECKPOINT_VERSION,
        "step": int(step),
        "seed": int(seed),
        "physics": physics_of(a),
        "fingerprint": fp,
        "log_bytes": logger.tell(),
        "traj_bytes": traj.tell() if traj is not None else None,
        "traj_file": traj.path.name if traj is not None else None,
    }
    arrays = {
        "positions": th.positions.numpy(),
        "velocities": th.velocities.numpy(),
        "cell": np.asarray(th.evaluator.cell, dtype=np.float64),
        **integ.state(),
    }
    tmp = path.with_name("." + path.stem + ".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def load_checkpoint(path: pathlib.Path, a, fp):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    if meta.get("version") != CHECKPOINT_VERSION:
        raise InputError(f"{path} has checkpoint version {meta.get('version')}, "
                         f"this script writes {CHECKPOINT_VERSION}")
    if meta["fingerprint"] != fp:
        raise InputError(
            f"{path} belongs to a different system or model: checkpoint "
            f"{meta['fingerprint']} vs now {fp}. Use the same --structure, "
            "--repeat and --model as the run being continued")
    now = physics_of(a)
    diff = [f"{k}: {meta['physics'].get(k)!r} -> {v!r}" for k, v in now.items()
            if meta["physics"].get(k) != v]
    diff += [f"{k}: {v!r} -> (unset)" for k, v in meta["physics"].items()
             if k not in now]
    if diff:
        raise InputError(
            "a restart must keep the physics of the run it continues; changed: "
            + "; ".join(diff) + ". To change them, start a new run from "
            "<out>/final.extxyz (or the last trajectory frame) with "
            "--keep-velocities and a new --out")
    return meta, arrays


def trim(path: pathlib.Path, nbytes) -> None:
    """Cut a file back to its size at the checkpoint (after a hard kill)."""
    if nbytes is not None and path.exists() and path.stat().st_size > nbytes:
        print(f"  trimming {path.name} back to the checkpoint "
              f"({path.stat().st_size - nbytes} bytes written after it)",
              flush=True)
        os.truncate(path, nbytes)


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
    """SIGTERM/SIGUSR1/SIGINT ask for a clean stop at the end of the current
    step, so the checkpoint is never taken halfway through a step. A second
    SIGINT quits immediately."""

    def __init__(self):
        self.reason = None

    def __call__(self, signum, frame):
        if self.reason is not None and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.reason = signal.Signals(signum).name
        print(f"\n  {self.reason} received: stopping after this step "
              "(send SIGINT again to quit at once)", flush=True)

    SIGNALS = (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)

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
    traj_path = out / f"traj.{a.traj_format}"
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
    fp = fingerprint(model, symbols)

    meta = arrays = None
    if a.restart:
        meta, arrays = load_checkpoint(ckpt_path, a, fp)
        start, seed = int(meta["step"]), int(meta["seed"])
        if start >= a.steps:
            print(f"{out}: already at step {start} >= --steps {a.steps}; "
                  "nothing to do (raise --steps to extend the run)")
            return 0
        if a.seed is not None and a.seed != seed:
            print(f"note: --seed ignored on restart; the run keeps seed {seed}")
        atoms.set_cell(arrays["cell"], scale_atoms=False)
        atoms.set_positions(arrays["positions"])
        trim(log_path, meta["log_bytes"])
        if meta.get("traj_file") == traj_path.name:
            trim(traj_path, meta["traj_bytes"])
    else:
        start = 0
        seed = a.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little") % 2**31

    velocities = None
    if a.restart:
        velocities = arrays["velocities"]
    elif a.keep_velocities:
        velocities = atoms.get_velocities()
        if velocities is None or not np.any(velocities):
            print("note: the structure file has no velocities; drawing them at "
                  f"{a.temperature} K instead", flush=True)
            velocities = None

    integ_name = {"nve": "NVE", "nvt": f"NVT ({a.thermostat})",
                  "npt": "NPT (MTK)"}[a.ensemble]
    counts = sorted(collections.Counter(symbols).items())
    comp = " ".join(f"{el} {c}" for el, c in counts)
    cellpar = atoms.cell.cellpar()
    print(f"deepmd-lr MD | {integ_name} | T {a.temperature:g} K | dt {a.dt:g} fs | "
          f"steps {start} -> {a.steps}", flush=True)
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
    if velocities is not None and not a.restart:
        # Every temperature and thermostat here counts 3N - 3 degrees of
        # freedom, which assumes the centre of mass is at rest. Kept
        # velocities (from a Langevin run, say) need not be.
        m = ev.masses()
        velocities = velocities - (m[:, None] * velocities).sum(0) / m.sum()
    th = Thermo(ev, atoms.get_positions(),
                velocities=np.zeros((n, 3)) if velocities is None else velocities,
                timestep=a.dt, temperature=a.temperature)
    integ = build_integrator(a, th, seed)
    if a.restart:
        integ.load_state(arrays)
    print(f"  ensemble {integ.name}; seed {seed}", flush=True)

    t0 = time.perf_counter()
    th.evaluate()
    print(f"  first force call {time.perf_counter() - t0:.1f} s (kernel "
          f"compilation, memory probe); skin {ev.skin:.2f} A", flush=True)
    report_charges(ev, symbols, a)
    if not a.restart:
        if a.relax_steps > 0:
            relax(th, a, symbols)
        if velocities is None:
            th.maxwell_boltzmann(a.temperature, seed=seed)
    integ.prepare()

    max_t = a.max_temperature
    if max_t is None:
        max_t = max(5.0 * a.temperature, 1500.0)

    run_json = out / "run.json"
    record = (json.loads(run_json.read_text())
              if a.restart and run_json.exists() else {"segments": []})
    segment = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "arguments": vars(a),
        "devices": devices,
        "seed": seed,
        "start_step": start,
        "n_atoms": n,
        "composition": dict(collections.Counter(symbols)),
        "environment": environment(devices),
        "skin_A": ev.skin,
    }
    record["segments"].append(segment)
    run_json.write_text(json.dumps(record, indent=2, default=str))

    logger = Logger(log_path, log_columns(a), append=a.restart)
    traj = (TrajectoryWriter(traj_path, a, symbols, append=a.restart)
            if a.traj_every > 0 else None)
    clock = [(start, time.perf_counter())]
    rows = []

    def log(step: int) -> None:
        now = time.perf_counter()
        s_prev, t_prev = clock[-1]
        row = {"step": step, "time[ps]": step * a.dt / 1000.0,
               "T[K]": th.temperature, "Epot[eV]": th.energy,
               "Ekin[eV]": th.kinetic_energy}
        row["Etot[eV]"] = row["Epot[eV]"] + row["Ekin[eV]"]
        econs = integ.conserved()
        if econs is not None:
            row["Econs[eV]"] = econs
        if a.ensemble == "npt":
            row["P[GPa]"] = th.pressure / units.GPa
            row["V[A^3]"] = th.volume
        row["ms/step"] = ((now - t_prev) / (step - s_prev) * 1e3
                          if step > s_prev else float("nan"))
        clock.append((step, now))
        logger.write(row)
        rows.append(row)
        if max_t > 0 and not row["T[K]"] <= max_t:          # NaN-safe
            raise Unstable(f"temperature {row['T[K]']:.0f} K exceeds "
                           f"--max-temperature {max_t:g} K at step {step}")

    stop = StopFlag()
    stop.install()
    deadline = time.monotonic() + a.max_hours * 3600 if a.max_hours > 0 else None
    status = "finished"
    step = start
    try:
        if not a.restart:
            log(0)
            if traj is not None:
                traj.write(th, 0)
        while step < a.steps:
            integ.step(step)
            step += 1
            if not math.isfinite(th.energy):
                raise Unstable(f"non-finite energy at step {step}")
            if step % a.log_every == 0 or step == a.steps:
                log(step)
            if traj is not None and step % a.traj_every == 0:
                traj.write(th, step)
            saved = step % a.checkpoint_every == 0 or step == a.steps
            if saved:
                save_checkpoint(ckpt_path, th, integ, step, seed, a, fp, logger, traj)
            reason = stop.reason or (
                "wall-time budget" if deadline and time.monotonic() > deadline
                else None)
            if reason and step < a.steps:
                if not saved:
                    save_checkpoint(ckpt_path, th, integ, step, seed, a, fp,
                                    logger, traj)
                status = f"stopped at step {step} ({reason}); continue with --restart"
                break
    except Unstable as exc:
        bad = Atoms(symbols, positions=th.positions.numpy(),
                    cell=np.asarray(ev.cell), pbc=True)
        ase.io.write(out / "blowup.extxyz", bad)
        print(f"\n*** UNSTABLE: {exc}\n"
              f"    last configuration: {out / 'blowup.extxyz'}\n"
              "    Usual causes: timestep too large, an unrelaxed start "
              "(--relax-steps), a thermostat too weak for a field, or a "
              "structure outside the model's training data.", flush=True)
        segment.update(status=f"unstable: {exc}", end_step=step)
        run_json.write_text(json.dumps(record, indent=2, default=str))
        return 2
    finally:
        stop.restore()
        logger.close()
        if traj is not None:
            traj.close()

    final = Atoms(symbols, positions=th.positions.numpy(),
                  cell=np.asarray(ev.cell), pbc=True)
    final.set_velocities(th.velocities.numpy())
    ase.io.write(out / "final.extxyz", final)

    summary = summarize(th, ev, a, rows, clock, n, devices)
    print(f"\n{status}\n" + "\n".join(f"  {k}: {v}" for k, v in summary.items()),
          flush=True)
    segment.update(status=status, end_step=step, summary=summary)
    run_json.write_text(json.dumps(record, indent=2, default=str))
    return 0


def summarize(th, ev, a, rows, clock, n, devices) -> dict:
    s = {}
    if len(clock) >= 3:               # drop the first interval (compilation)
        (s1, t1), (s2, t2) = clock[1], clock[-1]
        if s2 > s1:
            ms = (t2 - t1) / (s2 - s1) * 1e3
            s["speed"] = (f"{ms:.1f} ms/step = {a.dt * 86.4 / ms:.3f} ns/day "
                          f"({n} atoms on {len(devices)} device(s))")
    s["halo"] = (f"skin {ev.skin:.2f} A, {ev.rebuild_count} rebuilds, "
                 f"{ev.skin_backoffs} out-of-memory backoffs")
    if len(rows) >= 2:
        half = rows[len(rows) // 2:]
        s["mean T (second half)"] = f"{np.mean([r['T[K]'] for r in half]):.1f} K"
        if a.ensemble == "npt":
            p_mean = np.mean([r["P[GPa]"] for r in half])
            v_mean = np.mean([r["V[A^3]"] for r in half])
            s["mean P (second half)"] = f"{p_mean:+.5f} GPa"
            s["mean V (second half)"] = f"{v_mean:.2f} A^3"
        key = {"nve": "Etot[eV]"}.get(a.ensemble,
                                      "Econs[eV]" if "Econs[eV]" in rows[0] else None)
        if key is not None:
            t = np.array([r["time[ps]"] for r in rows])
            e = np.array([r[key] for r in rows]) * 1e3 / n
            if np.ptp(t) > 0:
                # A fit over a short stretch mostly measures the fluctuation:
                # 4 rows over 0.15 ps read +0.43 where 0.5 ps gave +0.0095.
                short = np.ptp(t) < 0.5 or len(rows) < 20
                s[f"{key.split('[')[0]} drift"] = (
                    f"{np.polyfit(t, e, 1)[0]:+.4f} meV/atom/ps over "
                    f"{np.ptp(t):.3g} ps, {len(rows)} log rows"
                    + ("; too short to judge: use >= 0.5 ps and >= 20 rows"
                       if short else ""))
            if a.ensemble == "nve":
                s["Etot fluctuation"] = (f"{np.std(e):.4f} meV/atom rms (falls "
                                         "~4x when dt is halved)")
    return s


def _time_devices(a: argparse.Namespace, devices: list[str]):
    """ms/step for one device list, or NaN. Everything it builds is local, so
    it is all freed on return, before the next device count is built."""
    atoms, model = load_system(a, devices[0])
    n = len(atoms)
    ev = make_evaluator(model, atoms, devices, a)
    th = Thermo(ev, atoms.get_positions(), velocities=np.zeros((n, 3)),
                timestep=a.dt, temperature=a.temperature)
    th.maxwell_boltzmann(a.temperature, seed=1)
    integ = build_integrator(a, th, seed=1)
    th.evaluate()
    integ.prepare()
    warm = max(10, a.benchmark_steps // 5)       # compilation, memory probe
    for i in range(warm):
        integ.step(i)
    for d in set(devices):
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for i in range(warm, warm + a.benchmark_steps):
        integ.step(i)
    for d in set(devices):
        torch.cuda.synchronize(d)
    ms = (time.perf_counter() - t0) / a.benchmark_steps * 1e3
    return (ms if math.isfinite(th.energy) else float("nan")), ev.skin


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
    except Unstable as exc:            # a rejected relaxation
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
