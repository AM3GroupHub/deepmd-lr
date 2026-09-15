"""Shared pieces for the nvalchemiops MD examples.

nvalchemiops ships half-step *kernels* and no driver, so the integration loop
belongs to the caller. That is the point of these examples: each script is a
complete loop you can copy, not a wrapper around one.

Everything here works in the model's units -- angstrom, eV, amu, and ASE time
units (`A sqrt(amu/eV)`) -- which are mutually consistent, so no conversion
factors appear in the loops.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from ase.data import atomic_masses, atomic_numbers
from ase.units import fs, kB

from deepmd_lr.ase import load_model
from deepmd_lr.distributed import DistributedEvaluator

__all__ = [
    "Reporter", "build_evaluator", "kelvin", "kt", "maxwell_boltzmann",
    "scalar", "vec", "water_box",
]


# --- warp interop -----------------------------------------------------------
# `wp.from_torch` is a zero-copy view, so the trajectory never leaves the GPU
# for the integrator's sake. It needs contiguous float64.

def vec(t: torch.Tensor) -> wp.array:
    """A warp ``vec3d`` view of an (n, 3) float64 torch tensor."""
    return wp.from_torch(t.contiguous(), dtype=wp.vec3d)


def scalar(value: float, device: str, dtype=wp.float64, n: int = 1) -> wp.array:
    """A length-n warp array. Every kernel takes its scalars this way."""
    return wp.array(np.full(n, value), dtype=dtype, device=device)


# --- units ------------------------------------------------------------------
# THE conversion to get right. Every nvalchemiops dynamics entry point takes
# temperature as k_B T in ENERGY units, not in kelvin -- the opposite of the
# heavy `nvalchemi-toolkit`, whose thermostats took kelvin. Passing kelvin
# where kT is expected silently thermostats to ~11600x the temperature you
# meant; passing kT where kelvin is expected gives a system at rest.

def kt(temperature_K: float) -> float:
    """k_B T in eV, from a temperature in kelvin."""
    return kB * temperature_K


def kelvin(velocities: torch.Tensor, masses: torch.Tensor, ndof: int) -> float:
    """Instantaneous temperature in kelvin, from the kinetic energy."""
    ke2 = float((masses[:, None] * velocities**2).sum())
    return ke2 / (ndof * kB)


# --- system -----------------------------------------------------------------

def water_box(n_side: int = 8, spacing: float = 3.1, seed: int = 0):
    """A jittered lattice of water molecules: positions, symbols, cell."""
    rng = np.random.default_rng(seed)
    L = n_side * spacing
    pos, sym = [], []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                o = (np.array([i, j, k]) + 0.5) * spacing + rng.uniform(-0.3, 0.3, 3)
                pos += [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.93, 0.0]]
                sym += ["O", "H", "H"]
    return np.array(pos) % L, sym, np.eye(3) * L


def build_evaluator(model_path, symbols, cell, devices=None, **kwargs):
    """Load a `dp_lr` checkpoint and wrap it in a DistributedEvaluator.

    One device is the normal case; pass several to partition across GPUs.
    `kwargs` forwards `compute_virial` (NPT needs it), `efield`, `head`, ...
    """
    head = kwargs.pop("head", None)
    model = load_model(str(model_path), head=head)
    type_map = list(model.get_type_map())
    atype = np.array([type_map.index(s) for s in symbols], dtype=np.int64)
    if devices is None:
        devices = [str(next(model.parameters()).device)]
    ev = DistributedEvaluator(model, atype, np.asarray(cell), devices=devices,
                              **kwargs)
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    return ev, masses


def maxwell_boltzmann(masses: np.ndarray, temperature_K: float, seed: int = 0,
                      device: str = "cuda") -> torch.Tensor:
    """Velocities from a Maxwell-Boltzmann distribution, drift removed."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(kt(temperature_K) / masses)[:, None]
    v = rng.normal(0.0, 1.0, (len(masses), 3)) * sigma
    v -= (masses[:, None] * v).sum(0) / masses.sum()      # zero total momentum
    return torch.tensor(v, dtype=torch.float64, device=device)


# --- reporting --------------------------------------------------------------

class Reporter:
    """Print a fixed-width log and keep the history for later inspection."""

    def __init__(self, *columns: str, every: int = 50):
        self.columns = ("step", "time_fs", *columns)
        self.every = every
        self.rows: list[tuple] = []
        print("  " + "".join(f"{c:>13s}" for c in self.columns), flush=True)

    def __call__(self, step: int, time_fs: float, **values):
        row = (step, time_fs, *(values[c] for c in self.columns[2:]))
        self.rows.append(row)
        if step % self.every == 0 or step == 0:
            cells = [f"{row[0]:13d}", f"{row[1]:13.1f}"]
            cells += [f"{v:13.5f}" for v in row[2:]]
            print("  " + "".join(cells), flush=True)

    def drift(self, column: str, per_atom: int = 1) -> float:
        """meV/atom/ps of drift in a column, by least squares over the run."""
        i = self.columns.index(column)
        t = np.array([r[1] for r in self.rows]) / 1000.0        # ps
        y = np.array([r[i] for r in self.rows]) * 1000.0 / per_atom
        if len(t) < 2:
            return float("nan")
        return float(np.polyfit(t, y, 1)[0])


DT_DEFAULT = 0.5 * fs
