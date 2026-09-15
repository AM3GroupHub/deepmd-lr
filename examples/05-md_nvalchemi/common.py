"""Shared pieces for the nvalchemiops MD examples.

nvalchemiops ships half-step *kernels* and no driver, so the integration loop
belongs to the caller. That is the point of these examples: each script is a
complete loop you can copy, not a wrapper around one.

Everything here works in the model's units -- angstrom, eV, amu, and ASE time
units (`A sqrt(amu/eV)`) -- which are mutually consistent, so no conversion
factors appear in the loops.
"""

from __future__ import annotations

import ase.io
import numpy as np
import torch
import warp as wp
from ase.data import atomic_masses, atomic_numbers
from ase.units import fs, kB
from nvalchemiops.dynamics.utils import (
    compute_kinetic_energy,
    compute_temperature,
    initialize_velocities,
)

from deepmd_lr.ase import load_model
from deepmd_lr.distributed import DistributedEvaluator

__all__ = [
    "Reporter", "Thermo", "build_evaluator", "kt", "maxwell_boltzmann",
    "read_structure", "scalar", "vec",
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


class Thermo:
    """Kinetic energy and temperature, through nvalchemiops' own kernels.

    `compute_temperature` returns `k_B T` in energy units, like everything
    else in that library, and it already divides by `3N - 3` rather than `3N`
    -- i.e. it assumes the centre-of-mass motion was removed, which
    `initialize_velocities` does. Verified against both conventions rather
    than assumed.
    """

    def __init__(self, natoms: int, device: str):
        self._ke = wp.zeros(1, dtype=wp.float64, device=device)
        self._T = wp.zeros(1, dtype=wp.float64, device=device)
        self._n = wp.array(np.array([natoms]), dtype=wp.int32, device=device)

    def kinetic_energy(self, wv: wp.array, wm: wp.array) -> float:
        """Total kinetic energy in eV."""
        compute_kinetic_energy(wv, wm, self._ke)
        return float(self._ke.numpy()[0])

    def kelvin(self, wv: wp.array, wm: wp.array) -> float:
        """Instantaneous temperature in kelvin."""
        compute_kinetic_energy(wv, wm, self._ke)
        compute_temperature(self._ke, self._T, self._n)
        return float(self._T.numpy()[0]) / kB


# --- system -----------------------------------------------------------------

def read_structure(path, index: int = -1):
    """Read a starting structure with ASE: positions, symbols, cell.

    Anything `ase.io.read` understands works -- extxyz, POSCAR, CIF, an ASE
    trajectory. For a multi-frame file `index` picks the frame; the default
    takes the last, which is usually the most equilibrated one.

    A cell is required. These are periodic simulations, and a structure with
    no cell would silently be treated as a molecule in vacuum by the
    descriptor and rejected outright by the reciprocal-space term.
    """
    atoms = ase.io.read(str(path), index=index)
    if isinstance(atoms, list):                      # a slice was passed
        atoms = atoms[-1]
    cell = np.asarray(atoms.get_cell())
    if abs(np.linalg.det(cell)) < 1e-8:
        raise ValueError(
            f"{path} has no cell (volume {abs(np.linalg.det(cell)):.2e} A^3). "
            "These examples run periodic MD, so the structure needs one -- "
            "set it in the file, or add a vacuum-padded cell for a slab."
        )
    if not atoms.pbc.all():
        atoms.pbc = True                             # a cell implies periodicity
    return atoms.get_positions(), atoms.get_chemical_symbols(), cell


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
    """Maxwell-Boltzmann velocities, centre-of-mass motion removed.

    Straight through `nvalchemiops.dynamics.utils.initialize_velocities`, so
    the draw happens on the GPU and the result never touches the host. The
    three scratch arrays are its COM-removal workspace, zeroed internally.

    Returns the torch tensor backing the warp view, because the integrator
    kernels want a warp array while the reporting wants something indexable.
    """
    v = torch.zeros((len(masses), 3), dtype=torch.float64, device=device)
    wm = wp.array(np.asarray(masses, dtype=np.float64), dtype=wp.float64,
                  device=device)
    momentum = wp.zeros(1, dtype=wp.vec3d, device=device)
    total_mass = wp.zeros(1, dtype=wp.float64, device=device)
    com = wp.zeros(1, dtype=wp.vec3d, device=device)
    initialize_velocities(
        vec(v), wm, scalar(kt(temperature_K), device),
        momentum, total_mass, com, random_seed=seed, remove_com=True,
    )
    return v


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
