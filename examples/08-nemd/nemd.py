"""Interfacial thermal conductance by NEMD: a hot and a cold Langevin bath on
one slab, and the heat they exchange through the interface between them.

Geometry along z, bottom to top:

    vacuum | fixed_bottom | HOT bath | free: material A | material B | COLD bath
           | fixed_top | vacuum

The outer layers are held by ASE's `FixAtoms`, which the GPU classes honour
(they stay exactly where they start, at rest, and carry no degrees of
freedom). The two baths are Langevin thermostats at T_hot and T_cold on their
own atoms only; everything in between is NVE. In steady state the power the
hot bath injects equals the power the cold bath removes, and both flow
through the single interface, so the heat flux is J = P / A with no factor
1/2. The temperature profile T(z) then has a jump dT at the interface, and
the interfacial (Kapitza) conductance is G = J / dT. `analyze.py` does the
fits.

The integrator is `RegionLangevin` below, a subclass of `MolecularDynamics`
written the way `06/07_custom_berendsen.py` writes a thermostat of its own.
Each step is O(dt/2) on every bath, velocity Verlet(dt), O(dt/2) again, where
the O-step is v <- c1 v + c2 sqrt(kT/m) xi with c1 = exp(-friction dt/2) and
c2 = sqrt(1 - c1^2). The kinetic energy every O-step adds is charged to its
bath, on the device, so `md.heat` is exact to rounding and costs no host
synchronisation per step. Nothing is conserved, and no centre-of-mass motion
is removed during the run: the fixed layers hold the slab, and a running
correction would alter the heat account.

Three stages, on one evaluator:

  1. equilibration: `Langevin` on every mobile atom at T0 (`--equil-ps`);
  2. steady: the baths on for `--steady-ps`; the gradient builds up. The
     heat is then reset, so bath transients stay out of the account;
  3. production: `--run-ps` with the baths on; `thermo.log` carries T and
     the cumulative heat Q of each bath, `profile.csv` the block-averaged
     T(z) over the mobile atoms, `trajectory.extxyz` frames, and
     `checkpoint.npz` the state every `--checkpoint-ps`. `--resume`
     continues from the checkpoint, with the log, trajectory and profile
     trimmed back to it, and replays exactly the noise an uninterrupted run
     would have drawn.

With T_hot = T_cold = T0 the run is an equilibrium control: the long-time
average of each bath's power is zero, but the cumulative Q of one finite
trajectory random-walks and is not expected to stay near zero.

The groups come from a per-atom `group` array in the structure file
(fixed_bottom, hot, free, cold, fixed_top), or, when the file has none, from
z-thicknesses: `--fixed-thickness` at each end, then `--bath-thickness`.

    python nemd.py --model ../../path/to/model.ckpt.pt \
                   --structure interface.extxyz --out run1
    python nemd.py --model ... --structure ... --out run1 --resume
    python analyze.py --out run1
"""

import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
import torch
import warp as wp
from ase import units
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import thermalize_momenta
from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize,
)

from deepmd_lr.nvalchemi import (
    Langevin,
    MolecularDynamics,
    build_evaluator,
    read_structure,
)

GROUPS = ("fixed_bottom", "hot", "free", "cold", "fixed_top")


# ----------------------------------------------------------------------------
# The integrator
# ----------------------------------------------------------------------------


def _check_baths(baths, n_atoms):
    """``{name: (indices or bool mask, temperature_K)}`` -> ``{name: (sorted
    int64 indices, float)}``, refusing what the heat account could not
    attribute: an empty or overlapping bath, an index outside the system."""
    if not baths:
        raise ValueError("at least one bath is needed")
    out, seen = {}, np.zeros(n_atoms, dtype=bool)
    for name, (atoms_of, temperature_K) in dict(baths).items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"bath names must be non-empty strings, got {name!r}")
        sel = np.asarray(atoms_of)
        if sel.dtype == bool:
            if sel.shape != (n_atoms,):
                raise ValueError(f"bath {name!r}: a mask must have {n_atoms} entries")
            idx = np.flatnonzero(sel)
        else:
            idx = np.unique(sel.astype(np.int64))
        if len(idx) == 0:
            raise ValueError(f"bath {name!r} is empty")
        if idx.min() < 0 or idx.max() >= n_atoms:
            raise ValueError(f"bath {name!r} has an index outside [0, {n_atoms})")
        if seen[idx].any():
            other = next(k for k, (i, _) in out.items() if np.intersect1d(i, idx).size)
            raise ValueError(
                f"baths {other!r} and {name!r} share "
                f"{np.intersect1d(out[other][0], idx).size} atoms"
            )
        if not float(temperature_K) >= 0:
            raise ValueError(f"bath {name!r}: temperature_K must be >= 0")
        seen[idx] = True
        out[name] = (idx, float(temperature_K))
    return out


class RegionLangevin(MolecularDynamics):
    """Velocity Verlet with a Langevin bath on each of several atom groups,
    each at its own temperature: the NEMD integrator for thermal transport.

    Parameters
    ----------
    baths
        ``{name: (indices or bool mask, temperature_K)}``. Names become the
        log columns ``T_<name>[K]`` and ``Q_<name>[eV]``, in this order.
    friction
        In inverse ASE time units, as for `Langevin`: 1/friction is the
        bath's relaxation time, ``1 / (100 * units.fs)`` for 100 fs.
    seed
        The noise is drawn from a generator reseeded with ``seed + nsteps``
        every step, as `Langevin` does, so a restart from a checkpoint
        replays exactly the noise an uninterrupted run would have drawn.

    Atoms outside every bath are NVE. Fixed atoms (`FixAtoms`) may not be in
    a bath: `Thermo` zeroes their velocity before every force call, and the
    energy a bath gave them would vanish from the account.

    `heat` is the cumulative kinetic energy the O-steps of each bath added,
    in eV, exact for those O-steps alone: velocities changed from outside
    (an observer editing `atoms`, `set_temperature`) are not charged. There
    is no conserved quantity; ``Etot - sum(heat)`` fluctuates like NVE's
    Etot, which is the check that the account is right.

    The bath update is written with torch on the tensor behind the Thermo's
    warp view of the velocities, as `Thermo.evaluate` and the relaxations
    do. Torch's default stream and warp's device stream are ordered against
    each other; running this under a non-blocking torch stream would not be.
    """

    def __init__(self, evaluator, atoms, timestep, baths, friction, *,
                 seed=None, **kwargs):
        if not float(friction) > 0:
            raise ValueError("friction must be positive")
        self.baths = _check_baths(baths, len(atoms))
        # Read by the base constructor when it builds the log header.
        self._EXTRA_BUILTIN = tuple(
            col for name in self.baths for col in (f"T_{name}[K]", f"Q_{name}[eV]")
        )
        super().__init__(evaluator, atoms, timestep, **kwargs)
        th = self.thermo
        held = np.intersect1d(
            np.concatenate([idx for idx, _ in self.baths.values()]), th.fixed
        )
        if len(held):
            raise ValueError(
                f"{len(held)} bath atoms are fixed (FixAtoms); a fixed atom "
                "cannot be thermostatted"
            )
        self._v = wp.to_torch(th.velocities)        # the Thermo's tensor, aliased
        self._m = wp.to_torch(th.masses)
        self.friction = float(friction)
        self._c1 = math.exp(-0.5 * self.friction * self.timestep)
        self._c2 = math.sqrt(1.0 - self._c1**2)
        self._bath_tensors = []                     # per bath: idx, m (n, 1), sigma
        for idx, t in self.baths.values():
            i = torch.as_tensor(idx, device=self._v.device)
            m = self._m.index_select(0, i)[:, None]
            self._bath_tensors.append([i, m, torch.sqrt(units.kB * t / m)])
        self._heat = torch.zeros(len(self.baths), dtype=torch.float64,
                                 device=self._v.device)
        self._gen = torch.Generator(device=self._v.device)
        self.seed = (
            int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little")
        ) % 2**31

    def step(self):
        th = self.thermo
        self._gen.manual_seed((self.seed + self.nsteps) % 2**31)
        self._o_step()
        velocity_verlet_position_update(th.positions, th.velocities, th.forces,
                                        th.masses, self.dt)
        th.evaluate()
        velocity_verlet_velocity_finalize(th.velocities, th.forces, th.masses,
                                          self.dt)
        self._o_step()

    def _o_step(self):
        """v <- c1 v + c2 sqrt(kT/m) xi on each bath, over half a step; the
        kinetic energy it adds is charged to that bath. Device only."""
        for b, (idx, m, sigma) in enumerate(self._bath_tensors):
            v = self._v.index_select(0, idx)
            xi = torch.randn(v.shape, generator=self._gen, device=v.device,
                             dtype=v.dtype)
            v_new = self._c1 * v + self._c2 * sigma * xi
            self._v.index_copy_(0, idx, v_new)
            self._heat[b].add_(0.5 * (m * (v_new * v_new - v * v)).sum())

    @property
    def heat(self):
        """Cumulative energy each bath injected, in eV, since construction or
        the last `reset_heat`. One host read."""
        return dict(zip(self.baths, self._heat.tolist(), strict=True))

    def reset_heat(self):
        """Zero the account -- at the end of the steady stage."""
        self._heat.zero_()

    def temperature(self, region):
        """Kinetic temperature (K) of a bath, by name, or of any index array:
        3 degrees of freedom per atom, none removed, since a subset never
        conserves its momentum."""
        if isinstance(region, str):
            idx = self._bath_tensors[list(self.baths).index(region)][0]
        else:
            idx = torch.as_tensor(np.asarray(region, dtype=np.int64),
                                  device=self._v.device)
        v = self._v.index_select(0, idx)
        ke = 0.5 * (self._m.index_select(0, idx)[:, None] * v * v).sum()
        return float(ke) * 2.0 / (3 * len(idx) * units.kB)

    def set_temperature(self, name, temperature_K):
        """Change one bath's target in place (the heat account continues)."""
        if not float(temperature_K) >= 0:
            raise ValueError("temperature_K must be >= 0")
        b = list(self.baths).index(name)
        idx, _ = self.baths[name]
        self.baths[name] = (idx, float(temperature_K))
        m = self._bath_tensors[b][1]
        self._bath_tensors[b][2] = torch.sqrt(units.kB * float(temperature_K) / m)

    def _builtin_extras(self):
        heat = self._heat.tolist()
        row = {}
        for b, name in enumerate(self.baths):
            row[f"T_{name}[K]"] = self.temperature(name)
            row[f"Q_{name}[eV]"] = heat[b]
        return row

    def _params(self):                              # what a checkpoint records
        return {
            "baths": {n: [idx.tolist(), t] for n, (idx, t) in self.baths.items()},
            "friction": self.friction,
            "seed": self.seed,
        }

    def _state(self):
        return {"heat": self._heat.cpu().numpy()}

    def _load_state(self, arrays):
        self._heat.copy_(torch.as_tensor(arrays["heat"]))


# ----------------------------------------------------------------------------
# Groups, the interface, and the T(z) profile
# ----------------------------------------------------------------------------


def assign_groups(atoms, a):
    """Which atoms are fixed, hot, free and cold, and where the interface is.

    From the file's per-atom ``group`` array when it has one, else from the
    thicknesses on the command line, measured from the two faces of the slab
    as `read_structure` centred it. Atoms the file already holds (POSCAR's
    selective dynamics, extxyz's ``move_mask``) join the fixed set.
    """
    z = atoms.positions[:, 2]
    n = len(atoms)
    if "group" in atoms.arrays:
        labels = atoms.arrays["group"].astype(str)
        unknown = sorted(set(labels) - set(GROUPS))
        if unknown:
            raise ValueError(f"unknown group labels {unknown}; each atom must be "
                             f"one of {GROUPS}")
        masks = {g: labels == g for g in GROUPS}
        source = "the file's group array"
    else:
        lo, hi = z.min(), z.max()
        tf, tb = a.fixed_thickness, a.bath_thickness
        if tf < 0 or not tb > 0:
            raise ValueError("--fixed-thickness must be >= 0 and --bath-thickness > 0")
        masks = {
            "fixed_bottom": z < lo + tf,
            "hot": (z >= lo + tf) & (z < lo + tf + tb),
            "cold": (z > hi - tf - tb) & (z <= hi - tf),
            "fixed_top": z > hi - tf,
        }
        masks["free"] = ~(masks["fixed_bottom"] | masks["hot"]
                          | masks["cold"] | masks["fixed_top"])
        source = f"thicknesses: {tf:g} A fixed, {tb:g} A baths"
    for g in ("hot", "free", "cold"):
        if not masks[g].any():
            raise ValueError(f"group '{g}' is empty ({source})")
    present = [g for g in GROUPS if masks[g].any()]
    for lower, upper in zip(present, present[1:], strict=False):
        # By median, so molecules straddling a boundary do not fail a file
        # that is ordered; a mislabelled file still does.
        if np.median(z[masks[lower]]) >= np.median(z[masks[upper]]):
            raise ValueError(f"groups are not ordered along z: '{lower}' lies at "
                             f"{np.median(z[masks[lower]]):.2f} A, '{upper}' at "
                             f"{np.median(z[masks[upper]]):.2f} A")

    from_file = np.concatenate(
        [np.asarray(c.get_indices(), dtype=np.int64) for c in atoms.constraints]
        + [np.zeros(0, dtype=np.int64)]
    )
    fixed = np.union1d(np.flatnonzero(masks["fixed_bottom"] | masks["fixed_top"]),
                       from_file)
    for g in ("hot", "cold"):
        if np.intersect1d(fixed, np.flatnonzero(masks[g])).size:
            raise ValueError(f"the file fixes atoms inside the '{g}' bath")
    mobile = np.setdiff1d(np.arange(n), fixed)
    free = np.setdiff1d(np.flatnonzero(masks["free"]), fixed)
    interface_z = (a.interface_z if a.interface_z is not None
                   else _interface(atoms, free))
    if not z[free].min() < interface_z < z[free].max():
        raise ValueError(f"interface z {interface_z:.2f} A is outside the free "
                         f"region [{z[free].min():.2f}, {z[free].max():.2f}]")
    cell = np.asarray(atoms.get_cell())
    return dict(
        fixed=fixed, mobile=mobile, free=free,
        hot=np.flatnonzero(masks["hot"]), cold=np.flatnonzero(masks["cold"]),
        left=free[z[free] < interface_z], right=free[z[free] > interface_z],
        lower_side=np.flatnonzero(z < interface_z),
        interface_z=float(interface_z),
        area=float(np.linalg.norm(np.cross(cell[0], cell[1]))),
        source=source,
        z_range={g: [float(z[masks[g]].min()), float(z[masks[g]].max())]
                 for g in present},
    )


def _interface(atoms, free):
    """The plane between the two materials: midway between the highest free
    atom of the lower species and the lowest free atom of the upper one.
    Needs exactly two species in the free region; otherwise --interface-z."""
    sym = np.array(atoms.get_chemical_symbols())[free]
    z = atoms.positions[free, 2]
    species = sorted(set(sym), key=lambda s: z[sym == s].mean())
    if len(species) != 2:
        raise ValueError(
            f"the free region has {len(species)} species ({species}); pass "
            "--interface-z (in the coordinates after read_structure centred the slab)"
        )
    lower, upper = species
    return 0.5 * (z[sym == lower].max() + z[sym == upper].min())


class ZProfile:
    """Time-block averages of the kinetic temperature of the mobile atoms in
    bins along z. `sample` accumulates on the device; `flush`, at the end of
    a block, writes one row per bin and prints one line of progress."""

    COLUMNS = ("step", "time_ps", "z", "T", "n_samples")

    def __init__(self, md, mobile, edges, path, append=False):
        th = md.thermo
        self.md = md
        self.x, self.v = wp.to_torch(th.positions), wp.to_torch(th.velocities)
        self.m = wp.to_torch(th.masses)
        dev = self.x.device
        self.mobile = torch.as_tensor(np.asarray(mobile, dtype=np.int64), device=dev)
        self.edges = torch.as_tensor(np.asarray(edges, dtype=np.float64), device=dev)
        self.centres = 0.5 * (np.asarray(edges)[1:] + np.asarray(edges)[:-1])
        self.nb = len(self.centres)
        self.ke = torch.zeros(self.nb, dtype=torch.float64, device=dev)
        self.n = torch.zeros(self.nb, dtype=torch.float64, device=dev)
        self._ones = torch.ones(len(mobile), dtype=torch.float64, device=dev)
        fresh = not (append and os.path.exists(path) and os.path.getsize(path))
        self.fh = open(path, "a" if append else "w")
        if fresh:
            self.fh.write(",".join(self.COLUMNS) + "\n")
            self.fh.flush()

    def sample(self):
        if self.md.nsteps == 0:                     # the observers' start-up call
            return
        z = self.x.index_select(0, self.mobile)[:, 2].contiguous()
        b = torch.bucketize(z, self.edges) - 1
        ok = (b >= 0) & (b < self.nb)
        v = self.v.index_select(0, self.mobile)
        ke = 0.5 * self.m.index_select(0, self.mobile) * (v * v).sum(1)
        self.ke.index_add_(0, b[ok], ke[ok])
        self.n.index_add_(0, b[ok], self._ones[ok])

    def flush(self):
        md = self.md
        if md.nsteps == 0:
            return
        n, ke = self.n.cpu().numpy(), self.ke.cpu().numpy()
        t = 2 * ke / (3 * np.maximum(n, 1) * units.kB)
        t[n == 0] = np.nan
        time_ps = md.get_time() / (1000 * units.fs)
        for zc, ti, ni in zip(self.centres, t, n, strict=True):
            self.fh.write(f"{md.nsteps},{time_ps:.6f},{zc:.4f},{ti:.4f},{int(ni)}\n")
        self.fh.flush()
        self.ke.zero_()
        self.n.zero_()
        heat = md.heat
        print(f"  {time_ps:8.2f} ps  " + "  ".join(
            f"{name} {md.temperature(name):6.1f} K  Q {heat[name]:+9.3f} eV"
            for name in md.baths), flush=True)

    def close(self):
        self.fh.close()


def truncate_profile(path, max_step):
    """Drop the rows a killed run wrote after its last checkpoint."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        lines = fh.readlines()
    keep = lines[:1] + [ln for ln in lines[1:] if int(ln.split(",")[0]) <= max_step]
    with open(path, "w") as fh:
        fh.writelines(keep)


def extra_columns(g, has_charge):
    """Diagnostics beside the per-bath columns: the free region's temperature
    on each side of the interface, the mobile centre-of-mass speed (nothing
    removes it during the run), the largest force, and the charges when the
    model predicts them. All from `md.atoms`, which is synced before a row."""
    left, right, mobile = g["left"], g["right"], g["mobile"]

    def region_t(md, idx):
        p, m = md.atoms.get_momenta()[idx], md.atoms.get_masses()[idx]
        return float((p * p / m[:, None]).sum()) / (3 * len(idx) * units.kB)

    def v_com(md):
        p, m = md.atoms.get_momenta()[mobile], md.atoms.get_masses()[mobile]
        return float(np.linalg.norm(p.sum(0)) / m.sum()) * units.fs

    cols = {
        "T_left[K]": lambda md: region_t(md, left),
        "T_right[K]": lambda md: region_t(md, right),
        "v_com[A/fs]": v_com,
        "maxF[eV/A]": lambda md: float(
            np.abs(md.thermo.get_forces(apply_constraint=False)).max()),
    }
    if has_charge:
        # The early warning for a charge head that does not keep the bulk
        # neutral: Q_lower and M_z then grow with the material thickness and
        # E_lr/N with its square, and the run fails some picoseconds later.
        lower = g["lower_side"]

        def charge(md):
            return np.asarray(md.results["charge"]).reshape(-1)

        def m_z(md):
            dz = md.atoms.positions[:, 2] - g["interface_z"]
            return float((charge(md) * dz).sum())

        n = len(g["mobile"]) + len(g["fixed"])
        cols["q_sum"] = lambda md: float(charge(md).sum())
        cols["q_absmax"] = lambda md: float(np.abs(charge(md)).max())
        cols["Q_lower[e]"] = lambda md: float(charge(md)[lower].sum())
        cols["M_z[eA]"] = m_z
        cols["E_lr/N[meV]"] = lambda md: 1e3 * float(
            md.results["energy_long_range"]) / n
    return cols


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--structure", required=True,
                    help="a slab, pbc T T F, with a `group` array or plain")
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--head", default=None, help="branch of a multi-head checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=None,
                    help="GPUs to partition over (default: every visible one)")
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--tau", type=float, default=100.0,
                    help="fs; the baths' relaxation time, friction = 1/tau")
    ap.add_argument("--t0", type=float, default=300.0, help="K, equilibration")
    ap.add_argument("--t-hot", type=float, default=330.0, help="K")
    ap.add_argument("--t-cold", type=float, default=270.0, help="K")
    ap.add_argument("--equil-ps", type=float, default=20.0)
    ap.add_argument("--steady-ps", type=float, default=50.0,
                    help="baths on, heat not counted")
    ap.add_argument("--run-ps", type=float, default=200.0, help="production")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fixed-thickness", type=float, default=5.0,
                    help="A held at each face (without a group array)")
    ap.add_argument("--bath-thickness", type=float, default=10.0,
                    help="A of bath inside each fixed layer (without a group array)")
    ap.add_argument("--interface-z", type=float, default=None,
                    help="A; default: between the two species of the free region")
    ap.add_argument("--bin", type=float, default=2.0, help="A, the T(z) bins")
    ap.add_argument("--profile-every", type=int, default=10, help="steps")
    ap.add_argument("--block-ps", type=float, default=5.0,
                    help="one T(z) row per bin per block")
    ap.add_argument("--log-every", type=int, default=100, help="steps")
    ap.add_argument("--traj-every", type=int, default=5000, help="steps")
    ap.add_argument("--checkpoint-ps", type=float, default=5.0,
                    help="a multiple of --block-ps")
    ap.add_argument("--compress", type=float, default=0.0,
                    help="table stride in A for the compressed DPA4C backbone "
                         "(0.01 is DeePMD-kit's default; 0 runs uncompressed). "
                         "Needs deepmd-kit built with its CUDA operators")
    ap.add_argument("--charge-filter", type=float, nargs=2, default=None,
                    metavar=("C_T", "DELTA"),
                    help="e; switch charges below about C_T off in simulation "
                         "(deepmd_lr.fitting.charge.filter_charges)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/checkpoint.npz")
    return ap.parse_args()


def main():
    a = parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_gpus = torch.cuda.device_count() if a.gpus is None else a.gpus
    if not 1 <= n_gpus <= torch.cuda.device_count():
        raise SystemExit(f"--gpus {n_gpus}, but {torch.cuda.device_count()} visible")
    devices = [f"cuda:{i}" for i in range(n_gpus)]

    atoms = read_structure(a.structure, a.index)
    if list(atoms.pbc) != [True, True, False]:
        raise SystemExit(f"the slab must be open along z (pbc T T F), got "
                         f"{atoms.pbc.tolist()}")
    g = assign_groups(atoms, a)
    atoms.set_constraint(FixAtoms(indices=g["fixed"]) if len(g["fixed"]) else [])
    ev = build_evaluator(a.model, atoms, devices=devices, head=a.head,
                         compress=a.compress if a.compress > 0 else False,
                         charge_filter=a.charge_filter)
    has_charge = bool(getattr(ev.model, "has_long_range", False))

    dt, friction = a.dt * units.fs, 1.0 / (a.tau * units.fs)

    def steps(ps):
        return int(round(ps * 1000 / a.dt))

    n_equil, n_steady, n_run = steps(a.equil_ps), steps(a.steady_ps), steps(a.run_ps)
    block_steps, ckpt_steps = steps(a.block_ps), steps(a.checkpoint_ps)
    if block_steps < 1 or block_steps % a.profile_every:
        raise SystemExit("--block-ps must be a positive multiple of --profile-every")
    if ckpt_steps < 1 or ckpt_steps % block_steps:
        raise SystemExit("--checkpoint-ps must be a positive multiple of --block-ps")
    if n_steady % block_steps:
        raise SystemExit("--steady-ps must be a multiple of --block-ps")

    baths = {"hot": (g["hot"], a.t_hot), "cold": (g["cold"], a.t_cold)}
    cols = extra_columns(g, has_charge)              # identical on resume
    z_mobile = atoms.positions[g["mobile"], 2]
    edges = np.arange(z_mobile.min() - a.bin, z_mobile.max() + 2 * a.bin, a.bin)
    print(f"NEMD  {len(atoms)} atoms on {', '.join(devices)}: {len(g['fixed'])} fixed, "
          f"{len(g['hot'])} hot, {len(g['free'])} free, {len(g['cold'])} cold "
          f"({g['source']})\n"
          f"      interface z {g['interface_z']:.2f} A, area {g['area']:.2f} A^2, "
          f"dt {a.dt} fs, tau {a.tau} fs, T0 {a.t0} K, hot {a.t_hot} K, "
          f"cold {a.t_cold} K, seed {a.seed}", flush=True)

    ckpt = out / "checkpoint.npz"
    t_start = time.time()
    if a.resume:
        if not ckpt.exists():
            raise SystemExit(f"--resume: {ckpt} does not exist")
        md = RegionLangevin.from_checkpoint(ev, atoms, ckpt, extra_columns=cols)
        truncate_profile(out / "profile.csv", md.nsteps)
        print(f"resumed at step {md.nsteps} "
              f"({md.get_time() / (1000 * units.fs):.2f} ps), heat {md.heat}",
              flush=True)
    else:
        thermalize_momenta(atoms, temperature_K=a.t0,
                           rng=np.random.default_rng(a.seed))   # fixed atoms rest
        # The mobile atoms' centre-of-mass momentum, removed once here and
        # never during the run (a running correction would alter the heat).
        p, m = atoms.get_momenta(), atoms.get_masses()
        mob = g["mobile"]
        p[mob] -= m[mob, None] * p[mob].sum(0) / m[mob].sum()
        p[g["fixed"]] = 0.0
        atoms.set_momenta(p, apply_constraint=False)
        if n_equil:
            eq = Langevin(ev, atoms, dt, temperature_K=a.t0, friction=friction,
                          seed=a.seed, logfile=out / "equil.log",
                          loginterval=a.log_every)
            eq.run(n_equil)
            print(f"equilibrated {a.equil_ps} ps at {a.t0} K: "
                  f"T {eq.thermo.temperature:.1f} K, {eq.performance()}", flush=True)
        md = RegionLangevin(
            ev, atoms, dt, baths, friction, seed=a.seed + 1,
            logfile=out / "thermo.log", loginterval=a.log_every, extra_columns=cols,
            trajectory=out / "trajectory.extxyz", traj_interval=a.traj_every,
            trajectory_properties=("energy", "forces", "velocities", "charges"),
            checkpoint=ckpt, checkpoint_interval=ckpt_steps,
            max_temperature=5 * max(a.t_hot, a.t0),
        )
    prof = ZProfile(md, g["mobile"], edges, out / "profile.csv", append=a.resume)
    md.attach(prof.sample, a.profile_every, sync_atoms=False)
    md.attach(prof.flush, block_steps, sync_atoms=False)

    if md.nsteps < n_steady:
        print(f"steady stage: baths on for {a.steady_ps} ps, heat not counted",
              flush=True)
        md.run(n_steady - md.nsteps)
        md.reset_heat()
        md.save_checkpoint()            # the periodic one preceded the reset
        print(f"production: {a.run_ps} ps from t = {a.steady_ps} ps, heat from zero",
              flush=True)
    md.run(n_steady + n_run - md.nsteps)
    prof.close()
    md.close()

    meta = {
        "model": a.model, "head": a.head, "structure": a.structure,
        "n_atoms": len(atoms), "groups": {k: len(g[k]) for k in
                                          ("fixed", "hot", "free", "cold", "mobile")},
        "group_source": g["source"], "z_range": g["z_range"],
        "interface_z": g["interface_z"], "area": g["area"],
        "charge_filter": a.charge_filter, "compress": a.compress,
        "dt_fs": a.dt, "tau_fs": a.tau, "T0": a.t0, "T_hot": a.t_hot,
        "T_cold": a.t_cold, "seed": a.seed, "equil_ps": a.equil_ps,
        "steady_ps": a.steady_ps, "run_ps": a.run_ps, "bin_A": a.bin,
        "block_ps": a.block_ps, "heat_eV": md.heat, "nsteps": md.nsteps,
        "performance": md.performance(), "wall_s": time.time() - t_start,
    }
    with open(out / "run_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    heat = md.heat
    print(f"\ndone: Q_hot {heat['hot']:+.3f} eV, Q_cold {heat['cold']:+.3f} eV over "
          f"{a.run_ps} ps of production\n  {md.performance()}\n"
          f"  python analyze.py --out {out}", flush=True)


if __name__ == "__main__":
    main()
