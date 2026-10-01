"""FIRE-Swap: Metropolis Monte Carlo over Mg <-> Nb B-site swaps in PMN,
Pb(Mg1/3 Nb2/3)O3, with a FIRE relaxation after every trial swap.

Each step picks a random Mg and a random Nb within `--swap-cutoff` (any Nb if
none is that close), exchanges them, relaxes the atoms and the cell, and
accepts with min(1, exp(-dE/kT)) on the relaxed energy. A bare swap costs
about +6 eV and the relaxation gives most of it back, so the decision rides
on the small remainder: the relaxation cannot be skipped, and it is where the
time goes. The settings follow Xu, Cai, Shi, Zhong & Xie, Phys. Rev. B 114,
024204 (2026), Appendix B: 600 K, fmax 0.02 eV/A, at most 200 FIRE steps,
dt 0.1, maxstep 0.2 A, swap cutoff 6.3 A. The stress tolerance is tighter
than theirs; see the last point below.

One evaluator is built once and every trial's `FIRE` runs on it, so the
relaxation never leaves the GPU. Four things make that work:

- **The swap exchanges positions, not species.** The evaluator's atom types
  are fixed for its life; moving the Mg to the Nb's site and the Nb to the
  Mg's is the same configuration. The two atoms jump ~4 A, and the
  evaluator's skin criterion rebuilds its neighbour list on its own.
- **A rejected trial is rolled back** to the last accepted state, its cell
  included, which is pushed to the evaluator too.
- **The frames carry the accepted state's energy**, never a rejected trial's.
- **The cell force is divided by 10** (`SwapFIRE` below), and the stress
  is relaxed to 0.02 GPa. The package's FIRE leaves the cell force whole,
  which suits a box far from its density; after one swap it makes the cell
  the stiff mode, and 8640 atoms took 151 FIRE steps a trial, some more
  than 400. ASE's FrechetCellFilter divides by N instead, which makes the
  cell so soft that 0.02 GPa was never reached in 400 steps; its own
  tolerance, fmax N / V = 0.24 GPa, leaves the 29,160-atom cell 0.6 eV
  above its minimum and the trial energies 0.03 eV (0.6 kT) apart. Divided
  by 10, every size took 82-95 steps to 0.02 GPa.

Step t draws its Mg, its Nb and its Metropolis number from
`default_rng([seed, t])`, so a resumed run continues the same chain.

    python fireswap.py --model pmn-lr-joint.pt --repeat 2 --steps 10000 --out run12
    python fireswap.py --model pmn-lr-joint.pt --repeat 2 --steps 20000 --out run12 \
                       --resume
    python analyze_order.py run12/fireswap.extxyz
"""

import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from deepmd_lr.nvalchemi import FIRE, build_evaluator, read_structure

LOG_HEADER = "step,E[eV],dE[eV],accepted,fire_steps,converged,trial[s],wall[s]\n"


class SwapFIRE(FIRE):
    """FIRE with the cell force divided by `CELL_FACTOR`: the cell is then
    about as stiff as an atom, neither the fast mode nor the slow one."""

    CELL_FACTOR = 10.0

    def _cell_force(self, h):
        return super()._cell_force(h) / self.CELL_FACTOR


def snapshot(atoms, energy):
    """The accepted state: positions, cell and its own energy."""
    frame = atoms.copy()
    frame.calc = SinglePointCalculator(frame, energy=energy)
    return frame


def write_atomic(path, atoms):
    write(f"{path}.tmp", atoms, format="extxyz")
    os.replace(f"{path}.tmp", path)


def propose(atoms, mg, nb, cutoff, rng):
    """A random Mg, and a random Nb within `cutoff` of it (any Nb if none)."""
    i = int(rng.choice(mg))
    near = nb[atoms.get_distances(i, nb, mic=True) < cutoff]
    return i, int(rng.choice(near if len(near) else nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--structure", default="pmn1080.xyz")
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="n x n x n supercell of --structure (2 = 8640 atoms)",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=100,
        help="total MC steps, a resumed run's earlier ones included",
    )
    ap.add_argument("--temperature", type=float, default=600.0, help="K")
    ap.add_argument("--fmax", type=float, default=0.02, help="eV/A")
    ap.add_argument("--max-relax", type=int, default=200, help="FIRE steps per trial")
    ap.add_argument("--swap-cutoff", type=float, default=6.3, help="A")
    ap.add_argument("--relax-cell", choices=("full", "none"), default="full")
    ap.add_argument("--smax", type=float, default=0.02, help="GPa")
    ap.add_argument(
        "--compress",
        action="store_true",
        help="compressed backbone (needs the CUDA-operator build)",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--save-every", type=int, default=100, help="trajectory frame interval"
    )
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--out", default="run")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    log, traj = out / "fireswap.log", out / "fireswap.extxyz"
    ckpt, ckpt_json, best = (
        out / "checkpoint.extxyz",
        out / "checkpoint.json",
        out / "best.extxyz",
    )
    relax_cell = None if a.relax_cell == "none" else a.relax_cell
    kT = a.temperature * units.kB

    if a.resume:
        ck = json.loads(ckpt_json.read_text())
        atoms = read(ckpt)
        seed, step, e_cur, e_best, n_acc = (
            ck[k] for k in ("seed", "step", "energy", "best", "accepted")
        )
        # Drop whatever was written after the checkpoint.
        lines = log.read_text().splitlines(keepends=True)
        log.write_text(
            "".join([lines[0]] + [x for x in lines[1:] if int(x.split(",")[0]) <= step])
        )
        frames = [f for f in read(traj, ":") if f.info["step"] <= step]
        write(traj, frames, format="extxyz")
        print(f"resumed at step {step}, E {e_cur:.5f} eV, seed {seed}")
    else:
        if log.exists():
            raise SystemExit(f"{log} exists: pass --resume, or choose another --out")
        out.mkdir(parents=True, exist_ok=True)
        atoms = read_structure(a.structure).repeat(a.repeat)
        seed = (
            a.seed
            if a.seed is not None
            else int(np.random.SeedSequence().entropy % 2**32)
        )
        step, n_acc = 0, 0
        log.write_text(LOG_HEADER)

    ev = build_evaluator(
        a.model, atoms, compress=a.compress, compute_virial=relax_cell is not None
    )
    sym = np.array(atoms.get_chemical_symbols())
    mg, nb = np.flatnonzero(sym == "Mg"), np.flatnonzero(sym == "Nb")
    smax = a.smax * units.GPa

    def relax(steps, check):
        opt = SwapFIRE(
            ev,
            atoms,
            relax_cell=relax_cell,
            dt=0.1,
            maxstep=0.2,
            logfile=None,
            check=check,
        )
        converged = opt.run(fmax=a.fmax, steps=steps, smax=smax)
        return opt.thermo.energy, opt.nsteps, converged

    print(
        f"{len(atoms)} atoms ({atoms.get_chemical_formula()})  T {a.temperature} K  "
        f"seed {seed}  relax_cell {a.relax_cell}  compress {a.compress}"
    )
    if not a.resume:
        # Stage A: a starting structure relaxed with another model is not
        # relaxed with this one.
        t0 = time.perf_counter()
        e_cur, n, _ = relax(10_000, check=True)
        print(
            f"stage A: {n} FIRE steps, {time.perf_counter() - t0:.1f} s, "
            f"E {e_cur:.5f} eV"
        )
        e_best = e_cur
        state = snapshot(atoms, e_cur)
        state.info["step"] = 0
        write(traj, state, format="extxyz")
        write_atomic(best, state)
    state = snapshot(atoms, e_cur)

    wall0, fire_total, trial_total, n_trials = time.perf_counter(), 0, 0.0, 0
    while step < a.steps:
        step += 1
        rng = np.random.default_rng([seed, step])
        i, j = propose(atoms, mg, nb, a.swap_cutoff, rng)
        u = rng.random()
        p = atoms.get_positions()
        p[[i, j]] = p[[j, i]]
        atoms.set_positions(p)
        t0 = time.perf_counter()
        e, n, converged = relax(a.max_relax, check=False)
        dt = time.perf_counter() - t0
        fire_total, trial_total, n_trials = (
            fire_total + n,
            trial_total + dt,
            n_trials + 1,
        )

        de = e - e_cur
        accepted = de < 0 or u < math.exp(-de / kT)
        if accepted:
            e_cur, n_acc = e, n_acc + 1
            state = snapshot(atoms, e)
        else:
            atoms.set_cell(state.cell, scale_atoms=False)
            atoms.set_positions(state.positions)
            if relax_cell is not None:
                ev.set_cell(state.cell)
        with open(log, "a") as fh:
            fh.write(
                f"{step},{e_cur:.5f},{de:.5f},{int(accepted)},{n},{int(converged)},"
                f"{dt:.3f},{time.perf_counter() - wall0:.1f}\n"
            )

        state.info["step"] = step
        if step % a.save_every == 0:
            write(traj, state, format="extxyz", append=True)
        if accepted and e_cur < e_best:
            e_best = e_cur
            write_atomic(best, state)
        if step % a.checkpoint_every == 0 or step == a.steps:
            write_atomic(ckpt, state)
            ckpt_json.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "step": step,
                        "energy": e_cur,
                        "best": e_best,
                        "accepted": n_acc,
                    }
                )
            )
        if step % 100 == 0 or step == a.steps:
            print(
                f"step {step:7d}  E {e_cur:.4f} eV  accepted {n_acc}  "
                f"{fire_total / n_trials:.1f} FIRE steps/trial  "
                f"{trial_total / n_trials:.2f} s/trial  "
                f"{1e3 * trial_total / max(fire_total, 1):.1f} ms/FIRE step",
                flush=True,
            )


if __name__ == "__main__":
    main()
