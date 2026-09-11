#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Convert a trajectory into a DeePMD-kit training system.

    python scripts/xyz_to_deepmd.py traj.xyz -o data
    python scripts/xyz_to_deepmd.py dump.traj -o data --valid-fraction 0.2
    python scripts/xyz_to_deepmd.py OUTCAR -o data --format vasp-out

Reading goes through ``ase.io.read(..., index=":")``, so any format ASE knows
works -- extxyz, ASE ``.traj``, VASP OUTCAR/XDATCAR, CP2K, Gaussian, and the
rest. ASE decides by extension unless ``--format`` says otherwise.

Assumes **every frame holds the same atoms in the same order**, which is the
ordinary case for one MD or relaxation trajectory, and is what lets the whole
thing be a single DeePMD system. The assumption is checked rather than
trusted: a mismatch is a clear error here instead of a confusing shape error
somewhere inside training. For a file mixing several species, split it first
-- a DeePMD system is one fixed composition and one fixed atom count.

What is written, under ``<out>/``::

    type.raw           per-atom indices, LOCAL to this system
    type_map.raw       the elements this system contains, sorted
    nopbc              empty marker, only when the frames are non-periodic
    set.000/coord.npy  (nframes, natoms * 3)
    set.000/energy.npy (nframes,)          when the frames carry energies
    set.000/force.npy  (nframes, natoms*3) when the frames carry forces
    set.000/box.npy    (nframes, 9)        only when periodic

``type.raw`` indices point into this system's own ``type_map.raw``, not into
the model's global ``type_map``; DeePMD-kit remaps them at load time. That is
what lets systems with different element subsets share one model.
"""

import argparse
import pathlib
import sys

import numpy as np


def load_frames(path, fmt=None, index=":"):
    """Read a trajectory through ASE and return it as a list of Atoms."""
    from ase.io import read  # imported late so --help works without ASE

    frames = read(path, index=index, format=fmt)
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise SystemExit(f"{path}: no frames read")
    return frames


def check_consistent(frames):
    """Require identical atoms, in identical order, in every frame.

    Cheap to verify and worth verifying: a trajectory that changes composition
    partway through would otherwise produce a system whose ``type.raw`` is
    silently wrong for most of its frames.
    """
    reference = frames[0].get_chemical_symbols()
    for i, atoms in enumerate(frames[1:], start=1):
        symbols = atoms.get_chemical_symbols()
        if symbols != reference:
            if len(symbols) != len(reference):
                detail = (
                    f"{len(reference)} atoms in frame 0, "
                    f"{len(symbols)} in frame {i}"
                )
            else:
                bad = next(
                    j for j, (a, b) in enumerate(zip(reference, symbols, strict=True))
                    if a != b
                )
                detail = (
                    f"frame {i} differs at atom {bad}: "
                    f"{reference[bad]!r} became {symbols[bad]!r}"
                )
            raise SystemExit(
                f"frames are not a single system -- {detail}.\n"
                "A DeePMD system is one fixed composition and atom count. Split "
                "the trajectory by composition and convert each part separately."
            )
    return reference


def collect(frames, symbols):
    """Pull coordinates, and whatever labels the frames actually carry.

    Energy and forces are read through the frame's calculator, which is where
    ASE puts them when reading a labelled trajectory. A frame without them is
    not an error -- coordinate-only data is still a valid system, it just
    needs the matching loss prefactors set to zero -- but a file where only
    *some* frames are labelled is, since the arrays would not line up.
    """
    natoms = len(symbols)
    coord = np.array([a.get_positions().reshape(-1) for a in frames], dtype=np.float64)

    def maybe(getter, name, shape):
        got = []
        for atoms in frames:
            try:
                got.append(np.asarray(getter(atoms), dtype=np.float64).reshape(-1))
            except Exception:  # noqa: BLE001 -- any calculator failure means "absent"
                got.append(None)
        present = [g is not None for g in got]
        if not any(present):
            return None
        if not all(present):
            missing = present.index(False)
            raise SystemExit(
                f"{name} is present in some frames but not all (frame {missing} "
                f"has none). Convert the labelled frames separately, or drop "
                f"{name} from the training loss."
            )
        return np.array(got).reshape(len(frames), *shape)

    energy = maybe(lambda a: a.get_potential_energy(), "energy", (1,))
    if energy is not None:
        energy = energy.reshape(-1)
    force = maybe(lambda a: a.get_forces(), "force", (natoms * 3,))

    periodic = bool(frames[0].pbc.any())
    for i, atoms in enumerate(frames):
        if bool(atoms.pbc.any()) != periodic:
            raise SystemExit(
                f"frame {i} is {'periodic' if not periodic else 'non-periodic'} "
                f"while frame 0 is not -- one system cannot be both."
            )
    box = (
        np.array([np.asarray(a.get_cell()).reshape(-1) for a in frames])
        if periodic
        else None
    )
    return coord, energy, force, box, periodic


def charge_spin_from(frames):
    """Per-frame ``[charge, multiplicity]``, if the frames say anything.

    Read from ``atoms.info`` (extxyz puts header keys there) falling back to
    ASE's own ``initial_charges``. Written only when at least one frame is not
    the neutral singlet default, since the file is otherwise pure noise -- and
    DeePMD-kit reads it only when the descriptor sets ``add_chg_spin_ebd``.
    """
    rows, interesting = [], False
    for atoms in frames:
        info = atoms.info
        charge = next(
            (float(info[k]) for k in ("charge", "total_charge", "Charge") if k in info),
            None,
        )
        if charge is None:
            try:
                total = float(atoms.get_initial_charges().sum())
                charge = total if abs(total) > 1e-8 else 0.0
            except Exception:  # noqa: BLE001
                charge = 0.0
        spin = next(
            (
                float(info[k])
                for k in ("spin", "multiplicity", "Multiplicity", "spin_multiplicity")
                if k in info
            ),
            1.0,
        )
        if abs(charge) > 1e-8 or abs(spin - 1.0) > 1e-8:
            interesting = True
        rows.append([charge, spin])
    return np.array(rows, dtype=np.float64) if interesting else None


def write_system(out, indices, arrays, atype, type_map, periodic):
    """Write one split (train or valid) as a DeePMD npy system."""
    coord, energy, force, box, charge_spin = arrays
    out.mkdir(parents=True, exist_ok=True)
    (out / "set.000").mkdir(exist_ok=True)
    np.save(out / "set.000/coord.npy", coord[indices])
    if energy is not None:
        np.save(out / "set.000/energy.npy", energy[indices])
    if force is not None:
        np.save(out / "set.000/force.npy", force[indices])
    if periodic:
        np.save(out / "set.000/box.npy", box[indices])
    else:
        # Its presence is the whole mechanism: DeePMD-kit's loader sees this
        # file and hands the model box=None.
        (out / "nopbc").touch()
    if charge_spin is not None:
        np.save(out / "set.000/charge_spin.npy", charge_spin[indices])
    (out / "type.raw").write_text("\n".join(map(str, atype)) + "\n")
    (out / "type_map.raw").write_text("\n".join(type_map) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", help="trajectory in any format ASE can read")
    ap.add_argument("-o", "--out", required=True, help="dataset directory to write")
    ap.add_argument("--format", default=None, help="override ASE's format guess")
    ap.add_argument("--index", default=":", help="ASE slice, e.g. '::10' or '-100:'")
    ap.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        help="fraction held out as a validation system (0 writes train only)",
    )
    ap.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    ap.add_argument(
        "--no-split",
        action="store_true",
        help="write one system at <out> rather than <out>/train and <out>/valid",
    )
    args = ap.parse_args()

    frames = load_frames(args.input, args.format, args.index)
    symbols = check_consistent(frames)
    coord, energy, force, box, periodic = collect(frames, symbols)
    charge_spin = charge_spin_from(frames)

    type_map = sorted(set(symbols))
    atype = np.array([type_map.index(s) for s in symbols], dtype=np.int64)
    arrays = (coord, energy, force, box, charge_spin)
    out = pathlib.Path(args.out)

    if args.no_split or args.valid_fraction <= 0 or len(frames) < 2:
        write_system(out, np.arange(len(frames)), arrays, atype, type_map, periodic)
        splits = {str(out): len(frames)}
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(frames))
        n_valid = max(1, int(round(len(frames) * args.valid_fraction)))
        write_system(out / "valid", order[:n_valid], arrays, atype, type_map, periodic)
        write_system(out / "train", order[n_valid:], arrays, atype, type_map, periodic)
        splits = {
            str(out / "train"): len(frames) - n_valid,
            str(out / "valid"): n_valid,
        }

    labels = [n for n, v in (("energy", energy), ("force", force)) if v is not None]
    print(f"{args.input} -> {out}", file=sys.stderr)
    print(f"  {len(frames)} frames x {len(symbols)} atoms", file=sys.stderr)
    print(f"  type_map {type_map}", file=sys.stderr)
    print(
        f"  {'periodic' if periodic else 'non-periodic (nopbc)'}"
        f"  labels: {', '.join(labels) if labels else 'none (coordinates only)'}"
        f"{'  +charge_spin' if charge_spin is not None else ''}",
        file=sys.stderr,
    )
    for path, n in splits.items():
        print(f"  {path}  {n} frames", file=sys.stderr)
    if charge_spin is not None:
        print(
            "  note: charge_spin is read only when the descriptor sets "
            "`add_chg_spin_ebd: true`",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
