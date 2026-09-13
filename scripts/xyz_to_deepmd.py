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
somewhere inside training. With ``--split-by-composition``, frames are grouped
and stably sorted by element, including coordinates, forces and atom arrays.
A DeePMD system is one fixed composition and one fixed atom count.

The output must be absent or empty. Existing datasets are never overwritten,
so obsolete labels or a stale ``nopbc`` marker cannot survive a rerun.

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
import collections
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


def composition_key(atoms):
    """Sorted symbol tuple — a frame's DeePMD system identity."""
    return tuple(sorted(atoms.get_chemical_symbols()))


def formula(key):
    """A readable directory name for a composition, e.g. ``C1H1O4S2``."""
    counts = collections.Counter(key)
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def group_by_composition(frames):
    """Split a mixed trajectory into one bucket per composition.

    Keyed on the *sorted* symbols, so the same molecule written with its atoms
    in a different order lands in one bucket rather than two. Charge and spin
    are deliberately NOT part of the key: they vary per frame and are carried
    in ``charge_spin.npy``, so the same molecule in two charge states is still
    one system. Sort each frame stably by element, carrying its labels along.
    """
    buckets = collections.OrderedDict()
    for atoms in frames:
        buckets.setdefault(composition_key(atoms), []).append(sort_frame(atoms))
    return buckets


def sort_frame(atoms):
    """Canonical element order without dropping ASE arrays or calculator labels."""
    from ase.calculators.calculator import PropertyNotImplementedError
    from ase.calculators.singlepoint import SinglePointCalculator

    results = {}
    if atoms.calc is not None:
        # Collect lazy calculators before slicing, which does not retain calc.
        for name in ("energy", "forces"):
            try:
                atoms.calc.get_property(name, atoms)
            except PropertyNotImplementedError:
                pass
        results = dict(atoms.calc.results)
    order = np.argsort(atoms.get_chemical_symbols(), kind="stable")
    sorted_atoms = atoms[order]
    for name in (
        "forces",
        "charges",
        "magmoms",
        "energies",
        "stresses",
        "born_effective_charges",
    ):
        if name in results:
            results[name] = np.asarray(results[name])[order].copy()
    if results:
        sorted_atoms.calc = SinglePointCalculator(sorted_atoms, **results)
    return sorted_atoms


def require_empty_output(out):
    """Never mix new arrays with labels or boundary markers from an old run."""
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise SystemExit(
            f"output {out} already exists and is not an empty directory; "
            "choose a new output directory. Existing datasets are not overwritten."
        )


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
                    f"{len(reference)} atoms in frame 0, {len(symbols)} in frame {i}"
                )
            else:
                bad = next(
                    j
                    for j, (a, b) in enumerate(zip(reference, symbols, strict=True))
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


def pad_vacuum(frames, target, axis=2):
    """Re-pad the cell along one axis and centre the slab in it.

    A slab's reference labels are computed with that axis non-periodic, so the
    cell length carries no physics -- but the model is handed a 3D-periodic
    cell, and there the length matters enormously. If the vacuum gap is
    narrower than twice the descriptor cutoff, atoms near one face fall inside
    the cutoff of the next image's opposite face, and the model is asked to
    reproduce forces from an interaction the labels never contained.

    Padding is only half of it: the slab still interacts with its images
    *electrostatically* across the vacuum, and for a dipolar interface that
    error barely shrinks as the gap grows. Cancelling it is what
    `long_range.reciprocal.slab_correction` is for. Padding and the correction
    are a pair, not alternatives.
    """
    for atoms in frames:
        # Save the labels first. ASE's SinglePointCalculator invalidates
        # itself when the geometry changes, so setting positions below would
        # silently discard the energies and forces and leave a
        # coordinates-only dataset that trains on nothing.
        results = dict(getattr(atoms.calc, "results", {})) if atoms.calc else {}
        cell = np.array(atoms.get_cell())
        if target <= cell[axis, axis]:
            raise SystemExit(
                f"--vacuum {target} is not larger than the existing cell "
                f"length {cell[axis, axis]:.3f} along axis {axis}; padding "
                "would compress the cell rather than extend it."
            )
        pos = atoms.get_positions()
        span = pos[:, axis].max() - pos[:, axis].min()
        if span >= target:
            raise SystemExit(
                f"the slab spans {span:.2f} A along axis {axis}, which does "
                f"not fit in a {target} A cell."
            )
        cell[axis, axis] = target
        atoms.set_cell(cell)
        # Centre it, so the vacuum is shared between the two faces rather than
        # sitting entirely above the slab.
        pos[:, axis] += (target - span) / 2.0 - pos[:, axis].min()
        atoms.set_positions(pos)
        if results:
            # Re-attach. A rigid translation along one axis leaves energies and
            # forces unchanged, so the saved values are still the right ones.
            from ase.calculators.singlepoint import (  # noqa: PLC0415
                SinglePointCalculator,
            )

            atoms.calc = SinglePointCalculator(atoms, **results)
    return frames


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
    require_empty_output(out)
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
        "--vacuum",
        type=float,
        default=None,
        metavar="Z",
        help="re-pad the cell along z to Z angstrom and centre the slab in it. "
        "For a 2D slab the gap must exceed twice the descriptor cutoff, or the "
        "model sees across the vacuum into the next periodic image.",
    )
    ap.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        help="fraction held out as a validation system (0 writes train only)",
    )
    ap.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    ap.add_argument(
        "--split-by-composition",
        action="store_true",
        help="the input mixes species: write one system per composition, into "
        "<out>/<formula>/. A DeePMD system is one fixed composition and atom "
        "count, so a mixed file must be split before it can be used.",
    )
    ap.add_argument(
        "--no-split",
        action="store_true",
        help="write one system at <out> rather than <out>/train and <out>/valid",
    )
    args = ap.parse_args()

    require_empty_output(pathlib.Path(args.out))

    frames = load_frames(args.input, args.format, args.index)

    if args.split_by_composition:
        buckets = group_by_composition(frames)
        print(
            f"{args.input} -> {args.out}\n  {len(frames)} frames -> "
            f"{len(buckets)} system(s)",
            file=sys.stderr,
        )
        for key, group in buckets.items():
            name = formula(key)
            convert_one(group, pathlib.Path(args.out) / name, args, label=name)
        return

    convert_one(frames, pathlib.Path(args.out), args, label=None)


def convert_one(frames, out, args, label=None):
    """Convert one already-grouped set of frames into a DeePMD system."""
    require_empty_output(out)
    symbols = check_consistent(frames)
    if args.vacuum is not None:
        frames = pad_vacuum(frames, args.vacuum)
    coord, energy, force, box, periodic = collect(frames, symbols)
    charge_spin = charge_spin_from(frames)

    type_map = sorted(set(symbols))
    atype = np.array([type_map.index(s) for s in symbols], dtype=np.int64)
    arrays = (coord, energy, force, box, charge_spin)
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
    if label is None:
        print(f"{args.input} -> {out}", file=sys.stderr)
    else:
        print(f"  {label}", file=sys.stderr)
    print(f"  {len(frames)} frames x {len(symbols)} atoms", file=sys.stderr)
    print(f"  type_map {type_map}", file=sys.stderr)
    if args.vacuum is not None:
        span = np.ptp(frames[0].get_positions()[:, 2])
        print(
            f"  padded z to {args.vacuum} A; slab spans {span:.2f} A, "
            f"vacuum gap {args.vacuum - span:.2f} A",
            file=sys.stderr,
        )
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
