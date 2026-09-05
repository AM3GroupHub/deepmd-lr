# SPDX-License-Identifier: LGPL-3.0-or-later
"""Turn the UMA LiFSI trajectory into a periodic DeePMD training set.

The trajectory is a **slab**: `pbc="T T F"`, a Li-metal electrode with liquid
electrolyte above it, 28.08 x 28.08 x 106.33 A with only ~2 A of gap. Two
things have to happen before it can train a 3D-periodic model.

**Add vacuum.** With a 2 A gap, declaring the cell 3D-periodic would put 427
atoms at the top face within the 6 A descriptor cutoff of 256 atoms at the
bottom face -- neighbours that the UMA labels (computed without z-periodicity)
never saw. The model would be asked to reproduce forces from an interaction
that is not in the reference. The gap must exceed twice the descriptor cutoff;
the default 20 A is comfortably past that.

**Keep it physically 2D.** Padding the cell does not make the physics
3D-periodic: the slab still interacts with its images through the vacuum, and
for a dipolar electrode/electrolyte interface that error is large and barely
shrinks as vacuum is added. That is what `slab_correction` in the model's
`long_range` block is for -- padding and the correction are a pair, not
alternatives.

Coordinates are also unwrapped in x and y (they run outside the cell), so they
are wrapped back in. z is left alone: it is the open direction.
"""

import argparse
import pathlib

import numpy as np

ATOMS_PER_FRAME = 6520
LINES_PER_FRAME = ATOMS_PER_FRAME + 2


def parse_frame(lines):
    """Parse one extxyz frame into (symbols, positions, forces, cell, energy)."""
    header = lines[1]
    lattice = header.split('Lattice="')[1].split('"')[0]
    cell = np.fromstring(lattice, sep=" ").reshape(3, 3)
    energy = float(header.split("energy=")[1].split()[0])

    body = np.array([ln.split() for ln in lines[2:]], dtype=object)
    symbols = body[:, 0].astype(str)
    values = body[:, 1:7].astype(np.float64)
    return symbols, values[:, :3], values[:, 3:], cell, energy


def stream_frames(path, wanted):
    """Yield (index, parsed frame) for the requested indices, in one pass.

    The file is ~2 GB, so it is streamed rather than loaded, and only the
    selected frames are parsed.
    """
    wanted = set(wanted)
    buffer, index = [], 0
    with open(path) as handle:
        for line in handle:
            buffer.append(line)
            if len(buffer) == LINES_PER_FRAME:
                if index in wanted:
                    yield index, parse_frame(buffer)
                buffer, index = [], index + 1
    if buffer and len(buffer) == LINES_PER_FRAME and index in wanted:
        yield index, parse_frame(buffer)


def count_frames(path):
    with open(path) as handle:
        lines = sum(1 for _ in handle)
    return lines // LINES_PER_FRAME, lines % LINES_PER_FRAME


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="LiFSI_EC_DMC_1M_T300.xyz")
    ap.add_argument("--out", default="data")
    ap.add_argument("--frames", type=int, default=300,
                    help="configurations to sample, spread over the trajectory")
    ap.add_argument("--skip", type=int, default=0,
                    help="discard this many leading frames (equilibration)")
    ap.add_argument("--vacuum", type=float, default=20.0,
                    help="gap along z, in A; must exceed 2 x descriptor cutoff")
    ap.add_argument("--valid-fraction", type=float, default=0.2)
    args = ap.parse_args()

    traj = pathlib.Path(args.traj)
    total, remainder = count_frames(traj)
    if remainder:
        print(f"note: ignoring a trailing partial frame ({remainder} lines)")
    usable = total - args.skip
    if usable < args.frames:
        raise SystemExit(
            f"asked for {args.frames} frames but only {usable} are available "
            f"after skipping {args.skip}"
        )
    # Spread the sample over the whole trajectory rather than taking a block,
    # so the set covers the run instead of one correlated stretch.
    picks = np.unique(
        np.linspace(args.skip, total - 1, args.frames).round().astype(int)
    )
    print(f"{total} frames in the trajectory; sampling {len(picks)} "
          f"(stride ~{(total - args.skip) / len(picks):.1f})")

    coords, forces, boxes, energies = [], [], [], []
    symbols = cell_z = None
    for count, (index, (sym, pos, frc, cell, energy)) in enumerate(
        stream_frames(traj, picks), start=1
    ):
        if symbols is None:
            symbols = sym
        elif not np.array_equal(symbols, sym):
            raise SystemExit(f"frame {index} has a different atom ordering")

        pos = pos.copy()
        # Wrap the periodic directions back into the cell; leave z, the open
        # direction, exactly as the reference had it.
        for axis in (0, 1):
            length = cell[axis, axis]
            pos[:, axis] -= np.floor(pos[:, axis] / length) * length
        # Centre the slab in a padded cell so both faces get the same gap.
        span = pos[:, 2].max() - pos[:, 2].min()
        new_z = span + args.vacuum
        pos[:, 2] += (new_z - span) / 2 - pos[:, 2].min()
        padded = cell.copy()
        padded[2, 2] = new_z
        cell_z = new_z

        coords.append(pos.reshape(-1))
        forces.append(frc.reshape(-1))
        boxes.append(padded.reshape(-1))
        energies.append(energy)
        if count % 50 == 0:
            print(f"  parsed {count}/{len(picks)}")

    coords = np.array(coords)
    forces = np.array(forces)
    boxes = np.array(boxes)
    energies = np.array(energies)

    type_map = sorted(str(s) for s in set(symbols))
    atype = np.array([type_map.index(s) for s in symbols])

    n_valid = max(int(len(coords) * args.valid_fraction), 1)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(coords))
    splits = {"valid": order[:n_valid], "train": order[n_valid:]}

    out = pathlib.Path(args.out)
    for name, idx in splits.items():
        d = out / name
        (d / "set.000").mkdir(parents=True, exist_ok=True)
        np.savetxt(d / "type.raw", atype, fmt="%d")
        (d / "type_map.raw").write_text("\n".join(type_map) + "\n")
        np.save(d / "set.000/coord.npy", coords[idx])
        np.save(d / "set.000/box.npy", boxes[idx])
        np.save(d / "set.000/energy.npy", energies[idx])
        np.save(d / "set.000/force.npy", forces[idx])
        print(f"{name:6s}: {len(idx)} frames -> {d}")

    print(f"\n{ATOMS_PER_FRAME} atoms/frame, type_map {type_map}")
    print(f"cell: {boxes[0][0]:.2f} x {boxes[0][4]:.2f} x {cell_z:.2f} A "
          f"({args.vacuum:.1f} A vacuum along z)")
    print(f"energy: {energies.mean():.1f} +- {energies.std():.1f} eV")
    print(f"force RMS: {np.sqrt((forces ** 2).mean()):.4f} eV/A")
    print("\nThe cell is now formally 3D-periodic, but the physics is still")
    print("2D. Train with slab_correction: true and pbc: [true, true, false].")


if __name__ == "__main__":
    main()
