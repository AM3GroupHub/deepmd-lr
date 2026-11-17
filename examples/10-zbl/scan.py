"""Push one pair together inside a real frame, with and without ``zbl``.

    python scan.py model.ckpt.pt frame.xyz --window 0.7 0.9 --data train.xyz

For every pair of elements in the frame, the first atom of one and its nearest
neighbour of the other are brought from their own distance down to 0.3
covalent-radius sums, the neighbour moving along the bond. The trained model
and the same weights with ``model.set_zbl({"window": ...})`` are compared, and
below the pair's shortest fitted contact (``--data``; the frame's own distance
without it) the curve with the term is judged: the energy must rise
monotonically, reach at least half the pair's ZBL energy, and keep the
long-range part and the two charges bounded.

Moving one atom can run it into a third one first: the table names the atom
nearest to the moved one, and that contact is then what the row shows. To see
what such a path should look like, write the scanned geometries with
``--frames``, evaluate them with something you trust (DFT, another model) and
hand the result back with ``--reference``; it is drawn as a third line:

    frames = ase.io.read("scan_frames.xyz", ":")
    for atoms in frames:
        atoms.calc = trusted
        atoms.get_potential_energy()
    ase.io.write("scan_reference.xyz", frames)
"""

import argparse
import itertools
import json

import numpy as np
import torch
from ase.data import atomic_numbers, covalent_radii
from ase.io import read, write
from ase.neighborlist import neighbor_list

from deepmd_lr.ase import LongRangeCalculator, load_model


def radius_sum(a: str, b: str) -> float:
    return float(covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]])


def pair_floors(frames, cutoff: float = 3.0) -> dict[str, float]:
    """The shortest distance of every pair of elements in ``frames``."""
    floors: dict[str, float] = {}
    for atoms in frames:
        i, j, d = neighbor_list("ijd", atoms, cutoff)
        symbols = np.array(atoms.get_chemical_symbols())
        for a, b in {
            tuple(sorted(p)) for p in zip(symbols[i], symbols[j], strict=True)
        }:
            close = d[(symbols[i] == a) & (symbols[j] == b)].min()
            floors[f"{a}-{b}"] = min(floors.get(f"{a}-{b}", np.inf), float(close))
    return floors


def evaluate(calc, atoms) -> tuple:
    """Energy, its short- and long-range parts, and the charges (or None)."""
    atoms = atoms.copy()
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    results = calc.results
    return (
        energy,
        results.get("energy_short_range", energy),
        results.get("energy_long_range", 0.0),
        results.get("charges"),
    )


def zbl_pair(model, a: str, b: str, radii) -> np.ndarray:
    """The model's switched ZBL energy of one ``a``-``b`` pair at ``radii``."""
    type_map = list(model.get_type_map())
    device = model.zbl.numbers.device
    atype = torch.tensor([type_map.index(a), type_map.index(b)], device=device)
    edges = torch.tensor([[0, 1], [1, 0]], device=device)
    mask = torch.ones(2, dtype=torch.bool, device=device)
    energies = []
    for r in radii:
        vec = torch.tensor(
            [[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        energies.append(float(model.zbl.atomic_energy(atype, edges, vec, mask).sum()))
    return np.array(energies)


def scan(calcs: dict, frame, a: str, b: str, n: int, frames: list) -> dict:
    """Bring the first ``a`` atom and its nearest ``b`` together; the
    geometries are appended to ``frames``."""
    symbols = np.array(frame.get_chemical_symbols())
    i = int(np.flatnonzero(symbols == a)[0])
    others = np.flatnonzero(symbols == b)
    others = others[others != i]
    j = int(others[frame.get_distances(i, others, mic=True).argmin()])
    bond = frame.get_distance(i, j, mic=True, vector=True)
    r0 = float(np.linalg.norm(bond))
    radii = np.linspace(r0, 0.3 * radius_sum(a, b), n)
    rest = [k for k in range(len(frame)) if k not in (i, j)]
    out = {"atoms": [i, j], "r": radii.tolist(), "third": []}
    reference = {name: evaluate(calc, frame) for name, calc in calcs.items()}
    for name in calcs:
        out[name] = {key: [] for key in ("total", "short", "long", "q_i", "q_j")}
    for r in radii:
        atoms = frame.copy()
        atoms.positions[j] = atoms.positions[i] + bond / r0 * r
        atoms.info = {"scan": f"{a}-{b}", "distance": float(r)}
        frames.append(atoms)
        if rest:
            gaps = atoms.get_distances(j, rest, mic=True)
            k = rest[int(gaps.argmin())]
            out["third"].append(f"{symbols[k]}{k} {gaps.min():.2f}")
        for name, calc in calcs.items():
            energy, short, long, charges = evaluate(calc, atoms)
            row = out[name]
            row["total"].append(energy - reference[name][0])
            row["short"].append(short - reference[name][1])
            row["long"].append(long - reference[name][2])
            row["q_i"].append(np.nan if charges is None else float(charges[i]))
            row["q_j"].append(np.nan if charges is None else float(charges[j]))
    return out


def judge(result: dict, floor: float, zbl: np.ndarray) -> dict:
    """The curve with the term, below ``floor``: monotone, repulsive, bounded."""
    r = np.array(result["r"])
    below = r < floor
    row = {key: np.array(value)[below] for key, value in result["zbl"].items()}
    total, long = row["total"], row["long"]
    charge = np.nanmax(np.abs([row["q_i"], row["q_j"]])) if below.any() else 0.0
    checks = {
        "finite": bool(np.isfinite(total).all()),
        "monotone": bool((np.diff(total) >= -1e-3).all()),
        "repulsive": bool(below.any() and total[-1] >= 0.5 * zbl[below][-1]),
        "long_range_bounded": bool(
            (np.abs(long) <= np.maximum(0.5 * np.abs(total), 1.0)).all()
        ),
        "charges_bounded": bool(not charge > 2.0),
    }
    return {"pass": all(checks.values()), **checks}


def plot(report: dict, path: str, title: str) -> None:
    """One column per pair: the energy on top, the larger of the two charges
    below (long-range models). Same colour, same model, in every panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e8e7e2", "#fcfcfb"
    series = {"zbl": ("with zbl", "#2a78d6"), "plain": ("trained model", "#eb6834")}
    reference = (report.get("reference_label"), "#1baf7a")
    pairs = report["pairs"]
    charged = report["long_range"]
    rows = 2 if charged else 1
    fig, axes = plt.subplots(
        rows,
        len(pairs),
        figsize=(4.2 * len(pairs), 3.4 * rows),
        facecolor=surface,
        squeeze=False,
        sharex="col",
    )
    for column, (name, entry) in enumerate(pairs.items()):
        r = np.array(entry["r"])
        top = axes[0, column]
        for key, (label, colour) in series.items():
            top.plot(r, entry[key]["total"], color=colour, lw=2, label=label)
        shown = [entry["zbl"]["total"], entry["zbl_pair"]]
        if "reference" in entry:
            top.plot(
                r, entry["reference"], color=reference[1], lw=2, label=reference[0]
            )
            shown.append(entry["reference"])
        top.plot(r, entry["zbl_pair"], color=muted, lw=1.2, label="ZBL of the pair")
        top.set_yscale("symlog", linthresh=1.0)
        # The axis follows the curves worth reading; where the trained model
        # falls by orders of magnitude more, it leaves the panel.
        low, high = np.min(shown), np.max(shown)
        top.set_ylim(min(-10.0, 3.0 * low), max(10.0, 3.0 * high))
        verdict = "pass" if entry["judgement"]["pass"] else "FAIL"
        top.set_title(f"{name}   {verdict}", loc="left", fontsize=10, color=ink)
        if charged:
            bottom = axes[1, column]
            for key, (_label, colour) in series.items():
                largest = np.nanmax(
                    np.abs([entry[key]["q_i"], entry[key]["q_j"]]), axis=0
                )
                bottom.plot(r, largest, color=colour, lw=2)
            bottom.set_yscale("symlog", linthresh=1.0)
        for ax in axes[:, column]:
            ax.axvspan(*entry["window"], color=grid, lw=0)
            ax.axvline(entry["floor"], color=muted, lw=1)
            ax.set_facecolor(surface)
            ax.grid(True, color=grid, lw=1)
            ax.tick_params(colors=muted, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(grid)
        axes[-1, column].set_xlabel("pair distance (A)", color=muted, fontsize=9)
    axes[0, 0].set_ylabel("energy change (eV)", color=muted, fontsize=9)
    if charged:
        axes[1, 0].set_ylabel(
            "largest |charge| of the pair (e)", color=muted, fontsize=9
        )
    # One legend for the figure, above the panels, where no curve runs.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=muted,
    )
    fig.suptitle(
        f"{title}: shaded = the window, vertical line = shortest fitted contact",
        x=0.01,
        ha="left",
        fontsize=10,
        color=ink,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=surface)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="training checkpoint")
    parser.add_argument("frame", help="structure file holding the frame")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--window", type=float, nargs=2, required=True)
    parser.add_argument("--data", help="training set, for the pair floors")
    parser.add_argument("--points", type=int, default=29)
    parser.add_argument("--frames", help="write the scanned geometries here")
    parser.add_argument("--reference", help="those geometries, with energies")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--json")
    parser.add_argument("--png")
    args = parser.parse_args()

    frame = read(args.frame, index=args.index)
    plain = load_model(args.model)
    model = load_model(args.model)
    model.set_zbl({"window": args.window})
    calcs = {"plain": LongRangeCalculator(plain), "zbl": LongRangeCalculator(model)}
    floors = pair_floors(read(args.data, index=":")) if args.data else {}
    for name, floor in floors.items():
        print(
            f"shortest {name} in the data: {floor:.3f} A = "
            f"{floor / radius_sum(*name.split('-')):.2f} covalent-radius sums"
        )

    report = {
        "model": args.model,
        "window": args.window,
        "long_range": bool(getattr(model, "has_long_range", False)),
        "reference_label": args.reference_label,
        "pairs": {},
    }
    # The frame itself first, then every scanned geometry in order.
    start = frame.copy()
    start.info = {"scan": "start", "distance": 0.0}
    frames = [start]
    trusted: dict[str, list[float]] = {}
    if args.reference:
        evaluated = read(args.reference, index=":")
        zero = evaluated[0].get_potential_energy()
        for atoms in evaluated[1:]:
            trusted.setdefault(atoms.info["scan"], []).append(
                atoms.get_potential_energy() - zero
            )
    elements = sorted(set(frame.get_chemical_symbols()))
    for a, b in itertools.combinations_with_replacement(elements, 2):
        name = f"{a}-{b}"
        result = scan(calcs, frame, a, b, args.points, frames)
        if name in trusted:
            if len(trusted[name]) != len(result["r"]):
                raise SystemExit(f"--reference holds another scan of {name}")
            result["reference"] = trusted[name]
        result["zbl_pair"] = zbl_pair(model, a, b, result["r"]).tolist()
        result["window"] = [x * radius_sum(a, b) for x in args.window]
        result["floor"] = floors.get(name, result["r"][0])
        result["judgement"] = judge(
            result, result["floor"], np.array(result["zbl_pair"])
        )
        report["pairs"][name] = result
        print(
            f"\n{name}: atoms {result['atoms']}, window "
            f"{result['window'][0]:.3f}-{result['window'][1]:.3f} A, "
            f"floor {result['floor']:.3f} A -> {result['judgement']}"
        )
        print(
            "     r   plain dE     zbl dE  short-range  long-range     q_i     q_j"
            "  ZBL pair  reference  nearest third atom"
        )
        for k, r in enumerate(result["r"]):
            row = result["zbl"]
            print(
                f"  {r:5.3f} {result['plain']['total'][k]:10.3g} "
                f"{row['total'][k]:10.3f} {row['short'][k]:12.3f} "
                f"{row['long'][k]:11.3f} {row['q_i'][k]:7.3f} {row['q_j'][k]:7.3f} "
                f"{result['zbl_pair'][k]:9.3f} "
                + (f"{result['reference'][k]:10.3f}" if name in trusted else " " * 10)
                + f"  {result['third'][k] if result['third'] else ''}"
            )
    if args.frames:
        write(args.frames, frames)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=1)
    if args.png:
        plot(report, args.png, f"window {args.window}")


if __name__ == "__main__":
    main()
