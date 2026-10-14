#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Calibrate a ``dp_lr`` model's Born effective charges against DFPT labels.

    python scripts/calibrate_born.py model.pt --fit valid_BEC.xyz --test test_BEC.xyz
    python scripts/calibrate_born.py model.pt --fit valid_BEC.xyz --plot figure_bec
    python scripts/calibrate_born.py model.pt --test test_BEC.xyz --scale 1.754

The protocol
------------
A ``dp_lr`` model is trained on energies and forces only. Its charges are never
supervised, so the Born tensors it reports (``deepmd_lr/born.py``) carry two
free factors that the energy fit does not pin down: an overall **scale**, which
is the electronic screening ``sqrt(eps_inf)`` (``docs/born-charges.md``), and
an overall **sign**, because the long-range energy depends on ``q_i q_j`` and
training lands on either charge branch. Both are determined *after* training
by one least-squares fit of the raw tensors onto DFPT labels, over every frame,
atom and tensor component::

    s = sum(Z_raw * Z_dft) / sum(Z_raw * Z_raw)

``|s|`` is ``sqrt(eps_inf)`` and ``sign(s)`` the charge sign. The fit is made
on a validation set (``--fit``) and the errors are then reported on a held-out
set (``--test``) at that fixed ``s``, so the test numbers never see their own
labels. This is the procedure behind the water model's reported Born-charge
errors (``sqrt(eps_inf) = 1.754``, test RMSE 0.049 e), and the one NEP's
``charge_mode`` baselines were scored with for the comparison.

The fitted factor is a calibration of the model's Born tensors, not the
physical dielectric constant: the water fit gives ``eps_inf = 3.08`` where
DFPT's own electronic dielectric constant is 1.84. Pass ``--eps-key epsilon``
when the frames carry that tensor in ``atoms.info`` and the script reports it
alongside for comparison. Born charges are never trained on; calibrating the
scale afterwards is the whole treatment.

Inputs
------
Frames in any format ASE reads (``ase.io.read(..., index=":")``). Each atom
carries its DFPT tensor as nine numbers under ``--key`` (default ``BEC``),
laid out row-major as ``[polarization, displacement]`` -- the same axis order
as ``calc.results["born_charge"]``, so nothing is transposed. Frames may also
carry energies and forces; when they do, the force and offset-removed energy
errors are reported as well, since the same calls produce them.

Compressed models need ``DP_CUDA_INFER=1`` for the table kernel; ``2`` is
refused for Born tensors by the model itself (``docs/born-charges.md``).

Outputs
-------
``--out`` (default ``calibrate_born.json``): the fit and every error
statistic. ``--plot PREFIX``: parity figures of the diagonal and off-diagonal
components on the test set (or the fit set), coloured by element.
``--dump-dir``: the raw arrays (``bec_pred.npy`` with ``s`` applied,
``bec_ref.npy``, ``forces_*``, ``energy_*``, ``symbols.npy``) in
``[frame, atom, polarization, displacement]`` layout.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

__all__ = [
    "born_statistics",
    "calibrate",
    "evaluate_frames",
    "fit_scale",
    "read_frames",
]


def read_frames(path, key: str = "BEC") -> list:
    """Every frame of ``path`` that carries per-atom Born labels under ``key``."""
    from ase.io import read

    frames = read(str(path), index=":")
    if not frames:
        raise ValueError(f"{path}: no frames")
    missing = [i for i, a in enumerate(frames) if key not in a.arrays]
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} of {len(frames)} frames have no per-atom "
            f"array {key!r} (first: frame {missing[0]})"
        )
    for i, a in enumerate(frames):
        if a.arrays[key].shape != (len(a), 9):
            raise ValueError(
                f"{path} frame {i}: {key!r} has shape {a.arrays[key].shape}, "
                f"expected ({len(a)}, 9)"
            )
    return frames


def evaluate_frames(calc, frames: list, key: str = "BEC") -> dict:
    """Run ``calc`` on every frame and gather raw Born tensors next to the labels.

    ``calc`` is a :class:`deepmd_lr.ase.LongRangeCalculator` built with
    ``efield=[0, 0, 0]`` (so Born tensors are produced) and ``eps_inf=1`` (so
    they are raw). Returns per-frame arrays: ``z_raw``/``z_ref`` as
    ``(nframes, natoms, 3, 3)``, ``charges`` as ``(nframes, natoms)``, and
    ``forces_*``/``energy_*`` when the frames carry those labels.
    """
    if getattr(calc, "efield", None) is None:
        raise ValueError(
            "build the calculator with efield=[0, 0, 0]: Born tensors are only "
            "produced under a field, and a zero field changes nothing else"
        )
    if float(getattr(calc, "eps_inf", 1.0)) != 1.0:
        raise ValueError("build the calculator with eps_inf=1: the fit wants raw Z")
    symbols = np.array(frames[0].get_chemical_symbols())
    z_raw, z_ref, charges, f_pred, f_ref, e_pred, e_ref = [], [], [], [], [], [], []
    for i, atoms in enumerate(frames):
        if np.any(np.array(atoms.get_chemical_symbols()) != symbols):
            raise ValueError(f"frame {i}: atoms differ from frame 0 (same system?)")
        labels = atoms.calc.results if atoms.calc is not None else {}
        f_lab = atoms.get_forces() if "forces" in labels else None
        e_lab = atoms.get_potential_energy() if "energy" in labels else None
        a = atoms.copy()
        a.calc = calc
        f_pred.append(a.get_forces())
        e_pred.append(a.get_potential_energy())
        if "born_charge" not in calc.results:
            raise ValueError(
                "the model reports no Born tensors: a model without a long_range "
                "block has no charges to build them from"
            )
        z_raw.append(np.asarray(calc.results["born_charge"]).reshape(len(a), 3, 3))
        charges.append(np.asarray(calc.results["charges"]).reshape(len(a)))
        z_ref.append(np.asarray(atoms.arrays[key], dtype=float).reshape(len(a), 3, 3))
        if f_lab is not None:
            f_ref.append(f_lab)
        if e_lab is not None:
            e_ref.append(e_lab)
    out = {
        "symbols": symbols,
        "z_raw": np.array(z_raw),
        "z_ref": np.array(z_ref),
        "charges": np.array(charges),
    }
    if len(f_ref) == len(frames):
        out["forces_pred"], out["forces_ref"] = np.array(f_pred), np.array(f_ref)
    if len(e_ref) == len(frames):
        out["energy_pred"], out["energy_ref"] = np.array(e_pred), np.array(e_ref)
    return out


def fit_scale(z_raw: np.ndarray, z_ref: np.ndarray) -> float:
    """Least-squares ``s`` with ``z_ref ~ s * z_raw`` over every component.

    Negative means the model sits on the negative charge branch; ``|s|`` is
    ``sqrt(eps_inf)``.
    """
    z_raw = np.asarray(z_raw, dtype=float)
    z_ref = np.asarray(z_ref, dtype=float)
    if z_raw.shape != z_ref.shape:
        raise ValueError(f"shape mismatch: {z_raw.shape} vs {z_ref.shape}")
    denominator = float((z_raw * z_raw).sum())
    if denominator == 0.0:
        raise ValueError("the model's Born tensors are identically zero")
    return float((z_raw * z_ref).sum() / denominator)


def born_statistics(z: np.ndarray, z_ref: np.ndarray, symbols: np.ndarray) -> dict:
    """MAE/RMSE over all, diagonal and off-diagonal components; isotropic means."""
    d = z - z_ref
    eye = np.eye(3, dtype=bool)
    out = {
        "mae_e": float(np.abs(d).mean()),
        "rmse_e": float(np.sqrt((d**2).mean())),
        "diag_mae_e": float(np.abs(d[..., eye]).mean()),
        "diag_rmse_e": float(np.sqrt((d[..., eye] ** 2).mean())),
        "offdiag_mae_e": float(np.abs(d[..., ~eye]).mean()),
        "offdiag_rmse_e": float(np.sqrt((d[..., ~eye] ** 2).mean())),
        "isotropic": {},
    }
    for el in dict.fromkeys(symbols):
        sel = symbols == el
        out["isotropic"][el] = {
            "model": _isotropic(z[:, sel]),
            "dft": _isotropic(z_ref[:, sel]),
            "n_atoms": int(sel.sum()),
        }
    return out


def _isotropic(z: np.ndarray) -> float:
    """Mean of tr(Z)/3 over the atoms and frames given."""
    return float(np.trace(z, axis1=-2, axis2=-1).mean() / 3)


def _force_energy_statistics(data: dict) -> dict:
    out = {}
    if "forces_pred" in data:
        d = data["forces_pred"] - data["forces_ref"]
        out["force_mae_eV_per_A"] = float(np.abs(d).mean())
        out["force_rmse_eV_per_A"] = float(np.sqrt((d**2).mean()))
    if "energy_pred" in data:
        # A DP model's energy carries an arbitrary per-element constant, which no
        # dynamics can feel; the offset-removed error is the one that compares models.
        nat = data["z_raw"].shape[1]
        de = data["energy_pred"] - data["energy_ref"]
        out["energy_offset_meV_per_atom"] = float(de.mean() / nat * 1e3)
        out["energy_mae_offset_removed_meV_per_atom"] = float(
            np.abs(de - de.mean()).mean() / nat * 1e3
        )
        out["energy_rmse_offset_removed_meV_per_atom"] = float(
            np.sqrt(((de - de.mean()) ** 2).mean()) / nat * 1e3
        )
    return out


def _dft_epsilon(frames: list, eps_key: str) -> dict | None:
    if not frames or eps_key not in frames[0].info:
        return None
    iso = np.array([np.trace(np.reshape(a.info[eps_key], (3, 3))) / 3 for a in frames])
    return {
        "key": eps_key,
        "eps_inf_mean": float(iso.mean()),
        "eps_inf_std": float(iso.std()),
        "sqrt_eps_inf_mean": float(np.sqrt(iso).mean()),
    }


def calibrate(
    calc,
    fit_frames: list | None,
    test_frames: list | None,
    *,
    key: str = "BEC",
    scale: float | None = None,
    eps_key: str | None = None,
) -> tuple[dict, dict]:
    """Fit ``s`` on ``fit_frames`` (or take ``scale``) and score both sets.

    Returns ``(summary, data)``; ``data`` holds the arrays of each set with
    ``bec_pred`` already multiplied by ``s``.
    """
    if fit_frames is None and scale is None:
        raise ValueError("give frames to fit on, or a --scale to apply")
    summary, data = {}, {}
    if fit_frames is not None:
        fit = evaluate_frames(calc, fit_frames, key)
        s = fit_scale(fit["z_raw"], fit["z_ref"]) if scale is None else float(scale)
        data["fit"] = fit
    else:
        s = float(scale)
    summary["scale"] = s
    summary["sqrt_eps_inf"] = abs(s)
    summary["eps_inf"] = s * s
    summary["charge_sign"] = 1.0 if s >= 0 else -1.0
    summary["scale_source"] = "fit" if scale is None else "given"
    for name, frames in (("fit", fit_frames), ("test", test_frames)):
        if frames is None:
            continue
        d = data[name] if name in data else evaluate_frames(calc, frames, key)
        data[name] = d
        d["bec_pred"], d["bec_ref"] = s * d["z_raw"], d["z_ref"]
        block = {
            "n_frames": int(d["z_raw"].shape[0]),
            "n_atoms": int(d["z_raw"].shape[1]),
            "born": born_statistics(d["bec_pred"], d["bec_ref"], d["symbols"]),
            "mean_abs_charge_e": float(np.abs(d["charges"]).mean()),
        }
        block.update(_force_energy_statistics(d))
        if eps_key is not None:
            eps = _dft_epsilon(frames, eps_key)
            if eps is not None:
                block["dft_epsilon"] = eps
        summary[name] = block
    return summary, data


def plot_parity(data: dict, prefix: str) -> list[str]:
    """Diagonal and off-diagonal parity plots of one set, one file each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#d1332e", "#8a8a8a", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    symbols = data["symbols"]
    elements = list(dict.fromkeys(symbols))
    eye = np.eye(3, dtype=bool)
    pred, ref = data["bec_pred"], data["bec_ref"]
    lo = min(pred.min(), ref.min())
    hi = max(pred.max(), ref.max())
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    written = []
    for tag, mask, label in (
        ("diag", eye, r"$Z^*_{\alpha\alpha}$ [e]"),
        ("offdiag", ~eye, r"$Z^*_{\alpha\beta}$ [e]"),
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([lo, hi], [lo, hi], "--", lw=1.2, color="0.3")
        for k, el in enumerate(elements):
            sel = symbols == el
            ax.scatter(
                pred[:, sel][..., mask].ravel(),
                ref[:, sel][..., mask].ravel(),
                s=6,
                alpha=0.3,
                lw=0,
                color=palette[k % len(palette)],
                label=el,
                rasterized=True,
            )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("model " + label)
        ax.set_ylabel("DFT " + label)
        ax.legend(frameon=False, markerscale=3)
        fig.tight_layout()
        out = f"{prefix}_{tag}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        written.append(out)
    return written


def _dump(data: dict, out_dir: pathlib.Path) -> None:
    for name, d in data.items():
        target = out_dir / name
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "symbols.npy", d["symbols"])
        for k in (
            "bec_pred",
            "bec_ref",
            "forces_pred",
            "forces_ref",
            "energy_pred",
            "energy_ref",
            "charges",
        ):
            if k in d:
                np.save(target / f"{k}.npy", d[k])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model", help="dp_lr checkpoint (.pt)")
    ap.add_argument("--fit", help="frames with DFPT labels to fit the scale on")
    ap.add_argument("--test", help="held-out frames scored at the fitted scale")
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="skip the fit and apply this s (sign included)",
    )
    ap.add_argument(
        "--key",
        default="BEC",
        help="per-atom array holding the 9 "
        "DFPT tensor components [polarization, displacement]",
    )
    ap.add_argument(
        "--eps-key",
        default=None,
        help="atoms.info key of the DFPT dielectric tensor, reported "
        "next to the fit (e.g. 'epsilon')",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--compress",
        nargs="?",
        const=0.01,
        type=float,
        default=False,
        help="evaluate the compressed model (table stride, default 0.01)",
    )
    ap.add_argument("--out", default="calibrate_born.json")
    ap.add_argument("--plot", default=None, help="prefix for the parity figures")
    ap.add_argument("--dump-dir", default=None, help="write the arrays here")
    a = ap.parse_args(argv)
    if a.fit is None and a.test is None:
        ap.error("give --fit and/or --test")

    from deepmd_lr.ase import LongRangeCalculator

    calc = LongRangeCalculator(
        a.model, device=a.device, efield=[0.0, 0.0, 0.0], compress=a.compress
    )
    fit_frames = read_frames(a.fit, a.key) if a.fit else None
    test_frames = read_frames(a.test, a.key) if a.test else None
    summary, data = calibrate(
        calc, fit_frames, test_frames, key=a.key, scale=a.scale, eps_key=a.eps_key
    )
    summary["model"] = str(a.model)
    summary["compress"] = a.compress
    summary["key"] = a.key
    pathlib.Path(a.out).write_text(json.dumps(summary, indent=2) + "\n")
    if a.plot:
        shown = data["test" if "test" in data else "fit"]
        summary["figures"] = plot_parity(shown, a.plot)
    if a.dump_dir:
        _dump(data, pathlib.Path(a.dump_dir))
    print(json.dumps(summary, indent=2))
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
