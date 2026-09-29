"""The conductance from a `nemd.py` run: bath powers, the T(z) profile, and
G = J / dT.

Production is everything after the steady stage (`time[ps] > steady_ps`
in thermo.log). The power each bath delivers is the slope of its cumulative
heat, P_hot from Q_hot(t) and P_cold from -Q_cold(t); in steady state the
two agree, and their mismatch is printed as the check. The flux through the
single interface is J = P / A. The profile is the sample-weighted mean of
profile.csv over production; a line is fitted to T(z) on each side of the
interface, over the free region with `--margin` angstrom left out on either
side of the plane, and the two lines extrapolated to the plane give dT. The
same slopes give each side's conductivity, kappa = J / |dT/dz|, which is
finite-size limited at these lengths and only a consistency check.

Units: Q in eV and t in ps give P in eV/ps; A in A^2 gives J in eV/(ps A^2),
and 1 eV ps^-1 A^-2 = 1.602e13 W m^-2, so G comes out in W m^-2 K^-1
(printed in MW m^-2 K^-1); 1 eV ps^-1 A^-1 K^-1 = 1.602e3 W m^-1 K^-1.

    python analyze.py --out run1
"""

import argparse
import json
import pathlib

import numpy as np
from ase import units

EV_PS_A2 = units._e / (1e-12 * 1e-20)        # W/m^2 per eV ps^-1 A^-2
EV_PS_A_K = units._e / (1e-12 * 1e-10)       # W/(m K) per eV ps^-1 A^-1 K^-1


def read_log(path):
    """The `#`-headed whitespace table the MD classes write -> {column: array}."""
    with open(path) as fh:
        names = fh.readline().lstrip("#").split()
    data = np.loadtxt(path, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(names)}


def fit_power(t, q, nblocks):
    """Slope of q(t) in eV/ps, and its standard error from `nblocks` block
    slopes (each block is a fit of its own)."""
    slope = np.polyfit(t, q, 1)[0]
    edges = np.linspace(t.min(), t.max(), nblocks + 1)
    blocks = [np.polyfit(t[m], q[m], 1)[0]
              for lo, hi in zip(edges, edges[1:], strict=False)
              if (m := (t >= lo) & (t <= hi)).sum() >= 3]
    err = np.std(blocks, ddof=1) / np.sqrt(len(blocks)) if len(blocks) > 1 else np.nan
    return slope, err


def mean_profile(prof, t_min):
    """Sample-weighted mean T per bin over the rows after t_min."""
    rows = prof[prof["time_ps"] > t_min]
    z = np.unique(rows["z"])
    t, n = np.zeros(len(z)), np.zeros(len(z))
    for i, zc in enumerate(z):
        r = rows[rows["z"] == zc]
        ok = r["n_samples"] > 0
        n[i] = r["n_samples"][ok].sum()
        t[i] = (r["T"][ok] * r["n_samples"][ok]).sum() / n[i] if n[i] else np.nan
    return z, t, n


def fit_side(z, t, lo, hi):
    """T = a z + b over the bins with lo < z < hi; returns a, b and their
    standard errors (nan with fewer than three bins)."""
    m = (z > lo) & (z < hi) & np.isfinite(t)
    if m.sum() < 2:
        raise ValueError(f"only {m.sum()} bins between z = {lo:.1f} and {hi:.1f} A; "
                         "use a finer --bin, a smaller --margin, or thicker free "
                         "layers")
    if m.sum() == 2:
        return (*np.polyfit(z[m], t[m], 1), np.nan, np.nan)
    p, cov = np.polyfit(z[m], t[m], 1, cov=True)
    return p[0], p[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the run directory of nemd.py")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="A left out of the fits on each side of the interface")
    ap.add_argument("--nblocks", type=int, default=5, help="blocks for the errors")
    ap.add_argument("--skip-ps", type=float, default=0.0,
                    help="production time to drop before the fits")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    with open(out / "run_meta.json") as fh:
        meta = json.load(fh)
    log = read_log(out / "thermo.log")
    prof = np.genfromtxt(out / "profile.csv", delimiter=",", names=True)

    t_min = meta["steady_ps"] + a.skip_ps
    prod = log["time[ps]"] > t_min
    if prod.sum() < 4:
        raise SystemExit(f"only {prod.sum()} production rows after {t_min} ps")
    t = log["time[ps]"][prod]
    p_hot, e_hot = fit_power(t, log["Q_hot[eV]"][prod], a.nblocks)
    p_cold, e_cold = fit_power(t, -log["Q_cold[eV]"][prod], a.nblocks)
    power = 0.5 * (p_hot + p_cold)
    power_err = 0.5 * np.hypot(e_hot, e_cold)
    flux = power / meta["area"]                              # eV ps^-1 A^-2

    z, temp, n = mean_profile(prof, t_min)
    zi, margin = meta["interface_z"], a.margin
    hot_top, cold_bottom = meta["z_range"]["hot"][1], meta["z_range"]["cold"][0]
    a_l, b_l, ea_l, eb_l = fit_side(z, temp, hot_top, zi - margin)
    a_r, b_r, ea_r, eb_r = fit_side(z, temp, zi + margin, cold_bottom)
    t_left, t_right = a_l * zi + b_l, a_r * zi + b_r
    d_t = t_left - t_right
    d_t_err = np.hypot(np.hypot(ea_l * zi, eb_l), np.hypot(ea_r * zi, eb_r))
    g = flux / d_t * EV_PS_A2                                 # W m^-2 K^-1
    g_err = g * np.hypot(power_err / power, d_t_err / d_t)
    kappa = {"left": flux / abs(a_l) * EV_PS_A_K, "right": flux / abs(a_r) * EV_PS_A_K}

    print(f"production {t.min():.1f} - {t.max():.1f} ps, {prod.sum()} rows; "
          f"area {meta['area']:.2f} A^2; interface z {zi:.2f} A")
    print(f"  P_hot  {p_hot:+.4f} +- {e_hot:.4f} eV/ps")
    print(f"  P_cold {p_cold:+.4f} +- {e_cold:.4f} eV/ps   "
          f"(mismatch {abs(p_hot - p_cold) / abs(power) * 100:.1f} % of the mean: "
          "the steady-state check)")
    print(f"  J      {flux:.3e} eV/(ps A^2) = {flux * EV_PS_A2 / 1e9:.3f} GW/m^2")
    print(f"  T(z) left  = {a_l:+.3f} z + {b_l:.1f}  -> {t_left:.1f} K at the plane")
    print(f"  T(z) right = {a_r:+.3f} z + {b_r:.1f}  -> {t_right:.1f} K at the plane")
    print(f"  dT     {d_t:.2f} +- {d_t_err:.2f} K")
    print(f"  G      {g / 1e6:.1f} +- {g_err / 1e6:.1f} MW m^-2 K^-1")
    print(f"  kappa  left {kappa['left']:.2f}, right {kappa['right']:.2f} W m^-1 K^-1 "
          "(finite-size limited; a consistency check only)")

    result = {
        "production_ps": [float(t.min()), float(t.max())],
        "P_hot_eV_ps": [p_hot, e_hot], "P_cold_eV_ps": [p_cold, e_cold],
        "J_eV_ps_A2": flux, "J_W_m2": flux * EV_PS_A2,
        "T_left_K": t_left, "T_right_K": t_right, "dT_K": [d_t, d_t_err],
        "G_W_m2K": [g, g_err], "kappa_W_mK": kappa,
        "fit_left": [a_l, b_l], "fit_right": [a_r, b_r], "margin_A": margin,
    }
    with open(out / "analysis.json", "w") as fh:
        json.dump(result, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(t, log["Q_hot[eV]"][prod], label="Q_hot")
    ax1.plot(t, -log["Q_cold[eV]"][prod], label="-Q_cold")
    tt = np.array([t.min(), t.max()])
    for p_, q_ in ((p_hot, log["Q_hot[eV]"][prod]), (p_cold, -log["Q_cold[eV]"][prod])):
        ax1.plot(tt, q_.mean() + p_ * (tt - t.mean()), "k--", lw=0.8)
    ax1.set_xlabel("t [ps]")
    ax1.set_ylabel("heat injected [eV]")
    ax1.set_title(f"P = {power:.4f} eV/ps, J = {flux * EV_PS_A2 / 1e9:.2f} GW/m$^2$")
    ax1.legend()

    ax2.plot(z, temp, "o", ms=4, label="T(z), production mean")
    for (a_, b_), (lo, hi) in (((a_l, b_l), (hot_top, zi)),
                               ((a_r, b_r), (zi, cold_bottom))):
        zz = np.array([lo, hi])
        ax2.plot(zz, a_ * zz + b_, "k-", lw=1.2)
    for lo, hi in (meta["z_range"]["hot"], meta["z_range"]["cold"]):
        ax2.axvspan(lo, hi, color="0.9")
    ax2.axvline(zi, color="0.5", ls=":")
    ax2.annotate("", (zi, t_left), (zi, t_right),
                 arrowprops={"arrowstyle": "<->", "color": "C3"})
    ax2.text(zi, 0.5 * (t_left + t_right), f"  dT = {d_t:.1f} K", color="C3")
    ax2.set_xlabel("z [A]")
    ax2.set_ylabel("T [K]")
    ax2.set_title(f"G = {g / 1e6:.0f} +- {g_err / 1e6:.0f} MW m$^{{-2}}$ K$^{{-1}}$")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "nemd.png", dpi=130)
    print(f"  wrote {out / 'analysis.json'} and {out / 'nemd.png'}")


if __name__ == "__main__":
    main()
