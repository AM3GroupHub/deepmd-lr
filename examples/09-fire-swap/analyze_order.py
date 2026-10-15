"""B-site chemical order of a FIRE-Swap trajectory, frame by frame.

Maps Mg/Nb onto the ideal n x n x n simple-cubic B lattice (n from the number
of B atoms) and reports the nearest-neighbour pair counts (Mg-Mg, Nb-Nb,
Nb-Mg), the Mg on the beta_II sublattice and the order parameter O(s), the Nb
clusters (connected Nb with at most one Mg nearest neighbour), and the
fraction of a replicated start's periodicity that has been broken.

    python analyze_order.py run/fireswap.extxyz [order.dat] [--period P]

`--period` is the replica period in B sites: 6 for any `--repeat` of the
6x6x6 `pmn1080.xyz`, the default.
"""

import argparse
from collections import deque

import numpy as np
from ase.io import iread

AXES = (0, 1, 2)


def b_site_lattice(atoms, n):
    """An (n, n, n) array of 'Mg'/'Nb' on the B lattice, and each site's atom index."""
    sym = np.array(atoms.get_chemical_symbols())
    b = np.where((sym == "Mg") | (sym == "Nb"))[0]
    if len(b) != n**3:
        raise ValueError(f"{len(b)} B atoms, expected {n**3}")
    frac = atoms.cell.scaled_positions(atoms.positions[b]) % 1.0
    ijk = np.floor(frac * n).astype(int) % n
    lat = np.full((n, n, n), "", dtype="<U2")
    idx = np.full((n, n, n), -1, dtype=int)
    lat[tuple(ijk.T)] = sym[b]
    idx[tuple(ijk.T)] = b
    if (idx < 0).any():
        raise ValueError("B atoms do not map one-to-one onto the ideal lattice")
    return lat, idx


def parity(n):
    i, j, k = np.indices((n, n, n))
    return (i + j + k) % 2


def order_stats(lat, beta2_parity=None, period=None):
    """Order statistics of one B lattice. beta2_parity=None picks the sublattice
    holding fewer Mg; periodicity_broken is reported only for a period < n."""
    n = lat.shape[0]
    mg = lat == "Mg"
    nb = ~mg
    par = parity(n)
    if beta2_parity is None:
        beta2_parity = int(mg[par == 1].sum() < mg[par == 0].sum())

    # each unordered NN pair counted once: +1 neighbour along each axis
    mm = sum(int((mg & np.roll(mg, -1, a)).sum()) for a in AXES)
    nn = sum(int((nb & np.roll(nb, -1, a)).sum()) for a in AXES)
    npairs = 3 * n**3
    s = np.where(nb, 1, -1)
    order = sum(int((s * np.roll(s, -1, a)).sum()) for a in AXES) / npairs

    # Nb clusters (the paper's definition): Nb with <= 1 Mg among its 6 neighbours
    n_mg_nn = sum(np.roll(mg, sh, a).astype(int) for a in AXES for sh in (1, -1))
    member = nb & (n_mg_nn <= 1)
    seen = np.zeros_like(member)
    sizes = []
    for start in zip(*np.where(member), strict=True):
        if seen[start]:
            continue
        seen[start] = True
        q, size = deque([start]), 0
        while q:
            u = q.popleft()
            size += 1
            for a in AXES:
                for sh in (1, -1):
                    w = list(u)
                    w[a] = (w[a] + sh) % n
                    w = tuple(w)
                    if member[w] and not seen[w]:
                        seen[w] = True
                        q.append(w)
        sizes.append(size)

    out = {
        "n_sites": n**3,
        "MgMg": mm,
        "NbNb": nn,
        "NbMg": npairs - mm - nn,
        "beta2_parity": beta2_parity,
        "beta2_Mg": int(mg[par == beta2_parity].sum()),
        "O": round(order, 6),
        "Nb_clustered": int(member.sum()),
        "n_clusters": len(sizes),
        "largest_cluster": max(sizes) if sizes else 0,
    }
    if period is not None and period < n:
        # sites differing from their image one period away, averaged over the axes
        out["periodicity_broken"] = round(
            float(np.mean([(lat != np.roll(lat, period, a)).mean() for a in AXES])), 6
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory")
    ap.add_argument("out", nargs="?", default="order.dat")
    ap.add_argument("--period", type=int, default=6)
    a = ap.parse_args()
    out, period = a.out, a.period
    beta2 = None
    rows = []
    for f, atoms in enumerate(iread(a.trajectory)):
        n_b = sum(s in ("Mg", "Nb") for s in atoms.get_chemical_symbols())
        n = round(n_b ** (1 / 3))
        lat, _ = b_site_lattice(atoms, n)
        st = order_stats(lat, beta2, period)
        beta2 = st["beta2_parity"]  # fix beta_II from the first frame
        e = (
            atoms.calc.results.get("energy")
            if atoms.calc is not None
            else atoms.info.get("energy")
        )
        rows.append((f, np.nan if e is None else float(np.sum(e)), st))
    keys = [k for k in rows[0][2] if k not in ("n_sites", "beta2_parity")]
    with open(out, "w") as fh:
        fh.write("frame energy_eV " + " ".join(keys) + "\n")
        for f, e, st in rows:
            fh.write(f"{f} {e:.5f} " + " ".join(str(st[k]) for k in keys) + "\n")
    print(f"{len(rows)} frames -> {out} (beta_II parity {beta2})")


if __name__ == "__main__":
    main()
