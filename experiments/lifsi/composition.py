"""Is the Li a stoichiometric counter-ion, or an electrode slab?"""
import numpy as np
from ase.io import iread

atoms = next(iread("LiFSI_EC_DMC_1M_T300.xyz", index=":"))
z = atoms.positions[:, 2]
sym = np.array(atoms.get_chemical_symbols())

print(f"FSI units (from N/S/F): N={np.sum(sym=='N')} "
      f"S={np.sum(sym=='S')//2} F={np.sum(sym=='F')//2}")
print(f"Li atoms             : {np.sum(sym=='Li')}")
print(f"  -> charge balance wants {np.sum(sym=='N')} Li for the salt\n")

print("z distribution by element (A):")
for e in ("Li", "N", "S", "F", "C", "O", "H"):
    m = sym == e
    if m.sum():
        print(f"  {e:2s} n={m.sum():5d}  z {z[m].min():7.2f} .. {z[m].max():7.2f}  "
              f"median {np.median(z[m]):7.2f}")

li = z[sym == "Li"]
print("\nLi z-histogram (counts per 10 A):")
hist, edges = np.histogram(li, bins=np.arange(0, 111, 10))
for c, lo in zip(hist, edges[:-1], strict=True):
    print(f"  z {lo:5.0f}-{lo+10:3.0f}: {c:5d} {'#' * min(int(c / 15), 60)}")

# nearest-neighbour spacing among the dense Li population -> metal or solvated?
dense = atoms.positions[(sym == "Li") & (z < 30)]
if len(dense) > 1:
    d = np.linalg.norm(dense[:, None, :] - dense[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    print(f"\nLi (z<30) nearest-neighbour distance: median "
          f"{np.median(d.min(axis=1)):.2f} A  (bcc Li metal is 3.04 A)")
