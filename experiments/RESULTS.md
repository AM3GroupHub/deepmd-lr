# Phase 5 — does the long-range term buy anything?

## Data situation

No periodic dataset was available on this machine. Everything under
`/home/admin/storage/dataset` — `omol25`, `opoly26`, and all three
`electrolyte` sets — is **non-periodic** (`pbc = [F F F]`): molecular
clusters in the OMol25 style. PME requires 3D periodicity, so none of it can
drive this comparison. The only periodic real data is DeePMD-kit's water
example (240 training frames, 192 atoms).

So the primary experiment is a controlled synthetic benchmark, with the water
set as a secondary real-data check.

> **Corrected after the first run.** The long-range term originally computed
> the *full* Ewald sum (real + reciprocal + self). It should only supply the
> reciprocal half: the screened real-space part decays as `erfc(alpha r)/r`
> and is already inside the descriptor cutoff, so the short-range network
> represents it directly — including it here asks the network to learn to
> subtract something it can already describe. Every number below is from the
> corrected, reciprocal-only term. The correction improved the result
> substantially (force gain 2.6x → 4.9x) and removed an energy regression on
> real water.

## Benchmark design

A periodic molten-salt-like system, 32 ions (±1) in a 14 Å cubic cell, with a
ground truth that is known exactly:

```
E = E_repulsion(Born-Mayer, cutoff 5 Å) + E_Coulomb(q = ±1, exact Ewald)
```

The repulsive part lies entirely inside the 6 Å descriptor cutoff, so a plain
DPA4C model can represent it exactly. The Coulomb part cannot be: **17.7% of
the total force signal comes from Coulomb interaction beyond 6 Å**, measured
by differencing exact Ewald against Coulomb truncated at the cutoff. That
0.37 eV/Å is the floor no strictly local 6 Å model can beat.

The reference labels use nvalchemi's *exact* `ewald_summation`, not the PME
path the model uses, so label and model do not share an implementation.

800 training frames, 200 held out, 5000 steps, identical settings for both
models apart from the `long_range` block.

## Result

Held-out validation, 200 frames:

| model    | E RMSE (meV/atom) | F RMSE (eV/Å) |
|----------|------------------:|--------------:|
| baseline |            187.31 |        0.4075 |
| joint    |             34.04 |        0.0826 |
|          |       **5.5×**    |    **4.9×**   |

The baseline lands at 0.41 eV/Å, essentially exactly the 0.37 eV/Å floor
predicted from the beyond-cutoff signal — it has hit the wall the analysis
said it would. The joint model reaches 0.083 eV/Å, **five times below that
floor**, which is only possible by representing interaction beyond the
cutoff.

## The control: is it just extra parameters?

The joint model has a second network, so the fair objection is that it wins
on capacity rather than physics. Re-running the whole comparison on an
identical dataset with the Coulomb term **removed**:

| dataset             | baseline F RMSE | joint F RMSE | ratio |
|---------------------|----------------:|-------------:|------:|
| with Coulomb        |          0.4094 |       0.0832 | 4.9×  |
| repulsion only      |          0.0243 |       0.0257 | 0.94× |

With no long-range physics present, the long-range machinery gives **no
benefit at all** (a hair worse, within noise). The gain is specifically from
capturing long-range physics, not from capacity. Both models reach 0.024 eV/Å
on the repulsion-only problem — 17× better than the baseline manages when
Coulomb is present — confirming the local part is easy and fully learnable.

## Learned charges, and a symmetry worth knowing about

The charge head is never shown a charge. It recovers:

| type | true | learned mean |    std |
|------|-----:|-------------:|-------:|
| Na   | +1.00 |      +0.9604 | 0.0137 |
| Cl   | −1.00 |      −0.9604 | 0.0139 |

Magnitude recovered to within 4%, relative assignment correct, small spread
across configurations.

**The global sign is a genuine degeneracy, and this run merely happened to
land on the physical branch.** The Coulomb energy is bilinear in charge, so
every product `q_i q_j` is unchanged under `q → −q`; for a neutral cell the
two branches are *exactly* degenerate. Verified numerically to 1.4e-14 and
pinned by `TestChargeSignDegeneracy`. An earlier run of this same benchmark
converged to the inverted branch (−0.9364 / +0.9364), which is equally
optimal. No amount of energy or force data can break the tie.

Practical consequence: a charge head trained on energies and forces alone
gives magnitudes and relative signs, with the overall sign arbitrary. Anyone
comparing learned charges to a population analysis (Bader, Mulliken, RESP)
must break the symmetry themselves — with a charge label, a sign constraint,
or a fixed reference atom.

The residual 6.4% magnitude deficit is a second, softer degeneracy: the split
between short-range and long-range is not unique, because DPA4C can absorb
Coulomb interaction that falls inside its cutoff.

## Real water (secondary)

DeePMD-kit's water example, 240 training frames, 4000 steps, mean over the
last quarter of the learning curve:

| model    | E RMSE (meV/atom) | F RMSE (eV/Å) |
|----------|------------------:|--------------:|
| baseline |             1.341 |        0.0962 |
| joint    |             1.038 |        0.0895 |

A 1.29× energy and 1.07× force improvement. **This is suggestive, not
demonstrated.** 240 frames of neutral water is too small to resolve an effect
this size, and neutral water at a 6 Å cutoff has modest long-range force
content to begin with. What it does establish is that the long-range term
neither breaks nor destabilizes training on real DFT data. (With the earlier
full-Ewald term this comparison showed an energy *regression*, 0.79×, which
the reciprocal-only correction removed.)

## Reproducing

```bash
cd experiments/longrange_benchmark
python generate_data.py                      # main dataset
python generate_data.py --out data_norc --no-coulomb   # ablation
python headroom.py                           # the 17.7% figure
for t in baseline joint; do
  mkdir -p run_$t && cd run_$t && ln -sfn ../data data
  dp --pt-expt train ../input_$t.json && cd ..
done
python analyze.py                            # table + learned charges
```
