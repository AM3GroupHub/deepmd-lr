# LiFSI / EC-DMC interface — training results

**Headline: at this budget the long-range term did not help.** The apparent
improvement over the baseline is entirely explained by the two-stage schedule.
Diagnosis and what would change it are below.

## Setup

300 frames sampled from the UMA trajectory (240 train / 60 validation), 6520
atoms each, padded to 28.08 × 28.08 × 124.38 Å with 20 Å of vacuum and trained
with the Yeh–Berkowitz slab correction. See `TRAINING_LiFSI.md`.

Three runs, all ending at the same 1900-step budget:

| run | schedule | long-range |
|---|---|---|
| baseline | 1900 steps, lr 1e-3 → 1e-5 | no |
| control | 1500 warm-up + 400 fine-tune, lr 5e-4 → 1e-5 | no |
| joint | 1500 warm-up + 400 fine-tune, same lr, grafted | yes |

## Held-out results (60 frames)

| model | E RMSE (meV/atom) | F RMSE (eV/Å) | F MAE |
|---|---:|---:|---:|
| baseline | 10.105 | 0.3770 | 0.2621 |
| control | 10.395 | **0.3371** | 0.2298 |
| joint | 9.977 | 0.3398 | 0.2320 |

Joint beats the *baseline* by 1.11× on force — but the **control matches it
(0.3371 vs 0.3398, control marginally better)**. The gain came from splitting
the budget into a warm-up plus a low-LR fine-tune, not from the physics. This
is exactly why the control exists: against the baseline alone the result would
have looked like a real 11% win.

## Why the long-range term did nothing

The long-range energy is **+0.34 ± 0.04 eV** against a total of −34065 eV —
0.001%. The charges explain it:

| | learned | physically expected |
|---|---:|---:|
| rms charge | 0.0038 e | ~0.3–0.8 e for the ions |
| max abs charge | 0.029 e | ~0.8 e (Li⁺, FSI⁻) |

The charges are **~200× too small**, so the electrostatic energy — quadratic
in charge — is ~4×10⁴ times too small to matter.

Tracking the charge head through the fine-tune shows what happened:

| checkpoint | max abs q | rms q | E_long-range |
|---|---:|---:|---:|
| grafted init | 1.33e-02 | 2.27e-03 | 4.997 eV |
| after 400 joint steps | 2.86e-02 | 3.77e-03 | 0.310 eV |

The charges grew ~2×, but the long-range **energy fell by 16×**. The optimizer
did not grow the term into something useful; it rearranged the charges to
*suppress* it. With a term that small the gradient pushing charges upward is
itself tiny (it scales with q), so escaping the small-charge region is slow —
the same stationary-point problem that makes an exactly zeroed head untrainable,
in a milder form.

**400 joint steps from a 0.01-scaled head is far too little.** This is an
under-training result, not evidence that the physics is absent.

## What would change it

1. **Far more joint steps** — thousands, not 400. At 4.5 s/step that is hours;
   the short-range warm-up is 5× cheaper, so the split is worth keeping.
2. **A chemically informed charge initialization.** Seeding the charge head's
   per-element output bias near formal charges (Li ≈ +0.8, F/O/N negative)
   would start the term at a useful magnitude instead of asking the optimizer
   to find it from ~0. This is the highest-value change and is cheap to
   implement — `--charge-init-scale` currently only rescales a random head.
3. **Check the headroom first.** For the synthetic benchmark 17.7% of the force
   signal lay beyond the 6 Å cutoff. The equivalent number is not known here,
   because it needs reference charges. If it is small, no amount of training
   will make the term pay.

## One qualitative signal worth noting

Even at these tiny magnitudes the charge profile has the right topology for an
electrode–electrolyte interface: net **+0.327 e** on the metal side (z < 30 Å)
against **−0.321 e** across the electrolyte, i.e. charge separation at the
interface with the metal positive. That is the sign an electric double layer
would have. At 0.3 e total it is far too weak to claim as a physical result —
it is reported as a direction, not a finding.

A reminder from `TRAINING_LiFSI.md`: charges on *metallic* Li are a fitting
device, not observables. A metal screens fields and delocalises charge, so the
electrolyte-side numbers are the ones with a physical reading.
