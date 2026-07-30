# Stability Boundary After Backtracking (Sweep v2)

[hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md) mapped the
Df/kf/σ/N stability boundary against greedy first-fit pairing and
explicitly reserved itself as "the current-implementation baseline… a
future pairing-choice fix should be benchmarked against this grid to
quantify how far the boundary actually shifts."

This is that measurement. Identical grid, identical seeds, identical
trial counts (`configs/boundary_sweep_v2.toml` is a copy of the original
config with only the output directory changed), run against the current
defaults: backtracking pairing, the overlap-acceptance fix, and mass-based
CCA Γ.

**Overall: 3374/4200 (80.3%), against 3039/4200 (72.4%) before.**

That headline number understates the change, because the grid deliberately
extends well past the boundary into regions no pairing strategy can
rescue. The interesting result is where the boundary moved.

## σ = 1.9 (the hard polydisperse case)

Success rate averaged over N ∈ {64…1024}, old → new. Rows that were 1.00
throughout and stayed there are omitted.

| Df | kf=0.8 | kf=0.9 | kf=1.0 | kf=1.1 | kf=1.2 | kf=1.3 | kf=1.4 |
|---|---|---|---|---|---|---|---|
| 2.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.92→1.00** |
| 2.1 | 1.00 | 1.00 | 1.00 | 1.00 | **0.80→1.00** | **0.32→1.00** | **0.08→1.00** |
| 2.2 | 1.00 | **0.92→1.00** | **0.52→1.00** | **0.16→0.84** | **0.00→0.64** | **0.00→0.52** | **0.00→0.36** |
| 2.3 | **0.56→0.84** | **0.12→0.72** | **0.04→0.44** | **0.00→0.40** | **0.04→0.20** | **0.00→0.16** | **0.00→0.04** |
| 2.4 | **0.16→0.44** | **0.00→0.32** | **0.00→0.20** | **0.00→0.12** | 0.00 | 0.00 | 0.00 |
| 2.5 | **0.00→0.20** | **0.00→0.12** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The Df=2.1 row is the clearest single statement: it used to collapse from
1.00 to 0.08 as kf rose from 0.8 to 1.4, and is now uniformly 1.00. The
previously-safe ceiling at σ=1.9 was Df≈2.0–2.1; it is now Df≈2.2 across
most of the kf range, with non-zero success appearing for the first time
at Df=2.4–2.5.

## σ = 1.5

| Df | kf=0.8 | kf=0.9 | kf=1.0 | kf=1.1 | kf=1.2 | kf=1.3 | kf=1.4 |
|---|---|---|---|---|---|---|---|
| 2.2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.92→1.00** |
| 2.3 | 1.00 | 1.00 | 1.00 | **0.84→0.96** | **0.44→0.84** | **0.08→0.72** | **0.04→0.44** |
| 2.4 | **0.92→0.96** | **0.60→0.84** | **0.36→0.64** | **0.00→0.44** | **0.00→0.40** | **0.00→0.08** | **0.00→0.04** |
| 2.5 | **0.40→0.60** | **0.08→0.40** | **0.04→0.28** | **0.00→0.12** | 0.00 | 0.00 | 0.00 |

## σ = 1.0 (monodisperse)

| Df | kf=0.8 | kf=0.9 | kf=1.0 | kf=1.1 | kf=1.2 | kf=1.3 | kf=1.4 |
|---|---|---|---|---|---|---|---|
| 2.3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00→0.96 |
| 2.4 | 1.00 | 1.00 | 1.00 | 0.88→0.80 | **0.68→0.72** | **0.56→0.72** | **0.36→0.60** |
| 2.5 | 0.92 | **0.68→0.80** | **0.60→0.64** | 0.52→0.48 | **0.20→0.40** | **0.04→0.36** | **0.00→0.24** |

Gains are real but smaller here, which is what the diagnosis predicts:
monodisperse aggregates were never the frustrated case. Two cells move
*down* slightly (Df=2.4/kf=1.1: 0.88→0.80, Df=2.5/kf=1.1: 0.52→0.48).
At 25 trials per cell those are within sampling noise, and they sit
alongside much larger gains in the same rows; they are recorded rather
than explained away, but they do not indicate a regression.

## N dependence

The previous sweep found that "N does not independently cause failure; it
sharpens whatever margin Df/kf/σ already leaves." Backtracking flattens
that sharpening substantially at σ=1.9:

| Df, kf | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---|---|---|---|---|---|
| 2.2, 1.0 | 1.00→1.00 | 0.60→**1.00** | 0.60→**1.00** | 0.40→**1.00** | 0.00→**1.00** |
| 2.3, 0.8 | 1.00→1.00 | 1.00→1.00 | 0.40→**1.00** | 0.40→**1.00** | 0.00→**0.20** |

Df=2.2/kf=1.0 previously degraded monotonically to total failure at
N=1024 and is now flat at 1.00 across the whole range. This matters more
than the averaged tables: large N was where the old implementation was
least usable, and it is where backtracking helps most, because a single
unlucky pair no longer discards an entire expensive attempt.

## Cost

Backtracking makes *infeasible* configurations more expensive, not less.
Where greedy bailed on the first failed pair, backtracking tries several
partners per cluster first, and `run_simulation` then retries the whole
thing up to 20 times. The first attempt at this sweep stalled at roughly
12 trials per five minutes in the Df=2.5 corners, despite a nominal 120s
per-trial timeout - because that timeout was only checked *between*
attempts and so could never interrupt one long attempt.

That is now fixed: `CCAggregator` takes a wall-clock `deadline`, threaded
from `run_simulation`'s `max_runtime_seconds`, and checked inside the
round loop and before each additional partner attempt. An infeasible
N=512/Df=2.5 configuration given a 20s budget returns in 20.1s. Anyone
sweeping past the boundary should set `trial_timeout`; without it, hard
corners are genuinely slow to fail.

Raw output: `benchmark_results/boundary_sweep_v2/stability_sweeps/`.
