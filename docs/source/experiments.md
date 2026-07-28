# CCA Sticking Method Comparison

This page retrospectively compares the CCA (cluster-cluster aggregation)
sticking strategies evaluated between early and mid-2026. The production
default is vanilla Fibonacci-spiral sticking with an incremental
active-set overlap check. Every alternative evaluated below, with one
exception (densification), failed to outperform it. Non-winning
implementations are archived under `pyfracval/experimental/` rather than
removed, since a different parameterization of the same idea may prove
useful later.

## Background: geometric frustration

In "hard" parameter regimes - high `Df`, low `kf`, wide polydispersity
(e.g. `Df=2.25, kf=0.95, rp_gstd=1.9`) - CCA sticking success rates
collapse to roughly 17-20%, independent of how the rotation search is
conducted. The underlying cause is geometric frustration: the fractal
scaling law forces two clusters to be placed at a contact distance
(`gamma_pc`) for which, at these parameter combinations, no overlap-free
relative orientation exists. No refinement of the search finds a solution
that does not exist at that fixed distance.

## Method comparison (hard regime: N=128, Df=2.25, kf=0.95, rp_gstd=1.9)

Source: `benchmark_results/profiles/method_comparison_hard_regime/`,
`benchmark_results/profiles/soft_quick_test/` (30 trials per method unless
noted; commit `d241350`).

| Method | Success rate | Median wall time | Verdict |
|---|---|---|---|
| Baseline (vanilla Fibonacci, single rotation mode) | 16.7% | 42.2 s | reference |
| Bounding-volume pair pre-filter | 16.7% | 41.0 s | no improvement |
| SSA (surface-accessibility) pair filter | 16.7% | 41.9 s | no improvement |
| Γ-expansion (gamma relaxation, max 3 attempts) | 16.7% | 41.4 s | no improvement |
| BV filter + Γ-expansion combined | 16.7% | 41.9 s | no improvement |
| FFT rigid-body docking (64³ grid) | 16.7% | 42.1 s | no improvement |
| FFT rigid-body docking (128³ grid) | 16.7% | 42.2 s | no improvement |
| Soft potential relaxation (paired baseline: 20.0%) | 20.0% | 35.6 s | no improvement over its own baseline (34.6 s) |
| Densification (generate at Df=1.8/kf=1.0, densify to target) | 100.0% | 2.1 s | qualitative improvement |
| Densification (generate at Df=2.0/kf=1.0, densify to target) | 100.0% | 2.6 s | qualitative improvement |

Every rigid-body enhancement to the sticking search - pair pre-filters, Γ
relaxation, combining them, and FFT-based rigid docking at two grid
resolutions - lands within noise of the vanilla baseline. Soft potential
relaxation is likewise statistically indistinguishable from its own paired
baseline. Densification is the only approach that changes the outcome
qualitatively: 100% success, roughly 20x faster than any rigid search
variant. It sidesteps the frustration problem by generating the aggregate
at an easier `Df`/`kf` and reshaping it afterward, rather than forcing
sticking at the hard target contact distance directly.

### Fractal accuracy of densified aggregates

Source: `benchmark_results/fractal_structure_validation.json`.

| Method | Success rate | Mean \|Rg error\| | Max \|Rg error\| |
|---|---|---|---|
| Baseline (rigid sticking, successes only) | 40.0% (12/30) | 1.67% | 3.52% |
| Densify (source Df=2.0) | 100.0% (30/30) | 0.42% | 0.93% |
| Densify (source Df=1.8) | 100.0% (30/30) | 1.04% | 1.84% |

Densified aggregates are both more likely to succeed and, on average, more
accurate against the theoretical radius-of-gyration scaling law than the
smaller set of successful rigid-body runs.

## Retry rotation modes: no measurable difference

Source: `benchmark_results/profiles/retry_mode_matrix_hard_v1/` (12 trials
per mode, hard regime, N=256 and N=512).

| Mode | N=256 success | N=256 median | N=512 success | N=512 median |
|---|---|---|---|---|
| `single` | 8.3% (1/12) | 57.6 s | 16.7% (2/12) | 115.8 s |
| `alternate` | 8.3% (1/12) | 56.4 s | 16.7% (2/12) | 115.8 s |
| `coarse_grid` | 8.3% (1/12) | 56.8 s | 16.7% (2/12) | 115.8 s |
| `coarse_to_fine` | 8.3% (1/12) | 58.3 s | 16.7% (2/12) | 114.2 s |

All four rotation-retry strategies produce identical success counts and
statistically indistinguishable timing at both sizes tested. Broadening
the rotation search does not help when no orientation is overlap-free at
the required contact distance.

## Candidate ordering policies: no measurable difference

Source: `benchmark_results/profiles/candidate_policy_probe_v1/` (8 trials
per policy, N=512).

| Policy | Success rate | Median wall time |
|---|---|---|
| `leaf_hybrid` | 12.5% (1/8) | 73.8 s |
| `leaf_score` | 12.5% (1/8) | 75.2 s |

Both policies produce the same outcome. This is a small probe (n=8 per
arm), but combined with the retry-mode result above it supports the
conclusion that *how* a contact pair is searched for matters far less than
*whether a valid contact pair exists at all* at the enforced `gamma_pc`.

## Implications for the codebase

- Production path (default, first-class code): vanilla Fibonacci sticking,
  single rotation mode, incremental active-set overlap checking, baseline
  candidate ordering. `pyfracval/cca/` (`pairing.py`, `candidates.py`,
  `sticking.py`, `fallbacks.py`, `aggregator.py`) is optimized for this
  path.
- Supported opt-in feature: densification (`pyfracval/densify.py`). It is
  the only mechanism in this codebase that reliably solves hard-regime
  generation, and does so faster than the "easy" regime's own rigid
  search.
- Archived, not removed (moved to `pyfracval/experimental/`, off the
  production path but reachable via the same config flags they always
  had - `cca/` retains a thin opt-in dispatch to each):
  - Extra retry rotation modes (`alternate`, `dual_jitter`, `coarse_grid`,
    `coarse_to_fine`) - `experimental/retry_modes.py`
  - Pair feasibility pre-filters (bounding-volume, SSA) -
    `experimental/pair_prefilters.py`
  - Γ-expansion - `experimental/gamma_expansion.py`
  - FFT rigid-body docking - `experimental/fft_docking.py`
  - Soft potential relaxation - `experimental/soft_relaxation.py`
  - Non-baseline candidate scoring policies (`leaf_soft`/`leaf_score`/
    `leaf_hybrid`) - `experimental/candidate_policies.py`
  - Soft-accept and rigid-repair (config flags `cca_soft_accept_*`/
    `cca_repair_*`): confirmed dead code (no reader anywhere) and removed
    outright rather than archived, since no implementation remained to
    preserve.

None of these were bugs. Each was a reasonable hypothesis, tested
rigorously, that did not move the outcome on this problem. They are
retained rather than deleted because a different parameterization of the
same idea - e.g. Γ-expansion with a substantially larger expansion budget,
or FFT docking at higher rotation sampling density - may behave
differently and could be revisited.

## Limitations and future work

Retry-mode and candidate-policy results above use small trial counts
(8-12 per arm) at a single hard-regime point. Distinguishing "no effect"
from "insufficient statistical power" would require a larger sweep across
multiple `(Df, kf, rp_gstd)` points; see
`configs/plausibility_step2_feature_matrix.toml` for the harness this was
originally built with (`benchmarks/build_feature_matrix_config.py`).
