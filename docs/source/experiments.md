# CCA Sticking: What We Tried, and What Actually Worked

This page is a retrospective on the CCA (cluster-cluster aggregation) sticking
experiments developed between early and mid-2026. The default production path
is **vanilla Fibonacci-spiral sticking with an incremental active-set overlap
check** — everything else described below was evaluated and, with one clear
exception (densification), did **not** outperform it. The code for the
non-winning approaches has been archived under `pyfracval/experimental/`
rather than deleted outright, in case the underlying ideas turn out to be
useful with a different angle later.

## The problem: geometric frustration

In "hard" parameter regimes — high `Df`, low `kf`, wide polydispersity (e.g.
`Df=2.25, kf=0.95, rp_gstd=1.9`) — CCA sticking success rates collapse to
roughly **17–20%**, no matter how the rotation search is conducted. The
underlying cause is **geometric frustration**: the fractal scaling law forces
two clusters to be placed at a contact distance (`gamma_pc`) that, for these
parameter combinations, has no overlap-free relative orientation. No amount of
smarter searching finds a solution that doesn't exist at that fixed distance.

## Head-to-head results (hard regime: N=128, Df=2.25, kf=0.95, rp_gstd=1.9)

Source: `benchmark_results/profiles/method_comparison_hard_regime/`,
`benchmark_results/profiles/soft_quick_test/` (30 trials per method unless
noted; commit `d241350`).

| Method | Success rate | Median wall time | Verdict |
|---|---|---|---|
| **Baseline (vanilla Fibonacci, single rotation mode)** | 16.7% | 42.2 s | reference |
| Bounding-volume pair pre-filter | 16.7% | 41.0 s | no improvement |
| SSA (surface-accessibility) pair filter | 16.7% | 41.9 s | no improvement |
| Γ-expansion (gamma relaxation, max 3 attempts) | 16.7% | 41.4 s | no improvement |
| BV filter + Γ-expansion combined | 16.7% | 41.9 s | no improvement |
| FFT rigid-body docking (64³ grid) | 16.7% | 42.1 s | no improvement |
| FFT rigid-body docking (128³ grid) | 16.7% | 42.2 s | no improvement |
| Soft potential relaxation (paired baseline: 20.0%) | 20.0% | 35.6 s | no improvement over its own baseline (34.6 s) |
| **Densification (generate at Df=1.8/kf=1.0, densify to target)** | **100.0%** | **2.1 s** | **wins decisively** |
| **Densification (generate at Df=2.0/kf=1.0, densify to target)** | **100.0%** | **2.6 s** | **wins decisively** |

Every rigid-body enhancement to the sticking search — pair pre-filters, Γ
relaxation, combining them, and switching to FFT-based rigid docking at two
grid resolutions — lands within noise of the vanilla baseline. Soft potential
relaxation is likewise statistically indistinguishable from its own
paired baseline. **Densification is the only approach that changes the
outcome qualitatively**: 100% success, roughly **20× faster** than any rigid
search variant, because it sidesteps the frustration problem entirely by
generating the aggregate at an easier `Df`/`kf` and reshaping it afterward
rather than trying to force sticking at the hard target contact distance.

### Fractal accuracy of densified aggregates

Source: `benchmark_results/fractal_structure_validation.json`.

| Method | Success rate | Mean \|Rg error\| | Max \|Rg error\| |
|---|---|---|---|
| Baseline (rigid sticking, successes only) | 40.0% (12/30) | 1.67% | 3.52% |
| Densify (source Df=2.0) | 100.0% (30/30) | **0.42%** | 0.93% |
| Densify (source Df=1.8) | 100.0% (30/30) | 1.04% | 1.84% |

Densified aggregates are not just far more likely to succeed — their radius of
gyration also matches the theoretical scaling law *more* accurately on average
than the (much smaller) set of successful rigid-body runs.

## Retry rotation modes — no measurable difference

Source: `benchmark_results/profiles/retry_mode_matrix_hard_v1/` (12 trials per
mode, hard regime, N=256 and N=512).

| Mode | N=256 success | N=256 median | N=512 success | N=512 median |
|---|---|---|---|---|
| `single` | 8.3% (1/12) | 57.6 s | 16.7% (2/12) | 115.8 s |
| `alternate` | 8.3% (1/12) | 56.4 s | 16.7% (2/12) | 115.8 s |
| `coarse_grid` | 8.3% (1/12) | 56.8 s | 16.7% (2/12) | 115.8 s |
| `coarse_to_fine` | 8.3% (1/12) | 58.3 s | 16.7% (2/12) | 114.2 s |

All four rotation-retry strategies produce identical success counts and
statistically indistinguishable timing at both sizes tested. Broadening the
rotation search does not help when the underlying problem is that *no*
orientation is overlap-free at the required contact distance.

## Candidate ordering policies — no measurable difference

Source: `benchmark_results/profiles/candidate_policy_probe_v1/` (8 trials per
policy, N=512).

| Policy | Success rate | Median wall time |
|---|---|---|
| `leaf_hybrid` | 12.5% (1/8) | 73.8 s |
| `leaf_score` | 12.5% (1/8) | 75.2 s |

Both policies land on the same outcome. This is a small probe (n=8 per arm),
but combined with the retry-mode result above, it reinforces that *how* you
search for a contact pair matters far less than *whether a valid contact pair
exists at all* at the enforced `gamma_pc`.

## What this means for the codebase

- **Production path (default, kept as first-class code):** vanilla Fibonacci
  sticking, single rotation mode, incremental active-set overlap checking,
  baseline candidate ordering. This is what `pyfracval/cca/` (`pairing.py`,
  `candidates.py`, `sticking.py`, `fallbacks.py`, `aggregator.py`) optimizes
  for.
- **Supported opt-in feature:** densification (`pyfracval/densify.py`). It is
  the only mechanism in this codebase that reliably solves hard-regime
  generation, and does so faster than the "easy" regime's own rigid search.
- **Archived, not deleted** (moved to `pyfracval/experimental/`, off the
  production path but still reachable via the same config flags they always
  had - `cca/` keeps a thin opt-in dispatch to each):
  - Extra retry rotation modes (`alternate`, `dual_jitter`, `coarse_grid`,
    `coarse_to_fine`) → `experimental/retry_modes.py`
  - Pair feasibility pre-filters (bounding-volume, SSA) →
    `experimental/pair_prefilters.py`
  - Γ-expansion → `experimental/gamma_expansion.py`
  - FFT rigid-body docking → `experimental/fft_docking.py`
  - Soft potential relaxation → `experimental/soft_relaxation.py`
  - Non-baseline candidate scoring policies (`leaf_soft`/`leaf_score`/
    `leaf_hybrid`) → `experimental/candidate_policies.py`
  - Soft-accept + rigid repair (config flags `cca_soft_accept_*`/
    `cca_repair_*`) — confirmed genuinely dead code (no reader anywhere) and
    deleted outright rather than archived; there was no implementation left
    to preserve.

None of these were bugs — they were reasonable hypotheses, tested rigorously,
and the data says they don't move the needle on this problem. They're kept
around (rather than deleted outright) because a different angle on the same
idea — e.g. Γ-expansion combined with a *much* larger expansion budget, or FFT
docking at higher rotation sampling density — might behave differently and
someone may want to pick the thread back up.

## Open item

Retry-mode and candidate-policy results above use small trial counts (8–12
per arm) at a single hard-regime point. If someone wants to be more confident
about "no effect" versus "not enough statistical power," a larger sweep across
multiple `(Df, kf, rp_gstd)` points would be the next step — see
`configs/plausibility_step2_feature_matrix.toml` for the harness this was
originally built with (`benchmarks/build_feature_matrix_config.py`).
