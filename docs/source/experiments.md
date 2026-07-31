# Comparison of CCA Sticking Methods

```{warning}
**The densification result on this page was withdrawn on 2026-07-30.**
The reported "100% success, 20x faster" figure counted aggregates that
were geometrically invalid — every densified aggregate examined carried
43–69% residual particle overlap — and was scored on radius of gyration,
a quantity densification optimizes directly and which therefore cannot
distinguish an aggregate at Df=2.1 from a compressed one at Df=1.8.
Evaluated against the density–density correlation function, densified
aggregates fall roughly 0.5 below their target Df. See
[correlation_validation.md](correlation_validation.md).

The rigid-search comparisons on this page are unaffected, and their
central finding stands: search strategy does not determine hard-regime
success. [backtracking_pairing.md](backtracking_pairing.md) identifies
the variable that does.
```

This page compares the CCA (cluster-cluster aggregation) sticking
strategies evaluated between early and mid-2026. The production default
is Fibonacci-spiral sticking with an incremental active-set overlap
check. None of the alternatives evaluated below outperformed it.
Non-default implementations are archived under `pyfracval/experimental/`
(see Implementation notes).

## Background: geometric frustration

In hard parameter regimes — high `Df`, low `kf`, wide polydispersity
(e.g. `Df=2.25, kf=0.95, rp_gstd=1.9`) — CCA sticking success rates fall
to roughly 17–20%, independent of how the rotation search is conducted.
The cause is geometric frustration: the fractal scaling law fixes the
contact distance (`gamma_pc`) at which two clusters must be placed, and
at these parameter combinations no overlap-free relative orientation
exists at that distance. Refining the search cannot recover a solution
that does not exist.

## Method comparison

Hard regime: N=128, Df=2.25, kf=0.95, rp_gstd=1.9. Source:
`benchmark_results/profiles/method_comparison_hard_regime/`,
`benchmark_results/profiles/soft_quick_test/` (30 trials per method
unless noted; commit `d241350`).

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
| Densification (generate at Df=1.8/kf=1.0, densify to target) | 100.0% | 2.1 s | withdrawn — see warning above |
| Densification (generate at Df=2.0/kf=1.0, densify to target) | 100.0% | 2.6 s | withdrawn — see warning above |

All rigid-body modifications to the sticking search — pair pre-filters,
Γ relaxation, their combination, and FFT-based rigid docking at two grid
resolutions — fall within noise of the baseline. Soft potential
relaxation is likewise statistically indistinguishable from its own
paired baseline. This pattern is consistent with the frustration
diagnosis: when no overlap-free orientation exists at the enforced
contact distance, the manner in which orientations are searched is
immaterial.

### Fractal accuracy of densified aggregates (withdrawn)

Source: `benchmark_results/fractal_structure_validation.json`. Retained
for the record; the accuracy metric used here (Rg agreement) is
insufficient for the reasons given in the warning above.

| Method | Success rate | Mean \|Rg error\| | Max \|Rg error\| |
|---|---|---|---|
| Baseline (rigid sticking, successes only) | 40.0% (12/30) | 1.67% | 3.52% |
| Densify (source Df=2.0) | 100.0% (30/30) | 0.42% | 0.93% |
| Densify (source Df=1.8) | 100.0% (30/30) | 1.04% | 1.84% |

## Retry rotation modes

Source: `benchmark_results/profiles/retry_mode_matrix_hard_v1/` (12
trials per mode, hard regime, N=256 and N=512).

| Mode | N=256 success | N=256 median | N=512 success | N=512 median |
|---|---|---|---|---|
| `single` | 8.3% (1/12) | 57.6 s | 16.7% (2/12) | 115.8 s |
| `alternate` | 8.3% (1/12) | 56.4 s | 16.7% (2/12) | 115.8 s |
| `coarse_grid` | 8.3% (1/12) | 56.8 s | 16.7% (2/12) | 115.8 s |
| `coarse_to_fine` | 8.3% (1/12) | 58.3 s | 16.7% (2/12) | 114.2 s |

The four rotation-retry strategies produce identical success counts and
statistically indistinguishable timing at both sizes tested. Broadening
the rotation search does not help when no orientation is overlap-free at
the required contact distance.

## Candidate ordering policies

Source: `benchmark_results/profiles/candidate_policy_probe_v1/` (8
trials per policy, N=512).

| Policy | Success rate | Median wall time |
|---|---|---|
| `leaf_hybrid` | 12.5% (1/8) | 73.8 s |
| `leaf_score` | 12.5% (1/8) | 75.2 s |

Both policies produce the same outcome. The sample is small (n=8 per
arm), but taken together with the retry-mode result above it supports
the conclusion that the manner in which a contact pair is searched
matters far less than whether a valid contact pair exists at the
enforced `gamma_pc`.

## Discussion

The production path remains vanilla Fibonacci sticking with a single
rotation mode, incremental active-set overlap checking, and baseline
candidate ordering; `pyfracval/cca/` (`pairing.py`, `candidates.py`,
`sticking.py`, `fallbacks.py`, `aggregator.py`) is organized around this
path. Each alternative evaluated here was a plausible hypothesis that
did not change the outcome when measured. The results are consistent
across all variants and support a single interpretation: hard-regime
failure is a property of the pairing and contact-distance constraints,
not of the search over orientations. Subsequent work on cluster pairing
([pairing_frustration.md](pairing_frustration.md),
[backtracking_pairing.md](backtracking_pairing.md)) confirmed this
interpretation and moved the boundary.

## Limitations

The retry-mode and candidate-policy comparisons use small trial counts
(8–12 per arm) at a single hard-regime point. Distinguishing "no
effect" from insufficient statistical power would require a larger
sweep across multiple `(Df, kf, rp_gstd)` points;
`configs/plausibility_step2_feature_matrix.toml` and
`benchmarks/build_feature_matrix_config.py` provide the harness this
was originally built with.

## Implementation notes

Non-winning implementations are archived under
`pyfracval/experimental/` rather than removed, since a different
parameterization of the same idea (e.g. Γ-expansion with a larger
expansion budget, or FFT docking at higher rotation sampling density)
may warrant revisiting. Each remains reachable through the same config
flags as before; `cca/` retains a thin opt-in dispatch:

- Retry rotation modes (`alternate`, `dual_jitter`, `coarse_grid`,
  `coarse_to_fine`) — `experimental/retry_modes.py`
- Pair feasibility pre-filters (bounding-volume, SSA) —
  `experimental/pair_prefilters.py`
- Γ-expansion — `experimental/gamma_expansion.py`
- FFT rigid-body docking — `experimental/fft_docking.py`
- Soft potential relaxation — `experimental/soft_relaxation.py`
- Non-baseline candidate scoring policies (`leaf_soft`/`leaf_score`/
  `leaf_hybrid`) — `experimental/candidate_policies.py`

The soft-accept and rigid-repair config flags (`cca_soft_accept_*`,
`cca_repair_*`) were confirmed to have no remaining implementation
(no reader anywhere in the codebase) and were removed outright rather
than archived.
