# Pipeline Baseline: Status of Stages and Variants

A single entry point summarizing the status of every pipeline stage and
algorithmic variant evaluated so far, cross-linking the detailed
writeups rather than duplicating them. Two datasets appear only on this
page: the attribution of failures to pipeline stage (PCA vs. CCA), and
a benchmark of `densify_method="voronoi"`.

## Stage and feature status

| Stage / feature | Status | Finding | Details |
|---|---|---|---|
| PCA subclustering (baseline) | production default | essentially never the terminal failure cause (see below) | this page |
| CCA vanilla Fibonacci sticking | production default | reference baseline | [experiments.md](experiments.md) |
| CCA backtracking pairing | production default | 5% → 100% single-shot success in the hard regime; boundary moved outward across the grid | [backtracking_pairing.md](backtracking_pairing.md), [boundary_sweep_v2.md](boundary_sweep_v2.md) |
| CCA pairing, greedy first-fit | superseded default | diagnosed bottleneck — 97.4% of hard-regime failures had a rescuable alternative pairing | [pairing_frustration.md](pairing_frustration.md) |
| CCA pairing, exact matching / leaf-weighted matching | opt-in | no measurable improvement (+0.2pp over 4200 trials); the cheap feasibility graph it optimizes over cannot predict sticking success | [matching_pairing.md](matching_pairing.md) |
| CCA overlap-failure census (`cca_overlap_census_enabled`) | opt-in, diagnostic | hard-regime failures involve a median 9/24 particles in the failing pair (~37.5%), not a small handful | [overlap_failure_census.md](overlap_failure_census.md) |
| CCA drop-rescue (`cca_drop_rescue_enabled`) | opt-in, recommended off | default budget inert by design; permissive budgets trade an inconsistent success effect for a systematic accuracy cost | [drop_rescue.md](drop_rescue.md) |
| CCA retry rotation modes (`alternate`, `dual_jitter`, `coarse_grid`, `coarse_to_fine`) | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA candidate ordering (`leaf_soft`, `leaf_score`, `leaf_hybrid`) | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA pair pre-filters (bounding-volume, SSA) | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA Γ-expansion | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA FFT rigid-body docking | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA soft potential relaxation | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| Densification, `radial` | opt-in | earlier favorable conclusion withdrawn: matches Rg but not the target Df (f(r) measures ~0.5 low), and pre-fix versions emitted invalid geometry | [correlation_validation.md](correlation_validation.md) |
| Densification, `voronoi` | opt-in | inferior to `radial` on both axes: ~11.6x slower and materially less accurate | this page, below |
| JAX/GPU kernel port | not pursued | numba faster by 1–4 orders of magnitude at every tested size | [gpu_acceleration.md](gpu_acceleration.md) |
| Structured event log (`event_log_path`) | opt-in, diagnostic | per-merge/per-run failure statistics; quantifies non-locality and non-nearness of failures | [event_logging.md](event_logging.md) |
| Feasibility warning | production default (advisory) | logistic model of the measured boundary; 97.7% agreement with the ≥50% feasibility call | [feasibility_criterion.md](feasibility_criterion.md) |
| Df/kf/σ/N stability boundary | characterized | grids before and after backtracking | [hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md), [boundary_sweep_v2.md](boundary_sweep_v2.md), [full_stability_sweep.md](full_stability_sweep.md) |

## Failure attribution by pipeline stage (PCA vs. CCA)

### Motivation

Earlier sweeps report end-to-end success/failure only.
`BenchmarkResult` has carried a `failure_stage` field since
`sticking_benchmark.py` was written, but it was never populated:
`run_simulation()` had no way to report which stage failed, so both the
sequential path (`StickingBenchmark.run_single_trial`) and the Dask
path (`stability_sweep.py`) hardcoded `"UNKNOWN"`. An optional
`diagnostics` dict parameter on `run_simulation()`, populated at each
retry-loop exit point with one of `PCA`, `CCA`, `TIMEOUT`, `PARAMS`, or
`RADII_GEN`, now supplies it to both paths.

The attribution records the last attempt made before a trial gives up
(retry exhaustion or wall-clock timeout) or succeeds — not a breakdown
of every one of the up to 20 internal retries. A trial where PCA fails
on attempts 1–5 and CCA fails on attempt 6 (the timeout point) is
attributed to CCA.

### Method

`configs/pipeline_stage_census.toml`: Df ∈ {1.4, 1.8, 2.0, 2.2, 2.5},
kf ∈ {0.6, 0.9, 1.0, 1.2, 1.4}, σ ∈ {1.0, 1.5, 1.9}, N ∈ {128, 512,
1024}, 4 seeds per combination — 225 combinations, 900 trials, run via
`benchmarks/stability_sweep.py` (Dask, local cluster). The grid spans
from comfortably safe to well past the known hard regime, so that both
PCA- and CCA-dominated failure territory would appear in the same run
if either exists.

### Results

515/900 trials succeeded (57.2%). Of the 385 failures:

| Failure stage | Count | Share |
|---|---:|---:|
| CCA | 311 | 80.8% |
| TIMEOUT | 74 | 19.2% |
| PCA | 0 | 0.0% |

PCA subclustering did not terminally fail once in this grid; every
failure that was not a wall-clock timeout was attributed to CCA. This
quantifies, across a spanning grid rather than one hard-regime point,
what [pairing_frustration.md](pairing_frustration.md) established
qualitatively: the CCA pairing/sticking stage is where the pipeline
fails.

The `TIMEOUT` share grows with N, since larger aggregates cost more per
attempt, making the 90 s budget more likely to expire before all 20
retries complete:

| N | CCA failures | TIMEOUT failures | Success rate |
|---:|---:|---:|---:|
| 128 | 117 | 0 | 61.0% (183/300) |
| 512 | 115 | 15 | 56.7% (170/300) |
| 1024 | 79 | 59 | 54.0% (162/300) |

Raw output: `benchmark_results/pipeline_stage_census/`.

## Densify method comparison (`radial` vs. `voronoi`)

### Method

`benchmarks/densify_method_comparison.py`, same hard-regime
target/source parameters as the densify table in
[experiments.md](experiments.md) (Df=2.25 target, source Df=2.0/kf=1.0,
N=128, σ=1.9), 30 seeds per method, with
`pyfracval.fractal.validate_fractal_structure` supplying the accuracy
columns.

### Results

| Method | Success rate | Avg \|Rg error\| | Rg within 5% | Avg wall time |
|---|---|---:|---:|---:|
| `radial` | 30/30 (100%) | +0.4% | 30/30 | 2.3s |
| `voronoi` | 30/30 (100%) | +8.5% | 8/30 | 26.7s |

Both methods report 100% "success" in the sense that `run_simulation`
returns a usable aggregate either way; the cost of `voronoi` is hidden
inside that figure. Its iterative migration frequently fails to
converge within `max_densify_iters`/`max_push_iters` and falls back to
best-result (logged as `Densification did not fully converge; using
best result`), producing aggregates whose radius of gyration misses the
target by an average of 8.5% (worst observed case: +7.8% with an
estimated empirical Df of 1.448 against a target of 2.25). `radial`
converges cleanly and is roughly 11.6× faster on average.

Raw output:
`benchmark_results/densify_method_comparison/densify_method_comparison.json`.

## Discussion

Two conclusions follow from the data on this page: (1) search-strategy
work belongs on the CCA side of the pipeline — PCA subclustering is not
a terminal cause of lost trials; and (2) `densify_method="voronoi"`
should not be preferred over `radial` at these settings — it is slower
and less accurate, and its 100% success rate masks an accuracy problem
visible only in `validate_fractal_structure`'s Rg/Df error columns
rather than the binary success flag. The broader caution about
densification as a whole — that Rg agreement alone is an insufficient
validation target — is established in
[correlation_validation.md](correlation_validation.md).

## Limitations

The PCA/CCA attribution captures only the last attempt's outcome per
trial, not a per-attempt census across all 20 retries; a trial that
fails PCA repeatedly before timing out on CCA shows no PCA attribution.
The `voronoi` comparison covers one regime and one source Df and has
not been swept across the broader Df/kf/σ/N grid the way the `radial`
path has.
