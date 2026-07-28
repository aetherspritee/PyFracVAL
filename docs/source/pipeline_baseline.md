# Pipeline Baseline: What Works, What Doesn't

A single entry point summarizing the status of every pipeline stage and
algorithmic variant evaluated so far, cross-linking the detailed writeups
rather than duplicating them, plus two pieces of data no earlier page
covers: which pipeline stage (PCA vs CCA) actually causes a failure, and
a first benchmark of `densify_method="voronoi"` (previously untested).

## Stage/feature status

| Stage / feature | Status | Verdict | Details |
|---|---|---|---|
| PCA subclustering (baseline) | production default | essentially never the terminal failure cause (see below) | this page |
| CCA vanilla Fibonacci sticking | production default | reference baseline | [experiments.md](experiments.md) |
| CCA pairing (greedy first-fit) | production default | diagnosed bottleneck - 97.4% of hard-regime failures had a rescuable alternative pairing available | [pairing_frustration.md](pairing_frustration.md) |
| CCA pairing, exact matching / leaf-weighted matching | opt-in | implemented and benchmarked - no measurable improvement (+0.2pp over 4200 trials); the cheap feasibility graph matching operates on is too optimistic to translate into real sticking success | [matching_pairing.md](matching_pairing.md) |
| CCA overlap-failure census (`cca_overlap_census_enabled`) | opt-in, diagnostic only | implemented - reveals hard-regime failures involve a median 9/24 particles in the failing pair (~37.5%), not a small handful; informs whether a drop-a-few-particles rescue is viable | [overlap_failure_census.md](overlap_failure_census.md) |
| CCA retry rotation modes (`alternate`, `dual_jitter`, `coarse_grid`, `coarse_to_fine`) | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA candidate ordering (`leaf_soft`, `leaf_score`, `leaf_hybrid`) | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA pair pre-filters (bounding-volume, SSA) | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA Γ-expansion | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA FFT rigid-body docking | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| CCA soft potential relaxation | opt-in, archived | no measurable benefit | [experiments.md](experiments.md) |
| Densification, `radial` | opt-in | the one feature that changes outcomes qualitatively - 100% success, ~20x faster than rigid search in the hard regime | [experiments.md](experiments.md) |
| Densification, `voronoi` | opt-in | worse than `radial` on both axes: ~11.6x slower and materially less accurate | this page, new data below |
| JAX/GPU kernel port | not pursued | numba wins by 1-4 orders of magnitude at every tested size | [gpu_acceleration.md](gpu_acceleration.md) |
| Df/kf/σ/N stability boundary | characterized | see full grid maps | [hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md), [full_stability_sweep.md](full_stability_sweep.md) |

## New data: PCA vs CCA failure attribution

### Background

Every earlier sweep reports end-to-end success/failure only.
`BenchmarkResult` has carried a `failure_stage` field since
`sticking_benchmark.py` was written, but it was never actually populated:
`run_simulation()` had no way to report which stage failed, so both the
sequential path (`StickingBenchmark.run_single_trial`) and the Dask path
(`stability_sweep.py`) hardcoded it to `"UNKNOWN"` (the Dask path
hardcoded the entire `failure_stage_counts` field to `{}`). Fixed by
adding an optional `diagnostics` dict parameter to `run_simulation()` that
both paths now thread through, populated at each retry-loop exit point
with one of `PCA`, `CCA`, `TIMEOUT`, `PARAMS`, or `RADII_GEN`.

Scope note: this attributes *the last attempt made* before a trial
finally gives up (retry exhaustion or wall-clock timeout) or succeeds -
not a full breakdown of every one of the up to 20 internal retry attempts
within a single trial. A trial where PCA fails on attempts 1-5 and CCA
fails on attempt 6 (the timeout point) is attributed to CCA.

### Method

`configs/pipeline_stage_census.toml`: Df ∈ {1.4, 1.8, 2.0, 2.2, 2.5}, kf ∈
{0.6, 0.9, 1.0, 1.2, 1.4}, σ ∈ {1.0, 1.5, 1.9}, N ∈ {128, 512, 1024}, 4
seeds/combo - 225 combinations, 900 trials, run via
`benchmarks/stability_sweep.py` (Dask, local cluster). Grid intentionally
spans from comfortably-safe to well past the known hard regime so both
PCA- and CCA-dominated failure territory would show up in the same run,
if either exists.

### Results

515/900 trials succeeded (57.2%) across this grid. Of the 385 failures:

| Failure stage | Count | Share |
|---|---:|---:|
| CCA | 311 | 80.8% |
| TIMEOUT | 74 | 19.2% |
| PCA | 0 | 0.0% |

PCA subclustering did not terminally fail even once in this grid - every
failure that wasn't a wall-clock timeout was attributed to CCA
aggregation. This is a direct, quantified confirmation (across a spanning
grid rather than one hard-regime point) of what
[pairing_frustration.md](pairing_frustration.md) already established
qualitatively: CCA, specifically its pairing/sticking stage, is where
this pipeline actually loses.

`TIMEOUT` share grows sharply with N, since larger aggregates cost more
per attempt, making the 90s budget more likely to run out before all 20
retries complete:

| N | CCA failures | TIMEOUT failures | Success rate |
|---:|---:|---:|---:|
| 128 | 117 | 0 | 61.0% (183/300) |
| 512 | 115 | 15 | 56.7% (170/300) |
| 1024 | 79 | 59 | 54.0% (162/300) |

Raw output: `benchmark_results/pipeline_stage_census/`.

## New data: densify method comparison (`radial` vs `voronoi`)

### Method

`benchmarks/densify_method_comparison.py`, same hard-regime target/source
parameters `experiments.md`'s existing densify table uses (Df=2.25
target, source Df=2.0/kf=1.0, N=128, σ=1.9), 30 seeds per method, using
`pyfracval.fractal.validate_fractal_structure` for the accuracy columns.

### Results

| Method | Success rate | Avg \|Rg error\| | Rg within 5% | Avg wall time |
|---|---|---:|---:|---:|
| `radial` | 30/30 (100%) | +0.4% | 30/30 | 2.3s |
| `voronoi` | 30/30 (100%) | +8.5% | 8/30 | 26.7s |

Both methods report 100% "success" in the sense that `run_simulation`
returns a usable aggregate either way - `voronoi`'s cost is hidden inside
that number. Its iterative migration frequently fails to converge within
`max_densify_iters`/`max_push_iters` and falls back to "best result"
(logged as `Densification did not fully converge; using best result`),
producing aggregates whose radius of gyration misses the target by an
average of 8.5% (worst individual case observed: +7.8% with an estimated
empirical Df of 1.448 against a target of 2.25 - a densify run that
converged in name only). `radial` converges cleanly and is roughly 11.6x
faster on average.

Raw output: `benchmark_results/densify_method_comparison/densify_method_comparison.json`.

## Discussion

Two actionable conclusions from the new data: (1) any future search-strategy
work belongs entirely on the CCA side of the pipeline - PCA subclustering
is not where this project is losing trials, at least not as a terminal
cause; and (2) `densify_method="voronoi"` should not be recommended over
`radial` at these settings. It isn't broken (it does produce a result),
but it is both slower and less accurate, and its 100% "success" rate
masks a real accuracy problem that only shows up by checking
`validate_fractal_structure`'s Rg/Df error columns rather than the
binary success flag alone - the general lesson `experiments.md`'s own
densify table already established.

## Limitations

The PCA/CCA attribution only captures the *last* attempt's outcome per
trial, not a full per-attempt census across all 20 retries - a trial that
fails PCA repeatedly before eventually succeeding or timing out on CCA
would show no PCA attribution at all. The `voronoi` comparison covers one
regime and one source Df; it has not been swept across the broader
Df/kf/σ/N grid the way the production `radial` path has.
