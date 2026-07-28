# A Wider Stability Sweep, and an Empirical Runtime Model

[hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md) mapped the
boundary right around PyFracVAL's established hard regime. This page
extends the grid much further in every dimension - closer in scope to
Tamer Areij's bachelor thesis {cite:p}`Areij2026Bachelorarbeit`, but wider
and finer than either the thesis or the previous sweep - and adds
something neither of them had: an empirical model for how long a
parameter combination takes to run, fit from real per-combination timing
data.

## Grid

`configs/full_thesis_replication_sweep.toml`:

| Parameter | Range | Count |
|---|---|---:|
| Df | 1.4 to 2.6, step 0.1 | 13 |
| kf | 0.5 to 1.5, step 0.1 | 11 |
| σ (rp_gstd) | 1.0, 1.25, 1.5, 1.75 | 4 |
| N | 128 to 1024, step 128 | 8 |
| seeds | 1, 2, 3, 4, 5 (literal, reused across every combination) | 5 |

`13 × 11 × 4 × 8 = 4576` combinations × 5 seeds = **22,880 trials**, run via
`benchmarks/stability_sweep.py`'s standard retry-inclusive metric (same as
the thesis's Seed 1/2/3 convention - see
[pairing_frustration.md](pairing_frustration.md) for why this differs from
the single-shot metric used there).

Raw output: `benchmark_results/full_thesis_replication_sweep/` (summary
JSON/CSV under `stability_sweeps/`, ~15,500 generated aggregate `.dat`
files under `aggregates/` - gitignored, not part of the repository).

## A real infrastructure lesson from running this

This sweep took roughly 20 hours wall-clock on a 16-core machine - not the
~2 hour estimate a straightforward "5.5x the previous sweep's size" scaling
would suggest. Two real bugs, found by actually running something at this
scale rather than a smaller smoke test, explain the gap:

1. **`get_client()`'s local cluster only used 4 of 16 cores.** Its
   docstring promised "defaults to the number of CPU cores when n_workers
   is None," but passing `None` straight through to `LocalCluster` doesn't
   deliver that - Dask's own default heuristic also factors in
   *available* system memory at cluster-start time, and silently picked 4
   workers on a 16-core/64GB machine that had a few GB already committed to
   unrelated running applications. Confirmed by connecting to the running
   scheduler's dashboard and counting actual worker processes. Fixed by
   resolving `n_workers = os.cpu_count()` explicitly rather than trusting
   Dask's memory-aware guess for what is, here, a purely CPU-bound batch
   workload.
2. **The slowest trials cluster at the tail of a `Dask.as_completed()`
   queue.** Fast trials finish (and get reported) first regardless of
   submission order; the grid's largest-N, most-extreme-Df/kf combinations
   were both submitted last (`sizes` is the outermost loop) *and*
   individually the most expensive - a double penalty that made the
   remaining-time estimate balloon well past a linear extrapolation from
   early progress.

Neither is fixed by "wait longer" - the worker-count fix is committed and
will apply to the next sweep; the queue-tail effect is inherent to
`as_completed()` and worth planning around (e.g., submitting hardest
combinations first) rather than "fixing," if a future sweep needs a
tighter time budget.

## Results

### Success rate vs Df

```{image} _static/sweep/success_vs_df.png
:alt: Success rate vs Df, collapsed over kf, N, sigma
:width: 600px
```

A clean, unimodal curve peaking at **Df=2.0** (not Df=1.8, where the thesis
found 100%) - explained entirely by the wider kf range: kf=0.5 (below the
thesis's floor of 1.0) already fails at Df=1.8 in this grid, pulling the
Df=1.8 average down to ~81% once it's included. This is a genuine
extension of the thesis's finding, not a contradiction of it: the "safe"
Df region depends on which kf range you're averaging over.

### Df × kf heatmap

```{image} _static/sweep/df_kf_heatmap.png
:alt: Success rate heatmap over Df and kf
:width: 600px
```

The safe band (Df≈1.8-2.3, green) is bounded by kf on *both* sides, sloping
in the direction the thesis already established (higher Df needs lower kf
and vice versa) - but this grid reaches new territory the thesis never
tested: at **Df=1.4, every tested kf from 0.5 to 1.5 fails outright**. The
thesis found Df=1.4 recoverable, but only at kf=2.2 - outside this grid's
range entirely. Low-Df instability apparently needs a much larger kf
correction than high-Df instability needs in the opposite direction.

### Success rate vs N and σ

```{image} _static/sweep/success_vs_n_sigma.png
:alt: Success rate vs N, by sigma
:width: 600px
```

Confirms the thesis's finding with much finer N resolution (8 steps vs.
their 4): success rate degrades smoothly and monotonically with both N and
σ, with no sharp cliffs - consistent with N/σ acting as amplifiers of
whatever margin Df/kf already leaves, not independent failure causes (see
[hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md) for the
same conclusion at a finer boundary-focused grid).

## An empirical runtime model

The summary only records per-combination averages (5 seeds), not
individual trial rows - `stability_sweep.py`'s Dask path never implemented
raw per-trial JSONL output (a separate gap, not fixed here). To keep each
fit honest about which cost regime it describes, only "clean" combinations
are used: all 5 seeds succeeded (2,402 combinations), or all 5 failed
(1,985 combinations) - the 189 mixed combinations are excluded, since their
average blends two very different cost profiles (a success stops at the
first working attempt; a failure exhausts all 20 internal retries).

`X = (N/kf)^(1/Df)` - the same dimensionless "how many particle-diameters
must the aggregate span" ratio `benchmarks/analyze_stability.py` already
uses for stability maps - is the primary feature, alongside raw N. Model:
`log(runtime_s) = a + b·log(N) + c·log(X)`, fit by ordinary least squares.

```{image} _static/sweep/speed_vs_x.png
:alt: Runtime vs X for success and failure paths
:width: 900px
```

| Regime | n | Equation | R² |
|---|---:|---|---:|
| Success (5/5 seeds) | 2402 | `runtime_s ≈ exp(-5.58) · N^1.82 · X^-1.37` | 0.50 |
| Failure, raw (0/5 seeds, incl. timeout-capped) | 1985 | `runtime_s ≈ exp(-1.28) · N^1.17 · X^-0.70` | 0.69 |
| Failure, uncensored | 1397 | `runtime_s ≈ exp(-1.39) · N^1.12 · X^-0.60` | 0.48 |

**A censoring caveat that matters here**: `trial_timeout=90` in the sweep
config caps any single trial's wall-clock budget. 588 of the 1,985 pure-
failure combinations (29.6%) hit that cap rather than naturally exhausting
all 20 internal retries - visible as a hard ceiling in the raw failure-path
scatter. The "raw" fit above describes *practical wait time given this
timeout setting* (still a real, useful number if you're asking "how long
before PyFracVAL gives up"); the "uncensored" fit, restricted to
combinations that finished under 85s on their own, is the better estimate
of the *actual* algorithmic cost of a failure, at the price of a lower R²
(removing the artificially flat ceiling also removes an easy source of
apparent fit quality).

**Reading the exponents**: successful runs scale worse than linearly with N
(`N^1.82`) - consistent with PCA subclustering and CCA merging both doing
more than O(N) work as the particle count grows. Failures scale closer to
linearly (`N^1.1-1.2`) - failing trials spend most of their time on the
first few CCA rounds before giving up, rather than scaling with the full
aggregate size the way a successful full build does. Both regimes show
*negative* exponents on X, meaning - within each regime - a geometrically
harder combination that still lands in that regime tends to run faster,
not slower. This isn't as strange as it looks: X isn't independent of N (X
= (N/kf)^(1/Df)), so at fixed N, a combination that produces a large X is
one with unusually low kf or Df, and those don't necessarily take longer
to resolve one way or the other - they just resolve differently.

**Honest caveat on fit quality**: R² of 0.48-0.69 means this model
explains roughly half to two-thirds of the variance - a genuinely useful
order-of-magnitude estimate for planning a sweep's compute budget, not a
precise predictor for any single combination. The visibly bimodal
structure in both scatter plots (a lower band running up to X≈20, a
distinct upper band beyond it) suggests there are at least two qualitatively
different cost regimes this simple 3-parameter model doesn't fully
separate - a `Df`/`kf`-aware regression, or a finer split of the failure
population by which pipeline stage typically kills the run (PCA vs. CCA),
would be a natural next step if a tighter fit is worth the effort later.

### Runtime distribution

```{image} _static/sweep/runtime_distribution.png
:alt: Runtime distribution, success vs failure
:width: 600px
```

The practical planning takeaway, independent of the fitted model: a
successful combination in this grid almost always finishes in under 10s;
a failing one costs anywhere from a few seconds up to the full 90s
timeout, with real mass sitting right at that cap.
