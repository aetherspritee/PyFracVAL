# Matching-Based CCA Pairing: a Negative Result

[pairing_frustration.md](pairing_frustration.md) diagnosed the CCA
pairing bottleneck: in the hard regime, 97.4% of failures (38/39) had a
valid alternative pairing available in the same cluster pool that greedy
first-fit never considered. This page reports the implementation and
benchmarking of the direct response — exact maximum-cardinality matching
in place of greedy first-fit — and its outcome: no measurable
improvement (+0.2 percentage points over 4200 trials). The cause is
identified in the Discussion. The result is reported in the same manner
as the project's prior negative results (candidate ordering, pair
prefilters, Γ-expansion, FFT docking, soft relaxation,
`densify_method="voronoi"`).

## Method

`pyfracval/cca/matching.py` adds two alternatives to
`_generate_pairs()`'s greedy first-fit, selected via the
`cca_pairing_strategy` config flag (`"greedy"` default, `"matching"`,
`"matching_leaf_weighted"`):

- `max_cardinality_matching`: exact maximum-cardinality matching via
  memoized brute-force DP. Round pool sizes are small (bounded by
  roughly `1/n_subcl_percentage`, empirically ≤16), so this is cheap
  and exact; a Blossom-algorithm implementation is unnecessary at this
  scale.
- `leaf_weighted_matching`: the same matching, with total edge weight
  (from a per-cluster-pair leaf classification, reusing
  `candidates.py`'s per-particle leaf mask aggregated into a
  per-cluster leaf fraction) as a tiebreaker among cardinality-optimal
  solutions. Cardinality is optimized first and never sacrificed for
  weight, since giving up a matchable pair for a higher-weight edge
  would directly contradict the diagnosed problem.

Both operate over the same inexpensive gamma-feasibility graph
`_generate_pairs()`'s greedy loop already builds (`gamma_real and
gamma_pc < sum_rmax`, with the existing 10% relaxation fallback) —
deliberately not the graph `pairing_frustration_probe.py` builds for
offline diagnosis, which tests every edge by attempting production
sticking (3 retries each). That census is appropriate for a one-off
diagnostic; running it inside `_generate_pairs()` on the hot path would
be prohibitively slow and would perturb the RNG state real trials
depend on. This design constraint turns out to explain the negative
result — see Discussion.

## Results

### Single-shot (frustration-probe regime and seeds)

`benchmarks/pairing_strategy_frustration_rerun.py`: same hard and
easy-control regimes, same 40 seeds, same single-shot (no internal
retry) methodology as the original probe, varying only
`cca_pairing_strategy`.

| Regime | greedy | matching | matching_leaf_weighted |
|---|---:|---:|---:|
| Hard (Df=2.25, kf=0.95, σ=1.9) | 2.5% (1/40) | 5.0% (2/40) | 2.5% (1/40) |
| Easy control (Df=1.8, kf=1.0, σ=1.5) | 100% (40/40) | 100% (40/40) | 100% (40/40) |

A one-trial difference at n=40 is not distinguishable from noise. Plain
matching does not measurably outperform greedy first-fit; leaf-weighted
matching performs identically to greedy on these seeds.

### Retry-inclusive boundary sweep

`benchmarks/pairing_strategy_sweep.py --config
configs/hard_regime_boundary_sweep.toml --strategies matching`: the full
840-combination, 4200-trial grid from
[hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md), run with
`cca_pairing_strategy="matching"` and compared cell-by-cell against the
greedy baseline (same literal seeds 1–5 per combination in both runs).

| Strategy | Total successes |
|---|---:|
| greedy (baseline) | 3039/4200 (72.4%) |
| matching | 3048/4200 (72.6%) |

The net difference is +9 trials (+0.2 percentage points) across the
grid. Of the 840 combinations, 51 (6.1%) differ at all between the two
strategies, and the differences run in both directions in roughly equal
measure (largest single swing +3 at one cell, against several −2 and +2
swings elsewhere). There is no systematic improvement concentrated
anywhere in the grid; the pattern is consistent with noise from each
internal-retry attempt consuming a different amount of shared RNG state
once the pairing decision diverges between strategies, even from the
same literal seed.

`matching_leaf_weighted` was not re-run at full-grid scale: the
single-shot comparison above shows it performing identically to greedy
on the exact seeds where an effect was expected, and a second 4200-trial
run for a strategy with no signal in the targeted test was not
considered a justified use of the compute budget.

Raw output: `benchmark_results/pairing_strategy_frustration_rerun.json`,
`benchmark_results/hard_regime_boundary_sweep_matching/`.

## Discussion

The 97.4% "rescuable" figure from
[pairing_frustration.md](pairing_frustration.md) was computed against a
feasibility graph built from actual sticking outcomes (each edge tested
via the real production sticking path, 3 retries). Matching over that
graph would find rescuing pairings in nearly every case — that is what
the diagnostic showed. But `_generate_pairs()` cannot afford to build
that graph: it runs before any geometry has been attempted, using only
the inexpensive gamma-feasibility test (`gamma_pc < sum_rmax`), which is
a necessary but not sufficient condition for two clusters sticking
without overlap. The cheap graph is systematically denser and more
optimistic than the real one. Maximizing cardinality over the
too-permissive graph does not preferentially select edges that succeed
at the sticking stage; it selects a different feasible-looking pairing,
which fails at approximately the same rate as greedy's first-fit choice,
because both draw from the same pool of geometrically plausible pairs
without information about which the sticking search will resolve.

The same argument explains the absence of a leaf-weighting effect: the
weighting can only re-rank among cardinality-optimal solutions within
the same optimistic graph, and carries no additional information about
sticking outcomes.

The lever the diagnosis actually identified — repartnering when a chosen
pair fails to stick, using outcome information a precomputed graph does
not have — is a different design: backtracking within `_run_iteration`
rather than a smarter upfront graph. That design is implemented and
evaluated in [backtracking_pairing.md](backtracking_pairing.md).

## Status

`cca_pairing_strategy="matching"`/`"matching_leaf_weighted"` are
implemented, tested (`tests/test_cca_matching.py`), and available as
opt-in config values. They do not move hard-regime success rates and are
not promoted beyond opt-in.
