# Matching-Based CCA Pairing: Implemented, Benchmarked, No Measurable Benefit

[pairing_frustration.md](pairing_frustration.md) diagnosed the CCA pairing
bottleneck: in the hard regime, 97.4% of failures (38/39) had a valid
alternative pairing available in the same cluster pool that greedy
first-fit never considered. This page implements the matching-based fix
that finding motivated, benchmarks it against the same grids used to
characterize the current implementation, and reports the result: no
measurable improvement, for a specific, identifiable reason explained
below. This is a negative result, reported the same way prior negative
results in this project have been (candidate ordering, pair prefilters,
Γ-expansion, FFT docking, soft relaxation, `densify_method="voronoi"`).

## Method

`pyfracval/cca/matching.py` adds two alternatives to
`_generate_pairs()`'s greedy first-fit, selected via a new
`cca_pairing_strategy` config flag (`"greedy"` default, `"matching"`,
`"matching_leaf_weighted"`):

- `max_cardinality_matching`: exact maximum-cardinality matching via
  memoized brute-force DP, generalizing
  `benchmarks/pairing_frustration_probe.py`'s existing
  `_max_matching_size` (which only counted a matching's size) into one
  that reconstructs the actual pair assignment. Round pool sizes are
  small (bounded by roughly `1/n_subcl_percentage`, empirically <=16), so
  this is cheap and exact - a real Blossom-algorithm implementation is
  unnecessary complexity at this scale.
- `leaf_weighted_matching`: the same matching, with total edge weight
  (from a per-cluster-pair leaf classification, reusing
  `candidates.py`'s per-particle leaf mask aggregated into a per-cluster
  leaf fraction) as a tiebreaker among cardinality-optimal solutions -
  cardinality is optimized first and never sacrificed for weight, since
  giving up a matchable pair for a "nicer" edge would directly contradict
  the diagnosed problem.

Both operate over the same cheap gamma-feasibility graph
`_generate_pairs()`'s greedy loop already builds (`gamma_real and
gamma_pc < sum_rmax`, with the existing 10% relaxation fallback) -
deliberately not the expensive graph
`pairing_frustration_probe.py` builds for offline diagnosis, which tests
every edge by actually attempting production sticking (3 retries each).
That census is appropriate for a one-off diagnostic; running it inside
`_generate_pairs()` on the hot path would be far too slow and would
perturb the RNG state real trials depend on. This design choice turns out
to be the reason the result below is negative - see Discussion.

## Results

### Single-shot (exact frustration-probe regime and seeds)

`benchmarks/pairing_strategy_frustration_rerun.py`: same hard/easy-control
regimes, same 40 seeds, same single-shot (no internal retry) methodology
as the original probe, varying only `cca_pairing_strategy`.

| Regime | greedy | matching | matching_leaf_weighted |
|---|---:|---:|---:|
| Hard (Df=2.25, kf=0.95, σ=1.9) | 2.5% (1/40) | 5.0% (2/40) | 2.5% (1/40) |
| Easy control (Df=1.8, kf=1.0, σ=1.5) | 100% (40/40) | 100% (40/40) | 100% (40/40) |

A one-trial difference at n=40 is not distinguishable from noise. Plain
matching does not measurably outperform greedy; leaf-weighted matching
performs identically to greedy on these exact seeds.

### Retry-inclusive boundary sweep

`benchmarks/pairing_strategy_sweep.py --config
configs/hard_regime_boundary_sweep.toml --strategies matching`: the full
840-combination, 4200-trial grid from
[hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md), run with
`cca_pairing_strategy="matching"` and compared cell-by-cell against the
existing greedy baseline (same literal seeds 1-5 per combination in both
runs).

| Strategy | Total successes |
|---|---:|
| greedy (existing baseline) | 3039/4200 (72.4%) |
| matching | 3048/4200 (72.6%) |

Net difference: +9 trials (+0.2 percentage points) across the entire
grid. Of the 840 combinations, 51 (6.1%) show any difference at all
between the two strategies, and those differences run in both directions
in roughly equal measure (the largest single swing is +3 at one cell,
against several -2 and +2 swings elsewhere) - not a systematic
improvement concentrated anywhere in the grid, consistent with noise from
each internal-retry attempt consuming a different amount of shared RNG
state once the pairing decision itself diverges between strategies, even
starting from the same literal seed.

`matching_leaf_weighted` was not re-run at this full-grid scale: the
single-shot result above already shows it performing identically to
greedy on the exact seeds the effect was expected to show up in, and
running the full 4200-trial grid a second time for a strategy the
targeted test already found no signal in was not judged worth the
compute budget.

Raw output: `benchmark_results/pairing_strategy_frustration_rerun.json`,
`benchmark_results/hard_regime_boundary_sweep_matching/`.

## Discussion

The 97.4% "rescuable" figure from `pairing_frustration.md` was computed
against a feasibility graph built from **actual sticking outcomes** (each
edge tested via the real production sticking path, 3 retries). Matching
over that graph would indeed find rescuing pairings almost every time -
that is precisely what the diagnostic showed. But `_generate_pairs()`
cannot afford to build that graph: it runs before any geometry has been
attempted, using only the cheap gamma-feasibility test (`gamma_pc <
sum_rmax`), which is a **necessary but not sufficient** condition for two
clusters actually sticking without overlap. The cheap graph is
systematically denser and more optimistic than the real one. Maximizing
cardinality over the wrong (too-permissive) graph does not reliably
select edges that will actually succeed at the sticking stage - it just
picks *a* different feasible-looking pairing, which fails at roughly the
same rate as greedy's first-fit choice, because both are drawing from the
same pool of "looks geometrically plausible" pairs without knowing which
ones the sticking search will actually resolve.

This also explains why leaf-weighting shows no effect: it can only
re-rank among cardinality-optimal solutions in the same cheap,
optimistic graph - it has no more information about which pairs will
actually stick than plain matching does.

The real lever `pairing_frustration.md` identified - repartnering when a
chosen pair fails to *actually* stick, using outcome information matching
only doesn't have access to - is a fundamentally different design:
backtracking within `_run_iteration` (retry with a different partner
after a real stick failure, informed by what specifically failed) rather
than a smarter *upfront* graph computed before any sticking is attempted.
That is not implemented here.

## Status and future work

`cca_pairing_strategy="matching"`/`"matching_leaf_weighted"` are
implemented, tested (`tests/test_cca_matching.py`), and available as
opt-in config values, but do not move hard-regime success rates and are
not being promoted beyond opt-in. The TODO.md item this page closes is
being replaced with a more specific one: a backtracking-based pairing
strategy that retries a failed pair's cluster with a different partner
using the *real* sticking outcome (not a precomputed graph), which is the
design the evidence above actually points to.
