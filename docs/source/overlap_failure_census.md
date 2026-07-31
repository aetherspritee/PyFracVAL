# Statistical Overlap-Failure Census

A binary success/fail outcome records whether a CCA sticking attempt
worked, but not how close it came or how many particles were involved.
This page describes an opt-in diagnostic that measures the latter, and
reports what the data show about the premise of the "drop a few
particles" rescue idea evaluated in [drop_rescue.md](drop_rescue.md).

## Method

`pyfracval/overlap_statistics.py::compute_overlap_census` runs a full
(non-early-exit) pairwise scan between two clusters and returns severity
data: the number of overlapping particle pairs, the particles involved
on each side, the overlap-fraction distribution, and cluster sizes. The
hot-path scalar overlap functions in `pyfracval/overlap.py` return only
a single max-overlap float and exit as soon as any pair exceeds
tolerance — that early exit is why numba outperforms JAX by orders of
magnitude (see [gpu_acceleration.md](gpu_acceleration.md)) and is not
modified here.

The census is wired in as a strictly opt-in hook
(`cca_overlap_census_enabled`, default `False`) at the point where
`pyfracval/cca/fallbacks.py::_perform_cca_sticking` gives up on a pair
(candidate list exhausted). It censuses the last-attempted candidate
placement — whatever `coords1_stick`/`current_coords2` held when the
loop exited — not a record of every attempt tried. If the final attempt
never got past initial placement (no rotation was tried), there is no
geometry to census and the hook is a no-op. When enabled, the result is
stored on `CCAggregator._last_overlap_census` and threaded out through
`run_simulation`'s `diagnostics` hook (added for
[pipeline_baseline.md](pipeline_baseline.md)) into
`BenchmarkResult.overlap_census`.

## Results

`benchmarks/overlap_census_probe.py`: same hard and easy-control regimes
as `pairing_frustration_probe.py`, N=128, 40 seeds, using the
retry-inclusive `run_simulation` path (the metric `--max-attempts`
exposes to users) with the census enabled.

| Regime | Success rate | Failures censused |
|---|---:|---:|
| Hard (Df=2.25, kf=0.95, σ=1.9) | 32.5% (13/40) | 23/27 |
| Easy control (Df=1.8, kf=1.0, σ=1.5) | 100% (40/40) | 0/0 |

(4 of 27 hard-regime failures produced no census: their final attempt
never reached rotation — see Limitations.)

All 23 censused failures have a combined cluster-pair size of exactly
24 particles — consistent with
[pairing_frustration.md](pairing_frustration.md)'s finding that failures
concentrate at the first CCA round, merging PCA subclusters of roughly
12 particles each.

| Offending particles (both clusters) | Count |
|---|---:|
| <=5 | 2 (8.7%) |
| 6-10 | 12 (52.2%) |
| 11-20 | 9 (39.1%) |
| >20 | 0 |

Mean 10.3 and median 9 offending particles, out of 24 total in the
failing pair. Overlap severity skews high: of 217 overlapping pairs
recorded across the 23 failures, 137 (63.1%) fall in the most severe
bucket (overlap fraction > 0.3).

### N=512

The same census at N=512 (same Df/kf/σ, 40 seeds):

| Regime | Success rate | Failures censused |
|---|---:|---:|
| Hard, N=512 | 2.5% (1/40) | 10/39 |

Cluster-pair size at failure is again a single fixed value (100 rather
than 24): N=512 hard-regime failures also concentrate at round 1.
Offending particles: mean 20.3, median 20, out of 100 — a smaller
relative fraction (20%) than N=128's 37.5%, but a larger absolute
count. Census coverage is much lower here (10/39, 25.6%, vs. N=128's
85.2%): most N=512 failures never reach the rotation-search stage —
see Limitations. Raw output:
`benchmark_results/overlap_census_probe_n512.json`.

## Discussion

The offending-particle counts are a large fraction of a small cluster
pair (median 9/24, ~37.5%), not a small fraction of a large one. This
bears directly on the "drop a few particles and retry" idea: at this
regime and N, a typical failure is not two 512-particle aggregates with
five troublemakers but two ~12-particle PCA subclusters with over a
third of all particles implicated, at severities well past a marginal
near-miss (most overlaps exceed 30% of the smaller particle's radius).
Dropping that many particles from a 12-particle cluster is a
structurally significant change, and a fixed "drop ≤5" budget — the
scale suggested by the "two 512-particle aggregates" example that
motivated the idea — would not have covered 91.3% of the failures
observed here.

The N=512 comparison resolves part of the question this raises. The
relative offending fraction does shrink with N (20% vs. 37.5%), a real
if modest trend in the direction the "5 out of 512" framing assumed —
but the absolute count needed (median 20) still exceeds any small fixed
budget, and this remains round-1 data (larger initial subclusters, not
a late-round merge of already-large aggregates). The motivating example
specifically describes a late-round merge, which neither N tested here
samples: every hard-regime failure observed, at both N=128 and N=512,
occurs at round 1. Whether a genuinely late-round failure looks
different remains open; answering it would require a probe that waits
for (or forces) a later-round failure rather than sampling whichever
round fails first.

## Limitations

4 of 27 hard-regime failures at N=128 produced no census, and 29 of 39
at N=512: their last attempt failed at initial rigid placement, before
any rotation, leaving no rotated geometry to scan. The much larger
uncensused share at N=512 is itself informative — most large-cluster
failures terminate earlier in the attempt pipeline than small-cluster
ones, a region the census as scoped cannot observe. The census also
covers only the single failing pair per round, not a full-pool
feasibility census of the kind `pairing_frustration_probe.py` performs
offline; this is deliberately cheaper and runs inside the production
retry loop. Finally, the last-attempted placement is not necessarily
the closest-to-success attempt, since candidates are tried in shuffled
(or policy-ordered) order rather than ranked by prior overlap severity;
tracking the minimum-overlap attempt across the whole loop would be a
natural refinement.

Raw output: `benchmark_results/overlap_census_probe.json`.
