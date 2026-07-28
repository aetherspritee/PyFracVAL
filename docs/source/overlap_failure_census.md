# Statistical Overlap-Failure Census

Binary success/fail tells you *whether* a CCA sticking attempt worked, not
*how close* it got or *how many* particles were actually the problem.
This page adds an opt-in diagnostic that answers the second question, and
reports what the data actually shows about the "drop a few particles"
rescue idea before Phase 3 attempts to build it.

## Method

`pyfracval/overlap_statistics.py::compute_overlap_census` runs a full
(non-early-exit) pairwise scan between two clusters and returns severity
data: how many particle pairs overlap, which particles are involved on
each side, the overlap-fraction distribution, and cluster sizes.
`pyfracval/overlap.py`'s hot-path scalar overlap functions return only a
single max-overlap float and exit the instant any pair exceeds tolerance
- that early exit is why numba beats JAX by orders of magnitude
(see [gpu_acceleration.md](gpu_acceleration.md)) and is not touched here.

Wired in as a strictly opt-in hook (`cca_overlap_census_enabled`, default
`False`) at the one place `pyfracval/cca/fallbacks.py::_perform_cca_sticking`
gives up on a pair (candidate list exhausted): it censuses the
*last-attempted* candidate placement - whatever `coords1_stick`/
`current_coords2` held when the loop exited - not a full record of every
attempt tried. If the very last attempt never got past initial placement
(no rotation was ever tried), there is no geometry to census and the hook
no-ops. When enabled, the result is stashed on
`CCAggregator._last_overlap_census` and threaded out through
`run_simulation`'s existing `diagnostics` hook (added for
[pipeline_baseline.md](pipeline_baseline.md)) into
`BenchmarkResult.overlap_census` for benchmark harnesses to consume.

## Results

`benchmarks/overlap_census_probe.py`: same hard/easy-control regimes as
`pairing_frustration_probe.py`, N=128, 40 seeds, but using the
retry-inclusive `run_simulation` path (the metric `--max-attempts`
actually gives users) with the census enabled.

| Regime | Success rate | Failures censused |
|---|---:|---:|
| Hard (Df=2.25, kf=0.95, σ=1.9) | 32.5% (13/40) | 23/27 |
| Easy control (Df=1.8, kf=1.0, σ=1.5) | 100% (40/40) | 0/0 |

(4 of 27 hard-regime failures got no census: their final attempt never
reached rotation - see Limitations.)

Every one of the 23 censused failures has a combined cluster-pair size of
exactly 24 particles - consistent with
[pairing_frustration.md](pairing_frustration.md)'s finding that failures
concentrate at the very first CCA round, merging PCA subclusters of
roughly 12 particles each.

| Offending particles (both clusters) | Count |
|---|---:|
| <=5 | 2 (8.7%) |
| 6-10 | 12 (52.2%) |
| 11-20 | 9 (39.1%) |
| >20 | 0 |

Mean 10.3, median 9 offending particles - out of only 24 total in the
failing pair. Overlap severity skews toward the high end: of 217 total
overlapping pairs recorded across these 23 failures, 137 (63.1%) fall in
the most severe bucket (overlap fraction > 0.3).

### N=512, for comparison

The same census at N=512 (same Df/kf/σ, 40 seeds):

| Regime | Success rate | Failures censused |
|---|---:|---:|
| Hard, N=512 | 2.5% (1/40) | 10/39 |

Cluster-pair size at failure is again a single fixed value (100, not
24) - N=512 hard-regime failures also concentrate at round 1, same as
N=128. Offending particles: mean 20.3, median 20, out of 100 - a smaller
*relative* fraction (20%) than N=128's 37.5%, though a larger *absolute*
count. Census coverage is much lower here (10/39, 25.6%, vs. N=128's
85.2%): most N=512 failures never reach the rotation-search stage at
all - see Limitations for what that means for the census's scope. Raw
output: `benchmark_results/overlap_census_probe_n512.json`.

## Discussion

The offending-particle counts above are a fraction of a *small* cluster
pair (median 9/24, ~37.5%), not a small fraction of a *large* one. This
matters directly for the "drop a few particles and retry" idea motivating
Phase 3: at this regime and this N, a typical failure is not "512 vs. 512
particles with 5 troublemakers" - it is two ~12-particle PCA subclusters
where over a third of all particles are implicated, and severity is
usually well past a marginal near-miss (most overlaps exceed 30% of the
smaller particle's radius). Dropping that many particles from a
12-particle cluster is a structurally significant change, not a minor
correction, and a fixed "drop <=5" budget (the scale suggested by the
"two 512-particle aggregates" example) would not have covered 91.3% of
the failures observed here.

The N=512 comparison answers part of the open question this raises, but
not all of it: the *relative* offending fraction does shrink with N (20%
vs. 37.5%), a real, if modest, trend in the direction the original "5 out
of 512" framing hoped for - but the *absolute* count needed (median 20)
is still far larger than a small fixed budget covers, and this is still
round-1 data (larger initial subclusters, not a late-round merge of
already-large aggregates). The example that motivated Phase 3 specifically
describes a *late*-round merge, which neither N tested here samples -
every hard-regime failure observed, at both N=128 and N=512, happens at
round 1. Whether a genuinely late-round failure looks different again is
still open; it would need a probe that specifically waits for (or forces)
a later-round failure rather than sampling whichever round fails first.

## Limitations

4 of 27 hard-regime failures at N=128 produced no census (10/39 uncensused
at N=512, a much larger share): their last attempt never reached the
rotation-search stage (initial rigid placement itself failed for that
specific candidate pair), so there was no rotated geometry to scan. That
N=512's uncensused share is so much larger is itself worth noting - most
large-cluster failures apparently fail even earlier in the attempt
pipeline than small-cluster ones do, which the census as currently scoped
can't say anything about. The census also only covers the *single*
failing pair per round,
not a full-pool feasibility census the way
`pairing_frustration_probe.py`'s offline diagnostic does - this is
deliberately cheaper and runs inside the real production retry loop
rather than as a separate single-shot tool. "Last-attempted placement" is
not necessarily the *closest*-to-success attempt tried, since candidates
are tried in shuffled (or policy-ordered) order, not ranked by prior
overlap severity - a future refinement could track the minimum-overlap
attempt across the whole loop instead of just the final one.

Raw output: `benchmark_results/overlap_census_probe.json`.
