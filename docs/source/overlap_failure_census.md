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

This is not necessarily true at every N or every round - the example that
motivated Phase 3 describes a *late*-round merge of two already-large
aggregates, which this probe did not sample (every hard-regime failure at
N=128 happens at round 1, before any cluster gets large). Whether
late-round failures between large clusters look different - fewer,
more localized offending particles relative to cluster size - is an open
question Phase 3 needs to check directly rather than assume, e.g. by
running this same census at larger N where more CCA rounds occur before
a failure can happen.

## Limitations

4 of 27 hard-regime failures produced no census: their last attempt never
reached the rotation-search stage (initial rigid placement itself failed
for that specific candidate pair), so there was no rotated geometry to
scan. The census also only covers the *single* failing pair per round,
not a full-pool feasibility census the way
`pairing_frustration_probe.py`'s offline diagnostic does - this is
deliberately cheaper and runs inside the real production retry loop
rather than as a separate single-shot tool. "Last-attempted placement" is
not necessarily the *closest*-to-success attempt tried, since candidates
are tried in shuffled (or policy-ordered) order, not ranked by prior
overlap severity - a future refinement could track the minimum-overlap
attempt across the whole loop instead of just the final one.

Raw output: `benchmark_results/overlap_census_probe.json`.
