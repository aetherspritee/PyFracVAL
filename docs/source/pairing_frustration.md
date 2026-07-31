# Cluster Pairing and Geometric Frustration in CCA Sticking

[experiments.md](experiments.md) established that no search-strategy
modification to CCA sticking — rotation modes, candidate scoring, pair
prefilters, Γ-expansion, FFT docking, soft relaxation — produces a
measurable difference in hard-regime success rates. All of these operate
within a fixed, already-chosen pair of clusters at a fixed,
scaling-law-derived contact distance γ. This page examines a variable
none of them touch: which clusters are paired in the first place.

## Motivation

`pyfracval/cca/pairing.py::_generate_pairs` chooses merge partners by
greedy first-fit: for each cluster, the first later cluster (in
arbitrary index order) that passes a loose feasibility gate is taken,
and the search stops. `pyfracval/cca/aggregator.py::_run_iteration` then
attempts to stick each chosen pair; if any single pair fails, the entire
round is aborted and the PCA/CCA attempt restarts from scratch with a
fresh shuffle, discarding every other pair that stuck successfully in
that round.

This leaves open a question the earlier retrospective did not address:
when a hard-regime run fails, is it because no collision-free
configuration exists for that cluster pool (frustration is fundamental,
as concluded for search strategy), or because greedy first-fit selected
a poor partner for one cluster while a different pairing of the same
pool would have succeeded?

## Method

`benchmarks/pairing_frustration_probe.py` runs single-shot PCA/CCA
trials (no internal retry, so each seed maps to exactly one outcome).
Whenever a trial fails, a census is run on the exact cluster pool at the
failing round:

1. For every pair `(i, j)` in the pool, the production sticking path
   (`_perform_cca_sticking`) is attempted, with three independent
   retries per pair to rule out chance, and the outcome recorded,
   producing a feasibility graph.
2. The graph is tested for a perfect matching (every cluster paired, or
   exactly one left over as the pass-through "odd one out" CCA already
   supports) via brute-force matching with memoization. Round pool sizes
   are small (bounded by roughly `1 / n_subcl_percentage`, ~11 for the
   default, independent of N), so this is exact and cheap; the Blossom
   algorithm is not needed at this scale.
3. The exact pair the real run attempted is re-tested, to distinguish
   "this pair required a genuinely different partner" from "this pair
   would have worked on retry, with no repartnering needed."

The census consumes additional draws from the shared RNG. Its
bit-generator state is saved and restored around the census so the
outcome of the real run downstream is not perturbed.

## Results

40 single-shot trials per regime, N=128:

| Regime | Params | Success rate | CCA failures | Rescuable by pairing | Universally frustrated |
|---|---|---:|---:|---:|---:|
| Hard | Df=2.25, kf=0.95, σ=1.9 | 2.5% | 39 | 38 (97.4%) | 1 (2.6%) |
| Easy control | Df=1.8, kf=1.0, σ=1.5 | 100% | 0 | - | - |

The 2.5% single-shot rate is lower than the 16.7% figure reported in
[experiments.md](experiments.md) for the same nominal regime because
that figure includes the internal 20-attempt retry loop each real
CLI/`run_simulation` call uses; this probe bypasses that loop to obtain
a clean seed-to-outcome mapping. The two are different metrics, not a
contradiction.

Characteristics of the failing rounds:

- Every failure occurs at the first CCA round — the initial ~11 PCA
  subclusters merging down to ~6 — rather than at later rounds with a
  handful of large clusters.
- The feasibility graphs are dense, not sparse: edge density ranges
  40–95% of all possible pairs (median ~70%). Most pairings of the pool
  succeed; greedy first-fit selects one that does not and the round is
  discarded.
- The killing pair's degree in the feasibility graph is almost always
  well above zero (e.g. `[9,8]`, `[7,10]`, `[10,10]`): the cluster that
  caused the round to fail typically had other viable partners that
  greedy first-fit never considered.
- Of the 38 rescuable failures, 20 (53%) require a genuinely different
  partner: re-testing the exact pair the real run attempted also failed
  on census. These cases are not explained by retrying with a different
  rotation; a different pairing choice is required.

Raw data: `benchmark_results/pairing_frustration_probe.json`.

## Discussion

The earlier conclusion that frustration is fundamental and no search
strategy helps is correct for everything that was tested, but pairing
choice was outside its scope, and the present data identify it as the
relevant lever. Round pools are small (~11 clusters), failures
concentrate almost entirely at round 1, and the feasibility graph is
dense enough that a different pairing strategy — a true matching in
place of greedy first-fit, or backtracking to a different partner on
merge failure instead of aborting the whole attempt — has substantial,
low-cost leverage on hard-regime success rates.

[hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md) maps the
Df/kf/σ/N boundary around the hard regime on the unmodified
implementation, forming the baseline against which a pairing-choice fix
is evaluated.

## Status

This page documents a diagnosis, not a fix. The follow-up
implementations and their outcomes are reported in
[matching_pairing.md](matching_pairing.md) (negative result) and
[backtracking_pairing.md](backtracking_pairing.md) (adopted).
