# Cluster Pairing and Geometric Frustration in CCA Sticking

[experiments.md](experiments.md) established that every search-strategy
enhancement tried for CCA sticking - rotation modes, candidate scoring,
pair prefilters, Γ-expansion, FFT docking, soft relaxation - produces no
measurable difference in hard-regime success rates. All of those operate
inside a fixed, already-chosen pair of clusters at a fixed,
scaling-law-derived contact distance γ. This page examines a variable none
of them touch: which clusters are paired in the first place.

## Motivation

`pyfracval/cca/pairing.py::_generate_pairs` chooses merge partners by
greedy first-fit: for each cluster, it takes the first later cluster (in
arbitrary index order) that passes a loose feasibility gate, then stops
searching. `pyfracval/cca/aggregator.py::_run_iteration` then attempts to
stick each chosen pair; if any single pair fails, the entire round is
aborted (lines 321-326) and the PCA/CCA attempt restarts from scratch with
a fresh shuffle, discarding every other pair that did stick successfully
in that round.

This raises a question the retrospective did not address: when a
hard-regime run fails, is it because no collision-free configuration
exists for that cluster pool (frustration is fundamental, as
`experiments.md` concluded for search strategy), or because greedy
first-fit selected a poor partner for one cluster while a different
pairing of the same pool would have succeeded?

## Method

`benchmarks/pairing_frustration_probe.py` runs single-shot PCA/CCA trials
(no internal retry, so each seed maps to exactly one outcome). Whenever a
trial fails, it runs a census on the exact cluster pool at the failing
round:

1. For every pair `(i, j)` in the pool, attempt the production sticking
   path (`_perform_cca_sticking`, three independent retries per pair to
   rule out chance) and record success or failure, building a feasibility
   graph.
2. Test whether that graph admits a perfect matching (every cluster
   paired, or exactly one left over as the pass-through "odd one out" CCA
   already supports) via brute-force matching with memoization. Round pool
   sizes are small (bounded by roughly `1 / n_subcl_percentage`, ~11 for
   the default, independent of N), so this is exact and cheap; the
   Blossom algorithm is not needed at this scale.
3. Re-tests the exact pair the real run attempted, to distinguish "this
   pair needed a genuinely different partner" from "this pair would have
   worked on retry, with no repartnering needed."

The census consumes additional draws from the shared RNG. Its
bit-generator state is saved and restored around the census so it does
not perturb the outcome of the real run downstream.

## Results

40 single-shot trials per regime, N=128:

| Regime | Params | Success rate | CCA failures | Rescuable by pairing | Universally frustrated |
|---|---|---:|---:|---:|---:|
| Hard | Df=2.25, kf=0.95, σ=1.9 | 2.5% | 39 | 38 (97.4%) | 1 (2.6%) |
| Easy control | Df=1.8, kf=1.0, σ=1.5 | 100% | 0 | - | - |

(The 2.5% single-shot rate is lower than the 16.7% figure reported in
`experiments.md` for the same nominal regime, because that number includes
the internal 20-attempt retry loop each real CLI/`run_simulation` call
uses; this probe bypasses that loop to obtain a clean seed-to-outcome
mapping. It is a different metric, not a contradiction.)

Characterizing the failing rounds:

- Every failure occurs at the very first CCA round - the initial ~11 PCA
  subclusters merging down to ~6 - rather than on later rounds with a
  handful of large, awkward clusters.
- The feasibility graphs are dense, not sparse: edge density ranges
  40-95% of all possible pairs (median ~70%). Most pairings of the pool
  succeed; greedy first-fit selects a poor one and discards the round.
- The killing pair's degree in the feasibility graph is almost always
  well above zero (e.g. `[9,8]`, `[7,10]`, `[10,10]`): the cluster that
  caused the round to fail typically had other viable partners that
  greedy never considered.
- Of the 38 rescuable failures, 20 (53%) require a genuinely different
  partner: re-testing the exact pair the real run attempted also failed
  on census. This is not explained by "retry and get a different
  rotation"; a different pairing choice is required.

Raw data: `benchmark_results/pairing_frustration_probe.json`.

## Discussion

The retrospective's conclusion that frustration is fundamental and no
search strategy helps is correct for everything it tested, but pairing
choice was outside its scope, and it is the relevant lever. Round sizes
are small (~11 clusters), failures concentrate almost
entirely at round 1, and the feasibility graph is dense enough that a
smarter pairing strategy - a true matching in place of greedy first-fit,
or backtracking to a different partner on merge failure instead of
aborting the whole attempt - has real, low-cost leverage on hard-regime
success rates.

See [hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md) for a
follow-up sweep mapping the Df/kf/σ/N boundary around the hard regime on
the current (unmodified) implementation, forming the baseline a future
pairing-choice fix would be evaluated against.

## Status and future work

This page documents a diagnosis, not a fix. `_generate_pairs`'s greedy
first-fit is unchanged; no matching-based or backtracking pairing
strategy has yet been implemented or benchmarked end-to-end. That
implementation is the natural next step this finding motivates.
