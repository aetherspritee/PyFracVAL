# CCA Pairing Frustration: A Lever the Retrospective Missed

[experiments.md](experiments.md) established that every *search-strategy*
enhancement tried for CCA sticking - rotation modes, candidate scoring, pair
prefilters, Γ-expansion, FFT docking, soft relaxation - makes no measurable
difference to hard-regime success rates. All of those operate **inside** a
fixed, already-chosen pair of clusters at a fixed, scaling-law-derived
contact distance γ. This page investigates something none of them touch:
**which clusters get paired in the first place.**

## The mechanism nobody had tested

`pyfracval/cca/pairing.py::_generate_pairs` chooses merge partners by greedy
first-fit - for each cluster, take the first later cluster (in arbitrary
index order) that passes a loose feasibility gate, then stop looking.
`pyfracval/cca/aggregator.py::_run_iteration` then attempts to stick each
chosen pair; if *any single pair* fails, the **entire round is aborted**
(lines 321-326) and the whole PCA+CCA attempt restarts from scratch with a
fresh shuffle - discarding every other pair that *did* stick successfully in
that round.

This raised an obvious question the retrospective never asked: when a hard-
regime run fails, is it because collision-free configurations genuinely
don't exist for that cluster pool (frustration is fundamental, as
`experiments.md` concluded for search strategy) - or because greedy
first-fit picked a bad partner for one cluster while a different pairing of
the same pool would have gone through cleanly?

## Method

`benchmarks/pairing_frustration_probe.py` runs single-shot PCA+CCA trials
(no internal retry, so each seed maps to exactly one outcome) and, whenever
a trial fails, runs a **census** on the exact cluster pool at the failing
round:

1. For every pair `(i, j)` in the pool, attempt the real production sticking
   path (`_perform_cca_sticking`, 3 independent retries per pair to rule out
   bad luck) and record whether it succeeds - building a feasibility graph.
2. Test whether that graph admits a **perfect matching** (every cluster
   paired, or exactly one left over as the pass-through "odd one out" CCA
   already supports) via brute-force matching with memoization - round pool
   sizes are small (bounded by roughly `1 / n_subcl_percentage`, ≈11 for the
   default, regardless of N), so this is cheap and exact, no need for
   Blossom-algorithm machinery.
3. Additionally re-tests the *exact* pair the real run tried, to distinguish
   "this needs a genuinely different partner" from "this pair would have
   worked on a retry, no repartnering needed."

The census consumes extra draws from the shared RNG; its bit-generator state
is saved and restored around the census so it cannot perturb what the real
run does downstream.

## Results

40 single-shot trials per regime, N=128:

| Regime | Params | Success rate | CCA failures | Rescuable by pairing | Universally frustrated |
|---|---|---:|---:|---:|---:|
| Hard | Df=2.25, kf=0.95, σ=1.9 | 2.5% | 39 | **38 (97.4%)** | 1 (2.6%) |
| Easy control | Df=1.8, kf=1.0, σ=1.5 | 100% | 0 | — | — |

(The 2.5% single-shot rate is lower than `experiments.md`'s 16.7% figure for
the same nominal regime because that number includes the internal 20-attempt
retry loop each real CLI/`run_simulation` call gets; this probe deliberately
bypasses that for a clean seed-to-outcome mapping. Different metric, not a
contradiction - if anything it makes 97.4% more striking, not less.)

Digging into the failing rounds themselves:

- **Every single failure happens at the very first CCA round** - the initial
  ~11 PCA subclusters merging down to ~6 - not on later rounds with a
  handful of large, awkward clusters as might be expected.
- **The feasibility graphs are dense, not sparse**: edge density ranges
  40-95% of all possible pairs (median ~70%). Most pairings of the pool
  work fine; greedy just grabbed a bad one and threw the whole round away.
- **The killing pair's degrees are almost always well above zero**
  (e.g. `[9,8]`, `[7,10]`, `[10,10]`) - the cluster that killed the round
  typically had plenty of other options that greedy never looked at.
- Of the 38 rescuable failures, **20 (53%) demonstrably need a genuinely
  different partner** - re-testing the exact same pair the real run tried
  also failed on census. This isn't just "retry and get lucky on a
  different rotation"; a different pairing choice is what's needed.

Raw data: `benchmark_results/pairing_frustration_probe.json`.

## What this means

The retrospective's "frustration is fundamental, no search strategy helps"
conclusion is correct for everything it tested - but pairing *choice* was
never in scope, and it turns out to be the actual lever. Round sizes are
small (~11 clusters), failures cluster overwhelmingly at round 1, and the
feasibility graph is dense enough that a smarter pairing strategy - a real
matching instead of greedy first-fit, or backtracking to a different
partner when a merge fails instead of aborting the whole attempt - has real,
cheap leverage on hard-regime success rates.

This connects directly to prior work: Tamer Areij's bachelor thesis
{cite:p}`Areij2026Bachelorarbeit`, which ran a 1512-run full-factorial
stability sweep on an earlier PyFracVAL version, explicitly flagged in its
outlook that a closer look at *why* specific cluster combinations fail
during CCA - "genauere Betrachtung der CCA-Fehlschläge... [könnten] später
möglicherweise Verbesserungen am Algorithmus oder bessere Wiederholungs-
und Abbruchstrategien entstehen" (a closer look at CCA failures... could
lead to algorithm improvements or better retry/abort strategies) - was left
for future work. This page is that follow-up.

See [hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md) for a
follow-up sweep mapping the Df/kf/σ/N boundary around the hard regime on
the current (unfixed) implementation - the baseline a future pairing-choice
fix would be compared against.

## Not yet done

This page documents a diagnosis, not a fix. `_generate_pairs`' greedy
first-fit is unchanged; no matching-based or backtracking pairing strategy
has been implemented or benchmarked end-to-end yet. That's the natural next
step this finding opens up.
