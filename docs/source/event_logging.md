# Failure Statistics: the Structured Event Log

Ordinary logging answers "did this run work". A paper needs something
harder: over thousands of runs, **where** generation fails, **why**, and
**how badly** — sliceable by the physics (`Df`, `kf`, `σ_p,geo`, `N`).
Free-text log lines cannot be aggregated, and the in-memory `diagnostics`
dict `run_simulation` accepts is per-call and never persisted.

`pyfracval/event_log.py` writes one JSONL file that answers those
questions. Enable it with a single config key:

```toml
event_log_path = "benchmark_results/events.jsonl"
```

Nothing is written and no file is opened when it is unset. Setting it
also switches the overlap census on automatically — "how many particles
overlap" is the question the failure records mainly exist to answer, and
the census is what measures it.

## Record kinds

Three kinds share one file, each stamped with the same run context so
they can be sliced or joined together.

| Kind | One per | Answers |
|---|---|---|
| `merge` | CCA merge attempt | which round, which cluster pair, at what Γ, how much search was consumed, outcome, and how many particles overlapped at give-up |
| `pca_failure` | PCA subcluster that could not be built | at which particle index, with how many candidates, after how many search/swap attempts |
| `run` | aggregate generation attempt | outcome, failure stage and reason, attempts used, wall time, final geometry quality |

Every record carries `run_id`, `pid`, and the simulation parameters, so a
sweep can write all its workers to one path and still be separable.

`pca_failure` exists because PCA and CCA fail by different mechanisms and
a taxonomy that cannot tell them apart is not much use. PCA failures
happen while growing a *single* subcluster particle by particle — usually
because no already-placed particle sits at a workable distance for the
next monomer's Γ — and were previously invisible except as free text.

## Two overlap denominators, never conflated

This codebase measures overlap two ways, and they differ by a large
factor for wide size distributions:

| Field | Denominator | Meaning |
|---|---|---|
| `max_overlap_of_rsum` | `r_i + r_j` | the convention `tol_ov`, `overlap.py` and `quality.py` use — **the one comparable to the configured tolerance** |
| `max_overlap_of_rmin` | `min(r_i, r_j)` | how deeply the *smaller* particle is penetrated (`densify.py`'s convention) |

Before this was split, the census reported only the `min(r)` form under
the bare name "overlap fraction" while the same record's `min_overlap`
used the `r_sum` form. Two silently different scales in one row is a trap
for anyone building a table from it. Measured on real failures the two
medians are 0.618 and 1.785 — nearly 3× apart on the same pairs.

## Analysis

```
devenv shell -- uv run python benchmarks/analyze_event_log.py events.jsonl
devenv shell -- uv run python benchmarks/analyze_event_log.py events.jsonl --by Df rp_gstd
```

Sample output over 18 runs spanning an easy regime, a hard one, and a
wide-σ one:

```
RUNS: where does generation fail?
  total 18   success 12 ( 66.7%)
    failed at CCA                 6   33.3%
  successful runs with invalid geometry: 0  (  0.0%)
  Rg error vs scaling law (%): min=-1.44 median=-0.16 max=+1.32

PCA FAILURES: why can a subcluster not be built?
  total 1
    no_candidates                   1  100.0%
  failed at particle index: min=2 median=2 max=2

CCA MERGES: why do sticking attempts fail?
   round    stuck   failed  fail rate
       1      322     3727      92.0%
       2      148     2476      94.4%
       ...
       6        2        5      71.4%
  merges that needed a later partner (backtracking): 244  ( 41.4% of stuck)

OVERLAP AT FAILURE: how many particles, and how badly?
  censused failures: 4960 of 7270  ( 68.2%)
  offending particles       : min=2 median=11 max=52
  cluster-pair size         : min=15 median=24 max=96
  offending fraction        : min=0.07 median=0.45 max=1.00
  worst overlap / (ri+rj)   : median=0.618   <- comparable to tol_ov
  worst overlap / min(ri,rj): median=1.785   <- penetration of smaller particle
  best overlap reached      : median=2.514e-01
  failures with <=10% of the pair offending: 15  (  0.3%)

SLICED BY Df, rp_gstd
       Df rp_gstd   runs  success  merge fail  med offend  med frac
      1.8     1.5      6   100.0%        0.0%           -         -
      2.2     3.0      6   100.0%       70.8%           9      0.44
      2.4     1.9      6     0.0%       97.3%          11      0.46
```

## What the numbers already say

Three results fall straight out of this and are worth stating, because
each closes a question that was previously argued rather than measured.

**Failures are not near-misses.** The best overlap a failing merge
reaches has median ~0.25 against a `tol_ov` of 1e-6. No refinement of the
rotation search closes a gap five orders of magnitude wide, which is
consistent with every search-strategy experiment in
[experiments.md](experiments.md) coming out flat.

**Failures are not localized.** Only **0.3%** of censused failures have
10% or less of the cluster pair offending; the median is 45%. This is the
premise drop-rescue rests on, and it is quantitatively false at every
regime measured — see [drop_rescue.md](drop_rescue.md), which reaches the
same conclusion from the other direction.

**Difficulty concentrates in the early rounds.** Merge failure rate falls
monotonically with round (92% at round 1 down to 71% by round 6), which
is what makes backtracking effective: it operates *within* a round, where
the difficulty is.

Note the wide-σ row above: 100% of runs succeed despite 70.8% of
individual merges failing. Run-level and merge-level success are
different questions, and separating them is much of the point — retry and
backtracking convert a low per-merge probability into a high per-run one.

## Cost

The census runs once per failed merge, not on the hot path, and costs one
full non-early-exit pairwise scan of the two clusters. Attaching a log
also makes the sticking loop compute a true (non-early-exit) overlap per
exhausted candidate so `min_overlap` is honest rather than the
incremental scan's lower bound — pure overhead for anyone not collecting
statistics, which is why it is gated on the log being attached.
