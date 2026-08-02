# Failure Statistics: the Structured Event Log

Ordinary logging records whether an individual run worked. Aggregate
analysis — over thousands of runs, where generation fails, by what
mechanism, and how severely, sliced by the physical parameters (`Df`,
`kf`, `σ_p,geo`, `N`) — requires structured records: free-text log lines
cannot be aggregated, and the in-memory `diagnostics` dict accepted by
`run_simulation` is per-call and never persisted.

`pyfracval/event_log.py` writes one JSONL file serving this purpose,
enabled by a single config key:

```toml
event_log_path = "benchmark_results/events.jsonl"
```

Nothing is written and no file is opened when the key is unset. Setting
it also enables the overlap census automatically, since the failure
records exist chiefly to answer "how many particles overlap, and how
badly", and the census is the instrument that measures it.

## Record kinds

Three record kinds share one file, each stamped with the same run
context so they can be sliced or joined:

| Kind | One per | Records |
|---|---|---|
| `merge` | CCA merge attempt | round, cluster pair, Γ, search consumed, outcome, particles overlapping at give-up |
| `pca_failure` | PCA subcluster that could not be built | failing particle index, candidate count, search/swap attempts used |
| `run` | aggregate generation attempt | outcome, failure stage and reason, attempts used, wall time, final geometry quality |

Every record carries `run_id`, `pid`, and the simulation parameters, so
a sweep can write all its workers to one path and remain separable.

## Compression

A `.gz` suffix gzips the log as it is written:

```toml
event_log_path = "benchmark_results/events.jsonl.gz"
```

This matters at sweep scale. A merge record is ~640 bytes, of which
~180 is the run context re-serialized verbatim on every line, and the
4200-trial boundary sweep below emitted 523,812 of them for 323 MB.
Measured on those records, gzip level 6 gives **~9.5x** — the difference
between copying 330 MB and 35 MB off a cluster — with no loss of
records or precision.

A compressed log cannot use the atomic single-line append that lets many
processes share one plain file: a gzip stream has to stay open across
records, and a buffered stream cannot be shared. So a compressed log
writes **one shard per process** — `events.jsonl.gz` becomes
`events.pid1234.jsonl.gz`. Within a process, every `EventLog` on the same
path shares one stream, so the thousands of per-trial instances a sweep
creates still produce a single well-compressed file per worker. Pass the
shards, or the directory holding them, to the analyzer.

A run killed mid-write leaves its final gzip member without a trailer;
the analyzer reports the truncation and keeps every record decoded
before it.

`pca_failure` is a distinct kind because PCA and CCA fail by different
mechanisms, and a taxonomy that cannot distinguish them is of limited
use. PCA failures occur while growing a single subcluster particle by
particle — typically because no already-placed particle sits at a
workable distance for the next monomer's Γ — and were previously
visible only as free text.

## Two overlap denominators

The codebase measures overlap in two conventions, which differ by a
large factor for wide size distributions:

| Field | Denominator | Meaning |
|---|---|---|
| `max_overlap_of_rsum` | `r_i + r_j` | the convention used by `tol_ov`, `overlap.py` and `quality.py` — comparable to the configured tolerance |
| `max_overlap_of_rmin` | `min(r_i, r_j)` | penetration depth of the smaller particle (`densify.py`'s convention) |

Before these were split into separate fields, the census reported only
the `min(r)` form under the bare name "overlap fraction" while the same
record's `min_overlap` used the `r_sum` form — two silently different
scales in one row. Measured on real failures, the two medians are 0.618
and 1.785, nearly a factor of three apart on the same pairs; the field
names now state the denominator explicitly.

## Analysis

```
devenv shell -- uv run python benchmarks/analyze_event_log.py events.jsonl
devenv shell -- uv run python benchmarks/analyze_event_log.py events.jsonl --by Df rp_gstd
devenv shell -- uv run python benchmarks/analyze_event_log.py RUNDIR   # all shards
```

## Text logs during a sweep

The structured log replaces, rather than accompanies, free-text failure
output. `benchmarks/stability_sweep.py` therefore configures the
`pyfracval` logger at **ERROR** by default (`--log-level` to override),
including inside Dask workers, which are separate processes that would
otherwise fall through to Python's `logging.lastResort` handler and print
every WARNING and ERROR unformatted to stderr.

That fallback is what produced the ~100 MB `run.log` files next to the
older sweeps: 1.3M lines, 94% of them three PCA retry messages, none of
them read and all of them already recorded structurally.

Applied to the full boundary sweep — the same 4200-trial grid as
[boundary_sweep_v2.md](boundary_sweep_v2.md), re-run with logging on
(`configs/boundary_sweep_v3_eventlog.toml`, 523,812 records, 323 MB):

```
RUNS: where does generation fail?
  total 4164   success 3437 ( 82.5%)
    failed at TIMEOUT           420   10.1%
    failed at CCA               307    7.4%
  successful runs with invalid geometry: 0  (  0.0%)
  Rg error vs scaling law (%): min=-2.60 median=+0.58 max=+10.68
  wall time per run (s): min=0.04 median=1.03 max=157.84

PCA FAILURES: none recorded

CCA MERGES: attempts 519648   stuck 84150 ( 16.2%)
    failed_overlap             435372   83.8%
    failed_no_candidates          126    0.0%
   round    stuck   failed  fail rate
       1    49224   188662      79.3%
       2    21197   148276      87.5%
       3     7976    70437      89.8%
       4     4409    22180      83.4%
       7       52      124      70.5%
      10        6        0       0.0%
  merges that needed a later partner (backtracking): 24547  ( 29.2% of stuck)

OVERLAP AT FAILURE (226,578 censused failures)
  offending particles       : min=2 median=13 max=132
  cluster-pair size         : min=10 median=36 max=1024
  offending fraction        : min=0.02 median=0.36 max=1.00
  worst overlap / (ri+rj)   : median=0.639   <- comparable to tol_ov
  worst overlap / min(ri,rj): median=1.804   <- penetration of smaller particle
  best overlap reached      : median=2.327e-01
  failures with <=10% of the pair offending: 2445  (  1.1%)
```

Sliced, the two axes that matter:

| σ | runs | success | merge fail | med offending | med fraction |
|---|---:|---:|---:|---:|---:|
| 1.0 | 1400 | 93.5% | 34.2% | 18 | 0.21 |
| 1.5 | 1400 | 83.2% | 75.8% | 11 | 0.33 |
| 1.9 | 1364 | 70.6% | 92.3% | 14 | 0.38 |

| Df | runs | success | merge fail | med offending | med fraction |
|---|---:|---:|---:|---:|---:|
| 1.8 | 525 | 100.0% | 0.0% | 7 | 0.23 |
| 2.0 | 525 | 100.0% | 1.3% | 7 | 0.42 |
| 2.1 | 525 | 100.0% | 11.0% | 12 | 0.29 |
| 2.2 | 525 | 94.7% | 69.9% | 17 | 0.30 |
| 2.3 | 514 | 80.0% | 86.5% | 15 | 0.34 |
| 2.4 | 514 | 54.7% | 89.2% | 13 | 0.36 |
| 2.5 | 511 | 29.0% | 89.7% | 12 | 0.38 |

## Findings

**Generation fails in CCA, essentially never in PCA.** Across 4200
trials spanning the full grid — including σ=1.9 and Df=2.5 — there were
**zero** PCA failure records. Every failure is a CCA merge failure or a
timeout. This is worth stating precisely because PCA *can* fail (the
mechanism exists and is now instrumented; a σ=3.0 probe triggers it), it
simply does not within the parameter range anyone sweeps. Effort spent on
the PCA stage is effort spent on a non-problem.

**Timeouts are the largest single failure category**, at 10.1% against
7.4% for genuine CCA failure. These are runs that hit the 120 s per-trial
budget rather than exhausting their retries, so a portion of them are
"too slow" rather than "impossible". Any success rate quoted from a
timeout-bounded sweep is a statement about the budget as much as the
algorithm — see the caveat below.

**Failures are not near-misses.** The best overlap a failing merge
reaches has median 0.233 against a `tol_ov` of 1e-6 — five orders of
magnitude. No refinement of the rotation search closes that, which is
consistent with every search-strategy experiment in
[experiments.md](experiments.md) coming out flat.

**Failures are not localized.** Only **1.1%** of 226,578 censused
failures have 10% or less of the cluster pair offending; the median is
36%. This is the premise the drop-rescue mechanism rests on, measured
now at full scale, and it is quantitatively false — agreeing with
[drop_rescue.md](drop_rescue.md), which reaches the same conclusion by an
independent route.

**Difficulty peaks at round 3, it does not decline monotonically.** An
earlier 18-run sample on this page suggested a monotonic fall with round;
the full sweep shows failure rate *rising* from 79.3% (round 1) to 89.8%
(round 3) before falling away. Round 1 merges fresh PCA subclusters,
which are small and easy to place; the squeeze comes a couple of rounds
later, when clusters are large enough to be awkward but the pool is still
big enough that most pairings are attempted. Backtracking still applies —
it operates within whichever round is difficult.

**No successful run produced invalid geometry**, across 3437 aggregates.
That is the overlap-acceptance and densify fixes holding at scale, and it
is the check that would have caught the catalog overlap leak had it
existed then.

## Caveat: timeouts make a bounded sweep non-deterministic

The same grid and seeds gave 80.3% (v2, no logging) and 81.8% (v3, with
logging). Neither the log nor the census consumes randomness, so the
difference is not the instrumentation changing the algorithm: it is that
`trial_timeout` is wall-clock, so a trial near the budget can land either
side of it depending on machine load. Quoting a success rate from a
timeout-bounded sweep therefore carries a machine-dependent component,
and comparisons between sweeps should either use the same hardware and
load or drop the timeout.

Note also that 4164 run records were written for 4200 trials. The 36
missing are failures whose worker was killed by the sweep's own timeout
before the record could be written; all 3437 successes are present, and
the run-level totals agree with the sweep's independent summary.

## Cost

The census runs once per failed merge, off the hot path, and costs one
full non-early-exit pairwise scan of the two clusters. Attaching a log
also makes the sticking loop compute a true (non-early-exit) overlap
per exhausted candidate, so that the recorded `min_overlap` is exact
rather than the incremental scan's lower bound. Both costs are pure
overhead when no statistics are being collected, which is why they are
gated on the log being attached.
