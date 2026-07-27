# Mapping the Hard-Regime Boundary

[experiments.md](experiments.md) established a single "hard regime" data
point (Df=2.25, kf=0.95, σ=1.9) where success rates collapse to ~17-20%.
[pairing_frustration.md](pairing_frustration.md) diagnosed *why* single-shot
attempts fail there. This page maps the actual Df/kf/σ/N boundary around
that point on the *current* implementation - a baseline for whatever
pairing-choice fix follows, and a direct extension of prior work: Tamer
Areij's bachelor thesis {cite:p}`Areij2026Bachelorarbeit` ran a 1512-run
full-factorial sweep on an earlier PyFracVAL version but only covered
kf ≥ 1.0 and σ ≤ 1.5 - both bounds our hard regime sits outside of.

## Grid

`configs/hard_regime_boundary_sweep.toml`: Df ∈ [1.8, 2.5] step 0.1 (8),
kf ∈ [0.8, 1.4] step 0.1 (7), σ ∈ {1.0, 1.5, 1.9} (3), N ∈ {64, 128, 256,
512, 1024} (5), 5 seeds/combo → 840 combinations, 4200 trials. Unlike
`pairing_frustration_probe.py`'s single-shot methodology, this uses
`run_simulation`'s standard internal retry loop (up to 20 attempts per
trial) via `benchmarks/stability_sweep.py` - the same metric the thesis
used, and what `--max-attempts`/the CLI actually give a user in practice.
Local Dask (16 cores), ~4200 trials/~20-30 min wall clock.

Raw output: `benchmark_results/hard_regime_boundary_sweep/stability_sweeps/`.

**A caveat on the runtime columns in the raw data**: `stability_sweep.py`'s
Dask path records each task's `submit_time` when *all* 4200 tasks are
enqueued up front, not when a worker actually starts executing it - so
`avg_runtime_s`/`median_runtime_s` in the summary are dominated by
queue-wait for tasks scheduled late in a 4200-task/16-worker batch, not real
per-trial cost (a directly-timed single trial takes ~1s easy, ~16s for the
hardest tested corner - see [gpu_acceleration.md](gpu_acceleration.md)'s
timing methodology for the same kind of check). Success-rate numbers are
unaffected; only the timing columns are unreliable here. Not fixed in this
pass - noted for whoever next relies on those columns.

## The boundary (σ=1.9, success rate averaged over N=64..1024)

|  Df  | kf=0.8 | kf=0.9 | kf=1.0 | kf=1.1 | kf=1.2 | kf=1.3 | kf=1.4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1.9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 |
| 2.1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.32 | 0.08 |
| 2.2 | 1.00 | 0.92 | 0.52 | 0.16 | 0.00 | 0.00 | 0.00 |
| 2.3 | 0.56 | 0.12 | 0.04 | 0.00 | 0.04 | 0.00 | 0.00 |
| 2.4 | 0.16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2.5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Same table at σ=1.5 and σ=1.0 (monodisperse) for comparison:

| σ=1.5 | kf=0.8 | kf=0.9 | kf=1.0 | kf=1.1 | kf=1.2 | kf=1.3 | kf=1.4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 |
| 2.3 | 1.00 | 1.00 | 1.00 | 0.84 | 0.44 | 0.08 | 0.04 |
| 2.4 | 0.92 | 0.60 | 0.36 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2.5 | 0.40 | 0.08 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 |

| σ=1.0 | kf=0.8 | kf=0.9 | kf=1.0 | kf=1.1 | kf=1.2 | kf=1.3 | kf=1.4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2.4 | 1.00 | 1.00 | 1.00 | 0.88 | 0.68 | 0.56 | 0.36 |
| 2.5 | 0.92 | 0.68 | 0.60 | 0.52 | 0.20 | 0.04 | 0.00 |

(Df ≤ 2.1 is 100% or near-100% across all tested kf at every σ - omitted
above for space; full grid in the raw JSON/CSV.)

## Reading the map

**The collapse boundary shifts to lower Df as polydispersity increases** -
exactly the direction [the field literature predicts](pairing_frustration.md):
safe up to Df≈2.3 monodisperse, Df≈2.2 at σ=1.5, only Df≈2.0 at σ=1.9. A
Deep Research literature survey commissioned alongside this sweep
independently cites "algorithmic collapse around a fractal dimension of 2.2
to 2.3" for polydisperse rigid CCA, and an absolute monodisperse ceiling of
Df≈2.55 for size-symmetric merge strategies (matching FracVAL's design) -
both land almost exactly on what this sweep measures directly: at σ=1.0,
kf=0.8, Df=2.5 still succeeds 92% of the time, consistent with the ceiling
sitting a bit further out around Df≈2.55.

**The Df×kf interaction is sharp and directional, confirming the thesis's
qualitative finding with a much finer boundary**: at every σ, *lower* kf
survives further into high-Df territory than higher kf. At σ=1.9, Df=2.2:
kf=0.8 is still 100% while kf=1.1 has already dropped to 16%, a cliff over
a kf range of just 0.3.

**Our established hard regime sits almost exactly on the edge of this
cliff.** At N=128 (matching [pairing_frustration.md](pairing_frustration.md)'s
probe), Df=2.25/kf=0.95 is bracketed by:

| Df | kf | success_rate (5 seeds, N=128, σ=1.9) |
|---:|---:|---:|
| 2.2 | 0.9 | 1.00 (5/5) |
| 2.2 | 1.0 | 0.60 (3/5) |
| 2.3 | 0.9 | 0.00 (0/5) |
| 2.3 | 1.0 | 0.00 (0/5) |

A 100%-successful corner and a fully-collapsed corner, 0.05 apart in Df on
each side. This wasn't a coincidence of the original `experiments.md`
regime choice - it's a deliberately hard stress point sitting right on the
knife-edge, which is exactly why the pairing-frustration probe's *single-shot*
methodology (no internal retry) measured only 2.5% success there: retry
compounds a low per-attempt probability into a much higher eventual
success rate near the edge, but the per-attempt probability itself is what
the probe's census actually explains (see
[pairing_frustration.md](pairing_frustration.md)).

**N amplifies instability specifically at the boundary, confirming the
thesis's finding with a much cleaner signal** since this grid specifically
targets the edge rather than averaging over a wide safe region. Two
representative near-boundary points:

| Df | kf | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---:|---:|---:|---:|---:|---:|---:|
| 2.2 | 1.0 | 1.00 | 0.60 | 0.60 | 0.40 | 0.00 |
| 2.3 | 0.8 | 1.00 | 1.00 | 0.40 | 0.40 | 0.00 |

Points comfortably inside the safe region (e.g. Df=2.1, kf=1.0, σ=1.9) show
no such degradation - 100% at every tested N from 64 to 1024. N doesn't
independently cause failure; it sharpens whatever margin Df/kf/σ already
left.

## Overall

3039/4200 trials (72.4%) succeeded across the full grid - not a meaningful
number on its own since the grid deliberately spans well past the boundary
by design, but a sanity check that the grid placement was reasonable (not
uniformly easy or uniformly hard).

## What this sets up

This is the *current-implementation* baseline. The
[pairing-frustration diagnosis](pairing_frustration.md) and an independent
literature survey both point at CCA merge-ordering (not search strategy,
already ruled out in [experiments.md](experiments.md)) as the lever most
likely to move this boundary. A future pairing-choice fix - matching-based
pairing instead of greedy first-fit, or backtracking to a different partner
on merge failure - should be benchmarked against this exact grid to
quantify how far the boundary actually shifts.
