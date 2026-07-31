# Df/kf/σ Stability Boundary Near the Hard Regime

[experiments.md](experiments.md) established a single hard-regime data
point (Df=2.25, kf=0.95, σ=1.9) where success rates collapse to
~17–20%, and [pairing_frustration.md](pairing_frustration.md) diagnosed
why single-shot attempts fail there. This page maps the Df/kf/σ/N
boundary around that point on the then-current implementation (greedy
first-fit pairing), forming the baseline against which the subsequent
pairing fix is measured ([boundary_sweep_v2.md](boundary_sweep_v2.md)).
It covers territory (kf < 1.0, σ > 1.5) that earlier stability
characterizations of this project did not reach.

## Method

`configs/hard_regime_boundary_sweep.toml`: Df ∈ [1.8, 2.5] step 0.1
(8), kf ∈ [0.8, 1.4] step 0.1 (7), σ ∈ {1.0, 1.5, 1.9} (3), N ∈ {64,
128, 256, 512, 1024} (5), 5 seeds per combination — 840 combinations,
4200 trials. Unlike `pairing_frustration_probe.py`'s single-shot
methodology, this sweep uses `run_simulation`'s standard internal retry
loop (up to 20 attempts per trial) via
`benchmarks/stability_sweep.py` — the same retry-inclusive metric
exposed to users via `--max-attempts`. Run on a local Dask cluster (16
cores); ~4200 trials in ~20–30 minutes wall clock.

Raw output:
`benchmark_results/hard_regime_boundary_sweep/stability_sweeps/`.

A caveat applies to the runtime columns in the raw data:
`stability_sweep.py`'s Dask path records each task's `submit_time` when
all 4200 tasks are enqueued up front, not when a worker begins
executing it, so `avg_runtime_s` and `median_runtime_s` in the summary
are dominated by queue-wait for tasks scheduled late in a
4200-task/16-worker batch rather than by per-trial cost (a
directly-timed single trial takes ~1 s in the easy region, ~16 s at the
hardest tested corner; see
[gpu_acceleration.md](gpu_acceleration.md) for the timing methodology).
Success-rate figures are unaffected; the timing columns in this sweep
are unreliable and were not corrected.

## Results

Boundary map at σ=1.9, success rate averaged over N=64..1024:

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

The same table at σ=1.5 and σ=1.0 (monodisperse):

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

(Df ≤ 2.1 is at or near 100% across all tested kf at every σ and is
omitted above; the full grid is in the raw JSON/CSV.)

Across the full grid, 3039/4200 trials (72.4%) succeeded. This figure
is not meaningful on its own — the grid deliberately spans well past
the boundary — but confirms the grid placement was neither uniformly
easy nor uniformly hard.

## Discussion

The collapse boundary shifts to lower Df as polydispersity increases:
safe up to Df≈2.3 monodisperse, Df≈2.2 at σ=1.5, and Df≈2.0 at σ=1.9.
This direction is consistent with the field literature: a survey
conducted alongside this sweep independently cites algorithmic collapse
around Df 2.2–2.3 for polydisperse rigid CCA, and an absolute
monodisperse ceiling of Df≈2.55 for size-symmetric merge strategies
matching FracVAL's design. Both figures agree with the direct
measurements here: at σ=1.0, kf=0.8, Df=2.5 still succeeds 92% of the
time, consistent with a ceiling near Df≈2.55.

The Df×kf interaction is sharp and directional: at every σ, lower kf
survives further into high-Df territory. At σ=1.9, Df=2.2, kf=0.8
remains at 100% while kf=1.1 has dropped to 16% — a transition spanning
a kf range of only 0.3.

The established hard regime sits close to the edge of this transition.
At N=128 (matching the probe in
[pairing_frustration.md](pairing_frustration.md)), Df=2.25/kf=0.95 is
bracketed by:

| Df | kf | success_rate (5 seeds, N=128, σ=1.9) |
|---:|---:|---:|
| 2.2 | 0.9 | 1.00 (5/5) |
| 2.2 | 1.0 | 0.60 (3/5) |
| 2.3 | 0.9 | 0.00 (0/5) |
| 2.3 | 1.0 | 0.00 (0/5) |

A fully-successful and a fully-collapsed corner sit 0.05 apart in Df.
The regime chosen in [experiments.md](experiments.md) (Df=2.25,
kf=0.95) is thus a deliberately hard stress point on this transition,
which is also why the pairing-frustration probe's single-shot
methodology measured only 2.5% success there: near the boundary, retry
compounds a low per-attempt probability into a substantially higher
eventual success rate, while the per-attempt probability itself is what
the probe's census explains.

N amplifies instability specifically at the boundary. Two
representative near-boundary points:

| Df | kf | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---:|---:|---:|---:|---:|---:|---:|
| 2.2 | 1.0 | 1.00 | 0.60 | 0.60 | 0.40 | 0.00 |
| 2.3 | 0.8 | 1.00 | 1.00 | 0.40 | 0.40 | 0.00 |

Points comfortably inside the safe region (e.g. Df=2.1, kf=1.0, σ=1.9)
show no such degradation: 100% at every tested N from 64 to 1024. N
does not independently cause failure; it sharpens whatever margin
Df/kf/σ leaves.

## Implications

This sweep is the greedy-pairing baseline. The pairing-frustration
diagnosis and the independent literature survey both identify CCA merge
ordering — rather than search strategy, already ruled out in
[experiments.md](experiments.md) — as the lever most likely to move
this boundary. The backtracking pairing fix was subsequently
benchmarked against this exact grid;
[boundary_sweep_v2.md](boundary_sweep_v2.md) quantifies the shift.
