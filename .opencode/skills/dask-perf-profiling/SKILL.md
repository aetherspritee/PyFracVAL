---
name: dask-perf-profiling
description: >
  Reproducible performance and profiling workflow for PyFracVAL on local and
  remote Dask clusters. Use when benchmarking speedups, identifying bottlenecks,
  comparing environments, or validating distributed scaling behavior.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: performance
---

## What I do

- Run apples-to-apples local vs remote Dask benchmark sessions
- Collect actionable profiling artifacts (task stream, worker usage, scheduler behavior)
- Normalize throughput for fair comparisons across heterogeneous workers
- Classify likely bottleneck type and recommend the highest-impact next step

## When to use me

Use this skill when:

- A speed optimization improves local performance but not remote Dask performance
- You need a shared procedure for performance validation before/after code changes
- You want to understand if runtime is compute-bound, scheduler-bound, transfer-bound, or straggler-bound
- You need comparable benchmark evidence across multiple machines

## Inputs to gather first

- Scheduler endpoint (for remote runs), e.g. `tcp://host:8786`
- Test matrix: `N` values (typically `128,256` for quick profiling)
- Fixed seeds and fixed config parameters
- Number of aggregates per run and warmup task count
- Local worker count and expected remote worker layout

## Standard protocol

1. Preflight
   - Confirm worker visibility and metadata (hostname, threads, Python version)
   - Confirm package deployment path (latest wheel install if required)
   - Note version mismatch warnings; they are often relevant to performance

2. Baseline run (local)
   - Run benchmark with fixed config + seeds
   - Include warmup phase and measured phase
   - Capture JSON outputs and profiling artifacts

3. Remote run (same config)
   - Same seeds and aggregate count as local
   - Same warmup policy
   - Capture JSON outputs and profiling artifacts

4. Extract and compare metrics
   - Raw throughput (`agg/s`)
   - Throughput per thread
   - Throughput per effective thread (weighted by worker calibration)
   - Task wall/cpu stats, worker busy fraction, success rate

5. Classify bottleneck
   - Compute-bound: workers mostly busy, high CPU/wall, low scheduler overhead
   - Scheduler-bound: many short tasks, high wait/dispatch overhead
   - Transfer-bound: high data movement + idle gaps awaiting payloads
   - Straggler-bound: one/few workers consistently lagging

6. Recommend next action
   - Task granularity changes (batch/chunk sizing)
   - Worker targeting or heterogeneity handling
   - Serialization/payload reduction
   - Algorithm hotspot optimization

## Profiling artifacts to capture

- Benchmark JSON summary per run
- Dask performance report (HTML)
- Task stream trace / timeline
- Worker metadata snapshot
- Environment fingerprint (package version, Python, key dependency versions)

Suggested directory layout:

```text
benchmark_results/profiles/<timestamp>/
  local/
    summary.json
    performance_report.html
  remote/
    summary.json
    performance_report.html
  comparison.json
  notes.md
```

## Normalization policy

Use two normalized views side-by-side:

1. Simple normalization
   - `throughput_per_thread = throughput / total_threads`

2. Heterogeneity-aware normalization
   - Compute per-worker calibration score (median of repeated calibration runs)
   - Derive effective threads by weighting thread counts by calibration index
   - Use conservative clipping for outliers to avoid single-worker distortion

Do not rely on one metric alone. Keep raw throughput as a first-class output.

## Interpretation rubric

- If remote raw throughput is lower but per-thread is similar:
  likely fixed orchestration overhead dominates at small N
- If remote raw throughput is higher but busy fraction is low:
  capacity is underutilized; scaling headroom exists with larger workload
- If per-effective-thread collapses unexpectedly:
  calibration likely unstable or cluster heterogeneity dominates; rerun calibration with more repeats
- If success rate differs across environments:
  treat correctness/stability as the first bottleneck before performance

## Common pitfalls

- Comparing different seed sets between runs
- Comparing cold-start runs to warm runs
- Tiny workload sizes producing noisy metrics
- Ignoring version mismatch warnings between client/scheduler/workers
- Treating loopback cluster tests as true remote behavior

## Reporting template

Use this minimal report structure:

1. Run context
   - commit, config, seeds, worker topology
2. Metric table
   - local vs remote raw + normalized metrics
3. Bottleneck diagnosis
   - primary, secondary, confidence
4. Recommended actions
   - top 3, ordered by expected impact

## Exit criteria

A profiling pass is complete when:

- Local and remote runs used identical benchmark inputs
- Artifacts are saved and readable
- Bottleneck classification is justified by at least 3 independent signals
- Next optimization action is explicit and testable
