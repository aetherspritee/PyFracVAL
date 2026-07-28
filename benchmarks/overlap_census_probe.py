#!/usr/bin/env python3
"""Gather overlap-failure severity data across hard/easy regimes using the
opt-in cca_overlap_census_enabled flag, to answer: when CCA sticking
fails, how many particles are actually implicated, and how severely?

Reuses StickingBenchmark.run_single_trial() directly (no subclassing
needed - unlike pairing_frustration_probe.py, the census here is a real
production opt-in flag, not an offline diagnostic wrapper), with the
retry-inclusive run_simulation() path (the metric users actually see, via
--max-attempts), across the same hard/easy regimes
pairing_frustration_probe.py uses.

Usage:
    devenv shell -- uv run python benchmarks/overlap_census_probe.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sticking_benchmark import StickingBenchmark  # noqa: E402

N_SEEDS = 40
RESULTS_DIR = Path("benchmark_results")

REGIMES = {
    "hard": dict(N=128, Df=2.25, kf=0.95, rp_gstd=1.9, description="hard_regime"),
    "easy_control": dict(
        N=128, Df=1.8, kf=1.0, rp_gstd=1.5, description="easy_control"
    ),
}


def run_regime(name: str, params: dict) -> dict:
    benchmark = StickingBenchmark(output_dir=str(RESULTS_DIR / "overlap_census_probe"))
    params = {**params, "cca_overlap_census_enabled": True}

    results = []
    for seed in range(1, N_SEEDS + 1):
        result = benchmark.run_single_trial(
            params, trial_num=seed, category=f"overlap_census_{name}", seed=seed
        )
        results.append(result)
        status = "success" if result.success else "failed"
        n_off = None
        if result.overlap_census is not None:
            n_off = (
                result.overlap_census["n_particles_cluster1_offending"]
                + result.overlap_census["n_particles_cluster2_offending"]
            )
        print(f"  [{name}] seed {seed}/{N_SEEDS}: {status} (n_offending={n_off})")

    failures_with_census = [
        r for r in results if not r.success and r.overlap_census is not None
    ]
    n_success = sum(1 for r in results if r.success)

    offending_totals = [
        r.overlap_census["n_particles_cluster1_offending"]
        + r.overlap_census["n_particles_cluster2_offending"]
        for r in failures_with_census
    ]
    cluster_sizes = [
        r.overlap_census["cluster1_size"] + r.overlap_census["cluster2_size"]
        for r in failures_with_census
    ]
    # How many failures were "localized" - a small, fixed number of
    # offending particles - vs. involving a large fraction of both
    # clusters? Buckets chosen to span the range this project's own
    # "~5 out of 512" framing for a future drop-rescue budget suggests.
    localized_buckets = Counter()
    for n_off in offending_totals:
        if n_off <= 5:
            localized_buckets["<=5"] += 1
        elif n_off <= 10:
            localized_buckets["6-10"] += 1
        elif n_off <= 20:
            localized_buckets["11-20"] += 1
        else:
            localized_buckets[">20"] += 1

    severity_totals = Counter()
    for r in failures_with_census:
        for bucket, count in r.overlap_census["severity_histogram"].items():
            severity_totals[bucket] += count

    summary = {
        "regime": name,
        "params": params,
        "n_seeds": N_SEEDS,
        "n_success": n_success,
        "success_rate": n_success / N_SEEDS,
        "n_failures_with_census": len(failures_with_census),
        "offending_particle_counts": offending_totals,
        "cluster_size_at_failure": cluster_sizes,
        "localized_buckets": dict(localized_buckets),
        "severity_histogram_totals": dict(severity_totals),
        "mean_offending_particles": (
            sum(offending_totals) / len(offending_totals) if offending_totals else None
        ),
        "median_offending_particles": (
            sorted(offending_totals)[len(offending_totals) // 2]
            if offending_totals
            else None
        ),
    }
    print(
        f"[{name}] success_rate={summary['success_rate']:.1%} "
        f"failures_censused={len(failures_with_census)} "
        f"localized_buckets={dict(localized_buckets)}"
    )
    return summary


def main() -> None:
    print("=" * 80)
    print("Overlap census probe")
    print("=" * 80)

    summaries = [run_regime(name, params) for name, params in REGIMES.items()]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "overlap_census_probe.json"
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
