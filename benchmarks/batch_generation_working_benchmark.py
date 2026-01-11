#!/usr/bin/env python3
"""
Working Batch Generation Benchmark

Demonstrates batch parallelization speedup with N=128 where it reliably works.
N=256+ has issues with multiprocessing (see BATCH_PARALLELIZATION_ANALYSIS.md).
"""

import json
import subprocess
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyfracval.batch_runner import (
    generate_aggregates_parallel,
    generate_aggregates_sequential,
)


def main():
    """Run working batch generation benchmark."""
    print("=" * 80)
    print("BATCH GENERATION BENCHMARK (WORKING)")
    print("Using N=128 (reliable multiprocessing performance)")
    print("=" * 80)

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except:
        commit = "unknown"

    n_cores = cpu_count()
    print(f"Current commit: {commit}")
    print(f"Available CPU cores: {n_cores}")
    print()

    # Working configuration
    config = {
        "N": 128,  # N=128 works reliably
        "Df": 1.8,
        "kf": 1.0,
        "rp_g": 100.0,
        "rp_gstd": 1.3,
        "tol_ov": 1e-6,
        "n_subcl_percentage": 0.1,
        "ext_case": 0,
    }

    n_aggregates = 20
    print(f"Generating {n_aggregates} aggregates with N={config['N']} particles")
    print("=" * 80)

    results = {}

    # Sequential baseline
    print("\n--- Sequential (1 worker) ---")
    start = time.time()
    seq_results = generate_aggregates_sequential(
        n_aggregates=n_aggregates,
        config=config,
        output_base_dir="/tmp/batch_bench_working_seq",
        seed_start=11000,
    )
    seq_time = time.time() - start

    seq_success = sum(1 for success, _, _ in seq_results if success)

    results["sequential"] = {
        "n_workers": 1,
        "n_aggregates": n_aggregates,
        "successes": seq_success,
        "total_time": seq_time,
        "avg_time_per_aggregate": seq_time / n_aggregates,
        "throughput": n_aggregates / seq_time,
        "speedup": 1.0,
    }

    print(
        f"\nSequential: {seq_time:.2f}s total ({seq_time / n_aggregates:.2f}s per agg)"
    )

    # Parallel with 2, 4, and 8 workers
    for n_workers in [2, 4, 8]:
        print(f"\n--- Parallel ({n_workers} workers) ---")
        start = time.time()
        par_results = generate_aggregates_parallel(
            n_aggregates=n_aggregates,
            config=config,
            output_base_dir=f"/tmp/batch_bench_working_par_{n_workers}",
            seed_start=11000,
            n_workers=n_workers,
            show_progress=True,
        )
        par_time = time.time() - start

        par_success = sum(1 for success, _, _ in par_results if success)
        speedup = seq_time / par_time

        results[f"parallel_{n_workers}"] = {
            "n_workers": n_workers,
            "n_aggregates": n_aggregates,
            "successes": par_success,
            "total_time": par_time,
            "avg_time_per_aggregate": par_time / n_aggregates,
            "throughput": n_aggregates / par_time,
            "speedup": speedup,
        }

        print(
            f"Parallel ({n_workers}): {par_time:.2f}s total "
            f"({par_time / n_aggregates:.2f}s per agg), "
            f"{speedup:.2f}x speedup"
        )

    # Save results
    output_file = Path(f"benchmark_results/batch_generation_working_{commit}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "commit": commit,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "optimization": "Batch aggregate generation (N=128 working config)",
                "cpu_cores": n_cores,
                "config": config,
                "n_aggregates": n_aggregates,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Per Agg (s)':<13} {'Speedup':<10}")
    print("-" * 55)

    for config_name, data in sorted(results.items(), key=lambda x: x[1]["n_workers"]):
        workers = data["n_workers"]
        total_time = data["total_time"]
        per_agg = data["avg_time_per_aggregate"]
        speedup = data["speedup"]

        print(f"{workers:<10} {total_time:<12.2f} {per_agg:<13.2f} {speedup:<10.2f}x")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Check if we got speedup
    speedup_2 = results.get("parallel_2", {}).get("speedup", 0)
    speedup_4 = results.get("parallel_4", {}).get("speedup", 0)
    speedup_8 = results.get("parallel_8", {}).get("speedup", 0)

    if speedup_2 >= 1.5:
        print("✅ SUCCESS! Batch parallelization achieves speedup for N=128!")
        print(f"   2 workers: {speedup_2:.2f}x speedup")
        print(f"   4 workers: {speedup_4:.2f}x speedup")
        print(f"   8 workers: {speedup_8:.2f}x speedup")
        print()
        print("Why it works:")
        print(
            "   - N=128: ~0.5-1s per aggregate (long enough to amortize spawn overhead)"
        )
        print("   - Independent tasks (no dependencies)")
        print("   - Each worker uses optimized Phase 1 sequential code")
        print()
        print("Recommendation:")
        print("   For N≤128 and batches of 20+, use batch parallelization!")
        print("   Expected: 2-4x speedup on 4-8 cores")
        print()
        print("⚠️  Note: N=256+ currently has issues with multiprocessing")
        print("   Use sequential generation for larger aggregates")
    else:
        print("⚠️  Speedup less than expected")
        print(f"   2 workers: {speedup_2:.2f}x")
        print(f"   4 workers: {speedup_4:.2f}x")
        print(f"   8 workers: {speedup_8:.2f}x")

    print("=" * 80)


if __name__ == "__main__":
    main()
