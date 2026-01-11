#!/usr/bin/env python3
"""
Realistic Batch Generation Benchmark

Tests with N=256 (1-2s per aggregate) where multiprocessing overhead should
be amortized and we should see actual speedup.

Theory:
- N=64: 0.1s per aggregate → overhead dominates → slower
- N=256: 1-2s per aggregate → overhead amortized → faster!
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
    """Run realistic batch generation benchmark."""
    print("=" * 80)
    print("BATCH GENERATION BENCHMARK (REALISTIC)")
    print("Using N=256 (1-2s per aggregate)")
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

    # Realistic configuration
    config = {
        "N": 256,  # Takes ~1-2s per aggregate
        "Df": 1.8,  # Use 1.8 instead of 1.9 to avoid PCA failures
        "kf": 1.0,  # Use 1.0 instead of 1.2 to avoid PCA failures
        "rp_g": 100.0,
        "rp_gstd": 1.3,
        "tol_ov": 1e-6,
        "n_subcl_percentage": 0.1,
        "ext_case": 0,
    }

    n_aggregates = 16  # 16 aggregates = ~20-30s total
    print(f"Generating {n_aggregates} aggregates with N={config['N']} particles")
    print(f"Expected: ~1-2s per aggregate, ~{n_aggregates * 1.5:.0f}s sequential")
    print("=" * 80)

    results = {}

    # Sequential baseline
    print("\n--- Sequential (1 worker) ---")
    start = time.time()
    seq_results = generate_aggregates_sequential(
        n_aggregates=n_aggregates,
        config=config,
        output_base_dir="/tmp/batch_bench_seq_realistic",
        seed_start=4000,
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

    # Parallel with 4 and 8 workers
    for n_workers in [4, 8]:
        print(f"\n--- Parallel ({n_workers} workers) ---")
        start = time.time()
        par_results = generate_aggregates_parallel(
            n_aggregates=n_aggregates,
            config=config,
            output_base_dir=f"/tmp/batch_bench_par_realistic_{n_workers}",
            seed_start=4000,
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
    output_file = Path(f"benchmark_results/batch_generation_realistic_{commit}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "commit": commit,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "optimization": "Batch aggregate generation (realistic N=256)",
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
    speedup_4 = results.get("parallel_4", {}).get("speedup", 0)
    speedup_8 = results.get("parallel_8", {}).get("speedup", 0)

    if speedup_8 >= 2.0:
        print("✅ SUCCESS! Batch parallelization achieves speedup for realistic N!")
        print(f"   8 workers: {speedup_8:.2f}x speedup")
        print()
        print("Why it works now (vs N=64 failure):")
        print("   - N=256: ~1-2s per aggregate (not 0.1s)")
        print("   - Spawn overhead (~0.5s) amortized over long tasks")
        print("   - 16 tasks keep workers busy (2 tasks per worker)")
        print()
        print("Recommendation:")
        print("   For N≥256 and batches of 50+, use batch parallelization!")
        print("   Expected: 4-8x speedup on 8-16 cores")
    else:
        print("⚠️  Speedup less than expected")
        print(f"   4 workers: {speedup_4:.2f}x")
        print(f"   8 workers: {speedup_8:.2f}x")
        print()
        print("Possible reasons:")
        print("   - spawn() overhead still significant")
        print("   - Not enough tasks to keep all workers busy")
        print("   - System load or resource contention")
        print()
        print("Better results expected with:")
        print("   - More aggregates (50-100+)")
        print("   - Larger N (512+, 2-5s per aggregate)")

    print("=" * 80)


if __name__ == "__main__":
    main()
