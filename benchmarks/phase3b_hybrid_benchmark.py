#!/usr/bin/env python3
"""
Phase 3B Hybrid Strategy Benchmark

Measures performance of sequential rotation + parallel overlap checking.
This combines the benefits of early termination (sequential rotation)
with parallelization of the expensive O(n) overlap checks.

Compares against Phase 1 baseline (all sequential).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyfracval import utils
from pyfracval.main_runner import run_simulation


def test_configuration(config, trials=3):
    """Test a single configuration multiple times."""
    runtimes = []
    successes = 0

    for trial in range(trials):
        seed = 4000 + trial
        start = time.time()
        try:
            success, coords, radii = run_simulation(
                1, config, output_base_dir="/tmp/phase3b_bench", seed=seed
            )
            runtime = time.time() - start

            if success:
                successes += 1
                runtimes.append(runtime)
        except Exception as e:
            print(f"    Trial {trial} failed: {e}")
            continue

    if successes > 0:
        return {
            "success_rate": successes / trials,
            "avg_runtime": float(np.mean(runtimes)),
            "std_runtime": float(np.std(runtimes)),
            "min_runtime": float(np.min(runtimes)),
            "max_runtime": float(np.max(runtimes)),
            "all_runtimes": [float(r) for r in runtimes],
        }
    else:
        return None


def main():
    """Run Phase 3B benchmark."""
    print("=" * 80)
    print("Phase 3B: Hybrid Strategy Benchmark")
    print("Sequential Rotation + Parallel Overlap")
    print("=" * 80)

    # Get current commit
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

    print(f"Current commit: {commit}")
    print(f"Parallel overlap threshold: {utils.PARALLEL_OVERLAP_THRESHOLD}")
    print()

    # Test configurations
    configs = [
        {
            "name": "N=128, Optimal",
            "N": 128,
            "Df": 1.8,
            "kf": 1.0,
            "rp_gstd": 1.3,
        },
        {
            "name": "N=256, Optimal",
            "N": 256,
            "Df": 1.9,
            "kf": 1.2,
            "rp_gstd": 1.3,
        },
        {
            "name": "N=512, Optimal",
            "N": 512,
            "Df": 1.9,
            "kf": 1.1,
            "rp_gstd": 1.3,
        },
    ]

    results = {}

    for config_def in configs:
        name = config_def.pop("name")

        print(f"Testing: {name}")
        print(f"  Parameters: {config_def}")

        # Build full config
        full_config = {
            **config_def,
            "rp_g": 100.0,
            "tol_ov": 1e-6,
            "n_subcl_percentage": 0.1,
            "ext_case": 0,
        }

        result = test_configuration(full_config, trials=3)

        if result:
            print(f"  Success: {result['success_rate'] * 100:.0f}%")
            print(
                f"  Runtime: {result['avg_runtime']:.3f}s ± {result['std_runtime']:.3f}s"
            )
            print(
                f"  Range: [{result['min_runtime']:.3f}s - {result['max_runtime']:.3f}s]"
            )
            results[name] = result
        else:
            print(f"  Failed all trials")
            results[name] = None

        print()

    # Save results
    output_file = Path(f"benchmark_results/phase3b_hybrid_{commit}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "commit": commit,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "optimization": "Sequential rotation + parallel overlap (auto-dispatch)",
                "parallel_threshold": utils.PARALLEL_OVERLAP_THRESHOLD,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Results saved to: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, result in results.items():
        if result:
            print(
                f"{name:25s} {result['avg_runtime']:6.3f}s ± {result['std_runtime']:5.3f}s"
            )

    # Load Phase 1 results for comparison
    phase1_file = Path("benchmark_results/phase_comparison_b8009c3.json")
    if phase1_file.exists():
        with open(phase1_file, "r") as f:
            phase1_data = json.load(f)

        print("\n" + "=" * 80)
        print("SPEEDUP vs PHASE 1 (Sequential)")
        print("=" * 80)

        for name, result in results.items():
            if result and name in phase1_data["results"]:
                phase1_time = phase1_data["results"][name]["avg_runtime"]
                phase3b_time = result["avg_runtime"]
                speedup = phase1_time / phase3b_time
                print(f"{name:25s} {speedup:.2f}x")

        print("\nNote: Speedup increases with N due to parallel overlap threshold")
        print(f"      (parallel used when n > {utils.PARALLEL_OVERLAP_THRESHOLD})")


if __name__ == "__main__":
    main()
