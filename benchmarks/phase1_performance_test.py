#!/usr/bin/env python3
"""
Phase 1 Optimization Performance Test

Measures runtime with Phase 1 optimizations applied.
Compare against commit be672c1 (before optimization) for baseline.

Optimizations tested:
1. Early termination + bounding sphere pre-checks
2. Vectorized PCA candidate selection
3. TRACE logging overhead removal
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyfracval.main_runner import run_simulation


def test_configuration(config, trials=5):
    """Test a single configuration multiple times."""
    runtimes = []
    successes = 0

    for trial in range(trials):
        seed = 1000 + trial
        start = time.time()
        try:
            success, coords, radii = run_simulation(
                1, config, output_base_dir="/tmp/phase1_perf", seed=seed
            )
            runtime = time.time() - start

            if success:
                successes += 1
                runtimes.append(runtime)
        except Exception as e:
            print(f"  Trial {trial} failed: {e}")
            continue

    if successes > 0:
        avg_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)
        return {
            "success_rate": successes / trials,
            "avg_runtime": avg_runtime,
            "std_runtime": std_runtime,
            "all_runtimes": runtimes,
        }
    else:
        return {
            "success_rate": 0,
            "avg_runtime": None,
            "std_runtime": None,
            "all_runtimes": [],
        }


def main():
    """Run performance tests."""
    print("=" * 80)
    print("Phase 1 Optimization Performance Test")
    print("=" * 80)
    print()

    # Test configurations (representative sample)
    configs = [
        {
            "name": "Optimal (N=128)",
            "params": {"N": 128, "Df": 1.8, "kf": 1.0, "rp_gstd": 1.3},
        },
        {
            "name": "Optimal (N=256)",
            "params": {"N": 256, "Df": 1.9, "kf": 1.2, "rp_gstd": 1.3},
        },
        {
            "name": "Low Df (N=128)",
            "params": {"N": 128, "Df": 1.5, "kf": 1.5, "rp_gstd": 1.3},
        },
        {
            "name": "High Df (N=128)",
            "params": {"N": 128, "Df": 2.2, "kf": 0.8, "rp_gstd": 1.3},
        },
    ]

    results = []

    for config_def in configs:
        name = config_def["name"]
        params = config_def["params"]

        print(f"\nTesting: {name}")
        print(
            f"  N={params['N']}, Df={params['Df']}, kf={params['kf']}, sigma={params['rp_gstd']}"
        )

        # Build full config
        full_config = {
            **params,
            "rp_g": 100.0,
            "tol_ov": 1e-6,
            "n_subcl_percentage": 0.1,
            "ext_case": 0,
        }

        result = test_configuration(full_config, trials=5)

        if result["success_rate"] > 0:
            print(f"  Success: {result['success_rate'] * 100:.0f}%")
            print(
                f"  Avg runtime: {result['avg_runtime']:.3f}s ± {result['std_runtime']:.3f}s"
            )
        else:
            print(f"  Failed all trials")

        results.append(
            {
                "name": name,
                "config": params,
                **result,
            }
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful_results = [r for r in results if r["success_rate"] > 0]
    if successful_results:
        print(f"\nSuccessful configurations: {len(successful_results)}/{len(results)}")
        print(f"\nRuntime statistics:")
        for r in successful_results:
            print(f"  {r['name']}: {r['avg_runtime']:.3f}s ± {r['std_runtime']:.3f}s")

    # Save results
    output_file = Path("benchmark_results/phase1_performance.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "optimizations": [
                        "Early termination + bounding sphere pre-checks",
                        "Vectorized PCA candidate selection",
                        "TRACE logging overhead removal",
                    ],
                    "commit": "3064cb3",
                    "baseline_commit": "be672c1",
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nTo measure speedup, compare against baseline (commit be672c1):")
    print(f"  git checkout be672c1")
    print(f"  uv run python benchmarks/phase1_performance_test.py")


if __name__ == "__main__":
    main()
