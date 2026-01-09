#!/usr/bin/env python3
"""
Large Aggregate Test

Tests whether optimal parameter combinations from N=128 sweeps
scale to larger aggregate sizes (N=256, 512, 1024).

Usage:
    uv run python benchmarks/large_aggregate_test.py
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyfracval.main_runner import run_simulation

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")


@dataclass
class LargeAggregateResult:
    """Result from a single large aggregate test."""

    N: int
    Df: float
    kf: float
    sigma_p_geo: float
    trials: int
    successes: int
    success_rate: float
    avg_runtime: float
    successful_seeds: list[int]


class LargeAggregateBenchmark:
    """Test optimal parameters on larger aggregates."""

    def __init__(self, output_dir: str = "benchmark_results/large_aggregates"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_optimal_combinations(self) -> list[dict]:
        """
        Return optimal parameter combinations discovered from quick sweep.

        These had 100% success rate with N=128, sigma=1.3
        """
        return [
            {"Df": 1.6, "kf": 1.2, "sigma": 1.3, "name": "Loose fractal"},
            {"Df": 1.8, "kf": 1.0, "sigma": 1.3, "name": "Medium-loose fractal"},
            {"Df": 2.0, "kf": 1.0, "sigma": 1.3, "name": "Medium fractal"},
            {"Df": 2.2, "kf": 0.8, "sigma": 1.3, "name": "Medium-dense fractal"},
            {"Df": 2.4, "kf": 0.6, "sigma": 1.3, "name": "Dense fractal"},
        ]

    def test_single_configuration(
        self, N: int, Df: float, kf: float, sigma: float, trials: int = 5
    ) -> LargeAggregateResult:
        """Test a single configuration with multiple seeds."""
        successes = 0
        runtimes = []
        successful_seeds = []

        # Generate deterministic seeds
        np.random.seed(int((N * 10000 + Df * 1000 + kf * 100) % 2**31))
        test_seeds = np.random.randint(0, 2**31, size=trials)

        for seed in test_seeds:
            config = {
                "N": N,
                "Df": Df,
                "kf": kf,
                "rp_g": 100.0,
                "rp_gstd": sigma,
                "tol_ov": 1e-6,
                "n_subcl_percentage": 0.1,
                "ext_case": 0,
                "seed": int(seed),
            }

            start = time.time()
            try:
                success, coords, radii = run_simulation(
                    1, config, output_base_dir="/tmp/large_agg_test", seed=int(seed)
                )
                runtime = time.time() - start

                if success:
                    successes += 1
                    successful_seeds.append(int(seed))
                runtimes.append(runtime)

            except Exception as e:
                runtime = time.time() - start
                logging.error(f"Trial failed with exception: {e}")
                runtimes.append(runtime)

        success_rate = successes / trials if trials > 0 else 0.0
        avg_runtime = np.mean(runtimes) if runtimes else 0.0

        return LargeAggregateResult(
            N=N,
            Df=Df,
            kf=kf,
            sigma_p_geo=sigma,
            trials=trials,
            successes=successes,
            success_rate=success_rate,
            avg_runtime=avg_runtime,
            successful_seeds=successful_seeds,
        )

    def run_benchmark(self, N_values: list[int] = None) -> list[LargeAggregateResult]:
        """Run benchmark across different N values."""
        if N_values is None:
            N_values = [128, 256, 512, 1024]

        optimal_combos = self.get_optimal_combinations()
        results = []

        total_tests = len(N_values) * len(optimal_combos)
        current = 0

        print(f"Large Aggregate Benchmark")
        print(f"=" * 80)
        print(f"Testing {len(optimal_combos)} optimal combinations")
        print(f"Across N values: {N_values}")
        print(f"Total configurations: {total_tests}")
        print(f"Trials per configuration: 5")
        print(f"Total trials: {total_tests * 5}")
        print()

        start_time = time.time()

        for N in N_values:
            print(f"\n{'=' * 80}")
            print(f"Testing N = {N}")
            print(f"{'=' * 80}\n")

            for combo in optimal_combos:
                current += 1
                print(
                    f"[{current}/{total_tests}] N={N}, Df={combo['Df']:.1f}, "
                    f"kf={combo['kf']:.1f} ({combo['name']})...",
                    end=" ",
                    flush=True,
                )

                result = self.test_single_configuration(
                    N, combo["Df"], combo["kf"], combo["sigma"], trials=5
                )
                results.append(result)

                # Print result
                success_pct = result.success_rate * 100
                status = "✓" if success_pct >= 80 else "⚠" if success_pct >= 40 else "✗"
                print(
                    f"{status} {result.successes}/{result.trials} "
                    f"({success_pct:.0f}%) - {result.avg_runtime:.1f}s avg"
                )

        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"Benchmark completed in {elapsed / 60:.1f} minutes")
        print(f"{'=' * 80}\n")

        return results

    def save_results(
        self,
        results: list[LargeAggregateResult],
        filename: str = "large_agg_results.json",
    ):
        """Save benchmark results to JSON."""
        output_path = self.output_dir / filename
        data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_configurations": len(results),
                "description": "Testing optimal N=128 parameters on larger aggregates",
            },
            "results": [asdict(r) for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {output_path}")
        return output_path

    def analyze_results(self, results: list[LargeAggregateResult]):
        """Analyze and summarize benchmark results."""
        print("\n" + "=" * 80)
        print("LARGE AGGREGATE ANALYSIS")
        print("=" * 80)

        # Group by N
        N_values = sorted(set(r.N for r in results))

        for N in N_values:
            n_results = [r for r in results if r.N == N]
            print(f"\n{'=' * 80}")
            print(f"N = {N} particles")
            print(f"{'=' * 80}")

            # Overall stats
            total_trials = sum(r.trials for r in n_results)
            total_successes = sum(r.successes for r in n_results)
            overall_success = (
                total_successes / total_trials * 100 if total_trials > 0 else 0
            )

            avg_runtime = np.mean([r.avg_runtime for r in n_results])

            print(
                f"\nOverall: {total_successes}/{total_trials} ({overall_success:.1f}%)"
            )
            print(f"Average runtime: {avg_runtime:.1f}s")

            # Per-configuration breakdown
            print(f"\nPer-configuration results:")
            for r in n_results:
                status = (
                    "✓"
                    if r.success_rate >= 0.8
                    else "⚠"
                    if r.success_rate >= 0.4
                    else "✗"
                )
                print(
                    f"  {status} Df={r.Df:.1f}, kf={r.kf:.1f}: "
                    f"{r.successes}/{r.trials} ({r.success_rate * 100:.0f}%) - "
                    f"{r.avg_runtime:.1f}s avg"
                )

        # Scalability analysis
        print(f"\n{'=' * 80}")
        print("Scalability Analysis")
        print(f"{'=' * 80}")

        # Group by Df/kf combo and track across N
        combos = {}
        for r in results:
            key = (r.Df, r.kf)
            if key not in combos:
                combos[key] = []
            combos[key].append((r.N, r.success_rate, r.avg_runtime))

        for (Df, kf), data in sorted(combos.items()):
            print(f"\nDf={Df:.1f}, kf={kf:.1f}:")
            for N, sr, rt in sorted(data):
                status = "✓" if sr >= 0.8 else "⚠" if sr >= 0.4 else "✗"
                print(f"  {status} N={N:4d}: {sr * 100:5.0f}% success, {rt:6.1f}s avg")


def main():
    """Main entry point."""
    print("PyFracVAL Large Aggregate Scaling Test")
    print("=" * 80)
    print()
    print("Testing optimal parameters from N=128 sweep on larger aggregates")
    print()

    # Test N=128 (baseline), 256, 512, and 1024
    benchmark = LargeAggregateBenchmark()
    results = benchmark.run_benchmark(N_values=[128, 256, 512, 1024])

    # Save results
    benchmark.save_results(results)

    # Analyze
    benchmark.analyze_results(results)

    print("\n✅ Large aggregate benchmark complete!")


if __name__ == "__main__":
    main()
