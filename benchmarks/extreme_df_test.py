#!/usr/bin/env python3
"""
Extreme Df Boundary Testing

Tests algorithm limits at low Df (1.3-1.5) and high Df (2.5-2.9) values.
Uses optimal sigma=1.3 to isolate Df effects.

Usage:
    uv run python benchmarks/extreme_df_test.py
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
class ExtremeDfResult:
    """Result from testing an extreme Df value."""

    Df: float
    kf: float
    kf_strategy: str
    sigma_p_geo: float
    N: int
    trials: int
    successes: int
    success_rate: float
    avg_runtime: float
    successful_seeds: list[int]
    failure_modes: dict


class ExtremeDfBenchmark:
    """Test extreme Df values to find algorithmic boundaries."""

    def __init__(self, output_dir: str = "benchmark_results/extreme_df"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_kf_from_relationship(self, Df: float) -> float:
        """
        Calculate optimal kf using empirical relationship from sweep:
        kf_optimal ≈ 3.0 - 1.0 * Df
        """
        return 3.0 - 1.0 * Df

    def get_extreme_df_test_matrix(self) -> list[dict]:
        """
        Define test matrix for extreme Df values.

        Tests both:
        1. Empirical relationship kf
        2. Fixed kf values to see if relationship holds at extremes
        """
        test_cases = []

        # Low Df regime (approaching linear chain)
        low_df_values = [1.3, 1.35, 1.4, 1.45, 1.5]

        # High Df regime (approaching solid sphere)
        high_df_values = [2.5, 2.6, 2.7, 2.8, 2.9]

        # Test each Df with multiple kf strategies
        for df in low_df_values:
            # Empirical relationship
            kf_empirical = self.calculate_kf_from_relationship(df)
            test_cases.append(
                {
                    "Df": df,
                    "kf": kf_empirical,
                    "strategy": "empirical",
                    "regime": "low_Df",
                }
            )

            # Also test with kf=1.0 to see divergence
            test_cases.append(
                {"Df": df, "kf": 1.0, "strategy": "fixed_1.0", "regime": "low_Df"}
            )

        for df in high_df_values:
            # Empirical relationship
            kf_empirical = self.calculate_kf_from_relationship(df)

            # Clamp to reasonable range (kf should be > 0)
            if kf_empirical <= 0:
                kf_empirical = 0.3  # Minimum reasonable kf

            test_cases.append(
                {
                    "Df": df,
                    "kf": kf_empirical,
                    "strategy": "empirical",
                    "regime": "high_Df",
                }
            )

            # Also test with kf=1.0
            test_cases.append(
                {"Df": df, "kf": 1.0, "strategy": "fixed_1.0", "regime": "high_Df"}
            )

        return test_cases

    def test_single_configuration(
        self, Df: float, kf: float, strategy: str, trials: int = 5
    ) -> ExtremeDfResult:
        """Test a single Df/kf combination."""
        successes = 0
        runtimes = []
        successful_seeds = []
        failure_modes = {"pca": 0, "cca": 0, "exception": 0}

        # Generate deterministic seeds
        np.random.seed(int((Df * 1000 + kf * 100) % 2**31))
        test_seeds = np.random.randint(0, 2**31, size=trials)

        for seed in test_seeds:
            config = {
                "N": 128,
                "Df": Df,
                "kf": kf,
                "rp_g": 100.0,
                "rp_gstd": 1.3,  # Use optimal sigma to isolate Df effects
                "tol_ov": 1e-6,
                "n_subcl_percentage": 0.1,
                "ext_case": 0,
                "seed": int(seed),
            }

            start = time.time()
            try:
                success, coords, radii = run_simulation(
                    1, config, output_base_dir="/tmp/extreme_df_test", seed=int(seed)
                )
                runtime = time.time() - start

                if success:
                    successes += 1
                    successful_seeds.append(int(seed))
                else:
                    # Failure - try to categorize
                    # Fast failure (< 1s) likely PCA, slow failure likely CCA
                    if runtime < 1.0:
                        failure_modes["pca"] += 1
                    else:
                        failure_modes["cca"] += 1

                runtimes.append(runtime)

            except Exception as e:
                runtime = time.time() - start
                failure_modes["exception"] += 1
                runtimes.append(runtime)
                logging.debug(f"Exception for Df={Df}, kf={kf}, seed={seed}: {e}")

        success_rate = successes / trials if trials > 0 else 0.0
        avg_runtime = np.mean(runtimes) if runtimes else 0.0

        return ExtremeDfResult(
            Df=Df,
            kf=kf,
            kf_strategy=strategy,
            sigma_p_geo=1.3,
            N=128,
            trials=trials,
            successes=successes,
            success_rate=success_rate,
            avg_runtime=avg_runtime,
            successful_seeds=successful_seeds,
            failure_modes=failure_modes,
        )

    def run_benchmark(self) -> list[ExtremeDfResult]:
        """Run the extreme Df benchmark."""
        test_matrix = self.get_extreme_df_test_matrix()
        results = []

        total = len(test_matrix)
        current = 0

        print(f"Extreme Df Boundary Testing")
        print(f"=" * 80)
        print(f"Testing {total} configurations")
        print(f"Trials per configuration: 5")
        print(f"Total trials: {total * 5}")
        print(f"Sigma fixed at 1.3 (optimal)")
        print()

        start_time = time.time()

        # Group by regime
        low_df_tests = [t for t in test_matrix if t["regime"] == "low_Df"]
        high_df_tests = [t for t in test_matrix if t["regime"] == "high_Df"]

        print(f"\n{'=' * 80}")
        print(f"LOW Df REGIME (1.3 - 1.5) - Testing {len(low_df_tests)} configs")
        print(f"{'=' * 80}\n")

        for test in low_df_tests:
            current += 1
            print(
                f"[{current}/{total}] Df={test['Df']:.2f}, kf={test['kf']:.2f} "
                f"({test['strategy']})...",
                end=" ",
                flush=True,
            )

            result = self.test_single_configuration(
                test["Df"], test["kf"], test["strategy"], trials=5
            )
            results.append(result)

            success_pct = result.success_rate * 100
            status = "✓" if success_pct >= 80 else "⚠" if success_pct >= 40 else "✗"
            print(f"{status} {result.successes}/5 ({success_pct:.0f}%)")

        print(f"\n{'=' * 80}")
        print(f"HIGH Df REGIME (2.5 - 2.9) - Testing {len(high_df_tests)} configs")
        print(f"{'=' * 80}\n")

        for test in high_df_tests:
            current += 1
            print(
                f"[{current}/{total}] Df={test['Df']:.2f}, kf={test['kf']:.2f} "
                f"({test['strategy']})...",
                end=" ",
                flush=True,
            )

            result = self.test_single_configuration(
                test["Df"], test["kf"], test["strategy"], trials=5
            )
            results.append(result)

            success_pct = result.success_rate * 100
            status = "✓" if success_pct >= 80 else "⚠" if success_pct >= 40 else "✗"
            print(f"{status} {result.successes}/5 ({success_pct:.0f}%)")

        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"Benchmark completed in {elapsed / 60:.1f} minutes")
        print(f"{'=' * 80}\n")

        return results

    def save_results(
        self, results: list[ExtremeDfResult], filename: str = "extreme_df_results.json"
    ):
        """Save results to JSON."""
        output_path = self.output_dir / filename
        data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_configurations": len(results),
                "description": "Testing extreme Df values to find algorithmic boundaries",
                "sigma_fixed": 1.3,
                "N_fixed": 128,
            },
            "results": [asdict(r) for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {output_path}")
        return output_path

    def analyze_results(self, results: list[ExtremeDfResult]):
        """Analyze and display results."""
        print("\n" + "=" * 80)
        print("EXTREME Df ANALYSIS")
        print("=" * 80)

        # Low Df regime
        low_df_results = [r for r in results if r.Df < 2.0]
        print(f"\n{'=' * 80}")
        print(f"LOW Df REGIME (< 2.0)")
        print(f"{'=' * 80}")

        # Group by Df value
        df_values_low = sorted(set(r.Df for r in low_df_results))
        for df in df_values_low:
            df_tests = [r for r in low_df_results if r.Df == df]
            print(f"\nDf = {df:.2f}")
            for r in df_tests:
                status = (
                    "✓"
                    if r.success_rate >= 0.8
                    else "⚠"
                    if r.success_rate >= 0.4
                    else "✗"
                )
                print(
                    f"  {status} kf={r.kf:.2f} ({r.kf_strategy}): "
                    f"{r.successes}/{r.trials} ({r.success_rate * 100:.0f}%) - "
                    f"{r.avg_runtime:.1f}s avg"
                )
                if r.failure_modes["pca"] + r.failure_modes["cca"] > 0:
                    print(
                        f"     Failures: PCA={r.failure_modes['pca']}, "
                        f"CCA={r.failure_modes['cca']}, "
                        f"Exception={r.failure_modes['exception']}"
                    )

        # High Df regime
        high_df_results = [r for r in results if r.Df >= 2.5]
        print(f"\n{'=' * 80}")
        print(f"HIGH Df REGIME (≥ 2.5)")
        print(f"{'=' * 80}")

        df_values_high = sorted(set(r.Df for r in high_df_results))
        for df in df_values_high:
            df_tests = [r for r in high_df_results if r.Df == df]
            print(f"\nDf = {df:.2f}")
            for r in df_tests:
                status = (
                    "✓"
                    if r.success_rate >= 0.8
                    else "⚠"
                    if r.success_rate >= 0.4
                    else "✗"
                )
                print(
                    f"  {status} kf={r.kf:.2f} ({r.kf_strategy}): "
                    f"{r.successes}/{r.trials} ({r.success_rate * 100:.0f}%) - "
                    f"{r.avg_runtime:.1f}s avg"
                )
                if r.failure_modes["pca"] + r.failure_modes["cca"] > 0:
                    print(
                        f"     Failures: PCA={r.failure_modes['pca']}, "
                        f"CCA={r.failure_modes['cca']}, "
                        f"Exception={r.failure_modes['exception']}"
                    )

        # Key findings
        print(f"\n{'=' * 80}")
        print("KEY FINDINGS")
        print(f"{'=' * 80}")

        # Find boundaries
        successful_dfs = [r.Df for r in results if r.success_rate >= 0.8]
        failed_dfs = [r.Df for r in results if r.success_rate == 0]

        if successful_dfs:
            print(f"\nLowest successful Df: {min(successful_dfs):.2f}")
            print(f"Highest successful Df: {max(successful_dfs):.2f}")

        if failed_dfs:
            print(f"\nLowest failed Df: {min(failed_dfs):.2f}")
            print(f"Highest failed Df: {max(failed_dfs):.2f}")

        # Test if empirical relationship holds
        empirical_results = [r for r in results if r.kf_strategy == "empirical"]
        fixed_results = [r for r in results if r.kf_strategy == "fixed_1.0"]

        print(f"\n{'=' * 80}")
        print("EMPIRICAL RELATIONSHIP TEST (kf = 3.0 - Df)")
        print(f"{'=' * 80}")

        empirical_success = sum(1 for r in empirical_results if r.success_rate >= 0.8)
        fixed_success = sum(1 for r in fixed_results if r.success_rate >= 0.8)

        print(
            f"Empirical kf strategy: {empirical_success}/{len(empirical_results)} "
            f"({empirical_success / len(empirical_results) * 100:.0f}% reliable)"
        )
        print(
            f"Fixed kf=1.0 strategy: {fixed_success}/{len(fixed_results)} "
            f"({fixed_success / len(fixed_results) * 100:.0f}% reliable)"
        )

        print(
            f"\nConclusion: Empirical relationship "
            f"{'HOLDS' if empirical_success > fixed_success else 'BREAKS'} "
            f"at extreme Df values"
        )


def main():
    """Main entry point."""
    print("PyFracVAL Extreme Df Boundary Testing")
    print("=" * 80)
    print("Finding algorithmic limits at low and high Df values")
    print()

    benchmark = ExtremeDfBenchmark()
    results = benchmark.run_benchmark()

    # Save results
    benchmark.save_results(results)

    # Analyze
    benchmark.analyze_results(results)

    print("\n✅ Extreme Df benchmark complete!")


if __name__ == "__main__":
    main()
