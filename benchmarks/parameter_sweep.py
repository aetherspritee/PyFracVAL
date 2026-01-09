#!/usr/bin/env python3
"""
Parameter Space Sweep for PyFracVAL

Tests combinations of (Df, kf, sigma_p_geo) to identify:
1. Optimal regions with high success rates
2. Impossible regions where no kf helps
3. Heuristics for parameter selection

Usage:
    uv run python benchmarks/parameter_sweep.py [--quick]
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyfracval.main_runner import run_simulation

logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s - %(message)s"
)  # Suppress debug logs


@dataclass
class SweepResult:
    """Result from a single parameter combination test."""

    Df: float
    kf: float
    sigma_p_geo: float
    N: int
    trials: int
    successes: int
    success_rate: float
    avg_runtime: float
    seeds_used: list[int]


class ParameterSweep:
    """Systematic parameter space exploration."""

    def __init__(
        self,
        output_dir: str = "benchmark_results/parameter_sweep",
        quick_mode: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_mode = quick_mode

    def define_sweep_grid(self) -> dict:
        """Define parameter ranges to sweep."""
        if self.quick_mode:
            # Quick mode: fewer points for testing
            return {
                "Df": np.linspace(1.6, 2.4, 5),  # 5 points
                "kf": np.linspace(0.6, 1.8, 7),  # 7 points
                "sigma_p_geo": [1.3, 1.5, 2.0],  # 3 levels
                "N": 128,  # Fixed size
                "trials_per_combo": 3,  # 3 seeds per combination
            }
        else:
            # Full sweep: comprehensive coverage
            return {
                "Df": np.linspace(1.5, 2.5, 11),  # 11 points
                "kf": np.linspace(0.5, 2.0, 16),  # 16 points
                "sigma_p_geo": [1.3, 1.5, 2.0],  # 3 levels
                "N": 128,  # Fixed size
                "trials_per_combo": 5,  # 5 seeds per combination
            }

    def run_single_trial(self, config: dict, seed: int) -> tuple[bool, float]:
        """Run a single simulation trial."""
        start = time.time()
        try:
            success, coords, radii = run_simulation(1, config, seed=seed)
            runtime = time.time() - start
            return success, runtime
        except Exception as e:
            runtime = time.time() - start
            logging.error(f"Trial failed with exception: {e}")
            return False, runtime

    def test_parameter_combination(
        self, Df: float, kf: float, sigma: float, N: int, trials: int
    ) -> SweepResult:
        """Test a specific parameter combination with multiple seeds."""
        successes = 0
        runtimes = []
        seeds = []

        # Generate random seeds for this combination
        np.random.seed(int((Df * 1000 + kf * 100 + sigma * 10) % 2**31))
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

            success, runtime = self.run_single_trial(config, int(seed))
            if success:
                successes += 1
            runtimes.append(runtime)
            seeds.append(int(seed))

        success_rate = successes / trials if trials > 0 else 0.0
        avg_runtime = np.mean(runtimes) if runtimes else 0.0

        return SweepResult(
            Df=Df,
            kf=kf,
            sigma_p_geo=sigma,
            N=N,
            trials=trials,
            successes=successes,
            success_rate=success_rate,
            avg_runtime=avg_runtime,
            seeds_used=seeds,
        )

    def run_sweep(self) -> list[SweepResult]:
        """Execute the full parameter sweep."""
        grid = self.define_sweep_grid()
        results = []

        total_combos = len(grid["Df"]) * len(grid["kf"]) * len(grid["sigma_p_geo"])
        current = 0

        print(f"Parameter Sweep Configuration:")
        print(
            f"  Df range: {grid['Df'][0]:.2f} to {grid['Df'][-1]:.2f} ({len(grid['Df'])} points)"
        )
        print(
            f"  kf range: {grid['kf'][0]:.2f} to {grid['kf'][-1]:.2f} ({len(grid['kf'])} points)"
        )
        print(f"  sigma_p_geo: {grid['sigma_p_geo']}")
        print(f"  N: {grid['N']}")
        print(f"  Trials per combination: {grid['trials_per_combo']}")
        print(f"  Total combinations: {total_combos}")
        print(f"  Total trials: {total_combos * grid['trials_per_combo']}")
        print(
            f"  Estimated time: ~{(total_combos * grid['trials_per_combo']) // 60} minutes"
        )
        print()

        start_time = time.time()

        for sigma in grid["sigma_p_geo"]:
            print(f"\n{'=' * 60}")
            print(f"Testing sigma_p_geo = {sigma}")
            print(f"{'=' * 60}\n")

            for Df in grid["Df"]:
                for kf in grid["kf"]:
                    current += 1
                    print(
                        f"[{current}/{total_combos}] Testing Df={Df:.2f}, kf={kf:.2f}, sigma={sigma:.2f}...",
                        end=" ",
                        flush=True,
                    )

                    result = self.test_parameter_combination(
                        Df, kf, sigma, grid["N"], grid["trials_per_combo"]
                    )
                    results.append(result)

                    # Print result
                    success_pct = result.success_rate * 100
                    status = (
                        "✓" if success_pct >= 80 else "⚠" if success_pct >= 50 else "✗"
                    )
                    print(
                        f"{status} {result.successes}/{result.trials} ({success_pct:.0f}%)"
                    )

        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Sweep completed in {elapsed / 60:.1f} minutes")
        print(f"{'=' * 60}\n")

        return results

    def save_results(
        self, results: list[SweepResult], filename: str = "sweep_results.json"
    ):
        """Save sweep results to JSON."""
        output_path = self.output_dir / filename
        data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_combinations": len(results),
                "quick_mode": self.quick_mode,
            },
            "results": [asdict(r) for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {output_path}")
        return output_path

    def analyze_results(self, results: list[SweepResult]):
        """Analyze and summarize sweep results."""
        print("\n" + "=" * 60)
        print("PARAMETER SWEEP ANALYSIS")
        print("=" * 60)

        # Group by sigma
        for sigma in sorted(set(r.sigma_p_geo for r in results)):
            sigma_results = [r for r in results if r.sigma_p_geo == sigma]

            print(f"\n{'=' * 60}")
            print(f"sigma_p_geo = {sigma}")
            print(f"{'=' * 60}")

            # Best combinations
            best = sorted(sigma_results, key=lambda x: x.success_rate, reverse=True)[:5]
            print(f"\nTop 5 Best Combinations:")
            for i, r in enumerate(best, 1):
                print(
                    f"  {i}. Df={r.Df:.2f}, kf={r.kf:.2f}: "
                    f"{r.success_rate * 100:.0f}% ({r.successes}/{r.trials})"
                )

            # Worst combinations
            worst = sorted(sigma_results, key=lambda x: x.success_rate)[:5]
            print(f"\nWorst 5 Combinations:")
            for i, r in enumerate(worst, 1):
                print(
                    f"  {i}. Df={r.Df:.2f}, kf={r.kf:.2f}: "
                    f"{r.success_rate * 100:.0f}% ({r.successes}/{r.trials})"
                )

            # Identify impossible Df values (no kf helps)
            print(f"\nImpossible Df Analysis:")
            df_values = sorted(set(r.Df for r in sigma_results))
            for df in df_values:
                df_results = [r for r in sigma_results if r.Df == df]
                max_success = max(r.success_rate for r in df_results)
                best_kf = next(
                    r.kf for r in df_results if r.success_rate == max_success
                )

                if max_success < 0.3:  # Less than 30% success with any kf
                    print(
                        f"  ⛔ Df={df:.2f}: IMPOSSIBLE (best: {max_success * 100:.0f}% with kf={best_kf:.2f})"
                    )
                elif max_success < 0.7:  # Less than 70% success
                    print(
                        f"  ⚠️  Df={df:.2f}: DIFFICULT (best: {max_success * 100:.0f}% with kf={best_kf:.2f})"
                    )
                else:
                    print(
                        f"  ✅ Df={df:.2f}: FEASIBLE (best: {max_success * 100:.0f}% with kf={best_kf:.2f})"
                    )

            # Recommended kf ranges for each Df
            print(f"\nRecommended kf Ranges (success >= 80%):")
            for df in df_values:
                df_results = [r for r in sigma_results if r.Df == df]
                good_results = [r for r in df_results if r.success_rate >= 0.8]

                if good_results:
                    kf_values = sorted([r.kf for r in good_results])
                    kf_min, kf_max = min(kf_values), max(kf_values)
                    print(f"  Df={df:.2f}: kf ∈ [{kf_min:.2f}, {kf_max:.2f}]")
                else:
                    print(f"  Df={df:.2f}: No reliable kf found")

        # Overall statistics
        print(f"\n{'=' * 60}")
        print(f"Overall Statistics:")
        print(f"{'=' * 60}")

        total_trials = sum(r.trials for r in results)
        total_successes = sum(r.successes for r in results)
        overall_success = total_successes / total_trials * 100

        print(f"Total trials: {total_trials}")
        print(f"Total successes: {total_successes}")
        print(f"Overall success rate: {overall_success:.1f}%")

        excellent = len([r for r in results if r.success_rate >= 0.9])
        good = len([r for r in results if 0.7 <= r.success_rate < 0.9])
        poor = len([r for r in results if 0.3 <= r.success_rate < 0.7])
        failed = len([r for r in results if r.success_rate < 0.3])

        print(f"\nCombination Quality:")
        print(f"  Excellent (≥90%): {excellent}")
        print(f"  Good (70-90%): {good}")
        print(f"  Poor (30-70%): {poor}")
        print(f"  Failed (<30%): {failed}")


def main():
    """Main entry point."""
    import sys

    quick_mode = "--quick" in sys.argv

    print("PyFracVAL Parameter Space Sweep")
    print("=" * 60)
    print()

    if quick_mode:
        print("⚡ QUICK MODE: Testing reduced grid")
    else:
        print("🔬 FULL MODE: Comprehensive parameter sweep")

    print()

    sweep = ParameterSweep(quick_mode=quick_mode)
    results = sweep.run_sweep()

    # Save results
    filename = "sweep_results_quick.json" if quick_mode else "sweep_results.json"
    sweep.save_results(results, filename)

    # Analyze
    sweep.analyze_results(results)

    print("\n✅ Parameter sweep complete!")


if __name__ == "__main__":
    main()
