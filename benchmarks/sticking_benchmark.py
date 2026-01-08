"""Comprehensive benchmark suite for sticking process analysis.

This module provides systematic testing of the FracVAL sticking algorithm
across various parameter combinations to identify failure modes and
performance characteristics.

Usage:
    # Quick test (stable cases only)
    python benchmarks/sticking_benchmark.py

    # Run specific suite
    from benchmarks.sticking_benchmark import StickingBenchmark
    b = StickingBenchmark()
    b.run_suite('low_df', n_trials=10)

    # Run full benchmark
    b.run_all(n_trials=10)
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

from pyfracval.main_runner import run_simulation
from pyfracval.schemas import SimulationParameters


@dataclass
class BenchmarkResult:
    """Results from a single benchmark trial."""

    # Input parameters
    N: int
    Df: float
    kf: float
    rp_g: float
    rp_gstd: float
    tol_ov: float
    n_subcl_percentage: float
    ext_case: int
    seed: int
    description: str
    category: str

    # Output metrics
    success: bool
    runtime_seconds: float
    failure_stage: Optional[str] = None  # 'PCA' or 'CCA' or None
    failure_reason: Optional[str] = None  # Detailed error

    # Aggregate properties (if successful)
    final_N: Optional[int] = None
    final_Rg: Optional[float] = None

    # Performance metrics (if we add instrumentation)
    total_rotations_pca: Optional[int] = None
    total_rotations_cca: Optional[int] = None
    avg_rotations_per_particle: Optional[float] = None
    gamma_failures: Optional[int] = None
    candidate_failures: Optional[int] = None


class StickingBenchmark:
    """Comprehensive benchmark suite for sticking process analysis."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Define all test cases
        self.test_suites = {
            'stable': self._define_stable_cases(),
            'low_df': self._define_low_df_cases(),
            'high_df': self._define_high_df_cases(),
            'extreme_kf': self._define_extreme_kf_cases(),
            'polydisperse': self._define_polydisperse_cases(),
            'scaling': self._define_scaling_cases(),
            'corner': self._define_corner_cases(),
        }

    def _define_stable_cases(self) -> List[Dict]:
        """Known stable parameter combinations (baseline)."""
        return [
            {'N': 128, 'Df': 1.8, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Original paper example'},
            {'N': 128, 'Df': 2.0, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Default config'},
            {'N': 256, 'Df': 1.9, 'kf': 1.2, 'rp_gstd': 1.3, 'description': 'Moderate polydisperse'},
        ]

    def _define_low_df_cases(self) -> List[Dict]:
        """Low fractal dimension cases (known problematic)."""
        return [
            {'N': 128, 'Df': 1.5, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Lower bound'},
            {'N': 128, 'Df': 1.6, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Moderate low Df'},
            {'N': 128, 'Df': 1.7, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Threshold test'},
            {'N': 128, 'Df': 1.5, 'kf': 1.3, 'rp_gstd': 1.5, 'description': 'Low Df + high kf'},
            {'N': 256, 'Df': 1.6, 'kf': 1.0, 'rp_gstd': 1.2, 'description': 'Larger N, low Df'},
        ]

    def _define_high_df_cases(self) -> List[Dict]:
        """High fractal dimension cases (dense packing)."""
        return [
            {'N': 128, 'Df': 2.2, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Moderate high Df'},
            {'N': 128, 'Df': 2.4, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'High Df'},
            {'N': 128, 'Df': 2.5, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Upper bound'},
            {'N': 128, 'Df': 2.3, 'kf': 1.5, 'rp_gstd': 1.5, 'description': 'High Df + high kf'},
            {'N': 256, 'Df': 2.2, 'kf': 1.2, 'rp_gstd': 1.3, 'description': 'Larger N, high Df'},
        ]

    def _define_extreme_kf_cases(self) -> List[Dict]:
        """Extreme prefactor values."""
        return [
            {'N': 128, 'Df': 1.8, 'kf': 0.5, 'rp_gstd': 1.5, 'description': 'Very low kf'},
            {'N': 128, 'Df': 1.8, 'kf': 0.8, 'rp_gstd': 1.5, 'description': 'Low kf'},
            {'N': 128, 'Df': 1.8, 'kf': 1.8, 'rp_gstd': 1.5, 'description': 'High kf'},
            {'N': 128, 'Df': 1.8, 'kf': 2.0, 'rp_gstd': 1.5, 'description': 'Very high kf'},
        ]

    def _define_polydisperse_cases(self) -> List[Dict]:
        """Polydispersity stress tests."""
        return [
            {'N': 128, 'Df': 1.8, 'kf': 1.0, 'rp_gstd': 1.8, 'description': 'High polydispersity'},
            {'N': 128, 'Df': 1.8, 'kf': 1.0, 'rp_gstd': 2.0, 'description': 'Very high polydispersity'},
            {'N': 128, 'Df': 1.8, 'kf': 1.0, 'rp_gstd': 1.0, 'description': 'Monodisperse'},
            {'N': 128, 'Df': 2.2, 'kf': 1.0, 'rp_gstd': 1.8, 'description': 'High Df + polydisperse'},
        ]

    def _define_scaling_cases(self) -> List[Dict]:
        """Scaling with particle number."""
        return [
            {'N': 64,   'Df': 1.8, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Small N'},
            {'N': 128,  'Df': 1.8, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Medium N'},
            {'N': 256,  'Df': 1.8, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Large N'},
            {'N': 512,  'Df': 1.8, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Very large N'},
        ]

    def _define_corner_cases(self) -> List[Dict]:
        """Corner cases with combined extremes."""
        return [
            {'N': 128, 'Df': 1.5, 'kf': 0.8, 'rp_gstd': 1.8, 'description': 'Low everything'},
            {'N': 128, 'Df': 2.4, 'kf': 1.8, 'rp_gstd': 1.8, 'description': 'High everything'},
            {'N': 512, 'Df': 1.6, 'kf': 1.0, 'rp_gstd': 1.5, 'description': 'Large N + low Df'},
            {'N': 512, 'Df': 2.3, 'kf': 1.5, 'rp_gstd': 1.3, 'description': 'Large N + high Df'},
        ]

    def run_single_trial(
        self,
        params: Dict,
        trial_num: int,
        category: str
    ) -> BenchmarkResult:
        """Run a single benchmark trial.

        Args:
            params: Test case parameters
            trial_num: Trial number (for seed generation)
            category: Benchmark category name

        Returns:
            BenchmarkResult with success status and metrics
        """
        # Fill in defaults
        full_params = {
            'rp_g': 100.0,
            'tol_ov': 1e-6,
            'n_subcl_percentage': 0.1,
            'ext_case': 0,
            **params
        }

        # Generate deterministic seed
        seed = hash((category, params['description'], trial_num)) % (2**31)
        full_params['seed'] = seed

        print(f"  Trial {trial_num + 1}: {params['description']} (seed={seed})")

        start_time = time.time()

        try:
            success, final_coords, final_radii = run_simulation(
                iteration=trial_num,
                sim_config_dict=full_params,
                output_base_dir=str(self.output_dir / "aggregates" / category),
                seed=seed
            )

            runtime = time.time() - start_time

            result = BenchmarkResult(
                category=category,
                success=success,
                runtime_seconds=runtime,
                **{k: v for k, v in full_params.items() if k in
                   ['N', 'Df', 'kf', 'rp_g', 'rp_gstd', 'tol_ov',
                    'n_subcl_percentage', 'ext_case', 'seed', 'description']}
            )

            if success and final_coords is not None:
                result.final_N = final_coords.shape[0]
                # Calculate Rg if needed
                from pyfracval.utils import calculate_cluster_properties
                _, rg, _, _ = calculate_cluster_properties(
                    final_coords, final_radii,
                    full_params['Df'], full_params['kf']
                )
                result.final_Rg = rg
            else:
                # Try to determine failure stage from logs
                # (This requires enhanced logging in main_runner.py)
                result.failure_stage = "UNKNOWN"
                result.failure_reason = "Check logs"

            return result

        except Exception as e:
            runtime = time.time() - start_time
            return BenchmarkResult(
                category=category,
                success=False,
                runtime_seconds=runtime,
                failure_stage="EXCEPTION",
                failure_reason=str(e),
                **{k: v for k, v in full_params.items() if k in
                   ['N', 'Df', 'kf', 'rp_g', 'rp_gstd', 'tol_ov',
                    'n_subcl_percentage', 'ext_case', 'seed', 'description']}
            )

    def run_suite(
        self,
        category: str,
        n_trials: int = 10,
        save_individual: bool = True
    ) -> List[BenchmarkResult]:
        """Run all tests in a specific category.

        Args:
            category: Benchmark category to run
            n_trials: Number of trials per test case
            save_individual: Save individual case results

        Returns:
            List of all BenchmarkResults for this category
        """
        print(f"\n{'='*60}")
        print(f"Running benchmark suite: {category.upper()}")
        print(f"{'='*60}\n")

        test_cases = self.test_suites[category]
        all_results = []

        for test_params in test_cases:
            print(f"\nTest: {test_params['description']}")
            print(f"Parameters: N={test_params['N']}, Df={test_params['Df']}, "
                  f"kf={test_params['kf']}, rp_gstd={test_params.get('rp_gstd', 1.5)}")

            case_results = []
            for trial in range(n_trials):
                result = self.run_single_trial(test_params, trial, category)
                case_results.append(result)
                all_results.append(result)

            # Summary for this test case
            successes = sum(1 for r in case_results if r.success)
            avg_runtime = np.mean([r.runtime_seconds for r in case_results])
            print(f"  Success rate: {successes}/{n_trials} ({100*successes/n_trials:.1f}%)")
            print(f"  Avg runtime: {avg_runtime:.2f}s")

            if save_individual:
                self._save_case_results(category, test_params['description'], case_results)

        # Save suite summary
        self._save_suite_summary(category, all_results)

        return all_results

    def run_all(self, n_trials: int = 10):
        """Run all benchmark suites.

        Args:
            n_trials: Number of trials per test case

        Returns:
            Dictionary mapping category names to result lists
        """
        all_results = {}

        for category in self.test_suites.keys():
            results = self.run_suite(category, n_trials=n_trials)
            all_results[category] = results

        # Generate final report
        self._generate_final_report(all_results)

        return all_results

    def _save_case_results(self, category: str, description: str, results: List[BenchmarkResult]):
        """Save results for a single test case."""
        output_file = self.output_dir / f"{category}_{description.replace(' ', '_')}.json"

        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    def _save_suite_summary(self, category: str, results: List[BenchmarkResult]):
        """Save summary for entire suite."""
        output_file = self.output_dir / f"{category}_summary.json"

        summary = {
            'category': category,
            'total_trials': len(results),
            'successes': sum(1 for r in results if r.success),
            'success_rate': sum(1 for r in results if r.success) / len(results),
            'avg_runtime': np.mean([r.runtime_seconds for r in results]),
            'median_runtime': np.median([r.runtime_seconds for r in results]),
            'results': [asdict(r) for r in results]
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_final_report(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Generate comprehensive markdown report."""
        report_file = self.output_dir / "BENCHMARK_REPORT.md"

        with open(report_file, 'w') as f:
            f.write("# PyFracVAL Sticking Benchmark Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Overall Summary\n\n")
            f.write("| Category | Total Trials | Successes | Success Rate | Avg Runtime |\n")
            f.write("|----------|--------------|-----------|--------------|-------------|\n")

            for category, results in all_results.items():
                total = len(results)
                successes = sum(1 for r in results if r.success)
                rate = 100 * successes / total
                avg_time = np.mean([r.runtime_seconds for r in results])

                f.write(f"| {category} | {total} | {successes} | {rate:.1f}% | {avg_time:.2f}s |\n")

            f.write("\n## Detailed Results by Category\n\n")

            for category, results in all_results.items():
                f.write(f"### {category.upper()}\n\n")

                # Group by test case
                test_cases = {}
                for result in results:
                    desc = result.description
                    if desc not in test_cases:
                        test_cases[desc] = []
                    test_cases[desc].append(result)

                for desc, case_results in test_cases.items():
                    successes = sum(1 for r in case_results if r.success)
                    total = len(case_results)

                    # Get parameters from first result
                    r0 = case_results[0]

                    f.write(f"**{desc}**\n")
                    f.write(f"- Parameters: N={r0.N}, Df={r0.Df}, kf={r0.kf}, rp_gstd={r0.rp_gstd}\n")
                    f.write(f"- Success: {successes}/{total} ({100*successes/total:.1f}%)\n")
                    f.write(f"- Avg Runtime: {np.mean([r.runtime_seconds for r in case_results]):.2f}s\n")

                    if successes < total:
                        failures = [r for r in case_results if not r.success]
                        f.write(f"- Failures: {[r.failure_stage for r in failures]}\n")

                    f.write("\n")

        print(f"\n{'='*60}")
        print(f"Benchmark complete! Report saved to: {report_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    benchmark = StickingBenchmark()

    # Run quick test first (stable cases only)
    print("Running quick test on stable cases...")
    benchmark.run_suite('stable', n_trials=3)

    # Uncomment to run full benchmark
    # print("\nRunning full benchmark suite...")
    # benchmark.run_all(n_trials=10)
