#!/usr/bin/env python3
"""
Sigma Failure Mode Investigation

Deep dive into why wide particle size distributions (sigma > 1.5) fail.
Instruments the PCA process to understand exactly where and why failures occur.

Usage:
    uv run python benchmarks/sigma_failure_investigation.py
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

# Enable detailed logging for this investigation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FailureAnalysis:
    """Detailed failure mode analysis."""

    sigma: float
    Df: float
    kf: float
    seed: int
    success: bool
    failure_stage: str  # "radii_generation", "pca", "cca", "success"
    failure_details: dict
    particle_size_stats: dict
    runtime: float


class SigmaFailureInvestigation:
    """Investigate why wide size distributions fail."""

    def __init__(self, output_dir: str = "benchmark_results/sigma_investigation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_single_run(
        self, sigma: float, Df: float, kf: float, seed: int
    ) -> FailureAnalysis:
        """Run a single trial with detailed instrumentation."""

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing: sigma={sigma}, Df={Df}, kf={kf}, seed={seed}")
        logger.info(f"{'=' * 80}\n")

        config = {
            "N": 128,
            "Df": Df,
            "kf": kf,
            "rp_g": 100.0,
            "rp_gstd": sigma,
            "tol_ov": 1e-6,
            "n_subcl_percentage": 0.1,
            "ext_case": 0,
            "seed": seed,
        }

        start_time = time.time()
        failure_stage = "success"
        failure_details = {}
        particle_stats = {}

        try:
            # Generate particle radii to analyze distribution
            from pyfracval import particle_generation

            np.random.seed(seed)
            radii = particle_generation.lognormal_pp_radii(sigma, 100.0, 128)

            particle_stats = {
                "mean": float(np.mean(radii)),
                "std": float(np.std(radii)),
                "min": float(np.min(radii)),
                "max": float(np.max(radii)),
                "range": float(np.max(radii) - np.min(radii)),
                "size_ratio": float(np.max(radii) / np.min(radii)),
                "cv": float(np.std(radii) / np.mean(radii)),  # coefficient of variation
                "percentiles": {
                    "p10": float(np.percentile(radii, 10)),
                    "p25": float(np.percentile(radii, 25)),
                    "p50": float(np.percentile(radii, 50)),
                    "p75": float(np.percentile(radii, 75)),
                    "p90": float(np.percentile(radii, 90)),
                },
            }

            logger.info(f"Particle size distribution:")
            logger.info(f"  Mean: {particle_stats['mean']:.2f}")
            logger.info(f"  Std:  {particle_stats['std']:.2f}")
            logger.info(
                f"  Range: [{particle_stats['min']:.2f}, {particle_stats['max']:.2f}]"
            )
            logger.info(f"  Size ratio (max/min): {particle_stats['size_ratio']:.2f}x")
            logger.info(f"  Coefficient of variation: {particle_stats['cv']:.3f}")

            # Run simulation
            success, coords, radii_final = run_simulation(
                1, config, output_base_dir="/tmp/sigma_investigation", seed=seed
            )

            if not success:
                failure_stage = "pca_or_cca"
                failure_details = {
                    "message": "Simulation failed during PCA or CCA",
                    "likely_cause": "Geometric constraints from wide size distribution",
                }

        except Exception as e:
            success = False
            failure_stage = "exception"
            failure_details = {
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            }
            logger.error(f"Exception occurred: {e}", exc_info=True)

        runtime = time.time() - start_time

        return FailureAnalysis(
            sigma=sigma,
            Df=Df,
            kf=kf,
            seed=seed,
            success=success,
            failure_stage=failure_stage,
            failure_details=failure_details,
            particle_size_stats=particle_stats,
            runtime=runtime,
        )

    def compare_sigma_values(self) -> list[FailureAnalysis]:
        """Compare different sigma values with same Df/kf."""

        # Use a reliable Df/kf combo from our sweep (Df=2.0, kf=1.0)
        Df, kf = 2.0, 1.0

        # Test sigma values: 1.3 (good), 1.5 (marginal), 1.8, 2.0, 2.5 (bad)
        sigma_values = [1.3, 1.5, 1.8, 2.0, 2.5]
        seeds = [42, 123, 456, 789, 1024]  # 5 seeds per sigma

        results = []
        total = len(sigma_values) * len(seeds)
        current = 0

        print(f"Sigma Failure Investigation")
        print(f"=" * 80)
        print(f"Testing Df={Df}, kf={kf} with sigma values: {sigma_values}")
        print(f"Seeds per sigma: {len(seeds)}")
        print(f"Total trials: {total}")
        print()

        for sigma in sigma_values:
            print(f"\nTesting sigma = {sigma}")
            print("-" * 40)

            for seed in seeds:
                current += 1
                print(
                    f"  [{current}/{total}] seed={seed}...",
                    end=" ",
                    flush=True,
                )

                result = self.analyze_single_run(sigma, Df, kf, seed)
                results.append(result)

                status = "✓" if result.success else "✗"
                print(f"{status} ({result.runtime:.2f}s)")

        return results

    def save_results(
        self, results: list[FailureAnalysis], filename: str = "sigma_investigation.json"
    ):
        """Save investigation results."""
        output_path = self.output_dir / filename
        data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": "Investigation of wide particle size distribution failures",
                "total_trials": len(results),
            },
            "results": [asdict(r) for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def analyze_results(self, results: list[FailureAnalysis]):
        """Analyze investigation results."""
        print("\n" + "=" * 80)
        print("SIGMA FAILURE ANALYSIS")
        print("=" * 80)

        # Group by sigma
        sigma_values = sorted(set(r.sigma for r in results))

        print("\nSuccess Rate by Sigma:")
        print("-" * 40)
        for sigma in sigma_values:
            sigma_results = [r for r in results if r.sigma == sigma]
            successes = sum(1 for r in sigma_results if r.success)
            total = len(sigma_results)
            rate = successes / total * 100

            status = "✓" if rate >= 80 else "⚠" if rate >= 40 else "✗"
            print(f"{status} sigma={sigma:4.1f}: {successes}/{total} ({rate:5.1f}%)")

        # Analyze particle statistics correlation with failure
        print("\n" + "=" * 80)
        print("Particle Size Statistics vs Success")
        print("=" * 80)

        for sigma in sigma_values:
            sigma_results = [r for r in results if r.sigma == sigma]
            successes = [r for r in sigma_results if r.success]
            failures = [r for r in sigma_results if not r.success]

            print(f"\nsigma = {sigma}")
            print("-" * 40)

            if successes and failures:
                # Compare successful vs failed runs
                success_ratios = [
                    r.particle_size_stats["size_ratio"] for r in successes
                ]
                failure_ratios = [r.particle_size_stats["size_ratio"] for r in failures]

                print(f"Size ratio (max/min):")
                print(
                    f"  Successful runs: {np.mean(success_ratios):.2f} ± {np.std(success_ratios):.2f}"
                )
                print(
                    f"  Failed runs:     {np.mean(failure_ratios):.2f} ± {np.std(failure_ratios):.2f}"
                )

                success_cv = [r.particle_size_stats["cv"] for r in successes]
                failure_cv = [r.particle_size_stats["cv"] for r in failures]

                print(f"Coefficient of variation:")
                print(
                    f"  Successful runs: {np.mean(success_cv):.3f} ± {np.std(success_cv):.3f}"
                )
                print(
                    f"  Failed runs:     {np.mean(failure_cv):.3f} ± {np.std(failure_cv):.3f}"
                )

            elif successes:
                ratios = [r.particle_size_stats["size_ratio"] for r in successes]
                cvs = [r.particle_size_stats["cv"] for r in successes]
                print(f"All runs successful!")
                print(f"  Avg size ratio: {np.mean(ratios):.2f}")
                print(f"  Avg CV: {np.mean(cvs):.3f}")
            else:
                ratios = [r.particle_size_stats["size_ratio"] for r in failures]
                cvs = [r.particle_size_stats["cv"] for r in failures]
                print(f"All runs failed!")
                print(f"  Avg size ratio: {np.mean(ratios):.2f}")
                print(f"  Avg CV: {np.mean(cvs):.3f}")

        # Key findings
        print("\n" + "=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)

        all_successes = [r for r in results if r.success]
        all_failures = [r for r in results if not r.success]

        if all_successes and all_failures:
            success_ratios = [
                r.particle_size_stats["size_ratio"] for r in all_successes
            ]
            failure_ratios = [r.particle_size_stats["size_ratio"] for r in all_failures]

            print(f"\nOverall particle size ratio (max/min):")
            print(
                f"  Successful: {np.mean(success_ratios):.2f}x (range: {np.min(success_ratios):.2f}-{np.max(success_ratios):.2f})"
            )
            print(
                f"  Failed:     {np.mean(failure_ratios):.2f}x (range: {np.min(failure_ratios):.2f}-{np.max(failure_ratios):.2f})"
            )

            # Find threshold
            threshold = (np.max(success_ratios) + np.min(failure_ratios)) / 2
            print(f"\n  Estimated failure threshold: ~{threshold:.1f}x size ratio")

            success_cvs = [r.particle_size_stats["cv"] for r in all_successes]
            failure_cvs = [r.particle_size_stats["cv"] for r in all_failures]

            print(f"\nCoefficient of variation:")
            print(f"  Successful: {np.mean(success_cvs):.3f}")
            print(f"  Failed:     {np.mean(failure_cvs):.3f}")


def main():
    """Main entry point."""
    investigation = SigmaFailureInvestigation()

    print("Starting sigma failure investigation...")
    print("This will provide detailed logging for each trial.\n")

    results = investigation.compare_sigma_values()

    # Save results
    investigation.save_results(results)

    # Analyze
    investigation.analyze_results(results)

    print("\n✅ Investigation complete!")


if __name__ == "__main__":
    main()
