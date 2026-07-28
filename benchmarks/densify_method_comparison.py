#!/usr/bin/env python3
"""Head-to-head comparison of densify.py's two densification methods.

`densify_method` has two implementations - "radial" (radial compression
followed by overlap resolution) and "voronoi" (Voronoi-guided iterative
migration) - but docs/source/experiments.md's existing densification
accuracy table (produced by validate_fractal_structure.py) only exercises
the "radial" default. "voronoi" has never been benchmarked. This script
runs both at the same hard-regime target/source parameters
experiments.md's table already uses, so the two are directly comparable.

Usage:
    devenv shell -- uv run python benchmarks/densify_method_comparison.py
"""

import json
import time
from pathlib import Path

from pyfracval.fractal import validate_fractal_structure
from pyfracval.main_runner import run_simulation

HARD_REGIME = {
    "N": 128,
    "Df": 2.25,
    "kf": 0.95,
    "rp_g": 100.0,
    "rp_gstd": 1.9,
    "tol_ov": 1e-6,
    "n_subcl_percentage": 0.1,
    "ext_case": 0,
}

METHODS = [
    {
        "name": "densify_radial_df20",
        "algorithm": {
            "densify_enabled": True,
            "densify_source_df": 2.0,
            "densify_source_kf": 1.0,
            "densify_method": "radial",
            "densify_rtol": 0.05,
            "densify_max_push_iters": 50,
        },
    },
    {
        "name": "densify_voronoi_df20",
        "algorithm": {
            "densify_enabled": True,
            "densify_source_df": 2.0,
            "densify_source_kf": 1.0,
            "densify_method": "voronoi",
            "densify_rtol": 0.05,
            "densify_max_push_iters": 50,
        },
    },
]

SEEDS = list(range(100, 130))
MAX_RUNTIME = 300
OUTPUT_DIR = Path("benchmark_results/densify_method_comparison")


def main() -> None:
    results = {}
    for method in METHODS:
        name = method["name"]
        algo = method["algorithm"]
        print(f"\n{'=' * 60}")
        print(f"Method: {name}")
        print(f"{'=' * 60}")

        successes = 0
        validations = []

        for seed in SEEDS:
            sim_cfg = dict(HARD_REGIME)
            sim_cfg.update(algo)

            start = time.perf_counter()
            ok, coords, radii = run_simulation(
                -1,
                sim_cfg,
                str(OUTPUT_DIR / "aggregates"),
                seed=seed,
                max_runtime_seconds=MAX_RUNTIME,
            )
            elapsed = time.perf_counter() - start

            if not ok or coords is None:
                continue

            successes += 1
            target_df = sim_cfg["Df"]
            target_kf = sim_cfg["kf"]

            validation = validate_fractal_structure(coords, radii, target_df, target_kf)
            validation["seed"] = seed
            validation["elapsed_s"] = round(elapsed, 1)
            validations.append(validation)

            status = "OK" if validation["rg_ok"] else "Rg_ERR"
            print(
                f"  seed={seed}: N={validation['N']}, "
                f"Rg_theory={validation['theoretical_rg']:.1f}, "
                f"Rg_emp={validation['empirical_rg']:.1f}, "
                f"err={validation['rg_error_pct']:+.1f}%, "
                f"Df_emp={validation['empirical_Df']:.3f} "
                f"(target={target_df}), "
                f"R²={validation['fit_r_squared']:.4f}, "
                f"elapsed={elapsed:.1f}s, {status}"
            )

        rate = successes / len(SEEDS) * 100
        print(f"\n  Success rate: {successes}/{len(SEEDS)} = {rate:.1f}%")

        if validations:
            avg_rg_err = sum(v["rg_error_pct"] for v in validations) / len(validations)
            avg_df_err = sum(v["df_error"] for v in validations) / len(validations)
            rg_ok_count = sum(1 for v in validations if v["rg_ok"])
            avg_rsq = sum(v["fit_r_squared"] for v in validations) / len(validations)
            avg_elapsed = sum(v["elapsed_s"] for v in validations) / len(validations)
            print(f"  Avg Rg error: {avg_rg_err:+.1f}%")
            print(f"  Avg Df error: {avg_df_err:+.3f}")
            print(f"  Avg R² of Df fit: {avg_rsq:.4f}")
            print(f"  Rg within 5%: {rg_ok_count}/{len(validations)}")
            print(f"  Avg wall time: {avg_elapsed:.1f}s")

        results[name] = {
            "success_rate": rate,
            "successes": successes,
            "total": len(SEEDS),
            "validations": validations,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "densify_method_comparison.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Method':<25} {'Success':>8} {'Avg Rg Err':>12} {'Avg Df Err':>12} "
        f"{'Rg<5%':>8} {'Avg Time':>10}"
    )
    print("-" * 80)
    for name, data in results.items():
        validations = data.get("validations", [])
        if validations:
            avg_rg_err = sum(v["rg_error_pct"] for v in validations) / len(validations)
            avg_df_err = sum(v["df_error"] for v in validations) / len(validations)
            rg_ok = sum(1 for v in validations if v["rg_ok"])
            avg_elapsed = sum(v["elapsed_s"] for v in validations) / len(validations)
            print(
                f"{name:<25} {data['successes']}/{data['total']:>3}"
                f" {avg_rg_err:>+11.1f}% {avg_df_err:>+11.3f} "
                f"{rg_ok}/{len(validations):>3} {avg_elapsed:>9.1f}s"
            )
        else:
            print(
                f"{name:<25} {data['successes']}/{data['total']:>3} "
                f"{'N/A':>12} {'N/A':>12} {'N/A':>8} {'N/A':>10}"
            )


if __name__ == "__main__":
    main()
