#!/usr/bin/env python3
"""Validate aggregate *structure*, not just Rg, via f(r).

Two claims in this project rest on structure rather than on a single
scalar, and neither has been checked against the metric FracVAL itself
validates with (:cite:`Moran2019FracVAL` §4.2, the density-density
correlation function and its Df-3 log-log slope):

1. **Densification** (docs/source/experiments.md) generates at an easy
   Df/kf and reshapes to the target. It matches Rg well - but Rg is one
   number, and a reshaped aggregate could match it while having the wrong
   internal structure.
2. **Backtracking** (docs/source/backtracking_pairing.md) lets a
   frustrated cluster pass through a round unmerged. That changes the
   merge hierarchy, and the paper's own Appendix B is the only reason to
   expect hierarchy details not to matter.

This script measures f(r) for aggregates built each way and compares the
recovered Df against the prescribed one.

Usage:
    devenv shell -- uv run python benchmarks/correlation_validation.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from pyfracval.correlation import density_correlation, fit_correlation_slope
from pyfracval.main_runner import run_simulation
from pyfracval.quality import compute_aggregate_quality

logging.basicConfig(level=logging.CRITICAL)

RESULTS_DIR = Path("benchmark_results")
N_SEEDS = 6
N_PARTICLES = 512
N_ORIENTATIONS = 120
N_RADII = 32

# Arms chosen so each isolates one question.
ARMS = {
    # Structure of an ordinary, natively-built aggregate: the reference.
    "native_df1.8": dict(Df=1.8, kf=1.0, rp_gstd=1.5, densify=False),
    "native_df2.1": dict(Df=2.1, kf=1.0, rp_gstd=1.5, densify=False),
    # Same target, reached by reshaping instead of by direct generation.
    "densified_df2.1": dict(Df=2.1, kf=1.0, rp_gstd=1.5, densify=True),
    # A target only reachable by densification before backtracking, and a
    # region where pass-throughs are common now.
    "native_df2.3": dict(Df=2.3, kf=1.0, rp_gstd=1.9, densify=False),
    "densified_df2.3": dict(Df=2.3, kf=1.0, rp_gstd=1.9, densify=True),
}


def build(arm: dict, seed: int, out_dir: Path):
    cfg = {
        "N": N_PARTICLES,
        "Df": arm["Df"],
        "kf": arm["kf"],
        "rp_g": 100.0,
        "rp_gstd": arm["rp_gstd"],
        "tol_ov": 1e-6,
        "n_subcl_percentage": 0.1,
        "ext_case": 0,
        "seed": seed,
    }
    if arm["densify"]:
        cfg.update(
            {
                "densify_enabled": True,
                "densify_source_df": 1.8,
                "densify_source_kf": 1.0,
            }
        )
    ok, coords, radii = run_simulation(
        iteration=1,
        sim_config_dict=cfg,
        output_base_dir=str(out_dir),
        max_runtime_seconds=180.0,
    )
    if not ok or coords is None or radii is None:
        return None
    return coords, radii


def main() -> None:
    print("=" * 92)
    print(f"f(r) structural validation  (N={N_PARTICLES}, {N_SEEDS} seeds/arm)")
    print("=" * 92)
    out_dir = Path("/tmp/pyfracval_corr_validation")
    results = {}

    for name, arm in ARMS.items():
        rows = []
        t0 = time.time()
        for seed in range(1, N_SEEDS + 1):
            built = build(arm, seed, out_dir)
            if built is None:
                continue
            coords, radii = built
            rng = np.random.default_rng(1000 + seed)
            corr = density_correlation(
                coords,
                radii,
                n_orientations=N_ORIENTATIONS,
                n_radii=N_RADII,
                rng=rng,
            )
            fit = fit_correlation_slope(corr)
            q = compute_aggregate_quality(coords, radii, arm["Df"], arm["kf"], 1e-6)
            rows.append(
                {
                    "seed": seed,
                    "df_from_fr": fit["df_estimate"],
                    "fit_r_squared": fit["r_squared"],
                    "fit_points": fit["n_points"],
                    "rg_error_pct": q["rg_error_pct"],
                    "max_residual_overlap": q["max_residual_overlap"],
                    "overlap_ok": q["overlap_ok"],
                    "n_particles": q["n_particles"],
                }
            )
        elapsed = time.time() - t0

        if not rows:
            print(f"[{name:18s}] no successful builds")
            results[name] = {"arm": arm, "n_success": 0, "rows": []}
            continue

        df_est = np.array([r["df_from_fr"] for r in rows])
        df_est = df_est[~np.isnan(df_est)]
        rg_err = np.array([abs(r["rg_error_pct"]) for r in rows])
        r2 = np.array([r["fit_r_squared"] for r in rows])
        bad = sum(1 for r in rows if not r["overlap_ok"])

        print(
            f"[{name:18s}] built={len(rows)}/{N_SEEDS}  "
            f"target_Df={arm['Df']:.2f}  "
            f"Df_from_f(r)={df_est.mean():.2f}+-{df_est.std():.2f}  "
            f"(err {df_est.mean() - arm['Df']:+.2f})  "
            f"R2={np.nanmean(r2):.3f}  "
            f"|rg_err|={rg_err.mean():.2f}%  overlap_bad={bad}  {elapsed:.0f}s"
        )
        results[name] = {
            "arm": arm,
            "n_success": len(rows),
            "df_from_fr_mean": float(df_est.mean()) if df_est.size else None,
            "df_from_fr_std": float(df_est.std()) if df_est.size else None,
            "df_error": float(df_est.mean() - arm["Df"]) if df_est.size else None,
            "fit_r_squared_mean": float(np.nanmean(r2)),
            "mean_abs_rg_error_pct": float(rg_err.mean()),
            "n_overlap_violations": bad,
            "rows": rows,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "correlation_validation.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
