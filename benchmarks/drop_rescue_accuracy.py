#!/usr/bin/env python3
"""Validate the drop-rescue fallback: success-rate impact AND fractal
accuracy of rescued aggregates, not success rate alone - dropping
particles could distort Df/Rg scaling, so a raw success-rate number by
itself would be misleading (same principle as
docs/source/pipeline_baseline.md's densify_method="voronoi" finding,
which had a perfect success rate hiding a real accuracy problem).

Two checks against the exact hard-regime seeds pairing_frustration_probe.py
and pairing_strategy_frustration_rerun.py already use:
1. Success rate: baseline (no rescue) vs. drop-rescue at the config
   defaults (cca_drop_rescue_max_particles=5, max_fraction=0.02) and at a
   relaxed budget, to show how the rescue rate scales with budget.
2. Fractal accuracy: validate_fractal_structure on every rescued
   successful aggregate, same style as experiments.md's densify table.

Usage:
    devenv shell -- uv run python benchmarks/drop_rescue_accuracy.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pyfracval import particle_generation, utils
from pyfracval.cca import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.fractal import validate_fractal_structure
from pyfracval.pca_subclusters import Subclusterer
from pyfracval.schemas import SimulationParameters

N_SEEDS = 40
SEED_BASE = 1
RESULTS_DIR = Path("benchmark_results")

HARD_REGIME = dict(N=128, Df=2.25, kf=0.95, rp_gstd=1.9)

CONFIGS = {
    "baseline_no_rescue": OrchestratorAlgorithmConfig(),
    "drop_rescue_default_budget": OrchestratorAlgorithmConfig(
        cca_drop_rescue_enabled=True,
        cca_drop_rescue_max_particles=5,
        cca_drop_rescue_max_fraction=0.02,
    ),
    "drop_rescue_relaxed_budget": OrchestratorAlgorithmConfig(
        cca_drop_rescue_enabled=True,
        cca_drop_rescue_max_particles=5,
        cca_drop_rescue_max_fraction=0.25,
    ),
}


def run_one_trial(sim_params, algorithm_config, rng):
    """Single-shot (no internal retry) PCA+CCA attempt, same methodology
    as pairing_frustration_probe.py, so this is directly comparable to
    the 2.5% single-shot hard-regime baseline already measured there."""
    initial_radii = particle_generation.lognormal_pp_radii(
        sim_params.rp_gstd, sim_params.rp_g, sim_params.N, rng=rng
    )
    shuffled_radii = utils.shuffle_array(initial_radii, rng=rng)

    subcluster_runner = Subclusterer(
        initial_radii=shuffled_radii,
        df=sim_params.Df,
        kf=sim_params.kf,
        tol_ov=sim_params.tol_ov,
        n_subcl_percentage=sim_params.n_subcl_percentage,
        rp_g=sim_params.rp_g,
        rp_gstd=sim_params.rp_gstd,
        rng=rng,
        algorithm_config=algorithm_config,
    )
    if not subcluster_runner.run_subclustering() or subcluster_runner.not_able_pca:
        return None, 0

    num_clusters, not_able_pca_flag, pca_coords_radii, pca_i_orden, _ = (
        subcluster_runner.get_results()
    )
    if not_able_pca_flag or pca_coords_radii is None or pca_i_orden is None:
        return None, 0

    cca_runner = CCAggregator(
        initial_coords=pca_coords_radii[:, :3],
        initial_radii=pca_coords_radii[:, 3],
        initial_i_orden=pca_i_orden,
        n_total=sim_params.N,
        df=sim_params.Df,
        kf=sim_params.kf,
        tol_ov=sim_params.tol_ov,
        ext_case=sim_params.ext_case,
        rng=rng,
        algorithm_config=algorithm_config,
    )
    cca_result = cca_runner.run_cca()
    if cca_result is None or cca_runner.not_able_cca:
        return None, cca_runner._particles_dropped_total
    return cca_result, cca_runner._particles_dropped_total


def run_config(name: str, algorithm_config: OrchestratorAlgorithmConfig) -> dict:
    sim_params = SimulationParameters(
        N=HARD_REGIME["N"],
        Df=HARD_REGIME["Df"],
        kf=HARD_REGIME["kf"],
        rp_g=100.0,
        rp_gstd=HARD_REGIME["rp_gstd"],
        tol_ov=1e-6,
        n_subcl_percentage=0.1,
        ext_case=0,
    )

    n_success = 0
    validations = []
    t0 = time.time()

    for i in range(N_SEEDS):
        rng = np.random.default_rng(SEED_BASE + i)
        result, n_dropped = run_one_trial(sim_params, algorithm_config, rng)
        if result is None:
            continue
        n_success += 1
        coords, radii = result
        validation = validate_fractal_structure(
            coords, radii, sim_params.Df, sim_params.kf
        )
        validation["seed"] = SEED_BASE + i
        validation["n_particles_dropped"] = n_dropped
        validation["n_particles_actual"] = int(coords.shape[0])
        validations.append(validation)

    elapsed = time.time() - t0
    success_rate = n_success / N_SEEDS
    print(
        f"[{name}] success_rate={success_rate:.1%} ({n_success}/{N_SEEDS}) ({elapsed:.1f}s)"
    )

    summary = {
        "config_name": name,
        "n_seeds": N_SEEDS,
        "n_success": n_success,
        "success_rate": success_rate,
        "elapsed_s": elapsed,
        "validations": validations,
    }
    if validations:
        avg_rg_err = sum(v["rg_error_pct"] for v in validations) / len(validations)
        avg_df_err = sum(v["df_error"] for v in validations) / len(validations)
        rg_ok = sum(1 for v in validations if v["rg_ok"])
        n_with_drops = sum(1 for v in validations if v["n_particles_dropped"] > 0)
        avg_dropped = sum(v["n_particles_dropped"] for v in validations) / len(
            validations
        )
        summary.update(
            {
                "avg_rg_error_pct": avg_rg_err,
                "avg_df_error": avg_df_err,
                "rg_within_5pct": rg_ok,
                "n_rescued_successes": n_with_drops,
                "avg_particles_dropped": avg_dropped,
            }
        )
        print(
            f"  avg_rg_err={avg_rg_err:+.1f}%  avg_df_err={avg_df_err:+.3f}  "
            f"rg_ok={rg_ok}/{len(validations)}  rescued={n_with_drops}  "
            f"avg_dropped={avg_dropped:.1f}"
        )
    return summary


def main() -> None:
    print("=" * 80)
    print("Drop-rescue accuracy validation")
    print("=" * 80)

    results = {name: run_config(name, cfg) for name, cfg in CONFIGS.items()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "drop_rescue_accuracy.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
