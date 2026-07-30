#!/usr/bin/env python3
"""Benchmark: backtracking CCA pairing vs. greedy first-fit.

docs/source/pairing_frustration.md measured that ~97% of hard-regime CCA
round failures had a *different* pairing of the same cluster pool that
would have worked, and docs/source/matching_pairing.md then showed that
picking better pairs up front from the cheap gamma-feasibility graph does
not capture that, because the graph cannot predict which feasible-looking
pairs actually stick. Backtracking reacts to the real sticking outcome
instead: on failure, try the cluster's next feasible partner.

This script measures the resulting success rate, plus the two Gamma
faithfulness/stability flags added alongside it:

- ``cca_gamma_measured_rg``: feed each cluster's *measured* Rg into the
  next Gamma so per-merge deviations cannot accumulate.

The Gamma form itself is no longer configurable: CCA always solves the
mass form (Moran Eq. 6) and PCA always the count form, because the mass
form is unusable when the second body is a single monomer. Mass weighting
is expressed through the optional per-particle ``densities`` argument.

Methodology is deliberately identical to benchmarks/drop_rescue_accuracy.py
and benchmarks/pairing_frustration_probe.py: single-shot PCA+CCA attempts
with no internal retry, so one seed maps to exactly one outcome and the
numbers are directly comparable to the 2.5% single-shot hard-regime
baseline those already established. (Retry-inclusive rates, as reported by
docs/source/hard_regime_boundary_sweep.md, are much higher for every arm.)

Usage:
    devenv shell -- uv run python benchmarks/backtracking_pairing_benchmark.py
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

from pyfracval import particle_generation, utils
from pyfracval.cca import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.fractal import validate_fractal_structure
from pyfracval.pca_subclusters import Subclusterer
from pyfracval.schemas import SimulationParameters

logging.basicConfig(level=logging.CRITICAL)

N_SEEDS = 40
SEED_BASE = 1
RESULTS_DIR = Path("benchmark_results")

# Same hard regime every prior pairing/rescue experiment used.
HARD_REGIME = dict(N=128, Df=2.25, kf=0.95, rp_gstd=1.9)
# A control inside the safe region, to confirm nothing regresses there.
EASY_REGIME = dict(N=128, Df=1.8, kf=1.0, rp_gstd=1.5)

CONFIGS = {
    "greedy_baseline": OrchestratorAlgorithmConfig(cca_pairing_strategy="greedy"),
    "backtracking": OrchestratorAlgorithmConfig(cca_pairing_strategy="backtracking"),
    "backtracking_measured_rg": OrchestratorAlgorithmConfig(
        cca_pairing_strategy="backtracking",
        cca_gamma_measured_rg=True,
    ),
}


def run_one_trial(sim_params, algorithm_config, rng):
    """One single-shot PCA+CCA attempt. Returns (result, stats)."""
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
        return None, {}

    _, not_able_pca_flag, pca_coords_radii, pca_i_orden, _ = (
        subcluster_runner.get_results()
    )
    if not_able_pca_flag or pca_coords_radii is None or pca_i_orden is None:
        return None, {}

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
    stats = {
        "backtrack_rescued_merges": cca_runner._backtrack_rescued_merges,
        "backtrack_failed_edges": cca_runner._backtrack_failed_edges,
        "pass_through_clusters": cca_runner._pass_through_clusters,
    }
    if cca_result is None or cca_runner.not_able_cca:
        return None, stats
    return cca_result, stats


def run_config(name, algorithm_config, regime, regime_name) -> dict:
    sim_params = SimulationParameters(
        N=regime["N"],
        Df=regime["Df"],
        kf=regime["kf"],
        rp_g=100.0,
        rp_gstd=regime["rp_gstd"],
        tol_ov=1e-6,
        n_subcl_percentage=0.1,
        ext_case=0,
    )

    n_success = 0
    validations = []
    totals = {
        "backtrack_rescued_merges": 0,
        "backtrack_failed_edges": 0,
        "pass_through_clusters": 0,
    }
    t0 = time.time()

    for i in range(N_SEEDS):
        rng = np.random.default_rng(SEED_BASE + i)
        result, stats = run_one_trial(sim_params, algorithm_config, rng)
        for key in totals:
            totals[key] += stats.get(key, 0)
        if result is None:
            continue
        n_success += 1
        coords, radii = result
        validation = validate_fractal_structure(
            coords, radii, sim_params.Df, sim_params.kf
        )
        validation["seed"] = SEED_BASE + i
        validation["n_particles_actual"] = int(coords.shape[0])
        validations.append(validation)

    elapsed = time.time() - t0
    success_rate = n_success / N_SEEDS
    print(
        f"[{regime_name}/{name}] success={success_rate:6.1%} "
        f"({n_success}/{N_SEEDS})  {elapsed:6.1f}s  "
        f"rescued_merges={totals['backtrack_rescued_merges']:3d}  "
        f"pass_through={totals['pass_through_clusters']:3d}"
    )

    summary = {
        "config_name": name,
        "regime": regime_name,
        "n_seeds": N_SEEDS,
        "n_success": n_success,
        "success_rate": success_rate,
        "elapsed_s": elapsed,
        **totals,
        "validations": validations,
    }
    if validations:
        avg_rg_err = sum(v["rg_error_pct"] for v in validations) / len(validations)
        avg_abs_rg_err = sum(abs(v["rg_error_pct"]) for v in validations) / len(
            validations
        )
        rg_ok = sum(1 for v in validations if v["rg_ok"])
        summary.update(
            {
                "avg_rg_error_pct": avg_rg_err,
                "avg_abs_rg_error_pct": avg_abs_rg_err,
                "rg_within_5pct": rg_ok,
            }
        )
        print(
            f"    avg_rg_err={avg_rg_err:+6.2f}%  avg|rg_err|={avg_abs_rg_err:5.2f}%  "
            f"rg_ok={rg_ok}/{len(validations)}"
        )
    return summary


def main() -> None:
    print("=" * 88)
    print("Backtracking pairing vs. greedy first-fit (single-shot methodology)")
    print("=" * 88)

    results = {}
    for regime_name, regime in (("hard", HARD_REGIME), ("easy", EASY_REGIME)):
        print(f"\n--- {regime_name} regime: {regime} ---")
        for name, cfg in CONFIGS.items():
            results[f"{regime_name}/{name}"] = run_config(
                name, cfg, regime, regime_name
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "backtracking_pairing_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
