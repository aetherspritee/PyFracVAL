#!/usr/bin/env python3
"""Is drop-rescue still worth anything now that backtracking exists?

docs/source/drop_rescue.md measured drop-rescue against the *greedy*
pairing baseline, where hard-regime single-shot success was 2.5%. That
baseline is gone: backtracking reaches ~100% at the same point
(docs/source/backtracking_pairing.md), so the failures drop-rescue was
built to catch mostly no longer happen there.

The honest question is therefore not "does it help at the old hard
regime" but "does it help at the *new* failure frontier" - the Df/kf/sigma
region where backtracking still fails, per
docs/source/boundary_sweep_v2.md. This script measures that, at two points
on the new frontier, using the same single-shot methodology as every
prior pairing/rescue experiment.

Usage:
    devenv shell -- uv run python benchmarks/drop_rescue_after_backtracking.py
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
from backtracking_pairing_benchmark import run_one_trial  # noqa: E402

from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.quality import compute_aggregate_quality
from pyfracval.schemas import SimulationParameters

logging.basicConfig(level=logging.CRITICAL)

N_SEEDS = 40
RESULTS_DIR = Path("benchmark_results")

# Points where backtracking still fails roughly half the time, from
# docs/source/boundary_sweep_v2.md's sigma=1.9 table.
FRONTIER = {
    "frontier_df2.3_kf1.0": dict(N=128, Df=2.3, kf=1.0, rp_gstd=1.9),
    "frontier_df2.4_kf0.8": dict(N=128, Df=2.4, kf=0.8, rp_gstd=1.9),
}

CONFIGS = {
    "baseline": OrchestratorAlgorithmConfig(),
    # Conservative default budget.
    "drop_default": OrchestratorAlgorithmConfig(cca_drop_rescue_enabled=True),
    # Relaxed relative budget, absolute cap still on.
    "drop_relaxed": OrchestratorAlgorithmConfig(
        cca_drop_rescue_enabled=True,
        cca_drop_rescue_max_particles=5,
        cca_drop_rescue_max_fraction=0.25,
    ),
    # Absolute cap disabled entirely, so the relative budget actually
    # scales with cluster size (the N-aware mode).
    "drop_relative_only": OrchestratorAlgorithmConfig(
        cca_drop_rescue_enabled=True,
        cca_drop_rescue_max_particles=0,
        cca_drop_rescue_max_fraction=0.25,
    ),
}


def run(name, cfg, regime, regime_name) -> dict:
    sim = SimulationParameters(
        N=regime["N"],
        Df=regime["Df"],
        kf=regime["kf"],
        rp_g=100.0,
        rp_gstd=regime["rp_gstd"],
        tol_ov=1e-6,
        n_subcl_percentage=0.1,
        ext_case=0,
    )
    n_ok = 0
    dropped_total = 0
    short = 0
    rg_errs = []
    bad_overlap = 0
    t0 = time.time()

    for i in range(N_SEEDS):
        result, _ = run_one_trial(sim, cfg, np.random.default_rng(i + 1))
        if result is None:
            continue
        n_ok += 1
        coords, radii = result
        missing = sim.N - coords.shape[0]
        if missing:
            short += 1
            dropped_total += missing
        q = compute_aggregate_quality(coords, radii, sim.Df, sim.kf, sim.tol_ov)
        rg_errs.append(abs(q["rg_error_pct"]))
        if not q["overlap_ok"]:
            bad_overlap += 1

    elapsed = time.time() - t0
    print(
        f"[{regime_name}/{name:20s}] success={n_ok / N_SEEDS:6.1%} ({n_ok}/{N_SEEDS})  "
        f"rescued(short)={short:2d}  dropped={dropped_total:3d}  "
        f"mean|rg_err|={np.mean(rg_errs) if rg_errs else float('nan'):5.2f}%  "
        f"overlap_bad={bad_overlap}  {elapsed:5.1f}s"
    )
    return {
        "regime": regime_name,
        "config": name,
        "n_seeds": N_SEEDS,
        "n_success": n_ok,
        "success_rate": n_ok / N_SEEDS,
        "n_aggregates_short_of_N": short,
        "particles_dropped_total": dropped_total,
        "mean_abs_rg_error_pct": float(np.mean(rg_errs)) if rg_errs else None,
        "n_overlap_violations": bad_overlap,
        "elapsed_s": elapsed,
    }


def main() -> None:
    print("=" * 92)
    print("Drop-rescue at the post-backtracking failure frontier")
    print("=" * 92)
    out = {}
    for regime_name, regime in FRONTIER.items():
        print(f"\n--- {regime_name}: {regime} ---")
        for name, cfg in CONFIGS.items():
            out[f"{regime_name}/{name}"] = run(name, cfg, regime, regime_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "drop_rescue_after_backtracking.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
