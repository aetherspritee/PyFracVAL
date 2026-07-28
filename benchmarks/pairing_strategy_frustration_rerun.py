#!/usr/bin/env python3
"""Re-run pairing_frustration_probe.py's exact hard-regime seeds with the
new matching-based pairing strategies active, to directly check whether
the diagnosed 38/39 rescuable population (docs/source/pairing_frustration.md)
is actually rescued in the real single-shot production path - not just
"is a matching possible" (already answered by the original probe's
census), but "does using matching pairing in production actually succeed
on these seeds."

Same regime, same seed base, same single-shot (no internal retry)
methodology as pairing_frustration_probe.py, varying only
algorithm_config.cca_pairing_strategy.

Usage:
    devenv shell -- uv run python benchmarks/pairing_strategy_frustration_rerun.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from pairing_frustration_probe import SEED_BASE, run_one_trial  # noqa: E402

from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.schemas import SimulationParameters

N_SEEDS = 40
RESULTS_DIR = Path("benchmark_results")

STRATEGIES = ["greedy", "matching", "matching_leaf_weighted"]

REGIMES = {
    "hard": dict(N=128, Df=2.25, kf=0.95, rp_gstd=1.9),
    "easy_control": dict(N=128, Df=1.8, kf=1.0, rp_gstd=1.5),
}


def run_regime_strategy(name: str, params: dict, strategy: str) -> dict:
    sim_params = SimulationParameters(
        N=params["N"],
        Df=params["Df"],
        kf=params["kf"],
        rp_g=100.0,
        rp_gstd=params["rp_gstd"],
        tol_ov=1e-6,
        n_subcl_percentage=0.1,
        ext_case=0,
    )
    algorithm_config = OrchestratorAlgorithmConfig(cca_pairing_strategy=strategy)

    outcomes = {"success": 0, "cca_failed": 0, "pca_failed": 0}
    t0 = time.time()

    for i in range(N_SEEDS):
        rng = np.random.default_rng(SEED_BASE + i)
        trial = run_one_trial(sim_params, algorithm_config, rng)
        outcomes[trial["outcome"]] += 1

    elapsed = time.time() - t0
    success_rate = outcomes["success"] / N_SEEDS
    print(
        f"[{name}/{strategy}] success_rate={success_rate:.1%} "
        f"({outcomes['success']}/{N_SEEDS}) ({elapsed:.1f}s)"
    )
    return {
        "regime": name,
        "strategy": strategy,
        "params": params,
        "n_seeds": N_SEEDS,
        "elapsed_s": elapsed,
        "outcomes": outcomes,
        "success_rate": success_rate,
    }


def main() -> None:
    print("=" * 80)
    print("Pairing strategy frustration rerun")
    print("=" * 80)

    results = []
    for regime_name, params in REGIMES.items():
        for strategy in STRATEGIES:
            results.append(run_regime_strategy(regime_name, params, strategy))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pairing_strategy_frustration_rerun.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    print("\nSummary (hard regime):")
    for r in results:
        if r["regime"] == "hard":
            print(f"  {r['strategy']:<25} {r['success_rate']:.1%}")


if __name__ == "__main__":
    main()
