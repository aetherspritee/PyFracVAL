#!/usr/bin/env python3
"""Generate cluster data for all feasible (sigma, Df, N) combinations.

Reads the feasibility CSVs from the wide sweep, finds the best kf per combo,
and generates 5 clusters per combo using the Marvin Dask cluster.
Auto-retries failed generations with new seeds.  Cluster .dat files are saved
locally on the client machine (not on remote workers).

Usage:
    devenv shell -- uv run python scripts/generate_cluster_data.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dask.distributed import as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pyfracval.dask_runner import get_client  # noqa: E402
from pyfracval.fractal import calculate_cluster_properties  # noqa: E402
from pyfracval.main_runner import run_simulation  # noqa: E402
from pyfracval.schemas import (  # noqa: E402
    AggregateProperties,
    GenerationInfo,
    Metadata,
    SimulationParameters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLUSTERS_PER_COMBO = 5
MAX_ATTEMPTS_PER_COMBO = 100
SCHEDULER_ADDRESS = "tcp://marvin.bv.e-technik.tu-dortmund.de:8786"
OUTPUT_BASE = PROJECT_ROOT / "cluster_data"

# Workers save to a temp dir on their own filesystem (discarded); we save
# locally on the client after collecting coords/radii from the future.
_WORKER_OUTPUT_DIR = "/tmp/pyfracval_worker_output"

# Algorithm config for densify+retry (matches the wide_sweep_densify_retry TOML)
DENSIFY_RETRY_ALGORITHM: dict[str, object] = {
    "densify_enabled": True,
    "densify_source_df": 2.0,
    "densify_source_kf": 1.0,
    "densify_method": "radial",
    "densify_rtol": 0.05,
    "densify_max_push_iters": 50,
    "cca_retry_rotation_mode": "alternate",
    "cca_retry_escalate_after": 120,
    "cca_dual_jitter_interval": 5,
    "cca_dual_jitter_deg": 8.0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deterministic_seed(
    sigma: float,
    df_val: float,
    n_val: int,
    attempt: int,
    config_label: str,
) -> int:
    """Reproducible seed bound to (sigma, Df, N, attempt, config)."""
    return abs(hash((sigma, df_val, n_val, attempt, config_label))) % (2**31 - 1)


def parse_feasibility_csv(csv_path: Path) -> list[dict[str, object]]:
    """Read feasibility CSV, return combos with best kf per (sigma, Df, N)."""
    combos: dict[tuple[float, float, int], list[tuple[float, float]]] = defaultdict(
        list
    )
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            key = (float(row["sigma"]), float(row["Df"]), int(row["N"]))
            combos[key].append((float(row["kf"]), float(row["success_rate"])))

    result: list[dict[str, object]] = []
    for (sigma, df_val, n_val), kfs in combos.items():
        kfs.sort(key=lambda x: -x[1])
        result.append(
            {
                "sigma": sigma,
                "Df": df_val,
                "N": n_val,
                "kf": kfs[0][0],
                "success_rate": kfs[0][1],
            }
        )
    return sorted(result, key=lambda r: (r["sigma"], r["N"], r["Df"]))  # type: ignore[arg-type,return-value]


def build_sim_config(
    sigma: float,
    df_val: float,
    n_val: int,
    kf: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    config: dict[str, object] = {
        "N": n_val,
        "Df": df_val,
        "kf": kf,
        "rp_g": 1.0,
        "rp_gstd": sigma,
        "tol_ov": 1e-6,
        "n_subcl_percentage": 0.1,
        "ext_case": 0,
    }
    if extra:
        config.update(extra)
    return config


def output_dir_for_combo(
    sigma: float,
    df_val: float,
    n_val: int,
    config_label: str,
) -> Path:
    sigma_str = f"sigma_{sigma:.2f}".replace(".", "p")
    df_str = f"Df_{df_val:.2f}".replace(".", "p")
    return OUTPUT_BASE / config_label / f"{sigma_str}__{df_str}__N_{n_val}"


def save_cluster_locally(
    coords: np.ndarray,
    radii: np.ndarray,
    out_dir: Path,
    iteration: int,
    seed: int,
    sigma: float,
    df_val: float,
    n_val: int,
    kf: float,
) -> str:
    """Save a single cluster as a .dat file (YAML header + data columns).

    Returns the path to the saved file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute aggregate properties
    _total_mass, rg, cm, _r_max = calculate_cluster_properties(
        coords, radii, df_val, kf
    )

    sim_params = SimulationParameters(
        N=n_val,
        Df=df_val,
        kf=kf,
        rp_g=1.0,
        rp_gstd=sigma,
        tol_ov=1e-6,
        n_subcl_percentage=0.1,
        ext_case=0,
        seed=seed,
    )
    metadata = Metadata(
        generation_info=GenerationInfo(iteration=iteration),
        simulation_parameters=sim_params,
        aggregate_properties=AggregateProperties(
            N_particles_actual=int(coords.shape[0]),
            radius_of_gyration=float(rg) if rg is not None else None,
            center_of_mass=cm.tolist() if cm is not None else None,  # type: ignore[union-attr]
        ),
    )

    metadata.save_to_file(folderpath=str(out_dir), coords=coords, radii=radii)

    # Find the file that was just saved (save_to_file generates the filename)
    saved_files = sorted(out_dir.glob("fracval_*.dat"), key=lambda p: p.stat().st_mtime)
    if saved_files:
        return str(saved_files[-1])
    return str(out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tasks: list[tuple[str, Path, dict[str, object] | None]] = [
        (
            "vanilla",
            PROJECT_ROOT / "benchmark_results/plausibility/wide_sweep_feasible_kf.csv",
            None,
        ),
        (
            "densify_retry",
            PROJECT_ROOT
            / "benchmark_results/plausibility/wide_sweep_feasible_kf_densify_retry.csv",
            DENSIFY_RETRY_ALGORITHM,
        ),
    ]

    # Parse combos
    all_combos: dict[str, list[dict[str, object]]] = {}
    total_target = 0
    for label, csv_path, _algo in tasks:
        combos = parse_feasibility_csv(csv_path)
        all_combos[label] = combos
        n_target = len(combos) * CLUSTERS_PER_COMBO
        total_target += n_target
        logger.info(
            "%s: %d combos × %d = %d clusters target",
            label,
            len(combos),
            CLUSTERS_PER_COMBO,
            n_target,
        )
    logger.info("Total target: %d clusters", total_target)

    # Master index
    master_index: list[dict[str, object]] = []
    stats: dict[str, dict[str, int]] = {}

    try:
        from tqdm import tqdm  # noqa: F811

        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    t_start = time.time()

    with get_client(
        scheduler_address=SCHEDULER_ADDRESS, install_package=True
    ) as client:
        # Prevent nested process/thread pools inside Dask workers
        client.run(
            lambda: os.environ.__setitem__(
                "PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS", "1"
            )
        )

        for config_label, combos in all_combos.items():
            algo = DENSIFY_RETRY_ALGORITHM if config_label == "densify_retry" else None
            n_combos = len(combos)
            combo_successes = 0
            combo_failures = 0

            combo_iter = tqdm(combos, desc=config_label) if _HAS_TQDM else combos

            for combo_idx, combo in enumerate(combo_iter):
                sigma = float(combo["sigma"])  # type: ignore[arg-type]
                df_val = float(combo["Df"])  # type: ignore[arg-type]
                n_val = int(combo["N"])  # type: ignore[arg-type]
                kf = float(combo["kf"])  # type: ignore[arg-type]

                out_dir = output_dir_for_combo(sigma, df_val, n_val, config_label)
                sim_config = build_sim_config(sigma, df_val, n_val, kf, algo)

                successes = 0
                total_attempts = 0

                while (
                    successes < CLUSTERS_PER_COMBO
                    and total_attempts < MAX_ATTEMPTS_PER_COMBO
                ):
                    needed = CLUSTERS_PER_COMBO - successes
                    batch_size = min(needed, MAX_ATTEMPTS_PER_COMBO - total_attempts)

                    # Submit a batch of tasks (worker saves to /tmp, we save locally)
                    futures: dict[object, tuple[int, int]] = {}
                    for j in range(batch_size):
                        attempt_num = total_attempts + j
                        seed = deterministic_seed(
                            sigma, df_val, n_val, attempt_num, config_label
                        )
                        fut = client.submit(
                            run_simulation,
                            attempt_num,
                            sim_config,
                            _WORKER_OUTPUT_DIR,
                            seed,
                        )
                        futures[fut] = (attempt_num, seed)

                    # Collect results and save locally
                    for future in as_completed(futures):
                        attempt_num, seed = futures[future]
                        total_attempts += 1
                        try:
                            success, coords, radii = future.result()
                            if success and coords is not None and radii is not None:
                                filepath = save_cluster_locally(
                                    coords,
                                    radii,
                                    out_dir,
                                    attempt_num,
                                    seed,
                                    sigma,
                                    df_val,
                                    n_val,
                                    kf,
                                )
                                successes += 1
                                master_index.append(
                                    {
                                        "config": config_label,
                                        "sigma": sigma,
                                        "Df": df_val,
                                        "N": n_val,
                                        "kf": kf,
                                        "attempt": attempt_num,
                                        "seed": seed,
                                        "success": True,
                                        "filepath": filepath,
                                    }
                                )
                        except Exception as exc:
                            logger.warning(
                                "%s | σ=%.2f Df=%.1f N=%d | attempt %d failed: %s",
                                config_label,
                                sigma,
                                df_val,
                                n_val,
                                attempt_num,
                                exc,
                            )

                if successes >= CLUSTERS_PER_COMBO:
                    combo_successes += 1
                else:
                    combo_failures += 1
                    logger.error(
                        "%s | σ=%.2f Df=%.1f N=%d | FAILED: only %d/%d clusters "
                        "after %d attempts",
                        config_label,
                        sigma,
                        df_val,
                        n_val,
                        successes,
                        CLUSTERS_PER_COMBO,
                        total_attempts,
                    )

                # Progress every 20 combos
                if combo_idx % 20 == 19:
                    logger.info(
                        "%s progress: %d/%d combos done",
                        config_label,
                        combo_idx + 1,
                        n_combos,
                    )

            stats[config_label] = {
                "combos_total": n_combos,
                "combos_ok": combo_successes,
                "combos_failed": combo_failures,
                "clusters_generated": len(
                    [e for e in master_index if e["config"] == config_label]
                ),
            }

    t_end = time.time()

    # Write master index
    index_path = OUTPUT_BASE / "cluster_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w") as fh:
        json.dump(master_index, fh, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("CLUSTER GENERATION COMPLETE")
    print("=" * 60)
    for label, s in stats.items():
        print(
            f"  {label}: {s['combos_ok']}/{s['combos_total']} combos OK "
            f"({s['clusters_generated']} clusters)"
        )
    total_gen = sum(s["clusters_generated"] for s in stats.values())
    print(f"  Total clusters generated: {total_gen}")
    print(f"  Total elapsed: {t_end - t_start:.1f} s")
    print(f"  Master index: {index_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
