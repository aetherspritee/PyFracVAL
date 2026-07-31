#!/usr/bin/env python3
"""Generate a validated catalog of cluster data for downstream simulations.

Reads the feasibility CSVs from the wide sweep, picks the best kf per
(sigma, Df, N) combination, and generates ``CLUSTERS_PER_COMBO`` clusters
for each on a Dask cluster (remote if reachable, otherwise local).

Usage:
    devenv shell -- uv run python scripts/generate_cluster_data.py
    devenv shell -- uv run python scripts/generate_cluster_data.py --local
    devenv shell -- uv run python scripts/generate_cluster_data.py --limit 20

Design notes (this is a rewrite; see git history for the previous version)
-------------------------------------------------------------------------
Four problems in the previous implementation shaped this one.

**Saturation.** It walked combos one at a time, submitting a batch of at
most ``CLUSTERS_PER_COMBO`` tasks and then blocking until all of them
returned. Against a cluster offering ~148 concurrent slots that used
about five, i.e. a ~30x throughput loss. This version keeps a single
global work queue and a sliding in-flight window sized to the cluster, so
combos overlap and the cluster stays busy.

**Validation.** It hand-built ``AggregateProperties`` from coords/radii
and never called :func:`pyfracval.quality.compute_aggregate_quality`, so
a geometrically invalid aggregate was catalogued as ``success=True``.
That is how a batch of densified clusters with severe particle overlap
reached downstream consumers. Every cluster is now measured before it is
written, and rejected if it overlaps.

**Reproducibility.** Seeds came from ``hash()`` on a tuple containing a
string. Python randomizes string hashing per process unless
``PYTHONHASHSEED`` is fixed, so "deterministic_seed" was not deterministic
across runs. Seeds now come from ``hashlib.blake2b``.

**Runaway tasks.** No wall-clock budget was passed, so a single
infeasible combination could occupy a worker indefinitely. Each task now
carries ``max_runtime_seconds``.

Densification is deliberately not offered. It does not reach the
requested fractal dimension (measured ~0.5 low by the density-density
correlation function) and its overlap resolution does not converge except
for compressions so small they do essentially nothing. See
``docs/source/correlation_validation.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import socket
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
from pyfracval.quality import compute_aggregate_quality  # noqa: E402
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
#: Bounds a combination that never succeeds. Each attempt is already a
#: full run_simulation, which retries internally up to 20 times, so this
#: is a budget of retries-of-retries; 10 is generous for anything with a
#: non-trivial per-attempt success probability and keeps a hopeless
#: combination from occupying a worker for hours.
MAX_ATTEMPTS_PER_COMBO = 10
SCHEDULER_HOST = "marvin.bv.e-technik.tu-dortmund.de"
SCHEDULER_PORT = 8786
SCHEDULER_ADDRESS = f"tcp://{SCHEDULER_HOST}:{SCHEDULER_PORT}"
OUTPUT_BASE = PROJECT_ROOT / "cluster_data"
CONFIG_LABEL = "vanilla"

#: Per-task wall-clock budget. Bounds an infeasible combination instead of
#: letting it hold a worker until the campaign is abandoned.
TASK_TIMEOUT_S = 180.0

#: Workers write to their own filesystem and we discard it - the client
#: re-saves locally from the returned arrays, so the catalog lands on one
#: machine regardless of where the work ran.
_WORKER_OUTPUT_DIR = "/tmp/pyfracval_worker_output"

#: Env applied to every worker process. Thread limits matter here because
#: the cluster runs multi-threaded workers (4 threads/process): without
#: them, each of the 4 concurrent tasks would spawn its own BLAS/OpenMP
#: pool and oversubscribe the core count several times over.
_WORKER_ENV = {
    "PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deterministic_seed(
    sigma: float, df_val: float, n_val: int, attempt: int, label: str
) -> int:
    """Reproducible seed bound to (sigma, Df, N, attempt, config).

    Uses blake2b rather than ``hash()``: Python randomizes string hashing
    per process, so a ``hash()``-derived seed is not reproducible across
    runs - which defeats the point of naming it deterministic.
    """
    key = f"{sigma:.6g}|{df_val:.6g}|{n_val}|{attempt}|{label}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") % (
        2**31 - 1
    )


def parse_feasibility_csv(csv_path: Path) -> dict[tuple[float, float, int], float]:
    """Best-scoring kf per (sigma, Df, N) from a feasibility CSV."""
    combos: dict[tuple[float, float, int], list[tuple[float, float]]] = defaultdict(
        list
    )
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            key = (float(row["sigma"]), float(row["Df"]), int(row["N"]))
            combos[key].append((float(row["kf"]), float(row["success_rate"])))
    return {k: max(v, key=lambda x: x[1])[0] for k, v in combos.items()}


def build_combo_list(limit: int | None = None) -> list[dict]:
    """Union of the vanilla and densify feasibility grids.

    The densify grid is included deliberately even though densification
    itself is not used: those combinations were only reachable *via*
    densification when the sweeps were run, and backtracking pairing has
    since moved the feasibility boundary outward far enough that many are
    now reachable natively (docs/source/boundary_sweep_v2.md). Attempting
    them costs a bounded number of failed tasks and gains real coverage.
    """
    base = PROJECT_ROOT / "benchmark_results/plausibility"
    vanilla = parse_feasibility_csv(base / "wide_sweep_feasible_kf.csv")
    densify = parse_feasibility_csv(base / "wide_sweep_feasible_kf_densify_retry.csv")

    merged: dict[tuple[float, float, int], float] = dict(densify)
    merged.update(vanilla)  # prefer the kf the vanilla sweep liked

    combos = [
        {"sigma": s, "Df": d, "N": n, "kf": kf} for (s, d, n), kf in merged.items()
    ]
    combos.sort(key=lambda c: (c["N"], c["sigma"], c["Df"]))
    return combos[:limit] if limit else combos


def output_dir_for_combo(sigma: float, df_val: float, n_val: int, label: str) -> Path:
    sigma_str = f"sigma_{sigma:.2f}".replace(".", "p")
    df_str = f"Df_{df_val:.2f}".replace(".", "p")
    return OUTPUT_BASE / label / f"{sigma_str}__{df_str}__N_{n_val}"


def build_sim_config(combo: dict) -> dict:
    return {
        "N": combo["N"],
        "Df": combo["Df"],
        "kf": combo["kf"],
        "rp_g": 1.0,
        "rp_gstd": combo["sigma"],
        "tol_ov": 1e-6,
        "n_subcl_percentage": 0.1,
        "ext_case": 0,
    }


def save_cluster(
    coords: np.ndarray,
    radii: np.ndarray,
    combo: dict,
    out_dir: Path,
    iteration: int,
    seed: int,
    quality: dict,
) -> str:
    """Write one validated cluster, quality record included."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _mass, rg, cm, _r_max = calculate_cluster_properties(
        coords, radii, combo["Df"], combo["kf"]
    )

    sim_params = SimulationParameters(
        N=combo["N"],
        Df=combo["Df"],
        kf=combo["kf"],
        rp_g=1.0,
        rp_gstd=combo["sigma"],
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
            center_of_mass=cm.tolist() if cm is not None else None,
            n_particles_dropped=max(0, combo["N"] - int(coords.shape[0])),
            # Carried into the file so a consumer never has to trust the
            # catalog's success flag alone.
            max_residual_overlap=quality["max_residual_overlap"],
            n_overlapping_pairs=quality["n_overlapping_pairs"],
            overlap_ok=quality["overlap_ok"],
            measured_rg=quality["measured_rg"],
            rg_error_pct=quality["rg_error_pct"],
        ),
    )
    metadata.save_to_file(folderpath=str(out_dir), coords=coords, radii=radii)
    saved = sorted(out_dir.glob("fracval_*.dat"), key=lambda p: p.stat().st_mtime)
    return str(saved[-1]) if saved else str(out_dir)


def scheduler_reachable(timeout: float = 10.0) -> bool:
    try:
        with socket.create_connection((SCHEDULER_HOST, SCHEDULER_PORT), timeout):
            return True
    except OSError:
        return False


def workers_have_pyfracval(client) -> bool:
    """True when every worker can already import the package.

    Checked before attempting an install: registering an install plugin is
    a single blocking RPC that waits for every worker to finish a real pip
    install, and doing that unnecessarily against a large cluster is a
    good way to make the scheduler unresponsive.
    """

    def _probe():
        try:
            import pyfracval  # noqa: F401

            return True
        except Exception:
            return False

    try:
        results = client.run(_probe)
    except Exception as exc:
        logger.warning("Could not probe workers for pyfracval: %s", exc)
        return False
    return bool(results) and all(results.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", action="store_true", help="force a local cluster")
    ap.add_argument("--limit", type=int, help="only the first N combos (smoke test)")
    ap.add_argument("--clusters-per-combo", type=int, default=CLUSTERS_PER_COMBO)
    ap.add_argument("--label", default=CONFIG_LABEL)
    args = ap.parse_args()

    combos = build_combo_list(args.limit)
    target = len(combos) * args.clusters_per_combo
    logger.info(
        "%d combos x %d clusters = %d target",
        len(combos),
        args.clusters_per_combo,
        target,
    )

    use_remote = not args.local and scheduler_reachable()
    if not args.local and not use_remote:
        logger.warning(
            "Remote scheduler %s unreachable; falling back to a local cluster.",
            SCHEDULER_ADDRESS,
        )

    index_rows: list[dict] = []
    rejected: list[dict] = []
    t_start = time.time()

    client_kwargs = (
        {"scheduler_address": SCHEDULER_ADDRESS} if use_remote else {"n_workers": None}
    )
    with get_client(**client_kwargs) as client:
        if use_remote and not workers_have_pyfracval(client):
            logger.info("Workers lack pyfracval; installing the local wheel.")
            from pyfracval.dask_runner import _register_package

            _register_package(client)

        client.run(lambda: os.environ.update(_WORKER_ENV))

        n_slots = sum(
            w["nthreads"] for w in client.scheduler_info()["workers"].values()
        )
        # Keep roughly two tasks queued per slot: enough that a worker
        # never idles waiting for the client to submit, without building a
        # backlog so deep that finished work sits unclaimed.
        window = max(8, n_slots * 2)
        logger.info("Cluster offers %d slots; in-flight window %d", n_slots, window)

        # --- global work queue --------------------------------------------
        need = {i: args.clusters_per_combo for i in range(len(combos))}
        attempts = {i: 0 for i in range(len(combos))}
        # Queue combo indices only. The attempt number is assigned inside
        # submit(), at the moment the counter is incremented - deriving it
        # earlier let two failures for the same combo read the same "next"
        # value before either was submitted, producing duplicate attempt
        # numbers and therefore colliding output filenames.
        pending: list[int] = []
        for i in range(len(combos)):
            pending.extend([i] * args.clusters_per_combo)

        futures: dict = {}

        def submit(combo_idx: int):
            """Submit one attempt for a combo and return its future."""
            attempt = attempts[combo_idx]
            attempts[combo_idx] += 1
            combo = combos[combo_idx]
            seed = deterministic_seed(
                combo["sigma"], combo["Df"], combo["N"], attempt, args.label
            )
            fut = client.submit(
                run_simulation,
                attempt,
                build_sim_config(combo),
                _WORKER_OUTPUT_DIR,
                seed,
                TASK_TIMEOUT_S,
                pure=False,
            )
            futures[fut] = (combo_idx, attempt, seed)
            return fut

        for _ in range(min(window, len(pending))):
            submit(pending.pop(0))

        done = 0
        ac = as_completed(list(futures), with_results=False)
        for fut in ac:
            combo_idx, attempt, seed = futures.pop(fut)
            combo = combos[combo_idx]
            try:
                success, coords, radii = fut.result()
            except Exception as exc:
                success, coords, radii = False, None, None
                logger.debug("task raised: %s", exc)

            accepted = False
            if success and coords is not None and radii is not None:
                q = compute_aggregate_quality(
                    coords, radii, combo["Df"], combo["kf"], 1e-6
                )
                if q["overlap_ok"]:
                    out_dir = output_dir_for_combo(
                        combo["sigma"], combo["Df"], combo["N"], args.label
                    )
                    path = save_cluster(coords, radii, combo, out_dir, attempt, seed, q)
                    index_rows.append(
                        {
                            "config": args.label,
                            "sigma": combo["sigma"],
                            "Df": combo["Df"],
                            "N": combo["N"],
                            "kf": combo["kf"],
                            "attempt": attempt,
                            "seed": seed,
                            "success": True,
                            "n_particles": q["n_particles"],
                            "max_residual_overlap": q["max_residual_overlap"],
                            "n_overlapping_pairs": q["n_overlapping_pairs"],
                            "measured_rg": q["measured_rg"],
                            "rg_error_pct": q["rg_error_pct"],
                            "filepath": path,
                        }
                    )
                    need[combo_idx] -= 1
                    accepted = True
                else:
                    # Reached the requested particle count but the geometry
                    # is not physically valid. Recording these separately
                    # rather than dropping them keeps the rejection rate
                    # visible instead of silently shrinking the catalog.
                    rejected.append(
                        {
                            "sigma": combo["sigma"],
                            "Df": combo["Df"],
                            "N": combo["N"],
                            "kf": combo["kf"],
                            "seed": seed,
                            "max_residual_overlap": q["max_residual_overlap"],
                            "n_overlapping_pairs": q["n_overlapping_pairs"],
                        }
                    )

            # Queue a replacement attempt if this combo still needs one.
            if (
                not accepted
                and need[combo_idx] > 0
                and attempts[combo_idx] < MAX_ATTEMPTS_PER_COMBO
            ):
                pending.append(combo_idx)

            while pending and len(futures) < window:
                idx = pending.pop(0)
                if need[idx] <= 0 or attempts[idx] >= MAX_ATTEMPTS_PER_COMBO:
                    continue
                ac.add(submit(idx))

            done += 1
            if done % 100 == 0:
                remaining = sum(v for v in need.values() if v > 0)
                logger.info(
                    "%d tasks done | %d clusters saved | %d still needed | %d in flight",
                    done,
                    len(index_rows),
                    remaining,
                    len(futures),
                )

    elapsed = time.time() - t_start

    # --- write the catalog --------------------------------------------------
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    index_csv = OUTPUT_BASE / "cluster_index.csv"
    if index_rows:
        with index_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
    (OUTPUT_BASE / "cluster_index.json").write_text(json.dumps(index_rows, indent=2))
    if rejected:
        (OUTPUT_BASE / "rejected_clusters.json").write_text(
            json.dumps(rejected, indent=2)
        )

    complete = sum(1 for i in range(len(combos)) if need[i] <= 0)
    print("\n" + "=" * 66)
    print("CLUSTER GENERATION COMPLETE")
    print("=" * 66)
    print(f"  combos fully satisfied : {complete}/{len(combos)}")
    print(f"  clusters saved         : {len(index_rows)}/{target}")
    print(f"  rejected for overlap   : {len(rejected)}")
    print(f"  elapsed                : {elapsed / 60:.1f} min")
    print(f"  catalog                : {index_csv}")
    print("=" * 66)


if __name__ == "__main__":
    main()
