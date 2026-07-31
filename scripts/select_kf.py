#!/usr/bin/env python3
"""Choose a feasible kf for every (sigma, Df, N) cell, by measurement.

The generation grid is over ``(sigma, Df, N)``; ``kf`` is not a swept
axis but a value picked per cell so the cell is generable at all. The
previous selection came from a sweep run against an older implementation,
which left the current one with gaps - some cells were assigned a kf that
no longer works, and others a kf that never worked and only appeared to
because of the overlap-acceptance defect fixed earlier.

This re-measures the choice against the current code, with one rule
applied uniformly to every cell so the resulting catalog is consistent
rather than a mix of provenances:

    among the kf values that reach the required success rate,
    take the one closest to kf = 1.0

Preferring 1.0 keeps the catalog near the physically typical prefactor
and lets the boundary decide the rest: low Df needs a larger kf (it
shrinks Rg, and therefore the centre-of-mass separation two subclusters
must span), high Df needs a smaller one (it enlarges Rg, giving the
sticking search room to avoid overlap). The rule expresses that without
hard-coding either direction.

Usage:
    devenv shell -- uv run python scripts/select_kf.py
    devenv shell -- uv run python scripts/select_kf.py --trials 5 --out my.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dask.distributed import as_completed  # noqa: E402

from pyfracval.dask_runner import get_client  # noqa: E402
from pyfracval.main_runner import run_simulation  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

#: Physically reasonable prefactors. Anything outside this would make the
#: catalog hard to defend regardless of whether it generates.
#: Moran et al. explore kf from 0.1 to 2.7; this ladder stays inside that
#: published range. The upper end matters: low Df at large N needs a large
#: prefactor, because Rg = a (n/kf)^(1/Df) means raising kf shrinks Rg and
#: with it the separation two subclusters must span. Cutting the ladder at
#: 1.8 left six such cells unreachable purely as a ladder artefact.
KF_LADDER = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7]
KF_TARGET = 1.0
PROBE_TIMEOUT_S = 90.0


def probe(args) -> tuple[tuple, float, bool]:
    sigma, df, n, kf, seed = args
    ok, _, _ = run_simulation(
        iteration=seed,
        sim_config_dict={
            "N": n,
            "Df": df,
            "kf": kf,
            "rp_g": 1.0,
            "rp_gstd": sigma,
            "tol_ov": 1e-6,
            "n_subcl_percentage": 0.1,
            "ext_case": 0,
            "seed": seed,
        },
        output_base_dir="/tmp/pyfracval_kfprobe",
        max_runtime_seconds=PROBE_TIMEOUT_S,
    )
    return (sigma, df, n), kf, bool(ok)


def cells_from_existing_grids() -> list[tuple[float, float, int]]:
    """The (sigma, Df, N) cells the catalog is meant to cover."""
    base = PROJECT_ROOT / "benchmark_results/plausibility"
    cells: set[tuple[float, float, int]] = set()
    for name in (
        "wide_sweep_feasible_kf.csv",
        "wide_sweep_feasible_kf_densify_retry.csv",
    ):
        with (base / name).open() as fh:
            for row in csv.DictReader(fh):
                cells.add((float(row["sigma"]), float(row["Df"]), int(row["N"])))
    return sorted(cells, key=lambda c: (c[2], c[0], c[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=3, help="probe trials per (cell, kf)")
    ap.add_argument(
        "--require",
        type=float,
        default=1.0,
        help="required success fraction for a kf to be accepted (default: all)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results/plausibility/selected_kf.csv",
    )
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cells = cells_from_existing_grids()
    if args.limit:
        cells = cells[: args.limit]
    jobs = [
        (s, d, n, kf, seed)
        for (s, d, n) in cells
        for kf in KF_LADDER
        for seed in range(1, args.trials + 1)
    ]
    logger.info(
        "%d cells x %d kf x %d trials = %d probes",
        len(cells),
        len(KF_LADDER),
        args.trials,
        len(jobs),
    )

    results: dict[tuple, dict[float, list[bool]]] = {c: {} for c in cells}
    t0 = time.time()

    with get_client() as client:
        # Same worker environment the generator uses. Without
        # PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS, PCA spawns a
        # multiprocessing.Pool inside the Dask worker as soon as a cell
        # has enough subclusters (>=4), which fails outright - silently
        # reporting every such cell as infeasible rather than as broken.
        client.run(
            lambda: os.environ.update(
                {
                    "PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
        )
        n_slots = sum(
            w["nthreads"] for w in client.scheduler_info()["workers"].values()
        )
        window = max(8, n_slots * 2)
        logger.info("%d slots, in-flight window %d", n_slots, window)

        pending = list(jobs)
        futures = {}

        def submit(job):
            fut = client.submit(probe, job, pure=False)
            futures[fut] = job
            return fut

        for _ in range(min(window, len(pending))):
            submit(pending.pop(0))

        ac = as_completed(list(futures))
        done = 0
        for fut in ac:
            futures.pop(fut, None)
            try:
                cell, kf, ok = fut.result()
                results[cell].setdefault(kf, []).append(ok)
            except Exception as exc:
                logger.debug("probe failed: %s", exc)
            while pending and len(futures) < window:
                ac.add(submit(pending.pop(0)))
            done += 1
            if done % 500 == 0:
                logger.info("%d/%d probes done", done, len(jobs))

    # --- selection ---------------------------------------------------------
    rows = []
    unfillable = []
    for cell in cells:
        sigma, df, n = cell
        viable = [
            (kf, sum(v) / len(v))
            for kf, v in results[cell].items()
            if v and sum(v) / len(v) >= args.require
        ]
        if not viable:
            # Fall back to whatever did best, so the cell is at least
            # attempted during generation rather than dropped silently.
            best = max(
                ((kf, sum(v) / len(v)) for kf, v in results[cell].items() if v),
                key=lambda x: x[1],
                default=(KF_TARGET, 0.0),
            )
            unfillable.append((cell, best))
            chosen, rate = best
        else:
            chosen, rate = min(viable, key=lambda x: (abs(x[0] - KF_TARGET), -x[1]))
        rows.append(
            {
                "sigma": sigma,
                "Df": df,
                "kf": chosen,
                "N": n,
                "success_rate": rate,
                "trials": args.trials,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 66)
    print("kf SELECTION COMPLETE")
    print("=" * 66)
    print(f"  cells                    : {len(cells)}")
    print(f"  cells with a viable kf   : {len(cells) - len(unfillable)}")
    print(f"  cells with none          : {len(unfillable)}")
    print(f"  elapsed                  : {(time.time() - t0) / 60:.1f} min")
    print(f"  written                  : {args.out}")
    if unfillable:
        print("\n  cells with no fully-successful kf (best effort recorded):")
        for cell, (kf, rate) in unfillable[:20]:
            print(
                f"    sigma={cell[0]} Df={cell[1]} N={cell[2]} -> kf={kf} ({rate:.0%})"
            )
    print("=" * 66)


if __name__ == "__main__":
    main()
