"""Run stability sweeps over (Df, kf, N, rp_gstd).

This script builds on benchmarks/sticking_benchmark.py to measure
success rates for aggregate generation across parameter grids.

Dask mode (--dask):
    Trials for *each* parameter combination are dispatched as individual
    Dask tasks and collected with ``as_completed``.  Use ``--dask-scheduler``
    to connect to a running scheduler; otherwise a local cluster is started.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List

import numpy as np

from benchmarks.sticking_benchmark import StickingBenchmark


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _powers_of_two(max_n: int) -> List[int]:
    sizes = []
    n = 1
    while n <= max_n:
        sizes.append(n)
        n *= 2
    return sizes


def _float_grid(min_val: float, max_val: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    values = np.arange(min_val, max_val + (step / 10.0), step)
    return [float(f"{val:.6f}") for val in values]


def _format_param(value: float) -> str:
    return f"{value:.6g}"


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Df/kf/N/rp_gstd to identify stable parameter regions."
    )

    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Base output directory (default: benchmark_results)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of trials per parameter set (default: 30)",
    )
    parser.add_argument(
        "--sizes",
        default="powers-of-two",
        help=(
            "Comma-separated list of N values, or 'powers-of-two' "
            "(default: powers-of-two up to --max-size)."
        ),
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Maximum N when using powers-of-two (default: 1024)",
    )
    parser.add_argument(
        "--sigmas",
        default="1.0,1.5,2.0,2.5,3.0",
        help="Comma-separated list of rp_gstd values (default: 1.0,1.5,2.0,2.5,3.0)",
    )
    parser.add_argument("--df-min", type=float, default=1.4)
    parser.add_argument("--df-max", type=float, default=2.6)
    parser.add_argument("--df-step", type=float, default=0.2)
    parser.add_argument("--kf-min", type=float, default=0.6)
    parser.add_argument("--kf-max", type=float, default=1.6)
    parser.add_argument("--kf-step", type=float, default=0.2)
    parser.add_argument(
        "--df-values",
        default=None,
        help="Optional comma-separated Df values (overrides df-min/max/step)",
    )
    parser.add_argument(
        "--kf-values",
        default=None,
        help="Optional comma-separated kf values (overrides kf-min/max/step)",
    )
    parser.add_argument("--rp-g", type=float, default=100.0)
    parser.add_argument("--tol-ov", type=float, default=1e-6)
    parser.add_argument("--n-subcl-percentage", type=float, default=0.1)
    parser.add_argument("--ext-case", type=int, default=0)
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw per-trial results to JSONL",
    )

    parser.add_argument(
        "--trial-timeout",
        type=int,
        default=None,
        help=(
            "Wall-clock timeout in seconds per trial (default: no limit). "
            "Timed-out trials are counted as failures."
        ),
    )

    # --- Dask options ---
    parser.add_argument(
        "--dask",
        action="store_true",
        help="Distribute trials across a Dask cluster.",
    )
    parser.add_argument(
        "--dask-scheduler",
        default=None,
        metavar="URL",
        help=(
            "Address of a running Dask scheduler "
            "(e.g. tcp://host:8786).  "
            "When omitted, a LocalCluster is started."
        ),
    )
    parser.add_argument(
        "--dask-workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of workers for the local Dask cluster (default: all CPUs).",
    )

    return parser


def _run_sweep_sequential(
    args, sizes, sigmas, df_values, kf_values, benchmark, sweep_rows, raw_handle
):
    """Inner loop for the sequential (non-Dask) sweep."""
    total_combos = len(sizes) * len(sigmas) * len(df_values) * len(kf_values)
    combo_index = 0

    for n_val in sizes:
        for rp_gstd in sigmas:
            for df_val in df_values:
                for kf_val in kf_values:
                    combo_index += 1
                    label = (
                        f"N={n_val},Df={_format_param(df_val)},"
                        f"kf={_format_param(kf_val)},rp_gstd={_format_param(rp_gstd)}"
                    )
                    print(f"[{combo_index}/{total_combos}] {label}")

                    params = {
                        "N": n_val,
                        "Df": df_val,
                        "kf": kf_val,
                        "rp_gstd": rp_gstd,
                        "rp_g": args.rp_g,
                        "tol_ov": args.tol_ov,
                        "n_subcl_percentage": args.n_subcl_percentage,
                        "ext_case": args.ext_case,
                        "description": label,
                    }

                    case_results = []
                    for trial in range(args.trials):
                        result = benchmark.run_single_trial(
                            params,
                            trial_num=trial,
                            category="stability_sweep",
                            trial_timeout=args.trial_timeout,
                        )
                        case_results.append(result)

                        if raw_handle is not None:
                            raw_record = asdict(result)
                            raw_handle.write(json.dumps(raw_record))
                            raw_handle.write("\n")

                    _append_sweep_row(
                        sweep_rows,
                        n_val,
                        df_val,
                        kf_val,
                        rp_gstd,
                        args.trials,
                        case_results,
                    )


def _run_sweep_dask(
    args, sizes, sigmas, df_values, kf_values, benchmark, sweep_rows, raw_handle
):
    """Inner loop for the Dask-distributed sweep."""
    from dask.distributed import as_completed as dask_as_completed

    from pyfracval.dask_runner import get_client
    from pyfracval.main_runner import run_simulation

    total_combos = len(sizes) * len(sigmas) * len(df_values) * len(kf_values)
    total_trials = total_combos * args.trials
    print(
        f"Submitting {total_trials} tasks across {total_combos} combos "
        f"to Dask ({args.dask_scheduler or 'local cluster'}) …"
    )

    with get_client(
        scheduler_address=args.dask_scheduler,
        n_workers=args.dask_workers,
    ) as client:
        # Build a flat list of (combo_key, trial) → future
        combo_futures: dict = {}  # future → (combo_key, trial_index)
        combo_params: dict = {}  # combo_key → params dict

        combo_index = 0
        for n_val in sizes:
            for rp_gstd in sigmas:
                for df_val in df_values:
                    for kf_val in kf_values:
                        combo_index += 1
                        label = (
                            f"N={n_val},Df={_format_param(df_val)},"
                            f"kf={_format_param(kf_val)},rp_gstd={_format_param(rp_gstd)}"
                        )
                        params = {
                            "N": n_val,
                            "Df": df_val,
                            "kf": kf_val,
                            "rp_gstd": rp_gstd,
                            "rp_g": args.rp_g,
                            "tol_ov": args.tol_ov,
                            "n_subcl_percentage": args.n_subcl_percentage,
                            "ext_case": args.ext_case,
                            "description": label,
                        }
                        combo_key = (n_val, rp_gstd, df_val, kf_val)
                        combo_params[combo_key] = params

                        for trial in range(args.trials):
                            seed = abs(
                                hash((n_val, df_val, kf_val, rp_gstd, trial))
                            ) % (2**31)
                            fut = client.submit(
                                run_simulation,
                                trial,
                                params,
                                "/tmp/dask_sweep_output",
                                seed,
                                args.trial_timeout,
                            )
                            combo_futures[fut] = (combo_key, trial)

        # Collect results
        combo_results: dict = {k: [] for k in combo_params}

        try:
            from tqdm import tqdm as _tqdm

            completed_iter = _tqdm(
                dask_as_completed(combo_futures),
                total=total_trials,
                desc="Trials",
            )
        except ImportError:
            completed_iter = dask_as_completed(combo_futures)

        for future in completed_iter:
            combo_key, trial = combo_futures[future]
            try:
                success, _coords, _radii = future.result()
            except Exception:
                success = False
            combo_results[combo_key].append(success)

        # Build sweep rows in original order
        combo_index = 0
        for n_val in sizes:
            for rp_gstd in sigmas:
                for df_val in df_values:
                    for kf_val in kf_values:
                        combo_index += 1
                        combo_key = (n_val, rp_gstd, df_val, kf_val)
                        successes_list = combo_results[combo_key]
                        successes = sum(successes_list)
                        label = combo_params[combo_key]["description"]
                        print(
                            f"[{combo_index}/{total_combos}] {label} → "
                            f"{successes}/{args.trials} success"
                        )
                        sweep_rows.append(
                            {
                                "N": n_val,
                                "Df": df_val,
                                "kf": kf_val,
                                "rp_gstd": rp_gstd,
                                "trials": args.trials,
                                "successes": successes,
                                "success_rate": successes / args.trials,
                                "avg_runtime_s": 0.0,
                                "median_runtime_s": 0.0,
                                "failure_stage_counts": {},
                            }
                        )


def _append_sweep_row(
    sweep_rows, n_val, df_val, kf_val, rp_gstd, n_trials, case_results
):
    successes = sum(1 for r in case_results if r.success)
    runtimes = [r.runtime_seconds for r in case_results]
    failure_stages = [r.failure_stage or "NONE" for r in case_results if not r.success]
    stage_counts = {
        stage: failure_stages.count(stage) for stage in sorted(set(failure_stages))
    }
    sweep_rows.append(
        {
            "N": n_val,
            "Df": df_val,
            "kf": kf_val,
            "rp_gstd": rp_gstd,
            "trials": n_trials,
            "successes": successes,
            "success_rate": successes / n_trials,
            "avg_runtime_s": mean(runtimes),
            "median_runtime_s": median(runtimes),
            "failure_stage_counts": stage_counts,
        }
    )


def main() -> int:
    args = _build_parser().parse_args()

    if args.df_values:
        df_values = _parse_float_list(args.df_values)
    else:
        df_values = _float_grid(args.df_min, args.df_max, args.df_step)

    if args.kf_values:
        kf_values = _parse_float_list(args.kf_values)
    else:
        kf_values = _float_grid(args.kf_min, args.kf_max, args.kf_step)

    if args.sizes == "powers-of-two":
        sizes = _powers_of_two(args.max_size)
    else:
        sizes = _parse_int_list(args.sizes)

    sigmas = _parse_float_list(args.sigmas)

    output_root = Path(args.output_dir)
    summary_dir = output_root / "stability_sweeps"
    summary_dir.mkdir(parents=True, exist_ok=True)

    benchmark = StickingBenchmark(output_dir=str(output_root))

    total_combos = len(sizes) * len(sigmas) * len(df_values) * len(kf_values)
    print(
        "Running stability sweep with "
        f"{total_combos} parameter combinations and {args.trials} trials each."
    )

    sweep_rows: list = []
    raw_path = summary_dir / "stability_sweep_raw.jsonl"
    raw_handle = raw_path.open("w", encoding="utf-8") if args.save_raw else None

    try:
        sweep_start = time.time()

        if getattr(args, "dask", False):
            _run_sweep_dask(
                args,
                sizes,
                sigmas,
                df_values,
                kf_values,
                benchmark,
                sweep_rows,
                raw_handle,
            )
        else:
            _run_sweep_sequential(
                args,
                sizes,
                sigmas,
                df_values,
                kf_values,
                benchmark,
                sweep_rows,
                raw_handle,
            )

        sweep_runtime = time.time() - sweep_start
        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_combinations": total_combos,
            "trials_per_combo": args.trials,
            "sizes": sizes,
            "sigmas": sigmas,
            "df_values": df_values,
            "kf_values": kf_values,
            "total_runtime_s": sweep_runtime,
            "results": sweep_rows,
        }

        _write_json(summary_dir / "stability_sweep_summary.json", summary)

        csv_rows = []
        for row in sweep_rows:
            csv_rows.append(
                {
                    "N": row["N"],
                    "Df": row["Df"],
                    "kf": row["kf"],
                    "rp_gstd": row["rp_gstd"],
                    "trials": row["trials"],
                    "successes": row["successes"],
                    "success_rate": row["success_rate"],
                    "avg_runtime_s": row["avg_runtime_s"],
                    "median_runtime_s": row["median_runtime_s"],
                    "failure_stage_counts": json.dumps(row["failure_stage_counts"]),
                }
            )

        _write_csv(
            summary_dir / "stability_sweep_summary.csv",
            csv_rows,
            fieldnames=[
                "N",
                "Df",
                "kf",
                "rp_gstd",
                "trials",
                "successes",
                "success_rate",
                "avg_runtime_s",
                "median_runtime_s",
                "failure_stage_counts",
            ],
        )

        print("Sweep complete.")
        print(f"Summary: {summary_dir / 'stability_sweep_summary.json'}")
        print(f"CSV: {summary_dir / 'stability_sweep_summary.csv'}")
        if raw_handle is not None:
            print(f"Raw: {raw_path}")

    finally:
        if raw_handle is not None:
            raw_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
