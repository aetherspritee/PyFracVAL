#!/usr/bin/env python3
"""Unified benchmark for local-vs-remote Dask execution.

This benchmark runs the same task bundle in two environments:
1) Local Dask cluster
2) Remote Dask scheduler (or a loopback "remote-like" local cluster)

It reports both raw throughput and normalized metrics:
- Throughput per thread
- Throughput per effective thread (weighted by calibration performance)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dask.distributed import LocalCluster, as_completed, get_worker, performance_report

from pyfracval.dask_runner import get_client
from pyfracval.main_runner import run_simulation


def _git_commit_short() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _worker_info() -> dict[str, str | int]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "cpu_count": os.cpu_count() or 1,
    }


def _timed_run(
    trial_index: int,
    config: dict[str, Any],
    output_base_dir: str,
    seed: int,
    trial_timeout: float | None,
) -> dict[str, Any]:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    success, _coords, _radii = run_simulation(
        trial_index,
        config,
        output_base_dir,
        seed,
        trial_timeout,
    )

    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    try:
        worker_addr = get_worker().address
    except Exception:
        worker_addr = "unknown"

    return {
        "success": bool(success),
        "wall_s": end_wall - start_wall,
        "cpu_s": end_cpu - start_cpu,
        "worker": worker_addr,
    }


def _run_calibration(
    client,
    worker_threads: dict[str, int],
    seed_start: int,
) -> dict[str, float]:
    """Run one lightweight calibration task per worker.

    Returns mapping worker_address -> wall_seconds.
    """
    calibration_cfg = {
        "N": 128,
        "Df": 1.8,
        "kf": 1.0,
        "rp_g": 100.0,
        "rp_gstd": 1.5,
        "tol_ov": 1e-6,
        "n_subcl_percentage": 0.1,
        "ext_case": 0,
    }

    fut_to_worker: dict[Any, str] = {}
    for idx, worker_addr in enumerate(worker_threads):
        fut = client.submit(
            _timed_run,
            -1000 - idx,
            calibration_cfg,
            "/tmp/pyfracval_calibration",
            seed_start + idx,
            None,
            workers=[worker_addr],
            allow_other_workers=False,
            pure=False,
        )
        fut_to_worker[fut] = worker_addr

    calib: dict[str, float] = {}
    for fut in as_completed(fut_to_worker):
        worker_addr = fut_to_worker[fut]
        result = fut.result()
        calib[worker_addr] = float(result["wall_s"])

    return calib


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _run_environment_benchmark(
    env_name: str,
    client,
    config: dict[str, Any],
    n_aggregates: int,
    warmup_tasks: int,
    seed_start: int,
    trial_timeout: float | None,
    profile_output_dir: Path | None = None,
) -> dict[str, Any]:
    scheduler_info = client.scheduler_info()
    worker_threads = {
        addr: int(info.get("nthreads", 1))
        for addr, info in scheduler_info.get("workers", {}).items()
    }
    total_threads = sum(worker_threads.values())

    worker_meta = client.run(_worker_info, workers=list(worker_threads.keys()))

    # Disable nested multiprocessing in Dask workers.
    client.run(
        lambda: os.environ.__setitem__("PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS", "1")
    )

    profile_dir = profile_output_dir / env_name if profile_output_dir else None
    report_path = profile_dir / "dask_performance_report.html" if profile_dir else None

    def _run_workload() -> tuple[
        dict[str, float], dict[str, float], float, list[dict[str, Any]], float
    ]:
        # Warmup tasks to reduce JIT cold-start noise.
        warmup_futures = [
            client.submit(
                _timed_run,
                -1 - i,
                config,
                f"/tmp/pyfracval_unified_{env_name}_warmup",
                seed_start + i,
                trial_timeout,
                pure=False,
            )
            for i in range(warmup_tasks)
        ]
        for fut in as_completed(warmup_futures):
            fut.result()

        calibration = _run_calibration(client, worker_threads, seed_start + 10_000)
        calib_times = [t for t in calibration.values() if t > 0]
        calib_ref = _median(calib_times) if calib_times else 1.0

        worker_perf_index: dict[str, float] = {}
        for worker_addr in worker_threads:
            w_time = calibration.get(worker_addr)
            if w_time is None or w_time <= 0:
                worker_perf_index[worker_addr] = 1.0
            else:
                worker_perf_index[worker_addr] = calib_ref / w_time

        futures = {}
        bench_start = time.perf_counter()
        for i in range(n_aggregates):
            seed = seed_start + 100_000 + i
            fut = client.submit(
                _timed_run,
                i,
                config,
                f"/tmp/pyfracval_unified_{env_name}_run",
                seed,
                trial_timeout,
                pure=False,
            )
            futures[fut] = i

        task_results: list[dict[str, Any]] = []
        for fut in as_completed(futures):
            task_results.append(fut.result())

        bench_end = time.perf_counter()
        return calibration, worker_perf_index, bench_start, task_results, bench_end

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with performance_report(filename=str(report_path)):
            calibration, worker_perf_index, bench_start, task_results, bench_end = (
                _run_workload()
            )
    else:
        calibration, worker_perf_index, bench_start, task_results, bench_end = (
            _run_workload()
        )

    effective_threads = sum(
        worker_threads[w] * worker_perf_index.get(w, 1.0) for w in worker_threads
    )

    total_wall_s = bench_end - bench_start

    successes = sum(1 for r in task_results if r["success"])
    task_walls = [float(r["wall_s"]) for r in task_results]
    task_cpus = [float(r["cpu_s"]) for r in task_results]

    throughput = n_aggregates / total_wall_s if total_wall_s > 0 else 0.0
    throughput_per_thread = throughput / total_threads if total_threads > 0 else 0.0
    throughput_per_effective_thread = (
        throughput / effective_threads if effective_threads > 0 else 0.0
    )

    # Fraction of ideal thread-capacity consumed by task wall-time.
    thread_capacity_s = total_threads * total_wall_s if total_threads > 0 else 0.0
    busy_fraction = (
        sum(task_walls) / thread_capacity_s if thread_capacity_s > 0 else 0.0
    )

    profiling_artifacts: dict[str, str] = {}
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        if report_path is not None and report_path.exists():
            profiling_artifacts["performance_report_html"] = str(report_path)

        profile_json = profile_dir / "client_profile.json"
        try:
            _write_json(profile_json, client.profile())
            profiling_artifacts["client_profile_json"] = str(profile_json)
        except Exception as exc:
            profiling_artifacts["client_profile_error"] = str(exc)

        transfer_json = profile_dir / "transfer_logs.json"
        try:
            incoming = client.run(
                lambda dask_worker: list(dask_worker.transfer_incoming_log)[-500:]
            )
            outgoing = client.run(
                lambda dask_worker: list(dask_worker.transfer_outgoing_log)[-500:]
            )
            _write_json(transfer_json, {"incoming": incoming, "outgoing": outgoing})
            profiling_artifacts["transfer_logs_json"] = str(transfer_json)
        except Exception as exc:
            profiling_artifacts["transfer_logs_error"] = str(exc)

        scheduler_json = profile_dir / "scheduler_info.json"
        try:
            _write_json(scheduler_json, client.scheduler_info())
            profiling_artifacts["scheduler_info_json"] = str(scheduler_json)
        except Exception as exc:
            profiling_artifacts["scheduler_info_error"] = str(exc)

    return {
        "environment": env_name,
        "workers": len(worker_threads),
        "total_threads": total_threads,
        "effective_threads": effective_threads,
        "worker_threads": worker_threads,
        "worker_metadata": worker_meta,
        "worker_calibration_wall_s": calibration,
        "worker_performance_index": worker_perf_index,
        "n_aggregates": n_aggregates,
        "successes": successes,
        "success_rate": successes / n_aggregates if n_aggregates > 0 else 0.0,
        "total_wall_s": total_wall_s,
        "throughput_agg_per_s": throughput,
        "throughput_per_thread": throughput_per_thread,
        "throughput_per_effective_thread": throughput_per_effective_thread,
        "task_wall_mean_s": _mean(task_walls),
        "task_wall_median_s": _median(task_walls),
        "task_cpu_mean_s": _mean(task_cpus),
        "task_cpu_median_s": _median(task_cpus),
        "cpu_to_wall_ratio": (sum(task_cpus) / sum(task_walls))
        if sum(task_walls) > 0
        else 0.0,
        "thread_busy_fraction": busy_fraction,
        "profiling_artifacts": profiling_artifacts,
    }


def _build_config(args) -> dict[str, Any]:
    return {
        "N": args.n,
        "Df": args.df,
        "kf": args.kf,
        "rp_g": args.rp_g,
        "rp_gstd": args.rp_gstd,
        "tol_ov": args.tol_ov,
        "n_subcl_percentage": args.n_subcl_percentage,
        "ext_case": args.ext_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified local/remote Dask benchmark")
    parser.add_argument("--scheduler-address", type=str, default=None)
    parser.add_argument(
        "--remote-loopback-workers",
        type=int,
        default=0,
        help="Create a separate local cluster and treat it as remote for smoke tests",
    )
    parser.add_argument("--local-workers", type=int, default=None)
    parser.add_argument("--n-aggregates", type=int, default=12)
    parser.add_argument("--warmup-tasks", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=1431354440)
    parser.add_argument("--trial-timeout", type=float, default=None)

    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--df", type=float, default=1.8)
    parser.add_argument("--kf", type=float, default=1.0)
    parser.add_argument("--rp-g", dest="rp_g", type=float, default=100.0)
    parser.add_argument("--rp-gstd", dest="rp_gstd", type=float, default=1.5)
    parser.add_argument("--tol-ov", dest="tol_ov", type=float, default=1e-6)
    parser.add_argument(
        "--n-subcl-percentage", dest="n_subcl_percentage", type=float, default=0.1
    )
    parser.add_argument("--ext-case", dest="ext_case", type=int, default=0)

    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON path (default: benchmark_results/unified_local_remote_<timestamp>.json)",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="",
        help="Optional directory to save Dask profiling artifacts",
    )

    args = parser.parse_args()
    config = _build_config(args)

    print("=" * 80)
    print("UNIFIED LOCAL/REMOTE DASK BENCHMARK")
    print("=" * 80)
    print(f"Config: {config}")
    print(f"Aggregates per environment: {args.n_aggregates}")
    print(f"Warmup tasks: {args.warmup_tasks}")
    print("=" * 80)

    commit = _git_commit_short()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    profile_dir = Path(args.profile_dir) if args.profile_dir else None

    local_result: dict[str, Any]
    remote_result: dict[str, Any] | None = None

    with get_client(
        scheduler_address=None,
        n_workers=args.local_workers,
        install_package=False,
    ) as local_client:
        print("Running local benchmark...")
        local_result = _run_environment_benchmark(
            "local",
            local_client,
            config,
            args.n_aggregates,
            args.warmup_tasks,
            args.seed_start,
            args.trial_timeout,
            profile_output_dir=profile_dir,
        )

    loopback_cluster: LocalCluster | None = None
    remote_address = args.scheduler_address

    if remote_address is None and args.remote_loopback_workers > 0:
        loopback_cluster = LocalCluster(
            n_workers=args.remote_loopback_workers,
            threads_per_worker=1,
            processes=True,
        )
        remote_address = loopback_cluster.scheduler_address

    if remote_address is not None:
        with get_client(
            scheduler_address=remote_address,
            install_package=True,
        ) as remote_client:
            print(f"Running remote benchmark via {remote_address}...")
            remote_result = _run_environment_benchmark(
                "remote",
                remote_client,
                config,
                args.n_aggregates,
                args.warmup_tasks,
                args.seed_start,
                args.trial_timeout,
                profile_output_dir=profile_dir,
            )

    if loopback_cluster is not None:
        loopback_cluster.close()

    comparison: dict[str, float] = {}
    if remote_result is not None:
        local_tp = local_result["throughput_agg_per_s"]
        remote_tp = remote_result["throughput_agg_per_s"]
        local_tpt = local_result["throughput_per_thread"]
        remote_tpt = remote_result["throughput_per_thread"]
        local_tpe = local_result["throughput_per_effective_thread"]
        remote_tpe = remote_result["throughput_per_effective_thread"]

        comparison = {
            "remote_vs_local_raw_throughput_ratio": (
                remote_tp / local_tp if local_tp > 0 else 0.0
            ),
            "remote_vs_local_per_thread_ratio": (
                remote_tpt / local_tpt if local_tpt > 0 else 0.0
            ),
            "remote_vs_local_per_effective_thread_ratio": (
                remote_tpe / local_tpe if local_tpe > 0 else 0.0
            ),
            "local_threads": float(local_result["total_threads"]),
            "remote_threads": float(remote_result["total_threads"]),
            "local_effective_threads": float(local_result["effective_threads"]),
            "remote_effective_threads": float(remote_result["effective_threads"]),
        }

    payload = {
        "benchmark": "unified_local_remote_dask",
        "commit": commit,
        "started_at": started_at,
        "config": config,
        "n_aggregates": args.n_aggregates,
        "warmup_tasks": args.warmup_tasks,
        "local": local_result,
        "remote": remote_result,
        "comparison": comparison,
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"benchmark_results/unified_local_remote_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(
        f"Local:  {local_result['throughput_agg_per_s']:.3f} agg/s | "
        f"per-thread={local_result['throughput_per_thread']:.5f} | "
        f"per-effective-thread={local_result['throughput_per_effective_thread']:.5f}"
    )
    if remote_result is not None:
        print(
            f"Remote: {remote_result['throughput_agg_per_s']:.3f} agg/s | "
            f"per-thread={remote_result['throughput_per_thread']:.5f} | "
            f"per-effective-thread={remote_result['throughput_per_effective_thread']:.5f}"
        )
        print(
            "Ratios (remote/local): "
            f"raw={comparison['remote_vs_local_raw_throughput_ratio']:.3f}, "
            f"per-thread={comparison['remote_vs_local_per_thread_ratio']:.3f}, "
            f"per-effective-thread={comparison['remote_vs_local_per_effective_thread_ratio']:.3f}"
        )
    else:
        print(
            "Remote benchmark skipped (no --scheduler-address and no --remote-loopback-workers)."
        )

    print(f"Saved: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
