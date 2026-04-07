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
import concurrent.futures
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dask.distributed import LocalCluster, as_completed, get_worker, performance_report

from pyfracval import config as runtime_config
from pyfracval.config import OrchestratorConfig
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
    n_val = 256 if args.n is None else args.n
    df_val = 1.8 if args.df is None else args.df
    kf_val = 1.0 if args.kf is None else args.kf
    rp_g_val = 100.0 if args.rp_g is None else args.rp_g
    rp_gstd_val = 1.5 if args.rp_gstd is None else args.rp_gstd
    tol_ov_val = 1e-6 if args.tol_ov is None else args.tol_ov
    n_subcl_val = 0.1 if args.n_subcl_percentage is None else args.n_subcl_percentage
    ext_case_val = 0 if args.ext_case is None else args.ext_case
    return {
        "N": n_val,
        "Df": df_val,
        "kf": kf_val,
        "rp_g": rp_g_val,
        "rp_gstd": rp_gstd_val,
        "tol_ov": tol_ov_val,
        "n_subcl_percentage": n_subcl_val,
        "ext_case": ext_case_val,
    }


def _apply_algorithm_settings(algorithm_cfg: dict[str, Any]) -> None:
    mapping = {
        "use_cca_incremental_overlap": "USE_CCA_INCREMENTAL_OVERLAP",
        "cca_incremental_full_sync_period": "CCA_INCREMENTAL_FULL_SYNC_PERIOD",
        "cca_candidate_policy": "CCA_CANDIDATE_POLICY",
        "cca_score_topk_per_class": "CCA_SCORE_TOPK_PER_CLASS",
    }
    for key, attr in mapping.items():
        if key in algorithm_cfg and hasattr(runtime_config, attr):
            setattr(runtime_config, attr, algorithm_cfg[key])


@dataclass
class RunCase:
    run_name: str
    scheduler_address: str | None
    local_workers: int | None
    n_aggregates: int
    warmup_tasks: int
    seed_start: int
    trial_timeout: float | None
    sim_config: dict[str, Any]
    algorithm_cfg: dict[str, Any]
    output_json: Path
    profile_dir: Path | None


def _run_one_case(case: RunCase) -> Path:
    _apply_algorithm_settings(case.algorithm_cfg)
    config = case.sim_config

    commit = _git_commit_short()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    local_result: dict[str, Any]
    remote_result: dict[str, Any] | None = None

    with get_client(
        scheduler_address=None,
        n_workers=case.local_workers,
        install_package=False,
    ) as local_client:
        local_result = _run_environment_benchmark(
            "local",
            local_client,
            config,
            case.n_aggregates,
            case.warmup_tasks,
            case.seed_start,
            case.trial_timeout,
            profile_output_dir=case.profile_dir,
        )

    if case.scheduler_address is not None:
        with get_client(
            scheduler_address=case.scheduler_address,
            install_package=True,
        ) as remote_client:
            remote_result = _run_environment_benchmark(
                "remote",
                remote_client,
                config,
                case.n_aggregates,
                case.warmup_tasks,
                case.seed_start,
                case.trial_timeout,
                profile_output_dir=case.profile_dir,
            )

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
        "run_name": case.run_name,
        "commit": commit,
        "started_at": started_at,
        "config": config,
        "algorithm": case.algorithm_cfg,
        "n_aggregates": case.n_aggregates,
        "warmup_tasks": case.warmup_tasks,
        "local": local_result,
        "remote": remote_result,
        "comparison": comparison,
    }
    case.output_json.parent.mkdir(parents=True, exist_ok=True)
    case.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return case.output_json


def _load_orchestrator_config(path: Path) -> dict[str, Any]:
    return OrchestratorConfig.from_toml(path).model_dump(exclude_none=True)


def _discover_orchestrator_config() -> Path | None:
    candidates = [
        Path("benchmark_orchestrator.toml"),
        Path("configs/benchmark_orchestrator.toml"),
        Path("configs/unified_local_remote_orchestrator.toml"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _merge_simulation(defaults: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(run)
    return {
        "N": int(merged.get("N", merged.get("n", 256))),
        "Df": float(merged.get("Df", merged.get("df", 1.8))),
        "kf": float(merged.get("kf", 1.0)),
        "rp_g": float(merged.get("rp_g", 100.0)),
        "rp_gstd": float(merged.get("rp_gstd", 1.5)),
        "tol_ov": float(merged.get("tol_ov", 1e-6)),
        "n_subcl_percentage": float(merged.get("n_subcl_percentage", 0.1)),
        "ext_case": int(merged.get("ext_case", 0)),
    }


def _resolve_trial_timeout(value: Any) -> float | None:
    if value is None:
        return None
    timeout = float(value)
    if timeout <= 0:
        return None
    return timeout


def _as_int_list(value: Any, fallback: list[int]) -> list[int]:
    if value is None:
        return list(fallback)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _is_positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except Exception:
        return False


def _is_non_negative_int(value: Any) -> bool:
    try:
        return int(value) >= 0
    except Exception:
        return False


def _validate_scheduler(value: Any, label: str) -> str:
    scheduler = str(value).strip()
    if not scheduler:
        raise ValueError(f"{label} must be a non-empty scheduler value")
    if scheduler == "local" or scheduler.startswith("tcp://"):
        return scheduler
    raise ValueError(
        f"{label} must be 'local' or start with 'tcp://' (got: {scheduler!r})"
    )


def _validate_n_values(value: Any, label: str) -> list[int]:
    n_values = _as_int_list(value, [256])
    if not n_values:
        raise ValueError(f"{label} must be non-empty")
    if any(n <= 0 for n in n_values):
        raise ValueError(f"{label} must only contain positive integers")
    return n_values


def _build_cases_from_orchestrator(
    cfg: dict[str, Any], args
) -> tuple[list[RunCase], str]:
    defaults = cfg.get("defaults", {})
    runs = cfg.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config must define at least one [[runs]] entry")

    execution_mode = str(
        args.execution_mode_override
        if args.execution_mode_override is not None
        else defaults.get("execution_mode", "sequential")
    ).lower()
    if execution_mode not in {"sequential", "parallel"}:
        raise ValueError("defaults.execution_mode must be 'sequential' or 'parallel'")

    output_root = Path(str(defaults.get("output_root", "benchmark_results/profiles")))
    default_sim = defaults.get("simulation", {})
    default_algo = defaults.get("algorithm", {})

    cli_sim_overrides = {
        k: v
        for k, v in {
            "N": args.n,
            "Df": args.df,
            "kf": args.kf,
            "rp_g": args.rp_g,
            "rp_gstd": args.rp_gstd,
            "tol_ov": args.tol_ov,
            "n_subcl_percentage": args.n_subcl_percentage,
            "ext_case": args.ext_case,
        }.items()
        if v is not None
    }

    cases: list[RunCase] = []
    for idx, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{idx}] must be a TOML table")

        run_name = str(run.get("name", "")).strip()
        if not run_name:
            raise ValueError(f"runs[{idx}].name is required and must be non-empty")

        if args.scheduler_address is not None and len(runs) > 1:
            raise ValueError(
                "--scheduler-address with --config is only allowed when exactly one [[runs]] entry is defined; "
                "for multi-run matrices set scheduler per run in the TOML config"
            )
        scheduler_raw = (
            args.scheduler_address
            if args.scheduler_address is not None
            else run.get("scheduler", defaults.get("scheduler"))
        )
        if scheduler_raw is None:
            raise ValueError(
                f"runs[{idx}].scheduler is required (set to 'local' or tcp://...)"
            )
        scheduler = _validate_scheduler(scheduler_raw, f"runs[{idx}].scheduler")
        scheduler_address = None if scheduler == "local" else scheduler

        local_workers = (
            args.local_workers
            if args.local_workers is not None
            else run.get(
                "workers", run.get("local_workers", defaults.get("local_workers"))
            )
        )
        if local_workers is not None and not _is_positive_int(local_workers):
            raise ValueError(f"runs[{idx}].workers/local_workers must be >= 1")

        n_values_source = (
            args.n
            if args.n is not None
            else run.get("n_values", defaults.get("n_values", [256]))
        )
        n_values = _validate_n_values(n_values_source, f"runs[{idx}].n_values")

        repeats = int(
            run.get("repeats", defaults.get("repeats", 1))
            if args.repeats is None
            else args.repeats
        )
        if repeats < 1:
            raise ValueError(f"runs[{idx}].repeats must be >= 1")

        n_aggregates = int(
            run.get("n_aggregates", defaults.get("n_aggregates", 12))
            if args.n_aggregates is None
            else args.n_aggregates
        )
        if n_aggregates < 1:
            raise ValueError(f"runs[{idx}].n_aggregates must be >= 1")

        warmup_tasks = int(
            run.get("warmup_tasks", defaults.get("warmup_tasks", 2))
            if args.warmup_tasks is None
            else args.warmup_tasks
        )
        if warmup_tasks < 0:
            raise ValueError(f"runs[{idx}].warmup_tasks must be >= 0")

        seed_start = int(
            run.get("seed_start", defaults.get("seed_start", 1431354440))
            if args.seed_start is None
            else args.seed_start
        )

        trial_timeout = _resolve_trial_timeout(
            run.get("trial_timeout", defaults.get("trial_timeout"))
            if args.trial_timeout is None
            else args.trial_timeout
        )

        run_sim = run.get("simulation", {})
        if not isinstance(run_sim, dict):
            raise ValueError(f"runs[{idx}].simulation must be a table")
        run_sim = {**run_sim, **cli_sim_overrides}

        algo_cfg = dict(default_algo)
        run_algo = run.get("algorithm", {})
        if not isinstance(run_algo, dict):
            raise ValueError(f"runs[{idx}].algorithm must be a table")
        algo_cfg.update(run_algo)

        run_profile = bool(run.get("profile", defaults.get("profile", True)))

        for n in n_values:
            for rep in range(1, repeats + 1):
                sim_cfg = _merge_simulation(default_sim, {**run_sim, "N": int(n)})
                if (
                    args.output_json
                    and len(runs) == 1
                    and len(n_values) == 1
                    and repeats == 1
                ):
                    out_path = Path(args.output_json)
                else:
                    out_path = (
                        output_root / run_name / f"unified_N{int(n)}_rep{rep}.json"
                    )

                if args.profile_dir:
                    profile_dir = (
                        Path(args.profile_dir) / run_name / f"N{int(n)}_rep{rep}"
                    )
                elif run_profile:
                    profile_dir = (
                        output_root / run_name / f"N{int(n)}_rep{rep}" / "profiles"
                    )
                else:
                    profile_dir = None

                case_seed_start = seed_start + rep * 1_000_000 + int(n) * 1000
                cases.append(
                    RunCase(
                        run_name=f"{run_name}_N{int(n)}_rep{rep}",
                        scheduler_address=scheduler_address,
                        local_workers=(
                            None if local_workers is None else int(local_workers)
                        ),
                        n_aggregates=n_aggregates,
                        warmup_tasks=warmup_tasks,
                        seed_start=case_seed_start,
                        trial_timeout=trial_timeout,
                        sim_config=sim_cfg,
                        algorithm_cfg=algo_cfg,
                        output_json=out_path,
                        profile_dir=profile_dir,
                    )
                )

    return cases, execution_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified local/remote Dask benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional orchestrator TOML config. If omitted, auto-discovers benchmark_orchestrator.toml or configs/benchmark_orchestrator.toml.",
    )
    parser.add_argument(
        "--execution-mode-override",
        type=str,
        default=None,
        choices=["sequential", "parallel"],
        help="Override orchestrator execution mode when --config is used",
    )
    parser.add_argument(
        "--max-parallel-cases",
        type=int,
        default=None,
        help="Max concurrent benchmark cases when execution mode is parallel",
    )
    parser.add_argument(
        "--scheduler-address",
        type=str,
        default=None,
        help="Legacy single-run scheduler override. In --config mode this is only valid when exactly one [[runs]] entry exists.",
    )
    parser.add_argument(
        "--remote-loopback-workers",
        type=int,
        default=0,
        help="Create a separate local cluster and treat it as remote for smoke tests",
    )
    parser.add_argument("--local-workers", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--n-aggregates", type=int, default=None)
    parser.add_argument("--warmup-tasks", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--trial-timeout", type=float, default=None)

    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--df", type=float, default=None)
    parser.add_argument("--kf", type=float, default=None)
    parser.add_argument("--rp-g", dest="rp_g", type=float, default=None)
    parser.add_argument("--rp-gstd", dest="rp_gstd", type=float, default=None)
    parser.add_argument("--tol-ov", dest="tol_ov", type=float, default=None)
    parser.add_argument(
        "--n-subcl-percentage", dest="n_subcl_percentage", type=float, default=None
    )
    parser.add_argument("--ext-case", dest="ext_case", type=int, default=None)

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

    orchestrator_config_path: Path | None = None
    if args.config:
        orchestrator_config_path = Path(args.config)
    else:
        orchestrator_config_path = _discover_orchestrator_config()

    if orchestrator_config_path is not None:
        cfg = _load_orchestrator_config(orchestrator_config_path)
        cases, execution_mode = _build_cases_from_orchestrator(cfg, args)
        orchestrator_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 80)
        print("UNIFIED LOCAL/REMOTE DASK BENCHMARK (ORCHESTRATOR)")
        print("=" * 80)
        print(f"Config file: {orchestrator_config_path}")
        print(f"Execution mode: {execution_mode}")
        print(f"Cases: {len(cases)}")
        print("=" * 80)

        output_paths: list[str] = []
        if execution_mode == "parallel":
            print(
                "WARNING: parallel execution mode can reduce reproducibility due to shared-resource contention."
            )
            max_workers = args.max_parallel_cases
            if max_workers is None:
                max_workers = min(len(cases), max(1, os.cpu_count() or 1))
            if max_workers < 1:
                raise ValueError("--max-parallel-cases must be >= 1")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_case = {
                    pool.submit(_run_one_case, case): case for case in cases
                }
                for future in concurrent.futures.as_completed(future_to_case):
                    case = future_to_case[future]
                    out_path = future.result()
                    output_paths.append(str(out_path))
                    print(f"Completed: {case.run_name} -> {out_path}")
        else:
            for case in cases:
                out_path = _run_one_case(case)
                output_paths.append(str(out_path))
                print(f"Completed: {case.run_name} -> {out_path}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        summary_path = (
            Path(args.output_json)
            if args.output_json and len(cases) > 1
            else Path("benchmark_results") / f"unified_orchestrator_summary_{ts}.json"
        )
        _write_json(
            summary_path,
            {
                "benchmark": "unified_local_remote_orchestrator",
                "config_path": str(orchestrator_config_path),
                "execution_mode": execution_mode,
                "cases": len(cases),
                "outputs": output_paths,
                "started_at": orchestrator_started_at,
                "commit": _git_commit_short(),
            },
        )
        print("\n" + "=" * 80)
        print("ORCHESTRATOR RESULTS")
        print("=" * 80)
        print(f"Completed {len(output_paths)} case(s)")
        print(f"Summary: {summary_path}")
        print("=" * 80)
        return

    config = _build_config(args)
    n_aggregates = args.n_aggregates if args.n_aggregates is not None else 12
    warmup_tasks = args.warmup_tasks if args.warmup_tasks is not None else 2
    seed_start = args.seed_start if args.seed_start is not None else 1431354440

    print("=" * 80)
    print("UNIFIED LOCAL/REMOTE DASK BENCHMARK")
    print("=" * 80)
    print(f"Config: {config}")
    print(f"Aggregates per environment: {n_aggregates}")
    print(f"Warmup tasks: {warmup_tasks}")
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
            n_aggregates,
            warmup_tasks,
            seed_start,
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
                n_aggregates,
                warmup_tasks,
                seed_start,
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
        "n_aggregates": n_aggregates,
        "warmup_tasks": warmup_tasks,
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
