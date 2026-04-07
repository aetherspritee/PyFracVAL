#!/usr/bin/env python3
"""Unified entrypoint for benchmark workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_script(script_name: str, script_args: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "benchmarks" / script_name
    cmd = [sys.executable, str(script_path), *script_args]
    return subprocess.call(cmd, cwd=str(repo_root))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified benchmark entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("unified", help="Run unified local/remote benchmark")
    sub.add_parser("stability", help="Run stability sweep benchmark")

    p_sticking = sub.add_parser("sticking", help="Run sticking benchmark suite")
    p_sticking.add_argument(
        "--suite", type=str, default="stable", help='Suite name or "all"'
    )
    p_sticking.add_argument("--trials", type=int, default=3, help="Trials per case")
    p_sticking.add_argument("--output-dir", type=str, default="benchmark_results")

    p_analyze = sub.add_parser("analyze", help="Run analysis helpers")
    p_analyze_sub = p_analyze.add_subparsers(dest="analyze_command", required=True)
    p_analyze_sub.add_parser("dask-profiles", help="Analyze Dask profile outputs")
    p_analyze_sub.add_parser("stability", help="Analyze stability outputs")

    return parser


def _dispatch_sticking(suite: str, trials: int, output_dir: str) -> int:
    from benchmarks.sticking_benchmark import StickingBenchmark

    benchmark = StickingBenchmark(output_dir=output_dir)
    if suite == "all":
        benchmark.run_all(n_trials=trials)
    else:
        benchmark.run_suite(suite, n_trials=trials)
    return 0


def main() -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()

    if args.command == "unified":
        return _run_script("unified_local_remote_benchmark.py", unknown)

    if args.command == "stability":
        return _run_script("stability_sweep.py", unknown)

    if args.command == "sticking":
        if unknown:
            parser.error(f"unrecognized sticking args: {' '.join(unknown)}")
        return _dispatch_sticking(args.suite, args.trials, args.output_dir)

    if args.command == "analyze":
        if args.analyze_command == "dask-profiles":
            return _run_script("analyze_dask_profiling_results.py", unknown)
        if args.analyze_command == "stability":
            return _run_script("analyze_stability.py", unknown)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
