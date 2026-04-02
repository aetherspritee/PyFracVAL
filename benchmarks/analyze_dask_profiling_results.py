#!/usr/bin/env python3
"""Analyze unified local/remote Dask benchmark outputs.

This script aggregates one or more JSON files produced by
`benchmarks/unified_local_remote_benchmark.py` and reports:

- robust summary stats by N and environment
- remote/local ratios (raw + normalized)
- likely bottleneck categories
- optimization opportunities with rationale
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _expand_inputs(patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        p = Path(pattern)
        if p.exists() and p.is_file():
            files.add(p)
            continue
        for match in Path(".").glob(pattern):
            if match.is_file():
                files.add(match)
    return sorted(files)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    frac = idx - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _iqr(values: list[float]) -> float:
    if not values:
        return 0.0
    return _quantile(values, 0.75) - _quantile(values, 0.25)


def _p95(values: list[float]) -> float:
    return _quantile(values, 0.95)


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean": _mean(values),
        "median": _median(values),
        "stdev": _stdev(values),
        "iqr": _iqr(values),
        "p95": _p95(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _extract_record(
    payload: dict[str, Any], source_file: Path
) -> dict[str, Any] | None:
    if payload.get("benchmark") != "unified_local_remote_dask":
        return None

    config = payload.get("config", {})
    n = int(config.get("N", 0))
    if n <= 0:
        return None

    local = payload.get("local")
    remote = payload.get("remote")
    comparison = payload.get("comparison") or {}

    if not isinstance(local, dict) or not isinstance(remote, dict):
        return None

    remote_worker_calib = remote.get("worker_calibration_wall_s", {}) or {}
    calib_values = [
        _safe_float(v) for v in remote_worker_calib.values() if _safe_float(v) > 0
    ]
    calib_spread = (
        (max(calib_values) / min(calib_values)) if len(calib_values) >= 2 else 1.0
    )

    return {
        "source": str(source_file),
        "N": n,
        "local": {
            "throughput": _safe_float(local.get("throughput_agg_per_s")),
            "per_thread": _safe_float(local.get("throughput_per_thread")),
            "per_effective_thread": _safe_float(
                local.get("throughput_per_effective_thread")
            ),
            "busy": _safe_float(local.get("thread_busy_fraction")),
            "task_wall_median": _safe_float(local.get("task_wall_median_s")),
            "cpu_to_wall": _safe_float(local.get("cpu_to_wall_ratio")),
            "success_rate": _safe_float(local.get("success_rate")),
        },
        "remote": {
            "throughput": _safe_float(remote.get("throughput_agg_per_s")),
            "per_thread": _safe_float(remote.get("throughput_per_thread")),
            "per_effective_thread": _safe_float(
                remote.get("throughput_per_effective_thread")
            ),
            "busy": _safe_float(remote.get("thread_busy_fraction")),
            "task_wall_median": _safe_float(remote.get("task_wall_median_s")),
            "cpu_to_wall": _safe_float(remote.get("cpu_to_wall_ratio")),
            "success_rate": _safe_float(remote.get("success_rate")),
            "calibration_spread": calib_spread,
        },
        "ratios": {
            "raw": _safe_float(comparison.get("remote_vs_local_raw_throughput_ratio")),
            "per_thread": _safe_float(
                comparison.get("remote_vs_local_per_thread_ratio")
            ),
            "per_effective_thread": _safe_float(
                comparison.get("remote_vs_local_per_effective_thread_ratio")
            ),
        },
    }


def _diagnose(n_bucket: dict[str, Any]) -> dict[str, Any]:
    ratios_raw = n_bucket["ratios_raw"]
    ratios_pt = n_bucket["ratios_per_thread"]
    ratios_pet = n_bucket["ratios_per_effective_thread"]
    remote_busy = n_bucket["remote_busy"]
    remote_calib_spread = n_bucket["remote_calibration_spread"]
    remote_success = n_bucket["remote_success"]

    med_raw = _median(ratios_raw)
    med_pt = _median(ratios_pt)
    med_pet = _median(ratios_pet)
    med_busy = _median(remote_busy)
    med_calib_spread = _median(remote_calib_spread)
    med_success = _median(remote_success)

    findings: list[str] = []
    opportunities: list[dict[str, str]] = []

    if med_success < 1.0:
        findings.append(
            "Remote success rate below 100%; stability limits throughput confidence."
        )
        opportunities.append(
            {
                "action": "Address remote instability before speed tuning",
                "reason": "Performance conclusions are weak if failures/retries differ by environment.",
            }
        )

    if med_busy < 0.45:
        findings.append(
            "Remote worker busy fraction is low; cluster likely underutilized."
        )
        opportunities.append(
            {
                "action": "Increase task granularity (larger chunks / fewer tiny tasks)",
                "reason": "Low busy fraction suggests scheduler/dispatch overhead dominates useful compute.",
            }
        )

    if med_raw < 0.9 and med_pt < 0.9:
        findings.append("Remote underperforms local after thread normalization.")
        opportunities.append(
            {
                "action": "Profile scheduler overhead and task wait time",
                "reason": "Likely orchestration overhead at current task size.",
            }
        )

    if med_calib_spread > 3.0:
        findings.append(
            "Remote worker calibration spread is high; cluster heterogeneity is significant."
        )
        opportunities.append(
            {
                "action": "Split heterogeneous workers into separate benchmark pools",
                "reason": "Mixed node speeds can cause stragglers and misleading normalization.",
            }
        )

    if med_pet < 0.6 * max(med_pt, 1e-9):
        findings.append("Per-effective-thread ratio is far below per-thread ratio.")
        opportunities.append(
            {
                "action": "Harden calibration model (more repeats + clipping)",
                "reason": "Current effective-thread weighting may be unstable due to outlier workers.",
            }
        )

    if not findings:
        findings.append(
            "No dominant bottleneck detected; behavior appears balanced for current runs."
        )
        opportunities.append(
            {
                "action": "Increase workload size and re-profile",
                "reason": "Small-N runs can hide or distort true scaling trends.",
            }
        )

    primary = findings[0]
    return {
        "primary_finding": primary,
        "findings": findings,
        "opportunities": opportunities[:3],
    }


def analyze(files: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for file in files:
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec = _extract_record(payload, file)
        if rec is not None:
            records.append(rec)

    by_n: dict[int, dict[str, Any]] = {}
    for rec in records:
        n = int(rec["N"])
        bucket = by_n.setdefault(
            n,
            {
                "records": 0,
                "sources": [],
                "local_throughput": [],
                "remote_throughput": [],
                "local_per_thread": [],
                "remote_per_thread": [],
                "local_per_effective_thread": [],
                "remote_per_effective_thread": [],
                "local_busy": [],
                "remote_busy": [],
                "remote_success": [],
                "remote_calibration_spread": [],
                "ratios_raw": [],
                "ratios_per_thread": [],
                "ratios_per_effective_thread": [],
            },
        )
        bucket["records"] += 1
        bucket["sources"].append(rec["source"])
        bucket["local_throughput"].append(rec["local"]["throughput"])
        bucket["remote_throughput"].append(rec["remote"]["throughput"])
        bucket["local_per_thread"].append(rec["local"]["per_thread"])
        bucket["remote_per_thread"].append(rec["remote"]["per_thread"])
        bucket["local_per_effective_thread"].append(
            rec["local"]["per_effective_thread"]
        )
        bucket["remote_per_effective_thread"].append(
            rec["remote"]["per_effective_thread"]
        )
        bucket["local_busy"].append(rec["local"]["busy"])
        bucket["remote_busy"].append(rec["remote"]["busy"])
        bucket["remote_success"].append(rec["remote"]["success_rate"])
        bucket["remote_calibration_spread"].append(rec["remote"]["calibration_spread"])
        bucket["ratios_raw"].append(rec["ratios"]["raw"])
        bucket["ratios_per_thread"].append(rec["ratios"]["per_thread"])
        bucket["ratios_per_effective_thread"].append(
            rec["ratios"]["per_effective_thread"]
        )

    per_n: dict[str, Any] = {}
    global_findings: list[str] = []
    global_opportunities: list[dict[str, str]] = []

    for n in sorted(by_n):
        bucket = by_n[n]
        diag = _diagnose(bucket)
        per_n[str(n)] = {
            "records": bucket["records"],
            "summary": {
                "local_throughput": _summarize(bucket["local_throughput"]),
                "remote_throughput": _summarize(bucket["remote_throughput"]),
                "local_per_thread": _summarize(bucket["local_per_thread"]),
                "remote_per_thread": _summarize(bucket["remote_per_thread"]),
                "local_per_effective_thread": _summarize(
                    bucket["local_per_effective_thread"]
                ),
                "remote_per_effective_thread": _summarize(
                    bucket["remote_per_effective_thread"]
                ),
                "remote_busy_fraction": _summarize(bucket["remote_busy"]),
                "ratios": {
                    "raw": _summarize(bucket["ratios_raw"]),
                    "per_thread": _summarize(bucket["ratios_per_thread"]),
                    "per_effective_thread": _summarize(
                        bucket["ratios_per_effective_thread"]
                    ),
                },
            },
            "diagnosis": diag,
            "sources": bucket["sources"],
        }
        global_findings.append(f"N={n}: {diag['primary_finding']}")
        for opp in diag["opportunities"]:
            global_opportunities.append(opp)

    # Keep only unique opportunities by action text.
    seen_actions: set[str] = set()
    dedup_opportunities: list[dict[str, str]] = []
    for opp in global_opportunities:
        action = opp.get("action", "")
        if action and action not in seen_actions:
            seen_actions.add(action)
            dedup_opportunities.append(opp)

    return {
        "input_files": [str(f) for f in files],
        "total_records": len(records),
        "per_N": per_n,
        "global_findings": global_findings,
        "recommended_next_actions": dedup_opportunities[:5],
    }


def _print_report(analysis: dict[str, Any]) -> None:
    print("=" * 80)
    print("DASK PROFILING ANALYSIS")
    print("=" * 80)
    print(f"Input files: {len(analysis.get('input_files', []))}")
    print(f"Usable records: {analysis.get('total_records', 0)}")

    per_n = analysis.get("per_N", {})
    for n in sorted(per_n, key=lambda x: int(x)):
        item = per_n[n]
        ratios = item["summary"]["ratios"]
        raw_med = ratios["raw"]["median"]
        pt_med = ratios["per_thread"]["median"]
        pet_med = ratios["per_effective_thread"]["median"]
        busy_med = item["summary"]["remote_busy_fraction"]["median"]
        print("-" * 80)
        print(
            f"N={n} | runs={item['records']} | "
            f"ratio raw={raw_med:.3f}, per-thread={pt_med:.3f}, per-effective={pet_med:.3f}, "
            f"remote busy={busy_med:.3f}"
        )
        print(f"  primary finding: {item['diagnosis']['primary_finding']}")

    print("-" * 80)
    print("Top next actions:")
    for i, action in enumerate(analysis.get("recommended_next_actions", []), start=1):
        print(f"  {i}. {action.get('action')} -- {action.get('reason')}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Dask profiling benchmark outputs"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["benchmark_results/unified_local_vs_marvin_*.json"],
        help="Input files or glob patterns",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save analysis JSON",
    )
    args = parser.parse_args()

    files = _expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No input files found for analysis")

    analysis = analyze(files)
    _print_report(analysis)

    if args.output_json:
        out = Path(args.output_json)
    else:
        out = Path("benchmark_results/dask_profiling_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Saved analysis JSON: {out}")


if __name__ == "__main__":
    main()
