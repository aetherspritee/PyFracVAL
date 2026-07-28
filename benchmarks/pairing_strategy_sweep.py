#!/usr/bin/env python3
"""Re-run a stability sweep config with the matching-based CCA pairing
strategies active, for direct comparison against the existing greedy
baseline data already committed for the same config.

Reuses stability_sweep.py's grid resolution and sweep execution as-is via
SweepConfig.merged({"algorithm": {"cca_pairing_strategy": ...}}) - no
hand-written per-strategy TOML files.

Usage:
    devenv shell -- uv run python benchmarks/pairing_strategy_sweep.py \
        --config configs/hard_regime_boundary_sweep.toml \
        --strategies matching matching_leaf_weighted
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stability_sweep import (  # noqa: E402
    StickingBenchmark,
    _resolve_grid,
    _run_sweep_dask,
    _run_sweep_sequential,
    _write_csv,
    _write_json,
)

from pyfracval.config import SweepConfig


def run_one_strategy(base_cfg: SweepConfig, strategy: str) -> None:
    cfg = base_cfg.merged({"algorithm": {"cca_pairing_strategy": strategy}})
    # Keep each strategy's output fully separate from the greedy baseline
    # and from each other, rather than overwriting the same directory.
    cfg = cfg.merged({"output_dir": f"{base_cfg.output_dir}_{strategy}"})

    sizes, sigmas, df_values, kf_values = _resolve_grid(cfg)
    output_root = Path(cfg.output_dir)
    summary_dir = output_root / "stability_sweeps"
    summary_dir.mkdir(parents=True, exist_ok=True)

    benchmark = StickingBenchmark(output_dir=str(output_root))
    total_combos = len(sizes) * len(sigmas) * len(df_values) * len(kf_values)
    print(
        f"\n=== strategy={strategy}: {total_combos} combos x {cfg.trials} "
        f"trials -> {output_root} ==="
    )

    sweep_rows: list = []
    raw_handle = None  # save_raw not needed for this comparison

    sweep_start = time.time()
    if cfg.dask.enable:
        _run_sweep_dask(
            cfg, sizes, sigmas, df_values, kf_values, benchmark, sweep_rows, raw_handle
        )
    else:
        _run_sweep_sequential(
            cfg, sizes, sigmas, df_values, kf_values, benchmark, sweep_rows, raw_handle
        )
    sweep_runtime = time.time() - sweep_start

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "config": cfg.model_dump(),
        "total_combinations": total_combos,
        "trials_per_combo": cfg.trials,
        "sizes": sizes,
        "sigmas": sigmas,
        "df_values": df_values,
        "kf_values": kf_values,
        "total_runtime_s": sweep_runtime,
        "results": sweep_rows,
    }
    _write_json(summary_dir / "stability_sweep_summary.json", summary)

    csv_rows = [
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
        for row in sweep_rows
    ]
    _write_csv(
        summary_dir / "stability_sweep_summary.csv",
        csv_rows,
        fieldnames=list(csv_rows[0].keys()) if csv_rows else [],
    )

    total_trials = sum(r["trials"] for r in sweep_rows)
    total_successes = sum(r["successes"] for r in sweep_rows)
    print(
        f"strategy={strategy}: {total_successes}/{total_trials} "
        f"({100 * total_successes / total_trials:.1f}%) in {sweep_runtime:.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["matching", "matching_leaf_weighted"],
    )
    args = parser.parse_args()

    base_cfg = SweepConfig.from_file(args.config)
    for strategy in args.strategies:
        run_one_strategy(base_cfg, strategy)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
