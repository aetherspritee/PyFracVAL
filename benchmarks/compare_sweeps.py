"""Compare stability sweep summaries across multiple runs.

Usage::

    uv run python benchmarks/compare_sweeps.py \\
        benchmark_results/consistency/local_1/stability_sweeps/stability_sweep_summary.json \\
        benchmark_results/consistency/local_2/stability_sweeps/stability_sweep_summary.json \\
        benchmark_results/consistency/marvin_1/stability_sweeps/stability_sweep_summary.json \\
        benchmark_results/consistency/marvin_2/stability_sweeps/stability_sweep_summary.json

Prints a per-combo comparison table and a headline summary showing:
  - Which combos differ between any pair
  - Max / mean absolute difference in success_rate across all pairs
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _key(row: dict) -> Tuple:
    return (row["N"], round(row["Df"], 6), round(row["kf"], 6), row["rp_gstd"])


def _index(summary: dict) -> Dict[Tuple, dict]:
    return {_key(r): r for r in summary["results"]}


def _label(path: Path) -> str:
    # Use the grandparent directory name as a short label (e.g. "local_1")
    parts = path.parts
    try:
        # .../consistency/local_1/stability_sweeps/stability_sweep_summary.json
        return parts[-3]
    except IndexError:
        return path.stem


def _print_pair_diff(label_a: str, label_b: str, idx_a: dict, idx_b: dict) -> int:
    """Print differences between two indexed sweeps. Returns number of diffs."""
    common_keys = sorted(set(idx_a) & set(idx_b))
    diffs = [
        (k, idx_a[k]["success_rate"], idx_b[k]["success_rate"])
        for k in common_keys
        if idx_a[k]["success_rate"] != idx_b[k]["success_rate"]
    ]
    only_a = set(idx_a) - set(idx_b)
    only_b = set(idx_b) - set(idx_a)

    print(f"\n{'─' * 60}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'─' * 60}")

    if only_a:
        print(f"  [only in {label_a}] {len(only_a)} combos")
    if only_b:
        print(f"  [only in {label_b}] {len(only_b)} combos")

    if not diffs:
        print("  ✓ All success_rates IDENTICAL")
    else:
        abs_diffs = [abs(a - b) for _, a, b in diffs]
        print(
            f"  {len(diffs)}/{len(common_keys)} combos differ  "
            f"(max Δ={max(abs_diffs):.2f}, mean Δ={sum(abs_diffs) / len(abs_diffs):.3f})"
        )
        print(
            f"  {'N':>4}  {'Df':>6}  {'kf':>6}  {'sigma':>5}  "
            f"  {label_a:>10}  {label_b:>10}  {'Δ':>6}"
        )
        for (n, df, kf, sigma), rate_a, rate_b in sorted(diffs):
            delta = rate_b - rate_a
            marker = " ↑" if delta > 0 else " ↓"
            print(
                f"  {n:>4}  {df:>6.3f}  {kf:>6.3f}  {sigma:>5.1f}  "
                f"  {rate_a:>10.2f}  {rate_b:>10.2f}  {delta:>+6.2f}{marker}"
            )

    return len(diffs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare stability sweep summary JSON files."
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        metavar="PATH",
        help="Paths to stability_sweep_summary.json files to compare.",
    )
    parser.add_argument(
        "--group-local-remote",
        action="store_true",
        help=(
            "Print aggregate stats grouping 'local_*' vs 'marvin_*' pairs "
            "in addition to all pairwise comparisons."
        ),
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.summaries]
    if len(paths) < 2:
        print("Need at least 2 summary files to compare.")
        return 1

    loaded = []
    for p in paths:
        summary = _load(p)
        label = _label(p)
        idx = _index(summary)
        loaded.append((label, idx, summary))
        print(
            f"Loaded {label!r}: {len(summary['results'])} combos, "
            f"{summary['trials_per_combo']} trials each, "
            f"generated {summary['generated_at']}"
        )

    total_diffs = 0
    pair_count = 0
    for (label_a, idx_a, _), (label_b, idx_b, _) in combinations(loaded, 2):
        n_diff = _print_pair_diff(label_a, label_b, idx_a, idx_b)
        total_diffs += n_diff
        pair_count += 1

    print(f"\n{'═' * 60}")
    print(f"  {pair_count} pairs compared — {total_diffs} total differing combos")

    # Headline verdict
    same_platform_pairs = [
        (a, b)
        for (a, ia, _), (b, ib, _) in combinations(loaded, 2)
        if (a.startswith("local") and b.startswith("local"))
        or (a.startswith("marvin") and b.startswith("marvin"))
    ]
    cross_pairs = [
        (a, b)
        for (a, ia, _), (b, ib, _) in combinations(loaded, 2)
        if not (
            (a.startswith("local") and b.startswith("local"))
            or (a.startswith("marvin") and b.startswith("marvin"))
        )
    ]

    if same_platform_pairs:
        print("\n  Same-platform pairs (expect IDENTICAL results):")
        for a, b in same_platform_pairs:
            print(f"    {a} ↔ {b}")

    if cross_pairs:
        print("\n  Cross-platform pairs (divergence expected):")
        for a, b in cross_pairs:
            print(f"    {a} ↔ {b}")

    print(f"{'═' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
