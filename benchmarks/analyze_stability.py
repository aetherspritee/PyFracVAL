"""
analyze_stability.py — Stability analysis for PyFracVAL sweep results.

Usage:
    uv run python benchmarks/analyze_stability.py [summary.json ...]

If no files are given, tries to auto-discover all stability_sweep_summary.json
files under benchmark_results/.

Key ideas
---------
The fractal-aggregate equation is:

    N = kf · (R_g / r_geo)^Df

Rearranging for the "effective radius ratio":

    X ≡ (N / kf)^(1/Df)  =  R_g / r_geo

X is the dimensionless ratio of target gyration radius to geometric mean
particle radius.  A large X means the algorithm must build a structure that
spans many particle diameters — geometrically harder.  For a fixed (Df, sigma),
stability should collapse to a single boundary in X.

This script:
  1. Loads one or more stability_sweep_summary.json files and merges them.
  2. Computes X for every (N, Df, kf) combo.
  3. For each sigma, prints a stability map over X (bucketed) and over the
     original (Df, kf) plane.
  4. Identifies the maximum safe X per sigma (≥ threshold success rate).
  5. Recommends (Df, kf) presets suitable for research runs up to N=1024.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STABLE_THRESHOLD = 0.90  # success_rate ≥ this → "stable"
UNCERTAIN_THRESHOLD = 0.50  # below this → "unstable"

SIGMA_LABELS = {1.0: "mono", 1.5: "low", 2.0: "mod", 2.5: "high", 3.0: "very-high"}


def _label(sr: float) -> str:
    if sr >= STABLE_THRESHOLD:
        return "S"  # stable
    if sr >= UNCERTAIN_THRESHOLD:
        return "?"  # borderline
    return "X"  # unstable


def _load_results(paths: list[Path]) -> list[dict]:
    """Load and merge result rows from one or more summary JSON files."""
    # Key: (N, Df, kf, sigma) → best (highest-trial) row
    merged: dict[tuple, dict] = {}
    for p in paths:
        with p.open() as f:
            data = json.load(f)
        rows = data.get("results", [])
        for row in rows:
            key = (row["N"], round(row["Df"], 4), round(row["kf"], 4), row["rp_gstd"])
            existing = merged.get(key)
            if existing is None or row["trials"] > existing["trials"]:
                merged[key] = row
    return list(merged.values())


def _compute_x(row: dict) -> float:
    """X = (N / kf)^(1/Df) — effective radius ratio."""
    return (row["N"] / row["kf"]) ** (1.0 / row["Df"])


# ---------------------------------------------------------------------------
# Stability map: (Df, kf) plane per (N, sigma)
# ---------------------------------------------------------------------------


def print_df_kf_map(rows: list[dict]) -> None:
    # Group by (N, sigma)
    groups: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    for r in rows:
        groups[(r["N"], r["rp_gstd"])][(round(r["Df"], 4), round(r["kf"], 4))] = r[
            "success_rate"
        ]

    for (n, sigma), combos in sorted(groups.items()):
        dfs = sorted({d for (d, _) in combos})
        kfs = sorted({k for (_, k) in combos})
        print(f"\n  N={n:4d}  sigma={sigma}")
        print("         kf → " + "  ".join(f"{k:.1f}" for k in kfs))
        for df in dfs:
            cells = []
            for kf in kfs:
                sr = combos.get((df, kf))
                if sr is None:
                    cells.append(" . ")
                else:
                    lbl = _label(sr)
                    # Add rate in parens for borderline
                    if lbl == "?":
                        cells.append(f"?{int(sr * 10)}")
                    elif lbl == "X":
                        cells.append(" X ")
                    else:
                        cells.append(" S ")
            print(f"  Df={df:.1f} │ " + "  ".join(cells))
    print()


# ---------------------------------------------------------------------------
# Stability map: X-bins per sigma (collapsed over N, Df, kf)
# ---------------------------------------------------------------------------


def print_x_map(rows: list[dict]) -> None:
    import math

    sigmas = sorted({r["rp_gstd"] for r in rows})
    # Build X → list[success_rate] per sigma
    x_data: dict[float, list[tuple[float, float]]] = defaultdict(
        list
    )  # sigma → [(X, sr)]
    for r in rows:
        x = _compute_x(r)
        x_data[r["rp_gstd"]].append((x, r["success_rate"]))

    print("\n  Stability vs. X = (N/kf)^(1/Df)  [bin width = 2]\n")
    for sigma in sigmas:
        pts = x_data[sigma]
        if not pts:
            continue
        xs = [x for x, _ in pts]
        # Bin from 1 to max+2 in steps of 2
        max_x = max(xs)
        bins = list(range(1, int(max_x) + 4, 2))
        print(f"  sigma={sigma}:")
        for lo in bins:
            hi = lo + 2
            in_bin = [(x, sr) for x, sr in pts if lo <= x < hi]
            if not in_bin:
                continue
            mean_sr = sum(sr for _, sr in in_bin) / len(in_bin)
            min_sr = min(sr for _, sr in in_bin)
            max_sr = max(sr for _, sr in in_bin)
            bar = "█" * int(mean_sr * 20) + "░" * (20 - int(mean_sr * 20))
            lbl = _label(mean_sr)
            print(
                f"    X=[{lo:4.0f},{hi:4.0f})  n={len(in_bin):3d}  "
                f"mean={mean_sr:.2f} min={min_sr:.2f} max={max_sr:.2f}  "
                f"{bar} {lbl}"
            )
        print()


# ---------------------------------------------------------------------------
# Per-sigma: maximum safe X
# ---------------------------------------------------------------------------


def max_safe_x(rows: list[dict]) -> dict[float, float]:
    """For each sigma, find the largest X where ≥ threshold success is still observed."""
    # Group by (sigma, X_rounded)
    groups: dict[tuple[float, float], list[float]] = defaultdict(list)
    for r in rows:
        x = _compute_x(r)
        x_r = round(x, 1)
        groups[(r["rp_gstd"], x_r)].append(r["success_rate"])

    # Per sigma: max X where mean_sr >= threshold
    result: dict[float, float] = {}
    sigmas = sorted({r["rp_gstd"] for r in rows})
    for sigma in sigmas:
        safe_xs = []
        for (s, x), srs in groups.items():
            if s != sigma:
                continue
            if sum(srs) / len(srs) >= STABLE_THRESHOLD:
                safe_xs.append(x)
        result[sigma] = max(safe_xs) if safe_xs else 0.0
    return result


# ---------------------------------------------------------------------------
# Preset recommendations
# ---------------------------------------------------------------------------


def recommend_presets(rows: list[dict], target_ns: list[int]) -> None:
    """
    For each sigma, suggest (Df, kf) combos that are stable across all target N.
    Stable = success_rate >= STABLE_THRESHOLD for every tested N in target_ns.
    """
    # Only keep rows whose N is in target_ns (or all N if target_ns is empty)
    if target_ns:
        relevant = [r for r in rows if r["N"] in target_ns]
    else:
        relevant = rows

    # Group by (sigma, Df, kf): collect {N: success_rate}
    combo_data: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in relevant:
        key = (r["rp_gstd"], round(r["Df"], 4), round(r["kf"], 4))
        combo_data[key][r["N"]] = r["success_rate"]

    sigmas = sorted({r["rp_gstd"] for r in relevant})
    tested_ns = sorted({r["N"] for r in relevant})

    print(
        f"\n  Preset recommendations (stable across N={tested_ns}, threshold={STABLE_THRESHOLD:.0%})"
    )
    print(
        f"  ({'S' if target_ns else 'all N'} tested, sorted by X = (N/kf)^(1/Df) at N=max)\n"
    )

    for sigma in sigmas:
        max_n = max(tested_ns) if tested_ns else 128
        candidates = []
        for (s, df, kf), n_map in combo_data.items():
            if s != sigma:
                continue
            # Must be stable for ALL tested N values present in the data
            ns_here = list(n_map.keys())
            if not ns_here:
                continue
            all_stable = all(n_map.get(n, 0.0) >= STABLE_THRESHOLD for n in ns_here)
            if all_stable:
                x_at_max = (max_n / kf) ** (1.0 / df)
                candidates.append((df, kf, x_at_max, min(n_map.values()), len(ns_here)))

        # Sort by Df (ascending), then kf
        candidates.sort(key=lambda t: (t[0], t[1]))
        slabel = SIGMA_LABELS.get(sigma, str(sigma))
        print(f"  sigma={sigma} ({slabel} polydispersity):")
        if not candidates:
            print("    (no fully-stable combos found in current data)")
        else:
            print(
                f"    {'Df':>5}  {'kf':>5}  {'X@N=' + str(max_n):>10}  {'min_sr':>7}  {'N_tested':>8}"
            )
            for df, kf, x, min_sr, n_tested in candidates:
                marker = " ★" if df in (1.8, 2.0, 2.2) and kf in (0.8, 1.0, 1.2) else ""
                print(
                    f"    {df:5.2f}  {kf:5.2f}  {x:>10.2f}  {min_sr:7.2f}  {n_tested:>8}{marker}"
                )
        print()


# ---------------------------------------------------------------------------
# N-scaling check: does success_rate degrade with N for a fixed (Df, kf, sigma)?
# ---------------------------------------------------------------------------


def print_n_scaling(rows: list[dict]) -> None:
    # Group by (Df, kf, sigma)
    groups: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in rows:
        key = (round(r["Df"], 4), round(r["kf"], 4), r["rp_gstd"])
        groups[key][r["N"]] = r["success_rate"]

    ns_all = sorted({r["N"] for r in rows})
    if len(ns_all) < 2:
        print("  (only one N value in data — skip N-scaling check)\n")
        return

    print(f"\n  N-scaling: success_rate by N for each (Df, kf, sigma)\n")
    print(
        f"  {'Df':>5}  {'kf':>5}  {'sigma':>6}  "
        + "  ".join(f"N={n:4d}" for n in ns_all)
    )
    print("  " + "-" * (20 + 9 * len(ns_all)))

    for (df, kf, sigma), n_map in sorted(groups.items()):
        rates = [n_map.get(n) for n in ns_all]
        cells = []
        for r in rates:
            if r is None:
                cells.append("  .  ")
            else:
                cells.append(f" {r:.2f}")
        # Flag if there's a degradation of > 0.2 from smallest to largest N
        valid = [(n, n_map[n]) for n in ns_all if n in n_map]
        if len(valid) >= 2:
            _, sr_first = valid[0]
            _, sr_last = valid[-1]
            flag = " ↓" if sr_last < sr_first - 0.20 else ""
        else:
            flag = ""
        print(f"  {df:5.2f}  {kf:5.2f}  {sigma:6.1f}  " + "  ".join(cells) + flag)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _find_summaries(root: Path) -> list[Path]:
    return sorted(root.rglob("stability_sweep_summary.json"))


def main(argv: list[str] | None = None) -> None:  # noqa: PLR0912, PLR0915
    global STABLE_THRESHOLD, UNCERTAIN_THRESHOLD  # noqa: PLW0603
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="stability_sweep_summary.json file(s). Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmark_results"),
        help="Root directory for auto-discovery (default: benchmark_results)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=STABLE_THRESHOLD,
        help=f"Success-rate threshold for 'stable' (default: {STABLE_THRESHOLD})",
    )
    parser.add_argument(
        "--df-kf-map",
        action="store_true",
        help="Print per-(N, sigma) Df×kf stability maps (verbose)",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        nargs="*",
        default=[64, 128],
        metavar="N",
        help="N values to consider for preset recommendations (default: 64 128)",
    )
    args = parser.parse_args(argv)

    STABLE_THRESHOLD = args.threshold

    # Resolve input files
    files = list(args.files)
    if not files:
        files = _find_summaries(args.root)
        if not files:
            print(
                f"No stability_sweep_summary.json found under {args.root}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("  PyFracVAL Stability Analysis")
    print(f"{'=' * 70}")
    print(f"  Loaded {len(files)} file(s):")
    for f in files:
        print(f"    {f}")

    rows = _load_results(files)
    print(f"  Total unique (N, Df, kf, sigma) combos: {len(rows)}")
    ns = sorted({r["N"] for r in rows})
    print(f"  N values: {ns}")
    sigmas = sorted({r["rp_gstd"] for r in rows})
    print(f"  Sigma values: {sigmas}")
    dfs = sorted({round(r["Df"], 4) for r in rows})
    kfs = sorted({round(r["kf"], 4) for r in rows})
    print(f"  Df values: {dfs}")
    print(f"  kf values: {kfs}")
    print(f"  Stable threshold: {STABLE_THRESHOLD:.0%}")
    print()

    n_stable = sum(1 for r in rows if r["success_rate"] >= STABLE_THRESHOLD)
    n_uncertain = sum(
        1 for r in rows if UNCERTAIN_THRESHOLD <= r["success_rate"] < STABLE_THRESHOLD
    )
    n_unstable = sum(1 for r in rows if r["success_rate"] < UNCERTAIN_THRESHOLD)
    print(
        f"  Overall: {n_stable} stable ({n_stable / len(rows):.0%})  "
        f"{n_uncertain} borderline ({n_uncertain / len(rows):.0%})  "
        f"{n_unstable} unstable ({n_unstable / len(rows):.0%})"
    )

    # --- Df/kf map ---
    if args.df_kf_map:
        print(f"\n{'=' * 70}")
        print(
            "  Stability map: Df × kf plane  (S=stable ≥90%, ?=borderline, X=unstable)"
        )
        print(f"{'=' * 70}")
        print_df_kf_map(rows)

    # --- X map ---
    print(f"\n{'=' * 70}")
    print("  Stability vs. dimensionless X = (N/kf)^(1/Df)")
    print(f"{'=' * 70}")
    print_x_map(rows)

    # --- Max safe X per sigma ---
    safe_x = max_safe_x(rows)
    print(f"\n{'=' * 70}")
    print("  Maximum safe X per sigma (largest X with mean_sr >= threshold)")
    print(f"{'=' * 70}")
    for sigma in sorted(safe_x):
        print(f"  sigma={sigma}  max_safe_X = {safe_x[sigma]:.1f}")

    # --- N-scaling ---
    print(f"\n{'=' * 70}")
    print("  N-scaling check")
    print(f"{'=' * 70}")
    print_n_scaling(rows)

    # --- Preset recommendations ---
    print(f"\n{'=' * 70}")
    print("  Preset recommendations")
    print(f"{'=' * 70}")
    recommend_presets(rows, args.target_n)

    # --- Research preset summary (fixed target_n=all available) ---
    all_ns = sorted({r["N"] for r in rows})
    if len(all_ns) > 1 and all_ns != args.target_n:
        print(f"{'=' * 70}")
        print(f"  Preset recommendations across ALL tested N = {all_ns}")
        print(f"{'=' * 70}")
        recommend_presets(rows, all_ns)


if __name__ == "__main__":
    main()
