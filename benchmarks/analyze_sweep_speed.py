#!/usr/bin/env python3
"""Fit an empirical runtime model from a stability_sweep.py summary, and
render the thesis-style plots (success rate vs Df, Df x kf heatmaps,
runtime distributions, ...) as static images for docs embedding.

Usage:
    devenv shell -- uv run --extra plot python benchmarks/analyze_sweep_speed.py \
        benchmark_results/full_thesis_replication_sweep/stability_sweeps/stability_sweep_summary.json \
        --out-dir docs/source/_static/sweep

Methodology for the speed model
--------------------------------
The summary only has per-combination averages (avg_runtime_s over 5 seeds),
not individual per-trial rows - stability_sweep.py's Dask path never
implemented raw per-trial JSONL writing (a separate gap, not fixed here).
A combo whose 5 seeds are a mix of successes and failures has an
avg_runtime_s that blends two very different cost regimes (a success stops
at the first working attempt; a failure exhausts all 20 internal retries),
so this script restricts each fit to "clean" combinations - all 5 seeds
succeeded, or all 5 failed - to keep each regression honest about which
regime it's describing. Mixed combinations are reported but excluded from
the fits.

X = (N / kf)^(1/Df) is used as the primary feature alongside raw N - it's
the same dimensionless "how many particle-diameters must the aggregate
span" ratio benchmarks/analyze_stability.py already uses for stability
maps, and is the natural physically-motivated variable here rather than
throwing Df/kf in as independent raw features.

Model: log(runtime_s) = a + b*log(N) + c*log(X), fit separately for the
success-path and failure-path regimes via ordinary least squares
(numpy.linalg.lstsq - no new dependency needed for a 3-parameter linear fit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    return data["results"]


def _compute_x(row: dict) -> float:
    return (row["N"] / row["kf"]) ** (1.0 / row["Df"])


def fit_log_linear(rows: list[dict]) -> dict:
    """OLS fit of log(avg_runtime_s) ~ a + b*log(N) + c*log(X)."""
    n_vals = np.array([r["N"] for r in rows], dtype=float)
    x_vals = np.array([_compute_x(r) for r in rows], dtype=float)
    y_vals = np.array([r["avg_runtime_s"] for r in rows], dtype=float)

    # Guard against non-positive runtimes (shouldn't happen, but a fit on
    # log(0) would silently produce garbage otherwise).
    mask = y_vals > 0
    n_vals, x_vals, y_vals = n_vals[mask], x_vals[mask], y_vals[mask]

    log_n = np.log(n_vals)
    log_x = np.log(x_vals)
    log_y = np.log(y_vals)

    design = np.column_stack([np.ones_like(log_n), log_n, log_x])
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(design, log_y, rcond=None)
    a, b, c = coeffs

    pred = design @ coeffs
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n_points": int(mask.sum()),
        "a": float(a),
        "b_log_n": float(b),
        "c_log_x": float(c),
        "r_squared": r_squared,
        "equation": f"runtime_s ~= exp({a:.3f}) * N^{b:.3f} * X^{c:.3f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/source/_static/sweep"),
        help="Directory to write plot PNGs into.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.summary_json)
    pure_success = [r for r in rows if r["success_rate"] == 1.0]
    pure_fail = [r for r in rows if r["success_rate"] == 0.0]
    mixed = [r for r in rows if 0 < r["success_rate"] < 1]

    # trial_timeout in the sweep config caps any single trial's wall-clock
    # budget - a real fraction of failures hit that cap rather than
    # naturally exhausting all 20 internal retries, right-censoring their
    # recorded runtime. Fit the "natural" failure-cost model on the
    # uncensored subset; the raw (censored-inclusive) numbers are reported
    # separately since "how long will I actually wait before this gives up,
    # given the timeout" is also a real, useful question - just a different
    # one from "what does an uncapped failure actually cost."
    censor_threshold_s = 85.0
    pure_fail_censored = [
        r for r in pure_fail if r["avg_runtime_s"] >= censor_threshold_s
    ]
    pure_fail_uncensored = [
        r for r in pure_fail if r["avg_runtime_s"] < censor_threshold_s
    ]

    print("=" * 70)
    print(f"Loaded {len(rows)} combinations from {args.summary_json}")
    print(
        f"  pure success (5/5): {len(pure_success)}  "
        f"pure fail (0/5): {len(pure_fail)}  mixed: {len(mixed)}"
    )
    print(
        f"  of pure-fail: {len(pure_fail_censored)} "
        f"({len(pure_fail_censored) / len(pure_fail):.1%}) likely timeout-censored "
        f"(avg_runtime_s >= {censor_threshold_s}s)"
    )
    print("=" * 70)

    print("\n--- Success-path runtime model (all 5 seeds succeeded) ---")
    fit_success = fit_log_linear(pure_success)
    for k, v in fit_success.items():
        print(f"  {k}: {v}")

    print(
        "\n--- Failure-path runtime model, RAW (all 5 seeds failed, incl. timeout-capped) ---"
    )
    fit_fail_raw = fit_log_linear(pure_fail)
    for k, v in fit_fail_raw.items():
        print(f"  {k}: {v}")

    print("\n--- Failure-path runtime model, UNCENSORED (excludes timeout-capped) ---")
    fit_fail_uncensored = fit_log_linear(pure_fail_uncensored)
    for k, v in fit_fail_uncensored.items():
        print(f"  {k}: {v}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_speed_plots(
        pure_success,
        pure_fail_uncensored,
        fit_success,
        fit_fail_uncensored,
        args.out_dir,
    )
    _write_thesis_style_plots(rows, args.out_dir)

    print(f"\nWrote plots to {args.out_dir}")


def _write_speed_plots(
    pure_success, pure_fail, fit_success, fit_fail, out_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, data, fit, title, color in [
        (axes[0], pure_success, fit_success, "Success path (5/5 seeds)", "tab:green"),
        (axes[1], pure_fail, fit_fail, "Failure path (0/5 seeds)", "tab:red"),
    ]:
        x_vals = np.array([_compute_x(r) for r in data])
        y_vals = np.array([r["avg_runtime_s"] for r in data])
        ax.scatter(x_vals, y_vals, s=8, alpha=0.4, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("X = (N/kf)^(1/Df)")
        ax.set_ylabel("avg runtime (s)")
        ax.set_title(f"{title}\nR²={fit['r_squared']:.2f}, n={fit['n_points']}")
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "speed_vs_x.png", dpi=140)
    plt.close(fig)


def _write_thesis_style_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # --- Success rate vs Df (all other params collapsed) ---
    df_values = sorted({r["Df"] for r in rows})
    df_success = []
    for df in df_values:
        subset = [r for r in rows if r["Df"] == df]
        total = sum(r["trials"] for r in subset)
        succ = sum(r["successes"] for r in subset)
        df_success.append(succ / total if total else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_values, df_success, marker="o", color="tab:blue")
    ax.set_xlabel("Df")
    ax.set_ylabel("success rate")
    ax.set_title("Success rate vs Df (collapsed over kf, N, sigma)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_df.png", dpi=140)
    plt.close(fig)

    # --- Df x kf heatmap (collapsed over N, sigma) ---
    kf_values = sorted({r["kf"] for r in rows})
    grid = np.full((len(df_values), len(kf_values)), np.nan)
    df_index = {v: i for i, v in enumerate(df_values)}
    kf_index = {v: i for i, v in enumerate(kf_values)}
    counts = np.zeros_like(grid)
    for r in rows:
        i, j = df_index[r["Df"]], kf_index[r["kf"]]
        if np.isnan(grid[i, j]):
            grid[i, j] = 0.0
        grid[i, j] += r["successes"]
        counts[i, j] += r["trials"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_grid = grid / counts

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        rate_grid,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        extent=(
            min(kf_values) - 0.05,
            max(kf_values) + 0.05,
            min(df_values) - 0.05,
            max(df_values) + 0.05,
        ),
    )
    ax.set_xlabel("kf")
    ax.set_ylabel("Df")
    ax.set_title("Success rate: Df x kf (collapsed over N, sigma)")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(out_dir / "df_kf_heatmap.png", dpi=140)
    plt.close(fig)

    # --- Success rate vs N and sigma ---
    n_values = sorted({r["N"] for r in rows})
    sigma_values = sorted({r["rp_gstd"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sigma in sigma_values:
        rates = []
        for n in n_values:
            subset = [r for r in rows if r["N"] == n and r["rp_gstd"] == sigma]
            total = sum(r["trials"] for r in subset)
            succ = sum(r["successes"] for r in subset)
            rates.append(succ / total if total else 0.0)
        ax.plot(n_values, rates, marker="o", label=f"sigma={sigma}")
    ax.set_xlabel("N")
    ax.set_ylabel("success rate")
    ax.set_title("Success rate vs N, by sigma (collapsed over Df, kf)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_n_sigma.png", dpi=140)
    plt.close(fig)

    # --- Runtime distribution: success vs failure ---
    success_runtimes = [r["avg_runtime_s"] for r in rows if r["success_rate"] == 1.0]
    fail_runtimes = [r["avg_runtime_s"] for r in rows if r["success_rate"] == 0.0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(
        np.log10(min(min(success_runtimes), min(fail_runtimes))),
        np.log10(max(max(success_runtimes), max(fail_runtimes))),
        40,
    )
    ax.hist(
        success_runtimes,
        bins=bins,
        alpha=0.6,
        label="all 5 seeds succeeded",
        color="tab:green",
    )
    ax.hist(
        fail_runtimes, bins=bins, alpha=0.6, label="all 5 seeds failed", color="tab:red"
    )
    ax.set_xscale("log")
    ax.set_xlabel("avg runtime per combination (s)")
    ax.set_ylabel("count")
    ax.set_title("Runtime distribution: success vs failure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_distribution.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
