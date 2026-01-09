#!/usr/bin/env python3
"""
Visualization script for parameter sweep results.

Creates heatmaps and plots to visualize parameter space.

Usage:
    uv run python benchmarks/visualize_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_sweep_results(filepath: str) -> dict:
    """Load sweep results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def create_success_heatmap(results: list[dict], sigma: float, output_dir: Path):
    """Create heatmap of success rates for a given sigma."""

    # Extract unique Df and kf values
    df_values = sorted(set(r["Df"] for r in results if r["sigma_p_geo"] == sigma))
    kf_values = sorted(set(r["kf"] for r in results if r["sigma_p_geo"] == sigma))

    # Create matrix
    success_matrix = np.zeros((len(df_values), len(kf_values)))

    for r in results:
        if r["sigma_p_geo"] == sigma:
            i = df_values.index(r["Df"])
            j = kf_values.index(r["kf"])
            success_matrix[i, j] = r["success_rate"] * 100

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        success_matrix,
        xticklabels=[f"{kf:.2f}" for kf in kf_values],
        yticklabels=[f"{df:.2f}" for df in df_values],
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Success Rate (%)"},
    )
    plt.xlabel("kf (Fractal Prefactor)", fontsize=12)
    plt.ylabel("Df (Fractal Dimension)", fontsize=12)
    plt.title(
        f"Parameter Space Success Rates (sigma={sigma})", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    output_path = output_dir / f"heatmap_sigma_{sigma:.1f}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved heatmap: {output_path}")


def create_df_kf_relationship_plot(results: list[dict], sigma: float, output_dir: Path):
    """Plot the inverse Df-kf relationship."""

    # For each Df, find the kf with highest success rate
    df_values = sorted(set(r["Df"] for r in results if r["sigma_p_geo"] == sigma))

    optimal_kf = []
    for df in df_values:
        df_results = [r for r in results if r["Df"] == df and r["sigma_p_geo"] == sigma]
        best = max(df_results, key=lambda x: x["success_rate"])
        optimal_kf.append((df, best["kf"], best["success_rate"]))

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Df vs optimal kf
    dfs = [x[0] for x in optimal_kf]
    kfs = [x[1] for x in optimal_kf]
    success = [x[2] * 100 for x in optimal_kf]

    scatter = ax1.scatter(dfs, kfs, c=success, s=100, cmap="RdYlGn", vmin=0, vmax=100)
    ax1.plot(dfs, kfs, "k--", alpha=0.3, label="Trend")
    ax1.set_xlabel("Df (Fractal Dimension)", fontsize=12)
    ax1.set_ylabel("Optimal kf (Fractal Prefactor)", fontsize=12)
    ax1.set_title(
        f"Inverse Df-kf Relationship (sigma={sigma})", fontsize=13, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Success Rate (%)", fontsize=10)

    # Right plot: Success rate vs Df
    ax2.bar(dfs, success, color=[plt.cm.RdYlGn(s / 100) for s in success])
    ax2.axhline(80, color="orange", linestyle="--", label="80% threshold")
    ax2.set_xlabel("Df (Fractal Dimension)", fontsize=12)
    ax2.set_ylabel("Best Success Rate (%)", fontsize=12)
    ax2.set_title(f"Feasibility by Df (sigma={sigma})", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    plt.tight_layout()

    output_path = output_dir / f"df_kf_relationship_sigma_{sigma:.1f}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved relationship plot: {output_path}")


def create_sigma_comparison_plot(results: list[dict], output_dir: Path):
    """Compare success rates across different sigma values."""

    sigma_values = sorted(set(r["sigma_p_geo"] for r in results))

    fig, axes = plt.subplots(1, len(sigma_values), figsize=(5 * len(sigma_values), 5))

    if len(sigma_values) == 1:
        axes = [axes]

    for idx, sigma in enumerate(sigma_values):
        sigma_results = [r for r in results if r["sigma_p_geo"] == sigma]

        # Calculate overall success rate
        total_trials = sum(r["trials"] for r in sigma_results)
        total_successes = sum(r["successes"] for r in sigma_results)
        overall_success = (
            total_successes / total_trials * 100 if total_trials > 0 else 0
        )

        # Count by quality
        excellent = len([r for r in sigma_results if r["success_rate"] >= 0.9])
        good = len([r for r in sigma_results if 0.7 <= r["success_rate"] < 0.9])
        poor = len([r for r in sigma_results if 0.3 <= r["success_rate"] < 0.7])
        failed = len([r for r in sigma_results if r["success_rate"] < 0.3])

        # Create bar plot
        categories = [
            "Excellent\n(≥90%)",
            "Good\n(70-90%)",
            "Poor\n(30-70%)",
            "Failed\n(<30%)",
        ]
        counts = [excellent, good, poor, failed]
        colors = ["green", "yellow", "orange", "red"]

        axes[idx].bar(categories, counts, color=colors, alpha=0.7)
        axes[idx].set_ylabel("Number of Combinations", fontsize=10)
        axes[idx].set_title(
            f"sigma={sigma}\nOverall: {overall_success:.0f}%",
            fontsize=11,
            fontweight="bold",
        )
        axes[idx].grid(True, alpha=0.3, axis="y")

        # Add count labels
        for i, count in enumerate(counts):
            axes[idx].text(i, count + 1, str(count), ha="center", fontsize=9)

    plt.tight_layout()

    output_path = output_dir / "sigma_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved sigma comparison: {output_path}")


def create_runtime_analysis(results: list[dict], output_dir: Path):
    """Analyze runtime patterns."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Separate successful and failed runs
    successful = [r for r in results if r["success_rate"] > 0]
    failed = [r for r in results if r["success_rate"] == 0]

    # Runtime distribution
    success_runtimes = [r["avg_runtime"] for r in successful]
    failed_runtimes = [r["avg_runtime"] for r in failed]

    ax1.hist(
        [success_runtimes, failed_runtimes],
        bins=30,
        label=["Successful", "Failed"],
        color=["green", "red"],
        alpha=0.6,
    )
    ax1.set_xlabel("Runtime (seconds)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Runtime Distribution", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Runtime vs success rate
    all_runtimes = [r["avg_runtime"] for r in results]
    all_success_rates = [r["success_rate"] * 100 for r in results]

    ax2.scatter(all_success_rates, all_runtimes, alpha=0.5, s=50)
    ax2.set_xlabel("Success Rate (%)", fontsize=12)
    ax2.set_ylabel("Average Runtime (seconds)", fontsize=12)
    ax2.set_title("Runtime vs Success Rate", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(
        np.median([r for r in all_runtimes if r > 0.1]),
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="Median (successful)",
    )
    ax2.legend()

    plt.tight_layout()

    output_path = output_dir / "runtime_analysis.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved runtime analysis: {output_path}")


def main():
    """Main visualization script."""

    # Check for results files
    results_dir = Path("benchmark_results/parameter_sweep")
    output_dir = Path("benchmark_results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for results file
    results_file = results_dir / "sweep_results.json"
    if not results_file.exists():
        results_file = results_dir / "sweep_results_quick.json"

    if not results_file.exists():
        print("Error: No sweep results file found!")
        print(f"Looking for: {results_dir / 'sweep_results.json'}")
        return

    print(f"Loading results from: {results_file}")
    data = load_sweep_results(results_file)
    results = data["results"]

    print(f"Loaded {len(results)} results")
    print("Creating visualizations...")
    print()

    # Get unique sigma values
    sigma_values = sorted(set(r["sigma_p_geo"] for r in results))

    # Create heatmaps for each sigma
    for sigma in sigma_values:
        print(f"Processing sigma={sigma}...")
        create_success_heatmap(results, sigma, output_dir)
        create_df_kf_relationship_plot(results, sigma, output_dir)

    # Create comparison plots
    create_sigma_comparison_plot(results, output_dir)
    create_runtime_analysis(results, output_dir)

    print()
    print(f"✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
