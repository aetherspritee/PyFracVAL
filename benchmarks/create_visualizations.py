#!/usr/bin/env python3
"""
Create comprehensive visualizations for all benchmark results.

Works without seaborn - uses only matplotlib.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(filepath):
    """Load JSON data."""
    with open(filepath) as f:
        return json.load(f)


def create_sigma_investigation_plots():
    """Create plots for sigma investigation results."""
    print("Creating sigma investigation visualizations...")

    # Load data
    data_file = Path("benchmark_results/sigma_investigation/sigma_investigation.json")
    if not data_file.exists():
        print(f"  Skipping - file not found: {data_file}")
        return

    data = load_json(data_file)
    results = data["results"]

    # Create output directory
    output_dir = Path("benchmark_results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Success rate by sigma
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sigmas = sorted(set(r["sigma"] for r in results))
    success_rates = []
    size_ratios = []

    for sigma in sigmas:
        sigma_results = [r for r in results if r["sigma"] == sigma]
        successes = sum(1 for r in sigma_results if r["success"])
        success_rates.append(successes / len(sigma_results) * 100)

        # Average size ratio for this sigma
        avg_ratio = np.mean(
            [r["particle_size_stats"]["size_ratio"] for r in sigma_results]
        )
        size_ratios.append(avg_ratio)

    # Left plot: Success rate vs sigma
    colors = [
        "green" if sr >= 80 else "orange" if sr >= 40 else "red" for sr in success_rates
    ]
    bars = ax1.bar(
        range(len(sigmas)), success_rates, color=colors, alpha=0.7, edgecolor="black"
    )
    ax1.set_xticks(range(len(sigmas)))
    ax1.set_xticklabels([f"{s:.1f}" for s in sigmas])
    ax1.set_xlabel("Sigma (Geometric Std Dev)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Success Rate vs Particle Size Distribution", fontsize=14, fontweight="bold"
    )
    ax1.axhline(80, color="orange", linestyle="--", alpha=0.5, label="80% threshold")
    ax1.axhline(50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend()
    ax1.set_ylim(0, 105)

    # Add percentage labels on bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{rate:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Right plot: Size ratio vs sigma
    ax2.plot(sigmas, size_ratios, "o-", linewidth=2, markersize=10, color="steelblue")
    ax2.axhline(
        4.9, color="red", linestyle="--", linewidth=2, label="Failure threshold (~4.9x)"
    )
    ax2.fill_between(sigmas, 0, 4.9, alpha=0.2, color="green", label="Safe zone")
    ax2.fill_between(
        sigmas, 4.9, max(size_ratios) * 1.1, alpha=0.2, color="red", label="Danger zone"
    )
    ax2.set_xlabel("Sigma (Geometric Std Dev)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Size Ratio (max/min diameter)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Particle Size Ratio vs Distribution Width", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    ax2.set_yscale("log")

    # Add annotations
    for sigma, ratio in zip(sigmas, size_ratios):
        ax2.annotate(
            f"{ratio:.1f}x",
            (sigma, ratio),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    output_file = output_dir / "sigma_investigation.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")


def create_extreme_df_plots():
    """Create plots for extreme Df results."""
    print("Creating extreme Df visualizations...")

    # Load data
    data_file = Path("benchmark_results/extreme_df/extreme_df_results.json")
    if not data_file.exists():
        print(f"  Skipping - file not found: {data_file}")
        return

    data = load_json(data_file)
    results = data["results"]

    output_dir = Path("benchmark_results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Df range heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique Df values and strategies
    df_values = sorted(set(r["Df"] for r in results))
    strategies = ["empirical", "fixed_1.0"]

    # Create matrix
    success_matrix = np.zeros((len(strategies), len(df_values)))

    for i, strategy in enumerate(strategies):
        for j, df in enumerate(df_values):
            df_results = [
                r for r in results if r["Df"] == df and r["kf_strategy"] == strategy
            ]
            if df_results:
                success_matrix[i, j] = df_results[0]["success_rate"] * 100

    # Create heatmap
    im = ax.imshow(success_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(range(len(df_values)))
    ax.set_xticklabels([f"{df:.2f}" for df in df_values], rotation=45, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(["kf = 3.0 - Df\n(Empirical)", "kf = 1.0\n(Fixed)"])

    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(df_values)):
            value = success_matrix[i, j]
            color = "white" if value < 50 else "black"
            text = ax.text(
                j,
                i,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )

    ax.set_xlabel("Df (Fractal Dimension)", fontsize=12, fontweight="bold")
    ax.set_ylabel("kf Strategy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Extreme Df Success Rates: Empirical vs Fixed kf",
        fontsize=14,
        fontweight="bold",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Success Rate (%)", fontsize=11)

    plt.tight_layout()
    output_file = output_dir / "extreme_df_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")

    # Figure 2: Empirical relationship validation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Get empirical results only
    empirical_results = [r for r in results if r["kf_strategy"] == "empirical"]
    df_emp = [r["Df"] for r in empirical_results]
    kf_emp = [r["kf"] for r in empirical_results]
    success_emp = [r["success_rate"] * 100 for r in empirical_results]

    # Left plot: Df vs kf relationship
    scatter = ax1.scatter(
        df_emp,
        kf_emp,
        c=success_emp,
        s=200,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        edgecolors="black",
        linewidth=2,
    )

    # Plot theoretical line
    df_line = np.linspace(min(df_emp), max(df_emp), 100)
    kf_line = 3.0 - df_line
    ax1.plot(df_line, kf_line, "k--", linewidth=2, label="kf = 3.0 - Df", alpha=0.7)

    ax1.set_xlabel("Df (Fractal Dimension)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("kf (Fractal Prefactor)", fontsize=12, fontweight="bold")
    ax1.set_title("Empirical Relationship Validation", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label("Success Rate (%)", fontsize=10)

    # Right plot: Success rate by regime
    low_df = [r for r in empirical_results if r["Df"] < 2.0]
    high_df = [r for r in empirical_results if r["Df"] >= 2.5]

    regimes = ["Low Df\n(1.3-1.5)", "High Df\n(2.5-2.9)"]
    avg_success = [
        np.mean([r["success_rate"] * 100 for r in low_df]) if low_df else 0,
        np.mean([r["success_rate"] * 100 for r in high_df]) if high_df else 0,
    ]
    colors_regime = ["lightcoral", "lightblue"]

    bars = ax2.bar(
        regimes,
        avg_success,
        color=colors_regime,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax2.axhline(80, color="orange", linestyle="--", linewidth=2, label="80% reliable")
    ax2.set_ylabel("Average Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Success by Df Regime (Empirical kf)", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    # Add percentage labels
    for bar, val in zip(bars, avg_success):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    plt.tight_layout()
    output_file = output_dir / "extreme_df_relationship.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")


def create_parameter_sweep_plots():
    """Create plots for parameter sweep."""
    print("Creating parameter sweep visualizations...")

    # Load data
    data_file = Path("benchmark_results/parameter_sweep/sweep_results_quick.json")
    if not data_file.exists():
        print(f"  Skipping - file not found: {data_file}")
        return

    data = load_json(data_file)
    results = data["results"]

    output_dir = Path("benchmark_results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sigma values
    sigmas = sorted(set(r["sigma_p_geo"] for r in results))

    for sigma in sigmas:
        sigma_results = [r for r in results if r["sigma_p_geo"] == sigma]

        # Get unique Df and kf values
        df_values = sorted(set(r["Df"] for r in sigma_results))
        kf_values = sorted(set(r["kf"] for r in sigma_results))

        # Create success matrix
        success_matrix = np.zeros((len(df_values), len(kf_values)))

        for r in sigma_results:
            i = df_values.index(r["Df"])
            j = kf_values.index(r["kf"])
            success_matrix[i, j] = r["success_rate"] * 100

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(success_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(range(len(kf_values)))
        ax.set_xticklabels([f"{kf:.1f}" for kf in kf_values])
        ax.set_yticks(range(len(df_values)))
        ax.set_yticklabels([f"{df:.1f}" for df in df_values])

        # Add text annotations
        for i in range(len(df_values)):
            for j in range(len(kf_values)):
                value = success_matrix[i, j]
                color = "white" if value < 50 else "black"
                text = ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        ax.set_xlabel("kf (Fractal Prefactor)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Df (Fractal Dimension)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Success Rate Heatmap (sigma={sigma})", fontsize=14, fontweight="bold"
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Success Rate (%)", fontsize=11)

        plt.tight_layout()
        output_file = output_dir / f"heatmap_sigma_{sigma:.1f}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_file}")

    # Create Df-kf relationship plot for sigma=1.3
    sigma_1_3_results = [r for r in results if r["sigma_p_geo"] == 1.3]

    fig, ax = plt.subplots(figsize=(10, 6))

    # For each Df, find optimal kf (highest success rate)
    df_values = sorted(set(r["Df"] for r in sigma_1_3_results))
    optimal_points = []

    for df in df_values:
        df_results = [r for r in sigma_1_3_results if r["Df"] == df]
        best = max(df_results, key=lambda x: x["success_rate"])
        optimal_points.append((df, best["kf"], best["success_rate"] * 100))

    dfs = [p[0] for p in optimal_points]
    kfs = [p[1] for p in optimal_points]
    success = [p[2] for p in optimal_points]

    # Plot scatter
    scatter = ax.scatter(
        dfs,
        kfs,
        c=success,
        s=200,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        edgecolors="black",
        linewidth=2,
    )

    # Plot empirical relationship
    df_line = np.linspace(min(dfs), max(dfs), 100)
    kf_line = 3.0 - df_line
    ax.plot(df_line, kf_line, "k--", linewidth=2, label="kf = 3.0 - Df", alpha=0.7)

    ax.set_xlabel("Df (Fractal Dimension)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Optimal kf (Highest Success)", fontsize=12, fontweight="bold")
    ax.set_title("Df-kf Relationship (sigma=1.3)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Success Rate (%)", fontsize=10)

    plt.tight_layout()
    output_file = output_dir / "df_kf_relationship.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")


def create_summary_figure():
    """Create a summary figure combining all key findings."""
    print("Creating summary figure...")

    output_dir = Path("benchmark_results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Df range
    ax1 = fig.add_subplot(gs[0, 0])
    df_ranges = [
        ("Reliable\n1.4-2.5", 95, "green"),
        ("Marginal\n1.3, 2.6", 70, "orange"),
        ("Impossible\n<1.3, ≥2.7", 0, "red"),
    ]
    labels = [r[0] for r in df_ranges]
    values = [r[1] for r in df_ranges]
    colors = [r[2] for r in df_ranges]

    bars = ax1.bar(
        range(len(labels)),
        values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Df Range Feasibility", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{val}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Panel 2: Sigma limits
    ax2 = fig.add_subplot(gs[0, 1])
    sigma_data = [
        ("σ≤1.3\n(Safe)", 100, "green"),
        ("σ=1.4-1.5\n(Risky)", 75, "orange"),
        ("σ≥1.6\n(Fails)", 0, "red"),
    ]
    labels = [s[0] for s in sigma_data]
    values = [s[1] for s in sigma_data]
    colors = [s[2] for s in sigma_data]

    bars = ax2.bar(
        range(len(labels)),
        values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Sigma (Distribution Width) Limits", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{val}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Panel 3: kf strategy comparison
    ax3 = fig.add_subplot(gs[0, 2])
    strategies = ["Empirical\nkf=3.0-Df", "Fixed\nkf=1.0"]
    strategy_success = [74, 10]
    colors_strat = ["green", "red"]

    bars = ax3.bar(
        strategies,
        strategy_success,
        color=colors_strat,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax3.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax3.set_title("kf Strategy Comparison", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, strategy_success):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{val}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Panel 4: Size ratio threshold
    ax4 = fig.add_subplot(gs[1, :])
    sigma_values = [1.3, 1.5, 1.8, 2.0, 2.5]
    size_ratios = [2.7, 4.7, 9.6, 14.4, 34.0]
    success_rates = [100, 80, 0, 0, 0]

    # Create two y-axes
    ax4_twin = ax4.twinx()

    # Plot size ratios as bars
    bars = ax4.bar(
        range(len(sigma_values)),
        size_ratios,
        alpha=0.5,
        color="steelblue",
        edgecolor="black",
        linewidth=2,
        label="Size Ratio (max/min)",
    )

    # Plot success rate as line
    line = ax4_twin.plot(
        range(len(sigma_values)),
        success_rates,
        "ro-",
        linewidth=3,
        markersize=10,
        label="Success Rate",
    )

    # Add threshold line
    ax4.axhline(
        4.9, color="red", linestyle="--", linewidth=2, label="Failure Threshold (~4.9x)"
    )

    ax4.set_xticks(range(len(sigma_values)))
    ax4.set_xticklabels([f"{s:.1f}" for s in sigma_values])
    ax4.set_xlabel("Sigma (Geometric Std Dev)", fontsize=12, fontweight="bold")
    ax4.set_ylabel(
        "Size Ratio (max/min)", fontsize=12, fontweight="bold", color="steelblue"
    )
    ax4_twin.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold", color="red")
    ax4.set_title("Size Ratio vs Success Rate", fontsize=14, fontweight="bold")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    # Add title
    fig.suptitle(
        "PyFracVAL Parameter Space: Key Findings Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(output_dir / "summary_figure.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'summary_figure.png'}")


def main():
    """Main entry point."""
    print("=" * 80)
    print("PyFracVAL Benchmark Visualization Suite")
    print("=" * 80)
    print()

    # Create all visualizations
    create_sigma_investigation_plots()
    create_extreme_df_plots()
    create_parameter_sweep_plots()
    create_summary_figure()

    print()
    print("=" * 80)
    print("✅ All visualizations complete!")
    print("=" * 80)
    print()
    print("Visualizations saved to: benchmark_results/visualizations/")
    print()
    print("Generated files:")
    viz_dir = Path("benchmark_results/visualizations")
    if viz_dir.exists():
        for f in sorted(viz_dir.glob("*.png")):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
