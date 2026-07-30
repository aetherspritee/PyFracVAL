#!/usr/bin/env python3
"""Fit a predictive feasibility boundary from a stability sweep.

docs/source/hard_regime_boundary_sweep.md and
docs/source/boundary_sweep_v2.md *measure* where generation collapses.
This turns that measurement into a model that can be evaluated at
parameters the grid did not sample, so `pyfracval.feasibility` can warn
before a run rather than after twenty retries.

Model: logistic regression of per-trial success on

    [1, Df, kf, log10(sigma_eff), log10(N), Df*kf, Df*log10(sigma_eff)]

The two interaction terms are not decoration. Every sweep so far found
the Df x kf effect to be sign-flipping (at low Df larger kf helps, at
high Df smaller kf helps) and the collapse boundary to move down in Df as
sigma rises; a purely additive model cannot express either.

Fitted by plain gradient descent on the log-likelihood - the design
matrix is 840 rows by 7 columns, so this needs no optimizer dependency,
and keeping scikit-learn out of the runtime is worth a few lines here.

Usage:
    devenv shell -- uv run python benchmarks/fit_feasibility_boundary.py
    devenv shell -- uv run python benchmarks/fit_feasibility_boundary.py --csv PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_CSV = Path(
    "benchmark_results/boundary_sweep_v2/stability_sweeps/stability_sweep_summary.csv"
)
FEATURE_NAMES = [
    "intercept",
    "df",
    "kf",
    "log_sigma",
    "log_n",
    "df_kf",
    "df_log_sigma",
]


def build_design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, successes, trials) with one row per grid cell."""
    X, succ, tri = [], [], []
    for r in rows:
        df = float(r["Df"])
        kf = float(r["kf"])
        sigma = float(r["rp_gstd"])
        n = int(r["N"])
        log_sigma = math.log10(sigma) if sigma > 1.0 else 0.0
        log_n = math.log10(n)
        X.append([1.0, df, kf, log_sigma, log_n, df * kf, df * log_sigma])
        succ.append(int(r["successes"]))
        tri.append(int(r["trials"]))
    return np.array(X), np.array(succ, dtype=float), np.array(tri, dtype=float)


def fit_logistic(
    X: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    iters: int = 40000,
    lr: float = 0.05,
) -> np.ndarray:
    """Binomial logistic fit by gradient ascent on the log-likelihood."""
    # Standardize for conditioning, then transform coefficients back so
    # the exported model can be evaluated on raw features.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    mu[0], sd[0] = 0.0, 1.0  # leave the intercept column alone
    Xs = (X - mu) / sd

    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = Xs @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        grad = Xs.T @ (successes - trials * p)
        w += lr * grad / trials.sum()

    # Undo standardization: z = w·(x-mu)/sd = (w/sd)·x - sum(w*mu/sd)
    raw = w / sd
    raw[0] = w[0] - float(np.sum(w[1:] * mu[1:] / sd[1:]))
    return raw


def report(X, successes, trials, w) -> dict:
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    observed = successes / trials

    # Weighted metrics: each cell carries `trials` observations.
    mae = float(np.average(np.abs(p - observed), weights=trials))
    brier = float(np.average((p - observed) ** 2, weights=trials))
    # Accuracy of the >=50% "will this work" call, per trial.
    pred_yes = p >= 0.5
    obs_yes = observed >= 0.5
    agree = float(np.average((pred_yes == obs_yes).astype(float), weights=trials))

    return {
        "weighted_mae": mae,
        "brier_score": brier,
        "boundary_call_accuracy": agree,
        "n_cells": int(X.shape[0]),
        "n_trials": int(trials.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument(
        "--out", type=Path, default=Path("benchmark_results/feasibility_fit.json")
    )
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"No sweep summary at {args.csv}")

    rows = list(csv.DictReader(args.csv.open()))
    X, successes, trials = build_design(rows)
    w = fit_logistic(X, successes, trials)
    metrics = report(X, successes, trials, w)

    coeffs = dict(zip(FEATURE_NAMES, (float(v) for v in w)))

    print("=" * 76)
    print(f"Feasibility fit from {args.csv}")
    print("=" * 76)
    print(f"cells={metrics['n_cells']}  trials={metrics['n_trials']}")
    print("\nCoefficients (raw features):")
    for name, value in coeffs.items():
        print(f"  {name:14s} {value:+10.4f}")
    print("\nFit quality (trial-weighted):")
    print(f"  mean abs error in predicted success rate : {metrics['weighted_mae']:.3f}")
    print(f"  Brier score                              : {metrics['brier_score']:.3f}")
    print(
        f"  agreement on the >=50% feasibility call  : {metrics['boundary_call_accuracy']:.1%}"
    )

    # Show the implied Df ceiling, which is the number a user cares about.
    from pyfracval import feasibility

    feasibility.set_coefficients(coeffs)
    print("\nImplied max feasible Df (predicted success >= 50%):")
    print(f"  {'sigma':>6} {'kf':>5} " + " ".join(f"N={n:<5}" for n in (64, 256, 1024)))
    for sigma in (1.0, 1.5, 1.9):
        for kf in (0.8, 1.0, 1.2, 1.4):
            cells = []
            for n in (64, 256, 1024):
                d = feasibility.max_feasible_df(kf, sigma, n)
                cells.append(f"{d:.2f} " if d else "none ")
            print(f"  {sigma:6.1f} {kf:5.1f} " + " ".join(f"{c:<7}" for c in cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"coefficients": coeffs, "metrics": metrics}, indent=2)
    )
    print(f"\nWrote {args.out}")
    print("\nPaste these coefficients into pyfracval/feasibility.py::_COEFFS")
    print("and set _FITTED = True.")


if __name__ == "__main__":
    main()
