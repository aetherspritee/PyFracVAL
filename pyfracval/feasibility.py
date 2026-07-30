r"""Predict whether a (Df, kf, sigma, N) request is generable.

Moran et al. (2019) note only that FracVAL works "as long as the pair of
Df and kf falls in the valid range where it is possible to generate such
fractal aggregates", without characterizing that range. This module turns
the measured range into a *predictive* one, so the tool can say up front
that a request is unlikely to succeed instead of discovering it after
twenty retries.

The model is a logistic fit to
``benchmark_results/boundary_sweep_v2/`` - 4200 trials over
Df in [1.8, 2.5], kf in [0.8, 1.4], sigma in {1.0, 1.5, 1.9},
N in {64...1024}, against the current defaults. Coefficients are baked in
below rather than refitted at import; refresh them with
``benchmarks/fit_feasibility_boundary.py`` after any change that moves
the boundary.

Scope and honesty about it
--------------------------
This is a semi-empirical fit over the grid actually measured, not a
theory. It is interpolation inside that box and extrapolation outside it,
and it says nothing about *why* the boundary sits where it does. Treat a
low predicted probability as "expect this to be slow and often fail",
not as proof of impossibility - the sweep itself found non-zero success
at points the earlier implementation could not reach at all.
"""

import logging
import math

logger = logging.getLogger(__name__)

# Logistic coefficients from benchmarks/fit_feasibility_boundary.py.
# Features: [1, Df, kf, log10(sigma_eff), log10(N), Df*kf, Df*log10(sigma_eff)]
# where sigma_eff = max(sigma, 1.0). See that script for the fit report.
#: Fit quality on the grid it was trained on (trial-weighted): mean
#: absolute error 0.035 in predicted success rate, Brier score 0.011, and
#: 97.7% agreement with the measured >=50% feasibility call.
_COEFFS: dict[str, float] = {
    "intercept": 107.7910,
    "df": -29.4749,
    "kf": 3.2182,
    "log_sigma": -10.9551,
    "log_n": -6.7599,
    "df_kf": -7.3129,
    "df_log_sigma": -9.0494,
}

#: Set True once real coefficients are baked in.
_FITTED = True

# The grid the fit is interpolating within.
GRID_BOUNDS = {
    "df": (1.8, 2.5),
    "kf": (0.8, 1.4),
    "sigma": (1.0, 1.9),
    "n": (64, 1024),
}


def set_coefficients(coeffs: dict[str, float]) -> None:
    """Install fitted coefficients (used by the fitting script)."""
    global _FITTED
    _COEFFS.update(coeffs)
    _FITTED = True


def _features(df: float, kf: float, sigma: float, n: int) -> dict[str, float]:
    log_sigma = math.log10(max(sigma, 1.0) + 1e-12) if sigma > 1.0 else 0.0
    log_n = math.log10(max(n, 1))
    return {
        "intercept": 1.0,
        "df": df,
        "kf": kf,
        "log_sigma": log_sigma,
        "log_n": log_n,
        "df_kf": df * kf,
        "df_log_sigma": df * log_sigma,
    }


def estimate_success_probability(
    df: float, kf: float, sigma: float, n: int
) -> float | None:
    """Predicted per-trial success probability, or None if unfitted.

    "Per trial" means one ``run_simulation`` call including its internal
    retries - the same quantity the stability sweep reports.
    """
    if not _FITTED:
        return None
    feats = _features(df, kf, sigma, n)
    z = sum(_COEFFS.get(k, 0.0) * v for k, v in feats.items())
    return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


def max_feasible_df(
    kf: float, sigma: float, n: int, threshold: float = 0.5, step: float = 0.01
) -> float | None:
    """Largest Df whose predicted success probability still exceeds
    ``threshold``, scanned over the fitted range."""
    if not _FITTED:
        return None
    lo, hi = GRID_BOUNDS["df"]
    best = None
    x = lo
    while x <= hi + 1e-9:
        p = estimate_success_probability(x, kf, sigma, n)
        if p is not None and p >= threshold:
            best = x
        x += step
    return best


def out_of_fitted_range(df: float, kf: float, sigma: float, n: int) -> list[str]:
    """Which requested parameters sit outside the fitted grid."""
    outside = []
    for name, value in (("df", df), ("kf", kf), ("sigma", sigma), ("n", n)):
        lo, hi = GRID_BOUNDS[name]
        if value < lo or value > hi:
            outside.append(f"{name}={value} (fitted range {lo}-{hi})")
    return outside


def warn_if_difficult(
    df: float, kf: float, sigma: float, n: int, threshold: float = 0.5
) -> str | None:
    """Emit a warning when a request looks unlikely to succeed.

    Returns the warning text (also logged), or None when the request
    looks fine or no fit is available. Deliberately advisory: the fit is
    empirical, so this never blocks a run.
    """
    p = estimate_success_probability(df, kf, sigma, n)
    if p is None or p >= threshold:
        return None

    ceiling = max_feasible_df(kf, sigma, n, threshold=threshold)
    parts = [
        f"Requested Df={df}, kf={kf}, sigma={sigma}, N={n} has an estimated "
        f"success probability of {p:.0%} per attempt."
    ]
    if ceiling is not None:
        parts.append(f"At this kf/sigma/N, Df up to about {ceiling:.2f} is reliable.")
    else:
        parts.append(
            "No Df in the fitted range looks reliable at this kf/sigma/N; "
            "try a lower kf or a narrower size distribution."
        )
    outside = out_of_fitted_range(df, kf, sigma, n)
    if outside:
        parts.append("Note this is extrapolation: " + ", ".join(outside) + ".")
    message = " ".join(parts)
    logger.warning(message)
    return message
