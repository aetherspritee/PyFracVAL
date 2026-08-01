#!/usr/bin/env python3
"""Particle-cluster (PCA) construction-success probe.

Regenerates the ``tab:pcprobe`` grid of the paper: the fraction of PCA
sub-cluster constructions that complete at ``n = 12`` primary particles,
over ``Df x sigma x kf``.  The table had previously been produced by code
typed inline in a session and never stored; this module is its committed,
deterministic replacement.

What one trial is
-----------------
Exactly the call chain the production sub-cluster stage uses
(:func:`pyfracval.pca_subclusters._run_single_subcluster`), with retries
switched off so that the measurement is the *single-shot* construction
probability:

1. ``rng = numpy.random.default_rng(seed)``
2. ``radii = lognormal_pp_radii(sigma, RP_G, n, rng=rng)``
3. ``PCAggregator(radii, df, kf, TOL_OV, rng=rng, ...).run()``

A trial counts as a success when ``run()`` returns an array and
``not_able_pca`` is False.  The same ``rng`` feeds the radius draw and the
aggregator, so one integer seed determines the whole trial and the grid is
byte-for-byte reproducible.

Seeds
-----
Trial ``i`` of every cell uses seed ``SEED_BASE + i`` (``SEED_BASE = 0``).
Every cell therefore sees the same seed sequence, and the seed list is
written into the JSON output so a rerun can be checked against it.

Usage
-----
    devenv shell -- uv run python benchmarks/pc_probe.py --seeds 200

Writes ``benchmark_results/pc_probe/pc_probe_<seeds>seeds.json`` and
prints the table with Wilson 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyfracval import particle_generation  # noqa: E402
from pyfracval.config import OrchestratorAlgorithmConfig  # noqa: E402
from pyfracval.pca_agg import PCAggregator  # noqa: E402

# --- Fixed probe conditions (paper tab:pcprobe) ---------------------------
N_PARTICLES = 12
RP_G = 100.0
TOL_OV = 1e-6
SEED_BASE = 0
DEFAULT_SEEDS = 200

DFS: tuple[float, ...] = (1.8, 2.5)
SIGMAS: tuple[float, ...] = (1.0, 1.9)
KFS: tuple[float, ...] = (1.0, 1.4, 1.8, 2.2, 2.6, 3.0)

RESULTS_DIR = REPO_ROOT / "benchmark_results" / "pc_probe"

Z95 = 1.959963984540054


class MassGammaPCAggregator(PCAggregator):
    """PCA that solves the *mass* form of the Gamma equation.

    Experiment-only variant used by ``benchmarks/pc_gamma_form.py``.  The
    production aggregator solves the count form
    (:meth:`pyfracval.pca_agg.PCAggregator._gamma_calculation`, which
    documents why); this subclass flips only that one scalar substitution
    and changes nothing else in the search, so the two arms differ in the
    contact relation alone.
    """

    def _gamma_calculation(
        self, m2: float, rg2: float, use_mass: bool = False
    ) -> tuple[bool, float]:
        return super()._gamma_calculation(m2, rg2, use_mass=True)


def wilson_ci(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because most cells of the
    probe sit at or near 0 and 1, where the normal interval is degenerate.

    Parameters
    ----------
    successes : int
        Number of successful trials.
    trials : int
        Total number of trials; must be positive.
    z : float, optional
        Normal quantile, default the 95% two-sided value.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds, clipped to ``[0, 1]``.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = (z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def run_trial(
    df: float,
    kf: float,
    sigma: float,
    seed: int,
    n: int = N_PARTICLES,
    use_mass: bool = False,
) -> bool:
    """Run one single-shot PCA sub-cluster construction.

    Parameters
    ----------
    df, kf : float
        Target fractal dimension and prefactor.
    sigma : float
        Geometric standard deviation of the primary-particle radii.
    seed : int
        Seed for the single generator driving both the radius draw and the
        aggregator's placement search.
    n : int, optional
        Number of primary particles in the sub-cluster.
    use_mass : bool, optional
        Solve the mass form of the Gamma equation instead of the
        production count form.

    Returns
    -------
    bool
        True when the aggregator returned a completed sub-cluster.
    """
    rng = np.random.default_rng(seed)
    radii = particle_generation.lognormal_pp_radii(sigma, RP_G, n, rng=rng)
    cls = MassGammaPCAggregator if use_mass else PCAggregator
    runner = cls(
        radii,
        df,
        kf,
        TOL_OV,
        rng=rng,
        algorithm_config=OrchestratorAlgorithmConfig(),
    )
    result = runner.run()
    return result is not None and not runner.not_able_pca


def run_cell(
    df: float,
    kf: float,
    sigma: float,
    seeds: Sequence[int],
    n: int = N_PARTICLES,
    use_mass: bool = False,
) -> dict:
    """Run every seed of one grid cell and summarise it.

    Returns
    -------
    dict
        Cell record with the parameters, per-seed outcomes, success count,
        rate, Wilson 95% interval and elapsed wall time.
    """
    t0 = time.perf_counter()
    outcomes = [int(run_trial(df, kf, sigma, s, n=n, use_mass=use_mass)) for s in seeds]
    elapsed = time.perf_counter() - t0
    successes = int(sum(outcomes))
    lo, hi = wilson_ci(successes, len(seeds))
    return {
        "df": df,
        "kf": kf,
        "sigma": sigma,
        "n": n,
        "use_mass": use_mass,
        "trials": len(seeds),
        "successes": successes,
        "success_rate": successes / len(seeds),
        "ci95_low": lo,
        "ci95_high": hi,
        "elapsed_s": elapsed,
        "outcomes_by_seed": outcomes,
    }


def git_commit() -> str | None:
    """Return the current HEAD hash, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def environment(seeds: Sequence[int], n: int) -> dict:
    """Provenance block written into every result file."""
    return {
        "git_commit": git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_particles": n,
        "rp_g": RP_G,
        "tol_ov": TOL_OV,
        "seed_base": SEED_BASE,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
    }


def print_table(cells: Iterable[dict], kfs: Sequence[float]) -> None:
    """Print the probe grid as rate and Wilson interval, one row per (Df, sigma)."""
    by_key = {(c["df"], c["sigma"], c["kf"]): c for c in cells}
    rows = sorted({(c["df"], c["sigma"]) for c in cells})
    head = "Df    sigma  " + "".join(f"kf={k:<15.1f}" for k in kfs)
    print(head)
    print("-" * len(head))
    for df, sigma in rows:
        line = f"{df:<5.1f} {sigma:<6.1f} "
        for kf in kfs:
            c = by_key[(df, sigma, kf)]
            line += (
                f"{c['success_rate']:.3f}"
                f" [{c['ci95_low']:.2f},{c['ci95_high']:.2f}]".ljust(18)
            )
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"trials per cell (default {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--n", type=int, default=N_PARTICLES, help="particles per sub-cluster"
    )
    parser.add_argument(
        "--use-mass",
        action="store_true",
        help="solve the mass form of Gamma (experiment only; see pc_gamma_form.py)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output JSON path")
    args = parser.parse_args(argv)

    # A failing construction is the measurement here, not an incident:
    # silence the aggregator's per-failure ERROR records so the grid's own
    # output is readable. Logging has no effect on the RNG stream.
    logging.disable(logging.CRITICAL)

    seeds = [SEED_BASE + i for i in range(args.seeds)]
    out_path = args.out or (
        RESULTS_DIR
        / f"pc_probe_{args.seeds}seeds{'_mass' if args.use_mass else ''}.json"
    )

    t0 = time.perf_counter()
    cells = []
    for df in DFS:
        for sigma in SIGMAS:
            for kf in KFS:
                cell = run_cell(df, kf, sigma, seeds, n=args.n, use_mass=args.use_mass)
                cells.append(cell)
                print(
                    f"  Df={df} sigma={sigma} kf={kf}: "
                    f"{cell['successes']}/{cell['trials']} "
                    f"({cell['success_rate']:.3f})  [{cell['elapsed_s']:.1f} s]",
                    flush=True,
                )
    total = time.perf_counter() - t0

    payload = {
        "experiment": "pc_probe",
        "description": (
            "Single-shot PCA sub-cluster construction success at n=12 over "
            "Df x sigma x kf (paper table tab:pcprobe)."
        ),
        "environment": environment(seeds, args.n),
        "grid": {"df": list(DFS), "sigma": list(SIGMAS), "kf": list(KFS)},
        "use_mass_gamma": args.use_mass,
        "total_elapsed_s": total,
        "cells": cells,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    print_table(cells, KFS)
    print(f"\ntotal wall time: {total:.1f} s")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
