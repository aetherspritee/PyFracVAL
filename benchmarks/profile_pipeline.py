#!/usr/bin/env python3
"""Profile a full generation run and report where the time actually goes.

Deliberately profiles ``run_simulation`` end to end rather than a
micro-benchmark of one kernel, because the interesting costs in this
pipeline are distributional: which *stage* dominates depends on N, on how
often sticking fails, and on how many retries a regime forces. A kernel
benchmark cannot see any of that.

Reports cumulative time by function, and separately rolls the results up
into the pipeline's own stages (PCA, CCA sticking, overlap checks,
geometry, quality) so the output answers "what should I optimize" rather
than only "what is hot".

Usage:
    devenv shell -- uv run python benchmarks/profile_pipeline.py
    devenv shell -- uv run python benchmarks/profile_pipeline.py --n 1024 --regime hard
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import os
import pstats
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(level=logging.CRITICAL)

REGIMES = {
    "easy": dict(Df=1.8, kf=1.0, rp_gstd=1.5),
    "medium": dict(Df=2.1, kf=1.0, rp_gstd=1.5),
    "hard": dict(Df=2.3, kf=1.0, rp_gstd=1.9),
}

# Roll individual functions up into the stage a reader thinks in terms of.
STAGE_PATTERNS = [
    ("overlap checks", ("overlap.py", "_overlap", "overlap_")),
    ("CCA sticking", ("cca/sticking.py", "cca_kernels.py")),
    ("CCA pairing/candidates", ("cca/pairing.py", "cca/candidates.py", "matching.py")),
    ("CCA orchestration", ("cca/aggregator.py", "cca/fallbacks.py")),
    ("PCA", ("pca_agg.py", "pca_kernels.py", "pca_subclusters.py")),
    ("geometry", ("geometry.py",)),
    ("fractal/Rg", ("fractal.py",)),
    ("quality record", ("quality.py",)),
    ("particle generation", ("particle_generation.py",)),
    ("output/IO", ("schemas.py",)),
]


def classify(func_desc: str) -> str:
    for stage, patterns in STAGE_PATTERNS:
        if any(p in func_desc for p in patterns):
            return stage
    return "other"


def run_once(n: int, regime: str, seed: int, out_dir: Path):
    from pyfracval.main_runner import run_simulation

    params = REGIMES[regime]
    return run_simulation(
        iteration=1,
        sim_config_dict={
            "N": n,
            "Df": params["Df"],
            "kf": params["kf"],
            "rp_g": 100.0,
            "rp_gstd": params["rp_gstd"],
            "tol_ov": 1e-6,
            "n_subcl_percentage": 0.1,
            "ext_case": 0,
            "seed": seed,
        },
        output_base_dir=str(out_dir),
        max_runtime_seconds=300.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--regime", choices=list(REGIMES), default="medium")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path("/tmp/pyfracval_profile")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm up numba so JIT compilation is not charged to the profile.
    print("warming up JIT...")
    run_once(64, "easy", 1, out_dir)

    print(f"profiling N={args.n} regime={args.regime} seed={args.seed} ...")
    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    ok, coords, _ = run_once(args.n, args.regime, args.seed, out_dir)
    prof.disable()
    elapsed = time.time() - t0
    print(f"  success={ok}  wall={elapsed:.2f}s")

    stats = pstats.Stats(prof)
    total = stats.total_tt

    # --- Stage rollup ---
    stage_time: dict[str, float] = {}
    for func, (_cc, _nc, tt, _ct, _callers) in stats.stats.items():
        desc = f"{func[0]}:{func[1]}({func[2]})"
        stage_time[classify(desc)] = stage_time.get(classify(desc), 0.0) + tt

    print(
        f"\n{'=' * 70}\nTime by pipeline stage (tottime, total {total:.2f}s)\n{'=' * 70}"
    )
    for stage, tt in sorted(stage_time.items(), key=lambda kv: -kv[1]):
        if tt <= 0.0:
            continue
        print(f"  {stage:26s} {tt:7.3f}s  {100.0 * tt / total:5.1f}%")

    # --- Individual hotspots ---
    print(f"\n{'=' * 70}\nTop functions by cumulative time\n{'=' * 70}")
    buf = io.StringIO()
    stats.stream = buf
    stats.sort_stats("cumulative").print_stats(args.top)
    for line in buf.getvalue().splitlines():
        if "pyfracval" in line or "numpy" in line or "scipy" in line:
            print(line.rstrip())

    print(f"\n{'=' * 70}\nTop functions by self time\n{'=' * 70}")
    buf2 = io.StringIO()
    stats.stream = buf2
    stats.sort_stats("tottime").print_stats(args.top)
    for line in buf2.getvalue().splitlines():
        if "pyfracval" in line or "numpy" in line or "scipy" in line:
            print(line.rstrip())


if __name__ == "__main__":
    main()
