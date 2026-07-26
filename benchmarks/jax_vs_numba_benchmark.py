#!/usr/bin/env python3
"""Head-to-head timing: numba (current production kernels) vs JAX-CPU vs
JAX-GPU, for the two numba hotspots identified as JAX/GPU port candidates
(see docs/source/gpu_acceleration.md).

Usage:
    devenv shell -- uv run --group jax_bench python benchmarks/jax_vs_numba_benchmark.py

Requires the jax_bench dependency group and devenv.nix's CUDA env vars
(CUDA_HOME, LD_LIBRARY_PATH, NUMBA_CUDA_DRIVER) to see the GPU at all -
falls back to CPU-only comparison if no GPU backend is available. Use
``uv run --group jax_bench`` (as above), not ``uv sync --group jax_bench``
- the latter *replaces* the synced group set rather than extending it, and
will silently uninstall pytest/sphinx/etc. from the project venv until the
next ``devenv shell`` resync.

Benchmark methodology
----------------------
Each kernel/framework/size combination is warmed up once (outside the timed
loop) to pay JIT-compilation cost before timing - this reflects a *warm*
process, i.e. what caching (numba's on-disk cache=True, JAX's persistent
compilation cache, both wired up via devenv.nix) buys you across repeated
runs. Cold-start (first-ever compile) cost is reported separately.

Per-call timing includes host<->device transfer for JAX (arrays are fresh
numpy each call, mirroring how these kernels are actually invoked inside
the simulation loop - coordinates/radii differ every rotation attempt, so
keeping everything GPU-resident across the whole run isn't the real usage
pattern for these particular kernels).
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark_results"
N_TIMED_CALLS = 50
SEED = 20260726


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _random_cluster(n: int, rng: np.random.Generator, scale: float = 1.0):
    coords = rng.standard_normal((n, 3)) * scale * (n ** (1 / 3))
    radii = rng.uniform(0.8, 1.2, size=n)
    return coords.astype(np.float64), radii.astype(np.float64)


def _time_calls(fn, arg_batches, block_fn=None) -> dict:
    """Run fn once (warmup, untimed) then N_TIMED_CALLS times (timed).

    arg_batches: list of length 1 + N_TIMED_CALLS of arg tuples (index 0 is
    the warmup call).
    """
    t0 = time.perf_counter()
    result = fn(*arg_batches[0])
    if block_fn is not None:
        block_fn(result)
    cold_start_s = time.perf_counter() - t0

    timings = []
    for args in arg_batches[1:]:
        t0 = time.perf_counter()
        result = fn(*args)
        if block_fn is not None:
            block_fn(result)
        timings.append(time.perf_counter() - t0)

    return {
        "cold_start_us": cold_start_s * 1e6,
        "median_us": statistics.median(timings) * 1e6,
        "mean_us": statistics.mean(timings) * 1e6,
        "min_us": min(timings) * 1e6,
        "max_us": max(timings) * 1e6,
    }


def _jax_block(x):
    x.block_until_ready()


# ---------------------------------------------------------------------------
# Kernel benchmarks
# ---------------------------------------------------------------------------


def bench_rodrigues_rotation(sizes: list[int]) -> list[dict]:
    import jax_kernels

    from pyfracval.geometry import rodrigues_rotation

    rows = []
    for n in sizes:
        rng = _rng()
        batches_np = []
        for _ in range(1 + N_TIMED_CALLS):
            coords, _ = _random_cluster(n, rng)
            axis = rng.standard_normal(3)
            angle = float(rng.uniform(0, 2 * np.pi))
            batches_np.append((coords, axis, angle))

        row = {"kernel": "rodrigues_rotation", "n": n}

        numba_stats = _time_calls(
            lambda c, a, ang: rodrigues_rotation(c.copy(), a.copy(), ang),
            batches_np,
        )
        row["numba"] = numba_stats

        import jax

        cpu = jax.devices("cpu")[0]
        batches_cpu = [
            (jax.device_put(c, cpu), jax.device_put(a, cpu), jax.device_put(ang, cpu))
            for c, a, ang in batches_np
        ]
        row["jax_cpu"] = _time_calls(
            jax_kernels.rodrigues_rotation_jax, batches_cpu, block_fn=_jax_block
        )

        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if gpu_devices:
            gpu = gpu_devices[0]
            batches_gpu = [
                (
                    jax.device_put(c, gpu),
                    jax.device_put(a, gpu),
                    jax.device_put(ang, gpu),
                )
                for c, a, ang in batches_np
            ]
            row["jax_gpu"] = _time_calls(
                jax_kernels.rodrigues_rotation_jax, batches_gpu, block_fn=_jax_block
            )
        else:
            row["jax_gpu"] = None

        rows.append(row)
        print(f"  rodrigues_rotation n={n}: done")
    return rows


def bench_pairwise_overlap(sizes: list[int]) -> list[dict]:
    import jax_kernels

    from pyfracval.overlap import calculate_max_overlap_cca_fast

    rows = []
    for n in sizes:
        rng = _rng()
        batches_np = []
        for _ in range(1 + N_TIMED_CALLS):
            coords1, radii1 = _random_cluster(n, rng, scale=1.5)
            coords2, radii2 = _random_cluster(n, rng, scale=1.5)
            batches_np.append((coords1, radii1, coords2, radii2))

        row = {"kernel": "pairwise_overlap", "n": n}

        row["numba"] = _time_calls(calculate_max_overlap_cca_fast, batches_np)

        import jax

        cpu = jax.devices("cpu")[0]
        batches_cpu = [tuple(jax.device_put(x, cpu) for x in b) for b in batches_np]
        row["jax_cpu"] = _time_calls(
            jax_kernels.max_overlap_pairwise_jax, batches_cpu, block_fn=_jax_block
        )

        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if gpu_devices:
            gpu = gpu_devices[0]
            batches_gpu = [tuple(jax.device_put(x, gpu) for x in b) for b in batches_np]
            row["jax_gpu"] = _time_calls(
                jax_kernels.max_overlap_pairwise_jax, batches_gpu, block_fn=_jax_block
            )
        else:
            row["jax_gpu"] = None

        rows.append(row)
        print(f"  pairwise_overlap n={n}: done")
    return rows


def bench_batch_overlap_pca(
    n_agg_sizes: list[int], batch_sizes: list[int]
) -> list[dict]:
    import jax_kernels

    from pyfracval.pca_kernels import batch_check_overlaps_pca

    rows = []
    for n_agg in n_agg_sizes:
        for k in batch_sizes:
            rng = _rng()
            batches_np = []
            for _ in range(1 + N_TIMED_CALLS):
                coords_agg, radii_agg = _random_cluster(n_agg, rng, scale=1.5)
                candidates = rng.standard_normal((k, 3)) * (n_agg ** (1 / 3))
                radius_new = float(rng.uniform(0.8, 1.2))
                batches_np.append(
                    (
                        coords_agg,
                        radii_agg,
                        candidates.astype(np.float64),
                        radius_new,
                        1e-6,
                    )
                )

            row = {"kernel": "batch_overlap_pca", "n_agg": n_agg, "batch_k": k}

            row["numba"] = _time_calls(batch_check_overlaps_pca, batches_np)

            import jax

            def to_jax_args(b, device):
                coords_agg, radii_agg, candidates, radius_new, _tol = b
                return (
                    jax.device_put(coords_agg, device),
                    jax.device_put(radii_agg, device),
                    jax.device_put(candidates, device),
                    jax.device_put(radius_new, device),
                )

            cpu = jax.devices("cpu")[0]
            batches_cpu = [to_jax_args(b, cpu) for b in batches_np]
            row["jax_cpu"] = _time_calls(
                jax_kernels.batch_check_overlaps_pca_jax,
                batches_cpu,
                block_fn=_jax_block,
            )

            gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
            if gpu_devices:
                gpu = gpu_devices[0]
                batches_gpu = [to_jax_args(b, gpu) for b in batches_np]
                row["jax_gpu"] = _time_calls(
                    jax_kernels.batch_check_overlaps_pca_jax,
                    batches_gpu,
                    block_fn=_jax_block,
                )
            else:
                row["jax_gpu"] = None

            rows.append(row)
            print(f"  batch_overlap_pca n_agg={n_agg} k={k}: done")
    return rows


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 80)
    print("JAX vs numba kernel benchmark")
    print("=" * 80)

    import jax

    print(f"jax version: {jax.__version__}")
    print(f"jax devices: {jax.devices()}")
    print(f"numba threads: {os.environ.get('NUMBA_NUM_THREADS', 'default')}")
    print("=" * 80)

    results = {
        "meta": {
            "jax_version": jax.__version__,
            "jax_devices": [str(d) for d in jax.devices()],
            "n_timed_calls": N_TIMED_CALLS,
            "seed": SEED,
        },
        "rodrigues_rotation": bench_rodrigues_rotation([16, 64, 256, 1024, 4096, 8192]),
        "pairwise_overlap": bench_pairwise_overlap([16, 64, 256, 1024, 4096]),
        "batch_overlap_pca": bench_batch_overlap_pca(
            n_agg_sizes=[64, 512, 2048], batch_sizes=[1, 30, 90, 360]
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "jax_vs_numba_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
