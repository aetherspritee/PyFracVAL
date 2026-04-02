"""Batch generation of multiple aggregates in parallel using Dask.

This module provides functionality to generate multiple fractal aggregates
in parallel via a Dask distributed scheduler — either a local cluster or a
remote one (e.g. ``tcp://host:8786``).

Each trial is submitted as an independent Dask task, so workers can be on
any machine that is part of the Dask cluster.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from .dask_runner import get_client
from .main_runner import run_simulation

logger = logging.getLogger(__name__)


def generate_aggregates_parallel(
    n_aggregates: int,
    config: dict[str, Any],
    output_base_dir: str = "RESULTS",
    seed_start: int = 1000,
    n_workers: int | None = None,
    show_progress: bool = True,
    scheduler_address: str | None = None,
) -> list[tuple[bool, np.ndarray | None, np.ndarray | None]]:
    """Generate multiple fractal aggregates in parallel via Dask.

    Parameters
    ----------
    n_aggregates:
        Number of aggregates to generate.
    config:
        Simulation configuration dictionary (N, Df, kf, rp_g, rp_gstd, …).
    output_base_dir:
        Base directory for output files (default: ``"RESULTS"``).
    seed_start:
        Starting random seed.  Aggregate *i* uses ``seed_start + i``.
    n_workers:
        Workers for a local cluster.  Ignored when *scheduler_address* is set.
    show_progress:
        Show a ``tqdm`` progress bar while futures complete.
    scheduler_address:
        Remote Dask scheduler address (e.g. ``"tcp://host:8786"``).
        ``None`` → start a ``LocalCluster``.

    Returns
    -------
    list[tuple[bool, np.ndarray | None, np.ndarray | None]]
        One ``(success, coords, radii)`` tuple per aggregate, in submission
        order.
    """
    try:
        from tqdm import tqdm

        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    from dask.distributed import as_completed  # local import to keep optional

    print("=" * 80)
    print("BATCH AGGREGATE GENERATION (DASK)")
    print("=" * 80)
    print(f"Generating {n_aggregates} aggregates")
    print(
        f"Configuration: N={config.get('N')}, Df={config.get('Df')}, kf={config.get('kf')}"
    )
    print(f"Seed range: {seed_start} to {seed_start + n_aggregates - 1}")
    print(f"Output directory: {output_base_dir}")
    if scheduler_address:
        print(f"Dask scheduler: {scheduler_address}")
    else:
        print(f"Dask cluster: local (n_workers={n_workers!r})")
    print("=" * 80)

    start_time = time.time()

    with get_client(
        scheduler_address=scheduler_address,
        n_workers=n_workers,
        install_package=scheduler_address is not None,
    ) as client:
        # Avoid nested process pools inside Dask workers.
        client.run(
            lambda: os.environ.__setitem__(
                "PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS", "1"
            )
        )

        # Deterministic seeds: abs(hash((N, Df, kf, sigma, trial_index))) % 2**31
        n_val = config.get("N", 0)
        df_val = config.get("Df", 0.0)
        kf_val = config.get("kf", 0.0)
        sigma_val = config.get("rp_gstd", 1.0)

        def _seed(i: int) -> int:
            return abs(hash((n_val, df_val, kf_val, sigma_val, seed_start + i))) % (
                2**31
            )

        futures = {
            client.submit(
                run_simulation,
                i,
                config,
                output_base_dir,
                _seed(i),
            ): i
            for i in range(n_aggregates)
        }

        results: list[tuple[bool, np.ndarray | None, np.ndarray | None]] = [
            (False, None, None) for _ in range(n_aggregates)
        ]

        completed_iter: Any = as_completed(futures)
        if show_progress and _has_tqdm:
            from tqdm import tqdm as _tqdm

            completed_iter = _tqdm(
                completed_iter, total=n_aggregates, desc="Aggregates"
            )

        for future in completed_iter:
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as exc:
                logger.error(f"Aggregate {i} raised an exception: {exc}")
                results[i] = (False, None, None)

    end_time = time.time()
    total_time = end_time - start_time
    successes = sum(1 for s, _, _ in results if s)
    failures = n_aggregates - successes

    print("=" * 80)
    print("BATCH GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds")
    print(
        f"Successful: {successes}/{n_aggregates} ({successes / n_aggregates * 100:.1f}%)"
    )
    print(f"Failed: {failures}/{n_aggregates} ({failures / n_aggregates * 100:.1f}%)")
    print(f"Average time per aggregate: {total_time / n_aggregates:.2f} seconds")
    print(f"Throughput: {n_aggregates / total_time:.2f} aggregates/second")
    print("=" * 80)

    return results


def generate_aggregates_sequential(
    n_aggregates: int,
    config: dict[str, Any],
    output_base_dir: str = "RESULTS",
    seed_start: int = 1000,
) -> list[tuple[bool, np.ndarray | None, np.ndarray | None]]:
    """Generate multiple aggregates sequentially (for comparison/debugging).

    Parameters
    ----------
    n_aggregates:
        Number of aggregates to generate.
    config:
        Simulation configuration dictionary.
    output_base_dir:
        Base directory for output files (default: ``"RESULTS"``).
    seed_start:
        Starting random seed.

    Returns
    -------
    list[tuple[bool, np.ndarray | None, np.ndarray | None]]
        One ``(success, coords, radii)`` tuple per aggregate.
    """
    print("=" * 80)
    print("BATCH AGGREGATE GENERATION (SEQUENTIAL)")
    print("=" * 80)
    print(f"Generating {n_aggregates} aggregates sequentially")
    print(
        f"Configuration: N={config.get('N')}, Df={config.get('Df')}, kf={config.get('kf')}"
    )
    print("=" * 80)

    results = []
    start_time = time.time()

    for i in range(n_aggregates):
        result = run_simulation(i, config, output_base_dir, seed_start + i)
        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == n_aggregates:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_aggregates - (i + 1)) / rate if rate > 0 else 0
            print(
                f"Progress: {i + 1}/{n_aggregates} aggregates "
                f"({(i + 1) / n_aggregates * 100:.1f}%) | "
                f"Rate: {rate:.1f} agg/s | "
                f"ETA: {eta:.0f}s"
            )

    end_time = time.time()
    total_time = end_time - start_time

    successes = sum(1 for success, _, _ in results if success)
    failures = n_aggregates - successes

    print("=" * 80)
    print("BATCH GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds")
    print(
        f"Successful: {successes}/{n_aggregates} ({successes / n_aggregates * 100:.1f}%)"
    )
    print(f"Failed: {failures}/{n_aggregates} ({failures / n_aggregates * 100:.1f}%)")
    print(f"Average time per aggregate: {total_time / n_aggregates:.2f} seconds")
    print(f"Throughput: {n_aggregates / total_time:.2f} aggregates/second")
    print("=" * 80)

    return results
