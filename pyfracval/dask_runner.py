"""Dask client helpers for distributed aggregate generation."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root: two levels up from this file (pyfracval/dask_runner.py → project root)
_PROJECT_ROOT = Path(__file__).parent.parent


def _build_wheel() -> Path:
    """Build a wheel for the local package and return its path."""
    logger.info("Building pyfracval wheel with 'uv build'…")
    result = subprocess.run(
        ["uv", "build", "--wheel"],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"'uv build' failed:\n{result.stdout}\n{result.stderr}")
    dist_dir = _PROJECT_ROOT / "dist"
    wheels = sorted(dist_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise FileNotFoundError(f"No wheel found in {dist_dir} after 'uv build'")
    wheel = wheels[-1]
    logger.info(f"Built wheel: {wheel.name} ({wheel.stat().st_size // 1024} KB)")
    return wheel


def _register_package(client) -> None:
    """Build a wheel and install it on all workers via client.run().

    ``client.run()`` sends a callable directly to each worker and executes it
    there, bypassing the plugin/scheduler machinery.

    The installer function is defined inline so cloudpickle serialises it
    **by value** (bytecode), not by reference to the pyfracval module — which
    would fail on the scheduler/workers before pyfracval is installed.
    """

    def _install(wheel_bytes: bytes, wheel_filename: str) -> str:
        import os
        import subprocess
        import sys
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        wheel_path = os.path.join(tmp_dir, wheel_filename)
        with open(wheel_path, "wb") as fh:
            fh.write(wheel_bytes)
        subprocess.check_call(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--reinstall",
                wheel_path,
            ]
        )
        return f"installed {wheel_filename} on worker (python {sys.version})"

    wheel_path = _build_wheel()
    wheel_bytes = wheel_path.read_bytes()
    wheel_filename = wheel_path.name
    logger.info(
        f"Installing {wheel_filename} ({len(wheel_bytes) // 1024} KB) "
        f"on scheduler and all workers…"
    )
    # Install on the scheduler first — it deserialises the task graph (including
    # pyfracval functions) before routing tasks to workers, so it needs the
    # package too.
    sched_msg = client.run_on_scheduler(_install, wheel_bytes, wheel_filename)
    logger.info(f"  scheduler: {sched_msg}")
    # Install on all workers.
    worker_addresses = list(client.scheduler_info()["workers"].keys())
    results = client.run(
        _install, wheel_bytes, wheel_filename, workers=worker_addresses
    )
    for worker_addr, msg in results.items():
        logger.info(f"  {worker_addr}: {msg}")
    logger.info("Scheduler and all workers have pyfracval installed.")


def get_client(
    scheduler_address: str | None = None,
    n_workers: int | None = None,
    install_package: bool = False,
):
    """Return a Dask distributed Client.

    If *scheduler_address* is given, connect to a running scheduler at that
    address (e.g. ``"tcp://host:8786"``).  Otherwise start a local
    ``LocalCluster`` with *n_workers* workers (defaults to the number of
    CPU cores when *n_workers* is ``None``).

    When *install_package* is ``True`` **and** a remote scheduler is used, the
    local ``pyfracval`` package is built into a wheel and installed on all
    workers via a ``WorkerPlugin`` before the client is returned.  This is
    required whenever the workers do not have ``pyfracval`` pre-installed
    (e.g. a generic Dask Docker image).

    Parameters
    ----------
    scheduler_address:
        Address of a remote Dask scheduler.  ``None`` → use a local cluster.
    n_workers:
        Number of workers for a local cluster.  Ignored when connecting to a
        remote scheduler.
    install_package:
        When ``True`` and using a remote scheduler, build + install
        ``pyfracval`` on all workers before returning.

    Returns
    -------
    dask.distributed.Client
    """
    from dask.distributed import Client, LocalCluster  # lazy import

    if scheduler_address is not None:
        logger.info(f"Connecting to remote Dask scheduler at {scheduler_address}")
        client = Client(scheduler_address)
        if install_package:
            _register_package(client)
        return client

    logger.info(f"Starting local Dask cluster with n_workers={n_workers!r}")
    cluster = LocalCluster(n_workers=n_workers)
    return Client(cluster)
