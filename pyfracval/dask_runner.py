"""Dask client helpers for distributed aggregate generation."""

from __future__ import annotations

import logging
import os
import subprocess
import tomllib
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


def _project_version() -> str:
    """Read project version from pyproject.toml."""
    pyproject = _PROJECT_ROOT / "pyproject.toml"
    with open(pyproject, "rb") as fh:
        data = tomllib.load(fh)
    return str(data["project"]["version"])


def install_wheel_on_workers(
    client,
    wheel_path: str | Path,
    package_name: str,
    expected_version: str,
) -> None:
    """Install an arbitrary wheel on a Dask scheduler and all its workers,
    verified via a runtime version fingerprint.

    Generic version of the mechanism this module has always used for
    installing *pyfracval* itself onto Dask Docker workers that don't have
    it preinstalled (e.g. a generic ``ghcr.io/dask/dask`` image) -- reusable
    for any package, not just this one. In particular: a *compiled*
    extension (e.g. pyfastmm's f2py extension) needs a wheel actually built
    for the worker's platform/Python ABI -- this function only ships and
    installs whatever wheel you hand it, it does not build one (see
    ``_register_package`` below for pyfracval's own "build one first" case).

    ``client.run()`` sends a callable directly to each worker and executes it
    there, bypassing the plugin/scheduler machinery. The installer function
    is defined inline so cloudpickle serialises it **by value** (bytecode),
    not by reference to a module -- which would fail on the scheduler/workers
    before *package_name* is installed there.

    Parameters
    ----------
    client:
        A connected ``dask.distributed.Client``.
    wheel_path:
        Path to a ``.whl`` file, already built for the *worker's* platform
        and Python version (this function does no cross-compilation or
        compatibility checking -- get that part right before calling this).
    package_name:
        The distribution/import name (e.g. ``"pyfracval"``, ``"pyfastmm"``,
        ``"spcwth"``) -- used for the post-install version check, the
        stale-module cache eviction, and the ``<PACKAGE>_INSTALLED_WHEEL``/
        ``<PACKAGE>_EXPECTED_VERSION`` env vars set on each worker.
    expected_version:
        Version string the installed wheel must report after installing,
        or this raises ``RuntimeError``.
    """
    wheel_path = Path(wheel_path)
    wheel_bytes = wheel_path.read_bytes()
    wheel_filename = wheel_path.name
    env_installed_wheel = f"{package_name.upper()}_INSTALLED_WHEEL"
    env_expected_version = f"{package_name.upper()}_EXPECTED_VERSION"
    logger.info(
        f"Installing {wheel_filename} ({len(wheel_bytes) // 1024} KB) "
        f"on scheduler and all workers…"
    )

    def _install_wheel_bytes_embedded(
        wheel_bytes: bytes,
        wheel_filename: str,
        package_name: str,
        expected_version: str,
        env_installed_wheel: str,
        env_expected_version: str,
    ) -> dict[str, str]:
        import importlib
        import os
        import subprocess
        import sys
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix=f"{package_name}_wheel_")
        wheel_path = os.path.join(tmp_dir, wheel_filename)
        with open(wheel_path, "wb") as fh:
            fh.write(wheel_bytes)

        install_cmds = [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--force-reinstall",
                wheel_path,
            ],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                wheel_path,
            ],
        ]
        last_exc: Exception | None = None
        for cmd in install_cmds:
            try:
                subprocess.check_call(cmd)
                last_exc = None
                break
            except Exception as exc:  # pragma: no cover - env-specific
                last_exc = exc
        if last_exc is not None:
            raise RuntimeError(
                "Failed to install wheel on scheduler/worker using pip and uv pip"
            ) from last_exc

        importlib.invalidate_caches()
        for mod_name in list(sys.modules):
            if mod_name == package_name or mod_name.startswith(package_name + "."):
                sys.modules.pop(mod_name, None)

        os.environ[env_installed_wheel] = wheel_filename
        os.environ[env_expected_version] = expected_version

        return {
            "python": sys.executable,
            "pid": str(os.getpid()),
            "installed_wheel": wheel_filename,
            "expected_version": expected_version,
        }

    def _worker_fingerprint_embedded(package_name: str) -> dict[str, str]:
        import importlib
        import os
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as pkg_version

        try:
            module = importlib.import_module(package_name)
        except ImportError:
            module = None

        try:
            runtime_version = pkg_version(package_name)
        except PackageNotFoundError:
            runtime_version = "unknown"

        return {
            "version": str(runtime_version),
            "module_file": str(getattr(module, "__file__", "unknown")),
            "pid": str(os.getpid()),
        }

    # Install on the scheduler first using embedded functions in this scope
    # (cloudpickle serialises them by value).
    sched_msg = client.run_on_scheduler(
        _install_wheel_bytes_embedded,
        wheel_bytes,
        wheel_filename,
        package_name,
        expected_version,
        env_installed_wheel,
        env_expected_version,
    )
    logger.info(f"  scheduler: {sched_msg}")
    # Install on all workers.
    worker_addresses = list(client.scheduler_info()["workers"].keys())
    results = client.run(
        _install_wheel_bytes_embedded,
        wheel_bytes,
        wheel_filename,
        package_name,
        expected_version,
        env_installed_wheel,
        env_expected_version,
        workers=worker_addresses,
    )
    for worker_addr, msg in results.items():
        logger.info(f"  {worker_addr}: {msg}")

    fingerprints = client.run(
        _worker_fingerprint_embedded, package_name, workers=worker_addresses
    )
    for worker_addr, fp in fingerprints.items():
        logger.info(
            "  %s version=%s file=%s pid=%s",
            worker_addr,
            fp.get("version"),
            fp.get("module_file"),
            fp.get("pid"),
        )
        if fp.get("version") != expected_version:
            raise RuntimeError(
                f"Worker {worker_addr} version mismatch: "
                f"expected {expected_version}, got {fp.get('version')}"
            )

    logger.info(
        f"Scheduler and all workers have {package_name} installed and "
        "verified at runtime."
    )


def _register_package(client) -> None:
    """Build pyfracval's own wheel and install it on all workers.

    Thin wrapper around :func:`install_wheel_on_workers` -- kept for
    backwards compatibility and as pyfracval's own "build from local source
    first" use case; other packages (e.g. a compiled extension like
    pyfastmm, which needs a wheel built for the *worker's* platform, not
    whatever this machine happens to be) call
    :func:`install_wheel_on_workers` directly with an already-built wheel
    instead of going through this function.
    """
    wheel_path = _build_wheel()
    expected_version = _project_version()
    install_wheel_on_workers(client, wheel_path, "pyfracval", expected_version)


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

    if n_workers is None:
        # LocalCluster(n_workers=None) does NOT simply use all CPU cores -
        # Dask's own default heuristic also factors in currently-available
        # system memory and can pick far fewer workers than cores on a
        # machine with other things running (observed: 4 workers on a
        # 16-core/64GB desktop with a few GB already committed to unrelated
        # apps). Resolve explicitly to the actual core count so a
        # compute-bound batch workload (a parameter sweep) gets what the
        # docstring above promises, rather than a memory-conservative guess.
        n_workers = os.cpu_count() or 1

    logger.info(f"Starting local Dask cluster with n_workers={n_workers!r}")
    cluster = LocalCluster(n_workers=n_workers)
    return Client(cluster)
