"""Dask client helpers for distributed aggregate generation."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_client(
    scheduler_address: str | None = None,
    n_workers: int | None = None,
):
    """Return a Dask distributed Client.

    If *scheduler_address* is given, connect to a running scheduler at that
    address (e.g. ``"tcp://host:8786"``).  Otherwise start a local
    ``LocalCluster`` with *n_workers* workers (defaults to the number of
    CPU cores when *n_workers* is ``None``).

    Parameters
    ----------
    scheduler_address:
        Address of a remote Dask scheduler.  ``None`` → use a local cluster.
    n_workers:
        Number of workers for a local cluster.  Ignored when connecting to a
        remote scheduler.

    Returns
    -------
    dask.distributed.Client
    """
    from dask.distributed import Client, LocalCluster  # lazy import

    if scheduler_address is not None:
        logger.info(f"Connecting to remote Dask scheduler at {scheduler_address}")
        return Client(scheduler_address)

    logger.info(f"Starting local Dask cluster with n_workers={n_workers!r}")
    cluster = LocalCluster(n_workers=n_workers)
    return Client(cluster)
