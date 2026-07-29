"""Load and filter a ``cluster_index.csv`` master index of generated aggregates.

``cluster_index.csv`` is written by ``scripts/generate_cluster_data.py`` as a
flat table -- one row per generated ``.dat`` file, with the generation
parameters (sigma/Df/N/kf/...) that would otherwise only be recoverable by
parsing directory names or re-reading every file's YAML header via
``pyfracval.schemas.Metadata.from_file()``. This module is the native way to
query that table instead: no pandas dependency (only an optional/plot
dependency of this package, not a core one), just the stdlib ``csv`` module
and a small pydantic model (``pyfracval.schemas.ClusterEntry``).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from pyfracval.schemas import ClusterEntry


def load_catalog(
    index_path: str | Path,
    data_root: str | Path | None = None,
) -> list[ClusterEntry]:
    """Load every row of a ``cluster_index.csv`` as a list of ``ClusterEntry``.

    *index_path*'s ``filepath`` column holds an absolute path from wherever
    the data was originally generated -- not portable across machines or
    repos that keep their own copy of the same aggregate files. When
    *data_root* is given, each entry's ``filepath`` is re-based under it
    instead of trusting the stored absolute path: the last 3 path
    components (``<config>/<sigma_..__Df_..__N_..>/<filename>.dat``) are the
    part of the layout ``scripts/generate_cluster_data.py`` actually
    guarantees, so re-joining just those under *data_root* is reliable
    regardless of where the CSV says the file used to live. When
    *data_root* is ``None``, ``filepath`` is used as-is (verbatim from the
    CSV).
    """
    index_path = Path(index_path)
    data_root = Path(data_root) if data_root is not None else None

    entries: list[ClusterEntry] = []
    with open(index_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filepath = Path(row["filepath"])
            if data_root is not None:
                filepath = data_root.joinpath(*filepath.parts[-3:])
            entries.append(
                ClusterEntry(
                    config=row["config"],
                    sigma=float(row["sigma"]),
                    Df=float(row["Df"]),
                    N=int(row["N"]),
                    kf=float(row["kf"]),
                    attempt=int(row["attempt"]),
                    seed=int(row["seed"]),
                    success=row["success"].strip().lower() == "true",
                    filepath=filepath,
                )
            )
    return entries


def _matches(value: float | int | str, allowed: object) -> bool:
    if allowed is None:
        return True
    if isinstance(allowed, (list, tuple, set, frozenset)):
        return value in allowed
    return value == allowed


def filter_catalog(
    entries: Iterable[ClusterEntry],
    *,
    sigma: float | Iterable[float] | None = None,
    Df: float | Iterable[float] | None = None,
    N: int | Iterable[int] | None = None,
    kf: float | Iterable[float] | None = None,
    config: str | Iterable[str] | None = None,
    success_only: bool = True,
) -> list[ClusterEntry]:
    """Filter catalog entries by generation parameters.

    Each keyword accepts either a single value (exact match) or an iterable
    of allowed values; ``None`` (the default for all but *success_only*)
    means "don't filter on this field". *success_only* (default ``True``)
    drops any entry with ``success=False`` before applying the other
    filters -- a failed generation attempt has no usable geometry.
    """
    result = []
    for entry in entries:
        if success_only and not entry.success:
            continue
        if not _matches(entry.sigma, sigma):
            continue
        if not _matches(entry.Df, Df):
            continue
        if not _matches(entry.N, N):
            continue
        if not _matches(entry.kf, kf):
            continue
        if not _matches(entry.config, config):
            continue
        result.append(entry)
    return result
