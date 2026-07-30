"""Structured, append-only event log for aggregate generation.

Existing telemetry answers "did this run succeed". A paper needs
something else: **where** generation fails, **why**, and **how badly** -
pooled over thousands of runs and sliceable by the physics
(``Df``, ``kf``, ``sigma_p,geo``, ``N``). Free-text log lines cannot be
aggregated, and the in-memory ``diagnostics`` dict ``run_simulation``
accepts is per-call and never persisted.

This module writes one JSONL file carrying three record kinds, each
stamped with the same run context so they can be sliced or joined
together:

``merge``
    One CCA merge attempt: which round, which cluster pair, the contact
    distance attempted, how much of the search was consumed, the
    outcome, and - when a census ran - how many particles ended up
    overlapping and by how much.
``pca_failure``
    A PCA subcluster that could not be built: at which particle index,
    with how many candidate partners available, after how many
    search/swap attempts. Without this, "where does it fail" cannot
    distinguish a PCA failure from a CCA one with any detail, even
    though both occur.
``run``
    One completed or abandoned aggregate: outcome, failure stage and
    reason, attempts consumed, wall time, and the final geometry's
    measured quality.

Enable with ``OrchestratorAlgorithmConfig.event_log_path``. Nothing is
written and no file is opened when it is unset.

Concurrency: records are written as single ``write()`` calls of one line
each in append mode, which is atomic enough on POSIX for several Dask
workers to share one path. Every record carries ``run_id`` and ``pid`` so
interleaved lines can always be separated again.
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MergeEvent:
    """One CCA merge attempt between two clusters.

    Attributes
    ----------
    round_index : int
        Which CCA round (1-based). Round 1 merges the initial PCA
        subclusters - the round essentially every hard-regime failure
        occurs in (docs/source/pairing_frustration.md).
    pool_size : int
        Number of clusters in the pool at the start of this round.
    cluster_idx1, cluster_idx2 : int
        Indices of the two clusters within the round's pool.
    n1, n2 : int
        Particle counts of the two clusters.
    gamma_pc : float
        Scaling-law contact distance the merge was attempted at.
    gamma_real : bool
        Whether the Gamma equation had a real solution at all.
    sum_rmax : float
        ``r_max1 + r_max2``; the cheap feasibility gate compares this
        against ``gamma_pc``.
    outcome : str
        One of ``stuck``, ``stuck_relaxed_tol``, ``failed_no_candidates``,
        ``failed_overlap``, ``rescued_soft_relaxation``, ``rescued_drop``,
        ``failed_gamma_not_real``, ``skipped_bv_filter``.
    candidates_tried : int
        Number of candidate (s1, s2) monomer pairs attempted.
    n_feasible_pairs : int
        Size of the candidate matrix - the search space available.
    rotations_used : int
        Rotation steps consumed on the final candidate attempted.
    min_overlap : float
        Best (smallest) max-overlap reached, normalized by ``r_i + r_j``
        so it is directly comparable to ``tol_ov``.
    n_offending_particles : int | None
        Distinct particles involved in a residual overlap at give-up.
    n_pairs_overlapping : int | None
        Overlapping pairs at give-up.
    max_overlap_of_rsum : float | None
        Worst residual overlap at give-up, normalized by ``r_i + r_j``
        and therefore comparable to ``tol_ov``.
    max_overlap_of_rmin : float | None
        The same overlap normalized by ``min(r_i, r_j)``. Recorded
        separately because the two denominators differ by a large factor
        for wide size distributions and must never be conflated.
    n_particles_dropped : int
        Particles removed by drop-rescue, when that fallback succeeded.
    attempt_index : int
        0-based index of this partner attempt for ``cluster_idx1`` within
        the round; non-zero only under backtracking pairing, and what
        distinguishes "first choice worked" from "third choice worked".
    """

    round_index: int
    pool_size: int
    cluster_idx1: int
    cluster_idx2: int
    n1: int
    n2: int
    gamma_pc: float
    gamma_real: bool
    sum_rmax: float
    outcome: str
    candidates_tried: int = 0
    n_feasible_pairs: int = 0
    rotations_used: int = 0
    min_overlap: float = float("inf")
    n_offending_particles: int | None = None
    n_pairs_overlapping: int | None = None
    max_overlap_of_rsum: float | None = None
    max_overlap_of_rmin: float | None = None
    n_particles_dropped: int = 0
    attempt_index: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class PcaFailureEvent:
    """A PCA subcluster that could not be completed.

    PCA failure is a distinct mechanism from CCA sticking failure: it
    happens while growing a *single* subcluster particle by particle, and
    the usual cause is that no already-placed particle sits at a workable
    distance for the next monomer's Gamma. Recording it separately is
    what lets a failure taxonomy attribute blame correctly instead of
    lumping everything under "the run failed".
    """

    subcluster_index: int
    subcluster_size: int
    particle_index: int
    reason: str
    search_attempts: int = 0
    n_candidates: int = 0
    gamma_real: bool = True
    gamma_pc: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class RunEvent:
    """One aggregate generation attempt, start to finish."""

    outcome: str
    failure_stage: str | None = None
    failure_reason: str | None = None
    attempts_used: int = 0
    elapsed_s: float = 0.0
    n_particles_actual: int = 0
    n_particles_dropped: int = 0
    max_residual_overlap: float | None = None
    n_overlapping_pairs: int | None = None
    overlap_ok: bool | None = None
    measured_rg: float | None = None
    rg_error_pct: float | None = None
    extra: dict = field(default_factory=dict)


_KINDS = {
    MergeEvent: "merge",
    PcaFailureEvent: "pca_failure",
    RunEvent: "run",
}


def _json_safe(value):
    """Coerce values JSON cannot represent, rather than losing records."""
    if isinstance(value, float):
        # inf/nan are not valid JSON; null is the honest encoding of
        # "never measured" for these fields.
        if value != value or value == float("inf") or value == float("-inf"):
            return None
    return value


class EventLog:
    """Append-only JSONL sink for generation events."""

    def __init__(
        self,
        path: str | Path,
        context: dict | None = None,
        run_id: str | None = None,
    ):
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.pid = os.getpid()
        #: Stamped onto every record. Carrying the simulation parameters
        #: here is what makes a pooled sweep log sliceable by physics
        #: rather than an undifferentiated pile of attempts.
        self.context = dict(context or {})
        self._failed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # A broken log path must never take down a simulation run.
            logger.warning(f"Event log disabled - cannot create {self.path}: {exc}")
            self._failed = True

    def set_context(self, **kwargs) -> None:
        """Merge additional fields into the per-record context."""
        self.context.update(kwargs)

    def record(self, event) -> None:
        """Append one event. Never raises: logging is diagnostic, and a
        full disk or bad path must not abort an aggregate mid-build."""
        if self._failed:
            return
        kind = _KINDS.get(type(event))
        if kind is None:
            logger.warning(f"Event log: unknown event type {type(event)!r}")
            return

        payload = {k: _json_safe(v) for k, v in asdict(event).items()}
        payload["kind"] = kind
        payload["run_id"] = self.run_id
        payload["pid"] = self.pid
        for key, value in self.context.items():
            # Context must not silently shadow a record's own fields.
            payload.setdefault(key, value)

        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Event log write failed, disabling: {exc}")
            self._failed = True
