"""Per-merge event logging for CCA aggregation.

Every CCA merge attempt - success or failure - can be recorded as one
JSONL record, turning ordinary production runs into sweep data. This
exists because the pre-existing telemetry (``profile_*`` counters,
:mod:`pyfracval.overlap_statistics`'s census) only reports *aggregate*
totals at the end of a run or a one-off snapshot at the give-up point;
neither answers "which merges were hard, and how did they fail" across a
population of runs.

Opt-in via ``OrchestratorAlgorithmConfig.cca_merge_log_path``; nothing is
written and no file is opened when that is unset.

Concurrency: records are written as single ``write()`` calls of one line
each, in append mode. That is atomic enough on POSIX for the parallel
(Dask/multiprocessing) case that several workers may share one path, as
long as lines stay under the platform pipe/file atomicity limit - they
are a few hundred bytes here. Each record additionally carries a
``run_id`` and ``pid`` so interleaved records can always be separated
again downstream.
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
        Which CCA round (1-based) this attempt happened in. Round 1 is the
        merge of the initial PCA subclusters - the round essentially every
        hard-regime failure occurs in (docs/source/pairing_frustration.md).
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
        Size of the candidate matrix (how many pairs passed the Eq. 10
        shell test) - the search space the attempt had available.
    rotations_used : int
        Rotation steps consumed on the final candidate attempted.
    min_overlap : float
        Best (smallest) max-overlap achieved across the attempt. Compare
        against ``tol_ov`` to see how close a failure came.
    n_offending_particles : int | None
        From the overlap census, when enabled: how many distinct particles
        were involved in a residual overlap at give-up.
    n_particles_dropped : int
        Particles removed by drop-rescue, when that fallback succeeded.
    attempt_index : int
        0-based index of this partner attempt for ``cluster_idx1`` within
        the round - non-zero values only occur under backtracking pairing
        and are what distinguishes "first choice worked" from "third
        choice worked".
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
    n_particles_dropped: int = 0
    attempt_index: int = 0
    extra: dict = field(default_factory=dict)


class MergeEventLog:
    """Append-only JSONL sink for :class:`MergeEvent` records."""

    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.pid = os.getpid()
        self._failed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # A broken log path must never take down a simulation run.
            logger.warning(f"Merge log disabled - cannot create {self.path}: {exc}")
            self._failed = True

    def record(self, event: MergeEvent) -> None:
        """Append one event. Never raises: logging is diagnostic, and a
        full disk or a bad path must not abort an aggregate mid-build."""
        if self._failed:
            return
        payload = asdict(event)
        payload["run_id"] = self.run_id
        payload["pid"] = self.pid
        # inf/-inf are not valid JSON; JSON's null is the honest encoding
        # of "no overlap value was ever recorded for this attempt".
        min_ov = payload.get("min_overlap")
        if min_ov is not None and (min_ov == float("inf") or min_ov != min_ov):
            payload["min_overlap"] = None
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Merge log write failed, disabling: {exc}")
            self._failed = True
