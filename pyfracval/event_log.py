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

Compression
-----------
Give the path a ``.gz`` suffix and records are gzipped as they are
written. This matters at sweep scale: a merge record is ~640 bytes of
which ~180 is the run context re-serialized verbatim on every line, and a
boundary sweep emits on the order of a million of them. Measured on a
real sweep log, gzip level 6 gives ~9.5x - the difference between
shipping 330 MB and 35 MB off a cluster - and costs nothing in fidelity.

Concurrency
-----------
Uncompressed, records are written as single ``write()`` calls of one line
each in append mode, which is atomic enough on POSIX for several Dask
workers to share one path.

Compressed, that trick is unavailable: a gzip stream has to stay open
across records (re-opening per record would emit a one-line gzip member
each time, which is *larger* than the plain text it replaces), and a
buffered stream cannot be shared by concurrent writers. So a compressed
log writes one shard per process - ``events.jsonl.gz`` becomes
``events.pid1234.jsonl.gz`` - and the analyzer takes many files. Within a
process every ``EventLog`` on the same path shares one writer, so the
thousands of per-trial instances a sweep creates still produce a single
well-compressed stream.

Forks are handled explicitly, because ``pca_subclusters`` runs its
subclusters through a forked ``Pool``: streams are finalised before a
fork so no child inherits a live buffer, and a child that writes gets
its own shard.

Every record carries ``run_id`` and ``pid`` either way, so records can
always be separated again.
"""

import atexit
import gzip
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Level 6 is the measured knee: ~9.5x on real merge records, where level
#: 9 buys under 3% more for several times the CPU.
GZIP_LEVEL = 6

#: Records between explicit flushes of a compressed stream. Each flush is
#: a sync point that costs a little ratio, so it is not done per record;
#: this bounds how much a killed run can lose to the write buffer.
FLUSH_EVERY = 500


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


class _Writer:
    """Process-wide byte sink for one configured log path.

    Shared by every :class:`EventLog` targeting that path, because a
    sweep builds one ``EventLog`` per trial and a compressed stream must
    be opened exactly once per file.
    """

    def __init__(self, nominal: Path, compressed: bool):
        #: The path as configured, before per-process sharding.
        self.nominal = nominal
        self.compressed = compressed
        self.failed = False
        self.path = nominal
        self._handle = None
        self._since_flush = 0
        self._owner_pid = -1
        self._retarget()

    def _retarget(self) -> None:
        """Bind this writer to the current process.

        A forked child inherits ``_WRITERS`` wholesale, so it would
        otherwise append to the shard its parent is still writing. The
        first write from a new pid re-resolves the shard instead.
        """
        self._handle = None
        self._since_flush = 0
        self._owner_pid = os.getpid()
        self.path = (
            _shard_path(self.nominal, self._owner_pid)
            if self.compressed
            else self.nominal
        )

    def write(self, line: str) -> None:
        if self.failed:
            return
        if os.getpid() != self._owner_pid:
            self._retarget()
        try:
            if self.compressed:
                if self._handle is None:
                    self._handle = gzip.open(
                        self.path, "at", encoding="utf-8", compresslevel=GZIP_LEVEL
                    )
                self._handle.write(line)
                self._since_flush += 1
                if self._since_flush >= FLUSH_EVERY:
                    self._handle.flush()
                    self._since_flush = 0
            else:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Event log write failed, disabling: {exc}")
            self.failed = True
            self.close()

    def close(self) -> None:
        # Only the process that opened the stream may finalise it. In a
        # forked child this is a no-op, which keeps the inherited atexit
        # hook from stamping a trailer into the parent's file.
        if self._handle is not None and os.getpid() == self._owner_pid:
            try:
                self._handle.close()
            except OSError:
                pass
        self._handle = None


#: Configured path -> writer. Keyed before sharding, so every EventLog
#: aimed at one log in one process resolves to a single stream.
_WRITERS: dict[Path, _Writer] = {}


def _get_writer(path: Path, compressed: bool) -> _Writer:
    writer = _WRITERS.get(path)
    if writer is None:
        writer = _Writer(path, compressed)
        _WRITERS[path] = writer
    return writer


def close_all() -> None:
    """Close every open compressed stream. Registered with ``atexit``.

    A gzip member is only valid once its trailer is written, so a log
    abandoned without this is truncated at the last flush.
    """
    for writer in _WRITERS.values():
        writer.close()


atexit.register(close_all)

# Finalise every stream *before* a fork, so no child can inherit a live
# gzip buffer. Abandoning the inherited object in the child is not
# enough: its garbage collector still flushes, and because parent and
# child share one file offset those bytes land in the parent's log as
# duplicated records. `pca_subclusters` forks a Pool per trial, so this
# is a real path, not a theoretical one.
#
# The cost is one gzip member boundary per fork. Decoders concatenate
# members transparently, and a trial's worth of records is far more than
# enough for the compressor to earn its dictionary back.
os.register_at_fork(before=close_all)


def _shard_path(path: Path, pid: int) -> Path:
    """Insert ``.pid<N>`` before the suffixes of a compressed log path.

    ``events.jsonl.gz`` -> ``events.pid1234.jsonl.gz``.
    """
    suffixes = "".join(path.suffixes[-2:])
    base = path.name[: -len(suffixes)] if suffixes else path.name
    return path.with_name(f"{base}.pid{pid}{suffixes}")


class EventLog:
    """Append-only JSONL sink for generation events.

    Parameters
    ----------
    path : str or Path
        Destination file. A ``.gz`` suffix turns on gzip compression and
        per-process sharding (see the module docstring).
    context : dict, optional
        Fields stamped onto every record written through this instance.
    run_id : str, optional
        Identifier shared by all records of one aggregate; generated when
        omitted.
    """

    def __init__(
        self,
        path: str | Path,
        context: dict | None = None,
        run_id: str | None = None,
    ):
        #: The path as configured, before per-process sharding.
        self.nominal_path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.pid = os.getpid()
        #: Stamped onto every record. Carrying the simulation parameters
        #: here is what makes a pooled sweep log sliceable by physics
        #: rather than an undifferentiated pile of attempts.
        self.context = dict(context or {})
        self._failed = False

        # One shard per process when compressed: a buffered gzip stream
        # cannot be shared across processes the way atomic line appends can.
        self.compressed = self.nominal_path.suffix == ".gz"

        try:
            self.nominal_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # A broken log path must never take down a simulation run.
            logger.warning(
                f"Event log disabled - cannot create {self.nominal_path}: {exc}"
            )
            self._failed = True

        self._writer = _get_writer(self.nominal_path, self.compressed)

    @property
    def path(self) -> Path:
        """The file actually being written - a per-process shard when
        compressed, the configured path otherwise."""
        return self._writer.path

    def set_context(self, **kwargs) -> None:
        """Merge additional fields into the per-record context."""
        self.context.update(kwargs)

    def record(self, event) -> None:
        """Append one event. Never raises: logging is diagnostic, and a
        full disk or bad path must not abort an aggregate mid-build."""
        if self._failed or self._writer.failed:
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
            line = json.dumps(payload) + "\n"
        except (TypeError, ValueError) as exc:
            logger.warning(f"Event log: unserializable record dropped: {exc}")
            return
        self._writer.write(line)

    def close(self) -> None:
        """Flush and close this path's stream.

        Optional - ``atexit`` handles the normal case. Call it when a
        long-lived process must finalise a log before reading it back.
        """
        self._writer.close()
