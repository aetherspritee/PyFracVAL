"""Thin CCAggregator orchestrator composed from the CCA mixins.

Owns instance state (coords, radii, i_orden, telemetry counters) and the
top-level ``run_cca()`` entry point; delegates the actual pairing,
candidate selection, sticking, and fallback logic to the mixins in
:mod:`pyfracval.cca.pairing`, :mod:`pyfracval.cca.candidates`,
:mod:`pyfracval.cca.sticking`, and :mod:`pyfracval.cca.fallbacks`.
"""

import logging
import math
import time
from typing import Tuple

import numpy as np

from .. import fractal
from ..config import OrchestratorAlgorithmConfig
from .candidates import _CandidatesMixin
from .fallbacks import _FallbacksMixin
from .pairing import CCA_PAIRING_FACTOR, _PairingMixin
from .sticking import _StickingMixin

logger = logging.getLogger(__name__)


class CCAggregator(_PairingMixin, _CandidatesMixin, _StickingMixin, _FallbacksMixin):
    """Performs Cluster-Cluster Aggregation (CCA).

    Takes pre-generated subclusters (defined by coordinates, radii, and
    the `i_orden` index map) and iteratively aggregates them in pairs.
    The pairing and sticking process attempts to preserve the target
    fractal dimension (Df) and prefactor (kf) using the Gamma_pc method
    derived from :cite:p:`Moran2019FracVAL`. Includes overlap checking
    and rotation (`_cca_reintento`) during sticking.

    Parameters
    ----------
    initial_coords : np.ndarray
        Nx3 array containing coordinates of all particles from all subclusters.
    initial_radii : np.ndarray
        N array containing radii corresponding to `initial_coords`.
    initial_i_orden : np.ndarray
        Mx3 array [[start, end, count], ...] defining the subclusters within
        the initial coordinates and radii arrays.
    n_total : int
        Total number of primary particles (N).
    df : float
        Target fractal dimension for the final aggregate.
    kf : float
        Target fractal prefactor for the final aggregate.
    tol_ov : float
        Maximum allowable overlap fraction between particles during sticking.
    ext_case : int
        Flag (0 or 1) controlling the geometric criteria used in CCA
        candidate selection (`_cca_select_candidates`) and sticking
        (`_cca_sticking_v1`). See :cite:p:`Moran2019FracVAL` Appendix C.

    Attributes
    ----------
    N : int
        Total number of primary particles.
    df, kf, tol_ov, ext_case : float/int
        Stored simulation parameters.
    coords, radii : np.ndarray
        Current coordinates and radii, updated after each iteration.
    i_orden : np.ndarray
        Current cluster index map, updated after each iteration.
    i_t : int
        Current number of clusters remaining.
    not_able_cca : bool
        Flag indicating if the CCA process failed.
    """

    def __init__(
        self,
        initial_coords: np.ndarray,
        initial_radii: np.ndarray,
        initial_i_orden: np.ndarray,
        n_total: int,
        df: float,
        kf: float,
        tol_ov: float,
        ext_case: int,
        rng: np.random.Generator | None = None,
        algorithm_config: OrchestratorAlgorithmConfig | None = None,
        initial_densities: np.ndarray | None = None,
        deadline: float | None = None,
    ):
        if initial_coords.shape[0] != n_total or initial_radii.shape[0] != n_total:
            raise ValueError(
                f"Initial coords/radii length mismatch (Coords: {initial_coords.shape[0]}, Radii: {initial_radii.shape[0]}, Expected: {n_total})"
            )
        if initial_i_orden.ndim != 2 or initial_i_orden.shape[1] != 3:
            raise ValueError("initial_i_orden must be an Mx3 array")
        # Ensure i_orden covers all particles
        if initial_i_orden.shape[0] > 0 and (initial_i_orden[-1, 1] + 1) != n_total:
            logger.warning(
                f"initial_i_orden last index ({initial_i_orden[-1, 1]}) does not match N-1 ({n_total - 1}). Total particles in i_orden: {np.sum(initial_i_orden[:, 2])}"
            )
            # This could indicate an issue from PCA subclustering stage.

        self.N: int = n_total
        self.df = df
        self.kf = kf
        self.tol_ov = tol_ov
        self.ext_case = ext_case  # 0 or 1
        self.algorithm_config: OrchestratorAlgorithmConfig = (
            algorithm_config
            if algorithm_config is not None
            else OrchestratorAlgorithmConfig()
        )

        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        # Current state of the simulation
        self.coords = initial_coords.copy()
        self.radii = initial_radii.copy()
        # Optional per-particle densities, kept in lockstep with radii
        # through every reorder, merge and drop. None means uniform.
        _dens = fractal.resolve_densities(
            initial_densities, n_total, context="CCAggregator densities"
        )
        self.densities = _dens.copy() if _dens is not None else None
        self.i_orden = initial_i_orden.copy()  # Shape (i_t, 3) [start, end, count]
        self.i_t = self.i_orden.shape[0]  # Current number of clusters

        self.not_able_cca = False
        # Absolute time.time() after which the run gives up mid-flight.
        # Without this the only wall-clock check lives between whole
        # PCA+CCA attempts, so a single attempt is uninterruptible - and
        # backtracking made single attempts far more expensive in
        # infeasible regimes, since it tries several partners per cluster
        # before conceding instead of bailing on the first failure.
        self.deadline = deadline
        self.timed_out = False

        # Timing accumulators (used when self.algorithm_config.profile_timing is True)
        self._t_cluster_props: float = 0.0
        self._t_select_candidates: float = 0.0
        self._t_sticking_v1: float = 0.0
        self._t_overlap_check: float = 0.0
        self._t_rotation: float = 0.0
        self._n_overlap_calls: int = 0
        self._n_rotation_calls: int = 0

        # Incremental overlap telemetry (active-set + full-check)
        self._active_calls: int = 0
        self._full_calls: int = 0
        self._active_pairs_checked: int = 0
        self._full_pairs_checked: int = 0
        self._active_nonempty_hits: int = 0
        self._full_periodic_syncs: int = 0
        self._full_final_validations: int = 0

        # Candidate statistics by leaf class (LL/LN/NN)
        self._cand_attempts_ll: int = 0
        self._cand_attempts_ln: int = 0
        self._cand_attempts_nn: int = 0
        self._cand_success_ll: int = 0
        self._cand_success_ln: int = 0
        self._cand_success_nn: int = 0

        # Candidate score telemetry
        self._cand_score_attempt_sum: float = 0.0
        self._cand_score_attempt_count: int = 0
        self._cand_score_success_sum: float = 0.0
        self._cand_score_success_count: int = 0
        self._cand_score_attempt_high: int = 0
        self._cand_score_attempt_low: int = 0
        self._cand_score_success_high: int = 0
        self._cand_score_success_low: int = 0

        # Retry-mode telemetry
        self._retry_mode_counts: dict[str, int] = {}
        self._retry_mode_success_counts: dict[str, int] = {}
        self._retry_mode_success_attempt_sum: dict[str, int] = {}

        # Gamma expansion and pair feasibility telemetry
        self._gamma_expansion_hits: int = 0
        self._gamma_expansion_successes: int = 0
        self._gamma_expansion_total_steps: int = 0
        self._bv_filter_rejects: int = 0
        self._ssa_filter_rejects: int = 0

        # Opt-in overlap-failure census (cca_overlap_census_enabled), set by
        # fallbacks.py::_run_overlap_census_on_failure - see
        # docs/source/overlap_failure_census.md. None when disabled or
        # before any failure has been censused.
        self._last_overlap_census = None
        self._last_overlap_failure_geometry = None

        # Drop-rescue telemetry (cca_drop_rescue_enabled) - see
        # docs/source/drop_rescue.md.
        self._drop_rescue_attempts: int = 0
        self._drop_rescue_successes: int = 0
        self._particles_dropped_total: int = 0

        # FFT docking telemetry
        self._fft_docking_attempts: int = 0
        self._fft_docking_successes: int = 0

        # Soft relaxation telemetry
        self._soft_relaxation_attempts: int = 0
        self._soft_relaxation_successes: int = 0

        # Backtracking-pairing telemetry: merges that only succeeded
        # because a *later* partner was tried, and edges proven not to
        # stick. The first is the direct measure of what backtracking buys
        # over greedy first-fit.
        self._backtrack_rescued_merges: int = 0
        self._backtrack_failed_edges: int = 0
        self._pass_through_clusters: int = 0

        # Per-merge diagnostics. _last_sticking_stats is a side-channel
        # filled in by the sticking loop (same pattern as
        # _last_overlap_census) and drained by _record_merge_event.
        self._round_index: int = 1
        self._last_sticking_stats: dict = {}
        self._merge_log = None
        merge_log_path = self.algorithm_config.cca_merge_log_path
        if merge_log_path:
            from ..merge_log import MergeEventLog

            self._merge_log = MergeEventLog(merge_log_path)

    # --------------------------------------------------------------------------
    # Helper methods for CCA specific calculations
    # --------------------------------------------------------------------------

    def _get_cluster_data(self, cluster_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts coords and radii for a specific cluster index (0-based)."""
        if cluster_idx < 0 or cluster_idx >= self.i_t:
            raise IndexError(
                f"Cluster index {cluster_idx} out of bounds (0 to {self.i_t - 1})"
            )

        start_idx = self.i_orden[cluster_idx, 0]
        end_idx = self.i_orden[cluster_idx, 1] + 1  # Make exclusive for slicing
        count = self.i_orden[cluster_idx, 2]

        if start_idx < 0 or end_idx > self.N or count <= 0 or start_idx >= end_idx:
            # Return empty arrays for invalid/empty clusters defined in i_orden
            # logger.warning(f"Cluster {cluster_idx} has invalid definition in i_orden: start={start_idx}, end={end_idx-1}, count={count}. Returning empty.")
            return np.array([]).reshape(0, 3), np.array([])

        cluster_coords = self.coords[start_idx:end_idx, :]
        cluster_radii = self.radii[start_idx:end_idx]

        # Basic check
        if cluster_coords.shape[0] != count or cluster_radii.shape[0] != count:
            logger.warning(
                f"Mismatch between i_orden count ({count}) and sliced data length for cluster {cluster_idx} (Coords: {cluster_coords.shape[0]}, Radii: {cluster_radii.shape[0]})."
            )
            # Attempt to use the sliced data length if possible
            # Or handle as error? Let's proceed with caution.

        return cluster_coords, cluster_radii

    def _out_of_time(self) -> bool:
        """True once the wall-clock deadline has passed (latching)."""
        if self.deadline is None:
            return False
        if self.timed_out:
            return True
        if time.time() >= self.deadline:
            self.timed_out = True
            logger.warning("CCA aborting: wall-clock deadline reached mid-aggregation.")
            return True
        return False

    def _get_cluster_densities(self, cluster_idx: int) -> np.ndarray | None:
        """Densities of one cluster's particles, or None for uniform density."""
        if self.densities is None:
            return None
        start_idx = self.i_orden[cluster_idx, 0]
        end_idx = self.i_orden[cluster_idx, 1] + 1
        if start_idx < 0 or end_idx > self.densities.shape[0] or start_idx >= end_idx:
            return np.array([])
        return self.densities[start_idx:end_idx]

    # --------------------------------------------------------------------------
    # Main CCA Iteration Logic
    # --------------------------------------------------------------------------

    def _attempt_pair_merge(
        self,
        k: int,
        other: int,
        cluster_props_cache: dict | None,
        pool_size: int,
        attempt_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Attempt one cluster merge, including every configured fallback.

        The single place a pair of clusters is turned into a merged one:
        rigid sticking first, then (if enabled) soft relaxation, then
        drop-rescue. Shared by every pairing strategy so they cannot drift
        apart in which fallbacks they honour, and the one place merge
        events are recorded from.

        Also the single place per-particle densities are realigned. Every
        sticking path - rigid, soft relaxation, FFT docking, drop-rescue -
        returns its rows as ``[cluster1 rows..., cluster2 rows...]``, so
        the matching densities can be rebuilt here from the two clusters'
        own density slices rather than being threaded through each
        sticking routine's signature.

        Returns the merged ``(coords, radii, densities)`` (densities None
        for uniform density), or None if every route failed.
        """
        self._last_sticking_stats = {}
        # Clear the census side-channel too: it is only repopulated when a
        # failure gets far enough to run one, so leaving the previous
        # pair's census in place would attribute its offending-particle
        # counts to this merge (visible as offending fractions above 1.0).
        self._last_overlap_census = None
        self._last_overlap_failure_geometry = None
        outcome = "stuck"
        n_dropped = 0
        dens1 = self._get_cluster_densities(k)
        dens2 = self._get_cluster_densities(other)
        drop_indices: tuple[list[int], list[int]] | None = None

        stick_result = self._perform_cca_sticking_with_expansion(
            k, other, cluster_props_cache
        )

        # Try soft relaxation fallback if enabled and rigid sticking failed
        if (
            stick_result is None
            and self.algorithm_config.cca_soft_relaxation_enabled
            and self.algorithm_config.cca_soft_relaxation_fallback_only
        ):
            self._soft_relaxation_attempts += 1
            logger.info(
                f"Rigid sticking failed for pair ({k}, {other}), "
                f"trying soft relaxation fallback..."
            )
            stick_result = self._try_soft_relaxation_sticking(
                k, other, cluster_props_cache
            )
            if stick_result is not None:
                self._soft_relaxation_successes += 1
                outcome = "rescued_soft_relaxation"
                logger.info(f"Soft relaxation succeeded for pair ({k}, {other})")

        # Try drop-rescue if enabled and every prior fallback failed
        # (docs/source/drop_rescue.md). Depends on the overlap census
        # populated by _perform_cca_sticking_with_expansion's failure path
        # above - self.algorithm_config's validator guarantees
        # cca_overlap_census_enabled is also True whenever
        # cca_drop_rescue_enabled is.
        if (
            stick_result is None
            and self.algorithm_config.cca_drop_rescue_enabled
            and self._last_overlap_census is not None
            and self._last_overlap_failure_geometry is not None
        ):
            from .rescue import retry_sticking_with_drops, select_drop_candidates

            drop = select_drop_candidates(
                self._last_overlap_census,
                self.algorithm_config.cca_drop_rescue_max_particles,
                self.algorithm_config.cca_drop_rescue_max_fraction,
            )
            if drop is not None:
                self._drop_rescue_attempts += 1
                drop_idx1, drop_idx2 = drop
                c1, r1, c2, r2 = self._last_overlap_failure_geometry
                stick_result = retry_sticking_with_drops(
                    c1, r1, c2, r2, drop_idx1, drop_idx2, self.tol_ov
                )
                if stick_result is not None:
                    self._drop_rescue_successes += 1
                    n_dropped = len(drop_idx1) + len(drop_idx2)
                    self._particles_dropped_total += n_dropped
                    outcome = "rescued_drop"
                    drop_indices = (list(drop_idx1), list(drop_idx2))
                    logger.info(f"Drop-rescue succeeded for pair ({k}, {other})")

        if stick_result is None:
            outcome = self._last_sticking_stats.get("failure_reason", "failed_overlap")
        elif outcome == "stuck" and self._last_sticking_stats.get("used_adaptive_tol"):
            outcome = "stuck_relaxed_tol"

        self._record_merge_event(k, other, outcome, pool_size, attempt_index, n_dropped)
        if stick_result is None:
            return None

        merged_coords, merged_radii = stick_result
        merged_densities = self._merge_densities(
            dens1, dens2, drop_indices, merged_radii.shape[0]
        )
        return merged_coords, merged_radii, merged_densities

    def _merge_densities(
        self,
        dens1: np.ndarray | None,
        dens2: np.ndarray | None,
        drop_indices: tuple[list[int], list[int]] | None,
        expected_len: int,
    ) -> np.ndarray | None:
        """Rebuild a merged cluster's densities from its two parents'.

        Relies on every sticking path emitting ``[cluster1 rows...,
        cluster2 rows...]`` in the parents' own particle order, with
        drop-rescue removing the censused indices from each side while
        leaving the survivors' relative order intact.
        """
        if dens1 is None or dens2 is None:
            return None

        if drop_indices is not None:
            drop1, drop2 = drop_indices
            keep1 = np.setdiff1d(
                np.arange(dens1.shape[0]), np.asarray(drop1, dtype=int)
            )
            keep2 = np.setdiff1d(
                np.arange(dens2.shape[0]), np.asarray(drop2, dtype=int)
            )
            merged = np.concatenate((dens1[keep1], dens2[keep2]))
        else:
            merged = np.concatenate((dens1, dens2))

        if merged.shape[0] != expected_len:
            # Alignment is a correctness invariant, not a nicety: a
            # mismatch means densities would silently attach to the wrong
            # particles from here on. Fail loudly instead.
            raise RuntimeError(
                f"Density/particle misalignment after merge: rebuilt "
                f"{merged.shape[0]} densities for {expected_len} particles. "
                f"A sticking path must have changed its row ordering."
            )
        return merged

    def _record_merge_event(
        self,
        k: int,
        other: int,
        outcome: str,
        pool_size: int,
        attempt_index: int,
        n_dropped: int,
    ) -> None:
        """Append one record to the merge event log, when one is configured."""
        if self._merge_log is None:
            return

        from ..merge_log import MergeEvent

        stats = self._last_sticking_stats or {}
        census = self._last_overlap_census
        n_offending = None
        if census is not None and not outcome.startswith("stuck"):
            n_offending = int(
                census.n_particles_cluster1_offending
                + census.n_particles_cluster2_offending
            )

        self._merge_log.record(
            MergeEvent(
                round_index=self._round_index,
                pool_size=pool_size,
                cluster_idx1=int(k),
                cluster_idx2=int(other),
                n1=int(stats.get("n1", 0)),
                n2=int(stats.get("n2", 0)),
                gamma_pc=float(stats.get("gamma_pc", 0.0)),
                gamma_real=bool(stats.get("gamma_real", False)),
                sum_rmax=float(stats.get("sum_rmax", 0.0)),
                outcome=outcome,
                candidates_tried=int(stats.get("candidates_tried", 0)),
                n_feasible_pairs=int(stats.get("n_feasible_pairs", 0)),
                rotations_used=int(stats.get("rotations_used", 0)),
                min_overlap=float(stats.get("min_overlap", float("inf"))),
                n_offending_particles=n_offending,
                n_particles_dropped=int(n_dropped),
                attempt_index=int(attempt_index),
            )
        )

    def _assemble_next_round(self, merged_units: list) -> bool:
        """Install a round's resulting clusters as the next round's state.

        ``merged_units`` is the list of ``(coords, radii, densities)``
        produced by this round, one entry per surviving cluster. Rebuilds
        ``coords``/``radii``/``densities``/``i_orden``/``i_t`` from it.
        """
        if not merged_units:
            logger.error("No clusters formed in CCA iteration.")
            self.not_able_cca = True
            return False

        total = sum(unit[0].shape[0] for unit in merged_units)
        coords_next = np.zeros((total, 3), dtype=self.coords.dtype)
        radii_next = np.zeros(total, dtype=self.radii.dtype)
        densities_next = (
            np.zeros(total, dtype=float) if self.densities is not None else None
        )
        i_orden_next = np.zeros((len(merged_units), 3), dtype=int)

        fill_idx = 0
        for idx, (unit_coords, unit_radii, unit_densities) in enumerate(merged_units):
            count = unit_coords.shape[0]
            coords_next[fill_idx : fill_idx + count, :] = unit_coords
            radii_next[fill_idx : fill_idx + count] = unit_radii
            if densities_next is not None and unit_densities is not None:
                densities_next[fill_idx : fill_idx + count] = unit_densities
            i_orden_next[idx, 0] = fill_idx
            i_orden_next[idx, 1] = fill_idx + count - 1
            i_orden_next[idx, 2] = count
            fill_idx += count

        self.coords = coords_next
        self.radii = radii_next
        self.densities = densities_next
        self.i_orden = i_orden_next
        self.i_t = len(merged_units)
        return True

    def _run_iteration_backtracking(self) -> bool:
        """One CCA round that retries partners instead of aborting.

        The production pairing strategy. Where the greedy path commits to
        a partner per cluster up front and fails the entire round (and
        thus the whole PCA+CCA attempt) the moment any one chosen pair
        will not stick, this one reacts to the *actual* sticking outcome:
        on failure it tries the cluster's next feasible partner.

        This distinction is the whole point.
        docs/source/matching_pairing.md showed that choosing better pairs
        up front from the cheap gamma-feasibility graph does not help,
        because that graph is necessary-but-not-sufficient - it cannot
        predict which feasible-looking pairs actually stick. Only a real
        attempt tells you that, so only a strategy that reacts to real
        attempts can exploit what docs/source/pairing_frustration.md
        measured: in ~97% of hard-regime round failures, some *other*
        pairing of the very same pool would have worked.

        Cost is bounded by ``cca_backtracking_max_partners`` attempts per
        cluster, against a baseline that discards the round's successful
        merges and restarts PCA from scratch (up to 20 times).
        """
        from .matching import build_feasibility_graph

        pool_size = self.i_t
        logger.info(
            f"--- CCA Iteration Start (backtracking) - Clusters: {pool_size} ---"
        )

        cluster_props = self._compute_cluster_props()
        adj = build_feasibility_graph(
            cluster_props, self._calculate_cca_gamma, CCA_PAIRING_FACTOR
        )
        nodes = [i for i in range(self.i_t) if cluster_props[i][0] > 0.0]
        if not nodes:
            logger.error("CCA round has no non-empty clusters.")
            self.not_able_cca = True
            return False

        unpaired = set(nodes)
        merged_units: list = []
        n_merges = 0
        n_pass_through = 0
        # Edges proven not to stick this round; never retried from the
        # other endpoint either, since sticking is symmetric.
        failed_edges: set[frozenset] = set()
        max_partners = max(1, int(self.algorithm_config.cca_backtracking_max_partners))
        allow_pass_through = bool(self.algorithm_config.cca_backtracking_pass_through)

        while unpaired:
            if self._out_of_time():
                self.not_able_cca = True
                return False
            # Most-constrained-first: handle the cluster with the fewest
            # remaining options while it still has any, rather than
            # stranding it after its only partners are taken. The index
            # tiebreak keeps this deterministic for a given seed.
            k = min(unpaired, key=lambda i: (len(adj[i] & unpaired), i))
            unpaired.discard(k)

            partners = [
                p
                for p in adj[k]
                if p in unpaired and frozenset((k, p)) not in failed_edges
            ]
            partners.sort(key=lambda p: (len(adj[p] & unpaired), p))

            merged_partner = None
            merged_result = None
            for attempt_index, partner in enumerate(partners[:max_partners]):
                # Each extra partner is a full candidate/rotation search;
                # stop spending them once the budget is gone.
                if self._out_of_time():
                    self.not_able_cca = True
                    return False
                result = self._attempt_pair_merge(
                    k,
                    partner,
                    cluster_props,
                    pool_size=pool_size,
                    attempt_index=attempt_index,
                )
                if result is not None:
                    merged_partner = partner
                    merged_result = result
                    if attempt_index > 0:
                        self._backtrack_rescued_merges += 1
                        logger.info(
                            f"Backtracking rescued pair ({k}, {partner}) "
                            f"on partner attempt {attempt_index + 1}."
                        )
                    break
                failed_edges.add(frozenset((k, partner)))
                self._backtrack_failed_edges += 1

            if merged_result is not None and merged_partner is not None:
                unpaired.discard(merged_partner)
                merged_units.append(merged_result)
                n_merges += 1
                continue

            # No partner stuck. Carrying the cluster into the next round
            # unmerged keeps every *other* successful merge in this round,
            # which is precisely what the old abort-the-round behaviour
            # threw away.
            if not allow_pass_through and partners:
                logger.error(
                    f"Cluster {k} found no workable partner and pass-through is disabled."
                )
                self.not_able_cca = True
                return False
            coords_k, radii_k = self._get_cluster_data(k)
            if coords_k.shape[0] > 0:
                merged_units.append((coords_k, radii_k, self._get_cluster_densities(k)))
                n_pass_through += 1

        # A round where nothing merged makes no progress; letting it
        # continue would spin forever on an unchanged pool.
        if n_merges == 0 and pool_size > 1:
            logger.error(
                f"CCA round made no progress: {pool_size} clusters, none merged."
            )
            self.not_able_cca = True
            return False

        if n_pass_through:
            self._pass_through_clusters += n_pass_through
            logger.info(
                f"CCA round: {n_merges} merged, {n_pass_through} passed through unmerged."
            )

        self._round_index += 1
        if not self._assemble_next_round(merged_units):
            return False
        logger.info(f"--- CCA Iteration End - Clusters Remaining: {self.i_t} ---")
        return True

    def _run_iteration(self) -> bool:
        """Performs one iteration of the CCA process."""
        if str(self.algorithm_config.cca_pairing_strategy).lower() == "backtracking":
            return self._run_iteration_backtracking()

        logger.info(f"--- CCA Iteration Start - Clusters: {self.i_t} ---")

        # Sort clusters by size (optional, matches Fortran)
        # self.i_orden = utils.sort_clusters(self.i_orden) # Sorts by count

        # Generate pairs
        gen_result = self._generate_pairs()
        if gen_result is None or self.not_able_cca:
            logger.error("Failed to generate valid pairs.")
            self.not_able_cca = True
            return False  # Cannot continue
        id_agglomerated, cluster_props_cache = gen_result

        # Identify monomers
        id_monomers = self._identify_monomers()
        if id_monomers is None:
            logger.error("Failed to identify monomers.")
            self.not_able_cca = True
            return False

        # --- Agglomerate Pairs ---
        num_clusters_next = math.ceil(self.i_t / 2.0)
        merged_units: list = []

        considered = np.zeros(self.i_t, dtype=int)  # Track processed clusters (0-based)
        processed_pairs = set()  # Track (idx1, idx2) tuples already processed

        for k in range(self.i_t):  # Iterate cluster index 0 to i_t-1
            if considered[k] == 1:
                continue

            # Find partner 'other' for cluster k
            partners = np.where(id_agglomerated[k, :] == 1)[0]
            other = -1  # Initialize 'other' index

            if len(partners) == 0:
                # Should only happen if it's an empty cluster that wasn't skipped, or error.
                logger.warning(f"Cluster {k} is not considered but has no partners.")
                continue  # Skip this presumably empty or problematic cluster
            elif len(partners) == 1 and partners[0] == k:
                # This is the self-paired odd cluster
                other = k
            else:
                # Find the first valid, available partner
                for p in partners:
                    if k == p:
                        continue  # Skip self-reference unless it's the only one
                    pair_tuple = tuple(sorted((k, p)))
                    if considered[p] == 0 and pair_tuple not in processed_pairs:
                        other = p
                        processed_pairs.add(pair_tuple)
                        break
                if other == -1:
                    # All partners were already considered, or it's the odd one remaining
                    if id_agglomerated[k, k] == 1 and self.i_t % 2 != 0:
                        other = k  # It's the odd one
                    else:
                        # Should have been marked considered earlier
                        # logger.debug(f"Cluster {k} seems orphaned.")
                        continue  # Skip

            # --- Process the pair (k, other) ---
            if k == other:  # Handle single cluster (odd number case)
                # logger.info(f"Passing through single cluster {k}")
                coords_k, radii_k = self._get_cluster_data(k)
                count_k = coords_k.shape[0]
                if count_k == 0:
                    # logger.info(f"  Skipping empty single cluster {k}")
                    considered[k] = 1
                    continue  # Skip empty cluster

                combined_coords = coords_k
                combined_radii = radii_k
                combined_densities = self._get_cluster_densities(k)
                considered[k] = 1
            else:  # Handle a pair (k, other)
                # logger.info(f"Attempting to stick pair ({k}, {other})")
                stick_result = self._attempt_pair_merge(
                    k, other, cluster_props_cache, pool_size=self.i_t
                )

                if stick_result is None:
                    logger.info(
                        f"Sticking failed for pair ({k}, {other}). Cannot continue."
                    )
                    self.not_able_cca = True
                    return False  # Critical failure

                combined_coords, combined_radii, combined_densities = stick_result
                considered[k] = 1
                considered[other] = 1

            merged_units.append((combined_coords, combined_radii, combined_densities))

        # --- Post-Iteration Update ---
        # Check if expected number of clusters were formed
        if len(merged_units) != num_clusters_next:
            logger.warning(
                f"CCA iteration formed {len(merged_units)} clusters, expected {num_clusters_next}."
            )
            # This could happen if empty clusters were skipped.
            if not merged_units and self.i_t > 1:  # Check if any clusters remain
                logger.error("No clusters formed in CCA iteration.")
                self.not_able_cca = True
                return False
        # _assemble_next_round sizes everything from the units actually
        # produced, so a round that drops particles (drop-rescue) or skips
        # an empty cluster needs no separate trimming pass.
        self._round_index += 1
        if not self._assemble_next_round(merged_units):
            return False

        logger.info(f"--- CCA Iteration End - Clusters Remaining: {self.i_t} ---")
        return True  # Iteration successful

    def run_cca(self) -> Tuple[np.ndarray, np.ndarray] | None:
        """Run the complete CCA process until only one cluster remains.

        Repeatedly calls `_run_iteration` which performs pairing and sticking
        for the current set of clusters. Updates the internal state
        (`coords`, `radii`, `i_orden`, `i_t`) after each iteration.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            A tuple containing:
                - final_coords (np.ndarray): Nx3 coordinates of the final aggregate.
                - final_radii (np.ndarray): N radii of the final aggregate.

            Returns None if the aggregation process fails at any stage
            (sets `self.not_able_cca` to True).
        """
        cca_iteration = 1
        while self.i_t > 1:
            success = self._run_iteration()
            if not success:
                self.not_able_cca = True
                logger.error("CCA aggregation failed.")
                return None
            cca_iteration += 1

        # Final checks after loop terminates
        if self.not_able_cca:
            return None

        if self.i_t != 1:
            logger.error(f"CCA finished but i_t = {self.i_t} (expected 1).")
            self.not_able_cca = True
            return None

        # Check for NaNs/Infs in the final result
        if (
            np.any(np.isnan(self.coords))
            or np.any(np.isnan(self.radii))
            or np.any(np.isinf(self.coords))
            or np.any(np.isinf(self.radii))
        ):
            logger.error("NaN or Inf detected in final CCA coordinates/radii.")
            self.not_able_cca = True
            return None

        logger.info("CCA aggregation completed successfully.")
        if self.algorithm_config.profile_timing:
            t_total = (
                self._t_cluster_props
                + self._t_select_candidates
                + self._t_sticking_v1
                + self._t_overlap_check
                + self._t_rotation
            )
            print(
                f"\n[PROFILE] CCA timing summary (N={self.N}):\n"
                f"  cluster_props   : {self._t_cluster_props:7.3f}s\n"
                f"  select_cands    : {self._t_select_candidates:7.3f}s\n"
                f"  sticking_v1     : {self._t_sticking_v1:7.3f}s\n"
                f"  overlap_check   : {self._t_overlap_check:7.3f}s  ({self._n_overlap_calls} calls)\n"
                f"  rotation        : {self._t_rotation:7.3f}s  ({self._n_rotation_calls} calls)\n"
                f"  accounted total : {t_total:7.3f}s"
            )
            if self._active_calls + self._full_calls > 0:
                total_calls = self._active_calls + self._full_calls
                total_pairs = self._active_pairs_checked + self._full_pairs_checked
                active_avg_pairs = (
                    self._active_pairs_checked / self._active_calls
                    if self._active_calls
                    else 0.0
                )
                full_avg_pairs = (
                    self._full_pairs_checked / self._full_calls
                    if self._full_calls
                    else 0.0
                )
                print(
                    f"\n[PROFILE] CCA overlap checks:\n"
                    f"  active checks   : {self._active_calls:7d}  ({100.0 * self._active_calls / total_calls:5.1f}%)  avg_pairs={active_avg_pairs:8.1f}\n"
                    f"  full checks     : {self._full_calls:7d}  ({100.0 * self._full_calls / total_calls:5.1f}%)  avg_pairs={full_avg_pairs:8.1f}\n"
                    f"  total pairs chk : {total_pairs:7d}\n"
                    f"  active nonempty : {self._active_nonempty_hits:7d}\n"
                    f"  periodic full   : {self._full_periodic_syncs:7d}\n"
                    f"  final full val  : {self._full_final_validations:7d}"
                )
            if self.algorithm_config.profile_cca_leaf_stats:
                attempts_total = (
                    self._cand_attempts_ll
                    + self._cand_attempts_ln
                    + self._cand_attempts_nn
                )
                success_total = (
                    self._cand_success_ll
                    + self._cand_success_ln
                    + self._cand_success_nn
                )

                def _pct(part: int, whole: int) -> float:
                    return 100.0 * part / whole if whole > 0 else 0.0

                def _rate(success: int, attempts: int) -> float:
                    return 100.0 * success / attempts if attempts > 0 else 0.0

                print(
                    f"\n[PROFILE] CCA candidate leaf-class stats:\n"
                    f"  attempts total  : {attempts_total:7d}\n"
                    f"    LL attempts   : {self._cand_attempts_ll:7d} ({_pct(self._cand_attempts_ll, attempts_total):5.1f}%)\n"
                    f"    LN attempts   : {self._cand_attempts_ln:7d} ({_pct(self._cand_attempts_ln, attempts_total):5.1f}%)\n"
                    f"    NN attempts   : {self._cand_attempts_nn:7d} ({_pct(self._cand_attempts_nn, attempts_total):5.1f}%)\n"
                    f"  success total   : {success_total:7d}\n"
                    f"    LL success    : {self._cand_success_ll:7d} (rate={_rate(self._cand_success_ll, self._cand_attempts_ll):5.1f}%)\n"
                    f"    LN success    : {self._cand_success_ln:7d} (rate={_rate(self._cand_success_ln, self._cand_attempts_ln):5.1f}%)\n"
                    f"    NN success    : {self._cand_success_nn:7d} (rate={_rate(self._cand_success_nn, self._cand_attempts_nn):5.1f}%)"
                )
            if self.algorithm_config.profile_cca_candidate_score:
                att_n = self._cand_score_attempt_count
                suc_n = self._cand_score_success_count
                att_mean = self._cand_score_attempt_sum / att_n if att_n else 0.0
                suc_mean = self._cand_score_success_sum / suc_n if suc_n else 0.0
                high_att = self._cand_score_attempt_high
                low_att = self._cand_score_attempt_low
                high_suc = self._cand_score_success_high
                low_suc = self._cand_score_success_low

                def _rate(success: int, attempts: int) -> float:
                    return 100.0 * success / attempts if attempts > 0 else 0.0

                print(
                    f"\n[PROFILE] CCA candidate score stats:\n"
                    f"  attempts scored : {att_n:7d}  mean_score={att_mean:7.4f}\n"
                    f"  success scored  : {suc_n:7d}  mean_score={suc_mean:7.4f}\n"
                    f"  high-score (>=0.70): attempts={high_att:7d}, success={high_suc:7d}, rate={_rate(high_suc, high_att):5.1f}%\n"
                    f"  low-score  (<0.40): attempts={low_att:7d}, success={low_suc:7d}, rate={_rate(low_suc, low_att):5.1f}%"
                )
            if (
                self.algorithm_config.profile_cca_retry_modes
                and self._retry_mode_counts
            ):
                mode_items = sorted(
                    self._retry_mode_counts.items(), key=lambda item: item[0]
                )
                lines = []
                for mode, attempts in mode_items:
                    success = self._retry_mode_success_counts.get(mode, 0)
                    rate = 100.0 * success / attempts if attempts > 0 else 0.0
                    success_attempt_sum = self._retry_mode_success_attempt_sum.get(
                        mode, 0
                    )
                    mean_success_attempt = (
                        float(success_attempt_sum) / float(success)
                        if success > 0
                        else 0.0
                    )
                    lines.append(
                        f"    {mode:16s} attempts={attempts:7d} success={success:7d} rate={rate:5.1f}% mean_success_attempt={mean_success_attempt:7.2f}"
                    )
                print("\n[PROFILE] CCA retry-mode stats:\n" + "\n".join(lines))
        # Return only the valid part of the arrays corresponding to the final cluster
        final_count = self.i_orden[0, 2]
        return self.coords[:final_count, :], self.radii[:final_count]
