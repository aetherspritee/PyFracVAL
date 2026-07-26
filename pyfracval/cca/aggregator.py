"""Thin CCAggregator orchestrator composed from the CCA mixins.

Owns instance state (coords, radii, i_orden, telemetry counters) and the
top-level ``run_cca()`` entry point; delegates the actual pairing,
candidate selection, sticking, and fallback logic to the mixins in
:mod:`pyfracval.cca.pairing`, :mod:`pyfracval.cca.candidates`,
:mod:`pyfracval.cca.sticking`, and :mod:`pyfracval.cca.fallbacks`.
"""

import logging
import math
from typing import Tuple

import numpy as np

from ..config import OrchestratorAlgorithmConfig
from .candidates import _CandidatesMixin
from .fallbacks import _FallbacksMixin
from .pairing import _PairingMixin
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
        self.i_orden = initial_i_orden.copy()  # Shape (i_t, 3) [start, end, count]
        self.i_t = self.i_orden.shape[0]  # Current number of clusters

        self.not_able_cca = False

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

        # FFT docking telemetry
        self._fft_docking_attempts: int = 0
        self._fft_docking_successes: int = 0

        # Soft relaxation telemetry
        self._soft_relaxation_attempts: int = 0
        self._soft_relaxation_successes: int = 0

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

    # --------------------------------------------------------------------------
    # Main CCA Iteration Logic
    # --------------------------------------------------------------------------

    def _run_iteration(self) -> bool:
        """Performs one iteration of the CCA process."""
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
        coords_next = np.zeros_like(self.coords)
        radii_next = np.zeros_like(self.radii)
        i_orden_next = np.zeros((num_clusters_next, 3), dtype=int)

        considered = np.zeros(self.i_t, dtype=int)  # Track processed clusters (0-based)
        processed_pairs = set()  # Track (idx1, idx2) tuples already processed
        fill_idx = 0  # Index for coords_next/radii_next
        next_cluster_idx = 0  # Index for i_orden_next

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
                considered[k] = 1
            else:  # Handle a pair (k, other)
                # logger.info(f"Attempting to stick pair ({k}, {other})")
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
                        logger.info(
                            f"Soft relaxation succeeded for pair ({k}, {other})"
                        )

                if stick_result is None:
                    logger.info(
                        f"Sticking failed for pair ({k}, {other}). Cannot continue."
                    )
                    self.not_able_cca = True
                    return False  # Critical failure

                combined_coords, combined_radii = stick_result
                considered[k] = 1
                considered[other] = 1

            # --- Update next iteration arrays ---
            num_added = combined_coords.shape[0]
            if fill_idx + num_added > self.N:
                logger.error("Exceeding total particle count N during CCA iteration.")
                self.not_able_cca = True
                return False

            if next_cluster_idx >= num_clusters_next:
                logger.error(
                    "Exceeding expected number of clusters for next CCA iteration."
                )
                self.not_able_cca = True
                return False

            coords_next[fill_idx : fill_idx + num_added, :] = combined_coords
            radii_next[fill_idx : fill_idx + num_added] = combined_radii

            i_orden_next[next_cluster_idx, 0] = fill_idx
            i_orden_next[next_cluster_idx, 1] = fill_idx + num_added - 1
            i_orden_next[next_cluster_idx, 2] = num_added

            fill_idx += num_added
            next_cluster_idx += 1

        # --- Post-Iteration Update ---
        # Check if expected number of clusters were formed
        if next_cluster_idx != num_clusters_next:
            logger.warning(
                f"CCA iteration formed {next_cluster_idx} clusters, expected {num_clusters_next}."
            )
            # This could happen if empty clusters were skipped.
            if next_cluster_idx == 0 and self.i_t > 1:  # Check if any clusters remain
                logger.error("No clusters formed in CCA iteration.")
                self.not_able_cca = True
                return False
            # Adjust i_orden_next size if fewer clusters were formed
            i_orden_next = i_orden_next[:next_cluster_idx, :]
            num_clusters_next = next_cluster_idx  # Update expected count

        # Update state for the next iteration
        self.coords = coords_next
        self.radii = radii_next
        self.i_orden = i_orden_next
        self.i_t = num_clusters_next

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
