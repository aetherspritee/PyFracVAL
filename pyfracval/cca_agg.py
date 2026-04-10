"""Implements the Cluster-Cluster Aggregation (CCA) algorithm."""

import logging
import math
from time import perf_counter
from typing import Set, Tuple

import numpy as np

from . import config, utils
from .fft_docking import fft_dock_sticking
from .soft_relaxation import soft_sticking
from .logs import TRACE_LEVEL_NUM

logger = logging.getLogger(__name__)


def _pair_key(i: int, j: int, n2: int) -> int:
    """Pack pair indices (i,j) into a single integer key."""
    return i * n2 + j


def _pair_unpack(key: int, n2: int) -> tuple[int, int]:
    """Unpack integer pair key into (i,j)."""
    return key // n2, key % n2


class CCAggregator:
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

        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        # Current state of the simulation
        self.coords = initial_coords.copy()
        self.radii = initial_radii.copy()
        self.i_orden = initial_i_orden.copy()  # Shape (i_t, 3) [start, end, count]
        self.i_t = self.i_orden.shape[0]  # Current number of clusters

        self.not_able_cca = False

        # Timing accumulators (used when config.PROFILE_TIMING is True)
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

    def _calculate_cca_gamma(self, props1: Tuple, props2: Tuple) -> Tuple[bool, float]:
        """Calculates Gamma_pc between two clusters based on their properties."""
        m1, rg1, _, _, radii1 = props1
        m2, rg2, _, _, radii2 = props2
        return utils.gamma_calculation(
            m1,
            rg1,
            radii1,
            m2,
            rg2,
            radii2,
            self.df,
            self.kf,
        )
        # n1 = len(radii1)
        # n2 = len(radii2)

        # if n1 == 0 or n2 == 0:
        #     return False, 0.0

        # m3 = m1 + m2
        # n3 = n1 + n2

        # combined_radii = np.concatenate((radii1, radii2))
        # rg3 = utils.calculate_rg(combined_radii, n3, self.df, self.kf)

        # gamma_pc = 0.0
        # gamma_real = False
        # try:
        #     term1 = (m3**2) * (rg3**2)
        #     term2 = m3 * (m1 * rg1**2 + m2 * rg2**2)
        #     denominator = m1 * m2

        #     if term1 > term2 and denominator > 1e-12:
        #         gamma_pc = np.sqrt((term1 - term2) / denominator)
        #         gamma_real = True
        # except (ValueError, ZeroDivisionError, OverflowError) as e:
        #     logger.warning(f"CCA Gamma calculation failed: {e}")
        #     gamma_real = False

        # return gamma_real, gamma_pc

    def _identify_monomers(self) -> np.ndarray | None:
        """Creates an array mapping each monomer index (0..N-1) to its cluster index (0..i_t-1)."""
        try:
            id_monomers = np.zeros(self.N, dtype=int) - 1  # Initialize with -1
            for cluster_idx in range(self.i_t):
                start_idx = self.i_orden[cluster_idx, 0]
                end_idx = self.i_orden[cluster_idx, 1] + 1
                if (
                    start_idx < end_idx and start_idx >= 0 and end_idx <= self.N
                ):  # Valid range check
                    id_monomers[start_idx:end_idx] = cluster_idx
            # Check if all monomers were assigned
            if np.any(id_monomers < 0):
                unassigned = np.where(id_monomers < 0)[0]
                logger.warning(
                    f"{len(unassigned)} monomers not assigned to any cluster based on i_orden. Indices: {unassigned[:10]}..."
                )
                # This shouldn't happen if i_orden is correct. Force assign or error?
                # Let's allow it but CCA might fail later if it tries to access them.
            return id_monomers
        except IndexError:
            logger.error("Index out of bounds in _identify_monomers. Check i_orden.")
            return None

    # --------------------------------------------------------------------------
    # Pair Generation Logic
    # --------------------------------------------------------------------------

    def _generate_pairs(self) -> np.ndarray | None:
        """
        Generates the ID_agglomerated matrix indicating potential pairs.
        Applies a relaxation factor if the strict condition fails.
        Returns the matrix or None on failure.
        """
        # --- RELAXATION FACTOR ---
        # Allow gamma_pc to be slightly larger than sum_rmax if needed.
        # Start with a higher value to test if it allows pairing.
        # If this works, you might fine-tune it later (e.g., 1.10, 1.05).
        CCA_PAIRING_FACTOR = 1.10  # Relaxed pairing (10% over sum_rmax; gamma expansion handles the rest)
        strict_pairing_used = True  # Flag to track if relaxation was needed
        # -------------------------

        id_agglomerated = np.zeros((self.i_t, self.i_t), dtype=int)
        cluster_props = {}  # Cache properties

        # Pre-calculate properties (as before)
        for i in range(self.i_t):
            coords_i, radii_i = self._get_cluster_data(i)
            if coords_i.shape[0] == 0:
                cluster_props[i] = (0.0, 0.0, np.zeros(3), 0.0, np.array([]))
                continue
            m_i, rg_i, cm_i, r_max_i = utils.calculate_cluster_properties(
                coords_i,
                radii_i,
                self.df,
                self.kf,  # Use target Df/kf
            )
            cluster_props[i] = (m_i, rg_i, cm_i, r_max_i, radii_i)
            logger.debug(
                f"Cluster {i}: N={len(radii_i)}, Rg={rg_i:.3f}, Rmax={r_max_i:.3f}, Mass={m_i:.2e}"
            )

        # Check TRACE logging once (optimization: avoid check in inner loop)
        trace_enabled = logger.isEnabledFor(TRACE_LEVEL_NUM)

        # Pairing loop
        for i in range(self.i_t):
            if np.sum(id_agglomerated[i, :]) > 0 or cluster_props[i][0] == 0.0:
                continue

            m1, rg1, _, r_max1, radii1 = cluster_props[i]
            props1 = (m1, rg1, None, r_max1, radii1)
            partner_found = False

            for j in range(i + 1, self.i_t):
                if np.sum(id_agglomerated[:, j]) > 0 or cluster_props[j][0] == 0.0:
                    continue

                m2, rg2, _, r_max2, radii2 = cluster_props[j]
                props2 = (m2, rg2, None, r_max2, radii2)

                gamma_real, gamma_pc = self._calculate_cca_gamma(props1, props2)
                sum_rmax = r_max1 + r_max2

                # --- Check Strict and Relaxed Conditions ---
                strict_condition = gamma_real and gamma_pc < sum_rmax
                # Apply factor ONLY if gamma is real
                relaxed_condition = (
                    gamma_real and gamma_pc < sum_rmax * CCA_PAIRING_FACTOR
                )

                # Log trace information
                if trace_enabled:  # TRACE level (checked once for performance)
                    logger.log(
                        TRACE_LEVEL_NUM,
                        f"Pair ({i},{j}): G={gamma_pc:.3f}, R1+R2={sum_rmax:.3f}, StrictOK={strict_condition}, RelaxOK={relaxed_condition} (Factor={CCA_PAIRING_FACTOR})",
                    )

                # --- Apply Pairing Logic ---
                pair_marked = False
                if strict_condition:
                    id_agglomerated[i, j] = 1
                    id_agglomerated[j, i] = 1
                    partner_found = True
                    pair_marked = True
                    logger.debug(
                        f"  Pair ({i},{j}): Success! Marked for aggregation (Strict)."
                    )

                elif relaxed_condition:  # Check relaxed only if strict failed
                    id_agglomerated[i, j] = 1
                    id_agglomerated[j, i] = 1
                    partner_found = True
                    pair_marked = True
                    strict_pairing_used = False  # Set flag
                    logger.warning(
                        f"  Pair ({i},{j}): Marked using RELAXED condition "
                        f"(Gamma={gamma_pc:.3f} vs SumRmax={sum_rmax:.3f}). "
                        f"Final Df/kf may deviate slightly from target ({self.df:.2f}/{self.kf:.2f})."
                    )
                # --------------------------

                if pair_marked:
                    break  # Found partner for i

            if not partner_found and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"No suitable partner found for cluster {i} after checking all j > {i}."
                )

        # --- Handle the odd cluster out (Logic remains the same) ---
        if self.i_t % 2 != 0:
            paired_status = np.sum(id_agglomerated, axis=0) + np.sum(
                id_agglomerated, axis=1
            )
            unpaired_indices = np.where(paired_status == 0)[0]
            actual_unpaired = [
                idx for idx in unpaired_indices if cluster_props[idx][0] > 0.0
            ]
            if len(actual_unpaired) == 1:
                loc = actual_unpaired[0]
                id_agglomerated[loc, loc] = 1
                logger.debug(f"Marked cluster {loc} as the odd one out (pass-through).")
            elif len(actual_unpaired) > 1:
                logger.warning(
                    f"Found {len(actual_unpaired)} non-empty unpaired clusters ({actual_unpaired}) "
                    f"for odd i_t={self.i_t} even after checking pairs. Pairing may fail."
                )

        # --- Final check: Ensure all non-empty clusters are accounted for ---
        final_paired_status = np.sum(id_agglomerated, axis=0) + np.sum(
            id_agglomerated, axis=1
        )
        should_be_paired_mask = np.array(
            [cluster_props[idx][0] > 0.0 for idx in range(self.i_t)]
        )
        if np.any(final_paired_status[should_be_paired_mask] == 0):
            failed_indices = np.where(
                (final_paired_status == 0) & should_be_paired_mask
            )[0]
            logger.error(
                f"Could not find pairs for all non-empty clusters even with relaxation factor {CCA_PAIRING_FACTOR}. Failed indices: {failed_indices}"
            )
            logger.error("Consider increasing the target Df or kf.")
            self.not_able_cca = True
            return None

        if not strict_pairing_used:
            logger.warning(
                f"CCA pairing required relaxation (Factor={CCA_PAIRING_FACTOR}). Final aggregate properties may deviate slightly from target Df/kf."
            )

        logger.debug("Pair generation completed.")
        return id_agglomerated, cluster_props

    # --------------------------------------------------------------------------
    # CCA Sticking Logic (Methods corresponding to CCA subroutine and its calls)
    # --------------------------------------------------------------------------

    def _cca_select_candidates(
        self, coords1, radii1, cm1, coords2, radii2, cm2, gamma_pc, gamma_real
    ) -> np.ndarray:
        """Generates the n1 x n2 matrix of potential sticking pairs between clusters."""
        n1 = coords1.shape[0]
        n2 = coords2.shape[0]
        list_matrix = np.zeros((n1, n2), dtype=int)

        if not gamma_real or n1 == 0 or n2 == 0:
            return list_matrix

        # Distances of particles from their respective CMs
        dist1 = np.linalg.norm(coords1 - cm1, axis=1)
        dist2 = np.linalg.norm(coords2 - cm2, axis=1)

        if self.ext_case == 1:
            d1_min = dist1 - radii1
            d1_max = dist1 + radii1
            d2_min = dist2 - radii2
            d2_max = dist2 + radii2

            # Use broadcasting for efficient comparison
            d1max_col = d1_max[:, np.newaxis]
            d2max_row = d2_max[np.newaxis, :]
            d1min_col = d1_min[:, np.newaxis]
            d2min_row = d2_min[np.newaxis, :]

            cond1 = (d1max_col + d2max_row) > gamma_pc
            abs_diff = np.abs(d2max_row - d1max_col)
            cond2a = abs_diff < gamma_pc
            cond2b = ((d2max_row - d1max_col) > gamma_pc) & (
                (d2min_row - d1max_col) < gamma_pc
            )
            cond2c = ((d1max_col - d2max_row) > gamma_pc) & (
                (d1min_col - d2max_row) < gamma_pc
            )

            list_matrix[cond1 & (cond2a | cond2b | cond2c)] = 1

        elif self.ext_case == 0:
            d1_max = dist1 + radii1
            d2_max = dist2 + radii2

            d1max_col = d1_max[:, np.newaxis]
            d2max_row = d2_max[np.newaxis, :]

            cond1 = (d1max_col + d2max_row) > gamma_pc
            cond2 = np.abs(d2max_row - d1max_col) < gamma_pc
            list_matrix[cond1 & cond2] = 1

        return list_matrix

    def _cca_pick_candidate_pair(
        self, list_matrix: np.ndarray, tried_pairs: Set[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        Selects a random available candidate pair (cand1_idx, cand2_idx) from the list matrix.
        Avoids pairs in tried_pairs. Returns (-1, -1) if none available.
        """
        valid_indices = np.argwhere(list_matrix > 0)  # Get indices where value is 1
        available_pairs = []
        for idx_pair in valid_indices:
            pair = tuple(idx_pair)
            if pair not in tried_pairs:
                available_pairs.append(pair)

        if not available_pairs:
            return -1, -1

        # Select a random pair from the available ones
        selected_pair_idx = int(self._rng.integers(len(available_pairs)))
        return available_pairs[selected_pair_idx]

    def _cca_sticking_v1(
        self, cluster1_data, cluster2_data, cand1_idx, cand2_idx, gamma_pc, gamma_real
    ) -> Tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Performs the initial sticking placement for CCA (corresponds to CCA_Sticking_process_v1).
        Handles translation of cluster 2, finding contact point, rotating cluster 1,
        finding second contact point, rotating cluster 2.

        Args:
            cluster1_data: Tuple (coords1, radii1, cm1)
            cluster2_data: Tuple (coords2, radii2, cm2)
            cand1_idx, cand2_idx: Indices of selected contact particles.
            gamma_pc, gamma_real: Pre-calculated gamma values.

        Returns:
            Tuple: (coords1_out, cm1_out, coords2_out, cm2_out, theta_a, vec_0, i_vec, j_vec)
                   Returns (None, ..., None) on failure.
        """
        coords1_in, radii1, cm1_in = cluster1_data
        coords2_in, radii2, cm2_in = cluster2_data
        n1 = coords1_in.shape[0]
        n2 = coords2_in.shape[0]

        # Work with copies
        coords1 = coords1_in.copy()
        coords2 = coords2_in.copy()
        cm1 = cm1_in.copy()
        cm2 = cm2_in.copy()  # Will be updated

        # --- Step 1: Translate Cluster 2 ---
        vec_cm1_p1 = coords1[cand1_idx] - cm1
        vec_cm1_p1 /= np.linalg.norm(vec_cm1_p1)
        if np.linalg.norm(vec_cm1_p1) < 1e-9:
            logger.warning("CCA Stick V1 - Selected particle coincides with CM1.")
            vec_cm1_p1 = np.array([1.0, 0.0, 0.0])  # Arbitrary direction

        cm2_target = cm1 + gamma_pc * vec_cm1_p1
        desplazamiento = cm2_target - cm2
        coords2 += desplazamiento  # Translate all particles
        cm2 = cm2_target  # Update CM2 position

        # --- Step 2: Find Initial Contact Point on Surface of Sphere 1 ---
        # Based on Fortran logic: This involves potentially complex intersection of
        # surfaces defined by Dmin/Dmax distances from CMs.
        contact_point = None
        point_valid = False

        # Re-calculate Dmin/max with translated coords2
        dist1 = np.linalg.norm(coords1[cand1_idx] - cm1)
        d1_min = dist1 - radii1[cand1_idx]
        d1_max = dist1 + radii1[cand1_idx]
        dist2 = np.linalg.norm(coords2[cand2_idx] - cm2)  # Use updated coords2/cm2
        d2_min = dist2 - radii2[cand2_idx]
        d2_max = dist2 + radii2[cand2_idx]

        spheres_1_ext = np.array([cm1[0], cm1[1], cm1[2], d1_min, d1_max])
        spheres_2_ext = np.array(
            [cm2[0], cm2[1], cm2[2], d2_min, d2_max]
        )  # Use updated cm2

        case = 0
        if self.ext_case == 1:
            # Determine case based on Dmin/max overlap relative to gamma_pc
            gamma_pc_thresh = gamma_pc
            if (d1_max + d2_max) > gamma_pc_thresh:
                abs_diff = abs(d2_max - d1_max)
                if abs_diff < gamma_pc_thresh:
                    case = 1
                elif (d2_max - d1_max > gamma_pc_thresh) and (
                    d2_min - d1_max < gamma_pc_thresh
                ):
                    case = 2
                elif (d1_max - d2_max > gamma_pc_thresh) and (
                    d1_min - d2_max < gamma_pc_thresh
                ):
                    case = 3

            if case > 0:
                x_cp, y_cp, z_cp, point_valid = utils.random_point_sc(
                    case, spheres_1_ext, spheres_2_ext
                )
                if point_valid:
                    contact_point = np.array([x_cp, y_cp, z_cp])
            # else: point_valid remains False

        elif self.ext_case == 0:
            # Use intersection of spheres defined by D1max and D2max
            sphere_1 = np.concatenate((cm1, [d1_max]))
            sphere_2 = np.concatenate((cm2, [d2_max]))  # Use updated cm2
            x_cp, y_cp, z_cp, _, _, _, _, point_valid = utils.two_sphere_intersection(
                sphere_1, sphere_2, rng=self._rng
            )
            if point_valid:
                contact_point = np.array([x_cp, y_cp, z_cp])

        if not point_valid or contact_point is None:
            logger.warning(
                f"CCA Stick V1 - Failed to find initial contact point (ext_case={self.ext_case}, case={case})."
            )
            return (
                None,
                None,
                None,
                0.0,
                np.zeros(4),
                np.zeros(3),
                np.zeros(3),
            )  # Failure

        # Refine contact point to be on surface of particle cand1_idx
        # Vector from particle center towards the calculated contact_point
        vec_p1_contact = contact_point - coords1[cand1_idx]
        vec_p1_contact /= np.linalg.norm(vec_p1_contact)
        if np.linalg.norm(vec_p1_contact) < 1e-9:
            # logger.warning("CCA Stick V1 - Contact point direction undefined.")
            # If direction is undefined, maybe stick along original cm1-p1 vector?
            temp = coords1[cand1_idx] - cm1
            final_contact_point_p1 = coords1[cand1_idx] + radii1[
                cand1_idx
            ] * temp / np.linalg.norm(temp)
        else:
            final_contact_point_p1 = (
                coords1[cand1_idx] + radii1[cand1_idx] * vec_p1_contact
            )

        # --- Step 3: Rotate Cluster 1 ---
        target_p1 = final_contact_point_p1
        current_p1 = coords1[cand1_idx]
        v1_rot = current_p1 - cm1
        v2_rot = target_p1 - cm1

        norm_v1 = np.linalg.norm(v1_rot)
        norm_v2 = np.linalg.norm(v2_rot)

        # Calculate rotation axis and angle
        rot_axis1 = np.zeros(3)
        rot_angle1 = 0.0
        perform_rot1 = True

        if norm_v1 > 1e-9 and norm_v2 > 1e-9:
            v1_u = v1_rot / norm_v1
            v2_u = v2_rot / norm_v2
            dot_prod = np.dot(v1_u, v2_u)

            if abs(dot_prod) > 1.0 - 1e-9:  # Collinear
                if dot_prod < 0:  # Anti-aligned
                    rot_angle1 = np.pi
                    # Find perpendicular axis
                    if abs(v1_u[0]) < 1e-9 and abs(v1_u[1]) < 1e-9:
                        rot_axis1 = np.array([1.0, 0.0, 0.0])
                    else:
                        rot_axis1 = np.array([-v1_u[1], v1_u[0], 0.0])
                else:  # Aligned
                    perform_rot1 = False  # No rotation needed
            else:  # Standard rotation
                rot_angle1 = np.arccos(np.clip(dot_prod, -1.0, 1.0))
                rot_axis1 = np.cross(v1_u, v2_u)
        else:  # One vector is zero length
            perform_rot1 = False

        # Apply rotation 1
        if perform_rot1 and np.linalg.norm(rot_axis1) > 1e-9 and abs(rot_angle1) > 1e-9:
            coords1_rel = coords1 - cm1
            coords1_rel_rotated = utils.rodrigues_rotation(
                coords1_rel, rot_axis1, rot_angle1
            )
            coords1 = coords1_rel_rotated + cm1
            # Update CM? No, rotation is around CM.

        # --- Step 4: Find Second Contact Point (Sphere Intersection) ---
        center_A = coords1[cand1_idx]  # Use updated coords1
        radius_A = radii1[cand1_idx] + radii2[cand2_idx]
        sphere_A = np.concatenate((center_A, [radius_A]))

        center_B = cm2  # Use updated cm2
        radius_B = np.linalg.norm(coords2[cand2_idx] - center_B)  # Use updated coords2
        sphere_B = np.concatenate((center_B, [radius_B]))

        x_cp2, y_cp2, z_cp2, theta_a, vec_0, i_vec, j_vec, intersection_valid = (
            utils.two_sphere_intersection(sphere_A, sphere_B, rng=self._rng)
        )

        if not intersection_valid:
            logger.debug(
                f"CCA Stick V1 - Failed sphere intersection A/B. cand1={cand1_idx}, cand2={cand2_idx}"
            )
            distAB = np.linalg.norm(center_A - center_B)
            logger.debug(
                f"  Dist={distAB:.4f}, R_A={radius_A:.4f}, R_B={radius_B:.4f}, Sum={radius_A + radius_B:.4f}"
            )
            return (
                None,
                None,
                None,
                0.0,
                np.zeros(4),
                np.zeros(3),
                np.zeros(3),
            )  # Failure

        final_contact_point_p2 = np.array([x_cp2, y_cp2, z_cp2])

        # --- Step 5: Rotate Cluster 2 ---
        target_p2 = final_contact_point_p2
        current_p2 = coords2[cand2_idx]  # Use updated coords2
        v1_rot = current_p2 - cm2
        v2_rot = target_p2 - cm2

        norm_v1 = np.linalg.norm(v1_rot)
        norm_v2 = np.linalg.norm(v2_rot)

        rot_axis2 = np.zeros(3)
        rot_angle2 = 0.0
        perform_rot2 = True

        if norm_v1 > 1e-9 and norm_v2 > 1e-9:
            v1_u = v1_rot / norm_v1
            v2_u = v2_rot / norm_v2
            dot_prod = np.dot(v1_u, v2_u)
            if abs(dot_prod) > 1.0 - 1e-9:
                if dot_prod < 0:
                    rot_angle2 = np.pi
                    if abs(v1_u[0]) < 1e-9 and abs(v1_u[1]) < 1e-9:
                        rot_axis2 = np.array([1.0, 0.0, 0.0])
                    else:
                        rot_axis2 = np.array([-v1_u[1], v1_u[0], 0.0])
                else:
                    perform_rot2 = False
            else:
                rot_angle2 = np.arccos(np.clip(dot_prod, -1.0, 1.0))
                rot_axis2 = np.cross(v1_u, v2_u)
        else:
            perform_rot2 = False

        if perform_rot2 and np.linalg.norm(rot_axis2) > 1e-9 and abs(rot_angle2) > 1e-9:
            coords2_rel = coords2 - cm2
            coords2_rel_rotated = utils.rodrigues_rotation(
                coords2_rel, rot_axis2, rot_angle2
            )
            coords2 = coords2_rel_rotated + cm2
            # Update CM? No.

        # Return final state after initial sticking
        return coords1, coords2, cm2, theta_a, vec_0, i_vec, j_vec

    def _cca_reintento(
        self,
        coords2_in: np.ndarray,
        cm2: np.ndarray,
        cand2_idx: int,
        vec_0: np.ndarray,
        i_vec: np.ndarray,
        j_vec: np.ndarray,
        attempt: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """Thin wrapper — delegates to JIT kernel (PyFracVAL-dsa).

        Rotates cluster 2 to the next Fibonacci-spiral position on the
        intersection circle.  The heavy lifting is done by
        ``utils._cca_reintento_kernel`` which is @njit-compiled to eliminate
        Python dispatch overhead and numpy scalar overhead for every step.
        """
        x0, y0, z0, r0 = vec_0
        coords2_out = utils._cca_reintento_kernel(
            coords2_in,
            cm2,
            cand2_idx,
            float(x0),
            float(y0),
            float(z0),
            float(r0),
            float(i_vec[0]),
            float(i_vec[1]),
            float(i_vec[2]),
            float(j_vec[0]),
            float(j_vec[1]),
            float(j_vec[2]),
            int(attempt),
        )
        return coords2_out, 0.0  # theta_a_new no longer needed by caller

    @staticmethod
    def _rotate_cluster_about_cm(
        coords_in: np.ndarray,
        cm: np.ndarray,
        axis: np.ndarray,
        angle_rad: float,
    ) -> np.ndarray:
        """Rotate a full cluster around its center of mass."""
        if np.linalg.norm(axis) <= 1.0e-12 or abs(float(angle_rad)) <= 1.0e-12:
            return coords_in
        coords_rel = coords_in - cm
        coords_rel_rot = utils.rodrigues_rotation(coords_rel, axis, float(angle_rad))
        return coords_rel_rot + cm

    @staticmethod
    def _normalize_axis(
        axis: np.ndarray, fallback: np.ndarray | None = None
    ) -> np.ndarray:
        axis_out = np.array(axis, dtype=float)
        axis_norm = float(np.linalg.norm(axis_out))
        if axis_norm > 1.0e-12:
            return axis_out / axis_norm
        if fallback is not None:
            fb = np.array(fallback, dtype=float)
            fb_norm = float(np.linalg.norm(fb))
            if fb_norm > 1.0e-12:
                return fb / fb_norm
        return np.array([1.0, 0.0, 0.0], dtype=float)

    def _apply_retry_rotation_mode(
        self,
        coords1_stick: np.ndarray,
        coords2_current: np.ndarray,
        coords1_base: np.ndarray,
        coords2_base: np.ndarray,
        cm1: np.ndarray,
        cm2_stick: np.ndarray,
        cand1_idx: int,
        cand2_idx: int,
        vec_0: np.ndarray,
        i_vec: np.ndarray,
        j_vec: np.ndarray,
        axis_anchor: np.ndarray,
        axis_moving: np.ndarray,
        intento: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Generate next retry pose according to configured retry mode."""
        mode_cfg = str(getattr(config, "CCA_RETRY_ROTATION_MODE", "single")).lower()
        if mode_cfg not in {
            "single",
            "alternate",
            "dual_jitter",
            "coarse_grid",
            "coarse_to_fine",
        }:
            mode_cfg = "single"

        if mode_cfg == "coarse_grid":
            sweep_steps = int(max(1, getattr(config, "CCA_COARSE_SWEEP_STEPS", 10)))
            spin_anchor_steps = int(
                max(1, getattr(config, "CCA_COARSE_SPIN_ANCHOR_STEPS", 6))
            )
            spin_moving_steps = int(
                max(1, getattr(config, "CCA_COARSE_SPIN_MOVING_STEPS", 6))
            )
            total = sweep_steps * spin_anchor_steps * spin_moving_steps
            idx = (int(intento) - 1) % total
            block = spin_anchor_steps * spin_moving_steps
            sweep_idx = idx // block
            rem = idx % block
            anchor_idx = rem // spin_moving_steps
            moving_idx = rem % spin_moving_steps

            sweep_attempt = int(
                round((float(sweep_idx + 1) / float(sweep_steps)) * 360.0)
            )
            sweep_attempt = max(1, min(360, sweep_attempt))
            coords2_swept, _ = self._cca_reintento(
                coords2_base,
                cm2_stick,
                cand2_idx,
                vec_0,
                i_vec,
                j_vec,
                attempt=sweep_attempt,
            )

            anchor_angle = (2.0 * config.PI * float(anchor_idx)) / float(
                spin_anchor_steps
            )
            moving_angle = (2.0 * config.PI * float(moving_idx)) / float(
                spin_moving_steps
            )

            coords1_next = self._rotate_cluster_about_cm(
                coords1_base,
                cm1,
                axis_anchor,
                anchor_angle,
            )
            coords2_next = self._rotate_cluster_about_cm(
                coords2_swept,
                cm2_stick,
                axis_moving,
                moving_angle,
            )
            return coords1_next, coords2_next, "coarse_grid"

        if mode_cfg == "coarse_to_fine":
            sweep_steps = int(max(1, getattr(config, "CCA_COARSE_SWEEP_STEPS", 10)))
            spin_anchor_steps = int(
                max(1, getattr(config, "CCA_COARSE_SPIN_ANCHOR_STEPS", 6))
            )
            spin_moving_steps = int(
                max(1, getattr(config, "CCA_COARSE_SPIN_MOVING_STEPS", 6))
            )
            total = sweep_steps * spin_anchor_steps * spin_moving_steps
            coarse_fraction = float(
                getattr(config, "CCA_COARSE_FINE_COARSE_FRACTION", 0.67)
            )
            coarse_fraction = min(max(coarse_fraction, 0.05), 0.95)
            coarse_budget = max(1, min(total - 1, int(round(total * coarse_fraction))))

            if int(intento) <= coarse_budget:
                if coarse_budget == 1:
                    idx = 0
                else:
                    idx = int(
                        round(
                            ((int(intento) - 1) * (total - 1))
                            / float(coarse_budget - 1)
                        )
                    )
                block = spin_anchor_steps * spin_moving_steps
                sweep_idx = idx // block
                rem = idx % block
                anchor_idx = rem // spin_moving_steps
                moving_idx = rem % spin_moving_steps

                sweep_attempt = int(
                    round((float(sweep_idx + 1) / float(sweep_steps)) * 360.0)
                )
                sweep_attempt = max(1, min(360, sweep_attempt))
                coords2_swept, _ = self._cca_reintento(
                    coords2_base,
                    cm2_stick,
                    cand2_idx,
                    vec_0,
                    i_vec,
                    j_vec,
                    attempt=sweep_attempt,
                )

                anchor_angle = (2.0 * config.PI * float(anchor_idx)) / float(
                    spin_anchor_steps
                )
                moving_angle = (2.0 * config.PI * float(moving_idx)) / float(
                    spin_moving_steps
                )
                coords1_next = self._rotate_cluster_about_cm(
                    coords1_base,
                    cm1,
                    axis_anchor,
                    anchor_angle,
                )
                coords2_next = self._rotate_cluster_about_cm(
                    coords2_swept,
                    cm2_stick,
                    axis_moving,
                    moving_angle,
                )
                return coords1_next, coords2_next, "coarse_to_fine_coarse"

            refine_idx = int(intento) - coarse_budget
            refine_deg = float(
                max(0.0, getattr(config, "CCA_COARSE_FINE_SPIN_DEG", 12.0))
            )
            refine_rad = np.deg2rad(refine_deg)
            phi = 2.0 * config.PI * float(refine_idx) / float(config.GOLDEN_RATIO)
            angle_anchor = refine_rad * float(np.sin(phi))
            angle_moving = refine_rad * float(np.cos(phi))
            coords1_next = self._rotate_cluster_about_cm(
                coords1_stick,
                cm1,
                axis_anchor,
                angle_anchor,
            )
            coords2_next = self._rotate_cluster_about_cm(
                coords2_current,
                cm2_stick,
                axis_moving,
                angle_moving,
            )
            return coords1_next, coords2_next, "coarse_to_fine_refine"

        escalate_after = int(max(0, getattr(config, "CCA_RETRY_ESCALATE_AFTER", 0)))
        use_mode = mode_cfg if intento > escalate_after else "single"

        coords1_next = coords1_stick
        coords2_next = coords2_current

        if use_mode == "single":
            coords2_next, _ = self._cca_reintento(
                coords2_current,
                cm2_stick,
                cand2_idx,
                vec_0,
                i_vec,
                j_vec,
                attempt=intento,
            )
            return coords1_next, coords2_next, "single"

        if use_mode == "alternate":
            if intento % 2 == 0:
                phi = 2.0 * config.PI * float(intento) / float(config.GOLDEN_RATIO)
                axis = np.array([i_vec[0], i_vec[1], i_vec[2]], dtype=float)
                axis = self._normalize_axis(axis, fallback=np.array([1.0, 0.0, 0.0]))
                coords1_next = self._rotate_cluster_about_cm(
                    coords1_stick,
                    cm1,
                    axis,
                    -phi,
                )
                return coords1_next, coords2_next, "alternate_anchor"

            coords2_next, _ = self._cca_reintento(
                coords2_current,
                cm2_stick,
                cand2_idx,
                vec_0,
                i_vec,
                j_vec,
                attempt=intento,
            )
            return coords1_next, coords2_next, "alternate_moving"

        coords2_next, _ = self._cca_reintento(
            coords2_current,
            cm2_stick,
            cand2_idx,
            vec_0,
            i_vec,
            j_vec,
            attempt=intento,
        )
        jitter_interval = int(max(1, getattr(config, "CCA_DUAL_JITTER_INTERVAL", 5)))
        if intento % jitter_interval == 0:
            jitter_deg = float(max(0.0, getattr(config, "CCA_DUAL_JITTER_DEG", 8.0)))
            jitter_rad = np.deg2rad(jitter_deg)
            if jitter_rad > 0.0:
                axis = self._rng.normal(size=3)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm > 1.0e-12:
                    axis = axis / axis_norm
                    angle = float(self._rng.uniform(-jitter_rad, jitter_rad))
                    coords1_next = self._rotate_cluster_about_cm(
                        coords1_stick,
                        cm1,
                        axis,
                        angle,
                    )
                    return coords1_next, coords2_next, "dual_jitter"
        return coords1_next, coords2_next, "dual_moving"

    @staticmethod
    def _pair_overlap(
        coords1: np.ndarray,
        radii1: np.ndarray,
        coords2: np.ndarray,
        radii2: np.ndarray,
        i: int,
        j: int,
    ) -> float:
        """Compute overlap for a single pair (i,j)."""
        dx = coords1[i, 0] - coords2[j, 0]
        dy = coords1[i, 1] - coords2[j, 1]
        dz = coords1[i, 2] - coords2[j, 2]
        d_sq = dx * dx + dy * dy + dz * dz
        radius_sum = radii1[i] + radii2[j]
        r_sq = radius_sum * radius_sum
        if d_sq > r_sq:
            return -np.inf
        dist = math.sqrt(d_sq)
        return 1.0 - dist / radius_sum

    @staticmethod
    def _leaf_mask_for_cluster(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Return bool mask of leaf-like monomers (degree <= 1) within a cluster."""
        n = coords.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        if n == 1:
            return np.ones(1, dtype=bool)

        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        d_sq = np.sum(diffs * diffs, axis=2)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        thr_sq = (r_sum + 1.0e-9) * (r_sum + 1.0e-9)
        contact = d_sq <= thr_sq
        np.fill_diagonal(contact, False)
        degree = np.sum(contact, axis=1)
        return degree <= 1

    def _record_candidate_attempt(self, leaf1: bool, leaf2: bool) -> str:
        if leaf1 and leaf2:
            self._cand_attempts_ll += 1
            return "LL"
        if leaf1 or leaf2:
            self._cand_attempts_ln += 1
            return "LN"
        self._cand_attempts_nn += 1
        return "NN"

    def _record_candidate_success(self, cls: str) -> None:
        if cls == "LL":
            self._cand_success_ll += 1
        elif cls == "LN":
            self._cand_success_ln += 1
        else:
            self._cand_success_nn += 1

    @staticmethod
    def _candidate_leaf_class(leaf1: bool, leaf2: bool) -> str:
        if leaf1 and leaf2:
            return "LL"
        if leaf1 or leaf2:
            return "LN"
        return "NN"

    @staticmethod
    def _candidate_score(
        coords1: np.ndarray,
        radii1: np.ndarray,
        cm1: np.ndarray,
        cand1_idx: int,
        coords2: np.ndarray,
        radii2: np.ndarray,
        cm2: np.ndarray,
        cand2_idx: int,
        gamma_pc: float,
        leaf_cls: str,
    ) -> float:
        """Heuristic candidate score in [0,1], larger means more promising."""
        d1 = float(np.linalg.norm(coords1[cand1_idx] - cm1))
        d2 = float(np.linalg.norm(coords2[cand2_idx] - cm2))
        s1 = d1 + float(radii1[cand1_idx])
        s2 = d2 + float(radii2[cand2_idx])

        # Radial compatibility against gamma shell relation.
        err = abs((s1 + s2) - gamma_pc) / max(gamma_pc, 1.0e-12)
        tau_r = 0.08
        radial = math.exp(-err / tau_r)

        # Class prior from observed empirical success tendency.
        if leaf_cls == "LL":
            leaf_prior = 1.0
        elif leaf_cls == "LN":
            leaf_prior = 0.45
        else:
            leaf_prior = 0.10

        score = 0.5 * leaf_prior + 0.5 * radial
        return max(0.0, min(1.0, score))

    def _record_candidate_score_attempt(self, score: float) -> None:
        self._cand_score_attempt_sum += score
        self._cand_score_attempt_count += 1
        if score >= 0.70:
            self._cand_score_attempt_high += 1
        if score < 0.40:
            self._cand_score_attempt_low += 1

    def _record_candidate_score_success(self, score: float) -> None:
        self._cand_score_success_sum += score
        self._cand_score_success_count += 1
        if score >= 0.70:
            self._cand_score_success_high += 1
        if score < 0.40:
            self._cand_score_success_low += 1

    def _scan_active_collisions(
        self,
        coords1: np.ndarray,
        radii1: np.ndarray,
        coords2: np.ndarray,
        radii2: np.ndarray,
        pair_keys: set[int],
        n2: int,
    ) -> tuple[float, set[int]]:
        """Check overlap only for currently active collision pairs."""
        max_overlap = 0.0
        active: set[int] = set()
        if not pair_keys:
            return max_overlap, active

        for key in pair_keys:
            i, j = _pair_unpack(key, n2)
            overlap = self._pair_overlap(coords1, radii1, coords2, radii2, i, j)
            if overlap > max_overlap:
                max_overlap = overlap
            if overlap > self.tol_ov:
                active.add(key)

        return max_overlap, active

    def _full_overlap_check(
        self,
        coords1: np.ndarray,
        radii1: np.ndarray,
        coords2: np.ndarray,
        radii2: np.ndarray,
    ) -> float:
        """Run global overlap check using fast early-termination kernel."""
        self._full_calls += 1
        return utils.calculate_max_overlap_cca_auto(
            coords1,
            radii1,
            coords2,
            radii2,
            tolerance=self.tol_ov,
        )

    # ------------------------------------------------------------------
    # Gamma Expansion and Pair Feasibility Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _bounding_volume_precheck(
        gamma_pc: float,
        r_max1: float,
        r_max2: float,
        gamma_real: bool,
    ) -> bool:
        """Check if two clusters can physically stick at distance gamma_pc."""
        if not gamma_real or gamma_pc <= 0.0:
            return False
        factor = float(getattr(config, "CCA_BV_DEEP_PENETRATION_FACTOR", 0.8))
        rmax_diff = abs(r_max1 - r_max2)
        if gamma_pc < rmax_diff * factor:
            logger.debug(
                f"BV pre-check reject: gamma={gamma_pc:.3f} < "
                f"|r_max1-r_max2|*{factor}={rmax_diff * factor:.3f}"
            )
            return False
        if gamma_pc > r_max1 + r_max2:
            logger.debug(
                f"BV pre-check reject: gamma={gamma_pc:.3f} > "
                f"r_max1+r_max2={r_max1 + r_max2:.3f}"
            )
            return False
        return True

    @staticmethod
    def _surface_accessible_mask(
        coords: np.ndarray,
        radii: np.ndarray,
        cm: np.ndarray,
        r_max: float,
        min_exposure: float | None = None,
    ) -> np.ndarray:
        """Compute surface accessibility mask for each monomer."""
        n = coords.shape[0]
        if n <= 1:
            return np.ones(n, dtype=bool)
        if min_exposure is None:
            min_exposure = float(getattr(config, "CCA_SSA_MIN_EXPOSURE", 0.3))
        dist_to_cm = np.linalg.norm(coords - cm[np.newaxis, :], axis=1)
        radial_fraction = dist_to_cm / max(r_max, 1.0e-12)
        mean_r = float(np.mean(radii))
        contact_dist_sq = (2.5 * mean_r) ** 2
        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        d_sq = np.sum(diffs * diffs, axis=2)
        neighbor_count = np.sum((d_sq < contact_dist_sq) & (d_sq > 0), axis=1)
        max_neighbors = max(n - 1, 1)
        isolation_fraction = 1.0 - neighbor_count / max_neighbors
        exposure = 0.5 * radial_fraction + 0.5 * isolation_fraction
        return exposure >= min_exposure

    def _perform_cca_sticking_with_expansion(
        self,
        cluster_idx1: int,
        cluster_idx2: int,
        cluster_props_cache: dict | None = None,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """Wrapper that adds gamma expansion and pair feasibility filtering.

        Wraps _perform_cca_sticking() with:
        1. Bounding volume pre-check (opt-in)
        2. SSA candidate filtering (opt-in)
        3. Incremental gamma expansion on failure (opt-in)
        """
        coords1_in, radii1_in = self._get_cluster_data(cluster_idx1)
        coords2_in, radii2_in = self._get_cluster_data(cluster_idx2)
        n1 = coords1_in.shape[0]
        n2 = coords2_in.shape[0]
        if n1 == 0 or n2 == 0:
            logger.error(
                f"Cannot stick empty cluster(s): idx1({n1} particles), idx2({n2} particles)"
            )
            return None

        _t0 = perf_counter() if config.PROFILE_TIMING else 0.0
        if (
            cluster_props_cache is not None
            and cluster_idx1 in cluster_props_cache
            and cluster_idx2 in cluster_props_cache
        ):
            _p1 = cluster_props_cache[cluster_idx1]
            _p2 = cluster_props_cache[cluster_idx2]
            m1, rg1, cm1, r_max1 = _p1[0], _p1[1], _p1[2], _p1[3]
            m2, rg2, cm2, r_max2 = _p2[0], _p2[1], _p2[2], _p2[3]
        else:
            m1, rg1, cm1, r_max1 = utils.calculate_cluster_properties(
                coords1_in, radii1_in, self.df, self.kf
            )
            m2, rg2, cm2, r_max2 = utils.calculate_cluster_properties(
                coords2_in, radii2_in, self.df, self.kf
            )
        if config.PROFILE_TIMING:
            self._t_cluster_props += perf_counter() - _t0

        # --- Pair Feasibility Pre-Filter ---
        pair_filter = str(
            getattr(config, "CCA_PAIR_FEASIBILITY_FILTER", "none")
        ).lower()
        if pair_filter == "bounding_volume":
            props1_bv = (m1, rg1, cm1, r_max1, radii1_in)
            props2_bv = (m2, rg2, cm2, r_max2, radii2_in)
            gamma_real_bv, gamma_pc_bv = self._calculate_cca_gamma(props1_bv, props2_bv)
            if not self._bounding_volume_precheck(
                gamma_pc_bv, r_max1, r_max2, gamma_real_bv
            ):
                self._bv_filter_rejects += 1
                logger.info(
                    f"CCA pair ({cluster_idx1}, {cluster_idx2}): "
                    f"Rejected by BV pre-check (gamma={gamma_pc_bv:.3f})"
                )
                return None

        # --- FFT Docking Method (opt-in) ---
        sticking_method = str(
            getattr(config, "CCA_STICKING_METHOD", "fibonacci")
        ).lower()
        if sticking_method == "fft_docking":
            props1_fft = (m1, rg1, cm1, r_max1, radii1_in)
            props2_fft = (m2, rg2, cm2, r_max2, radii2_in)
            gamma_real_fft, gamma_pc_fft = self._calculate_cca_gamma(
                props1_fft, props2_fft
            )
            if (
                hasattr(self, "_gamma_pc_override")
                and self._gamma_pc_override is not None
            ):
                gamma_pc_fft = self._gamma_pc_override
                gamma_real_fft = self._gamma_real_override
            self._fft_docking_attempts += 1
            fft_result = fft_dock_sticking(
                coords1=coords1_in,
                radii1=radii1_in,
                coords2=coords2_in,
                radii2=radii2_in,
                cm1=cm1,
                cm2=cm2,
                gamma_pc=gamma_pc_fft,
                gamma_real=gamma_real_fft,
                tol_ov=self.tol_ov,
                grid_size=int(getattr(config, "CCA_FFT_GRID_SIZE", 64)),
                num_rotations=int(getattr(config, "CCA_FFT_NUM_ROTATIONS", 70)),
                top_k_peaks=int(getattr(config, "CCA_FFT_TOP_K_PEAKS", 10)),
                gamma_tolerance=float(getattr(config, "CCA_FFT_GAMMA_TOLERANCE", 0.10)),
                min_peak_distance=int(getattr(config, "CCA_FFT_MIN_PEAK_DISTANCE", 3)),
            )
            if fft_result is not None:
                self._fft_docking_successes += 1
                logger.info(
                    f"CCA FFT docking SUCCESS for pair ({cluster_idx1}, {cluster_idx2})"
                )
                return fft_result
            logger.info(
                f"CCA FFT docking FAILED for pair "
                f"({cluster_idx1}, {cluster_idx2}), falling back to gamma expansion"
            )

        # --- First attempt: fibonacci method ---
        result = self._perform_cca_sticking(
            cluster_idx1, cluster_idx2, cluster_props_cache
        )
        if result is not None:
            return result

        # --- Gamma Expansion ---
        gamma_expansion_enabled = bool(
            getattr(config, "CCA_GAMMA_EXPANSION_ENABLED", False)
        )
        if not gamma_expansion_enabled:
            return None

        gamma_expansion_step = float(getattr(config, "CCA_GAMMA_EXPANSION_STEP", 0.02))
        gamma_expansion_max_factor = float(
            getattr(config, "CCA_GAMMA_EXPANSION_MAX_FACTOR", 1.05)
        )
        gamma_expansion_mass_exponent = float(
            getattr(config, "CCA_GAMMA_EXPANSION_MASS_EXPONENT", -0.75)
        )
        gamma_expansion_max_attempts = int(
            getattr(config, "CCA_GAMMA_EXPANSION_MAX_ATTEMPTS", 3)
        )
        n_total = n1 + n2
        self._gamma_expansion_hits += 1

        props1 = (m1, rg1, cm1, r_max1, radii1_in)
        props2 = (m2, rg2, cm2, r_max2, radii2_in)
        _, gamma_pc_original = self._calculate_cca_gamma(props1, props2)

        for attempt in range(1, gamma_expansion_max_attempts + 1):
            expansion_delta = (
                gamma_expansion_step
                * (n_total**gamma_expansion_mass_exponent)
                * attempt
            )
            gamma_pc_expanded = gamma_pc_original * (1.0 + expansion_delta)

            if gamma_pc_expanded > gamma_pc_original * gamma_expansion_max_factor:
                logger.info(
                    f"CCA gamma expansion hit max factor "
                    f"{gamma_expansion_max_factor:.3f} for clusters "
                    f"{cluster_idx1}, {cluster_idx2}. Giving up."
                )
                return None

            # Recompute gamma from fractal scaling law for physical consistency
            rg3_exp = utils.calculate_rg(
                np.concatenate((radii1_in, radii2_in)),
                n_total,
                self.df,
                self.kf,
            )
            m1_h, m2_h = float(n1), float(n2)
            m3_h = float(n_total)
            term1 = (m3_h**2) * (rg3_exp**2)
            term2 = m3_h * (m1_h * rg1**2 + m2_h * rg2**2)
            denom = m1_h * m2_h
            radicand = term1 - term2

            gamma_real_exp = (denom > 0) and (radicand >= 0)
            if gamma_real_exp:
                gamma_pc_rec = float(np.sqrt(radicand / denom))
                gamma_pc = min(
                    gamma_pc_expanded, gamma_pc_rec * gamma_expansion_max_factor
                )
                gamma_pc = max(gamma_pc, gamma_pc_original)
            else:
                gamma_pc = gamma_pc_expanded
            gamma_real = True

            self._gamma_expansion_total_steps += 1
            logger.info(
                f"CCA gamma expansion ({cluster_idx1},{cluster_idx2}): "
                f"attempt {attempt}/{gamma_expansion_max_attempts}, "
                f"gamma {gamma_pc_original:.4f} -> {gamma_pc:.4f} "
                f"(factor={gamma_pc / gamma_pc_original:.4f})"
            )

            # Override gamma via temp attribute
            self._gamma_pc_override = gamma_pc
            self._gamma_real_override = gamma_real
            try:
                result = self._perform_cca_sticking(
                    cluster_idx1, cluster_idx2, cluster_props_cache
                )
            finally:
                self._gamma_pc_override = None
                self._gamma_real_override = None

            if result is not None:
                self._gamma_expansion_successes += 1
                return result

        logger.warning(
            f"CCA sticking failed for clusters {cluster_idx1}, {cluster_idx2} "
            f"after {gamma_expansion_max_attempts} gamma expansions."
        )
        return None

    def _perform_cca_sticking(
        self,
        cluster_idx1: int,
        cluster_idx2: int,
        cluster_props_cache: dict | None = None,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Manages the process of sticking two clusters (idx1, idx2).
        Corresponds to the Fortran CCA subroutine.

        Args:
            cluster_idx1, cluster_idx2: Cluster indices to stick.
            cluster_props_cache: Optional dict from _generate_pairs mapping cluster
                index to (m, rg, cm, r_max, radii) tuples. When provided, avoids
                redundant calculate_cluster_properties calls (PyFracVAL-58z).

        Returns:
            Tuple(combined_coords, combined_radii) or None if sticking fails.
        """
        # --- Get Data for the two clusters ---
        coords1_in, radii1_in = self._get_cluster_data(cluster_idx1)
        coords2_in, radii2_in = self._get_cluster_data(cluster_idx2)
        n1 = coords1_in.shape[0]
        n2 = coords2_in.shape[0]

        if n1 == 0 or n2 == 0:
            logger.error(
                f"Cannot stick empty cluster(s): idx1({n1} particles), idx2({n2} particles)"
            )
            return None  # Cannot stick empty clusters

        # --- Calculate Properties and Gamma ---
        # FIX (PyFracVAL-58z): use cached props from _generate_pairs when available
        _t0 = perf_counter() if config.PROFILE_TIMING else 0.0
        if (
            cluster_props_cache is not None
            and cluster_idx1 in cluster_props_cache
            and cluster_idx2 in cluster_props_cache
        ):
            _p1 = cluster_props_cache[cluster_idx1]
            _p2 = cluster_props_cache[cluster_idx2]
            m1, rg1, cm1, r_max1 = _p1[0], _p1[1], _p1[2], _p1[3]
            m2, rg2, cm2, r_max2 = _p2[0], _p2[1], _p2[2], _p2[3]
        else:
            m1, rg1, cm1, r_max1 = utils.calculate_cluster_properties(
                coords1_in, radii1_in, self.df, self.kf
            )
            m2, rg2, cm2, r_max2 = utils.calculate_cluster_properties(
                coords2_in, radii2_in, self.df, self.kf
            )
        if config.PROFILE_TIMING:
            self._t_cluster_props += perf_counter() - _t0
        props1 = (m1, rg1, cm1, r_max1, radii1_in)
        props2 = (m2, rg2, cm2, r_max2, radii2_in)
        gamma_real, gamma_pc = self._calculate_cca_gamma(props1, props2)

        # Check for gamma override from expansion wrapper
        if hasattr(self, "_gamma_pc_override") and self._gamma_pc_override is not None:
            gamma_pc = self._gamma_pc_override
            gamma_real = self._gamma_real_override

        # --- Generate Candidate List ---
        _t0 = perf_counter() if config.PROFILE_TIMING else 0.0
        list_matrix = self._cca_select_candidates(
            coords1_in, radii1_in, cm1, coords2_in, radii2_in, cm2, gamma_pc, gamma_real
        )

        # Apply SSA filter if enabled
        pair_filter = str(
            getattr(config, "CCA_PAIR_FEASIBILITY_FILTER", "none")
        ).lower()
        if pair_filter == "ssa":
            ssa_mask_1 = self._surface_accessible_mask(
                coords1_in, radii1_in, cm1, r_max1
            )
            ssa_mask_2 = self._surface_accessible_mask(
                coords2_in, radii2_in, cm2, r_max2
            )
            original_count = int(np.sum(list_matrix > 0))
            for i_loc in range(list_matrix.shape[0]):
                for j_loc in range(list_matrix.shape[1]):
                    if list_matrix[i_loc, j_loc] > 0 and (
                        not ssa_mask_1[i_loc] or not ssa_mask_2[j_loc]
                    ):
                        list_matrix[i_loc, j_loc] = 0
            filtered_count = int(np.sum(list_matrix > 0))
            if filtered_count < original_count:
                self._ssa_filter_rejects += original_count - filtered_count
                logger.debug(
                    f"SSA filter: {original_count} -> {filtered_count} candidates "
                    f"for pair ({cluster_idx1}, {cluster_idx2})"
                )

        leaf_mask_1 = self._leaf_mask_for_cluster(coords1_in, radii1_in)
        leaf_mask_2 = self._leaf_mask_for_cluster(coords2_in, radii2_in)
        if config.PROFILE_TIMING:
            self._t_select_candidates += perf_counter() - _t0

        if np.sum(list_matrix) == 0:
            logger.warning(
                f"No initial candidates found for sticking clusters {cluster_idx1} and {cluster_idx2}. Gamma_real={gamma_real}"
            )
            # Can this happen if _generate_pairs said they *could* pair? Maybe due to Rmax vs Gamma criteria?
            # Or if gamma_real is false.
            return None  # Sticking fails if no candidates

        # --- Sticking Attempt Loop ---
        # FIX (PyFracVAL-2yb): Build candidate list ONCE and shuffle it.
        # Previously _cca_pick_candidate_pair rebuilt the full list on every attempt
        # → O((n1×n2)²) total work. Now O(n1×n2) to build once, O(1) per attempt.
        _candidate_indices = np.argwhere(list_matrix > 0)
        self._rng.shuffle(_candidate_indices)

        candidate_policy = str(config.CCA_CANDIDATE_POLICY).lower()

        # Optional soft leaf-priority policy: LL first, then LN, then NN.
        if candidate_policy == "leaf_soft":
            ll: list[np.ndarray] = []
            ln: list[np.ndarray] = []
            nn: list[np.ndarray] = []
            for pair in _candidate_indices:
                i = int(pair[0])
                j = int(pair[1])
                cls = self._candidate_leaf_class(
                    bool(leaf_mask_1[i]), bool(leaf_mask_2[j])
                )
                if cls == "LL":
                    ll.append(pair)
                elif cls == "LN":
                    ln.append(pair)
                else:
                    nn.append(pair)
            _candidate_indices = np.array(ll + ln + nn, dtype=int)
        elif candidate_policy in {"leaf_score", "leaf_hybrid"}:
            ll: list[np.ndarray] = []
            ln: list[np.ndarray] = []
            nn: list[np.ndarray] = []
            for pair in _candidate_indices:
                i = int(pair[0])
                j = int(pair[1])
                cls = self._candidate_leaf_class(
                    bool(leaf_mask_1[i]), bool(leaf_mask_2[j])
                )
                if cls == "LL":
                    ll.append(pair)
                elif cls == "LN":
                    ln.append(pair)
                else:
                    nn.append(pair)

            topk = int(getattr(config, "CCA_SCORE_TOPK_PER_CLASS", 0))

            def _score_and_sort(pairs: list[np.ndarray], cls: str) -> list[np.ndarray]:
                if not pairs:
                    return []
                n_score = len(pairs) if topk <= 0 else min(topk, len(pairs))
                scored: list[tuple[float, np.ndarray]] = []
                for pair in pairs[:n_score]:
                    i = int(pair[0])
                    j = int(pair[1])
                    score = self._candidate_score(
                        coords1_in,
                        radii1_in,
                        cm1,
                        i,
                        coords2_in,
                        radii2_in,
                        cm2,
                        j,
                        float(gamma_pc),
                        cls,
                    )
                    scored.append((score, pair))
                scored.sort(key=lambda x: x[0], reverse=True)
                scored_pairs = [p for _, p in scored]
                return scored_pairs + pairs[n_score:]

            if candidate_policy == "leaf_score":
                # Score order globally, optionally only top-k per class for speed.
                merged = (
                    _score_and_sort(ll, "LL")
                    + _score_and_sort(ln, "LN")
                    + _score_and_sort(nn, "NN")
                )
                _candidate_indices = np.array(merged, dtype=int)
            else:
                # Hybrid: keep leaf priority class order, score only within each class.
                merged = (
                    _score_and_sort(ll, "LL")
                    + _score_and_sort(ln, "LN")
                    + _score_and_sort(nn, "NN")
                )
                _candidate_indices = np.array(merged, dtype=int)

        sticking_successful = False
        final_coords1 = None
        final_coords2 = None

        attempts_tried = 0
        for attempt, (cand1_idx, cand2_idx) in enumerate(_candidate_indices):
            attempts_tried = attempt + 1
            leaf1 = bool(leaf_mask_1[cand1_idx])
            leaf2 = bool(leaf_mask_2[cand2_idx])
            cand_cls = self._candidate_leaf_class(leaf1, leaf2)
            self._record_candidate_attempt(leaf1, leaf2)
            cand_score = self._candidate_score(
                coords1_in,
                radii1_in,
                cm1,
                int(cand1_idx),
                coords2_in,
                radii2_in,
                cm2,
                int(cand2_idx),
                float(gamma_pc),
                cand_cls,
            )
            self._record_candidate_score_attempt(cand_score)
            # logger.info(f"  CCA Stick ({cluster_idx1},{cluster_idx2}): Trying pair ({cand1_idx}, {cand2_idx}). Attempt {attempt+1}/{len(_candidate_indices)}")

            # Perform initial sticking placement
            _t0 = perf_counter() if config.PROFILE_TIMING else 0.0
            stick_results = self._cca_sticking_v1(
                (coords1_in, radii1_in, cm1),
                (coords2_in, radii2_in, cm2),
                cand1_idx,
                cand2_idx,
                gamma_pc,
                gamma_real,
            )
            if config.PROFILE_TIMING:
                self._t_sticking_v1 += perf_counter() - _t0
            coords1_stick, coords2_stick, cm2_stick, theta_a, vec_0, i_vec, j_vec = (
                stick_results
            )

            if (
                coords1_stick is None or coords2_stick is None or cm2_stick is None
            ):  # Initial sticking failed for this pair
                # logger.info(f"    Initial sticking failed for pair ({cand1_idx}, {cand2_idx}).")
                continue  # Try next pair

            # Check initial overlap
            # cov_max = self._cca_overlap_check(
            #     coords1_stick, radii1_in, coords2_stick, radii2_in
            # )
            # Phase 3B: Auto-dispatch to parallel overlap for large N
            use_incremental = (
                config.USE_CCA_INCREMENTAL_OVERLAP and not config.USE_BATCH_ROTATION
            )
            full_sync_period = max(config.CCA_INCREMENTAL_FULL_SYNC_PERIOD, 1)

            active_collisions: set[int] = set()
            n2_local = radii2_in.shape[0]

            _t0 = perf_counter() if config.PROFILE_TIMING else 0.0
            if use_incremental:
                cov_max = self._full_overlap_check(
                    coords1_stick, radii1_in, coords2_stick, radii2_in
                )
                active_collisions = set()
            else:
                cov_max = utils.calculate_max_overlap_cca_auto(
                    coords1_stick,
                    radii1_in,
                    coords2_stick,
                    radii2_in,
                    tolerance=self.tol_ov,
                )
            if config.PROFILE_TIMING:
                self._t_overlap_check += perf_counter() - _t0
                self._n_overlap_calls += 1

            # Rotation attempts
            intento = 0
            retry_mode_cfg = str(
                getattr(config, "CCA_RETRY_ROTATION_MODE", "single")
            ).lower()
            if retry_mode_cfg in {"coarse_grid", "coarse_to_fine"}:
                max_rotations = int(
                    max(1, getattr(config, "CCA_COARSE_SWEEP_STEPS", 10))
                    * max(1, getattr(config, "CCA_COARSE_SPIN_ANCHOR_STEPS", 6))
                    * max(1, getattr(config, "CCA_COARSE_SPIN_MOVING_STEPS", 6))
                )
            else:
                max_rotations = 360  # From Fortran
            current_coords2 = coords2_stick.copy()  # Keep track of rotated coords2
            coords1_base = coords1_stick.copy()
            coords2_base = coords2_stick.copy()
            axis_anchor = self._normalize_axis(
                coords1_base[int(cand1_idx)] - cm1,
                fallback=np.array([i_vec[0], i_vec[1], i_vec[2]], dtype=float),
            )
            axis_moving = self._normalize_axis(
                coords2_base[int(cand2_idx)] - cm2_stick,
                fallback=np.array([i_vec[0], i_vec[1], i_vec[2]], dtype=float),
            )
            adaptive_tol_threshold = min(
                180, max_rotations
            )  # Relax tolerance after this many attempts
            relaxed_tol = 1.0e-5  # Relaxed tolerance (10x more lenient)
            used_adaptive_tol = False
            last_retry_mode = "single"

            # Choose rotation strategy based on configuration
            if config.USE_BATCH_ROTATION:
                # Batch rotation (Phase 3 - experimental, slower for N<1000)
                batch_size = config.ROTATION_BATCH_SIZE
                golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

                while cov_max > self.tol_ov and intento < max_rotations:
                    # Determine batch range
                    batch_start = intento
                    batch_end = min(intento + batch_size, max_rotations)
                    batch_count = batch_end - batch_start

                    if batch_count == 0:
                        break

                    # Generate batch of angles using Fibonacci spiral
                    attempts = np.arange(batch_start, batch_end)
                    angles = 2.0 * config.PI * attempts / golden_ratio

                    # Batch rotate cluster2 for all angles (parallel computation)
                    coords2_batch = utils.batch_rotate_cluster_cca(
                        current_coords2,
                        cm2_stick,
                        cand2_idx,
                        vec_0,
                        i_vec,
                        j_vec,
                        angles,
                    )

                    # Check overlaps for all rotated configurations (parallel)
                    overlaps = utils.batch_check_overlaps_cca(
                        coords1_stick,
                        radii1_in,
                        coords2_batch,
                        radii2_in,
                        self.tol_ov,
                    )

                    # Find first valid configuration (overlap <= tolerance)
                    valid_indices = np.where(overlaps <= self.tol_ov)[0]

                    if len(valid_indices) > 0:
                        # Found valid configuration
                        best_idx = valid_indices[0]
                        intento = batch_start + best_idx + 1
                        current_coords2 = coords2_batch[best_idx]
                        cov_max = overlaps[best_idx]

                        logger.trace(
                            f"    CCA Batch rotation {intento}: Found valid config with overlap={cov_max:.4e}"
                        )  # pyright: ignore
                        break  # Exit rotation loop
                    else:
                        # No valid configuration in this batch
                        # Use best (minimum overlap) from batch
                        best_idx = np.argmin(overlaps)
                        intento = batch_start + best_idx + 1
                        current_coords2 = coords2_batch[best_idx]
                        cov_max = overlaps[best_idx]

                        # Check adaptive tolerance
                        if intento >= adaptive_tol_threshold and cov_max <= relaxed_tol:
                            logger.info(
                                f"  CCA pair ({cand1_idx}, {cand2_idx}): Accepting relaxed tolerance "
                                f"(overlap={cov_max:.4e} <= {relaxed_tol:.4e}) after {intento} rotations."
                            )
                            used_adaptive_tol = True
                            break

                        logger.trace(
                            f"    CCA Batch {batch_start}-{batch_end}: Best overlap={cov_max:.4e} at attempt {intento}"
                        )  # pyright: ignore

                        # Continue to next batch
                        intento = batch_end

                    # Check if we've exceeded max rotations
                    if intento >= max_rotations and cov_max > self.tol_ov:
                        break  # Exit rotation loop for this pair
            else:
                # Sequential rotation with Fibonacci spiral (original algorithm).
                # Each step rotates from the previous position to the next Fibonacci
                # target angle, threading current_coords2 through the loop.
                while cov_max > self.tol_ov and intento < max_rotations:
                    intento += 1
                    _t0 = perf_counter() if config.PROFILE_TIMING else 0.0
                    coords1_rotated, coords2_rotated, retry_mode = (
                        self._apply_retry_rotation_mode(
                            coords1_stick,
                            current_coords2,
                            coords1_base,
                            coords2_base,
                            cm1,
                            cm2_stick,
                            int(cand1_idx),
                            int(cand2_idx),
                            vec_0,
                            i_vec,
                            j_vec,
                            axis_anchor,
                            axis_moving,
                            int(intento),
                        )
                    )
                    self._retry_mode_counts[retry_mode] = (
                        self._retry_mode_counts.get(retry_mode, 0) + 1
                    )
                    last_retry_mode = retry_mode
                    if config.PROFILE_TIMING:
                        self._t_rotation += perf_counter() - _t0
                        self._n_rotation_calls += 1

                    _t0 = perf_counter() if config.PROFILE_TIMING else 0.0
                    if use_incremental:
                        self._active_calls += 1
                        self._active_pairs_checked += len(active_collisions)
                        cov_l1, active_new = self._scan_active_collisions(
                            coords1_rotated,
                            radii1_in,
                            coords2_rotated,
                            radii2_in,
                            active_collisions,
                            n2_local,
                        )
                        active_collisions = active_new
                        if active_collisions:
                            self._active_nonempty_hits += 1
                        cov_max = cov_l1

                        if intento % full_sync_period == 0:
                            self._full_periodic_syncs += 1
                            cov_max = self._full_overlap_check(
                                coords1_rotated,
                                radii1_in,
                                coords2_rotated,
                                radii2_in,
                            )
                            active_collisions = set()
                    else:
                        cov_max = utils.calculate_max_overlap_cca_auto(
                            coords1_rotated,
                            radii1_in,
                            coords2_rotated,
                            radii2_in,
                            tolerance=self.tol_ov,
                        )

                    if config.PROFILE_TIMING:
                        self._t_overlap_check += perf_counter() - _t0
                        self._n_overlap_calls += 1

                    coords1_stick = coords1_rotated
                    current_coords2 = coords2_rotated
                    logger.trace(  # pyright: ignore
                        f"    Rotation {intento} [{retry_mode}]: Overlap = {cov_max:.4e}"
                    )

                    if intento >= adaptive_tol_threshold and cov_max <= relaxed_tol:
                        logger.info(
                            f"  CCA pair ({cand1_idx}, {cand2_idx}): Accepting "
                            f"relaxed tol (overlap={cov_max:.4e}) after {intento} rotations."
                        )
                        used_adaptive_tol = True
                        break

            # Check if overlap is acceptable
            if cov_max <= self.tol_ov or used_adaptive_tol:
                if use_incremental:
                    # Strict final validation with full overlap check before accept.
                    self._full_final_validations += 1
                    final_cov = utils.calculate_max_overlap_cca_auto(
                        coords1_stick,
                        radii1_in,
                        current_coords2,
                        radii2_in,
                        tolerance=self.tol_ov,
                    )
                    if final_cov > self.tol_ov and not used_adaptive_tol:
                        continue
                # logger.info(f"    Pair ({cand1_idx}, {cand2_idx}): Success! Overlap = {cov_max:.4e} after {intento} rotations.")
                sticking_successful = True
                self._record_candidate_success(cand_cls)
                self._record_candidate_score_success(cand_score)
                if intento > 0:
                    self._retry_mode_success_counts[last_retry_mode] = (
                        self._retry_mode_success_counts.get(last_retry_mode, 0) + 1
                    )
                    self._retry_mode_success_attempt_sum[last_retry_mode] = (
                        self._retry_mode_success_attempt_sum.get(last_retry_mode, 0)
                        + int(intento)
                    )
                final_coords1 = coords1_stick  # Cluster 1 might have rotated
                final_coords2 = (
                    current_coords2  # Use the final rotated coords for cluster 2
                )
                break  # Exit candidate pair loop successfully
            # else: continue to the next candidate pair

        # --- End of Sticking Attempt Loop ---

        if (
            sticking_successful
            and final_coords1 is not None
            and final_coords2 is not None
        ):
            # Combine results
            combined_coords = np.vstack((final_coords1, final_coords2))
            combined_radii = np.concatenate((radii1_in, radii2_in))
            return combined_coords, combined_radii
        else:
            logger.warning(
                f"CCA sticking failed for clusters {cluster_idx1} and {cluster_idx2} after trying {attempts_tried} pairs."
            )
            return None  # Failed to find non-overlapping configuration

    def _try_soft_relaxation_sticking(
        self,
        cluster_idx1: int,
        cluster_idx2: int,
        cluster_props_cache: dict | None = None,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """Attempt sticking using soft potential relaxation.

        This is a fallback method when rigid-body docking fails.
        Uses harmonic repulsion potentials and gradient descent to
        find a valid configuration.

        Returns:
            Tuple(combined_coords, combined_radii) or None if failed.
        """
        from . import config

        # Get cluster data
        coords1_in, radii1_in = self._get_cluster_data(cluster_idx1)
        coords2_in, radii2_in = self._get_cluster_data(cluster_idx2)
        n1 = coords1_in.shape[0]
        n2 = coords2_in.shape[0]

        if n1 == 0 or n2 == 0:
            return None

        # Get cluster properties
        if (
            cluster_props_cache is not None
            and cluster_idx1 in cluster_props_cache
            and cluster_idx2 in cluster_props_cache
        ):
            _p1 = cluster_props_cache[cluster_idx1]
            _p2 = cluster_props_cache[cluster_idx2]
            m1, rg1, cm1, r_max1 = _p1[0], _p1[1], _p1[2], _p1[3]
            m2, rg2, cm2, r_max2 = _p2[0], _p2[1], _p2[2], _p2[3]
        else:
            m1, rg1, cm1, r_max1 = utils.calculate_cluster_properties(
                coords1_in, radii1_in, self.df, self.kf
            )
            m2, rg2, cm2, r_max2 = utils.calculate_cluster_properties(
                coords2_in, radii2_in, self.df, self.kf
            )

        # Calculate gamma
        props1 = (m1, rg1, cm1, r_max1, radii1_in)
        props2 = (m2, rg2, cm2, r_max2, radii2_in)
        gamma_real, gamma_pc = self._calculate_cca_gamma(props1, props2)

        # Get candidate particles (use first available or compute)
        # For simplicity, use the particle closest to CM in each cluster
        dists1 = np.linalg.norm(coords1_in - cm1, axis=1)
        candidate_idx1 = int(np.argmin(dists1))
        dists2 = np.linalg.norm(coords2_in - cm2, axis=1)
        candidate_idx2 = int(np.argmin(dists2))

        # Get config parameters
        k_repulsion = float(getattr(config, "CCA_SOFT_RELAXATION_K_REPULSION", 10.0))
        k_gamma = float(getattr(config, "CCA_SOFT_RELAXATION_K_GAMMA", 1.0))
        gamma_tol = float(getattr(config, "CCA_SOFT_RELAXATION_GAMMA_TOLERANCE", 0.05))
        max_iters = int(getattr(config, "CCA_SOFT_RELAXATION_MAX_ITERS", 100))
        learning_rate = float(getattr(config, "CCA_SOFT_RELAXATION_LEARNING_RATE", 0.1))

        try:
            new_coords1, new_coords2, success, info = soft_sticking(
                coords1_in,
                radii1_in,
                coords2_in,
                radii2_in,
                gamma_pc,
                cm1,
                cm2,
                candidate_idx1,
                candidate_idx2,
                k_repulsion=k_repulsion,
                k_gamma=k_gamma,
                gamma_tolerance=gamma_tol,
                max_iters=max_iters,
                learning_rate=learning_rate,
            )

            if success:
                combined_coords = np.vstack((new_coords1, new_coords2))
                combined_radii = np.concatenate((radii1_in, radii2_in))
                logger.debug(
                    f"Soft relaxation converged: iters={info.get('iterations', 0)}, "
                    f"E={info.get('final_energy', 0):.4e}, "
                    f"gamma_err={info.get('gamma_error', 0):.4f}"
                )
                return combined_coords, combined_radii
            else:
                logger.debug(
                    f"Soft relaxation did not converge: "
                    f"max_ov={info.get('final_max_overlap', 1):.4e}, "
                    f"gamma_err={info.get('gamma_error', 0):.4f}"
                )
                return None

        except Exception as e:
            logger.warning(f"Soft relaxation failed with exception: {e}")
            return None

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
                    and getattr(config, "CCA_SOFT_RELAXATION_ENABLED", False)
                    and getattr(config, "CCA_SOFT_RELAXATION_FALLBACK_ONLY", True)
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
                logger.error(f"Exceeding total particle count N during CCA iteration.")
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
        if config.PROFILE_TIMING:
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
            if config.PROFILE_CCA_LEAF_STATS:
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
            if config.PROFILE_CCA_CANDIDATE_SCORE:
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
            if config.PROFILE_CCA_RETRY_MODES and self._retry_mode_counts:
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
