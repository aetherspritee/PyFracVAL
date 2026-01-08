"""Implements the Cluster-Cluster Aggregation (CCA) algorithm."""

import logging
import math
from typing import Set, Tuple

import numpy as np

from . import config, utils
from .logs import TRACE_LEVEL_NUM

logger = logging.getLogger(__name__)


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

        # Current state of the simulation
        self.coords = initial_coords.copy()
        self.radii = initial_radii.copy()
        self.i_orden = initial_i_orden.copy()  # Shape (i_t, 3) [start, end, count]
        self.i_t = self.i_orden.shape[0]  # Current number of clusters

        self.not_able_cca = False

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
        CCA_PAIRING_FACTOR = 1.50  # TEST: Start with 50% relaxation
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
                if logger.isEnabledFor(TRACE_LEVEL_NUM):  # TRACE level
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
        return id_agglomerated

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
        selected_pair_idx = np.random.randint(len(available_pairs))
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
                sphere_1, sphere_2
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
            utils.two_sphere_intersection(sphere_A, sphere_B)
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
        """
        Rotates cluster 2 to a new point on the intersection circle.
        Corresponds to CCA_Sticking_process_v1_reintento.

        Uses Fibonacci spiral sampling for optimal angular coverage.

        Args:
            coords2_in: Coordinates of cluster 2
            cm2: Center of mass of cluster 2
            cand2_idx: Index of candidate particle in cluster 2
            vec_0: [x0, y0, z0, r0] - center and radius of intersection circle
            i_vec: First basis vector for the circle plane
            j_vec: Second basis vector for the circle plane
            attempt: Rotation attempt number (0-indexed) for Fibonacci spiral

        Returns:
            tuple: (coords2_rotated, theta_a_new)
        """
        coords2 = coords2_in.copy()  # Work with copy
        n2 = coords2.shape[0]
        x0, y0, z0, r0 = vec_0

        # New angle using Fibonacci spiral for optimal coverage
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        theta_a_new = 2.0 * config.PI * attempt / golden_ratio
        target_p2 = np.zeros(3)
        target_p2[0] = (
            x0
            + r0 * np.cos(theta_a_new) * i_vec[0]
            + r0 * np.sin(theta_a_new) * j_vec[0]
        )
        target_p2[1] = (
            y0
            + r0 * np.cos(theta_a_new) * i_vec[1]
            + r0 * np.sin(theta_a_new) * j_vec[1]
        )
        target_p2[2] = (
            z0
            + r0 * np.cos(theta_a_new) * i_vec[2]
            + r0 * np.sin(theta_a_new) * j_vec[2]
        )

        # Rotate cluster 2 to align cand2_idx with target_p2
        current_p2 = coords2[cand2_idx]
        v1_rot = current_p2 - cm2
        v2_rot = target_p2 - cm2

        norm_v1 = np.linalg.norm(v1_rot)
        norm_v2 = np.linalg.norm(v2_rot)

        rot_axis = np.zeros(3)
        rot_angle = 0.0
        perform_rot = True

        if norm_v1 > 1e-9 and norm_v2 > 1e-9:
            v1_u = v1_rot / norm_v1
            v2_u = v2_rot / norm_v2
            dot_prod = np.dot(v1_u, v2_u)
            if abs(dot_prod) > 1.0 - 1e-9:
                if dot_prod < 0:
                    rot_angle = np.pi
                    if abs(v1_u[0]) < 1e-9 and abs(v1_u[1]) < 1e-9:
                        rot_axis = np.array([1.0, 0.0, 0.0])
                    else:
                        rot_axis = np.array([-v1_u[1], v1_u[0], 0.0])
                else:
                    perform_rot = False
            else:
                # Clamp dot_prod before acos due to potential precision issues
                dot_prod_clamped = np.clip(dot_prod, -1.0, 1.0)
                rot_angle = np.arccos(dot_prod_clamped)
                rot_axis = np.cross(v1_u, v2_u)
        else:
            perform_rot = False

        if perform_rot and np.linalg.norm(rot_axis) > 1e-9 and abs(rot_angle) > 1e-9:
            coords2_rel = coords2 - cm2
            coords2_rel_rotated = utils.rodrigues_rotation(
                coords2_rel, rot_axis, rot_angle
            )
            coords2 = coords2_rel_rotated + cm2
            # CM doesn't change

        return coords2, theta_a_new

    def _perform_cca_sticking(
        self, cluster_idx1: int, cluster_idx2: int
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Manages the process of sticking two clusters (idx1, idx2).
        Corresponds to the Fortran CCA subroutine.

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
        m1, rg1, cm1, r_max1 = utils.calculate_cluster_properties(
            coords1_in, radii1_in, self.df, self.kf
        )
        m2, rg2, cm2, r_max2 = utils.calculate_cluster_properties(
            coords2_in, radii2_in, self.df, self.kf
        )
        props1 = (m1, rg1, cm1, r_max1, radii1_in)
        props2 = (m2, rg2, cm2, r_max2, radii2_in)
        gamma_real, gamma_pc = self._calculate_cca_gamma(props1, props2)

        # --- Generate Candidate List ---
        list_matrix = self._cca_select_candidates(
            coords1_in, radii1_in, cm1, coords2_in, radii2_in, cm2, gamma_pc, gamma_real
        )

        if np.sum(list_matrix) == 0:
            logger.warning(
                f"No initial candidates found for sticking clusters {cluster_idx1} and {cluster_idx2}. Gamma_real={gamma_real}"
            )
            # Can this happen if _generate_pairs said they *could* pair? Maybe due to Rmax vs Gamma criteria?
            # Or if gamma_real is false.
            return None  # Sticking fails if no candidates

        # --- Sticking Attempt Loop ---
        tried_pairs: Set[Tuple[int, int]] = set()
        sticking_successful = False
        final_coords1 = None
        final_coords2 = None
        max_candidate_attempts = n1 * n2  # Try up to all possible pairs

        for attempt in range(max_candidate_attempts):
            cand1_idx, cand2_idx = self._cca_pick_candidate_pair(
                list_matrix, tried_pairs
            )

            if cand1_idx < 0:  # No more available pairs
                # logger.info(f"  CCA Stick ({cluster_idx1},{cluster_idx2}): No more candidate pairs to try.")
                break

            tried_pairs.add((cand1_idx, cand2_idx))
            # logger.info(f"  CCA Stick ({cluster_idx1},{cluster_idx2}): Trying pair ({cand1_idx}, {cand2_idx}). Attempt {attempt+1}/{max_candidate_attempts}")

            # Perform initial sticking placement
            stick_results = self._cca_sticking_v1(
                (coords1_in, radii1_in, cm1),
                (coords2_in, radii2_in, cm2),
                cand1_idx,
                cand2_idx,
                gamma_pc,
                gamma_real,
            )
            coords1_stick, coords2_stick, cm2_stick, theta_a, vec_0, i_vec, j_vec = (
                stick_results
            )

            if coords1_stick is None:  # Initial sticking failed for this pair
                # logger.info(f"    Initial sticking failed for pair ({cand1_idx}, {cand2_idx}).")
                continue  # Try next pair

            # Check initial overlap
            # cov_max = self._cca_overlap_check(
            #     coords1_stick, radii1_in, coords2_stick, radii2_in
            # )
            cov_max = utils.calculate_max_overlap_cca(
                coords1_stick,
                radii1_in,
                coords2_stick,
                radii2_in,
            )
            # logger.info(f"    Pair ({cand1_idx}, {cand2_idx}): Initial overlap = {cov_max:.4e}")

            # Rotation attempts if needed
            intento = 0
            max_rotations = 360  # From Fortran
            current_coords2 = coords2_stick.copy()  # Keep track of rotated coords2

            while cov_max > self.tol_ov and intento < max_rotations:
                intento += 1
                coords2_rotated, theta_a_new = self._cca_reintento(
                    current_coords2,
                    cm2_stick,
                    cand2_idx,  # Use CM after initial translation
                    vec_0,
                    i_vec,
                    j_vec,
                    attempt=intento,
                )
                # Check overlap with the rotated coords2
                # cov_max = self._cca_overlap_check(
                #     coords1_stick, radii1_in, coords2_rotated, radii2_in
                # )
                cov_max = utils.calculate_max_overlap_cca(
                    coords1_stick,
                    radii1_in,
                    coords2_rotated,
                    radii2_in,
                )

                # Update coords for next potential rotation
                current_coords2 = coords2_rotated
                logger.trace(f"    Rotation {intento}: Overlap = {cov_max:.4e}")  # pyright: ignore

                # Fortran logic for picking new *candidate* after 359 rotations is complex.
                # If max rotations fail here, we consider this candidate pair (cand1, cand2) failed.
                if intento >= max_rotations and cov_max > self.tol_ov:
                    # logger.info(f"    Pair ({cand1_idx}, {cand2_idx}): Failed after {max_rotations} rotations.")
                    break  # Exit rotation loop for this pair

            # Check if overlap is acceptable
            if cov_max <= self.tol_ov:
                # logger.info(f"    Pair ({cand1_idx}, {cand2_idx}): Success! Overlap = {cov_max:.4e} after {intento} rotations.")
                sticking_successful = True
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
                f"CCA sticking failed for clusters {cluster_idx1} and {cluster_idx2} after trying {attempt + 1} pairs."
            )
            return None  # Failed to find non-overlapping configuration

    # --------------------------------------------------------------------------
    # Main CCA Iteration Logic
    # --------------------------------------------------------------------------

    def _run_iteration(self) -> bool:
        """Performs one iteration of the CCA process."""
        logger.info(f"--- CCA Iteration Start - Clusters: {self.i_t} ---")

        # Sort clusters by size (optional, matches Fortran)
        # self.i_orden = utils.sort_clusters(self.i_orden) # Sorts by count

        # Generate pairs
        id_agglomerated = self._generate_pairs()
        if id_agglomerated is None or self.not_able_cca:
            logger.error("Failed to generate valid pairs.")
            self.not_able_cca = True
            return False  # Cannot continue

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
                stick_result = self._perform_cca_sticking(k, other)

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
        # Return only the valid part of the arrays corresponding to the final cluster
        final_count = self.i_orden[0, 2]
        return self.coords[:final_count, :], self.radii[:final_count]
