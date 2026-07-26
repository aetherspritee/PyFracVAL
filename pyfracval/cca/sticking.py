"""Rigid-body sticking, rotation, and overlap checking for CCA.

Mixed into :class:`pyfracval.cca.aggregator.CCAggregator` - these methods
assume ``self`` carries the CCAggregator instance state, set up in
``CCAggregator.__init__``.
"""

import logging
import math
from typing import Tuple

import numpy as np

from .. import utils

logger = logging.getLogger(__name__)

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


def _pair_key(i: int, j: int, n2: int) -> int:
    """Pack pair indices (i,j) into a single integer key."""
    return i * n2 + j


def _pair_unpack(key: int, n2: int) -> tuple[int, int]:
    """Unpack integer pair key into (i,j)."""
    return key // n2, key % n2


class _StickingMixin:
    """Rigid-body sticking, rotation, and overlap checking methods."""

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
        mode_cfg = str(self.algorithm_config.cca_retry_rotation_mode).lower()
        if mode_cfg not in {
            "single",
            "alternate",
            "dual_jitter",
            "coarse_grid",
            "coarse_to_fine",
        }:
            mode_cfg = "single"

        if mode_cfg == "coarse_grid":
            sweep_steps = int(max(1, self.algorithm_config.cca_coarse_sweep_steps))
            spin_anchor_steps = int(
                max(1, self.algorithm_config.cca_coarse_spin_anchor_steps)
            )
            spin_moving_steps = int(
                max(1, self.algorithm_config.cca_coarse_spin_moving_steps)
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

            anchor_angle = (2.0 * math.pi * float(anchor_idx)) / float(
                spin_anchor_steps
            )
            moving_angle = (2.0 * math.pi * float(moving_idx)) / float(
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
            sweep_steps = int(max(1, self.algorithm_config.cca_coarse_sweep_steps))
            spin_anchor_steps = int(
                max(1, self.algorithm_config.cca_coarse_spin_anchor_steps)
            )
            spin_moving_steps = int(
                max(1, self.algorithm_config.cca_coarse_spin_moving_steps)
            )
            total = sweep_steps * spin_anchor_steps * spin_moving_steps
            coarse_fraction = float(
                self.algorithm_config.cca_coarse_fine_coarse_fraction
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

                anchor_angle = (2.0 * math.pi * float(anchor_idx)) / float(
                    spin_anchor_steps
                )
                moving_angle = (2.0 * math.pi * float(moving_idx)) / float(
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
            refine_deg = float(max(0.0, self.algorithm_config.cca_coarse_fine_spin_deg))
            refine_rad = np.deg2rad(refine_deg)
            phi = 2.0 * math.pi * float(refine_idx) / float(_GOLDEN_RATIO)
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

        escalate_after = int(max(0, self.algorithm_config.cca_retry_escalate_after))
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
                phi = 2.0 * math.pi * float(intento) / float(_GOLDEN_RATIO)
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
        jitter_interval = int(max(1, self.algorithm_config.cca_dual_jitter_interval))
        if intento % jitter_interval == 0:
            jitter_deg = float(max(0.0, self.algorithm_config.cca_dual_jitter_deg))
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
