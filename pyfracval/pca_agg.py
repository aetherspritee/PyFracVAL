"""Implements Particle-Cluster Aggregation (PCA) for initial subclusters."""

import logging

import numpy as np

from . import config, utils
from .logs import TRACE_LEVEL_NUM

logger = logging.getLogger(__name__)

FLOATING_POINT_ERROR = 1e-9  # Defined in utils, but maybe locally needed? Use utils.


class PCAggregator:
    """Performs Particle-Cluster Aggregation (PCA).

    Builds a single cluster by sequentially adding individual primary
    particles (monomers) to a growing aggregate, attempting to match
    the target Df and kf via the Gamma_pc calculation at each step.
    Includes overlap checking and rotation to find valid placements.

    Parameters
    ----------
    initial_radii : np.ndarray
        1D array of radii for the primary particles to be aggregated.
    df : float
        Target fractal dimension for the aggregate.
    kf : float
        Target fractal prefactor for the aggregate.
    tol_ov : float
        Maximum allowable overlap fraction between particles.

    Attributes
    ----------
    N : int
        Total number of particles to aggregate.
    initial_mass : np.ndarray
        Calculated initial masses corresponding to `initial_radii`.
    coords : np.ndarray
        Nx3 array storing coordinates of particles as they are placed.
    radii : np.ndarray
        N array storing radii of particles as they are placed.
    mass : np.ndarray
        N array storing masses of particles as they are placed.
    n1 : int
        Number of particles currently in the aggregate.
    m1 : float
        Mass of the current aggregate.
    rg1 : float
        Radius of gyration of the current aggregate.
    cm : np.ndarray
        3D center of mass of the current aggregate.
    r_max : float
        Maximum distance from CM to any particle center in the aggregate.
    not_able_pca : bool
        Flag indicating if the aggregation process failed.
    """

    def __init__(self, initial_radii: np.ndarray, df: float, kf: float, tol_ov: float):
        self.N = len(initial_radii)
        if self.N < 2:
            raise ValueError("PCA requires at least 2 particles.")

        self.initial_radii = initial_radii.copy()
        # Calculate initial mass using utils consistently
        self.initial_mass = utils.calculate_mass(self.initial_radii)

        self.df = df
        self.kf = kf
        self.tol_ov = tol_ov

        # State variables for the growing cluster
        self.coords = np.zeros((self.N, 3), dtype=float)
        self.radii = np.zeros(self.N, dtype=float)
        self.mass = np.zeros(self.N, dtype=float)

        self.n1: int = 0  # Number of particles currently in the aggregate
        self.m1: float = 0.0  # Mass of the current aggregate
        self.rg1: float = 0.0  # Radius of gyration of the current aggregate
        self.cm = np.zeros(3)  # Center of mass of the current aggregate
        self.r_max: float = 0.0  # Max distance from CM in the current aggregate

        self.not_able_pca: bool = False

    def _random_point_sphere(self) -> tuple[float, float]:
        """Generates random angles (theta, phi) for a point on a sphere."""
        u, v = np.random.rand(2)
        # Use constant from config
        theta = 2.0 * config.PI * u
        phi = np.arccos(2.0 * v - 1.0)
        return theta, phi

    def _first_two_monomers(self):
        """Places the first two monomers."""
        if self.N < 2:
            return  # Should be caught by __init__ but safe check

        self.radii[0] = self.initial_radii[0]
        self.radii[1] = self.initial_radii[1]
        self.mass[0] = self.initial_mass[0]
        self.mass[1] = self.initial_mass[1]

        # Place first particle at origin
        self.coords[0, :] = 0.0

        # Place second particle touching the first (deterministically on X for consistency)
        # distance = self.radii[0] + self.radii[1]
        # self.coords[1, :] = [distance, 0.0, 0.0]

        # Alternative: random orientation (original Fortran way)
        theta, phi = self._random_point_sphere()
        distance = self.radii[0] + self.radii[1]
        self.coords[1, 0] = self.coords[0, 0] + distance * np.cos(theta) * np.sin(phi)
        self.coords[1, 1] = self.coords[0, 1] + distance * np.sin(theta) * np.sin(phi)
        self.coords[1, 2] = self.coords[0, 2] + distance * np.cos(phi)

        self.n1 = 2
        self.m1 = self.mass[0] + self.mass[1]
        # Use utils for rg calculation
        self.rg1 = utils.calculate_rg(self.radii[: self.n1], self.n1, self.df, self.kf)
        if self.m1 > utils.FLOATING_POINT_ERROR:  # Use utils tolerance
            self.cm = (
                self.coords[0] * self.mass[0] + self.coords[1] * self.mass[1]
            ) / self.m1
        else:
            self.cm = np.mean(self.coords[: self.n1], axis=0)

        # Initial r_max (max distance from CM)
        dist_0_cm = np.linalg.norm(self.coords[0] - self.cm)
        dist_1_cm = np.linalg.norm(self.coords[1] - self.cm)
        self.r_max = float(max(dist_0_cm, dist_1_cm))

    def _gamma_calculation(
        self,
        m2: float,
        rg2: float,
        heuristic: bool = True,
    ) -> tuple[bool, float]:
        """
        Calculates Gamma_pc for adding the next monomer (aggregate 2).
        """
        if heuristic:
            return utils.gamma_calculation(
                self.n1,
                self.rg1,
                self.radii[: self.n1],
                1,
                rg2,
                np.array([self.initial_radii[self.n1]]),
                self.df,
                self.kf,
            )
        else:
            return utils.gamma_calculation(
                self.m1,
                self.rg1,
                self.radii[: self.n1],
                m2,
                rg2,
                np.array([self.initial_radii[self.n1]]),
                self.df,
                self.kf,
            )
        # n1 = self.n1
        # n2 = 1
        # n3 = n1 + n2
        # m1 = self.m1
        # m3 = m1 + m2

        # if heuristic:
        #     m1 = n1
        #     m2 = n2
        #     m3 = n3

        # # Ensure index is valid before accessing initial_radii
        # if n1 >= self.N:
        #     logger.error(
        #         f"Gamma calculation requested for particle index {n1} >= N ({self.N})"
        #     )
        #     return False, 0.0

        # # Radii of particles already in cluster + the next one to be added
        # combined_radii = np.concatenate((self.radii[:n1], [self.initial_radii[n1]]))
        # rg3 = utils.calculate_rg(combined_radii, n3, self.df, self.kf)

        # # Heuristic from Fortran: ensure rg3 is not smaller than rg1
        # # (avoids issues if rg calculation is noisy for small N)
        # if self.rg1 > 0 and rg3 < self.rg1:
        #     logger.info(
        #         f"Gamma calc: Adjusted rg3 from {rg3:.2e} to match rg1 {self.rg1:.2e}"
        #     )
        #     rg3 = self.rg1

        # gamma_pc = 0.0
        # gamma_real = False

        # term1 = (m3**2) * (rg3**2)
        # term2 = m3 * (m1 * self.rg1**2 + m2 * rg2**2)  # rg2 is for monomer
        # denominator = m1 * m2
        # radicand = term1 - term2
        # try:
        #     gamma_pc = np.sqrt(radicand / denominator)
        #     gamma_real = True
        # except (ValueError, ZeroDivisionError, OverflowError) as e:
        #     logger.warning(f"Gamma calculation internal failed: {e}")
        #     logger.warning(
        #         f"Gamma_pc calculation non-real or denominator zero: "
        #         f"n1={n1}, m1={m1:.2e}, rg1={self.rg1:.2e}, "
        #         f"m2={m2:.2e}, rg2={rg2:.2e}, "
        #         f"m3={m3:.2e}, rg3={rg3:.2e} -> "
        #         f"radicand={radicand:.2e}, denominator={denominator:.2e}"
        #     )
        #     gamma_real = False

        # return gamma_real, gamma_pc

    def _select_candidates(
        self, radius_k: float, gamma_pc: float, gamma_real: bool
    ) -> tuple[np.ndarray, float]:
        """
        Generates the list of candidate particles (indices within 0 to n1-1)
        that monomer 'k' could stick to, based on Gamma_pc geometry.
        Returns the candidate indices and Rmax (max distance from CM).
        """
        candidates = []
        r_max_current = 0.0

        # If gamma is not real, sticking based on this criterion is impossible.
        if not gamma_real:
            logger.debug(
                "Gamma_pc not real in PCA candidate selection. No candidates selected."
            )
            return np.array([], dtype=int), self.r_max  # Return current r_max

        # Rmax needs to be tracked based on *all* particles in the cluster
        if self.n1 > 0:
            distances_sq = np.sum((self.coords[: self.n1] - self.cm) ** 2, axis=1)
            self.r_max = np.sqrt(np.max(distances_sq)) if distances_sq.size > 0 else 0.0

        logger.debug(
            f"  _select_candidates: Checking N1={self.n1} particles against Gamma_pc={gamma_pc:.4f}, R_k={radius_k:.4f}"
        )
        for i in range(self.n1):  # Iterate through particles 0 to n1-1
            dist_sq = np.sum((self.coords[i] - self.cm) ** 2)
            dist = np.sqrt(dist_sq)
            # r_max_current = max(r_max_current, dist) # Rmax updated above

            radius_i = self.radii[i]  # Radius of particle 'i' in the cluster

            radius_sum = radius_k + radius_i
            lower_dist_bound = gamma_pc - radius_sum
            upper_dist_bound = gamma_pc + radius_sum

            # Fortran conditions (translated):
            # 1) (R_k + R_i) <= Gamma_pc
            radius_sum_check = radius_sum <= gamma_pc + utils.FLOATING_POINT_ERROR

            # 2) dist > (Gamma_pc - R_k - R_i)
            lower_bound_check = dist > lower_dist_bound - utils.FLOATING_POINT_ERROR

            # 3) dist <= (Gamma_pc + R_k + R_i)
            upper_bound_check = dist <= upper_dist_bound + utils.FLOATING_POINT_ERROR

            logger.debug(
                f"    Cand i={i}: Dist={dist:.4f}, R_i={radius_i:.4f} | "
                f"Cond1 (Rk+Ri <= G): {radius_sum:.4f} <= {gamma_pc:.4f}? -> {radius_sum_check} | "
                f"Cond2 (Dist > G-Rk-Ri): {dist:.4f} > {lower_dist_bound:.4f}? -> {lower_bound_check} | "
                f"Cond3 (Dist <= G+Rk+Ri): {dist:.4f} <= {upper_dist_bound:.4f}? -> {upper_bound_check}"
            )

            if radius_sum_check and lower_bound_check and upper_bound_check:
                candidates.append(i)
                logger.debug(f"      -> Candidate {i} ADDED.")

        logger.debug(
            f"PCA selecting candidates for radius {radius_k:.2f} (gamma={gamma_pc:.3f}): Found {len(candidates)} candidates from {self.n1} particles."
        )
        return np.array(candidates, dtype=int), self.r_max  # Return updated r_max

    def _search_and_select_candidate(
        self, k: int, considered_indices: list[int], force_swap: bool = False
    ) -> tuple[int, float, float, bool, float, np.ndarray]:
        """
        Handles the complex logic of selecting a candidate, potentially swapping
        monomer 'k' with another if the initial attempt yields no candidates.
        Corresponds roughly to the loop calling `Search_list` and `Random_select_list`.

        Args:
            k: The index of the particle being added to the aggregate.
            considered_indices: List of particle indices already successfully placed.
            force_swap: If True, forces a particle swap even if candidates exist.
                       This is needed when all candidates fail the overlap check.

        Returns:
            tuple: (selected_idx, m2, rg2, gamma_real, gamma_pc, candidate_list)
                   Returns -1 for selected_idx if no candidate found after all attempts.
                   candidate_list is the list of indices (0 to n1-1) found for the *final* particle k.
        """
        available_monomers = list(range(k, self.N))  # Indices of unprocessed monomers
        tried_swaps = {k}  # Monomers tried *at position k*

        while True:
            # --- Try with current monomer k ---
            current_k_idx = k  # The actual index in the original list being processed
            current_k_radius = self.initial_radii[current_k_idx]
            current_k_mass = self.initial_mass[current_k_idx]
            # Rg of a single sphere = sqrt(3/5) * R => sqrt(0.6) * R
            current_k_rg = np.sqrt(0.6) * current_k_radius

            gamma_real, gamma_pc = self._gamma_calculation(current_k_mass, current_k_rg)
            logger.debug(
                f"PCA search k={k}: Radius={current_k_radius:.2f}, Gamma_real={gamma_real}, Gamma_pc={gamma_pc:.4f}"
            )

            candidates = np.array([], dtype=int)
            if gamma_real:
                # Rmax is updated inside _select_candidates
                candidates, self.r_max = self._select_candidates(
                    current_k_radius, gamma_pc, gamma_real
                )
                logger.debug(
                    f"PCA search k={k}: Found {len(candidates)} candidates: {candidates}"
                )

            if len(candidates) > 0 and not force_swap:
                # Select one candidate randomly (will be used as starting point in run loop)
                idx_in_candidates = np.random.randint(len(candidates))
                selected_initial_candidate = candidates[idx_in_candidates]
                logger.debug(
                    f"PCA search k={k}: Initial candidate {selected_initial_candidate} selected from {len(candidates)} options."
                )
                # Return the *full list* of candidates found for this k
                return (
                    selected_initial_candidate,
                    current_k_mass,
                    current_k_rg,
                    gamma_real,
                    gamma_pc,
                    candidates,  # Return the list of all candidates
                )
            else:
                # Need to swap: either no candidates OR force_swap=True
                if force_swap:
                    logger.debug(
                        f"PCA search k={k}: force_swap=True - will swap particle even though {len(candidates)} candidates exist."
                    )
                else:
                    logger.debug(
                        f"PCA search k={k}: No candidates found or gamma not real. Looking for swap."
                    )
                # --- No candidates: Try swapping k with an untried, available monomer ---
                # Find monomers eligible for swapping (not k itself, not already tried at pos k,
                # and not already successfully placed in the aggregate)
                eligible_for_swap = [
                    idx
                    for idx in available_monomers
                    if idx not in tried_swaps and idx not in considered_indices
                ]

                if not eligible_for_swap:
                    # No more monomers to swap with
                    logger.warning(
                        f"PCA k={k}: No candidates found and no more available monomers to swap with."
                    )
                    return (
                        -1,  # Indicate failure
                        current_k_mass,
                        current_k_rg,
                        gamma_real,
                        gamma_pc,
                        np.array([], dtype=int),  # Empty candidate list
                    )

                # Select a random monomer to swap with k
                swap_idx_in_eligible = np.random.randint(len(eligible_for_swap))
                swap_target_original_idx = eligible_for_swap[swap_idx_in_eligible]

                # Store original values for logging
                radius_before = self.initial_radii[k]
                radius_after = self.initial_radii[swap_target_original_idx]

                logger.info(
                    f"  PCA k={k}: SWAP - Particle radius {radius_before:.2f} → {radius_after:.2f} "
                    f"(swapping with index {swap_target_original_idx})"
                )

                # Perform the swap in the initial_radii and initial_mass arrays
                # Swap particle at index k with particle at swap_target_original_idx
                self.initial_radii[k], self.initial_radii[swap_target_original_idx] = (
                    self.initial_radii[swap_target_original_idx],
                    self.initial_radii[k],
                )
                self.initial_mass[k], self.initial_mass[swap_target_original_idx] = (
                    self.initial_mass[swap_target_original_idx],
                    self.initial_mass[k],
                )
                # Note: The particle originally at k is now at swap_target_original_idx
                # The particle originally at swap_target_original_idx is now at k

                # Mark the monomer *originally* at swap_target_original_idx as having been tried *at position k*
                tried_swaps.add(swap_target_original_idx)

                # Loop continues, recalculating gamma/candidates with the new monomer now at index k
                logger.debug(
                    f"PCA search k={k}: Swapped monomer from index {swap_target_original_idx} into position {k}. Retrying gamma/candidate search."
                )
                # The state of self.initial_radii/mass at index k has changed, restart loop

    def _sticking_process(
        self, k: int, selected_idx: int, gamma_pc: float
    ) -> tuple[np.ndarray | None, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the geometric parameters for placing monomer k based on
        intersection of two spheres.
        Sphere 1: Center = selected particle coords, Radius = R_sel + R_k
        Sphere 2: Center = CM of aggregate, Radius = Gamma_pc

        Returns:
            tuple: (coord_k_initial, theta_a, vec_0, i_vec, j_vec) or (None, ...) if intersection fails.
                   coord_k_initial is *one* point on the intersection circle.
                   Other return values define the circle for rotation (_reintento).
        """
        if selected_idx < 0 or selected_idx >= self.n1:
            logger.error(
                f"Sticking process called with invalid selected_idx={selected_idx} (n1={self.n1})"
            )
            return None, 0.0, np.zeros(4), np.zeros(3), np.zeros(3)
        if k < 0 or k >= self.N:
            logger.error(f"Sticking process called with invalid k={k} (N={self.N})")
            return None, 0.0, np.zeros(4), np.zeros(3), np.zeros(3)

        coord_sel = self.coords[selected_idx]
        radius_sel = self.radii[selected_idx]
        # Use initial radius of particle k before it's placed
        radius_k = self.initial_radii[k]

        # Define the two spheres for intersection
        sphere1_center = coord_sel
        sphere1_radius = radius_sel + radius_k
        sphere1 = np.concatenate((sphere1_center, [sphere1_radius]))

        sphere2_center = self.cm  # Use current aggregate CM
        sphere2_radius = gamma_pc
        sphere2 = np.concatenate((sphere2_center, [sphere2_radius]))

        intersection_valid = False
        try:
            # This utility finds the circle and returns *one* random point on it
            x_k, y_k, z_k, theta_a, vec_0, i_vec, j_vec, intersection_valid = (
                utils.two_sphere_intersection(sphere1, sphere2)
            )
        except Exception as e:
            logger.error(
                f"Error during two_sphere_intersection call: {e}", exc_info=True
            )
            intersection_valid = False  # Ensure it's false on exception

        if not intersection_valid:
            logger.warning(
                f"PCA sticking sphere intersection failed for k={k}, sel={selected_idx}."
            )
            # Log details to help diagnose intersection failures
            dist_centers = np.linalg.norm(sphere1_center - sphere2_center)
            radius_sum = sphere1_radius + sphere2_radius
            radius_diff = abs(sphere1_radius - sphere2_radius)
            logger.warning(
                f"  Intersection Fail Details: Center1={sphere1_center}, R1={sphere1_radius:.4f} | "
                f"Center2={sphere2_center}, R2={sphere2_radius:.4f} | "
                f"Dist={dist_centers:.4f}, R1+R2={radius_sum:.4f}, |R1-R2|={radius_diff:.4f}"
            )
            # Check conditions violated by intersection check in utils
            if dist_centers > radius_sum + utils.FLOATING_POINT_ERROR:
                logger.warning("  -> Spheres too far apart.")
            if dist_centers < radius_diff - utils.FLOATING_POINT_ERROR:
                logger.warning("  -> Sphere contained.")
            if (
                dist_centers < utils.FLOATING_POINT_ERROR
                and abs(radius_diff) < utils.FLOATING_POINT_ERROR
            ):
                logger.warning("  -> Spheres coincide.")

            return None, 0.0, np.zeros(4), np.zeros(3), np.zeros(3)  # Indicate failure

        # Return the initial point found and the circle parameters for rotation
        coord_k_initial = np.array([x_k, y_k, z_k])
        return coord_k_initial, theta_a, vec_0, i_vec, j_vec

    # Remove internal overlap check, use utils.calculate_max_overlap_pca
    # def _overlap_check(self, k: int) -> float: ...

    def _reintento(
        self, k: int, vec_0: np.ndarray, i_vec: np.ndarray, j_vec: np.ndarray, attempt: int = 0
    ) -> tuple[np.ndarray, float]:
        """
        Calculates a *new* point on the intersection circle defined by
        vec_0 (center, radius) and basis vectors i_vec, j_vec.
        This is used to rotate monomer k to try and resolve overlaps.

        Uses Fibonacci spiral sampling for optimal angular coverage, which avoids
        redundant angle sampling and provides better geometric exploration than
        pure random sampling.

        Args:
            k: Particle index
            vec_0: [x0, y0, z0, r0] - center and radius of intersection circle
            i_vec: First basis vector for the circle plane
            j_vec: Second basis vector for the circle plane
            attempt: Rotation attempt number (0-indexed) for Fibonacci spiral

        Returns:
             tuple: (coord_k_new, theta_a_new) - the new coordinates and the angle used.
        """
        x0, y0, z0, r0 = vec_0

        # If radius of intersection is near zero (touching point case), rotation is meaningless
        if r0 < utils.FLOATING_POINT_ERROR:
            logger.debug(
                f"Reintento k={k}: Intersection radius near zero, no rotation possible."
            )
            # Return the center of the 'circle' (the touch point)
            return np.array([x0, y0, z0]), 0.0

        # Generate angle using Fibonacci spiral for optimal coverage
        # Golden ratio provides quasi-uniform distribution without repetition
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        theta_a_new = 2.0 * config.PI * attempt / golden_ratio

        # Calculate new position using the circle equation
        coord_k_new = np.zeros(3)
        coord_k_new = (
            np.array([x0, y0, z0])
            + r0 * np.cos(theta_a_new) * i_vec
            + r0 * np.sin(theta_a_new) * j_vec
        )
        # coord_k_new[0] = (
        #     x0
        #     + r0 * np.cos(theta_a_new) * i_vec[0]
        #     + r0 * np.sin(theta_a_new) * j_vec[0]
        # )
        # coord_k_new[1] = (
        #     y0
        #     + r0 * np.cos(theta_a_new) * i_vec[1]
        #     + r0 * np.sin(theta_a_new) * j_vec[1]
        # )
        # coord_k_new[2] = (
        #     z0
        #     + r0 * np.cos(theta_a_new) * i_vec[2]
        #     + r0 * np.sin(theta_a_new) * j_vec[2]
        # )

        return coord_k_new, theta_a_new

    def run(self) -> np.ndarray | None:
        """Run the complete PCA process for all N particles.

        Sequentially adds particles from index 2 to N-1. For each particle k,
        it calculates Gamma_pc, finds potential sticking partners (`candidates`)
        in the existing aggregate (0..k-1), potentially swaps particle k with
        an unused one if no candidates are found initially.

        It then attempts to stick particle k to each candidate partner,
        calculating an initial position based on sphere intersections defined
        by Gamma_pc. If overlap occurs, it rotates particle k around the
        intersection circle (`_reintento`) up to `max_rotations` times.

        If a non-overlapping position is found for any candidate, the particle
        is successfully added, and aggregate properties are updated. If all
        candidates and all rotations fail for a particle k, or if the initial
        search/swap fails, the aggregation stops, `not_able_pca` is set True,
        and None is returned.

        Returns
        -------
        np.ndarray | None
            An Nx4 NumPy array [X, Y, Z, R] of the final aggregate if
            successful, otherwise None.
        """
        if self.N < 2:
            return None
        self._first_two_monomers()
        considered_indices = list(range(self.n1))

        for k in range(self.n1, self.N):
            logger.debug(f"--- PCA Step: Aggregating particle k={k} ---")

            # --- Outer loop to allow re-searching/swapping if all candidates fail overlap ---
            search_attempt = 0
            max_search_attempts = self.N
            sticking_successful = False

            while not sticking_successful and search_attempt < max_search_attempts:
                search_attempt += 1
                logger.debug(f"PCA k={k}: Search/Swap Attempt #{search_attempt}")

                # --- Perform Search/Swap ---
                # Force particle swap on retry attempts (when previous candidates failed)
                force_swap = (search_attempt > 1)
                search_result = self._search_and_select_candidate(
                    k, considered_indices, force_swap=force_swap
                )
                (
                    initial_candidate_idx,
                    m2,
                    rg2,
                    gamma_real,
                    gamma_pc,
                    candidates_list,
                ) = search_result

                # Check if search failed completely (no candidates even after swaps)
                # Don't need len(candidates_list) check here
                if initial_candidate_idx < 0 or not gamma_real:
                    logger.error(
                        f"PCA failed Search/Swap for k={k} (Attempt {search_attempt}). No valid gamma/candidates found even after swaps."
                    )
                    self.not_able_pca = True
                    return None  # Cannot continue if search itself fails

                # Store radius/mass for the current particle at index k
                radius_k_current = self.initial_radii[k]
                mass_k_current = self.initial_mass[k]

                # --- Try Sticking with Found Candidates ---
                if len(candidates_list) == 0:
                    logger.debug(
                        f"PCA k={k}, Attempt {search_attempt}: Search yielded Gamma but no candidates list. Retrying search/swap."
                    )
                    # Force the outer while loop to continue (effectively re-swaps)
                    # No need to do anything else, the while loop condition handles it
                    # This case might happen if _select_candidates fails geometrically
                    # even if gamma was real after a swap.
                    continue  # Go to next iteration of the outer while loop

                candidates_to_try = utils.shuffle_array(candidates_list.copy())
                logger.debug(
                    f"PCA k={k}, Attempt {search_attempt}: Trying {len(candidates_to_try)} candidates: {candidates_to_try}"
                )

                all_candidates_failed_overlap = True  # Assume failure until success

                for current_selected_idx in candidates_to_try:
                    logger.debug(
                        f"PCA k={k}: Trying candidate partner index {current_selected_idx}"
                    )
                    stick_result = self._sticking_process(
                        k, current_selected_idx, gamma_pc
                    )

                    if stick_result is None or stick_result[0] is None:
                        logger.debug(
                            f"  PCA k={k}, cand={current_selected_idx}: Sticking geometry failed."
                        )
                        continue  # Try next candidate

                    coord_k_initial, theta_a, vec_0, i_vec, j_vec = stick_result
                    self.coords[k] = coord_k_initial
                    self.radii[k] = radius_k_current
                    self.mass[k] = mass_k_current

                    cov_max = utils.calculate_max_overlap_pca(
                        self.coords[: self.n1],
                        self.radii[: self.n1],
                        self.coords[k],
                        self.radii[k],
                    )
                    logger.debug(
                        f"  PCA k={k}, cand={current_selected_idx}: Initial overlap = {cov_max:.4e}"
                    )

                    intento = 0
                    max_rotations = 360
                    while cov_max > self.tol_ov and intento < max_rotations:
                        intento += 1
                        coord_k_new, theta_a_new = self._reintento(
                            k, vec_0, i_vec, j_vec, attempt=intento
                        )
                        self.coords[k] = coord_k_new
                        cov_max = utils.calculate_max_overlap_pca(
                            self.coords[: self.n1],
                            self.radii[: self.n1],
                            self.coords[k],
                            self.radii[k],
                        )
                        # Add detailed trace log here if needed (as before)
                        if logger.isEnabledFor(TRACE_LEVEL_NUM):  # TRACE level
                            ov_details = []
                            for idx_agg in range(self.n1):
                                if idx_agg == current_selected_idx:
                                    continue  # Skip self-check with candidate? No needed here.
                                ov_agg = 1 - (
                                    np.linalg.norm(
                                        self.coords[k] - self.coords[idx_agg]
                                    )
                                    / (self.radii[k] + self.radii[idx_agg])
                                )
                                ov_details.append(f"vs{idx_agg}:{ov_agg:.2e}")
                            logger.log(
                                TRACE_LEVEL_NUM,
                                f"    PCA k={k}, cand={current_selected_idx}, Rot {intento}: Overlap = {cov_max:.4e} ({', '.join(ov_details)})",
                            )

                    if cov_max <= self.tol_ov:
                        logger.debug(
                            f"PCA k={k}: Sticking successful with cand {current_selected_idx} after {intento} rotations."
                        )
                        sticking_successful = True  # Set flag for outer loop
                        all_candidates_failed_overlap = (
                            False  # Mark success for this attempt
                        )
                        break  # Exit the 'for current_selected_idx' loop
                    else:
                        logger.debug(
                            f"  PCA k={k}, cand={current_selected_idx}: Failed overlap after {max_rotations} rotations."
                        )
                        # Continue to the next candidate in candidates_to_try

                # --- After trying all candidates for this search attempt ---
                if all_candidates_failed_overlap:
                    logger.warning(
                        f"PCA k={k}, Attempt {search_attempt}: All {len(candidates_to_try)} candidates failed overlap check. Retrying search/swap..."
                    )
                    # Reset temporary placement before potentially swapping particle k
                    self.coords[k] = 0.0
                    self.radii[k] = 0.0
                    self.mass[k] = 0.0
                    # The outer `while not sticking_successful` loop will continue
                # else: sticking_successful is True, outer while loop will exit

            # --- After the outer while loop ---
            if not sticking_successful:
                # This happens if max_search_attempts was reached
                logger.error(
                    f"PCA failed at k={k}. Could not find non-overlapping position "
                    f"after {max_search_attempts} search/swap attempts."
                )
                self.not_able_pca = True
                return None  # Critical failure

            # --- Update aggregate properties (only if sticking was successful) ---
            self.n1 += 1
            m_old = self.m1
            self.m1 += self.mass[k]  # Use mass that was set during successful attempt
            if self.m1 > utils.FLOATING_POINT_ERROR:
                self.cm = (self.cm * m_old + self.coords[k] * self.mass[k]) / self.m1
            else:
                self.cm = np.mean(self.coords[: self.n1], axis=0)
            self.rg1 = utils.calculate_rg(
                self.radii[: self.n1], self.n1, self.df, self.kf
            )
            considered_indices.append(k)
            logger.debug(
                f"--- PCA Step: Successfully added particle k={k}. Aggregate size n1={self.n1} ---"
            )

        # --- End of k loop ---
        # ... (final checks and return as before) ...
        if self.not_able_pca:
            return None
        final_data = np.hstack(
            (self.coords[: self.N], self.radii[: self.N].reshape(-1, 1))
        )
        if np.any(np.isnan(final_data)):
            logger.error("NaN detected in final PCA data.")
            self.not_able_pca = True
            return None
        logger.info(f"PCA run completed successfully for N={self.N} particles.")
        return final_data
