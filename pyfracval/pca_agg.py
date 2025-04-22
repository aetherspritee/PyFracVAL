# pca_agg.py
"""
Implements Particle-Cluster Aggregation (PCA) used for creating initial subclusters.
"""

from typing import List, Optional, Tuple

import config
import numpy as np
import utils  # Import utility functions


class PCAggregator:
    """
    Performs Particle-Cluster Aggregation (PCA) to form a single cluster
    from a given set of primary particles.
    """

    def __init__(self, initial_radii: np.ndarray, df: float, kf: float, tol_ov: float):
        self.N = len(initial_radii)
        if self.N < 2:
            raise ValueError("PCA requires at least 2 particles.")

        self.initial_radii = initial_radii.copy()
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

    def _random_point_sphere(self) -> Tuple[float, float]:
        """Generates random angles (theta, phi) for a point on a sphere."""
        u, v = np.random.rand(2)
        theta = 2.0 * config.PI * u
        phi = np.arccos(2.0 * v - 1.0)
        return theta, phi

    def _first_two_monomers(self):
        """Places the first two monomers."""
        self.radii[0] = self.initial_radii[0]
        self.radii[1] = self.initial_radii[1]
        self.mass[0] = self.initial_mass[0]
        self.mass[1] = self.initial_mass[1]

        # Place first particle at origin
        self.coords[0, :] = 0.0

        # Place second particle touching the first at a random orientation
        theta, phi = self._random_point_sphere()
        distance = self.radii[0] + self.radii[1]
        self.coords[1, 0] = self.coords[0, 0] + distance * np.cos(theta) * np.sin(phi)
        self.coords[1, 1] = self.coords[0, 1] + distance * np.sin(theta) * np.sin(phi)
        self.coords[1, 2] = self.coords[0, 2] + distance * np.cos(phi)

        self.n1 = 2
        self.m1 = self.mass[0] + self.mass[1]
        self.rg1 = utils.calculate_rg(self.radii[:2], self.n1, self.df, self.kf)
        if self.m1 > 1e-12:
            self.cm = (
                self.coords[0] * self.mass[0] + self.coords[1] * self.mass[1]
            ) / self.m1
        else:
            self.cm = np.mean(self.coords[:2], axis=0)

    def _gamma_calculation(self, m2: float, rg2: float) -> Tuple[bool, float]:
        """
        Calculates Gamma_pc for adding the next monomer (aggregate 2).
        Note: Fortran version had slightly different logic using n1, n2, n3 directly
              instead of masses, and used rg3_auxiliar. Let's stick closer
              to the physics-based mass calculation from CCA.
        """
        n2 = 1
        n3 = self.n1 + n2
        m3 = self.m1 + m2
        # Use combined radii up to the current point + the next one
        combined_radii = np.concatenate(
            (self.radii[: self.n1], [self.initial_radii[self.n1]])
        )
        rg3 = utils.calculate_rg(combined_radii, n3, self.df, self.kf)

        rg3_calc = rg3  # Store the calculated rg3
        rg3 = max(self.rg1, rg3_calc)  # Use the larger of rg1 or calculated rg3

        gamma_pc = 0.0
        gamma_real = False

        try:
            term1 = (m3**2) * (rg3**2)
            term2 = m3 * (self.m1 * self.rg1**2 + m2 * rg2**2)  # rg2 for monomer
            denominator = self.m1 * m2

            # Check if radicand is positive and denominator is non-zero
            if term1 > term2 and denominator > 1e-12:
                gamma_pc = np.sqrt((term1 - term2) / denominator)
                gamma_real = True
            # else: gamma_real remains False

        except (ValueError, ZeroDivisionError, OverflowError) as e:
            print(f"Warning: Gamma calculation failed in PCA: {e}")
            gamma_real = False

        return gamma_real, gamma_pc

    def _select_candidates(
        self, k: int, gamma_pc: float, gamma_real: bool
    ) -> Tuple[np.ndarray, float]:
        """
        Generates the list of candidate particles within the current aggregate
        that monomer 'k' could stick to. Similar to Random_select_list in Fortran PCA.
        Returns the candidate list (indices) and Rmax.
        """
        candidates = []
        r_max_current = 0.0
        radius_k = self.initial_radii[k]  # Radius of the incoming particle

        if not gamma_real:
            # If gamma is not real, potentially no sticking is possible based on this criterion.
            # Fortran code returns. What should happen here? Maybe allow sticking based on proximity?
            # For now, return empty list. Needs clarification on desired behavior.
            # Alternative: Stick to the closest particle? Or the one extending furthest?
            print(
                "Warning: Gamma_pc not real in PCA candidate selection. No candidates selected."
            )
            return np.array([], dtype=int), r_max_current

        for i in range(self.n1):  # Iterate through particles already in the cluster
            dist_sq = np.sum((self.coords[i] - self.cm) ** 2)
            dist = np.sqrt(dist_sq)
            r_max_current = max(r_max_current, dist)  # Keep track of Rmax

            radius_i = self.radii[i]  # Radius of particle 'i' in the cluster

            # Fortran condition: (dist > (Gamma_pc - R_k - R_i)) and (dist <= (Gamma_pc + R_k + R_i))
            # AND (R_k + R_i) <= Gamma_pc
            lower_bound = gamma_pc - radius_k - radius_i
            upper_bound = gamma_pc + radius_k + radius_i
            radius_sum_check = (radius_k + radius_i) <= gamma_pc

            # Add tolerance?
            # lower_bound -= 1e-9
            # upper_bound += 1e-9
            # radius_sum_check = (radius_k + radius_i) <= gamma_pc + 1e-9

            if radius_sum_check and (dist > lower_bound) and (dist <= upper_bound):
                candidates.append(i)

        return np.array(candidates, dtype=int), r_max_current

    def _search_and_select_candidate(
        self, k: int, considered_indices: List[int]
    ) -> Tuple[int, float, float, bool, float, np.ndarray]:
        """
        Handles the complex logic of selecting a candidate, potentially swapping
        monomer 'k' with another if the initial attempt yields no candidates.
        Corresponds roughly to the loop calling `Search_list` and `Random_select_list_pick_one`.

        Returns:
            tuple: (selected_idx, m2, rg2, gamma_real, gamma_pc)
                   Returns -1 for selected_idx if no candidate found.
        """
        available_monomers = list(
            range(k, self.N)
        )  # Indices of monomers not yet processed
        tried_swaps = {
            k
        }  # Keep track of which monomers have been tried in position 'k'

        while True:
            # --- Try with current monomer k ---
            current_k_radius = self.initial_radii[k]
            current_k_mass = self.initial_mass[k]
            current_k_rg = np.sqrt(0.6) * current_k_radius  # Rg of a single sphere

            gamma_real, gamma_pc = self._gamma_calculation(current_k_mass, current_k_rg)

            candidates, self.r_max = self._select_candidates(k, gamma_pc, gamma_real)

            if len(candidates) > 0:
                # Select one candidate randomly
                idx_in_candidates = np.random.randint(len(candidates))
                selected_real_idx = candidates[idx_in_candidates]
                # print(f"  PCA k={k}: Found {len(candidates)} candidates. Selected index {selected_real_idx}.")
                return (
                    selected_real_idx,
                    current_k_mass,
                    current_k_rg,
                    gamma_real,
                    gamma_pc,
                    candidates,  # <<< ADD THIS
                )
            else:
                # print(f"  PCA k={k}: No candidates found initially.")
                # --- No candidates: Try swapping k with an untried, available monomer ---
                # Find monomers available for swapping (not k and not already considered/tried)
                eligible_for_swap = [
                    idx
                    for idx in available_monomers
                    if idx not in tried_swaps and idx not in considered_indices
                ]

                if not eligible_for_swap:
                    # No more monomers to swap with
                    print(
                        f"  PCA k={k}: No candidates found and no more monomers to swap."
                    )
                    return (
                        -1,
                        current_k_mass,
                        current_k_rg,
                        gamma_real,
                        gamma_pc,
                        np.array([], dtype=int),  # <<< ADD EMPTY ARRAY
                    )

                # Select a random monomer to swap with k
                swap_idx_in_eligible = np.random.randint(len(eligible_for_swap))
                swap_target_idx = eligible_for_swap[swap_idx_in_eligible]
                # print(f"  PCA k={k}: Swapping with monomer index {swap_target_idx}.")

                # Perform the swap in the initial_radii and initial_mass arrays
                self.initial_radii[k], self.initial_radii[swap_target_idx] = (
                    self.initial_radii[swap_target_idx],
                    self.initial_radii[k],
                )
                self.initial_mass[k], self.initial_mass[swap_target_idx] = (
                    self.initial_mass[swap_target_idx],
                    self.initial_mass[k],
                )

                # Mark the swapped monomer as tried *in position k*
                tried_swaps.add(swap_target_idx)

                # Loop continues, recalculating gamma/candidates with the new monomer at index k

    def _sticking_process(
        self, k: int, selected_idx: int, gamma_pc: float
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Places monomer k based on intersection of two spheres.
        Sphere 1: Center = selected particle, Radius = R_sel + R_k
        Sphere 2: Center = CM of aggregate, Radius = Gamma_pc

        Returns:
            Tuple: (coord_k, theta_a, vec_0, i_vec, j_vec) or None if intersection fails.
        """
        coord_sel = self.coords[selected_idx]
        radius_sel = self.radii[selected_idx]
        radius_k = self.initial_radii[k]  # Use initial radius before it's placed

        sphere1 = np.concatenate((coord_sel, [radius_sel + radius_k]))
        sphere2 = np.concatenate((self.cm, [gamma_pc]))  # Use current aggregate CM

        # Call the more robust sphere intersection function
        x_k, y_k, z_k, theta_a, vec_0, i_vec, j_vec, valid = (
            utils.cca_two_sphere_intersection(sphere1, sphere2)
        )

        if not valid:
            print(
                f"Warning: PCA sticking process sphere intersection failed for k={k}, sel={selected_idx}."
            )
            # Check conditions: dist, r1, r2
            dist_check = np.linalg.norm(sphere1[:3] - sphere2[:3])
            r1_check = sphere1[3]
            r2_check = sphere2[3]
            print(
                f"  Dist={dist_check:.4f}, R1(sel+k)={r1_check:.4f}, R2(gamma)={r2_check:.4f}"
            )
            print(
                f"  R1+R2={r1_check + r2_check:.4f}, |R1-R2|={abs(r1_check - r2_check):.4f}"
            )
            # Handle failure - maybe return a default position or raise error?
            # Returning None for now, let the caller handle failure.
            return None, 0.0, np.zeros(4), np.zeros(3), np.zeros(3)

        coord_k = np.array([x_k, y_k, z_k])
        return coord_k, theta_a, vec_0, i_vec, j_vec

    def _overlap_check(self, k: int) -> float:
        """Checks overlap between monomer k and the existing aggregate (0 to k-1)."""
        max_overlap = 0.0
        coord_k = self.coords[k]
        radius_k = self.radii[k]  # Use the radius *after* it's assigned

        for j in range(self.n1):  # Compare with particles 0 to n1-1 (which is k-1 here)
            if j == k:
                continue  # Should not happen if called correctly before n1 update

            dist_sq = np.sum((coord_k - self.coords[j]) ** 2)
            dist = np.sqrt(dist_sq)
            radius_sum = radius_k + self.radii[j]

            if dist < radius_sum - 1e-9:  # Use tolerance
                overlap = (
                    (radius_sum - dist) / radius_sum if radius_sum > 1e-12 else 1.0
                )
                max_overlap = max(max_overlap, overlap)

        return max_overlap

    def _reintento(
        self, k: int, vec_0: np.ndarray, i_vec: np.ndarray, j_vec: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Rotates monomer k to a new point on the intersection circle."""
        x0, y0, z0, r0 = vec_0

        # Generate new random angle
        theta_a_new = 2.0 * config.PI * np.random.rand()

        # Calculate new position
        coord_k_new = np.zeros(3)
        coord_k_new[0] = (
            x0
            + r0 * np.cos(theta_a_new) * i_vec[0]
            + r0 * np.sin(theta_a_new) * j_vec[0]
        )
        coord_k_new[1] = (
            y0
            + r0 * np.cos(theta_a_new) * i_vec[1]
            + r0 * np.sin(theta_a_new) * j_vec[1]
        )
        coord_k_new[2] = (
            z0
            + r0 * np.cos(theta_a_new) * i_vec[2]
            + r0 * np.sin(theta_a_new) * j_vec[2]
        )

        return coord_k_new, theta_a_new

    def run(self) -> Optional[np.ndarray]:
        """
        Runs the PCA process to aggregate all N particles.

        Returns:
            An Nx4 NumPy array [X, Y, Z, R] of the final aggregate,
            or None if the aggregation fails.
        """
        self._first_two_monomers()

        considered_indices = list(range(2))  # Track indices of particles added

        for k in range(2, self.N):  # Start aggregation from the 3rd particle (index 2)
            # print(f"PCA Step: Aggregating particle k={k}")

            # Find candidate to stick to, potentially swapping k
            # This function returns: selected_idx, m2, rg2, gamma_real, gamma_pc
            search_result = self._search_and_select_candidate(k, considered_indices)
            selected_idx = search_result[
                0
            ]  # Initial candidate selected by the search function
            m2 = search_result[1]
            rg2 = search_result[2]
            gamma_real = search_result[3]
            gamma_pc = search_result[4]
            candidates = search_result[
                5
            ]  # Get the list of candidates generated internally

            if selected_idx < 0:  # _search_and_select_candidate failed entirely
                print(
                    f"Error: PCA failed. Could not find ANY valid sticking candidate (even after swaps) for particle {k}."
                )
                self.not_able_pca = True
                return None

            # Store original state before attempting this candidate
            # Note: search_and_select might have swapped initial_radii/mass
            radius_k_current = self.initial_radii[k]
            mass_k_current = self.initial_mass[k]

            # --- Sticking and Overlap Check Loop ---
            # ****** START OF REPLACEMENT ******
            sticking_successful = False
            candidates_to_try = list(
                candidates
            )  # Get the list generated by _select_candidates inside _search_and_select...
            utils.shuffle_array(candidates_to_try)  # Try them in random order

            # If the initial selected_idx wasn't in candidates, add it?
            # Or assume _select_candidates was called with the final k state by _search?
            # Let's assume candidates list is correct for the final state of monomer k.
            # Add the initially selected one if it wasn't in the list just in case? No, rely on _select_candidates output.

            if not candidates_to_try:
                print(
                    f"Warning: PCA - No candidates generated by _select_candidates for particle {k}, even though _search succeeded initially."
                )
                # This might indicate a logic mismatch or edge case. Proceed to failure.
                pass  # Will fall through to the failure print below

            for current_selected_idx in candidates_to_try:
                # print(f"  PCA k={k}: Trying candidate index {current_selected_idx}")

                # Attempt initial sticking position using the final gamma_pc from search
                coord_k, theta_a, vec_0, i_vec, j_vec = self._sticking_process(
                    k, current_selected_idx, gamma_pc
                )

                if coord_k is None:
                    # Sticking process failed for this candidate (e.g., no sphere intersection)
                    # print(f"    Initial sticking failed for candidate {current_selected_idx}.")
                    continue  # Try next candidate in candidates_to_try

                # Tentatively place particle k
                self.coords[k] = coord_k
                self.radii[k] = radius_k_current  # Assign radius now
                self.mass[k] = mass_k_current

                # Check initial overlap
                # Use the internal method, assuming it's correct
                # cov_max = self._overlap_check(k)
                cov_max = utils.calculate_max_overlap_pca(
                    self.coords[: self.n1],
                    self.radii[: self.n1],
                    self.coords[k],
                    self.radii[k],
                )
                # Or call the external one if you moved it:
                # print(f"    Candidate {current_selected_idx}: Initial overlap = {cov_max:.4e}")

                # Rotation attempts if needed
                intento = 0
                max_rotations = 360
                while cov_max > self.tol_ov and intento < max_rotations:
                    coord_k_new, theta_a_new = self._reintento(k, vec_0, i_vec, j_vec)
                    self.coords[k] = coord_k_new  # Update position for overlap check
                    # Check overlap again
                    cov_max = utils.calculate_max_overlap_pca(
                        self.coords[: self.n1],
                        self.radii[: self.n1],
                        self.coords[k],
                        self.radii[k],
                    )
                    # cov_max = self._overlap_check(k)
                    # Or external:
                    intento += 1
                    # print(f"    Candidate {current_selected_idx}, Rotation {intento}: Overlap = {cov_max:.4e}")

                # Check if overlap is now acceptable for this candidate
                if cov_max <= self.tol_ov:
                    # print(f"    Candidate {current_selected_idx}: Sticking successful after {intento} rotations.")
                    sticking_successful = True
                    break  # <<< EXIT the candidates_to_try loop, we found a working partner

            # --- End of loop trying different candidates ---

            if sticking_successful:
                # Update aggregate properties *after* successful placement
                self.n1 += 1
                m_old = self.m1
                # Use mass_k_current which corresponds to the particle actually placed at index k
                self.m1 += mass_k_current
                self.cm = (
                    (self.cm * m_old + self.coords[k] * mass_k_current) / self.m1
                    if self.m1 > 1e-12
                    else np.mean(self.coords[: self.n1], axis=0)
                )
                self.rg1 = utils.calculate_rg(
                    self.radii[: self.n1], self.n1, self.df, self.kf
                )
                considered_indices.append(k)  # Mark as added
            else:
                # If loop finished without sticking_successful being True
                # Use the initially selected_idx in the error message as that's what search started with
                print(
                    f"Error: PCA failed. Could not find non-overlapping position for particle {k} after trying all {len(candidates_to_try)} candidates (initial search selected {selected_idx})."
                )
                self.not_able_pca = True
                # Reset particle k state? Or just fail?
                self.coords[k] = 0.0  # Reset position
                self.radii[k] = 0.0
                self.mass[k] = 0.0
                return None  # Fail aggregation
            # ****** END OF REPLACEMENT ******

        # --- End of k loop ---
        if self.not_able_pca:
            return None

        # Return combined data
        final_data = np.hstack((self.coords, self.radii.reshape(-1, 1)))
        return final_data
