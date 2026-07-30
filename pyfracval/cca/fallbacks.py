"""Gamma expansion, pair feasibility prechecks, and sticking fallbacks for CCA.

Mixed into :class:`pyfracval.cca.aggregator.CCAggregator` - these methods
assume ``self`` carries the CCAggregator instance state, set up in
``CCAggregator.__init__``.
"""

import logging
import math
from time import perf_counter
from typing import Tuple

import numpy as np

from .. import cca_kernels, fractal, overlap
from ..experimental.fft_docking import fft_dock_sticking
from ..experimental.pair_prefilters import (
    bounding_volume_precheck,
    surface_accessible_mask,
)
from ..experimental.soft_relaxation import soft_sticking

logger = logging.getLogger(__name__)

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


class _FallbacksMixin:
    """Gamma expansion, pair prechecks, and sticking fallback methods."""

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

        _t0 = perf_counter() if self.algorithm_config.profile_timing else 0.0
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
            m1, rg1, cm1, r_max1 = fractal.calculate_cluster_properties(
                coords1_in,
                radii1_in,
                self.df,
                self.kf,
                densities=self._get_cluster_densities(cluster_idx1),
            )
            m2, rg2, cm2, r_max2 = fractal.calculate_cluster_properties(
                coords2_in,
                radii2_in,
                self.df,
                self.kf,
                densities=self._get_cluster_densities(cluster_idx2),
            )
        if self.algorithm_config.profile_timing:
            self._t_cluster_props += perf_counter() - _t0

        # --- Pair Feasibility Pre-Filter ---
        pair_filter = str(self.algorithm_config.cca_pair_feasibility_filter).lower()
        if pair_filter == "bounding_volume":
            props1_bv = (m1, rg1, cm1, r_max1, radii1_in)
            props2_bv = (m2, rg2, cm2, r_max2, radii2_in)
            gamma_real_bv, gamma_pc_bv = self._calculate_cca_gamma(props1_bv, props2_bv)
            if not bounding_volume_precheck(
                gamma_pc_bv,
                r_max1,
                r_max2,
                gamma_real_bv,
                algorithm_config=self.algorithm_config,
            ):
                self._bv_filter_rejects += 1
                self._last_sticking_stats = {
                    "n1": n1,
                    "n2": n2,
                    "gamma_pc": float(gamma_pc_bv),
                    "gamma_real": bool(gamma_real_bv),
                    "sum_rmax": float(r_max1 + r_max2),
                    "failure_reason": "skipped_bv_filter",
                }
                logger.info(
                    f"CCA pair ({cluster_idx1}, {cluster_idx2}): "
                    f"Rejected by BV pre-check (gamma={gamma_pc_bv:.3f})"
                )
                return None

        # --- FFT Docking Method (opt-in) ---
        sticking_method = str(self.algorithm_config.cca_sticking_method).lower()
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
                grid_size=int(self.algorithm_config.cca_fft_grid_size),
                num_rotations=int(self.algorithm_config.cca_fft_num_rotations),
                top_k_peaks=int(self.algorithm_config.cca_fft_top_k_peaks),
                gamma_tolerance=float(self.algorithm_config.cca_fft_gamma_tolerance),
                min_peak_distance=int(self.algorithm_config.cca_fft_min_peak_distance),
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

        # --- Gamma Expansion (archived, off by default - see
        # pyfracval/experimental/gamma_expansion.py) ---
        if not bool(self.algorithm_config.cca_gamma_expansion_enabled):
            return None

        from ..experimental.gamma_expansion import run_gamma_expansion

        return run_gamma_expansion(
            self,
            cluster_idx1,
            cluster_idx2,
            cluster_props_cache,
            n1,
            n2,
            m1,
            rg1,
            cm1,
            r_max1,
            radii1_in,
            m2,
            rg2,
            cm2,
            r_max2,
            radii2_in,
        )

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
        _t0 = perf_counter() if self.algorithm_config.profile_timing else 0.0
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
            m1, rg1, cm1, r_max1 = fractal.calculate_cluster_properties(
                coords1_in,
                radii1_in,
                self.df,
                self.kf,
                densities=self._get_cluster_densities(cluster_idx1),
            )
            m2, rg2, cm2, r_max2 = fractal.calculate_cluster_properties(
                coords2_in,
                radii2_in,
                self.df,
                self.kf,
                densities=self._get_cluster_densities(cluster_idx2),
            )
        if self.algorithm_config.profile_timing:
            self._t_cluster_props += perf_counter() - _t0
        props1 = (m1, rg1, cm1, r_max1, radii1_in)
        props2 = (m2, rg2, cm2, r_max2, radii2_in)
        gamma_real, gamma_pc = self._calculate_cca_gamma(props1, props2)

        # Check for gamma override from expansion wrapper
        if hasattr(self, "_gamma_pc_override") and self._gamma_pc_override is not None:
            gamma_pc = self._gamma_pc_override
            gamma_real = self._gamma_real_override

        # --- Generate Candidate List ---
        _t0 = perf_counter() if self.algorithm_config.profile_timing else 0.0
        list_matrix = self._cca_select_candidates(
            coords1_in, radii1_in, cm1, coords2_in, radii2_in, cm2, gamma_pc, gamma_real
        )

        # Apply SSA filter if enabled
        pair_filter = str(self.algorithm_config.cca_pair_feasibility_filter).lower()
        if pair_filter == "ssa":
            ssa_mask_1 = surface_accessible_mask(
                coords1_in,
                radii1_in,
                cm1,
                r_max1,
                algorithm_config=self.algorithm_config,
            )
            ssa_mask_2 = surface_accessible_mask(
                coords2_in,
                radii2_in,
                cm2,
                r_max2,
                algorithm_config=self.algorithm_config,
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
        if self.algorithm_config.profile_timing:
            self._t_select_candidates += perf_counter() - _t0

        # Side-channel for the merge event log / _attempt_pair_merge. Set
        # up before the early return below so a no-candidate failure is
        # still described accurately rather than reported as an overlap
        # failure it never got far enough to have.
        n_feasible_pairs = int(np.sum(list_matrix))
        self._last_sticking_stats = {
            "n1": n1,
            "n2": n2,
            "gamma_pc": float(gamma_pc),
            "gamma_real": bool(gamma_real),
            "sum_rmax": float(r_max1 + r_max2),
            "n_feasible_pairs": n_feasible_pairs,
            "candidates_tried": 0,
            "rotations_used": 0,
            "min_overlap": float("inf"),
            "used_adaptive_tol": False,
        }

        if np.sum(list_matrix) == 0:
            self._last_sticking_stats["failure_reason"] = (
                "failed_gamma_not_real" if not gamma_real else "failed_no_candidates"
            )
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

        # Production default is "baseline" (unranked shuffled order, set up
        # above). "leaf_soft"/"leaf_score"/"leaf_hybrid" are archived in
        # pyfracval/experimental/candidate_policies.py - benchmarked against
        # baseline with no measurable difference in success rate (see
        # docs/source/experiments.md), kept reachable via config.
        candidate_policy = str(self.algorithm_config.cca_candidate_policy).lower()
        if candidate_policy in {"leaf_soft", "leaf_score", "leaf_hybrid"}:
            from ..experimental.candidate_policies import reorder_candidates_by_policy

            _candidate_indices = reorder_candidates_by_policy(
                candidate_policy,
                _candidate_indices,
                leaf_mask_1,
                leaf_mask_2,
                coords1_in,
                radii1_in,
                cm1,
                coords2_in,
                radii2_in,
                cm2,
                gamma_pc,
                self.algorithm_config,
                candidate_leaf_class_fn=self._candidate_leaf_class,
                candidate_score_fn=self._candidate_score,
            )

        sticking_successful = False
        final_coords1 = None
        final_coords2 = None

        attempts_tried = 0
        best_overlap = float("inf")
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
            _t0 = perf_counter() if self.algorithm_config.profile_timing else 0.0
            stick_results = self._cca_sticking_v1(
                (coords1_in, radii1_in, cm1),
                (coords2_in, radii2_in, cm2),
                cand1_idx,
                cand2_idx,
                gamma_pc,
                gamma_real,
            )
            if self.algorithm_config.profile_timing:
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
                self.algorithm_config.use_cca_incremental_overlap
                and not self.algorithm_config.use_batch_rotation
            )
            full_sync_period = max(
                self.algorithm_config.cca_incremental_full_sync_period, 1
            )

            active_collisions: set[int] = set()
            n2_local = radii2_in.shape[0]

            _t0 = perf_counter() if self.algorithm_config.profile_timing else 0.0
            if use_incremental:
                cov_max = self._full_overlap_check(
                    coords1_stick, radii1_in, coords2_stick, radii2_in
                )
                active_collisions = set()
            else:
                cov_max = overlap.calculate_max_overlap_cca_auto(
                    coords1_stick,
                    radii1_in,
                    coords2_stick,
                    radii2_in,
                    tolerance=self.tol_ov,
                )
            if self.algorithm_config.profile_timing:
                self._t_overlap_check += perf_counter() - _t0
                self._n_overlap_calls += 1

            # Rotation attempts
            intento = 0
            retry_mode_cfg = str(self.algorithm_config.cca_retry_rotation_mode).lower()
            if retry_mode_cfg in {"coarse_grid", "coarse_to_fine"}:
                max_rotations = int(
                    max(1, self.algorithm_config.cca_coarse_sweep_steps)
                    * max(1, self.algorithm_config.cca_coarse_spin_anchor_steps)
                    * max(1, self.algorithm_config.cca_coarse_spin_moving_steps)
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
            if self.algorithm_config.use_batch_rotation:
                # Batch rotation (Phase 3 - experimental, slower for N<1000)
                batch_size = self.algorithm_config.rotation_batch_size
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
                    angles = 2.0 * math.pi * attempts / golden_ratio

                    # Batch rotate cluster2 for all angles (parallel computation)
                    coords2_batch = cca_kernels.batch_rotate_cluster_cca(
                        current_coords2,
                        cm2_stick,
                        cand2_idx,
                        vec_0,
                        i_vec,
                        j_vec,
                        angles,
                    )

                    # Check overlaps for all rotated configurations (parallel)
                    overlaps = cca_kernels.batch_check_overlaps_cca(
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

                        # Check adaptive tolerance. cov_max comes from a scan
                        # that early-exits at tol_ov, so it is only a lower
                        # bound on the true maximum here - re-evaluate at the
                        # threshold being compared against before accepting
                        # (see PCAggregator._true_overlap_at).
                        if intento >= adaptive_tol_threshold and cov_max <= relaxed_tol:
                            true_cov = overlap.calculate_max_overlap_cca_auto(
                                coords1_stick,
                                radii1_in,
                                current_coords2,
                                radii2_in,
                                tolerance=relaxed_tol,
                            )
                            if true_cov <= relaxed_tol:
                                cov_max = true_cov
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
                    _t0 = (
                        perf_counter() if self.algorithm_config.profile_timing else 0.0
                    )
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
                    if self.algorithm_config.profile_timing:
                        self._t_rotation += perf_counter() - _t0
                        self._n_rotation_calls += 1

                    _t0 = (
                        perf_counter() if self.algorithm_config.profile_timing else 0.0
                    )
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
                        cov_max = overlap.calculate_max_overlap_cca_auto(
                            coords1_rotated,
                            radii1_in,
                            coords2_rotated,
                            radii2_in,
                            tolerance=self.tol_ov,
                        )

                    if self.algorithm_config.profile_timing:
                        self._t_overlap_check += perf_counter() - _t0
                        self._n_overlap_calls += 1

                    coords1_stick = coords1_rotated
                    current_coords2 = coords2_rotated
                    logger.trace(  # pyright: ignore
                        f"    Rotation {intento} [{retry_mode}]: Overlap = {cov_max:.4e}"
                    )

                    if intento >= adaptive_tol_threshold and cov_max <= relaxed_tol:
                        # cov_max early-exits at tol_ov and is only a lower
                        # bound above it; re-evaluate at relaxed_tol before
                        # accepting (see PCAggregator._true_overlap_at).
                        true_cov = overlap.calculate_max_overlap_cca_auto(
                            coords1_rotated,
                            radii1_in,
                            coords2_rotated,
                            radii2_in,
                            tolerance=relaxed_tol,
                        )
                        if true_cov <= relaxed_tol:
                            cov_max = true_cov
                            logger.info(
                                f"  CCA pair ({cand1_idx}, {cand2_idx}): Accepting "
                                f"relaxed tol (overlap={cov_max:.4e}) after {intento} rotations."
                            )
                            used_adaptive_tol = True
                            break

            # Record how close this candidate actually came, for the merge
            # log. cov_max cannot be used directly: on the incremental path
            # it only rescans previously-active collision pairs, so it
            # reads 0.0 whenever that set is empty regardless of the true
            # overlap. Only measured when a merge log is attached - it
            # costs one full non-early-exit scan per exhausted candidate,
            # which is pure overhead for anyone not collecting statistics.
            if self._merge_log is not None:
                true_final_overlap = overlap.calculate_max_overlap_cca_auto(
                    coords1_stick,
                    radii1_in,
                    current_coords2,
                    radii2_in,
                    tolerance=float("inf"),
                )
                if true_final_overlap < best_overlap:
                    best_overlap = float(true_final_overlap)
            self._last_sticking_stats.update(
                {
                    "candidates_tried": attempts_tried,
                    "rotations_used": int(intento),
                    "min_overlap": best_overlap,
                    "used_adaptive_tol": bool(used_adaptive_tol),
                }
            )

            # Check if overlap is acceptable
            if cov_max <= self.tol_ov or used_adaptive_tol:
                if use_incremental:
                    # Strict final validation with full overlap check before accept.
                    self._full_final_validations += 1
                    final_cov = overlap.calculate_max_overlap_cca_auto(
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
            self._last_sticking_stats["failure_reason"] = "failed_overlap"
            logger.warning(
                f"CCA sticking failed for clusters {cluster_idx1} and {cluster_idx2} after trying {attempts_tried} pairs."
            )
            if self.algorithm_config.cca_overlap_census_enabled:
                self._run_overlap_census_on_failure(
                    coords1_stick, radii1_in, current_coords2, radii2_in
                )
            return None  # Failed to find non-overlapping configuration

    def _run_overlap_census_on_failure(
        self,
        coords1_last: np.ndarray | None,
        radii1: np.ndarray,
        coords2_last: np.ndarray | None,
        radii2: np.ndarray,
    ) -> None:
        """Opt-in post-failure diagnostic (docs/source/overlap_failure_census.md).

        Runs a full non-early-exit overlap scan against the last-attempted
        candidate placement (whatever the sticking loop above left
        coords1_stick/current_coords2 as when it exhausted candidates) and
        stashes the result on self._last_overlap_census for the caller
        (benchmark harness, or the drop-rescue fallback, see
        docs/source/drop_rescue.md) to consume. Also stashes the exact
        geometry the census was computed against on
        self._last_overlap_failure_geometry, since drop-rescue needs the
        actual coordinate arrays, not just the census's summary stats, to
        remove the offending particles and re-check.

        Silently no-ops if the very last attempt never got past initial
        placement (coords1_last/coords2_last still None) - there is no
        geometry to census in that case. Strictly additive: only runs when
        cca_overlap_census_enabled is set, never on the default hot path.
        """
        if coords1_last is None or coords2_last is None:
            self._last_overlap_census = None
            self._last_overlap_failure_geometry = None
            return
        from ..overlap_statistics import compute_overlap_census

        self._last_overlap_census = compute_overlap_census(
            coords1_last,
            radii1,
            coords2_last,
            radii2,
            max_pairs=self.algorithm_config.cca_overlap_census_max_pairs,
        )
        self._last_overlap_failure_geometry = (
            coords1_last,
            radii1,
            coords2_last,
            radii2,
        )

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
            m1, rg1, cm1, r_max1 = fractal.calculate_cluster_properties(
                coords1_in,
                radii1_in,
                self.df,
                self.kf,
                densities=self._get_cluster_densities(cluster_idx1),
            )
            m2, rg2, cm2, r_max2 = fractal.calculate_cluster_properties(
                coords2_in,
                radii2_in,
                self.df,
                self.kf,
                densities=self._get_cluster_densities(cluster_idx2),
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
        k_repulsion = float(self.algorithm_config.cca_soft_relaxation_k_repulsion)
        k_gamma = float(self.algorithm_config.cca_soft_relaxation_k_gamma)
        gamma_tol = float(self.algorithm_config.cca_soft_relaxation_gamma_tolerance)
        max_iters = int(self.algorithm_config.cca_soft_relaxation_max_iters)
        learning_rate = float(self.algorithm_config.cca_soft_relaxation_learning_rate)

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
