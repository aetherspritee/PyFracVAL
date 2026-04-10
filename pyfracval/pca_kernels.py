"""PCA-specific JIT kernels for PyFracVAL.

JIT-compiled batch functions for PCA position calculation and overlap checking.

Functions
---------
batch_calculate_positions_pca
    JIT batch calculation of particle positions during PCA.
batch_check_overlaps_pca
    JIT parallel overlap checker for PCA batch operations.
"""

import logging

import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)


@jit(parallel=True, fastmath=True, cache=True, nopython=True)
def batch_calculate_positions_pca(
    vec_0: np.ndarray,
    i_vec: np.ndarray,
    j_vec: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Calculate batch of positions on intersection circle for PCA.

    Uses Numba parallel loops to compute multiple rotation positions simultaneously.

    Parameters
    ----------
    vec_0 : np.ndarray
        [x0, y0, z0, r0] - center and radius of intersection circle
    i_vec : np.ndarray
        First basis vector (3D)
    j_vec : np.ndarray
        Second basis vector (3D)
    angles : np.ndarray
        Array of rotation angles (1D)

    Returns
    -------
    np.ndarray
        (N, 3) array of positions, one per angle
    """
    n_angles = angles.shape[0]
    positions = np.empty((n_angles, 3), dtype=np.float64)

    x0, y0, z0, r0 = vec_0

    # Parallel loop over angles
    for i in prange(n_angles):
        theta = angles[i]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Calculate position: center + r0 * (cos(theta)*i_vec + sin(theta)*j_vec)
        positions[i, 0] = x0 + r0 * (cos_theta * i_vec[0] + sin_theta * j_vec[0])
        positions[i, 1] = y0 + r0 * (cos_theta * i_vec[1] + sin_theta * j_vec[1])
        positions[i, 2] = z0 + r0 * (cos_theta * i_vec[2] + sin_theta * j_vec[2])

    return positions


@jit(parallel=True, fastmath=True, cache=True, nopython=True)
def batch_check_overlaps_pca(
    coords_agg: np.ndarray,
    radii_agg: np.ndarray,
    candidate_positions: np.ndarray,
    radius_new: float,
    tolerance: float,
) -> np.ndarray:
    """Check overlap for batch of candidate positions (PCA).

    Uses Numba parallel loops to evaluate multiple positions simultaneously.

    Parameters
    ----------
    coords_agg : np.ndarray
        Current aggregate coordinates (n_agg, 3)
    radii_agg : np.ndarray
        Current aggregate radii (n_agg,)
    candidate_positions : np.ndarray
        Batch of candidate positions to test (n_candidates, 3)
    radius_new : float
        Radius of new particle
    tolerance : float
        Overlap tolerance

    Returns
    -------
    np.ndarray
        (n_candidates,) array of max overlap values for each position
    """
    n_candidates = candidate_positions.shape[0]
    n_agg = coords_agg.shape[0]
    overlaps = np.empty(n_candidates, dtype=np.float64)

    # Parallel loop over candidate positions
    for idx in prange(n_candidates):
        coord_new = candidate_positions[idx]
        max_overlap = -np.inf

        # For each candidate, check against all aggregate particles
        for j in range(n_agg):
            coord_agg = coords_agg[j]
            radius_agg = radii_agg[j]
            radius_sum = radius_new + radius_agg

            # Compute squared distance
            d_sq = 0.0
            for dim in range(3):
                diff = coord_new[dim] - coord_agg[dim]
                d_sq += diff * diff

            # Bounding sphere pre-check
            radius_sum_sq = radius_sum * radius_sum
            if d_sq > radius_sum_sq:
                continue

            # Compute overlap
            dist = np.sqrt(d_sq)
            overlap = 1.0 - dist / radius_sum

            if overlap > max_overlap:
                max_overlap = overlap

            # Early termination check (can't break in prange, but helps inner loop)
            if overlap > tolerance:
                max_overlap = overlap
                break

        overlaps[idx] = max_overlap

    return overlaps
