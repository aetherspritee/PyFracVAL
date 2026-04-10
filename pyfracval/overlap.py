"""Overlap calculation dispatch for PyFracVAL.

Functions for computing maximum overlap between particle clusters,
with variants for PCA, CCA, fast, parallel, and auto-dispatch modes.

Constants
---------
PARALLEL_OVERLAP_THRESHOLD
    Minimum cluster size to trigger parallel overlap calculation.

Functions
---------
calculate_max_overlap_cca
    Full pairwise overlap calculation for CCA cluster pair.
calculate_max_overlap_pca
    Full pairwise overlap calculation for PCA cluster pair.
calculate_max_overlap_pca_fast
    Optimised overlapping-spheres overlap for PCA.
calculate_max_overlap_cca_fast
    Optimised overlapping-spheres overlap for CCA.
calculate_max_overlap_pca_parallel
    Parallel overlap calculation for PCA cluster pair.
calculate_max_overlap_cca_parallel
    Parallel overlap calculation for CCA cluster pair.
calculate_max_overlap_pca_auto
    Auto-dispatch overlap for PCA (chooses fast vs parallel).
calculate_max_overlap_cca_auto
    Auto-dispatch overlap for CCA (chooses fast vs parallel).
"""

import logging

import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)


@jit(parallel=True, fastmath=True, cache=True)
def calculate_max_overlap_cca(
    coords1: np.ndarray, radii1: np.ndarray, coords2: np.ndarray, radii2: np.ndarray
) -> float:
    """Calculate max overlap between two particle clusters (Numba optimized).

    Overlap is defined as `1 - distance / (radius1 + radius2)` for
    overlapping pairs, max(0).

    Parameters
    ----------
    coords1 : np.ndarray
        Nx3 coordinates of cluster 1.
    radii1 : np.ndarray
        N radii of cluster 1.
    coords2 : np.ndarray
        Mx3 coordinates of cluster 2.
    radii2 : np.ndarray
        M radii of cluster 2.

    Returns
    -------
    float
        Maximum overlap fraction found between any particle in cluster 1
        and any particle in cluster 2. Returns 0.0 if no overlap.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]

    total_pairs = n1 * n2

    if total_pairs == 0:
        return 0.0

    max_overlap_val = 0.0

    for k in prange(total_pairs):
        i = k % n1
        j = k // n1

        coord1 = coords1[i]
        radius1 = radii1[i]

        coord2 = coords2[j]
        radius2 = radii2[j]

        d_sq = 0.0
        for dim in range(3):  # Assuming 3D
            d_sq += (coord1[dim] - coord2[dim]) ** 2
        dist_ij = np.sqrt(d_sq)

        overlap = 1 - dist_ij / (radius1 + radius2)
        max_overlap_val = max(overlap, max_overlap_val)  # no racing condition

    return max_overlap_val


@jit(parallel=True, fastmath=True, cache=True)
def calculate_max_overlap_pca(
    coords_agg: np.ndarray,
    radii_agg: np.ndarray,
    coord_new: np.ndarray,
    radius_new: float,
) -> float:
    """Calculate max overlap between a new particle and an aggregate (Numba).

    Overlap is defined as `1 - distance / (radius_new + radius_agg)` for
    overlapping pairs, max(0).

    Parameters
    ----------
    coords_agg : np.ndarray
        Nx3 coordinates of the existing aggregate.
    radii_agg : np.ndarray
        N radii of the aggregate particles.
    coord_new : np.ndarray
        3D coordinates of the new particle.
    radius_new : float
        Radius of the new particle.

    Returns
    -------
    float
        Maximum overlap fraction found between the new particle and any
        particle in the aggregate. Returns 0.0 if no overlap.
    """
    n_agg = coords_agg.shape[0]

    if n_agg == 0:
        return 0.0

    max_overlap_val = 0.0

    for j in prange(n_agg):
        coord_agg = coords_agg[j]
        radius_agg = radii_agg[j]

        d_sq = 0.0
        for dim in range(3):
            d_sq += (coord_new[dim] - coord_agg[dim]) ** 2
        dist = np.sqrt(d_sq)

        overlap = 1 - dist / (radius_new + radius_agg)
        max_overlap_val = max(overlap, max_overlap_val)  # no racing condition

    return max_overlap_val


@jit(parallel=False, fastmath=True, cache=True)
def calculate_max_overlap_pca_fast(
    coords_agg: np.ndarray,
    radii_agg: np.ndarray,
    coord_new: np.ndarray,
    radius_new: float,
    tolerance: float = 1e-6,
) -> float:
    """Calculate max overlap with early termination (optimized for speed).

    This optimized version includes:
    1. Early termination: Returns immediately when overlap exceeds tolerance
    2. Bounding sphere pre-check: Avoids sqrt for particles far apart
    3. Sequential execution: Trades parallelization for early exit

    Overlap is defined as `1 - distance / (radius_new + radius_agg)`.

    Performance: ~2-3x faster than parallel version when overlap is found early.

    Parameters
    ----------
    coords_agg : np.ndarray
        Nx3 coordinates of the existing aggregate.
    radii_agg : np.ndarray
        N radii of the aggregate particles.
    coord_new : np.ndarray
        3D coordinates of the new particle.
    radius_new : float
        Radius of the new particle.
    tolerance : float, optional
        Overlap tolerance threshold for early termination (default: 1e-6).

    Returns
    -------
    float
        Maximum overlap fraction found. Returns immediately if overlap > tolerance.
    """
    n_agg = coords_agg.shape[0]

    if n_agg == 0:
        return 0.0

    max_overlap_val = 0.0

    for j in range(n_agg):
        coord_agg = coords_agg[j]
        radius_agg = radii_agg[j]

        # Calculate squared distance
        d_sq = 0.0
        for dim in range(3):
            d_sq += (coord_new[dim] - coord_agg[dim]) ** 2

        # Bounding sphere pre-check: skip sqrt if particles are far apart
        radius_sum = radius_new + radius_agg
        radius_sum_sq = radius_sum * radius_sum

        if d_sq > radius_sum_sq:
            # No overlap possible, skip this particle
            continue

        # Compute actual distance (only when needed)
        dist = np.sqrt(d_sq)

        # Calculate overlap
        overlap = 1.0 - dist / radius_sum

        # Update maximum
        if overlap > max_overlap_val:
            max_overlap_val = overlap

        # Early termination: return immediately if overlap exceeds tolerance
        if overlap > tolerance:
            return overlap

    return max_overlap_val


@jit(parallel=False, fastmath=True, cache=True)
def calculate_max_overlap_cca_fast(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    tolerance: float = 1e-6,
) -> float:
    """Calculate max overlap between clusters with early termination (optimized).

    This optimized version includes:
    1. Early termination: Returns immediately when overlap exceeds tolerance
    2. Bounding sphere pre-check: Avoids sqrt for particles far apart
    3. Sequential execution: Trades parallelization for early exit

    Overlap is defined as `1 - distance / (radius1 + radius2)`.

    Performance: ~2-3x faster than parallel version when overlap is found early.

    Parameters
    ----------
    coords1 : np.ndarray
        Nx3 coordinates of cluster 1.
    radii1 : np.ndarray
        N radii of cluster 1.
    coords2 : np.ndarray
        Mx3 coordinates of cluster 2.
    radii2 : np.ndarray
        M radii of cluster 2.
    tolerance : float, optional
        Overlap tolerance threshold for early termination (default: 1e-6).

    Returns
    -------
    float
        Maximum overlap fraction found. Returns immediately if overlap > tolerance.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]

    total_pairs = n1 * n2

    if total_pairs == 0:
        return 0.0

    max_overlap_val = 0.0

    # Nested loops for sequential scanning with early exit
    for i in range(n1):
        coord1 = coords1[i]
        radius1 = radii1[i]

        for j in range(n2):
            coord2 = coords2[j]
            radius2 = radii2[j]

            # Calculate squared distance
            d_sq = 0.0
            for dim in range(3):
                d_sq += (coord1[dim] - coord2[dim]) ** 2

            # Bounding sphere pre-check
            radius_sum = radius1 + radius2
            radius_sum_sq = radius_sum * radius_sum

            if d_sq > radius_sum_sq:
                # No overlap possible, skip this pair
                continue

            # Compute actual distance (only when needed)
            dist_ij = np.sqrt(d_sq)

            # Calculate overlap
            overlap = 1.0 - dist_ij / radius_sum

            # Update maximum
            if overlap > max_overlap_val:
                max_overlap_val = overlap

            # Early termination
            if overlap > tolerance:
                return overlap

    return max_overlap_val


# ============================================================================
# Phase 3B: Hybrid Strategy - Parallel Overlap with Sequential Rotation
# ============================================================================

# Threshold for using parallel overlap calculation (particles in aggregate)
PARALLEL_OVERLAP_THRESHOLD = 200


@jit(parallel=True, fastmath=True, cache=True, nopython=True)
def calculate_max_overlap_pca_parallel(
    coords_agg: np.ndarray,
    radii_agg: np.ndarray,
    coord_new: np.ndarray,
    radius_new: float,
) -> float:
    """Calculate max overlap for PCA with parallel execution (no early termination).

    This version uses Numba prange to parallelize overlap checks across all
    aggregate particles. Trade-off: No early termination, but faster for large N.

    Use for n_agg > PARALLEL_OVERLAP_THRESHOLD (~200 particles).

    Parameters
    ----------
    coords_agg : np.ndarray
        Current aggregate coordinates (n_agg, 3)
    radii_agg : np.ndarray
        Current aggregate radii (n_agg,)
    coord_new : np.ndarray
        New particle coordinates (3,)
    radius_new : float
        New particle radius

    Returns
    -------
    float
        Maximum overlap fraction found across all particles
    """
    n_agg = coords_agg.shape[0]
    overlaps = np.empty(n_agg, dtype=np.float64)

    # Parallel loop over all aggregate particles
    for j in prange(n_agg):
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
            overlaps[j] = -np.inf
            continue

        # Compute overlap
        dist = np.sqrt(d_sq)
        overlaps[j] = 1.0 - dist / radius_sum

    return np.max(overlaps)


@jit(parallel=True, fastmath=True, cache=True, nopython=True)
def calculate_max_overlap_cca_parallel(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
) -> float:
    """Calculate max overlap for CCA with parallel execution (no early termination).

    This version uses Numba prange to parallelize overlap checks. Computes all
    pair overlaps in parallel.

    Use for n1 * n2 > PARALLEL_OVERLAP_THRESHOLD (~200 pairs).

    Parameters
    ----------
    coords1 : np.ndarray
        Cluster 1 coordinates (n1, 3)
    radii1 : np.ndarray
        Cluster 1 radii (n1,)
    coords2 : np.ndarray
        Cluster 2 coordinates (n2, 3)
    radii2 : np.ndarray
        Cluster 2 radii (n2,)

    Returns
    -------
    float
        Maximum overlap fraction found across all particle pairs
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    total_pairs = n1 * n2

    # Flatten to 1D array for parallel processing
    all_overlaps = np.empty(total_pairs, dtype=np.float64)

    # Parallel loop over all pairs
    for pair_idx in prange(total_pairs):
        i = pair_idx // n2
        j = pair_idx % n2

        coord1 = coords1[i]
        radius1 = radii1[i]
        coord2 = coords2[j]
        radius2 = radii2[j]
        radius_sum = radius1 + radius2

        # Compute squared distance
        d_sq = 0.0
        for dim in range(3):
            diff = coord1[dim] - coord2[dim]
            d_sq += diff * diff

        # Bounding sphere pre-check
        radius_sum_sq = radius_sum * radius_sum
        if d_sq > radius_sum_sq:
            all_overlaps[pair_idx] = -np.inf
            continue

        # Compute overlap
        dist = np.sqrt(d_sq)
        all_overlaps[pair_idx] = 1.0 - dist / radius_sum

    return np.max(all_overlaps)


def calculate_max_overlap_pca_auto(
    coords_agg: np.ndarray,
    radii_agg: np.ndarray,
    coord_new: np.ndarray,
    radius_new: float,
    tolerance: float = 1e-6,
) -> float:
    """Auto-dispatch to parallel or sequential overlap check based on size.

    For large aggregates (n > PARALLEL_OVERLAP_THRESHOLD), uses parallel version
    without early termination. For small aggregates, uses sequential with early exit.

    Parameters
    ----------
    coords_agg : np.ndarray
        Current aggregate coordinates (n_agg, 3)
    radii_agg : np.ndarray
        Current aggregate radii (n_agg,)
    coord_new : np.ndarray
        New particle coordinates (3,)
    radius_new : float
        New particle radius
    tolerance : float, optional
        Overlap tolerance for early termination (default: 1e-6)

    Returns
    -------
    float
        Maximum overlap fraction
    """
    n_agg = coords_agg.shape[0]

    if n_agg > PARALLEL_OVERLAP_THRESHOLD:
        # Large aggregate: use parallel (no early termination)
        return calculate_max_overlap_pca_parallel(
            coords_agg, radii_agg, coord_new, radius_new
        )
    else:
        # Small aggregate: use sequential with early termination
        return calculate_max_overlap_pca_fast(
            coords_agg, radii_agg, coord_new, radius_new, tolerance
        )


def calculate_max_overlap_cca_auto(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    tolerance: float = 1e-6,
) -> float:
    """Check max overlap between two clusters during CCA sticking.

    FIX (PyFracVAL-xwx): Always use the sequential early-termination path.
    The previous parallel dispatch was counterproductive for CCA: in sticking,
    clusters are placed touching (high overlap probability), so early termination
    fires almost immediately. The parallel path computes ALL n1*n2 pairs even
    when the first pair already overlaps, making it 78x slower for large clusters.

    Benchmark (n1=n2=256, 65536 pairs): parallel=109µs, fast=1.4µs.

    Parameters
    ----------
    coords1 : np.ndarray
        Cluster 1 coordinates (n1, 3)
    radii1 : np.ndarray
        Cluster 1 radii (n1,)
    coords2 : np.ndarray
        Cluster 2 coordinates (n2, 3)
    radii2 : np.ndarray
        Cluster 2 radii (n2,)
    tolerance : float, optional
        Overlap tolerance for early termination (default: 1e-6)

    Returns
    -------
    float
        Maximum overlap fraction
    """
    # Always use sequential with early termination: CCA clusters are placed
    # touching so overlap is found immediately, making early exit dominant.
    return calculate_max_overlap_cca_fast(coords1, radii1, coords2, radii2, tolerance)
