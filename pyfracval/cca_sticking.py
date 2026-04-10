"""CCA sticking and rotation utility functions.

Pure functions for cluster rotation, axis normalisation, overlap checking,
and collision scanning used during CCA sticking operations.

Functions
---------
rotate_cluster_about_cm
    Rotate a full cluster around its centre of mass.
normalize_axis
    Normalise a rotation axis, falling back to a default or x-axis.
scan_active_collisions
    Scan for remaining collisions in an active-set overlap mask.
full_overlap_check
    Perform a full pairwise overlap check between two sets of particles.
"""

import logging

import numpy as np

from .geometry import rodrigues_rotation

logger = logging.getLogger(__name__)


def rotate_cluster_about_cm(
    coords_in: np.ndarray,
    cm: np.ndarray,
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate a full cluster around its centre of mass.

    Parameters
    ----------
    coords_in : np.ndarray
        (N, 3) array of particle centre coordinates.
    cm : np.ndarray
        (3,) centre-of-mass vector.
    axis : np.ndarray
        (3,) rotation axis (does not need to be unit-length).
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        (N, 3) rotated coordinates.
    """
    if np.linalg.norm(axis) <= 1.0e-12 or abs(float(angle_rad)) <= 1.0e-12:
        return coords_in
    coords_rel = coords_in - cm
    coords_rel_rot = rodrigues_rotation(coords_rel, axis, float(angle_rad))
    return coords_rel_rot + cm


def normalize_axis(axis: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalise a rotation axis; fall back to *fallback* or x-axis.

    Parameters
    ----------
    axis : np.ndarray
        (3,) rotation axis candidate.
    fallback : np.ndarray or None, optional
        Alternative axis to try if *axis* is near-zero.

    Returns
    -------
    np.ndarray
        Unit-length (3,) rotation axis.
    """
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


def scan_active_collisions(
    coords1: np.ndarray,
    radii1: np.ndarray,
    active_idx: np.ndarray,
    coords2_single: np.ndarray,
    radii2_single: np.ndarray,
) -> np.ndarray:
    """Scan for remaining collisions in an active-set overlap mask.

    Parameters
    ----------
    coords1 : np.ndarray
        (N, 3) coordinates of cluster 1 particles in the active set.
    radii1 : np.ndarray
        (N,) radii of cluster 1 particles.
    active_idx : np.ndarray
        Indices of active particles to check.
    coords2_single : np.ndarray
        (3,) coordinates of the candidate particle.
    radii2_single : np.ndarray
        (1,) radius of the candidate particle.

    Returns
    -------
    np.ndarray
        Indices into *active_idx* where collisions are found.
    """
    if len(active_idx) == 0:
        return np.array([], dtype=int)
    d_sq = np.sum((coords1[active_idx] - coords2_single[np.newaxis, :]) ** 2, axis=1)
    r_sum = radii1[active_idx] + float(radii2_single)
    overlap = d_sq < r_sum * r_sum
    return active_idx[overlap]


def full_overlap_check(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
) -> bool:
    """Perform a full pairwise overlap check between two clusters.

    Parameters
    ----------
    coords1, coords2 : np.ndarray
        (N, 3) and (M, 3) coordinate arrays.
    radii1, radii2 : np.ndarray
        (N,) and (M,) radius arrays.

    Returns
    -------
    bool
        True if any pair of particles overlaps.
    """
    n1, n2 = coords1.shape[0], coords2.shape[0]
    for i in range(n1):
        for j in range(n2):
            d_sq = float(np.sum((coords1[i] - coords2[j]) ** 2))
            r_sum = float(radii1[i]) + float(radii2[j])
            if d_sq < r_sum * r_sum:
                return True
    return False
