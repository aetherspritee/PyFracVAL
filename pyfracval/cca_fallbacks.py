"""CCA fallback and pre-check utilities for sticking operations.

Pure functions for bounding-volume pre-checks and surface-accessibility
masking used by the CCA fallback sticking methods.

Functions
---------
bounding_volume_precheck
    Fast bounding-volume pre-check for sticking feasibility.
surface_accessible_mask
    Compute surface accessibility mask for candidate selection.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def bounding_volume_precheck(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    penetration_factor: float = 0.8,
) -> tuple[bool, float | None]:
    """Fast bounding-volume pre-check for sticking feasibility.

    Rejects clearly infeasible pairs where the sum of bounding-sphere
    radii is smaller than the distance between centres of mass.

    Parameters
    ----------
    coords1, coords2 : np.ndarray
        (N, 3) and (M, 3) particle coordinate arrays.
    radii1, radii2 : np.ndarray
        (N,) and (M,) particle radius arrays.
    penetration_factor : float, optional
        Safety factor (default 0.8). Lower values are more permissive.

    Returns
    -------
    tuple[bool, float | None]
        ``(feasible, max_overlap)`` where *feasible* is True if the
        pair passes the pre-check, and *max_overlap* is the estimated
        maximum overlap (or None if infeasible).
    """
    cm1 = np.sum(coords1 * radii1[:, np.newaxis] ** 3, axis=0) / np.sum(radii1**3)
    cm2 = np.sum(coords2 * radii2[:, np.newaxis] ** 3, axis=0) / np.sum(radii2**3)
    r1_max = float(np.max(np.linalg.norm(coords1 - cm1, axis=1) + radii1))
    r2_max = float(np.max(np.linalg.norm(coords2 - cm2, axis=1) + radii2))
    d_cm = float(np.linalg.norm(cm1 - cm2))

    if d_cm > (r1_max + r2_max) * penetration_factor:
        return False, None

    return True, None


def surface_accessible_mask(
    coords: np.ndarray,
    radii: np.ndarray,
    cm: np.ndarray,
    r_max: float,
    min_exposure: float = 0.3,
) -> np.ndarray:
    """Compute surface accessibility mask for candidate particles.

    A particle is surface-accessible if a sufficient fraction of its
    solid angle is not occluded by neighbours.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) particle centre coordinates.
    radii : np.ndarray
        (N,) particle radii.
    cm : np.ndarray
        (3,) centre-of-mass vector.
    r_max : float
        Maximum radius of the cluster.
    min_exposure : float, optional
        Minimum exposed fraction to count as surface-accessible (default 0.3).

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates a surface-accessible particle.
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Direction from each particle to the centre of mass
    to_cm = cm[np.newaxis, :] - coords
    dist_to_cm = np.linalg.norm(to_cm, axis=1)

    # Particles near the surface are those far from cm relative to r_max
    surface_fraction = dist_to_cm / max(r_max, 1.0e-12)
    return surface_fraction >= (1.0 - min_exposure)
