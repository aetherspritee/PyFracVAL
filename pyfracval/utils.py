# utils.py
"""Utility functions for vector operations and array manipulation."""

import logging
from typing import Tuple

import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)

FLOATING_POINT_ERROR = 1e-9


def shuffle_array(arr: np.ndarray) -> np.ndarray:
    """Randomly shuffle elements of a 1D array in-place (Fisher-Yates).

    Modifies the input array directly. Mimics Fortran randsample behavior.

    Parameters
    ----------
    arr : np.ndarray
        The 1D NumPy array to shuffle.

    Returns
    -------
    np.ndarray
        The input `arr`, modified in-place.
    """
    n = len(arr)
    for i in range(n - 1):
        # Random index from i to n-1
        j = np.random.randint(i, n)
        # Swap elements
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def sort_clusters(i_orden: np.ndarray) -> np.ndarray:
    """Sort cluster information array `i_orden` by cluster size (count).

    Parameters
    ----------
    i_orden : np.ndarray
        The Mx3 NumPy array [start_idx, end_idx, count].

    Returns
    -------
    np.ndarray
        A new Mx3 array sorted by the 'count' column (column index 2).

    Raises
    ------
    ValueError
        If `i_orden` is not an Mx3 array.
    """
    if i_orden.ndim != 2 or i_orden.shape[1] != 3:
        raise ValueError("i_orden must be an Mx3 array")
    if i_orden.shape[0] == 0:
        return i_orden  # Return empty array if input is empty

    # Get the indices that would sort the 'count' column (index 2)
    sort_indices = np.argsort(i_orden[:, 2], kind="stable")
    return i_orden[sort_indices]


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the cross product of two 3D vectors.

    Parameters
    ----------
    a : np.ndarray
        The first 3D vector.
    b : np.ndarray
        The second 3D vector.

    Returns
    -------
    np.ndarray
        The 3D cross product vector.

    See Also
    --------
    numpy.cross : NumPy's implementation.
    """
    return np.cross(a, b)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Returns a zero vector if the input vector's norm is close to zero.

    Parameters
    ----------
    v : np.ndarray
        The vector (1D array) to normalize.

    Returns
    -------
    np.ndarray
        The normalized vector, or a zero vector if the norm is negligible.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-12:  # Use tolerance instead of exact zero check
        return np.zeros_like(v)
    return v / norm


def calculate_mass(radii: np.ndarray) -> np.ndarray:
    """Calculate particle mass from radii assuming constant density (prop. to R^3).

    Parameters
    ----------
    radii : np.ndarray
        Array of particle radii.

    Returns
    -------
    np.ndarray
        Array of corresponding particle masses.
    """
    return (4.0 / 3.0) * np.pi * (radii**3)


def calculate_rg(radii: np.ndarray, npp: int, df: float, kf: float) -> float:
    """Calculate the radius of gyration using the fractal scaling law.

    Implements the formula Rg = a * (N / kf)^(1/Df), where 'a' is the
    geometric mean radius calculated from the input `radii` array.
    See :cite:p:`Moran2019FracVAL`.

    Parameters
    ----------
    radii : np.ndarray
        Array of radii of particles in the cluster/aggregate.
    npp : int
        Number of primary particles (N) in the cluster.
    df : float
        Fractal dimension (Df).
    kf : float
        Fractal prefactor (kf).

    Returns
    -------
    float
        The calculated radius of gyration (Rg). Returns 0.0 if `npp` is 0,
        `kf` or `df` is zero, or if calculation fails (e.g., log error).
    """
    rg = 0.0
    if npp == 0 or kf == 0 or df == 0:
        return 0.0

    try:
        valid_r = radii[radii > 1e-12]  # Filter near-zero radii
        if len(valid_r) > 0:
            # Geometric mean radius
            log_r_mean = np.sum(np.log(valid_r)) / len(valid_r)
            geo_mean_r = np.exp(log_r_mean)
            rg = geo_mean_r * (npp / kf) ** (1.0 / df)
        # else: rg remains 0.0
    except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
        # Catch potential warnings from log(<=0) as well
        logger.warning(
            f"Could not calculate rg ({e}). npp={npp}, len(valid_r)={len(valid_r)}"
        )
        rg = 0.0  # Assign a default value

    return rg


def calculate_cluster_properties(
    coords: np.ndarray, radii: np.ndarray, df: float, kf: float
) -> Tuple[float, float, np.ndarray, float]:
    """Calculate aggregate properties: total mass, Rg, center of mass, Rmax.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle coordinates.
    radii : np.ndarray
        N array of particle radii.
    df : float
        Fractal dimension used for Rg calculation.
    kf : float
        Fractal prefactor used for Rg calculation.

    Returns
    -------
    tuple[float, float | None, np.ndarray | None, float]
        A tuple containing:
            - total_mass (float): Sum of individual particle masses.
            - rg (float | None): Radius of gyration calculated via `calculate_rg`,
              or None if calculation failed.
            - cm (np.ndarray | None): 3D center of mass coordinates, or None if
              calculation failed.
            - r_max (float): Maximum distance from the center of mass to any
              particle center in the aggregate.

        Returns (0.0, 0.0, np.zeros(3), 0.0) for empty inputs (N=0).
    """
    npp = coords.shape[0]
    if npp == 0:
        return 0.0, 0.0, np.zeros(3), 0.0

    mass_vec = calculate_mass(radii)
    total_mass = np.sum(mass_vec)

    if total_mass < 1e-12:  # Use tolerance
        cm = np.mean(coords, axis=0) if npp > 0 else np.zeros(3)
    else:
        cm = np.sum(coords * mass_vec[:, np.newaxis], axis=0) / total_mass

    rg = calculate_rg(radii, npp, df, kf)

    # Calculate max distance from CM
    if npp > 0:
        dist_from_cm = np.linalg.norm(coords - cm, axis=1)
        r_max = np.max(dist_from_cm)
    else:
        r_max = 0.0

    return total_mass, rg, cm, r_max


def rodrigues_rotation(
    vectors: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """Rotate vector(s) around an axis using Rodrigues' rotation formula.

    Parameters
    ----------
    vectors : np.ndarray
        A single 3D vector or an Nx3 array of vectors to rotate.
    axis : np.ndarray
        The 3D rotation axis (does not need to be normalized).
    angle : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The rotated vector or Nx3 array of rotated vectors. Returns the
        original vectors if the axis norm is near zero.

    Raises
    ------
    ValueError
        If input `vectors` is not 1D (3,) or 2D (N, 3).
    """
    axis = normalize(axis)
    if np.linalg.norm(axis) < FLOATING_POINT_ERROR:  # No rotation if axis is zero
        return vectors

    k = axis
    v = vectors
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Apply formula: v_rot = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
    # Handle both single vector and multiple vectors (Nx3)
    if v.ndim == 1:
        cross_kv = np.cross(k, v)
        dot_kv = np.dot(k, v)
        v_rot = v * cos_a + cross_kv * sin_a + k * dot_kv * (1.0 - cos_a)
    elif v.ndim == 2:
        cross_kv = np.cross(k[np.newaxis, :], v, axis=1)
        dot_kv = np.dot(v, k)  # Result is N element array
        v_rot = (
            v * cos_a
            + cross_kv * sin_a
            + k[np.newaxis, :] * dot_kv[:, np.newaxis] * (1.0 - cos_a)
        )
    else:
        raise ValueError("Input vectors must be 3D or Nx3")

    return v_rot


def two_sphere_intersection(
    sphere_1: np.ndarray, sphere_2: np.ndarray
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Find the intersection circle of two spheres and pick a random point.

    Calculates the center (x0, y0, z0) and radius (r0) of the intersection
    circle, defines basis vectors (i_vec, j_vec) for the circle's plane,
    and returns a random point (x, y, z) on that circle based on a random
    angle (theta).

    Handles edge cases: spheres too far, one contained, coincidence, touching.

    Parameters
    ----------
    sphere_1 : np.ndarray
        Definition of the first sphere: [x1, y1, z1, r1].
    sphere_2 : np.ndarray
        Definition of the second sphere: [x2, y2, z2, r2].

    Returns
    -------
    tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, bool]
        A tuple containing:
            - x, y, z (float): Coordinates of a random point on the intersection.
            - theta (float): Random angle (radians) used to generate the point.
            - vec_0 (np.ndarray): [x0, y0, z0, r0] - center and radius of the
              intersection circle (r0=0 if spheres touch at a point).
            - i_vec (np.ndarray): First basis vector of the intersection plane.
            - j_vec (np.ndarray): Second basis vector of the intersection plane.
            - valid (bool): True if a valid intersection (circle or point)
              exists, False otherwise (e.g., separate, contained, coincident).
    """
    x1, y1, z1, r1 = sphere_1
    x2, y2, z2, r2 = sphere_2
    center1 = sphere_1[:3]
    center2 = sphere_2[:3]
    v12 = center2 - center1
    distance = np.linalg.norm(v12)

    # Default invalid return values
    invalid_ret = (0.0, 0.0, 0.0, 0.0, np.zeros(4), np.zeros(3), np.zeros(3), False)

    # --- Check for edge cases ---
    # 1. Spheres are too far apart
    if distance > r1 + r2 + FLOATING_POINT_ERROR:
        logger.debug(
            f"TSI: Spheres too far apart (d={distance:.4f}, r1+r2={r1 + r2:.4f})"
        )
        return invalid_ret
    # 2. One sphere is contained within the other without touching
    if distance < abs(r1 - r2) - FLOATING_POINT_ERROR:
        logger.debug(
            f"TSI: Sphere contained within other (d={distance:.4f}, |r1-r2|={abs(r1 - r2):.4f})"
        )
        return invalid_ret
    # 3. Spheres coincide
    if distance < FLOATING_POINT_ERROR and abs(r1 - r2) < FLOATING_POINT_ERROR:
        logger.debug("TSI: Spheres coincide")
        # Intersection is the whole sphere surface - requires different handling if needed
        return invalid_ret  # Cannot define a unique circle

    # --- Handle Touching Point Case ---
    is_touching = False
    touch_point = np.zeros(3)
    if abs(distance - (r1 + r2)) < FLOATING_POINT_ERROR:  # Touching externally
        is_touching = True
        # Point is on the line segment between centers
        if distance > FLOATING_POINT_ERROR:
            touch_point = center1 + v12 * (r1 / distance)
        else:  # Should be caught by coincident case, but fallback
            touch_point = center1
    elif abs(distance - abs(r1 - r2)) < FLOATING_POINT_ERROR:  # Touching internally
        is_touching = True
        # Point is on the line extending from centers
        if distance > FLOATING_POINT_ERROR:
            if r1 > r2:
                touch_point = center1 + v12 * (r1 / distance)
            else:  # r2 > r1
                touch_point = center2 + (-v12) * (
                    r2 / distance
                )  # Point on sphere 2 surface
        else:  # Should be caught by coincident case
            touch_point = center1

    if is_touching:
        logger.debug(f"TSI: Spheres touching at point {touch_point}")
        # Return the single point, theta=0, r0=0
        vec_0_touch = np.concatenate((touch_point, [0.0]))
        # i_vec, j_vec are ill-defined, return zeros
        return (
            touch_point[0],
            touch_point[1],
            touch_point[2],
            0.0,
            vec_0_touch,
            np.zeros(3),
            np.zeros(3),
            True,
        )

    # --- Standard Intersection Case (Circle) ---
    try:
        # distance 'd' is already computed
        # distance from center1 to intersection plane:
        dist1_plane = (distance**2 - r2**2 + r1**2) / (2 * distance)

        # Radius of the intersection circle squared
        r0_sq = r1**2 - dist1_plane**2
        if r0_sq < -FLOATING_POINT_ERROR:  # Tolerance check for numerical issues
            logger.warning(
                f"TSI: Negative r0^2 ({r0_sq}) in sphere intersection. d={distance}, r1={r1}, r2={r2}"
            )
            return invalid_ret
        r0 = np.sqrt(max(0.0, r0_sq))  # Ensure non-negative before sqrt

        # Center of the intersection circle
        unit_v12 = v12 / distance
        center0 = center1 + unit_v12 * dist1_plane
        x0, y0, z0 = center0

        # Define basis vectors for the intersection plane (perpendicular to v12)
        # k_vec is the normal to the plane (unit_v12)
        k_vec = unit_v12

        # Find a vector i_vec perpendicular to k_vec robustly
        # If k_vec is close to x-axis, use y-axis for cross product, otherwise use x-axis
        if abs(np.dot(k_vec, np.array([1.0, 0.0, 0.0]))) < 0.9:
            cross_ref = np.array([1.0, 0.0, 0.0])
        else:
            cross_ref = np.array([0.0, 1.0, 0.0])

        j_vec = normalize(cross_product(k_vec, cross_ref))
        i_vec = normalize(
            cross_product(j_vec, k_vec)
        )  # Ensure i,j,k form right-handed system

        # Generate random angle theta
        theta = 2.0 * np.pi * np.random.rand()

        # Calculate random point on the circle
        point_on_circle = (
            center0 + r0 * np.cos(theta) * i_vec + r0 * np.sin(theta) * j_vec
        )
        x, y, z = point_on_circle

        vec_0 = np.array([x0, y0, z0, r0])
        return x, y, z, theta, vec_0, i_vec, j_vec, True

    except (ZeroDivisionError, ValueError) as e:
        logger.error(f"Error during sphere intersection calculation: {e}")
        return invalid_ret


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
