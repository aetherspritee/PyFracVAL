# utils.py
"""Utility functions for vector operations and array manipulation."""

import logging
from typing import Tuple

import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)

FLOATING_POINT_ERROR = 1e-9


def shuffle_array(arr: np.ndarray) -> np.ndarray:
    """
    Randomly shuffles the elements of a 1D array in place (Fisher-Yates).
    Matches the Fortran randsample logic more closely than permutation.

    Args:
        arr: The 1D NumPy array to shuffle.

    Returns:
        The shuffled array (modified in place).
    """
    n = len(arr)
    for i in range(n - 1):
        # Random index from i to n-1
        j = np.random.randint(i, n)
        # Swap elements
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def sort_clusters(i_orden: np.ndarray) -> np.ndarray:
    """
    Sorts the cluster information array `i_orden` based on the cluster count (column 2).

    Args:
        i_orden: The `Mx3` NumPy array [start_idx, end_idx, count].

    Returns:
        A new `Mx3` array sorted by count.
    """
    if i_orden.ndim != 2 or i_orden.shape[1] != 3:
        raise ValueError("i_orden must be an Mx3 array")
    if i_orden.shape[0] == 0:
        return i_orden  # Return empty array if input is empty

    # Get the indices that would sort the 'count' column (index 2)
    sort_indices = np.argsort(i_orden[:, 2], kind="stable")
    return i_orden[sort_indices]


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculates the cross product of two 3D vectors."""
    return np.cross(a, b)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector, returning zero vector if input norm is zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:  # Use tolerance instead of exact zero check
        return np.zeros_like(v)
    return v / norm


def calculate_mass(radii: np.ndarray) -> np.ndarray:
    """Calculates mass from radii assuming constant density."""
    return (4.0 / 3.0) * np.pi * (radii**3)


def calculate_rg(radii: np.ndarray, npp: int, df: float, kf: float) -> float:
    """
    Calculates the radius of gyration for a cluster/aggregate.

    Args:
        radii: Array of radii of particles in the cluster.
        npp: Number of particles in the cluster.
        df: Fractal dimension.
        kf: Fractal prefactor.

    Returns:
        Radius of gyration, or 0.0 if calculation fails.
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
    """
    Calculates properties for a cluster: mass, Rg, CM, Rmax.

    Args:
        coords: Nx3 array of coordinates.
        radii: N array of radii.
        df: Fractal dimension.
        kf: Fractal prefactor.

    Returns:
        Tuple: (total_mass, rg, cm, r_max)
               Returns (0.0, 0.0, [0,0,0], 0.0) for empty clusters.
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
    """
    Rotates one or more vectors around a given axis by a given angle using Rodrigues' formula.

    Args:
        vectors: An Nx3 array of vectors to rotate, or a single 3D vector.
        axis: The 3D rotation axis (will be normalized).
        angle: The rotation angle in radians.

    Returns:
        The rotated vectors (Nx3 or 3D).
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
    """
    Finds the intersection circle of two spheres and picks a random point on it.
    Corresponds to the Fortran CCA_Two_sphere_intersection.

    Args:
        sphere_1: [x1, y1, z1, r1]
        sphere_2: [x2, y2, z2, r2]

    Returns:
        tuple: (x, y, z, theta, vec_0, i_vec, j_vec, valid)
               x, y, z: Coordinates of a random point on the intersection circle.
               theta: The random angle used.
               vec_0: [x0, y0, z0, r0] center and radius of intersection circle.
               i_vec, j_vec: Basis vectors for the intersection plane.
               valid (bool): True if intersection is valid (circle or point), False otherwise.
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
    """
    Calculates maximum overlap between two sets of particles (clusters).
    Parallelized using Numba (array reduction workaround).

    Args:
        coords1 (Nx3): Coordinates of cluster 1.
        radii1 (N): Radii of cluster 1.
        coords2 (Mx3): Coordinates of cluster 2.
        radii2 (M): Radii of cluster 2.

    Returns:
        Maximum overlap value (0 if no overlap).
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
    """
    Calculates maximum overlap between a new particle and an existing aggregate.
    Parallelized using Numba (array reduction workaround).

    Args:
        coords_agg (Nx3): Coordinates of the aggregate.
        radii_agg (N): Radii of the aggregate particles.
        coord_new (3): Coordinates of the new particle.
        radius_new (float): Radius of the new particle.

    Returns:
        Maximum overlap value (0 if no overlap).
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
