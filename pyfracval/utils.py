# utils.py
"""Utility functions for vector operations and array manipulation."""

import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import jit, prange

logger = logging.getLogger(__name__)

FLOATING_POINT_ERROR = 1e-9


def shuffle_array(arr: np.ndarray) -> np.ndarray:
    """Randomly shuffle elements of a 1D array in-place (Fisher-Yates).

    Modifies the input array directly. Mimics Fortran randsample behavior.

    Parameters
    ----------
    arr :
        The 1D NumPy array to shuffle.

    Returns
    -------
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


# def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     """Calculate the cross product of two 3D vectors.

#     Parameters
#     ----------
#     a : np.ndarray
#         The first 3D vector.
#     b : np.ndarray
#         The second 3D vector.

#     Returns
#     -------
#     np.ndarray
#         The 3D cross product vector.

#     See Also
#     --------
#     numpy.cross : NumPy's implementation.
#     """
#     return np.cross(a, b)


# def normalize(v: np.ndarray) -> np.ndarray:
#     """Normalize a vector to unit length.

#     Returns a zero vector if the input vector's norm is close to zero.

#     Parameters
#     ----------
#     v : np.ndarray
#         The vector (1D array) to normalize.

#     Returns
#     -------
#     np.ndarray
#         The normalized vector, or a zero vector if the norm is negligible.
#     """
#     norm = np.linalg.norm(v)
#     if norm < 1e-12:  # Use tolerance instead of exact zero check
#         return np.zeros_like(v)
#     return v / norm


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
    # TODO: throw an error just in case
    if npp == 0 or kf == 0 or df == 0:
        return 0.0

    # TODO: check radii beforehand
    valid_r = radii[radii > 1e-12]  # Filter near-zero radii
    try:
        if len(valid_r) > 0:
            # Geometric mean radius
            log_r_mean = np.sum(np.log(valid_r)) / len(valid_r)
            geo_mean_r = np.exp(log_r_mean)
            rg = geo_mean_r * (npp / kf) ** (1.0 / df)
    except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
        # Catch potential warnings from log(<=0) as well
        logger.warning(
            f"Could not calculate rg ({e}). npp={npp}, len(valid_r)={len(valid_r)}"
        )

    return max(rg, 0.0)


def gamma_calculation(
    m1: float,
    rg1: float,
    radii1: npt.NDArray,
    m2: float,
    rg2: float,
    radii2: npt.NDArray,
    df: float,
    kf: float,
    heuristic: bool = True,
) -> tuple[bool, float]:
    """
    Calculates Gamma_pc for adding the next monomer (aggregate 2).
    """
    n1 = radii1.size
    n2 = radii2.size

    n3 = n1 + n2
    m3 = m1 + m2

    if heuristic:
        m1 = n1
        m2 = n2
        m3 = n3

    # Radii of particles already in cluster + the next one to be added
    combined_radii = np.concatenate((radii1, radii2))
    rg3 = calculate_rg(combined_radii, n3, df, kf)

    # Heuristic from Fortran: ensure rg3 is not smaller than rg1
    # (avoids issues if rg calculation is noisy for small N)
    if n2 == 1 and rg3 < rg1:
        logger.info(f"Gamma calc: Adjusted rg3 from {rg3:.2e} to match rg1 {rg1:.2e}")
        rg3 = rg1

    gamma_pc = 0.0
    gamma_real = False

    term1 = (m3**2) * (rg3**2)
    term2 = m3 * (m1 * rg1**2 + m2 * rg2**2)  # rg2 is for monomer
    denominator = m1 * m2
    radicand = term1 - term2
    try:
        gamma_pc = np.sqrt(radicand / denominator)
        gamma_real = True
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        logger.warning(f"Gamma calculation internal failed: {e}")
        logger.warning(
            f"Gamma_pc calculation non-real or denominator zero: "
            f"n1={n1}, m1={m1:.2e}, rg1={rg1:.2e}, "
            f"n2={n2}, m2={m2:.2e}, rg2={rg2:.2e}, "
            f"n3={n3}, m3={m3:.2e}, rg3={rg3:.2e} -> "
            f"radicand={radicand:.2e}, denominator={denominator:.2e}"
        )
        gamma_real = False

    return gamma_real, gamma_pc


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
    # No rotation if axis is zero
    if np.linalg.norm(axis) < FLOATING_POINT_ERROR:
        return vectors
    axis /= np.linalg.norm(axis)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Apply formula: v_rot = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
    # Handle both single vector and multiple vectors (Nx3)
    if vectors.ndim == 1:
        dot_kv = np.dot(axis, vectors)
        cross_kv = np.cross(axis, vectors)
    elif vectors.ndim == 2:
        axis = axis[np.newaxis, :]
        dot_kv = np.sum(axis * vectors, axis=1)[:, np.newaxis]
        cross_kv = np.cross(axis, vectors, axis=1)

    # elif vectors.ndim > 2:
    else:
        raise ValueError("Input vectors must be 3D or Nx3")
    v_rot = vectors * cos_a + cross_kv * sin_a + axis * dot_kv * (1.0 - cos_a)

    return v_rot


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


@jit(parallel=True, fastmath=True, cache=True, nopython=True)
def batch_check_overlaps_cca(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2_batch: np.ndarray,
    radii2: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Check overlap for batch of cluster2 configurations (CCA).

    Uses Numba parallel loops to evaluate multiple cluster configurations simultaneously.

    Parameters
    ----------
    coords1 : np.ndarray
        Cluster 1 coordinates (n1, 3)
    radii1 : np.ndarray
        Cluster 1 radii (n1,)
    coords2_batch : np.ndarray
        Batch of cluster 2 configurations (n_batch, n2, 3)
    radii2 : np.ndarray
        Cluster 2 radii (n2,) - same for all configurations
    tolerance : float
        Overlap tolerance

    Returns
    -------
    np.ndarray
        (n_batch,) array of max overlap values for each configuration
    """
    n_batch = coords2_batch.shape[0]
    n1 = coords1.shape[0]
    n2 = coords2_batch.shape[1]
    overlaps = np.empty(n_batch, dtype=np.float64)

    # Parallel loop over batch
    for batch_idx in prange(n_batch):
        coords2 = coords2_batch[batch_idx]
        max_overlap = -np.inf

        # Check all pairs between cluster1 and cluster2
        for i in range(n1):
            coord1 = coords1[i]
            radius1 = radii1[i]

            for j in range(n2):
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
                    continue

                # Compute overlap
                dist = np.sqrt(d_sq)
                overlap = 1.0 - dist / radius_sum

                if overlap > max_overlap:
                    max_overlap = overlap

                # Early termination for inner loops
                if overlap > tolerance:
                    max_overlap = overlap
                    break

            # If already over tolerance, no need to check more cluster1 particles
            if max_overlap > tolerance:
                break

        overlaps[batch_idx] = max_overlap

    return overlaps


def batch_rotate_cluster_cca(
    coords2_in: np.ndarray,
    cm2: np.ndarray,
    cand2_idx: int,
    vec_0: np.ndarray,
    i_vec: np.ndarray,
    j_vec: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Batch rotate cluster2 for multiple angles (CCA).

    For each angle, calculates the target position on the intersection circle,
    then rotates the entire cluster to align the candidate particle with that target.

    Parameters
    ----------
    coords2_in : np.ndarray
        Cluster 2 coordinates (n2, 3)
    cm2 : np.ndarray
        Center of mass of cluster 2 (3,)
    cand2_idx : int
        Index of candidate particle in cluster 2
    vec_0 : np.ndarray
        [x0, y0, z0, r0] - center and radius of intersection circle
    i_vec : np.ndarray
        First basis vector (3,)
    j_vec : np.ndarray
        Second basis vector (3,)
    angles : np.ndarray
        Array of rotation angles (n_angles,)

    Returns
    -------
    np.ndarray
        (n_angles, n2, 3) array of rotated cluster configurations
    """
    n_angles = angles.shape[0]
    n2 = coords2_in.shape[0]
    rotated_clusters = np.empty((n_angles, n2, 3), dtype=np.float64)

    x0, y0, z0, r0 = vec_0

    # Current position of candidate particle relative to CM
    current_p2 = coords2_in[cand2_idx]
    v1_rot = current_p2 - cm2
    norm_v1 = np.linalg.norm(v1_rot)

    # For each angle, calculate target and rotate cluster
    for i in range(n_angles):
        theta = angles[i]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Target position on intersection circle
        target_p2 = np.array(
            [
                x0 + r0 * (cos_theta * i_vec[0] + sin_theta * j_vec[0]),
                y0 + r0 * (cos_theta * i_vec[1] + sin_theta * j_vec[1]),
                z0 + r0 * (cos_theta * i_vec[2] + sin_theta * j_vec[2]),
            ]
        )

        # Vector from CM to target
        v2_rot = target_p2 - cm2
        norm_v2 = np.linalg.norm(v2_rot)

        # Determine rotation axis and angle
        if norm_v1 > 1e-9 and norm_v2 > 1e-9:
            v1_u = v1_rot / norm_v1
            v2_u = v2_rot / norm_v2
            dot_prod = np.dot(v1_u, v2_u)
            dot_prod = np.clip(dot_prod, -1.0, 1.0)

            if abs(dot_prod) > 1.0 - 1e-9:
                # Vectors are parallel or anti-parallel
                if dot_prod < 0:
                    # Anti-parallel: 180 degree rotation
                    rot_angle = np.pi
                    # Choose perpendicular axis
                    if abs(v1_u[0]) < 1e-9 and abs(v1_u[1]) < 1e-9:
                        rot_axis = np.array([1.0, 0.0, 0.0])
                    else:
                        rot_axis = np.array([-v1_u[1], v1_u[0], 0.0])
                        rot_axis /= np.linalg.norm(rot_axis)
                else:
                    # Parallel: no rotation needed
                    rotated_clusters[i] = coords2_in.copy()
                    continue
            else:
                # Normal case: compute rotation axis and angle
                rot_angle = np.arccos(dot_prod)
                rot_axis = np.cross(v1_u, v2_u)
                rot_axis /= np.linalg.norm(rot_axis)

            # Rotate cluster around CM
            coords_centered = coords2_in - cm2
            coords_rotated = rodrigues_rotation(coords_centered, rot_axis, rot_angle)
            rotated_clusters[i] = coords_rotated + cm2
        else:
            # Degenerate case: no rotation
            rotated_clusters[i] = coords2_in.copy()

    return rotated_clusters


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

    Note
    ----
    https://mathworld.wolfram.com/Sphere-SphereIntersection.html
    """
    invalid_ret = (0.0, 0.0, 0.0, 0.0, np.zeros(4), np.zeros(3), np.zeros(3), False)

    point1 = sphere_1[:3]
    point2 = sphere_2[:3]
    r1 = sphere_1[3]
    r2 = sphere_2[3]

    dp = point2 - point1
    distance = np.linalg.norm(dp)

    if distance > r1 + r2:
        logger.debug(
            f"TSI: Spheres too far apart (d={distance:.4f}, r1+r2={r1 + r2:.4f})"
        )
        return invalid_ret
    if distance < abs(r1 - r2):
        logger.debug(
            f"TSI: Sphere contained within other (d={distance:.4f}, |r1-r2|={abs(r1 - r2):.4f})"
        )
        return invalid_ret

    k_vec = dp / distance
    plane_distance = (distance**2 + r1**2 - r2**2) / (2 * distance)
    center0 = point1 + plane_distance * k_vec
    r0 = np.sqrt(r1**2 - plane_distance**2)

    if abs(np.dot(k_vec, np.array([1.0, 0.0, 0.0]))) < 1 / np.sqrt(3):
        cross_ref = np.array([1.0, 0.0, 0.0])
    elif abs(np.dot(k_vec, np.array([0.0, 1.0, 0.0]))) < 1 / np.sqrt(3):
        cross_ref = np.array([0.0, 1.0, 0.0])
    else:
        cross_ref = np.array([0.0, 0.0, 1.0])
    i_vec = np.cross(k_vec, cross_ref)
    i_vec /= np.linalg.norm(i_vec)
    j_vec = np.cross(i_vec, k_vec)
    j_vec /= np.linalg.norm(j_vec)

    theta = 2.0 * np.pi * np.random.rand()

    center_k = center0 + r0 * (np.cos(theta) * i_vec + np.sin(theta) * j_vec)
    if np.any(np.isnan(center_k)):
        print("""
              Seems like these two spheres do no intersect!
              Open up an issue and inform us about it :)
              """)
        return invalid_ret

    return (
        center_k[0],
        center_k[1],
        center_k[2],
        theta,
        np.array([center0[0], center0[1], center0[2], r0]),
        i_vec,
        j_vec,
        True,
    )

    # x1, y1, z1, r1 = sphere_1
    # x2, y2, z2, r2 = sphere_2
    # center1 = sphere_1[:3]
    # center2 = sphere_2[:3]
    # v12 = center2 - center1
    # distance = np.linalg.norm(v12)

    # # Default invalid return values
    # invalid_ret = (0.0, 0.0, 0.0, 0.0, np.zeros(4), np.zeros(3), np.zeros(3), False)

    # # --- Check for edge cases ---
    # # 1. Spheres are too far apart
    # if distance > r1 + r2 + FLOATING_POINT_ERROR:
    #     logger.debug(
    #         f"TSI: Spheres too far apart (d={distance:.4f}, r1+r2={r1 + r2:.4f})"
    #     )
    #     return invalid_ret
    # # 2. One sphere is contained within the other without touching
    # if distance < abs(r1 - r2) - FLOATING_POINT_ERROR:
    #     logger.debug(
    #         f"TSI: Sphere contained within other (d={distance:.4f}, |r1-r2|={abs(r1 - r2):.4f})"
    #     )
    #     return invalid_ret
    # # TODO: check 2 and 3 should be the same?
    # # 3. Spheres coincide
    # if distance < FLOATING_POINT_ERROR and abs(r1 - r2) < FLOATING_POINT_ERROR:
    #     logger.debug("TSI: Spheres coincide")
    #     # Intersection is the whole sphere surface - requires different handling if needed
    #     return invalid_ret  # Cannot define a unique circle

    # # --- Handle Touching Point Case ---
    # is_touching = False
    # touch_point = np.zeros(3)
    # if abs(distance - (r1 + r2)) < FLOATING_POINT_ERROR:  # Touching externally
    #     is_touching = True
    #     # Point is on the line segment between centers
    #     if distance > FLOATING_POINT_ERROR:
    #         touch_point = center1 + v12 * (r1 / distance)
    #     else:  # Should be caught by coincident case, but fallback
    #         touch_point = center1
    # elif abs(distance - abs(r1 - r2)) < FLOATING_POINT_ERROR:  # Touching internally
    #     is_touching = True
    #     # Point is on the line extending from centers
    #     if distance > FLOATING_POINT_ERROR:
    #         if r1 > r2:
    #             touch_point = center1 + v12 * (r1 / distance)
    #         else:  # r2 > r1
    #             touch_point = center2 + (-v12) * (
    #                 r2 / distance
    #             )  # Point on sphere 2 surface
    #     else:  # Should be caught by coincident case
    #         touch_point = center1

    # if is_touching:
    #     logger.debug(f"TSI: Spheres touching at point {touch_point}")
    #     # Return the single point, theta=0, r0=0
    #     vec_0_touch = np.concatenate((touch_point, [0.0]))
    #     # i_vec, j_vec are ill-defined, return zeros
    #     return (
    #         touch_point[0],
    #         touch_point[1],
    #         touch_point[2],
    #         0.0,
    #         vec_0_touch,
    #         np.zeros(3),
    #         np.zeros(3),
    #         True,
    #     )

    # # --- Standard Intersection Case (Circle) ---
    # try:
    #     # distance 'd' is already computed
    #     # distance from center1 to intersection plane:
    #     dist1_plane = (distance**2 - r2**2 + r1**2) / (2 * distance)

    #     # Radius of the intersection circle squared
    #     r0_sq = r1**2 - dist1_plane**2
    #     if r0_sq < -FLOATING_POINT_ERROR:  # Tolerance check for numerical issues
    #         logger.warning(
    #             f"TSI: Negative r0^2 ({r0_sq}) in sphere intersection. d={distance}, r1={r1}, r2={r2}"
    #         )
    #         return invalid_ret
    #     r0 = np.sqrt(max(0.0, r0_sq))  # Ensure non-negative before sqrt

    #     # Center of the intersection circle
    #     unit_v12 = v12 / distance
    #     center0 = center1 + unit_v12 * dist1_plane
    #     x0, y0, z0 = center0

    #     # Define basis vectors for the intersection plane (perpendicular to v12)
    #     # k_vec is the normal to the plane (unit_v12)
    #     k_vec = unit_v12

    #     # Find a vector i_vec perpendicular to k_vec robustly
    #     # If k_vec is close to x-axis, use y-axis for cross product, otherwise use x-axis
    #     if abs(np.dot(k_vec, np.array([1.0, 0.0, 0.0]))) < 0.9:
    #         cross_ref = np.array([1.0, 0.0, 0.0])
    #     else:
    #         cross_ref = np.array([0.0, 1.0, 0.0])

    #     j_vec = normalize(cross_product(k_vec, cross_ref))
    #     i_vec = normalize(cross_product(j_vec, k_vec))
    #     # Generate random angle theta
    #     theta = 2.0 * np.pi * np.random.rand()

    #     # Calculate random point on the circle
    #     point_on_circle = (
    #         center0 + r0 * np.cos(theta) * i_vec + r0 * np.sin(theta) * j_vec
    #     )
    #     x, y, z = point_on_circle

    #     vec_0 = np.array([x0, y0, z0, r0])
    #     return x, y, z, theta, vec_0, i_vec, j_vec, True

    # except (ZeroDivisionError, ValueError) as e:
    #     logger.error(f"Error during sphere intersection calculation: {e}")
    #     return invalid_ret


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
    """Auto-dispatch to parallel or sequential overlap check based on size.

    For large cluster pairs, uses parallel version without early termination.
    For small pairs, uses sequential with early exit.

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
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    total_pairs = n1 * n2

    if total_pairs > PARALLEL_OVERLAP_THRESHOLD:
        # Many pairs: use parallel (no early termination)
        return calculate_max_overlap_cca_parallel(coords1, radii1, coords2, radii2)
    else:
        # Few pairs: use sequential with early termination
        return calculate_max_overlap_cca_fast(
            coords1, radii1, coords2, radii2, tolerance
        )
