# utils.py
"""Utility functions for vector operations and array manipulation."""

import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import jit, prange

logger = logging.getLogger(__name__)

FLOATING_POINT_ERROR = 1e-9


def shuffle_array(
    arr: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Randomly shuffle elements of a 1D array in-place (Fisher-Yates).

    Modifies the input array directly. Mimics Fortran randsample behavior.

    Parameters
    ----------
    arr :
        The 1D NumPy array to shuffle.
    rng : np.random.Generator | None, optional
        A NumPy Generator instance for reproducible randomness. If None,
        a fresh Generator is created.

    Returns
    -------
        The input `arr`, modified in-place.
    """
    _rng = rng if rng is not None else np.random.default_rng()
    n = len(arr)
    for i in range(n - 1):
        # Random index from i to n-1
        j = int(_rng.integers(i, n))
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
    all_radii: npt.NDArray | None = None,
) -> tuple[bool, float]:
    """
    Calculates Gamma_pc for adding the next monomer (aggregate 2).

    Parameters
    ----------
    all_radii : np.ndarray, optional
        If provided, the geometric mean radius for rg3 is computed from this
        full set of radii (matching Fortran behaviour where R contains all N
        particles). When None the geometric mean is taken from the local
        combined set (radii1 + radii2).
    """
    n1 = radii1.size
    n2 = radii2.size

    n3 = n1 + n2
    m3 = m1 + m2

    if heuristic:
        m1 = n1
        m2 = n2
        m3 = n3

    # Use the full particle set for the geometric-mean radius when available,
    # matching Fortran: rg3 = geomean(R_all) * (n3/kf)^(1/Df)
    rg3_radii = all_radii if all_radii is not None else np.concatenate((radii1, radii2))
    rg3 = calculate_rg(rg3_radii, n3, df, kf)

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

    # Explicitly check for non-real conditions before calling sqrt.
    # np.sqrt(negative) produces nan + RuntimeWarning (not a Python exception),
    # so a try/except is insufficient — we must guard explicitly.
    if denominator <= 0.0:
        logger.warning(
            f"Gamma_pc calculation: denominator={denominator:.2e} <= 0 "
            f"(n1={n1}, m1={m1:.2e}, n2={n2}, m2={m2:.2e})"
        )
        gamma_real = False
    elif radicand < 0.0:
        logger.debug(
            f"Gamma_pc calculation: radicand={radicand:.2e} < 0 (non-real result). "
            f"n1={n1}, rg1={rg1:.2e}, n2={n2}, rg2={rg2:.2e}, rg3={rg3:.2e}"
        )
        gamma_real = False
    else:
        try:
            val = radicand / denominator
            gamma_pc = float(np.sqrt(val))
            # Sanity check: sqrt should never produce nan/inf here, but guard anyway
            if not np.isfinite(gamma_pc):
                logger.warning(
                    f"Gamma_pc calculation: non-finite result {gamma_pc} "
                    f"(radicand={radicand:.2e}, denominator={denominator:.2e})"
                )
                gamma_pc = 0.0
                gamma_real = False
            else:
                gamma_real = True
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logger.warning(f"Gamma calculation internal failed: {e}")
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
        # FIX (PyFracVAL-31m): delegate to JIT-compiled fast path
        return _rodrigues_rotation_2d(vectors, axis, cos_a, sin_a)

    # elif vectors.ndim > 2:
    else:
        raise ValueError("Input vectors must be 3D or Nx3")
    v_rot = vectors * cos_a + cross_kv * sin_a + axis * dot_kv * (1.0 - cos_a)

    return v_rot


@jit(nopython=True, fastmath=True, cache=True)
def _rodrigues_rotation_2d(
    vectors: np.ndarray, axis: np.ndarray, cos_a: float, sin_a: float
) -> np.ndarray:
    """JIT-compiled Rodrigues rotation for Nx3 arrays (PyFracVAL-31m).

    axis must already be normalised before calling.
    cos_a and sin_a must be pre-computed by the caller.

    Parameters
    ----------
    vectors : np.ndarray
        Shape (N, 3) array of vectors to rotate.
    axis : np.ndarray
        Shape (3,) normalised rotation axis.
    cos_a : float
        cos(angle)
    sin_a : float
        sin(angle)

    Returns
    -------
    np.ndarray
        Shape (N, 3) rotated vectors.
    """
    n = vectors.shape[0]
    result = np.empty((n, 3), dtype=vectors.dtype)
    kx, ky, kz = axis[0], axis[1], axis[2]
    one_minus_cos = 1.0 - cos_a
    for i in range(n):
        vx = vectors[i, 0]
        vy = vectors[i, 1]
        vz = vectors[i, 2]
        # dot(k, v)
        kdv = kx * vx + ky * vy + kz * vz
        # cross(k, v)
        cx = ky * vz - kz * vy
        cy = kz * vx - kx * vz
        cz = kx * vy - ky * vx
        result[i, 0] = vx * cos_a + cx * sin_a + kx * kdv * one_minus_cos
        result[i, 1] = vy * cos_a + cy * sin_a + ky * kdv * one_minus_cos
        result[i, 2] = vz * cos_a + cz * sin_a + kz * kdv * one_minus_cos
    return result


# Golden ratio constant for Fibonacci spiral (same as config.GOLDEN_RATIO)
_GOLDEN_RATIO = (1.0 + 2.23606797749979) / 2.0  # (1 + sqrt(5)) / 2
_TWO_PI = 6.283185307179586  # 2 * pi


@jit(nopython=True, fastmath=True, cache=True)
def _cca_reintento_kernel(
    coords2_in: np.ndarray,
    cm2: np.ndarray,
    cand2_idx: int,
    x0: float,
    y0: float,
    z0: float,
    r0: float,
    ivx: float,
    ivy: float,
    ivz: float,
    jvx: float,
    jvy: float,
    jvz: float,
    attempt: int,
) -> np.ndarray:
    """JIT-compiled CCA rotation kernel (PyFracVAL-dsa).

    Computes the Fibonacci-spiral rotation of cluster2 to its next candidate
    position on the intersection circle.  Replaces the Python-level
    _cca_reintento method body to eliminate CPython dispatch and scalar
    overhead for every Fibonacci step.

    Parameters
    ----------
    coords2_in : np.ndarray, shape (n2, 3)
        Current absolute coordinates of cluster 2.
    cm2 : np.ndarray, shape (3,)
        Centre-of-mass of cluster 2 (constant throughout rotation loop).
    cand2_idx : int
        Index of the candidate contact particle in cluster 2.
    x0, y0, z0, r0 : float
        Centre and radius of the intersection circle (vec_0 unpacked).
    ivx, ivy, ivz : float
        First basis vector of the intersection circle plane (i_vec unpacked).
    jvx, jvy, jvz : float
        Second basis vector of the intersection circle plane (j_vec unpacked).
    attempt : int
        Fibonacci step index (1-indexed).

    Returns
    -------
    np.ndarray, shape (n2, 3)
        Rotated coordinates.  Returns ``coords2_in`` unchanged when no
        rotation is needed (parallel to avoid a copy).
    """
    # --- 1. Target point on intersection circle (Fibonacci spiral) ----------
    theta = _TWO_PI * attempt / _GOLDEN_RATIO
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    tp_x = x0 + r0 * cos_t * ivx + r0 * sin_t * jvx
    tp_y = y0 + r0 * cos_t * ivy + r0 * sin_t * jvy
    tp_z = z0 + r0 * cos_t * ivz + r0 * sin_t * jvz

    # --- 2. Rotation axis and angle -----------------------------------------
    # v1 = current position of cand2 particle relative to cm2
    cm2x = cm2[0]
    cm2y = cm2[1]
    cm2z = cm2[2]
    v1x = coords2_in[cand2_idx, 0] - cm2x
    v1y = coords2_in[cand2_idx, 1] - cm2y
    v1z = coords2_in[cand2_idx, 2] - cm2z
    # v2 = target position relative to cm2
    v2x = tp_x - cm2x
    v2y = tp_y - cm2y
    v2z = tp_z - cm2z

    norm_v1 = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    norm_v2 = np.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)

    if norm_v1 < 1e-9 or norm_v2 < 1e-9:
        return coords2_in  # No rotation possible

    # Normalise
    u1x = v1x / norm_v1
    u1y = v1y / norm_v1
    u1z = v1z / norm_v1
    u2x = v2x / norm_v2
    u2y = v2y / norm_v2
    u2z = v2z / norm_v2

    dot = u1x * u2x + u1y * u2y + u1z * u2z

    if dot > 1.0 - 1e-9:
        # Already aligned — nothing to do
        return coords2_in

    rot_angle: float
    rax: float
    ray: float
    raz: float

    if dot < -(1.0 - 1e-9):
        # Anti-parallel — rotate 180° around a perpendicular axis
        rot_angle = 3.141592653589793  # pi
        if abs(u1x) < 1e-9 and abs(u1y) < 1e-9:
            rax = 1.0
            ray = 0.0
            raz = 0.0
        else:
            rax = -u1y
            ray = u1x
            raz = 0.0
    else:
        rot_angle = np.arccos(dot)
        # cross(u1, u2)
        rax = u1y * u2z - u1z * u2y
        ray = u1z * u2x - u1x * u2z
        raz = u1x * u2y - u1y * u2x

    # Normalise rotation axis
    rn = np.sqrt(rax * rax + ray * ray + raz * raz)
    if rn < 1e-9 or abs(rot_angle) < 1e-9:
        return coords2_in  # Degenerate — skip

    rax /= rn
    ray /= rn
    raz /= rn

    # --- 3. Apply Rodrigues rotation to all particles in cluster 2 ----------
    cos_a = np.cos(rot_angle)
    sin_a = np.sin(rot_angle)
    one_minus_cos = 1.0 - cos_a

    n2 = coords2_in.shape[0]
    result = np.empty((n2, 3), dtype=coords2_in.dtype)
    for i in range(n2):
        # Translate to cm2-centred frame
        vx = coords2_in[i, 0] - cm2x
        vy = coords2_in[i, 1] - cm2y
        vz = coords2_in[i, 2] - cm2z
        # Rodrigues: v_rot = v*cos + (k×v)*sin + k*(k·v)*(1-cos)
        kdv = rax * vx + ray * vy + raz * vz
        cx = ray * vz - raz * vy
        cy = raz * vx - rax * vz
        cz = rax * vy - ray * vx
        result[i, 0] = vx * cos_a + cx * sin_a + rax * kdv * one_minus_cos + cm2x
        result[i, 1] = vy * cos_a + cy * sin_a + ray * kdv * one_minus_cos + cm2y
        result[i, 2] = vz * cos_a + cz * sin_a + raz * kdv * one_minus_cos + cm2z
    return result


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


@jit(nopython=True, fastmath=True, cache=True)
def _two_sphere_intersection_kernel(
    sphere_1: np.ndarray, sphere_2: np.ndarray, theta: float
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    bool,
]:
    """JIT-compiled kernel for two_sphere_intersection (PyFracVAL-du0).

    Given two spheres and a pre-sampled angle theta, computes the intersection
    circle geometry and returns a point on the circle at that angle.

    Parameters
    ----------
    sphere_1 : np.ndarray
        [x1, y1, z1, r1]
    sphere_2 : np.ndarray
        [x2, y2, z2, r2]
    theta : float
        Pre-sampled angle in [0, 2*pi) for the point on the intersection circle.

    Returns
    -------
    tuple of 14 scalars: x, y, z, x0, y0, z0, r0, ix, iy, iz, jx, jy, jz, valid
        x, y, z    - point on the intersection circle at angle theta
        x0, y0, z0 - center of the intersection circle
        r0         - radius of the intersection circle
        ix, iy, iz - first basis vector of the intersection plane
        jx, jy, jz - second basis vector of the intersection plane
        valid      - True if intersection exists, False otherwise
    """
    x1 = sphere_1[0]
    y1 = sphere_1[1]
    z1 = sphere_1[2]
    r1 = sphere_1[3]

    x2 = sphere_2[0]
    y2 = sphere_2[1]
    z2 = sphere_2[2]
    r2 = sphere_2[3]

    dpx = x2 - x1
    dpy = y2 - y1
    dpz = z2 - z1
    distance = np.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)

    _invalid = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    if distance > r1 + r2:
        return _invalid
    if distance < abs(r1 - r2):
        return _invalid

    inv_d = 1.0 / distance
    kx = dpx * inv_d
    ky = dpy * inv_d
    kz = dpz * inv_d

    plane_distance = (distance * distance + r1 * r1 - r2 * r2) * (0.5 * inv_d)
    x0 = x1 + plane_distance * kx
    y0 = y1 + plane_distance * ky
    z0 = z1 + plane_distance * kz
    r0_sq = r1 * r1 - plane_distance * plane_distance
    r0 = np.sqrt(r0_sq) if r0_sq > 0.0 else 0.0

    # Choose cross-reference vector least aligned with k_vec
    abs_kx = abs(kx)
    abs_ky = abs(ky)
    abs_kz = abs(kz)
    inv_sqrt3 = 1.0 / np.sqrt(3.0)
    if abs_kx < inv_sqrt3:
        cx, cy, cz = 1.0, 0.0, 0.0
    elif abs_ky < inv_sqrt3:
        cx, cy, cz = 0.0, 1.0, 0.0
    else:
        cx, cy, cz = 0.0, 0.0, 1.0

    # i_vec = cross(k_vec, cross_ref), normalised
    ix = ky * cz - kz * cy
    iy = kz * cx - kx * cz
    iz = kx * cy - ky * cx
    i_norm = np.sqrt(ix * ix + iy * iy + iz * iz)
    inv_i = 1.0 / i_norm
    ix *= inv_i
    iy *= inv_i
    iz *= inv_i

    # j_vec = cross(i_vec, k_vec), normalised
    jx = iy * kz - iz * ky
    jy = iz * kx - ix * kz
    jz = ix * ky - iy * kx
    j_norm = np.sqrt(jx * jx + jy * jy + jz * jz)
    inv_j = 1.0 / j_norm
    jx *= inv_j
    jy *= inv_j
    jz *= inv_j

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = x0 + r0 * (cos_t * ix + sin_t * jx)
    y = y0 + r0 * (cos_t * iy + sin_t * jy)
    z = z0 + r0 * (cos_t * iz + sin_t * jz)

    return x, y, z, x0, y0, z0, r0, ix, iy, iz, jx, jy, jz, True


def two_sphere_intersection(
    sphere_1: np.ndarray, sphere_2: np.ndarray, rng: np.random.Generator | None = None
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
    _rng = rng if rng is not None else np.random.default_rng()
    invalid_ret = (0.0, 0.0, 0.0, 0.0, np.zeros(4), np.zeros(3), np.zeros(3), False)

    theta = 2.0 * np.pi * _rng.random()

    x, y, z, x0, y0, z0, r0, ix, iy, iz, jx, jy, jz, valid = (
        _two_sphere_intersection_kernel(sphere_1, sphere_2, theta)
    )

    if not valid:
        r1 = sphere_1[3]
        r2 = sphere_2[3]
        distance = float(np.linalg.norm(sphere_2[:3] - sphere_1[:3]))
        if distance > r1 + r2:
            logger.debug(
                f"TSI: Spheres too far apart (d={distance:.4f}, r1+r2={r1 + r2:.4f})"
            )
        else:
            logger.debug(
                f"TSI: Sphere contained within other (d={distance:.4f}, |r1-r2|={abs(r1 - r2):.4f})"
            )
        return invalid_ret

    return (
        x,
        y,
        z,
        theta,
        np.array([x0, y0, z0, r0]),
        np.array([ix, iy, iz]),
        np.array([jx, jy, jz]),
        True,
    )


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


def compute_empirical_rg(coords: np.ndarray, radii: np.ndarray) -> float:
    """Compute empirical Rg directly from particle coordinates (mass-weighted).

    Unlike ``calculate_rg`` which uses the fractal scaling law
    Rg = a*(N/kf)^(1/Df), this function measures Rg from the actual
    spatial distribution of particles.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle center coordinates.
    radii : np.ndarray
        N array of particle radii. Mass is proportional to r^3.

    Returns
    -------
    float
        Empirical (mass-weighted) radius of gyration.
    """
    if coords.shape[0] == 0:
        return 0.0
    masses = radii**3
    total_mass = np.sum(masses)
    if total_mass < 1e-30:
        return 0.0
    cm = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    dist_sq = np.sum((coords - cm[np.newaxis, :]) ** 2, axis=1)
    return float(np.sqrt(np.sum(dist_sq * masses) / total_mass))


def compute_pair_correlation_dimensions(
    coords: np.ndarray,
    radii: np.ndarray,
    n_bins: int = 50,
) -> dict[str, np.ndarray]:
    """Estimate fractal dimension from pair-correlation (mass-radius) scaling.

    Computes the cumulative mass M(r) as a function of radial distance
    from the centre of mass. For a fractal aggregate,
    M(r) ~ r^Df, so a log-log fit gives the empirical Df.

    The fit uses raw (un-normalised) cumulative mass because normalised
    mass fractions are in [0,1] whose logs are non-positive, breaking
    the log-linear regression.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle center coordinates.
    radii : np.ndarray
        N array of particle radii.
    n_bins : int
        Number of radial bins (default 50).

    Returns
    -------
    dict with keys:
        r_bins : np.ndarray — bin edge radii (n_bins+1,)
        r_centers : np.ndarray — bin centre radii (n_bins,)
        M_r : np.ndarray — cumulative normalised mass fraction within each radius (n_bins+1,)
        empirical_Df : float — slope of log(M) vs log(r) fit
        fit_r_squared : float — R^2 of the linear fit
        empirical_kf : float — estimated kf from the fit
    """
    n = coords.shape[0]
    if n < 2:
        return {
            "r_bins": np.array([]),
            "r_centers": np.array([]),
            "M_r": np.array([]),
            "empirical_Df": 0.0,
            "fit_r_squared": 0.0,
            "empirical_kf": 0.0,
        }

    masses = radii**3
    total_mass = np.sum(masses)
    cm = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    distances = np.linalg.norm(coords - cm[np.newaxis, :], axis=1)

    r_min = np.min(distances[distances > 0]) * 0.5
    r_max = np.max(distances) * 0.99
    if r_min >= r_max or r_min <= 0:
        return {
            "r_bins": np.array([]),
            "r_centers": np.array([]),
            "M_r": np.array([]),
            "empirical_Df": 0.0,
            "fit_r_squared": 0.0,
            "empirical_kf": 0.0,
        }

    r_bins = np.linspace(r_min, r_max, n_bins + 1)
    cumulative_mass = np.zeros(n_bins + 1)
    for i in range(n_bins + 1):
        mask = distances <= r_bins[i]
        cumulative_mass[i] = np.sum(masses[mask])

    M_r = cumulative_mass / total_mass
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Use raw cumulative mass for log-log fit (normalised values < 1 have negative logs)
    log_r = np.log(r_centers)
    log_M_raw = np.log(cumulative_mass[1:])  # skip bin edge at r_min

    valid = np.isfinite(log_M_raw) & np.isfinite(log_r) & (cumulative_mass[1:] > 0)
    if np.sum(valid) < 3:
        return {
            "r_bins": r_bins,
            "r_centers": r_centers,
            "M_r": M_r,
            "empirical_Df": 0.0,
            "fit_r_squared": 0.0,
            "empirical_kf": 0.0,
        }

    coeffs = np.polyfit(log_r[valid], log_M_raw[valid], 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    ss_res = np.sum((log_M_raw[valid] - (slope * log_r[valid] + intercept)) ** 2)
    ss_tot = np.sum((log_M_raw[valid] - np.mean(log_M_raw[valid])) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Estimate kf from: Rg = a * (N/kf)^(1/Df) => kf = N / (Rg/a)^Df
    # Using empirical fit: M(r) = exp(intercept) * r^slope
    # At r = Rg: M = total_mass, so: total_mass = exp(intercept) * Rg^slope
    # kf from fractal scaling: N = kf * (Rg/a)^Df
    empirical_rg = compute_empirical_rg(coords, radii)
    geo_mean_r = np.exp(np.mean(np.log(radii[radii > 1e-12])))
    if slope > 0 and empirical_rg > 0 and geo_mean_r > 0:
        empirical_kf = n / (empirical_rg / geo_mean_r) ** slope
    else:
        empirical_kf = 0.0

    return {
        "r_bins": r_bins,
        "r_centers": r_centers,
        "M_r": M_r,
        "empirical_Df": float(slope),
        "fit_r_squared": float(r_squared),
        "empirical_kf": float(empirical_kf),
    }


def validate_fractal_structure(
    coords: np.ndarray,
    radii: np.ndarray,
    target_df: float,
    target_kf: float,
    rg_rtol: float = 0.05,
) -> dict[str, float]:
    """Validate that generated aggregate matches target fractal parameters.

    Compares theoretical Rg (from scaling law) vs empirical Rg (from
    coordinates), and estimates the actual fractal dimension from
    mass-radius scaling.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle center coordinates.
    radii : np.ndarray
        N array of particle radii.
    target_df : float
        Target fractal dimension.
    target_kf : float
        Target fractal prefactor.
    rg_rtol : float
        Relative tolerance for Rg agreement (default 5%).

    Returns
    -------
    dict with keys:
        N : int — number of particles
        theoretical_rg : float — Rg from scaling law Rg = a*(N/kf)^(1/Df)
        empirical_rg : float — Rg measured from coordinates
        rg_error_pct : float — (empirical - theoretical)/theoretical * 100
        rg_ok : bool — |rg_error_pct| < rg_rtol * 100
        empirical_Df : float — Df estimated from mass-radius scaling
        target_Df : float — target fractal dimension
        df_error : float — empirical_Df - target_Df
        fit_r_squared : float — goodness of fit for Df estimation
        empirical_kf : float — estimated fractal prefactor
        target_kf : float — target fractal prefactor
    """
    n = coords.shape[0]
    theoretical_rg = calculate_rg(radii, n, target_df, target_kf)
    empirical_rg = compute_empirical_rg(coords, radii)
    rg_error_pct = (
        (empirical_rg - theoretical_rg) / theoretical_rg * 100
        if theoretical_rg > 0
        else 0.0
    )

    pair_corr = compute_pair_correlation_dimensions(coords, radii)

    return {
        "N": n,
        "theoretical_rg": theoretical_rg,
        "empirical_rg": empirical_rg,
        "rg_error_pct": rg_error_pct,
        "rg_ok": abs(rg_error_pct) < rg_rtol * 100,
        "empirical_Df": pair_corr["empirical_Df"],
        "target_Df": target_df,
        "df_error": pair_corr["empirical_Df"] - target_df,
        "fit_r_squared": pair_corr["fit_r_squared"],
        "empirical_kf": pair_corr["empirical_kf"],
        "target_kf": target_kf,
    }
