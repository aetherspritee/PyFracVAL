"""CCA-specific JIT kernels for PyFracVAL.

JIT-compiled helper functions used during CCA sticking and retry operations.

Constants
---------
_GOLDEN_RATIO
    Golden ratio constant used for Fibonacci spiral rotations.
_TWO_PI
    2 * pi constant for angle calculations.

Functions
---------
_cca_reintento_kernel
    JIT kernel for the CCA "reintento" (retry) overlap check.
batch_check_overlaps_cca
    JIT parallel overlap checker for CCA batch rotation.
batch_rotate_cluster_cca
    Rotate all positions in a CCA cluster around its centre of mass.
"""

import logging

import numpy as np
from numba import jit, prange

from .geometry import rodrigues_rotation

logger = logging.getLogger(__name__)

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
