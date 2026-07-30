"""CCA-specific JIT kernels for PyFracVAL.

JIT-compiled helper functions used during CCA sticking and retry operations.

These kernels accelerate retry and overlap stages in the CCA procedure from
:cite:p:`Moran2019FracVAL`.

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
import math

import numpy as np
from numba import jit, prange

from .geometry import _two_sphere_intersection_kernel, rodrigues_rotation

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


@jit(nopython=True, fastmath=True, cache=True)
def _rotate_about_point(coords, centre, ax, ay, az, angle):
    """Rotate every row of `coords` about `centre` by `angle` around axis (ax,ay,az).

    Axis need not be normalised; a degenerate axis or angle leaves the
    coordinates untouched.
    """
    n = coords.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    axis_norm = math.sqrt(ax * ax + ay * ay + az * az)
    if axis_norm < 1e-12 or abs(angle) < 1e-12:
        for i in range(n):
            out[i, 0] = coords[i, 0]
            out[i, 1] = coords[i, 1]
            out[i, 2] = coords[i, 2]
        return out

    inv = 1.0 / axis_norm
    kx = ax * inv
    ky = ay * inv
    kz = az * inv
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    one_minus_cos = 1.0 - cos_a
    cx0 = centre[0]
    cy0 = centre[1]
    cz0 = centre[2]

    for i in range(n):
        vx = coords[i, 0] - cx0
        vy = coords[i, 1] - cy0
        vz = coords[i, 2] - cz0
        kdv = kx * vx + ky * vy + kz * vz
        crx = ky * vz - kz * vy
        cry = kz * vx - kx * vz
        crz = kx * vy - ky * vx
        out[i, 0] = cx0 + vx * cos_a + crx * sin_a + kx * kdv * one_minus_cos
        out[i, 1] = cy0 + vy * cos_a + cry * sin_a + ky * kdv * one_minus_cos
        out[i, 2] = cz0 + vz * cos_a + crz * sin_a + kz * kdv * one_minus_cos
    return out


@jit(nopython=True, fastmath=True, cache=True)
def _align_rotation(v1x, v1y, v1z, v2x, v2y, v2z):
    """Axis and angle rotating vector v1 onto vector v2.

    Returns ``(ax, ay, az, angle, do_rotate)``. Mirrors the branch
    structure of the interpreted implementation exactly, including the
    anti-parallel case (where any perpendicular axis is valid) and the
    already-aligned case (where no rotation is applied at all).
    """
    n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0, 0.0, 0.0, 0.0, False

    u1x = v1x / n1
    u1y = v1y / n1
    u1z = v1z / n1
    u2x = v2x / n2
    u2y = v2y / n2
    u2z = v2z / n2
    dot = u1x * u2x + u1y * u2y + u1z * u2z

    if abs(dot) > 1.0 - 1e-9:
        if dot < 0.0:
            # Anti-parallel: pick any axis perpendicular to u1.
            if abs(u1x) < 1e-9 and abs(u1y) < 1e-9:
                return 1.0, 0.0, 0.0, math.pi, True
            return -u1y, u1x, 0.0, math.pi, True
        return 0.0, 0.0, 0.0, 0.0, False

    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    angle = math.acos(dot)
    ax = u1y * u2z - u1z * u2y
    ay = u1z * u2x - u1x * u2z
    az = u1x * u2y - u1y * u2x
    return ax, ay, az, angle, True


@jit(nopython=True, fastmath=True, cache=True)
def cca_sticking_v1_kernel(
    coords1_in,
    radii1,
    cm1,
    coords2_in,
    radii2,
    cm2_in,
    cand1_idx,
    cand2_idx,
    gamma_pc,
    theta_a,
    theta_b,
):
    """Fused ``ext_case=0`` sticking placement.

    The interpreted version of this spends most of its time on numpy
    dispatch for 3-vectors rather than on arithmetic: per N=1024
    aggregate it drove ~30k norm calls, ~24k array allocations and ~20k
    zeros. Fusing the whole placement into one compiled function removes
    all of that, and lets the two cluster transforms run as tight loops.

    Randomness is hoisted out: the two angles the sphere-sphere
    intersections would have sampled are passed in, so this stays pure
    and the caller keeps sole ownership of the RNG stream (which is what
    makes runs reproducible).

    Returns ``(coords1_out, coords2_out, cm2_out, vec0, i_vec, j_vec,
    ok)``; ``ok`` is False when either intersection has no solution, in
    which case the arrays are meaningless and the caller must skip them.
    """
    zeros4 = np.zeros(4, dtype=np.float64)
    zeros3 = np.zeros(3, dtype=np.float64)

    coords2 = coords2_in.copy()

    # --- Step 1: translate cluster 2 so |CM2 - CM1| == gamma_pc ---
    vx = coords1_in[cand1_idx, 0] - cm1[0]
    vy = coords1_in[cand1_idx, 1] - cm1[1]
    vz = coords1_in[cand1_idx, 2] - cm1[2]
    vnorm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vnorm < 1e-12:
        vx, vy, vz = 1.0, 0.0, 0.0
    else:
        vx /= vnorm
        vy /= vnorm
        vz /= vnorm

    cm2x = cm1[0] + gamma_pc * vx
    cm2y = cm1[1] + gamma_pc * vy
    cm2z = cm1[2] + gamma_pc * vz
    dx = cm2x - cm2_in[0]
    dy = cm2y - cm2_in[1]
    dz = cm2z - cm2_in[2]
    for i in range(coords2.shape[0]):
        coords2[i, 0] += dx
        coords2[i, 1] += dy
        coords2[i, 2] += dz

    cm2 = np.empty(3, dtype=np.float64)
    cm2[0] = cm2x
    cm2[1] = cm2y
    cm2[2] = cm2z

    # --- Step 2: first contact point, on the D1max/D2max spheres ---
    p1x = coords1_in[cand1_idx, 0]
    p1y = coords1_in[cand1_idx, 1]
    p1z = coords1_in[cand1_idx, 2]
    d1 = math.sqrt((p1x - cm1[0]) ** 2 + (p1y - cm1[1]) ** 2 + (p1z - cm1[2]) ** 2)
    d1_max = d1 + radii1[cand1_idx]

    q2x = coords2[cand2_idx, 0]
    q2y = coords2[cand2_idx, 1]
    q2z = coords2[cand2_idx, 2]
    d2 = math.sqrt((q2x - cm2x) ** 2 + (q2y - cm2y) ** 2 + (q2z - cm2z) ** 2)
    d2_max = d2 + radii2[cand2_idx]

    sph1 = np.empty(4, dtype=np.float64)
    sph1[0] = cm1[0]
    sph1[1] = cm1[1]
    sph1[2] = cm1[2]
    sph1[3] = d1_max
    sph2 = np.empty(4, dtype=np.float64)
    sph2[0] = cm2x
    sph2[1] = cm2y
    sph2[2] = cm2z
    sph2[3] = d2_max

    (cx, cy, cz, _x0, _y0, _z0, _r0, _ix, _iy, _iz, _jx, _jy, _jz, valid) = (
        _two_sphere_intersection_kernel(sph1, sph2, theta_a)
    )
    if not valid:
        return coords1_in, coords2, cm2, zeros4, zeros3, zeros3, False

    # Push the sampled point out onto candidate 1's own surface.
    ux = cx - p1x
    uy = cy - p1y
    uz = cz - p1z
    unorm = math.sqrt(ux * ux + uy * uy + uz * uz)
    if unorm < 1e-9:
        tx = p1x - cm1[0]
        ty = p1y - cm1[1]
        tz = p1z - cm1[2]
        tnorm = math.sqrt(tx * tx + ty * ty + tz * tz)
        target_x = p1x + radii1[cand1_idx] * tx / tnorm
        target_y = p1y + radii1[cand1_idx] * ty / tnorm
        target_z = p1z + radii1[cand1_idx] * tz / tnorm
    else:
        target_x = p1x + radii1[cand1_idx] * ux / unorm
        target_y = p1y + radii1[cand1_idx] * uy / unorm
        target_z = p1z + radii1[cand1_idx] * uz / unorm

    # --- Step 3: rotate cluster 1 so candidate 1 reaches that point ---
    ax, ay, az, angle, do_rot = _align_rotation(
        p1x - cm1[0],
        p1y - cm1[1],
        p1z - cm1[2],
        target_x - cm1[0],
        target_y - cm1[1],
        target_z - cm1[2],
    )
    if do_rot:
        coords1 = _rotate_about_point(coords1_in, cm1, ax, ay, az, angle)
    else:
        coords1 = coords1_in.copy()

    # --- Step 4: second contact point (point-touch between candidates) ---
    a_x = coords1[cand1_idx, 0]
    a_y = coords1[cand1_idx, 1]
    a_z = coords1[cand1_idx, 2]
    sphA = np.empty(4, dtype=np.float64)
    sphA[0] = a_x
    sphA[1] = a_y
    sphA[2] = a_z
    sphA[3] = radii1[cand1_idx] + radii2[cand2_idx]

    b_x = coords2[cand2_idx, 0]
    b_y = coords2[cand2_idx, 1]
    b_z = coords2[cand2_idx, 2]
    radius_b = math.sqrt((b_x - cm2x) ** 2 + (b_y - cm2y) ** 2 + (b_z - cm2z) ** 2)
    sphB = np.empty(4, dtype=np.float64)
    sphB[0] = cm2x
    sphB[1] = cm2y
    sphB[2] = cm2z
    sphB[3] = radius_b

    (ex, ey, ez, x0, y0, z0, r0, ix, iy, iz, jx, jy, jz, valid2) = (
        _two_sphere_intersection_kernel(sphA, sphB, theta_b)
    )
    if not valid2:
        return coords1, coords2, cm2, zeros4, zeros3, zeros3, False

    # --- Step 5: rotate cluster 2 so candidate 2 reaches that point ---
    ax2, ay2, az2, angle2, do_rot2 = _align_rotation(
        b_x - cm2x, b_y - cm2y, b_z - cm2z, ex - cm2x, ey - cm2y, ez - cm2z
    )
    if do_rot2:
        coords2 = _rotate_about_point(coords2, cm2, ax2, ay2, az2, angle2)

    vec0 = np.empty(4, dtype=np.float64)
    vec0[0] = x0
    vec0[1] = y0
    vec0[2] = z0
    vec0[3] = r0
    i_vec = np.empty(3, dtype=np.float64)
    i_vec[0] = ix
    i_vec[1] = iy
    i_vec[2] = iz
    j_vec = np.empty(3, dtype=np.float64)
    j_vec[0] = jx
    j_vec[1] = jy
    j_vec[2] = jz
    return coords1, coords2, cm2, vec0, i_vec, j_vec, True
