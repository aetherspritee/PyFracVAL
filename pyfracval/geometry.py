"""Geometric primitives for PyFracVAL.

Rodrigues rotation, sphere intersection, and related constants.

Functions
---------
rodrigues_rotation
    Rotate vectors around an axis using Rodrigues' formula.
two_sphere_intersection
    Compute the intersection circle of two overlapping spheres.

Constants
---------
FLOATING_POINT_ERROR
    Numerical tolerance for floating-point comparisons.
"""

import logging
from typing import Tuple

import numpy as np
from numba import jit

logger = logging.getLogger(__name__)
FLOATING_POINT_ERROR = 1e-9


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
