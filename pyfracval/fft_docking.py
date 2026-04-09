"""FFT-based rigid-body docking for CCA sticking.

Implements the Katchalski-Katzir cross-correlation algorithm adapted for
fractal aggregate sticking. Replaces the stochastic Fibonacci search with
deterministic translation evaluation via 3D FFT.

The module voxellises two clusters onto a 3D grid, computes FFT
cross-correlation for each rotation of cluster 2, extracts top-K peaks,
and validates each candidate with exact overlap + gamma_pc checks.

Reference: Katchalski-Katzir et al., PNAS 89(6):2195-2199, 1992.
"""

import logging
import math
from itertools import product
from typing import Tuple

import numpy as np
from numba import jit, prange

from . import utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba JIT kernels (cache enabled for fast re-use)
# ---------------------------------------------------------------------------


@jit(nopython=True, fastmath=True, cache=True)
def _voxelize_kernel(
    coords: np.ndarray,
    radii: np.ndarray,
    grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    n: int,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    mode: int,
) -> int:
    """Fill a 3D grid with voxel values for *n* particles.

    Parameters
    ----------
    coords : (n, 3) particle centres
    radii  : (n,)   particle radii
    grid   : (gx, gy, gz) output grid (modified in-place, float64)
    origin : (3,)   grid origin in real coords
    voxel_size : float
    n      : number of particles
    grid_nx, grid_ny, grid_nz : grid dimensions
    mode   : 0 = occupancy only (+1 everywhere inside sphere),
             1 = interior (-1), surface shell (+1)

    Returns
    -------
    Number of non-zero voxels set.
    """
    count = 0
    shell_width = max(voxel_size * 1.5, 0.5)
    for p in range(n):
        cx = (coords[p, 0] - origin[0]) / voxel_size
        cy = (coords[p, 1] - origin[1]) / voxel_size
        cz = (coords[p, 2] - origin[2]) / voxel_size
        r = radii[p] / voxel_size
        r_inner = r - shell_width / voxel_size
        if r_inner < 0.0:
            r_inner = 0.0
        i_min = max(int(cx - r) - 1, 0)
        i_max = min(int(cx + r) + 2, grid_nx)
        j_min = max(int(cy - r) - 1, 0)
        j_max = min(int(cy + r) + 2, grid_ny)
        k_min = max(int(cz - r) - 1, 0)
        k_max = min(int(cz + r) + 2, grid_nz)
        for i in range(i_min, i_max):
            dx = float(i) - cx
            for j in range(j_min, j_max):
                dy = float(j) - cy
                for k in range(k_min, k_max):
                    dz = float(k) - cz
                    dist_sq = dx * dx + dy * dy + dz * dz
                    if dist_sq <= r * r:
                        if dist_sq >= r_inner * r_inner:
                            grid[i, j, k] = 1.0
                            count += 1
                        elif mode == 1:
                            grid[i, j, k] = -1.0
    return count


@jit(nopython=True, fastmath=True, cache=True)
def _overlap_check_kernel(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    tol_ov_sq: float,
) -> float:
    """Compute max fractional overlap between two clusters.

    Returns the maximum cov/max_overlap value. Zero means no overlap.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    cov_max = 0.0
    for i in range(n1):
        ri = radii1[i]
        xi = coords1[i, 0]
        yi = coords1[i, 1]
        zi = coords1[i, 2]
        for j in range(n2):
            rj = radii2[j]
            dx = xi - coords2[j, 0]
            dy = yi - coords2[j, 1]
            dz = zi - coords2[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            r_sum = ri + rj
            if dist_sq < r_sum * r_sum:
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-12
                cov = (r_sum - dist) / min(ri, rj)
                if cov > cov_max:
                    cov_max = cov
    return cov_max


# ---------------------------------------------------------------------------
# Rotation sampling on SO(3)
# ---------------------------------------------------------------------------


def sample_so3_rotations(
    n_rotations: int = 70,
) -> list[np.ndarray]:
    """Generate near-uniform rotation matrices on SO(3).

    Uses the Hopf fibration approach: sample points on S2 for the
    rotation axis direction, then Niño et al. twist angles.

    For n_rotations ~ 70 this gives roughly 15-degree resolution.

    Parameters
    ----------
    n_rotations : int
        Number of rotation matrices to generate.

    Returns
    -------
    List of (3, 3) rotation matrices.
    """
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    rotations = []
    n_axis = max(int(math.sqrt(n_rotations)), 1)
    n_twist = max(n_rotations // n_axis, 1)
    idx = 0
    for axis_i in range(n_axis):
        z = 1.0 - (axis_i / max(n_axis - 1, 1)) * 2.0 if n_axis > 1 else 1.0
        r_xy = math.sqrt(max(1.0 - z * z, 0.0))
        phi = golden_angle * axis_i
        ax = r_xy * math.cos(phi)
        ay = r_xy * math.sin(phi)
        az = z
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-12:
            ax, ay, az = 0.0, 0.0, 1.0
            norm = 1.0
        ax /= norm
        ay /= norm
        az /= norm
        for twist_j in range(n_twist):
            if idx >= n_rotations:
                return rotations
            theta = golden_angle * twist_j
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            omc = 1.0 - cos_t
            K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
            R = (
                np.eye(3) * cos_t
                + sin_t * K
                + omc * np.outer([ax, ay, az], [ax, ay, az])
            )
            if abs(np.linalg.det(R) - 1.0) > 1e-6:
                continue
            rotations.append(R)
            idx += 1
    return rotations


# ---------------------------------------------------------------------------
# Voxelisation and FFT correlation
# ---------------------------------------------------------------------------


def voxelize_cluster(
    coords: np.ndarray,
    radii: np.ndarray,
    cm: np.ndarray,
    grid_size: int,
    voxel_size: float,
    mode: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a particle cluster to a 3D voxel grid.

    Parameters
    ----------
    coords : (N, 3) particle centres.
    radii  : (N,) particle radii.
    cm     : (3,) centre-of-mass (grid will be centred here).
    grid_size : int, side length of the cubic grid.
    voxel_size : float, size of each voxel in simulation units.
    mode   : 0 = surface shell only (+1); 1 = interior (-1), surface (+1).

    Returns
    -------
    grid : (grid_size, grid_size, grid_size) int8 array
    origin : (3,) real-coordinate of grid[0,0,0] corner
    """
    half = grid_size / 2.0 * voxel_size
    origin = cm - half
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    n = coords.shape[0]

    count = _voxelize_kernel(
        coords,
        radii,
        grid,
        origin,
        voxel_size,
        n,
        grid_size,
        grid_size,
        grid_size,
        mode,
    )
    logger.debug(
        f"Voxelise mode={mode}: set {count} voxels out of {grid_size**3}, voxel_size={voxel_size:.3f}"
    )
    return grid, origin


def compute_fft_correlation(
    grid1: np.ndarray,
    grid2: np.ndarray,
) -> np.ndarray:
    """Compute 3D cross-correlation via FFT.

    Uses the convolution theorem: correlation(A, B) = IFFT(FFT(A) * conj(FFT(B))).

    Parameters
    ----------
    grid1 : (gx, gy, gz) receptor grid (cluster 1).
    grid2 : (gx, gy, gz) ligand grid (cluster 2, already rotated).

    Returns
    -------
    corr : (gx, gy, gz) float64 correlation surface.
    """
    fft1 = np.fft.fftn(grid1.astype(np.float64))
    fft2 = np.fft.fftn(grid2.astype(np.float64))
    corr = np.real(np.fft.ifftn(fft1 * np.conj(fft2)))
    return corr


def extract_top_k_peaks(
    corr: np.ndarray,
    k: int = 10,
    min_distance: int = 3,
) -> list[Tuple[float, int, int, int]]:
    """Extract top-k peaks from a 3D correlation surface.

    A peak must be the local maximum within a *min_distance* cube.
    Returns list of (score, dx, dy, dz) sorted by score descending,
    where (dx, dy, dz) is the translation offset in grid units.
    """
    from scipy.ndimage import maximum_filter, label

    neighbourhood_size = 2 * min_distance + 1
    local_max = maximum_filter(corr, size=neighbourhood_size)
    detected = corr == local_max

    detected[corr < np.max(corr) * 0.1] = False

    labeled, num_features = label(detected)
    if num_features == 0:
        return []

    peaks = []
    for feature_id in range(1, num_features + 1):
        feature_mask = labeled == feature_id
        feature_vals = corr * feature_mask
        peak_idx = np.unravel_index(np.argmax(feature_vals), corr.shape)
        peak_score = corr[peak_idx]
        peaks.append(
            (float(peak_score), int(peak_idx[0]), int(peak_idx[1]), int(peak_idx[2]))
        )
    peaks.sort(key=lambda x: x[0], reverse=True)
    return peaks[:k]


# ---------------------------------------------------------------------------
# Placement validation
# ---------------------------------------------------------------------------


def validate_placement(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    cm1: np.ndarray,
    cm2_target: np.ndarray,
    gamma_pc: float,
    gamma_real: bool,
    tol_ov: float,
    gamma_tolerance: float = 0.10,
) -> Tuple[bool, float, float]:
    """Validate a candidate placement for sticking.

    Checks:
    1. Distance between CMs is approximately gamma_pc.
    2. No particle overlap beyond tolerance.

    Parameters
    ----------
    coords1, radii1 : cluster 1 particles
    coords2, radii2 : cluster 2 particles (already placed at candidate position)
    cm1 : centre-of-mass of cluster 1
    cm2_target : centre-of-mass of cluster 2 at candidate position
    gamma_pc : target inter-cluster distance
    gamma_real : whether gamma_pc is physically meaningful
    tol_ov : maximum allowable overlap fraction
    gamma_tolerance : relative tolerance on gamma_pc distance check

    Returns
    -------
    (valid, distance, max_overlap)
    """
    if not gamma_real or gamma_pc <= 0:
        return False, 0.0, 0.0
    actual_dist = float(np.linalg.norm(cm2_target - cm1))
    dist_ratio = actual_dist / gamma_pc if gamma_pc > 1e-12 else float("inf")
    if dist_ratio < (1.0 - gamma_tolerance) or dist_ratio > (1.0 + gamma_tolerance):
        return False, actual_dist, 0.0
    cov_max = _overlap_check_kernel(
        coords1.astype(np.float64),
        radii1.astype(np.float64),
        coords2.astype(np.float64),
        radii2.astype(np.float64),
        tol_ov * tol_ov,
    )
    return cov_max <= tol_ov, actual_dist, cov_max


# ---------------------------------------------------------------------------
# Main FFT docking entry point
# ---------------------------------------------------------------------------


def fft_dock_sticking(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    cm1: np.ndarray,
    cm2: np.ndarray,
    gamma_pc: float,
    gamma_real: bool,
    tol_ov: float,
    grid_size: int = 64,
    num_rotations: int = 70,
    top_k_peaks: int = 10,
    gamma_tolerance: float = 0.10,
    min_peak_distance: int = 3,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Attempt FFT-based rigid-body docking for two clusters.

    For each rotation of cluster 2:
    1. Voxelize both clusters onto 3D grids
    2. Compute FFT cross-correlation
    3. Extract top-K translation peaks
    4. Validate each candidate with exact overlap + gamma_pc checks

    Parameters
    ----------
    coords1, radii1 : cluster 1 particles
    coords2, radii2 : cluster 2 particles
    cm1, cm2 : centres-of-mass of clusters 1 and 2
    gamma_pc : target inter-cluster distance for sticking
    gamma_real : whether gamma_pc is physically valid
    tol_ov : maximum allowable overlap fraction
    grid_size : side length of cubic voxel grid
    num_rotations : number of SO(3) rotation samples
    top_k_peaks : number of translation candidates to evaluate per rotation
    gamma_tolerance : relative tolerance for gamma_pc distance check
    min_peak_distance : minimum grid-cell distance between peaks

    Returns
    -------
    (combined_coords, combined_radii) if sticking succeeds, None otherwise.
    """
    if not gamma_real or gamma_pc <= 0:
        logger.debug("FFT dock: gamma_real=False or gamma_pc<=0, skipping.")
        return None

    all_coords = np.concatenate([coords1, coords2], axis=0)
    all_radii = np.concatenate([radii1, radii2])
    r_max = float(np.max(all_radii)) * 2.0
    span = float(np.max(np.ptp(all_coords, axis=0))) + r_max * 2
    voxel_size = span / grid_size if grid_size > 0 else 1.0
    voxel_size = max(voxel_size, 1e-6)

    grid1, origin1 = voxelize_cluster(
        coords1, radii1, cm1, grid_size, voxel_size, mode=1
    )

    rotations = sample_so3_rotations(num_rotations)

    best_overlap = float("inf")
    best_result = None

    for rot_idx, R in enumerate(rotations):
        coords2_rot = (R @ coords2.T).T
        cm2_rot = R @ cm2

        grid2, origin2 = voxelize_cluster(
            coords2_rot, radii2, cm2_rot, grid_size, voxel_size, mode=1
        )

        corr = compute_fft_correlation(grid1, grid2)

        peaks = extract_top_k_peaks(corr, k=top_k_peaks, min_distance=min_peak_distance)

        for score, dx, dy, dz in peaks:
            shift_real = np.array([dx * voxel_size, dy * voxel_size, dz * voxel_size])

            cm2_offset = cm2_rot + shift_real - (origin2 - origin1)

            actual_dist = float(np.linalg.norm(cm2_offset - cm1))
            dist_ratio = actual_dist / gamma_pc if gamma_pc > 1e-12 else float("inf")

            if dist_ratio < 0.5 or dist_ratio > 1.5:
                continue

            offset = cm2_offset - cm2_rot
            coords2_placed = coords2_rot + offset[np.newaxis, :]

            valid, dist, cov_max = validate_placement(
                coords1,
                radii1,
                coords2_placed,
                radii2,
                cm1,
                cm2_offset,
                gamma_pc,
                gamma_real,
                tol_ov,
                gamma_tolerance=gamma_tolerance,
            )

            if valid:
                logger.info(
                    f"FFT dock: SUCCESS at rot={rot_idx}, peak=({dx},{dy},{dz}), "
                    f"dist={dist:.3f}, cov_max={cov_max:.6f}"
                )
                combined_coords = np.vstack([coords1, coords2_placed])
                combined_radii = np.concatenate([radii1, radii2])
                return combined_coords, combined_radii

            if cov_max < best_overlap:
                best_overlap = cov_max

    logger.info(
        f"FFT dock: FAILED after {len(rotations)} rotations. "
        f"Best overlap={best_overlap:.6f}"
    )
    return None
