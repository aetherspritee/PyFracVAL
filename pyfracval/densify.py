"""Post-aggregation densification for high-Df fractal aggregates.

Generates aggregates at a feasible Df (e.g. 2.0) then compresses them
toward the target Df (e.g. 2.25) using radial compression followed by
iterative overlap resolution.  Optionally uses Voronoi-guided migration
for better structural preservation.

Two methods are provided:
  - ``radial``: uniform radial compression toward CM, then overlap push-apart
  - ``voronoi``: Voronoi-guided migration of under-dense particles inward

Both methods are opt-in via ``config.DENSIFY_ENABLED``.
"""

import logging
from typing import Tuple

import numpy as np

from . import config, utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba JIT kernels
# ---------------------------------------------------------------------------

try:
    from numba import jit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    prange = range


@jit(nopython=True, fastmath=True, cache=True)
def _self_overlap_pairs_kernel(
    coords: np.ndarray,
    radii: np.ndarray,
    n: int,
    max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find overlapping particle pairs in a single aggregate.

    Returns arrays of (i, j, overlap_amount) for each overlapping pair,
    up to max_pairs pairs.
    """
    pair_i = np.empty(max_pairs, dtype=np.int64)
    pair_j = np.empty(max_pairs, dtype=np.int64)
    pair_ov = np.empty(max_pairs, dtype=np.float64)
    count = 0

    for i in range(n):
        xi = coords[i, 0]
        yi = coords[i, 1]
        zi = coords[i, 2]
        ri = radii[i]
        for j in range(i + 1, n):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dz = zi - coords[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            r_sum = ri + radii[j]
            if dist_sq < r_sum * r_sum:
                dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-12
                overlap = (r_sum - dist) / min(ri, radii[j])
                if count < max_pairs:
                    pair_i[count] = i
                    pair_j[count] = j
                    pair_ov[count] = overlap
                    count += 1
    return pair_i[:count], pair_j[:count], pair_ov[:count]


@jit(nopython=True, fastmath=True, cache=True)
def _radial_compress_kernel(
    coords: np.ndarray,
    cm: np.ndarray,
    alpha: float,
    n: int,
) -> np.ndarray:
    """Radially compress coordinates toward CM by factor alpha."""
    result = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        for d in range(3):
            result[i, d] = cm[d] + alpha * (coords[i, d] - cm[d])
    return result


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def _push_apart_kernel(
    coords: np.ndarray,
    radii: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_ov: np.ndarray,
    n_pairs: int,
    push_fraction: float,
    n: int,
) -> np.ndarray:
    """Push overlapping pairs apart along their connecting vector."""
    displacements = np.zeros((n, 3), dtype=np.float64)
    for k in range(n_pairs):
        i = pair_i[k]
        j = pair_j[k]
        ov = pair_ov[k]
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dz = coords[i, 2] - coords[j, 2]
        dist_sq = dx * dx + dy * dy + dz * dz
        dist = np.sqrt(dist_sq) if dist_sq > 1e-20 else 1e-10
        push = ov * push_fraction * 0.5
        ux = dx / dist
        uy = dy / dist
        uz = dz / dist
        displacements[i, 0] += push * ux
        displacements[i, 1] += push * uy
        displacements[i, 2] += push * uz
        displacements[j, 0] -= push * ux
        displacements[j, 1] -= push * uy
        displacements[j, 2] -= push * uz
    return displacements


# ---------------------------------------------------------------------------
# Python-level functions
# ---------------------------------------------------------------------------


def _compute_measured_rg(
    coords: np.ndarray, radii: np.ndarray, df: float, kf: float
) -> float:
    """Compute measured Rg from coordinates using the fractal scaling law."""
    n = coords.shape[0]
    return utils.calculate_rg(radii, n, df, kf)


def _compute_empirical_rg(coords: np.ndarray, radii: np.ndarray) -> float:
    """Compute Rg directly from coordinates (mass-weighted)."""
    cm = np.average(coords, axis=0, weights=radii**3)
    dist_sq = np.sum((coords - cm[np.newaxis, :]) ** 2, axis=1)
    return float(np.sqrt(np.average(dist_sq, weights=radii**3)))


def _find_overlaps(
    coords: np.ndarray,
    radii: np.ndarray,
    max_pairs: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all overlapping particle pairs and their overlap amounts."""
    n = coords.shape[0]
    return _self_overlap_pairs_kernel(
        coords.astype(np.float64),
        radii.astype(np.float64),
        n,
        max_pairs,
    )


def radial_compress(
    coords: np.ndarray,
    radii: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Radially compress coordinates toward CM by factor alpha.

    Parameters
    ----------
    coords : (N, 3) particle positions
    radii : (N,) particle radii
    alpha : compression factor (< 1.0 moves particles inward)

    Returns
    -------
    Compressed coordinates, shape (N, 3)
    """
    cm = np.average(coords, axis=0, weights=radii**3)
    return _radial_compress_kernel(
        coords.astype(np.float64), cm.astype(np.float64), alpha, coords.shape[0]
    )


def resolve_overlaps(
    coords: np.ndarray,
    radii: np.ndarray,
    tol_ov: float = 1e-4,
    max_iters: int = 50,
    push_fraction: float = 0.5,
    patience: int = 10,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, bool, int]:
    """Iteratively resolve overlapping particle pairs.

    Parameters
    ----------
    coords : (N, 3) particle positions
    radii : (N,) particle radii
    tol_ov : maximum allowed fractional overlap
    max_iters : maximum resolution iterations
    push_fraction : fraction of overlap to push apart each step
    patience : stop after this many non-improving iterations
    rng : random number generator for jitter

    Returns
    -------
    (resolved_coords, success, n_iters)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = coords.shape[0]
    current = coords.copy()
    best_max_ov = float("inf")
    stagnant = 0

    for iteration in range(max_iters):
        pair_i, pair_j, pair_ov = _find_overlaps(current, radii, max_pairs=n * 10)

        if len(pair_i) == 0:
            logger.info(f"Overlap resolution converged in {iteration} iterations.")
            return current, True, iteration

        max_ov = float(np.max(pair_ov)) if len(pair_ov) > 0 else 0.0

        if max_ov <= tol_ov:
            logger.info(
                f"Overlap resolution converged in {iteration} iterations "
                f"(max overlap={max_ov:.2e} <= tol={tol_ov:.2e})."
            )
            return current, True, iteration

        if max_ov < best_max_ov:
            best_max_ov = max_ov
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                logger.warning(
                    f"Overlap resolution stalled after {iteration} iterations "
                    f"(max overlap={max_ov:.4f}, best={best_max_ov:.4f})."
                )
                return current, False, iteration

        displacements = _push_apart_kernel(
            current.astype(np.float64),
            radii.astype(np.float64),
            pair_i,
            pair_j,
            pair_ov,
            len(pair_i),
            push_fraction,
            n,
        )
        current = (
            current
            + displacements
            + rng.normal(0, 0.1 * np.mean(radii), size=current.shape) * 0.01
        )

    logger.warning(
        f"Overlap resolution exhausted {max_iters} iterations "
        f"(max overlap={best_max_ov:.4f})."
    )
    return current, False, max_iters


def voronoi_local_density(
    coords: np.ndarray,
) -> np.ndarray:
    """Compute local number density from Voronoi cell volumes.

    Uses scipy.spatial.Voronoi for tessellation.  Volume is approximated
    as the volume of the circumscribed sphere of each Voronoi region
    (correct for convex polyhedra).

    Parameters
    ----------
    coords : (N, 3) particle positions

    Returns
    -------
    local_density : (N,) array of local number densities (1/volume)
    """
    from scipy.spatial import Voronoi

    n = coords.shape[0]
    if n < 5:
        return np.ones(n) / np.mean(np.var(coords, axis=0))

    vor = Voronoi(coords)

    volumes = np.full(n, np.inf)
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        verts = vor.vertices[region]
        n_v = len(verts)
        hull_vol = 0.0
        origin = verts[0]
        for j in range(1, n_v - 1):
            v1 = verts[j] - origin
            v2 = verts[j + 1] - origin
            cross = np.cross(v1, v2)
            hull_vol += abs(np.dot(origin, cross)) / 6.0
        volumes[i] = hull_vol if hull_vol > 0 else np.inf

    finite_mask = np.isfinite(volumes) & (volumes > 0)
    if not np.any(finite_mask):
        return np.ones(n) / np.mean(np.var(coords, axis=0))

    volumes[~finite_mask] = np.median(volumes[finite_mask])
    return 1.0 / volumes


def voronoi_migrate_step(
    coords: np.ndarray,
    radii: np.ndarray,
    rg_target: float,
    step_fraction: float = 0.02,
) -> Tuple[np.ndarray, float]:
    """One step of Voronoi-guided migration.

    Moves particles in under-dense regions (high Voronoi volume / low density)
    toward the center of mass, proportional to their local under-density.

    Parameters
    ----------
    coords : (N, 3) particle positions
    radii : (N,) particle radii
    rg_target : target radius of gyration
    step_fraction : fraction of CM-to-particle distance to move each step

    Returns
    -------
    (updated_coords, current_rg)
    """
    cm = np.average(coords, axis=0, weights=radii**3)
    local_density = voronoi_local_density(coords)
    median_density = np.median(local_density)

    under_dense = local_density < median_density
    under_dense_fraction = float(np.sum(under_dense)) / len(under_dense)

    step = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        if under_dense[i]:
            direction = cm - coords[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                weight = max(0.0, 1.0 - local_density[i] / median_density)
                step[i] = direction / norm * step_fraction * norm * weight

    new_coords = coords + step
    new_rg = _compute_empirical_rg(new_coords, radii)
    return new_coords, new_rg


def densify_aggregate(
    coords: np.ndarray,
    radii: np.ndarray,
    target_df: float,
    target_kf: float,
    tol_ov: float = 1e-4,
    max_push_iters: int = 50,
    max_densify_iters: int = 20,
    push_fraction: float = 0.5,
    push_patience: int = 10,
    rg_rtol: float = 0.02,
    method: str = "radial",
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Densify an aggregate from its current Df toward target Df.

    Parameters
    ----------
    coords : (N, 3) particle positions (from CCA at source Df)
    radii : (N,) particle radii
    target_df : target fractal dimension
    target_kf : target fractal prefactor
    tol_ov : maximum allowed fractional overlap during resolution
    max_push_iters : max overlap resolution iterations per densification step
    max_densify_iters : max densification iterations
    push_fraction : fraction of overlap to push apart each step
    push_patience : stop push-apart after this many stagnant iterations
    rg_rtol : relative tolerance on Rg target
    method : "radial" for radial compression, "voronoi" for Voronoi-guided
    rng : random number generator

    Returns
    -------
    (densified_coords, densified_radii, success)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = coords.shape[0]
    rg_target = utils.calculate_rg(radii, n, target_df, target_kf)
    rg_current = _compute_empirical_rg(coords, radii)

    logger.info(
        f"Densify: target_rg={rg_target:.2f}, current_rg={rg_current:.2f}, "
        f"method={method}, rg_rtol={rg_rtol}"
    )

    if rg_current <= rg_target * (1.0 + rg_rtol):
        logger.info("Densify: already at or below target Rg, no densification needed.")
        return coords, radii, True

    current_coords = coords.copy()
    best_coords = coords.copy()
    best_rg_error = abs(rg_current - rg_target) / rg_target

    if method == "radial":
        alpha = rg_target / rg_current
        logger.info(f"Densify: radial compression factor alpha={alpha:.4f}")

        compressed = radial_compress(current_coords, radii, alpha)
        compressed_rg = _compute_empirical_rg(compressed, radii)
        logger.info(
            f"Densify: after compression, rg={compressed_rg:.2f} (target={rg_target:.2f})"
        )

        resolved, success, n_iters = resolve_overlaps(
            compressed,
            radii,
            tol_ov=tol_ov,
            max_iters=max_push_iters,
            push_fraction=push_fraction,
            patience=push_patience,
            rng=rng,
        )

        final_rg = _compute_empirical_rg(resolved, radii)
        rg_error = abs(final_rg - rg_target) / rg_target
        logger.info(
            f"Densify: after overlap resolution, rg={final_rg:.2f}, "
            f"error={rg_error:.4f}, overlaps_resolved={success}"
        )

        if rg_error < best_rg_error:
            best_coords = resolved
            best_rg_error = rg_error

        if rg_error <= rg_rtol:
            return resolved, radii, True

    elif method == "voronoi":
        for diter in range(max_densify_iters):
            current_rg = _compute_empirical_rg(current_coords, radii)
            rg_error = abs(current_rg - rg_target) / rg_target

            if rg_error <= rg_rtol:
                logger.info(
                    f"Densify: Voronoi converged after {diter} iterations, "
                    f"rg={current_rg:.2f}"
                )
                return current_coords, radii, True

            step_frac = min(0.05, (current_rg - rg_target) / rg_current)
            new_coords, new_rg = voronoi_migrate_step(
                current_coords, radii, rg_target, step_fraction=step_frac
            )

            resolved, success, _ = resolve_overlaps(
                new_coords,
                radii,
                tol_ov=tol_ov,
                max_iters=max_push_iters,
                push_fraction=push_fraction,
                patience=push_patience,
                rng=rng,
            )

            new_rg = _compute_empirical_rg(resolved, radii)
            rg_error_new = abs(new_rg - rg_target) / rg_target

            if rg_error_new < best_rg_error:
                best_coords = resolved
                best_rg_error = rg_error_new

            if success or rg_error_new < rg_error:
                current_coords = resolved
            else:
                current_coords = resolved
                step_frac *= 0.5

            logger.info(
                f"Densify: Voronoi iter {diter}, rg={new_rg:.2f}, "
                f"error={rg_error_new:.4f}"
            )

    else:
        logger.error(f"Densify: unknown method '{method}'. Use 'radial' or 'voronoi'.")
        return best_coords, radii, False

    max_cov = utils.calculate_max_overlap_cca_auto(
        best_coords, radii, best_coords, radii, tolerance=tol_ov
    )
    overlap_ok = max_cov <= tol_ov
    final_rg = _compute_empirical_rg(best_coords, radii)
    rg_ok = abs(final_rg - rg_target) / rg_target <= rg_rtol * 2

    logger.info(
        f"Densify: final rg={final_rg:.2f} (target={rg_target:.2f}), "
        f"max_overlap={max_cov:.2e}, overlap_ok={overlap_ok}, rg_ok={rg_ok}"
    )

    return best_coords, radii, overlap_ok and rg_ok
