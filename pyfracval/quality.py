"""Final-aggregate quality measurement.

Answers two questions about a *finished* aggregate that nothing else in
the pipeline asks:

1. **Is the geometry actually valid?** ``success`` in the catalog means
   "PCA+CCA reached the requested particle count", which says nothing
   about whether the saved coordinates are overlap-free. Clusters marked
   successful have been found carrying severe residual overlap (see
   docs/source/catalog_overlap_leak.md). Measuring the finished geometry
   once, at the end, closes that whole class of leak structurally rather
   than by chasing individual entry points.
2. **Did it land on the scaling law?** The per-merge Gamma machinery
   *aims* at the prescribed Df/kf, but the pairing relaxation factor and
   the adaptive overlap tolerance both accept merges slightly off-target,
   and nothing measures the result.

Cheap enough to run unconditionally: one O(N^2) pass over a few hundred
to a few thousand particles, once per aggregate, against a generation
that took seconds to minutes.
"""

import logging

import numpy as np

from . import fractal

logger = logging.getLogger(__name__)


def max_self_overlap(
    coords: np.ndarray, radii: np.ndarray, min_overlap: float = 1e-12
) -> tuple[float, int]:
    """Largest and total count of residual overlaps within one aggregate.

    The overlap fraction uses the same normalization as the sticking
    tolerance ``tol_ov`` - ``((r_i + r_j) - d_ij) / (r_i + r_j)`` - so the
    result is directly comparable against it. (Note
    ``densify._self_overlap_pairs_kernel`` normalizes by ``min(r_i, r_j)``
    instead, for its own push-apart geometry; the two numbers are not
    interchangeable.)

    Parameters
    ----------
    min_overlap : float
        Ignore overlaps at or below this. Particles are placed in point
        contact, so a correctly-built aggregate has many pairs sitting at
        ``|overlap| ~ 1e-15`` purely from floating-point round-off in the
        rigid transforms. Counting those would report every healthy
        aggregate as having dozens of "overlapping pairs".

    Returns
    -------
    tuple[float, int]
        ``(max_overlap_fraction, n_overlapping_pairs)``. The fraction is
        0.0 when nothing overlaps beyond ``min_overlap``.
    """
    n = coords.shape[0]
    if n < 2:
        return 0.0, 0

    # pdist-style upper-triangle scan, vectorized in one shot. N is at
    # most a few thousand here, so the N^2 intermediate is fine and far
    # faster than a Python loop.
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]

    iu = np.triu_indices(n, k=1)
    dist_u = dist[iu]
    r_sum_u = r_sum[iu]

    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = (r_sum_u - dist_u) / r_sum_u
    overlapping = overlap > min_overlap
    n_pairs = int(np.count_nonzero(overlapping))
    if n_pairs == 0:
        return 0.0, 0
    return float(np.max(overlap[overlapping])), n_pairs


def compute_aggregate_quality(
    coords: np.ndarray,
    radii: np.ndarray,
    df: float,
    kf: float,
    tol_ov: float,
    n_particles_dropped: int = 0,
    densities: np.ndarray | None = None,
) -> dict:
    """Measure a finished aggregate against what was asked for.

    Parameters
    ----------
    coords, radii : np.ndarray
        The final aggregate geometry.
    df, kf : float
        The prescribed fractal parameters, for the scaling-law target.
    tol_ov : float
        The overlap tolerance the run was generated under; used to decide
        ``overlap_ok``.
    n_particles_dropped : int
        Particles removed by drop-rescue, passed through for the record.

    Returns
    -------
    dict
        ``max_residual_overlap``, ``n_overlapping_pairs``, ``overlap_ok``,
        ``measured_rg`` (paper Eq. 4), ``scaling_law_rg``,
        ``rg_error_pct``, ``n_particles``, ``n_particles_dropped``.
    """
    n = int(coords.shape[0])
    max_overlap, n_pairs = max_self_overlap(coords, radii)
    measured_rg = fractal.compute_empirical_rg_polydisperse(coords, radii, densities)
    scaling_law_rg = fractal.calculate_rg(radii, n, df, kf)
    rg_error_pct = (
        (measured_rg - scaling_law_rg) / scaling_law_rg * 100.0
        if scaling_law_rg > 0.0
        else float("nan")
    )

    # A tolerance-sized overlap is what the sticking loop was told to
    # allow, so only flag what exceeds it. The factor of 10 keeps
    # floating-point accumulation across many rigid transforms from
    # tripping the flag on geometry that is fine.
    overlap_ok = max_overlap <= max(tol_ov * 10.0, 1e-9)

    if not overlap_ok:
        logger.warning(
            f"Aggregate has residual overlap above tolerance: "
            f"max={max_overlap:.3e} over {n_pairs} pairs (tol_ov={tol_ov:.1e})"
        )

    return {
        "n_particles": n,
        "n_particles_dropped": int(n_particles_dropped),
        "max_residual_overlap": max_overlap,
        "n_overlapping_pairs": n_pairs,
        "overlap_ok": bool(overlap_ok),
        "measured_rg": measured_rg,
        "scaling_law_rg": scaling_law_rg,
        "rg_error_pct": rg_error_pct,
    }
