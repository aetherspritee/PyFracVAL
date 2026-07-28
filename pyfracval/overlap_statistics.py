"""Opt-in, off-hot-path overlap-failure census for CCA sticking attempts.

`pyfracval/overlap.py`'s CCA overlap-check functions return a single
scalar (max overlap fraction) and early-exit the instant any pair exceeds
tolerance - load-bearing for performance (see
docs/source/gpu_acceleration.md: numba beats JAX by 1-4 orders of
magnitude specifically because of this early-exit/branch-skipping
behavior). This module is deliberately *not* a modification of that hot
path. Instead, on a *failed* sticking attempt, it runs one full
(non-early-exit) pairwise scan between the two clusters to answer a
question the scalar check can't: how many particles overlap, and by how
much - severity data the binary success/fail signal throws away.

Modeled directly on `pyfracval/densify.py`'s `_self_overlap_pairs_kernel`
(same full-scan, no-early-exit, capped-output design), adapted from a
single-set self-overlap scan to a two-cluster cross-overlap scan.
"""

from typing import Tuple

import numpy as np

from .schemas import OverlapCensus

try:
    from numba import jit

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


_SEVERITY_BUCKETS = [0.05, 0.15, 0.3]
_SEVERITY_LABELS = ["0-0.05", "0.05-0.15", "0.15-0.3", "0.3+"]


@jit(nopython=True, fastmath=True, cache=True)
def _cross_overlap_pairs_kernel(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full (non-early-exit) scan of all n1*n2 pairs between two point
    sets. Returns (idx1, idx2, overlap_fraction) for every overlapping
    pair, up to max_pairs. No i<j exclusion needed - unlike
    densify.py's single-set self-overlap kernel, idx1 and idx2 index two
    distinct arrays, not the same one.

    overlap_fraction uses the same definition as densify.py's kernel:
    (r_sum - dist) / min(r_i, r_j), 0 at exact contact, growing as the
    spheres interpenetrate further relative to the smaller radius.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    pair_i = np.empty(max_pairs, dtype=np.int64)
    pair_j = np.empty(max_pairs, dtype=np.int64)
    pair_ov = np.empty(max_pairs, dtype=np.float64)
    count = 0

    for i in range(n1):
        xi = coords1[i, 0]
        yi = coords1[i, 1]
        zi = coords1[i, 2]
        ri = radii1[i]
        for j in range(n2):
            dx = xi - coords2[j, 0]
            dy = yi - coords2[j, 1]
            dz = zi - coords2[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            r_sum = ri + radii2[j]
            if dist_sq < r_sum * r_sum:
                dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-12
                ov = (r_sum - dist) / min(ri, radii2[j])
                if count < max_pairs:
                    pair_i[count] = i
                    pair_j[count] = j
                    pair_ov[count] = ov
                    count += 1

    return pair_i[:count], pair_j[:count], pair_ov[:count]


def _severity_histogram(overlaps: np.ndarray) -> dict[str, int]:
    hist = {label: 0 for label in _SEVERITY_LABELS}
    if overlaps.size == 0:
        return hist
    bucket_idx = np.searchsorted(_SEVERITY_BUCKETS, overlaps, side="right")
    for idx in bucket_idx:
        hist[_SEVERITY_LABELS[int(idx)]] += 1
    return hist


def compute_overlap_census(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    max_pairs: int = 4096,
) -> OverlapCensus:
    """Run the full cross-overlap scan and package the result as an
    OverlapCensus. Cost is O(n1*n2) with no early exit - intended to run
    once, on a failed attempt, never on the hot path."""
    pair_i, pair_j, pair_ov = _cross_overlap_pairs_kernel(
        np.ascontiguousarray(coords1, dtype=np.float64),
        np.ascontiguousarray(radii1, dtype=np.float64),
        np.ascontiguousarray(coords2, dtype=np.float64),
        np.ascontiguousarray(radii2, dtype=np.float64),
        max_pairs,
    )

    n_pairs = int(pair_i.shape[0])
    offending1 = sorted({int(i) for i in pair_i})
    offending2 = sorted({int(j) for j in pair_j})

    return OverlapCensus(
        n_pairs_overlapping=n_pairs,
        n_particles_cluster1_offending=len(offending1),
        n_particles_cluster2_offending=len(offending2),
        offending_indices_cluster1=offending1,
        offending_indices_cluster2=offending2,
        max_overlap_fraction=float(pair_ov.max()) if n_pairs else 0.0,
        mean_overlap_fraction=float(pair_ov.mean()) if n_pairs else 0.0,
        severity_histogram=_severity_histogram(pair_ov),
        cluster1_size=int(coords1.shape[0]),
        cluster2_size=int(coords2.shape[0]),
    )
