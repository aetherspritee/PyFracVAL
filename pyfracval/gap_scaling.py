"""Convert a target inter-particle gap into a position-only scale factor.

Ported from YASF-new's ``Config.cluster_gap_factor``/``cluster_gap_mode``
(``yasfpy/config.py``) -- the target gap is expressed as a multiple of the
mean particle radius (``target_gap = gap_factor * r_mean``), not an
absolute length, so it's meaningful across differently-scaled clusters.

This module only *computes* the scale -- it never mutates or returns scaled
coordinates. Applying the scale is a downstream concern (e.g. pyfastmm's
``ParticlesConfig.gap_factor``, which already does exactly
``positions *= gap_factor`` and needs nothing else changed): keeping that
split means there is exactly one place a cluster's geometry is actually
transformed, however many places compute *what* to transform it by.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

_VALID_MODES = {"average", "strict"}


def compute_gap_scale(
    coords: np.ndarray,
    radii: np.ndarray,
    gap_factor: float | None,
    mode: str = "average",
) -> float:
    """Return the position-only scale factor that achieves *gap_factor*.

    Parameters
    ----------
    coords : (N, 3) array
        Particle center positions.
    radii : (N,) array
        Particle radii, same units as *coords*.
    gap_factor : float or None
        Target minimum surface-to-surface gap between neighboring
        particles, as a multiple of the mean particle radius. ``None`` or
        ``0`` means no gap requirement -- returns ``1.0`` (the touching,
        as-generated case; a no-op for a caller that then does
        ``positions *= scale``). Must be ``>= 0``.
    mode : {"average", "strict"}
        How *gap_factor* becomes a scale factor:

        - ``"average"`` (default, matching YASF's own default): a cheap
          closed-form estimate from the mean radius,
          ``s = max(1, 1 + gap_factor / 2)``. Does not guarantee every
          pair individually clears the target gap for irregular
          (non-uniform-density) aggregates.
        - ``"strict"``: exact, via a KD-tree nearest-neighbor query --
          finds the scale that gives the *closest* pair exactly the
          target gap, which (since scaling every position by one global
          factor scales every pairwise distance by that same factor)
          guarantees every other pair, starting further apart, clears it
          too.

    Returns
    -------
    float
        The scale factor ``s`` such that ``coords * s`` achieves the
        requested gap. Always ``>= 1.0``.

    Raises
    ------
    ValueError
        If *gap_factor* is negative, *mode* is not one of the values
        above, or the resulting scale fails to eliminate all overlaps
        (checked directly, not assumed from the formula/computation).
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"gap_mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
        )
    if gap_factor is None or gap_factor == 0:
        return 1.0
    if gap_factor < 0:
        raise ValueError(f"gap_factor must be >= 0, got {gap_factor}")

    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    r_mean = float(np.mean(radii))
    target_gap = gap_factor * r_mean

    if mode == "average":
        scale = max(1.0, 1.0 + gap_factor / 2.0)
    else:
        r_max = float(np.max(radii))
        cutoff = 2 * r_max + target_gap
        tree = KDTree(coords)
        pairs = tree.query_pairs(r=cutoff, output_type="ndarray")

        scale = 1.0
        if pairs.size > 0:
            idx_i, idx_j = pairs[:, 0], pairs[:, 1]
            d_ij = np.linalg.norm(coords[idx_i] - coords[idx_j], axis=1)
            if np.any(d_ij <= 1e-12):
                raise ValueError(
                    "Duplicate particle positions detected -- cannot compute "
                    "a valid gap scale."
                )
            needed = (radii[idx_i] + radii[idx_j] + target_gap) / d_ij
            scale = max(1.0, float(np.max(needed)))

    _validate_no_overlaps(coords * scale, radii, gap_factor, mode)
    return scale


def _validate_no_overlaps(
    scaled_coords: np.ndarray,
    radii: np.ndarray,
    gap_factor: float,
    mode: str,
) -> None:
    """Confirm no pair of scaled particles overlaps -- a hard-fail safety
    net, not a formality: the "average" mode is a heuristic and can
    theoretically under-scale an irregular aggregate.

    Plain all-pairs distances (``pdist``), not a KD-tree -- this is a
    correctness check, not a hot path, and cluster sizes here (up to
    ~1000 spheres, ~500k pairs) are nowhere near where an O(N^2) pairwise
    distance matrix would actually matter; a tree-based cutoff query is
    real complexity bought for a performance concern that doesn't exist
    at this scale.
    """
    n = len(radii)
    if n < 2:
        return
    # pdist's condensed output is pairs (0,1),(0,2),...,(0,n-1),(1,2),...
    # in exactly the order np.triu_indices(n, k=1) produces -- this is
    # pdist/squareform's own documented correspondence, not a coincidence.
    dists = pdist(scaled_coords)
    i_idx, j_idx = np.triu_indices(n, k=1)
    min_required = radii[i_idx] + radii[j_idx]
    overlap_count = int(np.sum(dists < min_required))
    if overlap_count > 0:
        raise ValueError(
            f"Scaling for gap_factor={gap_factor} (mode={mode!r}) failed to "
            f"eliminate all overlaps -- {overlap_count} particle pair(s) "
            "still overlap after scaling. This implies unusually dense or "
            "irregular geometry; try mode='strict' if using 'average'."
        )
