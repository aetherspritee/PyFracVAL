"""Detect-and-drop rescue for CCA sticking failures.

docs/source/overlap_failure_census.md's real data is a caution, not a
green light: at N=128 hard regime, every failure happens at CCA round 1
between ~12-particle subclusters, with a median of 9/24 (~37.5%)
particles implicated - not the "a handful out of hundreds" scale the
idea was originally framed around. This module implements the mechanism
regardless (it may behave differently at larger N or later rounds, which
the census data didn't sample), gated by an explicit drop budget so it
only fires when a failure genuinely is localized - most of the failures
this project has actually measured will not qualify, and that is by
design, not a bug.

Design: rather than re-running the full candidate/rotation search on a
reduced cluster pair (expensive, and would need to plumb a raw-coords
entry point through the whole _perform_cca_sticking machinery), this
takes the *exact* geometric placement that was already censused as
failing, removes the specific particles the census identified as
offending, and checks whether *that* placement is now overlap-free. This
directly answers the question the user's own framing asked ("would
dropping the monomers make the cluster fusable") without a second search.
If removing those particles doesn't resolve the overlap (some other pair
is still too close), the rescue simply fails and the caller falls
through to the existing round-abort behavior - no retry-with-a-different-
placement is attempted here.

No backfill: a rescued merge produces fewer than the requested N
particles. The final aggregate's shortfall is reported via
AggregateProperties.n_particles_dropped rather than silently absorbed.
"""

import logging

import numpy as np

from .. import overlap
from ..schemas import OverlapCensus

logger = logging.getLogger(__name__)


def select_drop_candidates(
    census: OverlapCensus,
    max_drop_particles: int,
    max_drop_fraction: float,
) -> tuple[list[int], list[int]] | None:
    """Decide which particles to drop from each side of a failed pair,
    given its overlap census.

    Budget: min(max_drop_particles, ceil(max_drop_fraction * cluster_size))
    per side - an absolute safety cap (never drop more than
    max_drop_particles outright) *and* a relative cap (never drop more
    than max_drop_fraction of a small cluster, so a 20-particle cluster
    doesn't lose a quarter of itself just because the fixed budget allows
    it). Both bounds are configurable
    (OrchestratorAlgorithmConfig.cca_drop_rescue_max_particles /
    cca_drop_rescue_max_fraction) since the right values are an empirical
    question, not something to fix a priori - see
    docs/source/overlap_failure_census.md's severity-histogram data for
    what's actually observed.

    Because the two caps combine with ``min``, the absolute one dominates
    once clusters get large: at N=512's observed cluster-pair size of 100,
    ``min(5, ceil(0.25*100))`` is still 5, so the relative budget never
    actually engages no matter how it is set (docs/source/drop_rescue.md).
    Setting ``max_drop_particles`` to 0 (or negative) therefore means "no
    absolute cap - scale purely with cluster size", which is the only way
    to get a genuinely N-aware budget.

    Returns (drop_idx1, drop_idx2) if within budget on both sides, else
    None (rescue not attempted - caller falls through to the existing
    round-abort behavior).
    """
    relative1 = int(np.ceil(max_drop_fraction * census.cluster1_size))
    relative2 = int(np.ceil(max_drop_fraction * census.cluster2_size))
    if max_drop_particles > 0:
        budget1 = min(max_drop_particles, relative1)
        budget2 = min(max_drop_particles, relative2)
    else:
        budget1, budget2 = relative1, relative2

    if (
        census.n_particles_cluster1_offending > budget1
        or census.n_particles_cluster2_offending > budget2
    ):
        return None

    return (
        list(census.offending_indices_cluster1),
        list(census.offending_indices_cluster2),
    )


def retry_sticking_with_drops(
    coords1_failed: np.ndarray,
    radii1: np.ndarray,
    coords2_failed: np.ndarray,
    radii2: np.ndarray,
    drop_idx1: list[int],
    drop_idx2: list[int],
    tol_ov: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Remove the given particle indices from the *already-placed* failing
    geometry and check whether the result is overlap-free.

    Parameters mirror the last-censused placement
    (coords1_failed/coords2_failed - the same arrays
    pyfracval/overlap_statistics.py::compute_overlap_census was run
    against, not a fresh re-placement).

    Returns (combined_coords, combined_radii) on success, None if the
    reduced pair still overlaps (dropping those particles wasn't enough)
    or would empty a cluster entirely.
    """
    keep_mask1 = np.ones(coords1_failed.shape[0], dtype=bool)
    keep_mask1[drop_idx1] = False
    keep_mask2 = np.ones(coords2_failed.shape[0], dtype=bool)
    keep_mask2[drop_idx2] = False

    reduced_coords1 = coords1_failed[keep_mask1]
    reduced_radii1 = radii1[keep_mask1]
    reduced_coords2 = coords2_failed[keep_mask2]
    reduced_radii2 = radii2[keep_mask2]

    if reduced_coords1.shape[0] == 0 or reduced_coords2.shape[0] == 0:
        logger.warning(
            "Drop-rescue would empty a cluster entirely - aborting rescue attempt."
        )
        return None

    cov_max = overlap.calculate_max_overlap_cca_auto(
        reduced_coords1,
        reduced_radii1,
        reduced_coords2,
        reduced_radii2,
        tolerance=tol_ov,
    )
    if cov_max > tol_ov:
        logger.info(
            f"Drop-rescue: removing {len(drop_idx1)}+{len(drop_idx2)} particles "
            f"was not sufficient (remaining overlap={cov_max:.4e} > tol={tol_ov:.4e})."
        )
        return None

    logger.info(
        f"Drop-rescue succeeded: dropped {len(drop_idx1)} particle(s) from "
        f"cluster 1 ({coords1_failed.shape[0]} -> {reduced_coords1.shape[0]}) and "
        f"{len(drop_idx2)} from cluster 2 "
        f"({coords2_failed.shape[0]} -> {reduced_coords2.shape[0]})."
    )
    combined_coords = np.vstack((reduced_coords1, reduced_coords2))
    combined_radii = np.concatenate((reduced_radii1, reduced_radii2))
    return combined_coords, combined_radii
