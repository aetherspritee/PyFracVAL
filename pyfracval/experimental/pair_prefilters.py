"""Archived CCA pair-feasibility pre-filters.

The bounding-volume and SSA (surface-accessibility) prefilters were meant to
skip doomed candidate pairs before spending time on rigid-body sticking
attempts. Benchmarked head-to-head against no filtering at all (hard regime,
see ``docs/source/experiments.md``): both land within noise of the vanilla
baseline - neither improves success rate, and the extra bookkeeping doesn't
save meaningful wall time either. Kept reachable via
``cca_pair_feasibility_filter`` ("bounding_volume" / "ssa") for anyone who
wants to try a different angle later.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import OrchestratorAlgorithmConfig

logger = logging.getLogger(__name__)


def bounding_volume_precheck(
    gamma_pc: float,
    r_max1: float,
    r_max2: float,
    gamma_real: bool,
    algorithm_config: OrchestratorAlgorithmConfig | None = None,
) -> bool:
    """Check if two clusters can physically stick at distance gamma_pc."""
    if not gamma_real or gamma_pc <= 0.0:
        return False
    algorithm_config = algorithm_config or OrchestratorAlgorithmConfig()
    factor = float(algorithm_config.cca_bv_deep_penetration_factor)
    rmax_diff = abs(r_max1 - r_max2)
    if gamma_pc < rmax_diff * factor:
        logger.debug(
            f"BV pre-check reject: gamma={gamma_pc:.3f} < "
            f"|r_max1-r_max2|*{factor}={rmax_diff * factor:.3f}"
        )
        return False
    if gamma_pc > r_max1 + r_max2:
        logger.debug(
            f"BV pre-check reject: gamma={gamma_pc:.3f} > "
            f"r_max1+r_max2={r_max1 + r_max2:.3f}"
        )
        return False
    return True


def surface_accessible_mask(
    coords: np.ndarray,
    radii: np.ndarray,
    cm: np.ndarray,
    r_max: float,
    min_exposure: float | None = None,
    algorithm_config: OrchestratorAlgorithmConfig | None = None,
) -> np.ndarray:
    """Compute surface accessibility mask for each monomer."""
    n = coords.shape[0]
    if n <= 1:
        return np.ones(n, dtype=bool)
    if min_exposure is None:
        algorithm_config = algorithm_config or OrchestratorAlgorithmConfig()
        min_exposure = float(algorithm_config.cca_ssa_min_exposure)
    dist_to_cm = np.linalg.norm(coords - cm[np.newaxis, :], axis=1)
    radial_fraction = dist_to_cm / max(r_max, 1.0e-12)
    mean_r = float(np.mean(radii))
    contact_dist_sq = (2.5 * mean_r) ** 2
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    d_sq = np.sum(diffs * diffs, axis=2)
    neighbor_count = np.sum((d_sq < contact_dist_sq) & (d_sq > 0), axis=1)
    max_neighbors = max(n - 1, 1)
    isolation_fraction = 1.0 - neighbor_count / max_neighbors
    exposure = 0.5 * radial_fraction + 0.5 * isolation_fraction
    return exposure >= min_exposure
