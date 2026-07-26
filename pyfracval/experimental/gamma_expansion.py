"""Archived CCA gamma-expansion fallback.

Incrementally widens the contact distance (``gamma_pc``) and retries rigid
sticking when the first attempt fails. Benchmarked against no expansion at
all in the hard regime (see ``docs/source/experiments.md``): lands within
noise of the vanilla baseline - the underlying problem is geometric
frustration, and relaxing the contact-distance constraint by a few percent
doesn't open up an overlap-free orientation that wasn't already unreachable.

Kept reachable via ``cca_gamma_expansion_enabled`` for anyone who wants to
try a much larger expansion budget later.

Tightly coupled to :class:`pyfracval.cca.aggregator.CCAggregator` state
(telemetry counters, the ``_gamma_pc_override``/``_gamma_real_override``
side-channel ``_perform_cca_sticking`` reads, and ``_perform_cca_sticking``
itself) - takes the aggregator instance directly rather than trying to
decouple what is inherently an extension of its sticking-attempt loop.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

from .. import fractal

logger = logging.getLogger(__name__)


def run_gamma_expansion(
    aggregator: Any,
    cluster_idx1: int,
    cluster_idx2: int,
    cluster_props_cache: dict | None,
    n1: int,
    n2: int,
    m1: float,
    rg1: float,
    cm1: np.ndarray,
    r_max1: float,
    radii1_in: np.ndarray,
    m2: float,
    rg2: float,
    cm2: np.ndarray,
    r_max2: float,
    radii2_in: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Retry sticking with an incrementally widened gamma_pc until success."""
    algorithm_config = aggregator.algorithm_config
    gamma_expansion_step = float(algorithm_config.cca_gamma_expansion_step)
    gamma_expansion_max_factor = float(algorithm_config.cca_gamma_expansion_max_factor)
    gamma_expansion_mass_exponent = float(
        algorithm_config.cca_gamma_expansion_mass_exponent
    )
    gamma_expansion_max_attempts = int(
        algorithm_config.cca_gamma_expansion_max_attempts
    )
    n_total = n1 + n2
    aggregator._gamma_expansion_hits += 1

    props1 = (m1, rg1, cm1, r_max1, radii1_in)
    props2 = (m2, rg2, cm2, r_max2, radii2_in)
    _, gamma_pc_original = aggregator._calculate_cca_gamma(props1, props2)

    for attempt in range(1, gamma_expansion_max_attempts + 1):
        expansion_delta = (
            gamma_expansion_step * (n_total**gamma_expansion_mass_exponent) * attempt
        )
        gamma_pc_expanded = gamma_pc_original * (1.0 + expansion_delta)

        if gamma_pc_expanded > gamma_pc_original * gamma_expansion_max_factor:
            logger.info(
                f"CCA gamma expansion hit max factor "
                f"{gamma_expansion_max_factor:.3f} for clusters "
                f"{cluster_idx1}, {cluster_idx2}. Giving up."
            )
            return None

        # Recompute gamma from fractal scaling law for physical consistency
        rg3_exp = fractal.calculate_rg(
            np.concatenate((radii1_in, radii2_in)),
            n_total,
            aggregator.df,
            aggregator.kf,
        )
        m1_h, m2_h = float(n1), float(n2)
        m3_h = float(n_total)
        term1 = (m3_h**2) * (rg3_exp**2)
        term2 = m3_h * (m1_h * rg1**2 + m2_h * rg2**2)
        denom = m1_h * m2_h
        radicand = term1 - term2

        gamma_real_exp = (denom > 0) and (radicand >= 0)
        if gamma_real_exp:
            gamma_pc_rec = float(np.sqrt(radicand / denom))
            gamma_pc = min(gamma_pc_expanded, gamma_pc_rec * gamma_expansion_max_factor)
            gamma_pc = max(gamma_pc, gamma_pc_original)
        else:
            gamma_pc = gamma_pc_expanded
        gamma_real = True

        aggregator._gamma_expansion_total_steps += 1
        logger.info(
            f"CCA gamma expansion ({cluster_idx1},{cluster_idx2}): "
            f"attempt {attempt}/{gamma_expansion_max_attempts}, "
            f"gamma {gamma_pc_original:.4f} -> {gamma_pc:.4f} "
            f"(factor={gamma_pc / gamma_pc_original:.4f})"
        )

        # Override gamma via temp attribute
        aggregator._gamma_pc_override = gamma_pc
        aggregator._gamma_real_override = gamma_real
        try:
            result = aggregator._perform_cca_sticking(
                cluster_idx1, cluster_idx2, cluster_props_cache
            )
        finally:
            aggregator._gamma_pc_override = None
            aggregator._gamma_real_override = None

        if result is not None:
            aggregator._gamma_expansion_successes += 1
            return result

    logger.warning(
        f"CCA sticking failed for clusters {cluster_idx1}, {cluster_idx2} "
        f"after {gamma_expansion_max_attempts} gamma expansions."
    )
    return None
