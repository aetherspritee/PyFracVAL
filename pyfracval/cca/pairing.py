"""Pair generation and Gamma_pc calculation for CCA.

Mixed into :class:`pyfracval.cca.aggregator.CCAggregator` - these methods
assume ``self`` carries the CCAggregator instance state (coords, radii,
i_orden, df, kf, etc.), set up in ``CCAggregator.__init__``.
"""

import logging
from typing import Tuple

import numpy as np

from .. import fractal
from ..logs import TRACE_LEVEL_NUM

logger = logging.getLogger(__name__)


class _PairingMixin:
    """Pair generation and Gamma_pc calculation methods."""

    def _calculate_cca_gamma(self, props1: Tuple, props2: Tuple) -> Tuple[bool, float]:
        """Calculates Gamma_pc between two clusters based on their properties."""
        m1, rg1, _, _, radii1 = props1
        m2, rg2, _, _, radii2 = props2
        return fractal.gamma_calculation(
            m1,
            rg1,
            radii1,
            m2,
            rg2,
            radii2,
            self.df,
            self.kf,
        )

    def _identify_monomers(self) -> np.ndarray | None:
        """Creates an array mapping each monomer index (0..N-1) to its cluster index (0..i_t-1)."""
        try:
            id_monomers = np.zeros(self.N, dtype=int) - 1  # Initialize with -1
            for cluster_idx in range(self.i_t):
                start_idx = self.i_orden[cluster_idx, 0]
                end_idx = self.i_orden[cluster_idx, 1] + 1
                if (
                    start_idx < end_idx and start_idx >= 0 and end_idx <= self.N
                ):  # Valid range check
                    id_monomers[start_idx:end_idx] = cluster_idx
            # Check if all monomers were assigned
            if np.any(id_monomers < 0):
                unassigned = np.where(id_monomers < 0)[0]
                logger.warning(
                    f"{len(unassigned)} monomers not assigned to any cluster based on i_orden. Indices: {unassigned[:10]}..."
                )
                # This shouldn't happen if i_orden is correct. Force assign or error?
                # Let's allow it but CCA might fail later if it tries to access them.
            return id_monomers
        except IndexError:
            logger.error("Index out of bounds in _identify_monomers. Check i_orden.")
            return None

    def _generate_pairs(self) -> np.ndarray | None:
        """
        Generates the ID_agglomerated matrix indicating potential pairs.
        Applies a relaxation factor if the strict condition fails.
        Returns the matrix or None on failure.
        """
        # --- RELAXATION FACTOR ---
        # Allow gamma_pc to be slightly larger than sum_rmax if needed.
        # Start with a higher value to test if it allows pairing.
        # If this works, you might fine-tune it later (e.g., 1.10, 1.05).
        CCA_PAIRING_FACTOR = 1.10  # Relaxed pairing (10% over sum_rmax; gamma expansion handles the rest)
        strict_pairing_used = True  # Flag to track if relaxation was needed
        # -------------------------

        id_agglomerated = np.zeros((self.i_t, self.i_t), dtype=int)
        cluster_props = {}  # Cache properties

        # Pre-calculate properties (as before)
        for i in range(self.i_t):
            coords_i, radii_i = self._get_cluster_data(i)
            if coords_i.shape[0] == 0:
                cluster_props[i] = (0.0, 0.0, np.zeros(3), 0.0, np.array([]))
                continue
            m_i, rg_i, cm_i, r_max_i = fractal.calculate_cluster_properties(
                coords_i,
                radii_i,
                self.df,
                self.kf,  # Use target Df/kf
            )
            cluster_props[i] = (m_i, rg_i, cm_i, r_max_i, radii_i)
            logger.debug(
                f"Cluster {i}: N={len(radii_i)}, Rg={rg_i:.3f}, Rmax={r_max_i:.3f}, Mass={m_i:.2e}"
            )

        # Check TRACE logging once (optimization: avoid check in inner loop)
        trace_enabled = logger.isEnabledFor(TRACE_LEVEL_NUM)

        # Pairing loop
        for i in range(self.i_t):
            if np.sum(id_agglomerated[i, :]) > 0 or cluster_props[i][0] == 0.0:
                continue

            m1, rg1, _, r_max1, radii1 = cluster_props[i]
            props1 = (m1, rg1, None, r_max1, radii1)
            partner_found = False

            for j in range(i + 1, self.i_t):
                if np.sum(id_agglomerated[:, j]) > 0 or cluster_props[j][0] == 0.0:
                    continue

                m2, rg2, _, r_max2, radii2 = cluster_props[j]
                props2 = (m2, rg2, None, r_max2, radii2)

                gamma_real, gamma_pc = self._calculate_cca_gamma(props1, props2)
                sum_rmax = r_max1 + r_max2

                # --- Check Strict and Relaxed Conditions ---
                strict_condition = gamma_real and gamma_pc < sum_rmax
                # Apply factor ONLY if gamma is real
                relaxed_condition = (
                    gamma_real and gamma_pc < sum_rmax * CCA_PAIRING_FACTOR
                )

                # Log trace information
                if trace_enabled:  # TRACE level (checked once for performance)
                    logger.log(
                        TRACE_LEVEL_NUM,
                        f"Pair ({i},{j}): G={gamma_pc:.3f}, R1+R2={sum_rmax:.3f}, StrictOK={strict_condition}, RelaxOK={relaxed_condition} (Factor={CCA_PAIRING_FACTOR})",
                    )

                # --- Apply Pairing Logic ---
                pair_marked = False
                if strict_condition:
                    id_agglomerated[i, j] = 1
                    id_agglomerated[j, i] = 1
                    partner_found = True
                    pair_marked = True
                    logger.debug(
                        f"  Pair ({i},{j}): Success! Marked for aggregation (Strict)."
                    )

                elif relaxed_condition:  # Check relaxed only if strict failed
                    id_agglomerated[i, j] = 1
                    id_agglomerated[j, i] = 1
                    partner_found = True
                    pair_marked = True
                    strict_pairing_used = False  # Set flag
                    logger.warning(
                        f"  Pair ({i},{j}): Marked using RELAXED condition "
                        f"(Gamma={gamma_pc:.3f} vs SumRmax={sum_rmax:.3f}). "
                        f"Final Df/kf may deviate slightly from target ({self.df:.2f}/{self.kf:.2f})."
                    )
                # --------------------------

                if pair_marked:
                    break  # Found partner for i

            if not partner_found and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"No suitable partner found for cluster {i} after checking all j > {i}."
                )

        # --- Handle the odd cluster out (Logic remains the same) ---
        if self.i_t % 2 != 0:
            paired_status = np.sum(id_agglomerated, axis=0) + np.sum(
                id_agglomerated, axis=1
            )
            unpaired_indices = np.where(paired_status == 0)[0]
            actual_unpaired = [
                idx for idx in unpaired_indices if cluster_props[idx][0] > 0.0
            ]
            if len(actual_unpaired) == 1:
                loc = actual_unpaired[0]
                id_agglomerated[loc, loc] = 1
                logger.debug(f"Marked cluster {loc} as the odd one out (pass-through).")
            elif len(actual_unpaired) > 1:
                logger.warning(
                    f"Found {len(actual_unpaired)} non-empty unpaired clusters ({actual_unpaired}) "
                    f"for odd i_t={self.i_t} even after checking pairs. Pairing may fail."
                )

        # --- Final check: Ensure all non-empty clusters are accounted for ---
        final_paired_status = np.sum(id_agglomerated, axis=0) + np.sum(
            id_agglomerated, axis=1
        )
        should_be_paired_mask = np.array(
            [cluster_props[idx][0] > 0.0 for idx in range(self.i_t)]
        )
        if np.any(final_paired_status[should_be_paired_mask] == 0):
            failed_indices = np.where(
                (final_paired_status == 0) & should_be_paired_mask
            )[0]
            logger.error(
                f"Could not find pairs for all non-empty clusters even with relaxation factor {CCA_PAIRING_FACTOR}. Failed indices: {failed_indices}"
            )
            logger.error("Consider increasing the target Df or kf.")
            self.not_able_cca = True
            return None

        if not strict_pairing_used:
            logger.warning(
                f"CCA pairing required relaxation (Factor={CCA_PAIRING_FACTOR}). Final aggregate properties may deviate slightly from target Df/kf."
            )

        logger.debug("Pair generation completed.")
        return id_agglomerated, cluster_props
