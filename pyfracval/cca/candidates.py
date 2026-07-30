"""Candidate pair selection, scoring, and telemetry for CCA.

Mixed into :class:`pyfracval.cca.aggregator.CCAggregator` - these methods
assume ``self`` carries the CCAggregator instance state, set up in
``CCAggregator.__init__``.
"""

import logging
import math
from typing import Set, Tuple

import numpy as np
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


class _CandidatesMixin:
    """Candidate pair selection, scoring, and telemetry methods."""

    def _cca_select_candidates(
        self, coords1, radii1, cm1, coords2, radii2, cm2, gamma_pc, gamma_real
    ) -> np.ndarray:
        """Generates the n1 x n2 matrix of potential sticking pairs between clusters."""
        n1 = coords1.shape[0]
        n2 = coords2.shape[0]
        list_matrix = np.zeros((n1, n2), dtype=int)

        if not gamma_real or n1 == 0 or n2 == 0:
            return list_matrix

        # Distances of particles from their respective CMs
        dist1 = np.linalg.norm(coords1 - cm1, axis=1)
        dist2 = np.linalg.norm(coords2 - cm2, axis=1)

        if self.ext_case == 1:
            d1_min = dist1 - radii1
            d1_max = dist1 + radii1
            d2_min = dist2 - radii2
            d2_max = dist2 + radii2

            # Use broadcasting for efficient comparison
            d1max_col = d1_max[:, np.newaxis]
            d2max_row = d2_max[np.newaxis, :]
            d1min_col = d1_min[:, np.newaxis]
            d2min_row = d2_min[np.newaxis, :]

            cond1 = (d1max_col + d2max_row) > gamma_pc
            abs_diff = np.abs(d2max_row - d1max_col)
            cond2a = abs_diff < gamma_pc
            cond2b = ((d2max_row - d1max_col) > gamma_pc) & (
                (d2min_row - d1max_col) < gamma_pc
            )
            cond2c = ((d1max_col - d2max_row) > gamma_pc) & (
                (d1min_col - d2max_row) < gamma_pc
            )

            list_matrix[cond1 & (cond2a | cond2b | cond2c)] = 1

        elif self.ext_case == 0:
            d1_max = dist1 + radii1
            d2_max = dist2 + radii2

            d1max_col = d1_max[:, np.newaxis]
            d2max_row = d2_max[np.newaxis, :]

            cond1 = (d1max_col + d2max_row) > gamma_pc
            cond2 = np.abs(d2max_row - d1max_col) < gamma_pc
            list_matrix[cond1 & cond2] = 1

        return list_matrix

    def _cca_pick_candidate_pair(
        self, list_matrix: np.ndarray, tried_pairs: Set[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        Selects a random available candidate pair (cand1_idx, cand2_idx) from the list matrix.
        Avoids pairs in tried_pairs. Returns (-1, -1) if none available.
        """
        valid_indices = np.argwhere(list_matrix > 0)  # Get indices where value is 1
        available_pairs = []
        for idx_pair in valid_indices:
            pair = tuple(idx_pair)
            if pair not in tried_pairs:
                available_pairs.append(pair)

        if not available_pairs:
            return -1, -1

        # Select a random pair from the available ones
        selected_pair_idx = int(self._rng.integers(len(available_pairs)))
        return available_pairs[selected_pair_idx]

    @staticmethod
    def _leaf_mask_for_cluster(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Return bool mask of leaf-like monomers (degree <= 1) within a cluster."""
        n = coords.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        if n == 1:
            return np.ones(1, dtype=bool)

        # Upper triangle only, via pdist: broadcasting to an (n, n, 3)
        # difference array costs several times the memory and time for
        # the same answer, and this runs per cluster per sticking call
        # whenever leaf information is actually consumed.
        dist = pdist(coords)
        iu, ju = np.triu_indices(n, k=1)
        contact = dist <= (radii[iu] + radii[ju] + 1.0e-9)

        degree = np.zeros(n, dtype=np.int64)
        np.add.at(degree, iu[contact], 1)
        np.add.at(degree, ju[contact], 1)
        return degree <= 1

    def _record_candidate_attempt(self, leaf1: bool, leaf2: bool) -> str:
        if leaf1 and leaf2:
            self._cand_attempts_ll += 1
            return "LL"
        if leaf1 or leaf2:
            self._cand_attempts_ln += 1
            return "LN"
        self._cand_attempts_nn += 1
        return "NN"

    def _record_candidate_success(self, cls: str) -> None:
        if cls == "LL":
            self._cand_success_ll += 1
        elif cls == "LN":
            self._cand_success_ln += 1
        else:
            self._cand_success_nn += 1

    @staticmethod
    def _candidate_leaf_class(leaf1: bool, leaf2: bool) -> str:
        if leaf1 and leaf2:
            return "LL"
        if leaf1 or leaf2:
            return "LN"
        return "NN"

    @staticmethod
    def _candidate_score(
        coords1: np.ndarray,
        radii1: np.ndarray,
        cm1: np.ndarray,
        cand1_idx: int,
        coords2: np.ndarray,
        radii2: np.ndarray,
        cm2: np.ndarray,
        cand2_idx: int,
        gamma_pc: float,
        leaf_cls: str,
    ) -> float:
        """Heuristic candidate score in [0,1], larger means more promising."""
        d1 = float(np.linalg.norm(coords1[cand1_idx] - cm1))
        d2 = float(np.linalg.norm(coords2[cand2_idx] - cm2))
        s1 = d1 + float(radii1[cand1_idx])
        s2 = d2 + float(radii2[cand2_idx])

        # Radial compatibility against gamma shell relation.
        err = abs((s1 + s2) - gamma_pc) / max(gamma_pc, 1.0e-12)
        tau_r = 0.08
        radial = math.exp(-err / tau_r)

        # Class prior from observed empirical success tendency.
        if leaf_cls == "LL":
            leaf_prior = 1.0
        elif leaf_cls == "LN":
            leaf_prior = 0.45
        else:
            leaf_prior = 0.10

        score = 0.5 * leaf_prior + 0.5 * radial
        return max(0.0, min(1.0, score))

    def _record_candidate_score_attempt(self, score: float) -> None:
        self._cand_score_attempt_sum += score
        self._cand_score_attempt_count += 1
        if score >= 0.70:
            self._cand_score_attempt_high += 1
        if score < 0.40:
            self._cand_score_attempt_low += 1

    def _record_candidate_score_success(self, score: float) -> None:
        self._cand_score_success_sum += score
        self._cand_score_success_count += 1
        if score >= 0.70:
            self._cand_score_success_high += 1
        if score < 0.40:
            self._cand_score_success_low += 1
