"""Archived CCA candidate-pair ordering policies.

``leaf_soft`` (leaf-pairs-first ordering) and ``leaf_score``/``leaf_hybrid``
(heuristic-scored ordering) were benchmarked against the unranked
(shuffled) baseline ordering at N=512 in the hard regime (see
``docs/source/experiments.md``): both land on the same success rate as
baseline. *How* you search for a contact pair doesn't matter much when the
real question is *whether* a valid contact pair exists at all at the
enforced ``gamma_pc``.

Kept reachable via ``cca_candidate_policy`` for anyone who wants to try a
different scoring heuristic later. Leaf classification and scoring
themselves (``_candidate_leaf_class``/``_candidate_score`` in
``cca/candidates.py``) stay in production code since the per-attempt
telemetry they feed is diagnostic instrumentation used regardless of
policy, not part of what was benchmarked here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import OrchestratorAlgorithmConfig

LeafClassFn = Callable[[bool, bool], str]
ScoreFn = Callable[..., float]


def reorder_candidates_by_policy(
    candidate_policy: str,
    candidate_indices: np.ndarray,
    leaf_mask_1: np.ndarray,
    leaf_mask_2: np.ndarray,
    coords1: np.ndarray,
    radii1: np.ndarray,
    cm1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    cm2: np.ndarray,
    gamma_pc: float,
    algorithm_config: OrchestratorAlgorithmConfig,
    candidate_leaf_class_fn: LeafClassFn,
    candidate_score_fn: ScoreFn,
) -> np.ndarray:
    """Reorder shuffled candidate pairs according to an archived policy."""
    ll: list[np.ndarray] = []
    ln: list[np.ndarray] = []
    nn: list[np.ndarray] = []
    for pair in candidate_indices:
        i = int(pair[0])
        j = int(pair[1])
        cls = candidate_leaf_class_fn(bool(leaf_mask_1[i]), bool(leaf_mask_2[j]))
        if cls == "LL":
            ll.append(pair)
        elif cls == "LN":
            ln.append(pair)
        else:
            nn.append(pair)

    if candidate_policy == "leaf_soft":
        return np.array(ll + ln + nn, dtype=int)

    topk = int(algorithm_config.cca_score_topk_per_class)

    def _score_and_sort(pairs: list[np.ndarray], cls: str) -> list[np.ndarray]:
        if not pairs:
            return []
        n_score = len(pairs) if topk <= 0 else min(topk, len(pairs))
        scored: list[tuple[float, np.ndarray]] = []
        for pair in pairs[:n_score]:
            i = int(pair[0])
            j = int(pair[1])
            score = candidate_score_fn(
                coords1,
                radii1,
                cm1,
                i,
                coords2,
                radii2,
                cm2,
                j,
                float(gamma_pc),
                cls,
            )
            scored.append((score, pair))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored_pairs = [p for _, p in scored]
        return scored_pairs + pairs[n_score:]

    # Both leaf_score (global score order) and leaf_hybrid (score within
    # leaf-priority class) end up doing the same per-class scoring pass -
    # the two policies were probed head-to-head and found indistinguishable
    # (docs/source/experiments.md), so there's no separate "global" merge
    # left to preserve.
    merged = (
        _score_and_sort(ll, "LL")
        + _score_and_sort(ln, "LN")
        + _score_and_sort(nn, "NN")
    )
    return np.array(merged, dtype=int)
