"""Matching-based alternatives to `_generate_pairs()`'s greedy first-fit.

docs/source/pairing_frustration.md diagnosed the actual bottleneck in
hard-regime CCA sticking: 97.4% of failures had a valid alternative
pairing available in the same cluster pool that greedy first-fit never
considered, because it commits to the first feasible partner it finds per
cluster and never looks back. These functions replace that first-fit
choice with an exact maximum-cardinality matching over the same cheap
gamma-feasibility graph `_generate_pairs()` already computes - not the
expensive "actually attempt sticking" graph
`benchmarks/pairing_frustration_probe.py` builds for offline diagnosis
(3 retries per edge), which would be far too slow and RNG-perturbing to
run on the hot path.

Pure functions, no ``self`` - unit-testable directly and reusable from
both `_generate_pairs()` and any future caller.
"""

from typing import Callable

Adjacency = dict[int, set[int]]


def build_feasibility_graph(
    cluster_props: dict[int, tuple],
    gamma_fn: Callable[[tuple, tuple], tuple[bool, float]],
    pairing_factor: float,
) -> Adjacency:
    """Cheap gamma-feasibility adjacency over non-empty cluster indices.

    Factors out the same strict/relaxed gamma gate `_generate_pairs()`'s
    greedy loop computes per pair (``gamma_real and gamma_pc < sum_rmax *
    pairing_factor``), so both the greedy and matching code paths share
    one feasibility test rather than duplicating it. Does not distinguish
    strict vs. relaxed matches the way the greedy path's logging does -
    both are simply "feasible" edges here.

    Parameters
    ----------
    cluster_props : dict[int, tuple]
        cluster_idx -> (mass, rg, cm, r_max, radii), the same cache shape
        `_generate_pairs()` already builds. Clusters with mass 0.0 (empty)
        are excluded from the returned graph's nodes.
    gamma_fn : callable
        Bound `self._calculate_cca_gamma` - takes two (m, rg, cm, r_max,
        radii)-shaped props tuples, returns (gamma_real, gamma_pc).
    pairing_factor : float
        The relaxation factor (e.g. 1.10) applied to sum_rmax.
    """
    nodes = [i for i, props in cluster_props.items() if props[0] > 0.0]
    adj: Adjacency = {i: set() for i in nodes}

    for idx_a, i in enumerate(nodes):
        m1, rg1, _, r_max1, radii1 = cluster_props[i]
        props1 = (m1, rg1, None, r_max1, radii1)
        for j in nodes[idx_a + 1 :]:
            m2, rg2, _, r_max2, radii2 = cluster_props[j]
            props2 = (m2, rg2, None, r_max2, radii2)

            gamma_real, gamma_pc = gamma_fn(props1, props2)
            sum_rmax = r_max1 + r_max2
            if gamma_real and gamma_pc < sum_rmax * pairing_factor:
                adj[i].add(j)
                adj[j].add(i)

    return adj


def max_cardinality_matching(adj: Adjacency, nodes: list[int]) -> list[tuple[int, int]]:
    """Exact maximum-cardinality matching via memoized brute-force DP.

    Generalizes `benchmarks/pairing_frustration_probe.py::_max_matching_size`
    (which only counts a matching's size) into one that reconstructs the
    actual assignment. Round pool sizes are small - bounded by roughly
    ``1 / n_subcl_percentage``, empirically <=16 per the probe's own
    numbers - so ``2**n`` memoized states with O(n) work each is cheap
    relative to per-pair sticking cost; a real Blossom-algorithm
    implementation (O(n^3), handles arbitrary graphs including odd
    cycles a greedy DP can't) is unnecessary complexity at this scale.
    Do not "upgrade" this reflexively if round pool sizes ever grow
    significantly (e.g. a much smaller ``n_subcl_percentage`` default).

    Returns a list of (i, j) matched pairs; unmatched nodes are simply
    absent from the result (the odd-cluster-out / pass-through case is
    handled by the caller, same as the greedy path).
    """
    memo: dict[frozenset, tuple[int, list[tuple[int, int]]]] = {}

    def rec(remaining: frozenset) -> tuple[int, list[tuple[int, int]]]:
        if len(remaining) <= 1:
            return 0, []
        if remaining in memo:
            return memo[remaining]

        remaining_list = list(remaining)
        first = remaining_list[0]
        rest = remaining_list[1:]

        best_size, best_edges = rec(frozenset(rest))  # leave `first` unmatched
        for other in rest:
            if other in adj[first]:
                candidate_rest = frozenset(x for x in rest if x != other)
                cand_size, cand_edges = rec(candidate_rest)
                cand_size += 1
                if cand_size > best_size:
                    best_size = cand_size
                    best_edges = cand_edges + [(first, other)]

        memo[remaining] = (best_size, best_edges)
        return memo[remaining]

    _, edges = rec(frozenset(nodes))
    return edges


def leaf_weighted_matching(
    adj: Adjacency,
    nodes: list[int],
    edge_weight_fn: Callable[[int, int], float],
) -> list[tuple[int, int]]:
    """Maximum-cardinality matching, with total edge weight as a tiebreaker
    among cardinality-optimal solutions.

    Cardinality is optimized first rather than weight outright: sacrificing
    a matchable pair to chase a higher-weight edge elsewhere would directly
    contradict the diagnosed problem (rounds failing because too few
    clusters get paired at all, not because of which specific clusters get
    paired). Implemented by comparing ``(size, weight)`` tuples
    lexicographically at each DP step - Python's native tuple comparison
    already does exactly "maximize the first component, use the second as
    a tiebreaker," so this needs no separate two-pass DP.

    ``edge_weight_fn(i, j)`` should return the caller's per-edge weight
    (e.g. from a leaf-class classification of the cluster pair).
    """
    memo: dict[frozenset, tuple[int, float, list[tuple[int, int]]]] = {}

    def rec(remaining: frozenset) -> tuple[int, float, list[tuple[int, int]]]:
        if len(remaining) <= 1:
            return 0, 0.0, []
        if remaining in memo:
            return memo[remaining]

        remaining_list = list(remaining)
        first = remaining_list[0]
        rest = remaining_list[1:]

        best = rec(frozenset(rest))  # leave `first` unmatched
        for other in rest:
            if other in adj[first]:
                candidate_rest = frozenset(x for x in rest if x != other)
                c_size, c_weight, c_edges = rec(candidate_rest)
                candidate = (
                    c_size + 1,
                    c_weight + edge_weight_fn(first, other),
                    c_edges + [(first, other)],
                )
                if (candidate[0], candidate[1]) > (best[0], best[1]):
                    best = candidate

        memo[remaining] = best
        return best

    _, _, edges = rec(frozenset(nodes))
    return edges


def cluster_leaf_fraction(leaf_mask) -> float:
    """Fraction of leaf-classified (contact-degree <= 1) particles in a
    cluster, given its `_CandidatesMixin._leaf_mask_for_cluster` output."""
    if leaf_mask.size == 0:
        return 0.0
    return float(leaf_mask.mean())


def cluster_pair_leaf_class(
    leaf_fraction_i: float, leaf_fraction_j: float, threshold: float = 0.5
) -> str:
    """Classify an edge (i, j) as "LL"/"LN"/"NN" from each cluster's own
    leaf-fraction (mean of its per-particle leaf mask), using the same
    three-bucket structure as `candidates.py`'s per-particle
    `_candidate_leaf_class` for consistency - but operating on cluster
    pairs, not individual particles, which needed a new aggregation rule
    since none existed. The threshold (default 0.5, i.e. "more than half
    of this cluster's particles are surface-exposed leaves") is an
    unvalidated free parameter - treat it as such rather than asserting
    it's correct; see docs/source/matching_pairing.md for whether
    leaf-weighting shows any effect at all before tuning it further.
    """
    is_leaf_i = leaf_fraction_i > threshold
    is_leaf_j = leaf_fraction_j > threshold
    if is_leaf_i and is_leaf_j:
        return "LL"
    if is_leaf_i or is_leaf_j:
        return "LN"
    return "NN"
