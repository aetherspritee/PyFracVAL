"""Unit tests for pyfracval.cca.matching's matching-based pairing strategies."""

import numpy as np

from pyfracval.cca.matching import (
    build_feasibility_graph,
    cluster_leaf_fraction,
    cluster_pair_leaf_class,
    leaf_weighted_matching,
    max_cardinality_matching,
)
from pyfracval.cca_agg import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig


def _sorted_pairs(pairs):
    return sorted(tuple(sorted(p)) for p in pairs)


class TestMaxCardinalityMatching:
    def test_simple_chain_picks_two_disjoint_edges(self):
        # 0-1-2-3 chain: greedy first-fit on (0,1) would strand 2 and 3
        # apart if it also grabbed (2,3) is fine here, but the point is
        # the matcher must find *a* maximum matching, not just any one.
        adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        pairs = max_cardinality_matching(adj, [0, 1, 2, 3])
        assert len(pairs) == 2
        matched_nodes = {n for pair in pairs for n in pair}
        assert matched_nodes == {0, 1, 2, 3}

    def test_greedy_trap_is_solved_optimally(self):
        # Node 0 only connects to 1; node 1 ALSO connects to 2 and 3, which
        # connect to each other. Greedy first-fit visiting 0 first grabs
        # (0,1), stranding 2 and 3 unmatched even though (2,3) is
        # available. An optimal matcher must find (0,1) and (2,3) - the
        # exact "rescuable by a different pairing" scenario
        # pairing_frustration.md diagnosed.
        adj = {0: {1}, 1: {0, 2, 3}, 2: {1, 3}, 3: {1, 2}}
        pairs = max_cardinality_matching(adj, [0, 1, 2, 3])
        assert len(pairs) == 2
        matched_nodes = {n for pair in pairs for n in pair}
        assert matched_nodes == {0, 1, 2, 3}

    def test_odd_node_left_unmatched(self):
        adj = {0: {1}, 1: {0}, 2: set()}
        pairs = max_cardinality_matching(adj, [0, 1, 2])
        assert _sorted_pairs(pairs) == [(0, 1)]

    def test_no_edges_returns_empty(self):
        adj = {0: set(), 1: set(), 2: set()}
        assert max_cardinality_matching(adj, [0, 1, 2]) == []

    def test_all_edges_produces_provably_optimal_cardinality(self):
        # Complete graph on 5 nodes: optimal matching size is 2 (one node
        # necessarily left over).
        nodes = [0, 1, 2, 3, 4]
        adj = {i: {j for j in nodes if j != i} for i in nodes}
        pairs = max_cardinality_matching(adj, nodes)
        assert len(pairs) == 2
        matched_nodes = {n for pair in pairs for n in pair}
        assert len(matched_nodes) == 4  # exactly one node unmatched

    def test_returned_pairs_are_valid_edges_and_disjoint(self):
        adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1, 3}, 3: {2}}
        pairs = max_cardinality_matching(adj, [0, 1, 2, 3])
        seen = set()
        for i, j in pairs:
            assert j in adj[i] and i in adj[j]
            assert i not in seen and j not in seen
            seen.add(i)
            seen.add(j)


class TestLeafWeightedMatching:
    def test_prefers_higher_weight_among_equal_cardinality_solutions(self):
        # Two disjoint options of equal cardinality (1 edge each, since 0
        # only connects to 1 or 2, not both simultaneously): (0,1) weight
        # 1.0 vs (0,2) weight 0.1. Both leave the same number of nodes
        # unmatched (cardinality-equal), so weight must decide.
        adj = {0: {1, 2}, 1: {0}, 2: {0}}
        weights = {(0, 1): 1.0, (1, 0): 1.0, (0, 2): 0.1, (2, 0): 0.1}
        pairs = leaf_weighted_matching(
            adj, [0, 1, 2], edge_weight_fn=lambda i, j: weights[(i, j)]
        )
        assert _sorted_pairs(pairs) == [(0, 1)]

    def test_never_sacrifices_cardinality_for_weight(self):
        # (0,1) alone has a huge weight but leaves 2,3 unmatched (size 1).
        # (0,2)+(1,3) or (0,3)+(1,2) achieve size 2 with tiny weight.
        # Cardinality must win: expect 2 matched pairs, not 1.
        adj = {0: {1, 2, 3}, 1: {0, 2, 3}, 2: {0, 1}, 3: {0, 1}}

        def weight_fn(i, j):
            return 1000.0 if {i, j} == {0, 1} else 0.01

        pairs = leaf_weighted_matching(adj, [0, 1, 2, 3], edge_weight_fn=weight_fn)
        assert len(pairs) == 2
        matched_nodes = {n for pair in pairs for n in pair}
        assert matched_nodes == {0, 1, 2, 3}


class TestBuildFeasibilityGraph:
    def test_matches_pairwise_gamma_check(self):
        # cluster_props: idx -> (mass, rg, cm, r_max, radii)
        cluster_props = {
            0: (1.0, 5.0, None, 3.0, np.array([3.0])),
            1: (1.0, 5.0, None, 3.0, np.array([3.0])),
            2: (0.0, 0.0, None, 0.0, np.array([])),  # empty cluster, excluded
        }

        def gamma_fn(props1, props2):
            # Always "feasible", gamma_pc well under sum_rmax.
            return True, 1.0

        adj = build_feasibility_graph(cluster_props, gamma_fn, pairing_factor=1.10)
        assert set(adj.keys()) == {0, 1}  # empty cluster 2 excluded
        assert adj[0] == {1}
        assert adj[1] == {0}

    def test_infeasible_gamma_produces_no_edge(self):
        cluster_props = {
            0: (1.0, 5.0, None, 3.0, np.array([3.0])),
            1: (1.0, 5.0, None, 3.0, np.array([3.0])),
        }

        def gamma_fn(props1, props2):
            return True, 1000.0  # gamma_pc far exceeds sum_rmax

        adj = build_feasibility_graph(cluster_props, gamma_fn, pairing_factor=1.10)
        assert adj[0] == set()
        assert adj[1] == set()

    def test_gamma_not_real_produces_no_edge(self):
        cluster_props = {
            0: (1.0, 5.0, None, 3.0, np.array([3.0])),
            1: (1.0, 5.0, None, 3.0, np.array([3.0])),
        }

        def gamma_fn(props1, props2):
            return False, 1.0

        adj = build_feasibility_graph(cluster_props, gamma_fn, pairing_factor=1.10)
        assert adj[0] == set()


class TestClusterLeafHelpers:
    def test_cluster_leaf_fraction_empty_mask(self):
        assert cluster_leaf_fraction(np.zeros(0, dtype=bool)) == 0.0

    def test_cluster_leaf_fraction_mean(self):
        mask = np.array([True, True, False, False])
        assert cluster_leaf_fraction(mask) == 0.5

    def test_cluster_pair_leaf_class_ll(self):
        assert cluster_pair_leaf_class(0.9, 0.9) == "LL"

    def test_cluster_pair_leaf_class_nn(self):
        assert cluster_pair_leaf_class(0.1, 0.1) == "NN"

    def test_cluster_pair_leaf_class_ln(self):
        assert cluster_pair_leaf_class(0.9, 0.1) == "LN"
        assert cluster_pair_leaf_class(0.1, 0.9) == "LN"

    def test_cluster_pair_leaf_class_threshold_boundary(self):
        # Exactly at the threshold is not > threshold, so both count as
        # non-leaf.
        assert cluster_pair_leaf_class(0.5, 0.5, threshold=0.5) == "NN"


def _make_aggregator(n=64, df=1.8, kf=1.3, seed=42, algorithm_config=None):
    rng = np.random.RandomState(seed)
    coords = rng.randn(n, 3)
    radii = np.ones(n) * 10.0
    i_orden = np.array([[0, n - 1, n]])
    return CCAggregator(
        initial_coords=coords,
        initial_radii=radii,
        initial_i_orden=i_orden,
        n_total=n,
        df=df,
        kf=kf,
        tol_ov=1e-4,
        ext_case=0,
        algorithm_config=algorithm_config,
    )


class TestGeneratePairsMatchingStrategy:
    """Integration coverage: a real CCAggregator using the new strategies
    still produces a valid id_agglomerated matrix."""

    def test_matching_strategy_produces_valid_id_agglomerated(self):
        cfg = OrchestratorAlgorithmConfig(cca_pairing_strategy="matching")
        agg = _make_aggregator(n=88, algorithm_config=cfg)
        result = agg._generate_pairs()
        assert result is not None
        id_agglomerated, cluster_props = result
        # Symmetric
        assert np.array_equal(id_agglomerated, id_agglomerated.T)
        # Every non-empty cluster is accounted for (paired or odd-one-out).
        should_be_paired = np.array([cluster_props[i][0] > 0.0 for i in range(agg.i_t)])
        paired_status = np.sum(id_agglomerated, axis=0) + np.sum(
            id_agglomerated, axis=1
        )
        assert not np.any(paired_status[should_be_paired] == 0)

    def test_matching_leaf_weighted_strategy_produces_valid_id_agglomerated(self):
        cfg = OrchestratorAlgorithmConfig(cca_pairing_strategy="matching_leaf_weighted")
        agg = _make_aggregator(n=88, algorithm_config=cfg)
        result = agg._generate_pairs()
        assert result is not None
        id_agglomerated, cluster_props = result
        assert np.array_equal(id_agglomerated, id_agglomerated.T)
        should_be_paired = np.array([cluster_props[i][0] > 0.0 for i in range(agg.i_t)])
        paired_status = np.sum(id_agglomerated, axis=0) + np.sum(
            id_agglomerated, axis=1
        )
        assert not np.any(paired_status[should_be_paired] == 0)

    def test_greedy_strategy_is_unaffected_default(self):
        cfg = OrchestratorAlgorithmConfig()
        assert cfg.cca_pairing_strategy == "greedy"
        agg = _make_aggregator(n=88, algorithm_config=cfg)
        result = agg._generate_pairs()
        assert result is not None
