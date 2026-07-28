"""Unit tests for pyfracval.overlap_statistics's opt-in overlap-failure census."""

import numpy as np

from pyfracval.cca_agg import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.overlap_statistics import (
    _cross_overlap_pairs_kernel,
    compute_overlap_census,
)


def _make_aggregator(n=32, df=1.8, kf=1.3, seed=7, algorithm_config=None):
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


class TestCrossOverlapPairsKernel:
    def test_no_overlap_returns_empty(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[10.0, 0.0, 0.0]])
        radii2 = np.array([1.0])

        pair_i, pair_j, pair_ov = _cross_overlap_pairs_kernel(
            coords1, radii1, coords2, radii2, max_pairs=100
        )
        assert len(pair_i) == 0
        assert len(pair_j) == 0
        assert len(pair_ov) == 0

    def test_exact_contact_is_not_overlap(self):
        # dist == r_sum exactly: dist_sq < r_sum*r_sum is False at equality.
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[2.0, 0.0, 0.0]])
        radii2 = np.array([1.0])

        pair_i, _, _ = _cross_overlap_pairs_kernel(
            coords1, radii1, coords2, radii2, max_pairs=100
        )
        assert len(pair_i) == 0

    def test_known_overlap_fraction(self):
        # dist=0.5, r_sum=2 (r1=1, r2=1): overlap = (2-0.5)/min(1,1) = 1.5
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[0.5, 0.0, 0.0]])
        radii2 = np.array([1.0])

        pair_i, pair_j, pair_ov = _cross_overlap_pairs_kernel(
            coords1, radii1, coords2, radii2, max_pairs=100
        )
        assert list(pair_i) == [0]
        assert list(pair_j) == [0]
        np.testing.assert_allclose(pair_ov, [1.5])

    def test_scans_all_pairs_no_early_exit(self):
        # 3 particles in cluster1, all overlapping the single cluster2
        # particle - the whole point of this kernel vs. the hot-path
        # scalar check is that it doesn't stop at the first hit.
        coords1 = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        radii1 = np.array([1.0, 1.0, 1.0])
        coords2 = np.array([[0.0, 0.0, 0.0]])
        radii2 = np.array([1.0])

        pair_i, pair_j, pair_ov = _cross_overlap_pairs_kernel(
            coords1, radii1, coords2, radii2, max_pairs=100
        )
        assert len(pair_i) == 3
        assert set(pair_i) == {0, 1, 2}

    def test_respects_max_pairs_cap(self):
        n = 10
        coords1 = np.zeros((n, 3))
        radii1 = np.ones(n)
        coords2 = np.zeros((n, 3))
        radii2 = np.ones(n)

        pair_i, _, _ = _cross_overlap_pairs_kernel(
            coords1, radii1, coords2, radii2, max_pairs=5
        )
        assert len(pair_i) == 5


class TestComputeOverlapCensus:
    def test_no_overlap_census(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[10.0, 0.0, 0.0]])
        radii2 = np.array([1.0])

        census = compute_overlap_census(coords1, radii1, coords2, radii2)
        assert census.n_pairs_overlapping == 0
        assert census.n_particles_cluster1_offending == 0
        assert census.n_particles_cluster2_offending == 0
        assert census.max_overlap_fraction == 0.0
        assert census.mean_overlap_fraction == 0.0
        assert sum(census.severity_histogram.values()) == 0

    def test_offending_particle_counts_deduplicate(self):
        # cluster1 particle 0 overlaps both cluster2 particles - should
        # count as 1 offending particle on side 1, 2 on side 2.
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([2.0])
        coords2 = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        radii2 = np.array([1.0, 1.0])

        census = compute_overlap_census(coords1, radii1, coords2, radii2)
        assert census.n_pairs_overlapping == 2
        assert census.n_particles_cluster1_offending == 1
        assert census.n_particles_cluster2_offending == 2
        assert census.offending_indices_cluster1 == [0]
        assert census.offending_indices_cluster2 == [0, 1]

    def test_severity_histogram_buckets(self):
        # Construct pairs with known overlap fractions in each bucket.
        # overlap = (r_sum - dist) / min(r1, r2), r1=r2=1 => r_sum=2
        # dist=1.98 -> overlap=0.02 (bucket 0-0.05)
        # dist=1.90 -> overlap=0.10 (bucket 0.05-0.15)
        # dist=1.75 -> overlap=0.25 (bucket 0.15-0.3)
        # dist=1.00 -> overlap=1.00 (bucket 0.3+)
        coords1 = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        radii1 = np.ones(4)
        coords2 = np.array(
            [[1.98, 0.0, 0.0], [1.90, 0.0, 0.0], [1.75, 0.0, 0.0], [1.00, 0.0, 0.0]]
        )
        radii2 = np.ones(4)

        # Each cluster1[i] should only be compared against cluster2[i]
        # for this test to isolate one pair per bucket - use single-particle
        # calls instead to avoid unwanted cross pairs.
        for i in range(4):
            c = compute_overlap_census(
                coords1[i : i + 1],
                radii1[i : i + 1],
                coords2[i : i + 1],
                radii2[i : i + 1],
            )
            assert c.n_pairs_overlapping == 1

        c0 = compute_overlap_census(
            coords1[0:1], radii1[0:1], coords2[0:1], radii2[0:1]
        )
        assert c0.severity_histogram["0-0.05"] == 1
        c1 = compute_overlap_census(
            coords1[1:2], radii1[1:2], coords2[1:2], radii2[1:2]
        )
        assert c1.severity_histogram["0.05-0.15"] == 1
        c2 = compute_overlap_census(
            coords1[2:3], radii1[2:3], coords2[2:3], radii2[2:3]
        )
        assert c2.severity_histogram["0.15-0.3"] == 1
        c3 = compute_overlap_census(
            coords1[3:4], radii1[3:4], coords2[3:4], radii2[3:4]
        )
        assert c3.severity_histogram["0.3+"] == 1

    def test_cluster_sizes_recorded(self):
        coords1 = np.zeros((3, 3))
        radii1 = np.ones(3)
        coords2 = np.zeros((5, 3)) + 100.0
        radii2 = np.ones(5)

        census = compute_overlap_census(coords1, radii1, coords2, radii2)
        assert census.cluster1_size == 3
        assert census.cluster2_size == 5

    def test_max_and_mean_overlap_fraction(self):
        coords1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        radii1 = np.array([1.0, 1.0])
        coords2 = np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        radii2 = np.array([1.0, 1.0])

        # Pairs formed: (0,0) dist=1 -> ov=(2-1)/1=1.0
        #               (0,1) dist=1.5 -> ov=(2-1.5)/1=0.5
        #               (1,0) dist=1 -> ov=1.0
        #               (1,1) dist=1.5 -> ov=0.5
        census = compute_overlap_census(coords1, radii1, coords2, radii2)
        assert census.n_pairs_overlapping == 4
        np.testing.assert_allclose(census.max_overlap_fraction, 1.0)
        np.testing.assert_allclose(census.mean_overlap_fraction, 0.75)


class TestOverlapCensusHookIntegration:
    """Integration coverage for the opt-in hook wired into
    fallbacks.py::_perform_cca_sticking, without depending on a slow,
    RNG-dependent real hard-regime failure to occur."""

    def test_census_populated_directly_via_failure_hook(self):
        cfg = OrchestratorAlgorithmConfig(cca_overlap_census_enabled=True)
        agg = _make_aggregator(algorithm_config=cfg)
        assert agg._last_overlap_census is None

        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[0.5, 0.0, 0.0]])
        radii2 = np.array([1.0])
        agg._run_overlap_census_on_failure(coords1, radii1, coords2, radii2)

        assert agg._last_overlap_census is not None
        assert agg._last_overlap_census.n_pairs_overlapping == 1

    def test_hook_noop_when_last_attempt_coords_are_none(self):
        cfg = OrchestratorAlgorithmConfig(cca_overlap_census_enabled=True)
        agg = _make_aggregator(algorithm_config=cfg)

        radii1 = np.array([1.0])
        radii2 = np.array([1.0])
        agg._run_overlap_census_on_failure(None, radii1, None, radii2)

        assert agg._last_overlap_census is None

    def test_disabled_flag_means_last_overlap_census_stays_none_by_default(self):
        cfg = OrchestratorAlgorithmConfig()  # cca_overlap_census_enabled=False
        agg = _make_aggregator(algorithm_config=cfg)
        assert agg._last_overlap_census is None
