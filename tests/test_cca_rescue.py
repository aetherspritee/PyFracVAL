"""Unit tests for pyfracval.cca.rescue's detect-and-drop failure rescue."""

import numpy as np

from pyfracval.cca.rescue import retry_sticking_with_drops, select_drop_candidates
from pyfracval.cca_agg import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.schemas import OverlapCensus


def _make_census(
    n_off1: int,
    n_off2: int,
    cluster1_size: int,
    cluster2_size: int,
) -> OverlapCensus:
    return OverlapCensus(
        n_pairs_overlapping=max(n_off1, n_off2),
        n_particles_cluster1_offending=n_off1,
        n_particles_cluster2_offending=n_off2,
        offending_indices_cluster1=list(range(n_off1)),
        offending_indices_cluster2=list(range(n_off2)),
        max_overlap_fraction=0.5,
        mean_overlap_fraction=0.3,
        severity_histogram={"0-0.05": 0, "0.05-0.15": 0, "0.15-0.3": 1, "0.3+": 0},
        cluster1_size=cluster1_size,
        cluster2_size=cluster2_size,
    )


class TestSelectDropCandidates:
    def test_within_budget_returns_indices(self):
        census = _make_census(2, 3, cluster1_size=20, cluster2_size=20)
        result = select_drop_candidates(
            census, max_drop_particles=5, max_drop_fraction=0.5
        )
        assert result is not None
        drop1, drop2 = result
        assert drop1 == [0, 1]
        assert drop2 == [0, 1, 2]

    def test_over_absolute_budget_returns_none(self):
        census = _make_census(6, 1, cluster1_size=20, cluster2_size=20)
        result = select_drop_candidates(
            census, max_drop_particles=5, max_drop_fraction=0.5
        )
        assert result is None

    def test_over_relative_budget_returns_none(self):
        # cluster1_size=10, max_fraction=0.1 -> budget=ceil(0.1*10)=1,
        # but n_off1=2 exceeds it even though it's under max_drop_particles.
        census = _make_census(2, 0, cluster1_size=10, cluster2_size=20)
        result = select_drop_candidates(
            census, max_drop_particles=5, max_drop_fraction=0.1
        )
        assert result is None

    def test_combined_budget_takes_the_tighter_bound(self):
        # cluster_size=200, max_fraction=0.02 -> budget=ceil(0.02*200)=4,
        # max_drop_particles=5 -> effective budget is min(5,4)=4.
        census = _make_census(4, 0, cluster1_size=200, cluster2_size=200)
        assert (
            select_drop_candidates(census, max_drop_particles=5, max_drop_fraction=0.02)
            is not None
        )
        census_over = _make_census(5, 0, cluster1_size=200, cluster2_size=200)
        assert (
            select_drop_candidates(
                census_over, max_drop_particles=5, max_drop_fraction=0.02
            )
            is None
        )

    def test_zero_offending_is_within_any_budget(self):
        census = _make_census(0, 0, cluster1_size=10, cluster2_size=10)
        result = select_drop_candidates(
            census, max_drop_particles=1, max_drop_fraction=0.01
        )
        assert result == ([], [])


class TestRetryStickingWithDrops:
    def test_dropping_offending_particle_resolves_overlap(self):
        # cluster1: two particles, one far away (fine) and one overlapping
        # cluster2's single particle. Dropping the offending one should
        # leave a valid, non-overlapping combined result.
        coords1 = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        radii1 = np.array([1.0, 1.0])
        coords2 = np.array([[0.5, 0.0, 0.0]])
        radii2 = np.array([1.0])

        result = retry_sticking_with_drops(
            coords1,
            radii1,
            coords2,
            radii2,
            drop_idx1=[0],
            drop_idx2=[],
            tol_ov=1e-6,
        )
        assert result is not None
        combined_coords, combined_radii = result
        assert combined_coords.shape[0] == 2  # 1 kept from cluster1 + 1 from cluster2
        assert combined_radii.shape[0] == 2

    def test_insufficient_drop_still_overlapping_returns_none(self):
        # Both cluster1 particles overlap cluster2's particle; dropping
        # only one leaves the other still overlapping.
        coords1 = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        radii1 = np.array([1.0, 1.0])
        coords2 = np.array([[0.5, 0.0, 0.0]])
        radii2 = np.array([1.0])

        result = retry_sticking_with_drops(
            coords1,
            radii1,
            coords2,
            radii2,
            drop_idx1=[1],  # only drop one of the two offenders
            drop_idx2=[],
            tol_ov=1e-6,
        )
        assert result is None

    def test_dropping_entire_cluster_returns_none(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[0.5, 0.0, 0.0]])
        radii2 = np.array([1.0])

        result = retry_sticking_with_drops(
            coords1,
            radii1,
            coords2,
            radii2,
            drop_idx1=[0],  # empties cluster1 entirely
            drop_idx2=[],
            tol_ov=1e-6,
        )
        assert result is None

    def test_no_drop_needed_already_non_overlapping(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[3.0, 0.0, 0.0]])
        radii2 = np.array([1.0])

        result = retry_sticking_with_drops(
            coords1, radii1, coords2, radii2, drop_idx1=[], drop_idx2=[], tol_ov=1e-6
        )
        assert result is not None
        combined_coords, _ = result
        assert combined_coords.shape[0] == 2


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


class TestDropRescueConfigWiring:
    """Confirms the config auto-enable and instance-attribute wiring, not
    a full end-to-end CCA run (that's covered by the benchmark scripts,
    which are slow/RNG-dependent)."""

    def test_drop_rescue_auto_enables_census(self):
        cfg = OrchestratorAlgorithmConfig(cca_drop_rescue_enabled=True)
        assert cfg.cca_overlap_census_enabled is True

    def test_default_config_has_both_disabled(self):
        cfg = OrchestratorAlgorithmConfig()
        assert cfg.cca_drop_rescue_enabled is False
        assert cfg.cca_overlap_census_enabled is False

    def test_aggregator_initializes_drop_rescue_telemetry(self):
        agg = _make_aggregator()
        assert agg._drop_rescue_attempts == 0
        assert agg._drop_rescue_successes == 0
        assert agg._particles_dropped_total == 0
        assert agg._last_overlap_failure_geometry is None


class TestCoordsNextTrimming:
    """Regression coverage for the coords_next/radii_next trimming fix
    _run_iteration needed once particle counts could shrink mid-round."""

    def test_run_iteration_output_size_matches_fill_idx_without_drops(self):
        # Sanity check: with drop-rescue disabled (default), trimming to
        # fill_idx must be a no-op - coords/radii size stays exactly N
        # after an iteration, same as before this change.
        n = 16
        agg = _make_aggregator(n=n)
        ok = agg._run_iteration()
        assert ok
        assert agg.coords.shape[0] == n
        assert agg.radii.shape[0] == n
