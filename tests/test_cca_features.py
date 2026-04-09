"""Unit tests for CCA gamma expansion, BV filter, and SSA filter features."""

import numpy as np
import pytest

from pyfracval import config
from pyfracval.cca_agg import CCAggregator


def _make_aggregator(n=64, df=1.8, kf=1.3, seed=42):
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
    )


class TestBoundingVolumePrecheck:
    """Tests for _bounding_volume_precheck static method."""

    def test_reject_when_gamma_below_rmax_diff_times_factor(self):
        assert (
            CCAggregator._bounding_volume_precheck(
                5.0, 100.0, 10.0, gamma_real=True
            )
            is False
        )

    def test_accept_when_gamma_above_rmax_diff_times_factor(self):
        assert (
            CCAggregator._bounding_volume_precheck(
                80.0, 100.0, 10.0, gamma_real=True
            )
            is True
        )

    def test_reject_when_gamma_above_sum_of_rmax(self):
        assert (
            CCAggregator._bounding_volume_precheck(
                200.0, 100.0, 10.0, gamma_real=True
            )
            is False
        )

    def test_accept_when_gamma_between_bounds(self):
        assert (
            CCAggregator._bounding_volume_precheck(
                40.0, 50.0, 30.0, gamma_real=True
            )
            is True
        )

    def test_reject_when_gamma_real_false(self):
        assert (
            CCAggregator._bounding_volume_precheck(
                50.0, 100.0, 30.0, gamma_real=False
            )
            is False
        )

    def test_reject_when_gamma_zero_or_negative(self):
        assert (
            CCAggregator._bounding_volume_precheck(
                0.0, 100.0, 30.0, gamma_real=True
            )
            is False
        )
        assert (
            CCAggregator._bounding_volume_precheck(
                -5.0, 100.0, 30.0, gamma_real=True
            )
            is False
        )

    def test_equal_rmax_values(self):
        assert (
            CCAggregator._bounding_volume_precheck(
                10.0, 50.0, 50.0, gamma_real=True
            )
            is True
        )


class TestSurfaceAccessibleMask:
    """Tests for _surface_accessible_mask static method."""

    def test_single_particle_always_accessible(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        cm = np.array([0.0, 0.0, 0.0])
        r_max = 1.0
        mask = CCAggregator._surface_accessible_mask(coords, radii, cm, r_max)
        assert mask.shape == (1,)
        assert mask[0] is np.True_

    def test_all_accessible_with_zero_threshold(self):
        coords = np.array(
            [[0, 0, 0], [5, 0, 0], [-5, 0, 0], [0, 5, 0], [0, -5, 0]]
        )
        radii = np.ones(5)
        cm = np.array([0.0, 0.0, 0.0])
        r_max = 5.0
        mask = CCAggregator._surface_accessible_mask(
            coords, radii, cm, r_max, min_exposure=0.0
        )
        assert np.all(mask)

    def test_none_accessible_with_high_threshold(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        radii = np.ones(3)
        cm = np.mean(coords, axis=0)
        r_max = 1.0
        mask = CCAggregator._surface_accessible_mask(
            coords, radii, cm, r_max, min_exposure=0.99
        )
        assert np.all(~mask)

    def test_distant_particles_accessible(self):
        coords = np.array([[0, 0, 0], [100, 0, 0]])
        radii = np.ones(2)
        cm = np.array([50.0, 0, 0])
        r_max = 50.0
        mask = CCAggregator._surface_accessible_mask(
            coords, radii, cm, r_max, min_exposure=0.3
        )
        assert mask.shape == (2,)
        assert mask[0] and mask[1]

    def test_uses_config_default_when_none(self):
        orig = config.CCA_SSA_MIN_EXPOSURE
        config.CCA_SSA_MIN_EXPOSURE = 0.42
        try:
            coords = np.array([[0, 0, 0], [5, 0, 0]])
            radii = np.ones(2)
            cm = np.array([2.5, 0, 0])
            r_max = 5.0
            mask = CCAggregator._surface_accessible_mask(coords, radii, cm, r_max)
            assert mask.shape == (2,)
        finally:
            config.CCA_SSA_MIN_EXPOSURE = orig


class TestTelemetryCounters:
    """Tests for telemetry counter initialization."""

    def test_all_counters_initialized_to_zero(self):
        agg = _make_aggregator()
        assert agg._gamma_expansion_hits == 0
        assert agg._gamma_expansion_successes == 0
        assert agg._gamma_expansion_total_steps == 0
        assert agg._bv_filter_rejects == 0
        assert agg._ssa_filter_rejects == 0
