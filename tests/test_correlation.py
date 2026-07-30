"""Tests for the density-density correlation estimator.

The estimator is validated against cases with analytically known answers
before being trusted on aggregates, since an f(r) implementation that is
subtly wrong would still produce plausible-looking curves.
"""

import numpy as np
import pytest

from pyfracval.correlation import (
    density_correlation,
    fit_correlation_slope,
    sphere_intersection_volume,
)


class TestSphereIntersectionVolume:
    def test_disjoint_spheres_share_nothing(self):
        v = sphere_intersection_volume(
            np.array([1.0]), np.array([1.0]), np.array([3.0])
        )
        assert v[0] == 0.0

    def test_coincident_equal_spheres_share_their_whole_volume(self):
        v = sphere_intersection_volume(
            np.array([2.0]), np.array([2.0]), np.array([0.0])
        )
        assert v[0] == pytest.approx((4.0 / 3.0) * np.pi * 8.0)

    def test_containment_gives_smaller_sphere_volume(self):
        # A radius-1 sphere entirely inside a radius-5 one.
        v = sphere_intersection_volume(
            np.array([5.0]), np.array([1.0]), np.array([2.0])
        )
        assert v[0] == pytest.approx((4.0 / 3.0) * np.pi * 1.0)

    def test_touching_spheres_share_nothing(self):
        v = sphere_intersection_volume(
            np.array([1.0]), np.array([2.0]), np.array([3.0])
        )
        assert v[0] == pytest.approx(0.0, abs=1e-12)

    def test_half_overlap_matches_closed_form(self):
        # Two unit spheres at d=1. Closed form:
        # V = pi*(2r-d)^2*(d^2 + 2dr - 3r^2 + 2dr + 6r^2 - 3r^2)/(12d)
        # For r1=r2=1, d=1 this is pi*1*(1+2-3+2+6-3)/12 = 5*pi/12.
        v = sphere_intersection_volume(
            np.array([1.0]), np.array([1.0]), np.array([1.0])
        )
        assert v[0] == pytest.approx(5.0 * np.pi / 12.0)

    def test_vectorizes_over_arrays(self):
        r1 = np.array([1.0, 1.0, 5.0])
        r2 = np.array([1.0, 1.0, 1.0])
        d = np.array([3.0, 0.0, 2.0])
        v = sphere_intersection_volume(r1, r2, d)
        assert v.shape == (3,)
        assert v[0] == 0.0
        assert v[1] > 0.0
        assert v[2] > 0.0


class TestDensityCorrelation:
    def test_single_sphere_self_overlap_is_one_at_zero_shift(self):
        # f(r) is normalized by total volume, so a lone sphere displaced
        # by nothing must overlap itself completely.
        coords = np.zeros((1, 3))
        radii = np.array([1.0])
        # Smallest sampled radius is rp/10, small but not zero, so the
        # value should be close to (but under) 1.
        res = density_correlation(
            coords, radii, n_orientations=8, n_radii=5, rng=np.random.default_rng(0)
        )
        assert 0.8 < res["f"][0] <= 1.0

    def test_correlation_decreases_with_distance(self):
        rng = np.random.default_rng(2)
        coords = rng.normal(scale=5.0, size=(40, 3))
        radii = np.ones(40)
        res = density_correlation(coords, radii, n_orientations=16, n_radii=8, rng=rng)
        # Monotone decrease is not guaranteed pointwise from sampling
        # noise, but the far end must be far below the near end.
        assert res["f"][0] > res["f"][-1]
        assert res["f"][-1] < 0.5 * res["f"][0]

    def test_dense_uniform_ball_recovers_slope_near_zero(self):
        # A solid, space-filling ball is Df = 3, so f(r) should be nearly
        # flat (slope Df - 3 = 0) inside it. This is the estimator's
        # sanity check against a known dimension.
        rng = np.random.default_rng(7)
        n = 900
        # Uniformly fill a ball of radius R.
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        radius_ball = 12.0
        rad = radius_ball * rng.random(n) ** (1.0 / 3.0)
        coords = v * rad[:, None]
        radii = np.full(n, 1.0)

        res = density_correlation(coords, radii, n_orientations=24, n_radii=24, rng=rng)
        fit = fit_correlation_slope(res)
        # Expect Df near 3 for a filled ball. Tolerance is loose: this is
        # a finite, randomly-sampled ball, not an infinite continuum.
        assert fit["df_estimate"] == pytest.approx(3.0, abs=0.6)

    def test_empty_aggregate_rejected(self):
        with pytest.raises(ValueError):
            density_correlation(np.zeros((0, 3)), np.array([]))


class TestFitCorrelationSlope:
    def test_reports_when_window_is_too_short_to_fit(self):
        # A tiny aggregate has almost no scaling range between 2*rp and Rg.
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        res = density_correlation(
            coords, radii, n_orientations=4, n_radii=6, rng=np.random.default_rng(1)
        )
        fit = fit_correlation_slope(res)
        # Either it fits, or it honestly reports too few points - but it
        # must not invent a slope from one or two samples.
        assert fit["n_points"] >= 3 or np.isnan(fit["slope"])
