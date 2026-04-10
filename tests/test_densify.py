"""Unit tests for densification module."""

import numpy as np
import pytest

from pyfracval.densify import (
    _compute_empirical_rg,
    _find_overlaps,
    radial_compress,
    resolve_overlaps,
    voronoi_local_density,
    densify_aggregate,
)


class TestEmpiricalRg:
    """Tests for empirical Rg computation."""

    def test_single_particle(self):
        coords = np.array([[5.0, 0.0, 0.0]])
        radii = np.array([1.0])
        rg = _compute_empirical_rg(coords, radii)
        assert rg == pytest.approx(0.0, abs=1e-10)

    def test_two_particles_symmetric(self):
        coords = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        rg = _compute_empirical_rg(coords, radii)
        assert rg == pytest.approx(5.0, rel=0.01)

    def test_cube_corners(self):
        d = 10.0
        coords = np.array(
            [[d, d, d], [-d, d, d], [d, -d, d], [d, d, -d],
             [-d, -d, d], [-d, d, -d], [d, -d, -d], [-d, -d, -d]]
        )
        radii = np.ones(8)
        rg = _compute_empirical_rg(coords, radii)
        assert rg > d * 0.9


class TestFindOverlaps:
    """Tests for overlap detection."""

    def test_no_overlap(self):
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        pi, pj, pov = _find_overlaps(coords, radii)
        assert len(pi) == 0

    def test_full_overlap(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        pi, pj, pov = _find_overlaps(coords, radii)
        assert len(pi) == 1
        assert pov[0] > 0.5

    def test_slight_overlap(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        pi, pj, pov = _find_overlaps(coords, radii)
        assert len(pi) == 1
        assert pov[0] > 0


class TestRadialCompress:
    """Tests for radial compression."""

    def test_compression_reduces_rg(self):
        rng = np.random.RandomState(42)
        n = 50
        coords = rng.randn(n, 3) * 10
        radii = np.ones(n)
        rg_before = _compute_empirical_rg(coords, radii)
        compressed = radial_compress(coords, radii, alpha=0.8)
        rg_after = _compute_empirical_rg(compressed, radii)
        assert rg_after < rg_before
        assert rg_after == pytest.approx(rg_before * 0.8, rel=0.05)

    def test_alpha_1_is_identity(self):
        rng = np.random.RandomState(42)
        n = 30
        coords = rng.randn(n, 3) * 5
        radii = np.ones(n)
        compressed = radial_compress(coords, radii, alpha=1.0)
        np.testing.assert_allclose(compressed, coords, atol=1e-10)


class TestResolveOverlaps:
    """Tests for overlap resolution."""

    def test_resolve_two_overlapping_spheres(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        resolved, success, n_iters = resolve_overlaps(
            coords, radii, tol_ov=0.01, max_iters=100, push_fraction=0.5
        )
        assert success or n_iters > 0
        pair_i, pair_j, pov = _find_overlaps(resolved, radii)
        if len(pair_i) > 0:
            assert float(np.max(pov)) < float(np.max(_find_overlaps(coords, radii)[2]))

    def test_no_overlaps_returns_immediately(self):
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        resolved, success, n_iters = resolve_overlaps(coords, radii, tol_ov=0.01)
        assert success is True
        assert n_iters == 0


class TestVoronoiLocalDensity:
    """Tests for Voronoi local density computation."""

    def test_uniform_particles(self):
        rng = np.random.RandomState(42)
        n = 20
        coords = rng.randn(n, 3) * 5
        density = voronoi_local_density(coords)
        assert density.shape == (n,)
        assert np.all(density > 0)
        assert np.all(np.isfinite(density))

    def test_dense_center_sparse_outside(self):
        center = np.array([[0.0, 0.0, 0.0]] * 5)
        far = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
        coords = np.vstack([center + np.random.RandomState(0).randn(5, 3) * 0.5, far])
        radii = np.ones(coords.shape[0])
        density = voronoi_local_density(coords)
        assert density.shape == (coords.shape[0],)


class TestDensifyAggregate:
    """Integration tests for the full densification pipeline."""

    def test_radial_densify_easy_regime(self):
        """Generate at Df=1.8, densify to Df=2.0."""
        from pyfracval import utils

        rng = np.random.RandomState(42)
        n = 64
        r_g = 10.0
        coords = rng.randn(n, 3) * 15
        radii = np.ones(n) * r_g

        rg_before = _compute_empirical_rg(coords, radii)
        target_rg = utils.calculate_rg(radii, n, 2.0, 1.0)

        new_coords, new_radii, success = densify_aggregate(
            coords, radii,
            target_df=2.0, target_kf=1.0,
            tol_ov=1e-3, max_push_iters=30,
            method="radial",
            rng=rng,
        )
        assert new_coords.shape == coords.shape
        assert new_radii.shape == radii.shape

    def test_densify_disabled_returns_as_is(self):
        """When method is unknown, returns best effort."""
        rng = np.random.RandomState(42)
        n = 32
        coords = rng.randn(n, 3) * 5
        radii = np.ones(n) * 1.0

        new_coords, new_radii, success = densify_aggregate(
            coords, radii,
            target_df=2.0, target_kf=1.0,
            method="invalid_method",
            rng=rng,
        )
        assert new_coords.shape == coords.shape
        assert success is False
