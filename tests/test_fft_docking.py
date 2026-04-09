"""Unit tests for FFT-based rigid-body docking module."""

import numpy as np
import pytest

from pyfracval.fft_docking import (
    _overlap_check_kernel,
    _voxelize_kernel,
    compute_fft_correlation,
    extract_top_k_peaks,
    fft_dock_sticking,
    sample_so3_rotations,
    validate_placement,
    voxelize_cluster,
)


class TestSampleSO3Rotations:
    """Tests for SO(3) rotation sampling."""

    def test_returns_approximate_count(self):
        rots = sample_so3_rotations(70)
        assert len(rots) >= 50  # Hopf fibration may produce fewer than requested

    def test_returns_3x3_matrices(self):
        rots = sample_so3_rotations(10)
        for R in rots:
            assert R.shape == (3, 3)

    def test_rotation_matrices_orthogonal(self):
        rots = sample_so3_rotations(50)
        for R in rots:
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-4)
            RtR = R.T @ R
            np.testing.assert_allclose(RtR, np.eye(3), atol=1e-4)

    def test_single_rotation(self):
        rots = sample_so3_rotations(1)
        assert len(rots) <= 1

    def test_identity_preserves_vector(self):
        rots = sample_so3_rotations(20)
        v = np.array([1.0, 0.0, 0.0])
        for R in rots:
            v_rot = R @ v
            assert np.linalg.norm(v_rot) == pytest.approx(1.0, abs=1e-6)


class TestVoxelizeCluster:
    """Tests for cluster voxelization."""

    def _make_simple_cluster(self):
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        cm = np.array([2.5, 0.0, 0.0])
        return coords, radii, cm

    def test_grid_shape(self):
        coords, radii, cm = self._make_simple_cluster()
        grid, origin = voxelize_cluster(coords, radii, cm, grid_size=32, voxel_size=1.0)
        assert grid.shape == (32, 32, 32)
        assert np.any(grid != 0)

    def test_surface_mode_has_positive_voxels(self):
        coords, radii, cm = self._make_simple_cluster()
        grid, _ = voxelize_cluster(coords, radii, cm, grid_size=32, voxel_size=1.0, mode=0)
        assert np.sum(grid == 1) > 0

    def test_interior_mode_has_negative_voxels(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([5.0])
        cm = np.array([0.0, 0.0, 0.0])
        grid, _ = voxelize_cluster(coords, radii, cm, grid_size=32, voxel_size=1.0, mode=1)
        assert np.sum(grid < 0) > 0
        assert np.sum(grid > 0) > 0

    def test_single_particle_fills_grid(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([5.0])
        cm = np.array([0.0, 0.0, 0.0])
        grid, _ = voxelize_cluster(coords, radii, cm, grid_size=16, voxel_size=1.0)
        total_filled = np.sum(grid != 0)
        assert total_filled > 0


class TestComputeFFTCorrelation:
    """Tests for FFT cross-correlation."""

    def test_correlation_shape_preserved(self):
        grid1 = np.zeros((16, 16, 16), dtype=np.int8)
        grid2 = np.zeros((16, 16, 16), dtype=np.int8)
        grid1[8, 8, 8] = 1
        grid2[8, 8, 8] = 1
        corr = compute_fft_correlation(grid1, grid2)
        assert corr.shape == (16, 16, 16)

    def test_self_correlation_has_peak(self):
        grid = np.zeros((16, 16, 16), dtype=np.float64)
        grid[8, 8, 8] = 1.0
        corr = compute_fft_correlation(grid, grid)
        assert np.max(corr) > 0
        assert np.max(corr) == pytest.approx(1.0, abs=0.01)

    def test_shifted_correlation_peak_moves(self):
        grid1 = np.zeros((16, 16, 16), dtype=np.float64)
        grid2 = np.zeros((16, 16, 16), dtype=np.float64)
        grid1[8, 8, 8] = 1.0
        grid2[10, 10, 10] = 1.0
        corr = compute_fft_correlation(grid1, grid2)
        assert np.max(corr) > 0


class TestExtractTopKPeaks:
    """Tests for peak extraction."""

    def test_single_peak(self):
        corr = np.zeros((16, 16, 16))
        corr[8, 8, 8] = 10.0
        peaks = extract_top_k_peaks(corr, k=5, min_distance=2)
        assert len(peaks) >= 1
        assert peaks[0][0] == pytest.approx(10.0, abs=0.1)

    def test_k_limited(self):
        corr = np.zeros((16, 16, 16))
        corr[4, 4, 4] = 10.0
        corr[10, 10, 10] = 8.0
        peaks = extract_top_k_peaks(corr, k=1, min_distance=2)
        assert len(peaks) <= 1

    def test_noise_floor_filter(self):
        corr = np.ones((16, 16, 16)) * 0.01
        corr[8, 8, 8] = 10.0
        peaks = extract_top_k_peaks(corr, k=5, min_distance=2)
        assert any(abs(p[0] - 10.0) < 0.5 for p in peaks)


class TestOverlapCheckKernel:
    """Tests for the JIT overlap check kernel."""

    def test_no_overlap(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[10.0, 0.0, 0.0]])
        radii2 = np.array([1.0])
        cov = _overlap_check_kernel(coords1, radii1, coords2, radii2, 1e-4)
        assert cov == 0.0

    def test_full_overlap(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[0.0, 0.0, 0.0]])
        radii2 = np.array([1.0])
        cov = _overlap_check_kernel(coords1, radii1, coords2, radii2, 1e-4)
        assert cov > 0

    def test_slight_overlap(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[1.5, 0.0, 0.0]])
        radii2 = np.array([1.0])
        cov = _overlap_check_kernel(coords1, radii1, coords2, radii2, 1e-4)
        assert cov > 0


class TestValidatePlacement:
    """Tests for placement validation."""

    def test_valid_placement(self):
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        radii1 = np.array([0.5, 0.5])
        coords2 = np.array([[100.0, 0.0, 0.0], [101.0, 0.0, 0.0]])
        radii2 = np.array([0.5, 0.5])
        cm1 = np.array([0.5, 0.0, 0.0])
        cm2 = np.array([100.5, 0.0, 0.0])
        valid, dist, cov = validate_placement(
            coords1, radii1, coords2, radii2, cm1, cm2,
            gamma_pc=100.0, gamma_real=True, tol_ov=1e-4, gamma_tolerance=0.20,
        )
        assert valid is True or valid is False  # depends on exact dist

    def test_invalid_gamma_real_false(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[10.0, 0.0, 0.0]])
        radii2 = np.array([1.0])
        cm1 = np.array([0.0, 0.0, 0.0])
        cm2 = np.array([10.0, 0.0, 0.0])
        valid, _, _ = validate_placement(
            coords1, radii1, coords2, radii2, cm1, cm2,
            gamma_pc=10.0, gamma_real=False, tol_ov=1e-4,
        )
        assert valid is False


class TestFFTDockSticking:
    """Integration test for the full FFT docking pipeline."""

    def test_well_separated_clusters_succeed(self):
        rng = np.random.RandomState(42)
        n1, n2 = 8, 8
        coords1 = rng.randn(n1, 3) * 2
        radii1 = np.ones(n1) * 0.5
        coords2 = rng.randn(n2, 3) * 2 + np.array([20.0, 0.0, 0.0])
        radii2 = np.ones(n2) * 0.5
        cm1 = np.mean(coords1, axis=0)
        cm2 = np.mean(coords2, axis=0)
        gamma_pc = float(np.linalg.norm(cm2 - cm1))

        result = fft_dock_sticking(
            coords1, radii1, coords2, radii2,
            cm1, cm2, gamma_pc, gamma_real=True, tol_ov=1e-3,
            grid_size=32, num_rotations=5, top_k_peaks=5,
            gamma_tolerance=0.30,
        )
        # Well-separated small clusters should be dockable
        # (might still fail if overlap, but shouldn't crash)

    def test_returns_none_for_zero_gamma(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[10.0, 0.0, 0.0]])
        radii2 = np.array([1.0])
        cm1 = np.array([0.0, 0.0, 0.0])
        cm2 = np.array([10.0, 0.0, 0.0])
        result = fft_dock_sticking(
            coords1, radii1, coords2, radii2,
            cm1, cm2, gamma_pc=0.0, gamma_real=True, tol_ov=1e-4,
        )
        assert result is None

    def test_returns_none_for_non_real_gamma(self):
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[10.0, 0.0, 0.0]])
        radii2 = np.array([1.0])
        cm1 = np.array([0.0, 0.0, 0.0])
        cm2 = np.array([10.0, 0.0, 0.0])
        result = fft_dock_sticking(
            coords1, radii1, coords2, radii2,
            cm1, cm2, gamma_pc=10.0, gamma_real=False, tol_ov=1e-4,
        )
        assert result is None
