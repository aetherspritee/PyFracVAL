"""Tests for soft potential relaxation module."""

import numpy as np
import pytest

from pyfracval.soft_relaxation import (
    compute_forces,
    soft_relaxation,
    soft_sticking,
    _compute_max_overlap_kernel,
)


class TestComputeForces:
    """Tests for force computation."""

    def test_no_overlap_no_force(self):
        """Non-overlapping particles should have zero repulsive force."""
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        cm_target = np.array([2.5, 0.0, 0.0])  # Actual CM (no CM constraint force)

        forces, max_overlap, energy = compute_forces(
            coords, radii, k_repulsion=10.0, cm_target=cm_target, k_gamma=0.0
        )

        # Particles are 5 units apart, sum of radii is 2, so no overlap
        assert max_overlap == 0.0
        assert energy == 0.0
        # Forces should be zero (no overlap, no CM constraint)
        assert np.allclose(forces, 0.0)

    def test_overlap_produces_repulsion(self):
        """Overlapping particles should experience repulsive forces."""
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        cm_target = np.array([0.75, 0.0, 0.0])  # Midpoint

        forces, max_overlap, energy = compute_forces(
            coords, radii, k_repulsion=10.0, cm_target=cm_target, k_gamma=0.0
        )

        # Overlap: sum of radii = 2.0, distance = 1.5, overlap = 0.5
        assert max_overlap > 0.0
        assert energy > 0.0

        # Forces should push particles apart
        # Particle 0 should move left (negative x), particle 1 should move right (positive x)
        assert forces[0, 0] < 0  # Left particle pushed left
        assert forces[1, 0] > 0  # Right particle pushed right

    def test_cm_constraint(self):
        """CM constraint should pull toward target."""
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        cm_target = np.array([10.0, 0.0, 0.0])  # Far from actual CM

        forces, _, _ = compute_forces(
            coords, radii, k_repulsion=0.0, cm_target=cm_target, k_gamma=10.0
        )

        # Both particles should be pulled in +x direction
        assert np.all(forces[:, 0] > 0)


class TestSoftRelaxation:
    """Tests for soft relaxation function."""

    def test_converges_no_overlap(self):
        """Should converge immediately if no overlaps."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
            ]
        )
        radii = np.array([1.0, 1.0, 1.0])
        cm_target = np.mean(coords, axis=0)

        relaxed, success, info = soft_relaxation(
            coords,
            radii,
            cm_target,
            k_repulsion=10.0,
            k_gamma=1.0,
            max_iters=50,
            tol_overlap=1e-4,
        )

        assert success
        assert info["final_max_overlap"] <= 1e-4
        # Coordinates should not change much
        np.testing.assert_allclose(relaxed, coords, atol=0.1)

    def test_resolves_simple_overlap(self):
        """Should resolve simple two-particle overlap."""
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        cm_target = np.array([0.75, 0.0, 0.0])

        relaxed, success, info = soft_relaxation(
            coords,
            radii,
            cm_target,
            k_repulsion=50.0,
            k_gamma=10.0,
            max_iters=100,
            tol_overlap=1e-4,
            learning_rate=0.2,
        )

        # Should reduce overlap significantly
        assert info["final_max_overlap"] < 0.1
        # Distance should increase
        orig_dist = np.linalg.norm(coords[1] - coords[0])
        final_dist = np.linalg.norm(relaxed[1] - relaxed[0])
        assert final_dist > orig_dist

    def test_respects_max_iters(self):
        """Should respect max_iters limit."""
        # Create high overlap that won't converge quickly
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        cm_target = np.array([0.5, 0.0, 0.0])

        relaxed, success, info = soft_relaxation(
            coords,
            radii,
            cm_target,
            max_iters=5,
            tol_overlap=1e-6,  # Very strict tolerance
        )

        assert info["iterations"] <= 5


class TestSoftSticking:
    """Tests for soft sticking of two clusters."""

    def test_basic_sticking(self):
        """Test basic soft sticking of two clusters."""
        # Two small clusters
        coords1 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii1 = np.array([0.5, 0.5])
        coords2 = np.array([[10.0, 0.0, 0.0]])
        radii2 = np.array([0.5])

        cm1 = np.mean(coords1, axis=0)
        cm2 = np.mean(coords2, axis=0)
        gamma_pc = 5.0  # Target separation

        new_coords1, new_coords2, success, info = soft_sticking(
            coords1,
            radii1,
            coords2,
            radii2,
            gamma_pc,
            cm1,
            cm2,
            0,
            0,
            k_repulsion=10.0,
            k_gamma=1.0,
            max_iters=50,
        )

        # Check that clusters were moved
        assert new_coords1.shape == coords1.shape
        assert new_coords2.shape == coords2.shape
        # Gamma error should be reasonable
        assert info["gamma_error"] < 0.5  # Within 50%

    def test_overlap_resolution(self):
        """Test that overlapping clusters get resolved."""
        # Two overlapping clusters
        coords1 = np.array([[0.0, 0.0, 0.0]])
        radii1 = np.array([1.0])
        coords2 = np.array([[1.5, 0.0, 0.0]])  # Overlaps with coords1
        radii2 = np.array([1.0])

        cm1 = coords1[0].copy()
        cm2 = coords2[0].copy()
        gamma_pc = 5.0

        new_coords1, new_coords2, success, info = soft_sticking(
            coords1,
            radii1,
            coords2,
            radii2,
            gamma_pc,
            cm1,
            cm2,
            0,
            0,
            k_repulsion=20.0,
            k_gamma=5.0,
            max_iters=100,
            learning_rate=0.2,
        )

        # Compute final overlap
        final_dist = np.linalg.norm(new_coords2[0] - new_coords1[0])
        r_sum = radii1[0] + radii2[0]

        # Should reduce or eliminate overlap
        assert final_dist >= r_sum * 0.9 or info["final_max_overlap"] < 0.5


class TestMaxOverlapKernel:
    """Tests for the max overlap computation kernel."""

    def test_no_overlap(self):
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        max_ov = _compute_max_overlap_kernel(coords, radii)
        assert max_ov == 0.0

    def test_overlap(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        max_ov = _compute_max_overlap_kernel(coords, radii)
        # Overlap = (2.0 - 1.5) / 1.0 = 0.5
        assert max_ov == pytest.approx(0.5, abs=0.01)

    def test_multiple_overlaps(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0, 1.0])
        max_ov = _compute_max_overlap_kernel(coords, radii)
        # Max overlap should be from first two particles
        assert max_ov == pytest.approx(0.5, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
