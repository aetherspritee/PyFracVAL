"""Unit tests for pyfracval.gap_scaling -- porting YASF-new's
cluster_gap_factor semantics (target gap in mean radii -> position scale)."""

import numpy as np
import pytest

from pyfracval.gap_scaling import _validate_no_overlaps, compute_gap_scale


def _touching_pair(r=1.0, gap_from_touching=0.0):
    """Two spheres of radius r, centered symmetrically about the origin,
    separated by exactly 2r (+ gap_from_touching)."""
    d = 2 * r + gap_from_touching
    coords = np.array([[-d / 2, 0.0, 0.0], [d / 2, 0.0, 0.0]])
    radii = np.array([r, r])
    return coords, radii


def test_none_gap_factor_returns_1_no_op():
    coords, radii = _touching_pair()
    assert compute_gap_scale(coords, radii, gap_factor=None) == 1.0


def test_zero_gap_factor_returns_1_no_op():
    coords, radii = _touching_pair()
    assert compute_gap_scale(coords, radii, gap_factor=0) == 1.0


def test_negative_gap_factor_raises():
    coords, radii = _touching_pair()
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_gap_scale(coords, radii, gap_factor=-1.0)


def test_invalid_mode_raises():
    coords, radii = _touching_pair()
    with pytest.raises(ValueError, match="gap_mode"):
        compute_gap_scale(coords, radii, gap_factor=1.0, mode="bogus")


@pytest.mark.parametrize(
    "gamma,expected_scale",
    [(2.0, 2.0), (4.0, 3.0), (0.5, 1.25)],
)
def test_average_mode_matches_yasf_formula(gamma, expected_scale):
    # s = max(1, 1 + gamma/2) -- YASF's own closed-form estimate, doesn't
    # depend on actual geometry, only on gamma itself.
    coords, radii = _touching_pair(r=1.0)
    scale = compute_gap_scale(coords, radii, gap_factor=gamma, mode="average")
    assert scale == pytest.approx(expected_scale)


def test_strict_mode_achieves_exact_target_gap_for_closest_pair():
    r = 1.0
    coords, radii = _touching_pair(r=r)  # centers at distance 2r (touching)
    gamma = 2.0
    scale = compute_gap_scale(coords, radii, gap_factor=gamma, mode="strict")

    scaled_d = np.linalg.norm((coords[0] - coords[1]) * scale)
    achieved_gap = scaled_d - 2 * r
    target_gap = gamma * r  # r_mean == r here
    assert achieved_gap == pytest.approx(target_gap, rel=1e-9)


def test_strict_mode_no_close_pairs_needs_no_scaling():
    # Two spheres already far apart -- well beyond any plausible target gap
    # -- strict mode should not scale at all.
    coords = np.array([[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    scale = compute_gap_scale(coords, radii, gap_factor=2.0, mode="strict")
    assert scale == 1.0


def test_strict_mode_duplicate_positions_raise():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="Duplicate particle positions"):
        compute_gap_scale(coords, radii, gap_factor=1.0, mode="strict")


def test_scale_is_always_at_least_one():
    # A tiny gap_factor shouldn't ever produce a shrinking scale.
    coords, radii = _touching_pair()
    assert compute_gap_scale(coords, radii, gap_factor=1e-9, mode="average") >= 1.0


def test_validate_no_overlaps_raises_on_fully_overlapping_particles():
    # Direct test of the safety-net helper: a global scale >= 1 applied to
    # an already-valid (non-overlapping) cluster can mathematically never
    # introduce a new overlap (uniform dilation scales every pairwise
    # distance by the same factor) -- so this failure path is realistically
    # unreachable through compute_gap_scale() on real generated data. Test
    # it directly instead, with a pathological input it exists to catch.
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="failed to eliminate all overlaps"):
        _validate_no_overlaps(coords, radii, gap_factor=1.0, mode="average")


def test_validate_no_overlaps_passes_for_valid_geometry():
    coords, radii = _touching_pair()
    _validate_no_overlaps(coords, radii, gap_factor=1.0, mode="average")  # no raise
