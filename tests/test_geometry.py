"""Unit tests for pyfracval.geometry, focused on the ext_case=1 support
added for the previously-broken utils.random_point_sc call
(cca/sticking.py::_cca_sticking_v1)."""

import numpy as np
import pytest

from pyfracval.geometry import random_point_sc, spherical_cap_angle


class TestSphericalCapAngle:
    def test_symmetric_equal_radius_spheres_matches_closed_form(self):
        # Two equal-radius spheres along the x-axis: by symmetry the
        # intersection plane bisects the centers, giving a closed-form
        # phi_cr = arccos(d / (2r)).
        r = 10.0
        d = 12.0
        sphere_1 = np.array([0.0, 0.0, 0.0, r])
        sphere_2 = np.array([d, 0.0, 0.0, r])
        expected = np.arccos(d / (2.0 * r))
        assert spherical_cap_angle(sphere_1, sphere_2) == pytest.approx(
            expected, abs=1e-6
        )

    def test_symmetric_case_is_order_independent(self):
        r = 8.0
        d = 5.0
        sphere_1 = np.array([1.0, 2.0, 3.0, r])
        sphere_2 = np.array([1.0 + d, 2.0, 3.0, r])
        angle_12 = spherical_cap_angle(sphere_1, sphere_2)
        angle_21 = spherical_cap_angle(sphere_2, sphere_1)
        assert angle_12 == pytest.approx(angle_21, abs=1e-6)

    def test_angle_in_valid_range(self):
        sphere_1 = np.array([0.0, 0.0, 0.0, 10.0])
        sphere_2 = np.array([15.0, 0.0, 0.0, 8.0])
        angle = spherical_cap_angle(sphere_1, sphere_2)
        assert 0.0 <= angle <= np.pi


class TestRandomPointSC:
    def _spheres(self, gap_frac=0.3):
        # Two clusters' "shell" spheres: [x, y, z, d_min, d_max].
        spheres_1 = np.array([0.0, 0.0, 0.0, 40.0, 60.0])
        spheres_2 = np.array([90.0, 0.0, 0.0, 35.0, 55.0])
        return spheres_1, spheres_2

    @pytest.mark.parametrize("case", [1, 2, 3])
    def test_point_lies_on_expected_sphere_1_radius(self, case):
        spheres_1, spheres_2 = self._spheres()
        rng = np.random.default_rng(42)
        x, y, z, valid = random_point_sc(case, spheres_1, spheres_2, rng=rng)
        assert valid
        expected_r1 = spheres_1[4] if case in (1, 2) else spheres_1[3]
        dist = np.linalg.norm(np.array([x, y, z]) - spheres_1[:3])
        assert dist == pytest.approx(expected_r1, rel=1e-5)

    def test_invalid_case_returns_false(self):
        spheres_1, spheres_2 = self._spheres()
        rng = np.random.default_rng(1)
        x, y, z, valid = random_point_sc(0, spheres_1, spheres_2, rng=rng)
        assert valid is False
        assert (x, y, z) == (0.0, 0.0, 0.0)

    def test_coincident_centers_returns_invalid(self):
        spheres_1 = np.array([5.0, 5.0, 5.0, 10.0, 20.0])
        spheres_2 = np.array([5.0, 5.0, 5.0, 8.0, 18.0])
        rng = np.random.default_rng(2)
        _, _, _, valid = random_point_sc(2, spheres_1, spheres_2, rng=rng)
        assert valid is False

    def test_samples_vary_and_stay_within_cap_bounds(self):
        # case=2/3 always use phi_cr_min=0, so the polar angle (measured
        # from the center1->center2 axis) of every sample must lie in
        # [0, phi_cr_max], and repeated sampling should actually explore
        # that range rather than collapsing to one point.
        spheres_1, spheres_2 = self._spheres()
        rng = np.random.default_rng(7)
        axis = spheres_2[:3] - spheres_1[:3]
        axis = axis / np.linalg.norm(axis)

        phi_cr_max = spherical_cap_angle(
            np.array([*spheres_1[:3], spheres_1[4]]),
            np.array([*spheres_2[:3], spheres_2[3]]),
        )

        points = []
        for _ in range(200):
            x, y, z, valid = random_point_sc(2, spheres_1, spheres_2, rng=rng)
            assert valid
            points.append((x, y, z))

        angles = []
        for x, y, z in points:
            v = np.array([x, y, z]) - spheres_1[:3]
            v = v / np.linalg.norm(v)
            angle = np.arccos(np.clip(np.dot(v, axis), -1.0, 1.0))
            angles.append(angle)

        assert all(-1e-6 <= a <= phi_cr_max + 1e-6 for a in angles)
        assert max(angles) - min(angles) > 0.1 * phi_cr_max
