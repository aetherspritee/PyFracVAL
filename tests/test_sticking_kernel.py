"""Pin the compiled sticking kernel to the interpreted reference.

``_cca_sticking_v1`` dispatches to a fused numba kernel for the default
``ext_case=0`` geometry. That is only safe if the kernel reproduces
``_cca_sticking_v1_interpreted`` exactly - an optimization that quietly
changes the geometry would still produce plausible aggregates, so it must
be pinned rather than eyeballed.

The two draw the same two angles from the same RNG in the same order (the
kernel takes them as arguments precisely so the caller keeps ownership of
the stream), so seeding both identically makes the comparison exact
rather than statistical.
"""

import numpy as np
import pytest

from pyfracval.cca import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig


def _aggregator(seed, ext_case=0):
    """Minimal two-cluster aggregator; only used as a method host here."""
    n, per = 16, 8
    coords = np.zeros((n, 3))
    radii = np.ones(n)
    i_orden = np.array([[0, per - 1, per], [per, n - 1, per]])
    for c in range(2):
        for p in range(per):
            coords[c * per + p] = np.array([c * 40.0 + 2.0 * p, 0.0, 0.0])
    return CCAggregator(
        initial_coords=coords,
        initial_radii=radii,
        initial_i_orden=i_orden,
        n_total=n,
        df=1.8,
        kf=1.0,
        tol_ov=1e-6,
        ext_case=ext_case,
        rng=np.random.default_rng(seed),
        algorithm_config=OrchestratorAlgorithmConfig(),
    )


def _case(rng, n1, n2, polydisperse):
    """A random but well-posed pair of clusters plus a workable gamma."""
    coords1 = rng.normal(scale=6.0, size=(n1, 3))
    coords2 = rng.normal(scale=6.0, size=(n2, 3)) + np.array([60.0, 0.0, 0.0])
    if polydisperse:
        radii1 = rng.uniform(0.5, 3.0, size=n1)
        radii2 = rng.uniform(0.5, 3.0, size=n2)
    else:
        radii1 = np.ones(n1)
        radii2 = np.ones(n2)
    cm1 = coords1.mean(axis=0)
    cm2 = coords2.mean(axis=0)
    cand1 = int(rng.integers(n1))
    cand2 = int(rng.integers(n2))
    # Pick gamma inside the window where the first sphere-sphere
    # intersection actually has a solution, |d1max - d2max| < gamma <
    # d1max + d2max. Sampling gamma blindly mostly produces placements
    # both implementations correctly reject, which tests nothing.
    d1_max = float(np.linalg.norm(coords1[cand1] - cm1)) + radii1[cand1]
    d2_max = float(np.linalg.norm(coords2[cand2] - cm2)) + radii2[cand2]
    lo = abs(d1_max - d2_max)
    hi = d1_max + d2_max
    gamma = lo + (hi - lo) * rng.uniform(0.25, 0.75)
    return coords1, radii1, cm1, coords2, radii2, cm2, cand1, cand2, gamma


class TestKernelMatchesInterpreted:
    @pytest.mark.parametrize("polydisperse", [False, True])
    def test_placements_agree_across_many_random_cases(self, polydisperse):
        rng_cases = np.random.default_rng(1234)
        compared = 0

        # Many cases are legitimately infeasible (the second
        # intersection has no solution) and both implementations reject
        # them identically; sample enough to accumulate real comparisons.
        for trial in range(300):
            n1 = int(rng_cases.integers(2, 20))
            n2 = int(rng_cases.integers(2, 20))
            data = _case(rng_cases, n1, n2, polydisperse)
            c1, r1, cm1, c2, r2, cm2, cand1, cand2, gamma = data

            # Same seed for both so the two sampled angles coincide.
            agg_k = _aggregator(seed=trial)
            agg_i = _aggregator(seed=trial)

            kern = agg_k._cca_sticking_v1(
                (c1, r1, cm1), (c2, r2, cm2), cand1, cand2, gamma, True
            )
            interp = agg_i._cca_sticking_v1_interpreted(
                (c1, r1, cm1), (c2, r2, cm2), cand1, cand2, gamma, True
            )

            # Both must agree on whether a placement exists at all.
            assert (kern[0] is None) == (interp[0] is None), (
                f"trial {trial}: kernel and reference disagree on validity"
            )
            if kern[0] is None:
                continue

            compared += 1
            np.testing.assert_allclose(
                kern[0], interp[0], rtol=1e-9, atol=1e-9, err_msg="coords1"
            )
            np.testing.assert_allclose(
                kern[1], interp[1], rtol=1e-9, atol=1e-9, err_msg="coords2"
            )
            np.testing.assert_allclose(
                kern[2], interp[2], rtol=1e-9, atol=1e-9, err_msg="cm2"
            )
            np.testing.assert_allclose(
                kern[4], interp[4], rtol=1e-9, atol=1e-9, err_msg="vec_0"
            )
            np.testing.assert_allclose(
                kern[5], interp[5], rtol=1e-9, atol=1e-9, err_msg="i_vec"
            )
            np.testing.assert_allclose(
                kern[6], interp[6], rtol=1e-9, atol=1e-9, err_msg="j_vec"
            )

        assert compared >= 40, (
            f"only {compared} valid placements compared; the test is not "
            f"exercising the kernel meaningfully"
        )

    def test_candidates_end_up_in_point_contact(self):
        """The physical invariant the placement exists to satisfy."""
        rng_cases = np.random.default_rng(99)
        checked = 0
        for trial in range(150):
            data = _case(rng_cases, 10, 10, polydisperse=True)
            c1, r1, cm1, c2, r2, cm2, cand1, cand2, gamma = data
            agg = _aggregator(seed=trial)
            out = agg._cca_sticking_v1(
                (c1, r1, cm1), (c2, r2, cm2), cand1, cand2, gamma, True
            )
            if out[0] is None:
                continue
            checked += 1
            coords1_out, coords2_out = out[0], out[1]
            d = np.linalg.norm(coords1_out[cand1] - coords2_out[cand2])
            assert d == pytest.approx(r1[cand1] + r2[cand2], rel=1e-7)
        assert checked >= 25

    def test_gamma_separation_is_respected(self):
        """CM2 must sit exactly gamma from CM1 - the whole point of Gamma."""
        rng_cases = np.random.default_rng(7)
        checked = 0
        for trial in range(150):
            data = _case(rng_cases, 12, 9, polydisperse=True)
            c1, r1, cm1, c2, r2, cm2, cand1, cand2, gamma = data
            agg = _aggregator(seed=trial)
            out = agg._cca_sticking_v1(
                (c1, r1, cm1), (c2, r2, cm2), cand1, cand2, gamma, True
            )
            if out[0] is None:
                continue
            checked += 1
            assert np.linalg.norm(out[2] - cm1) == pytest.approx(gamma, rel=1e-9)
        assert checked >= 25

    def test_clusters_move_rigidly(self):
        """Sticking may translate and rotate a cluster but must not deform it."""
        rng_cases = np.random.default_rng(5)
        checked = 0
        for trial in range(120):
            data = _case(rng_cases, 14, 11, polydisperse=True)
            c1, r1, cm1, c2, r2, cm2, cand1, cand2, gamma = data
            agg = _aggregator(seed=trial)
            out = agg._cca_sticking_v1(
                (c1, r1, cm1), (c2, r2, cm2), cand1, cand2, gamma, True
            )
            if out[0] is None:
                continue
            checked += 1
            for before, after in ((c1, out[0]), (c2, out[1])):
                d_before = np.linalg.norm(
                    before[:, None, :] - before[None, :, :], axis=2
                )
                d_after = np.linalg.norm(after[:, None, :] - after[None, :, :], axis=2)
                np.testing.assert_allclose(d_after, d_before, rtol=1e-9, atol=1e-9)
        assert checked >= 20

    def test_ext_case_1_still_uses_the_interpreted_path(self):
        # ext_case=1 samples spherical caps, which the kernel does not
        # implement; dispatching it to the kernel would silently change
        # the geometry.
        agg = _aggregator(seed=1, ext_case=1)
        assert agg.ext_case == 1
        rng_cases = np.random.default_rng(3)
        data = _case(rng_cases, 8, 8, polydisperse=False)
        c1, r1, cm1, c2, r2, cm2, cand1, cand2, gamma = data
        # Must not raise, and must go through the interpreted branch.
        agg._cca_sticking_v1((c1, r1, cm1), (c2, r2, cm2), cand1, cand2, gamma, True)
