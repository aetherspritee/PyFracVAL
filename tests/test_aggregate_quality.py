"""Tests for pyfracval.quality and the overlap-acceptance bug it exposed.

The regression tests here cover the root cause of the catalog overlap
leak (docs/source/catalog_overlap_leak.md): the adaptive-tolerance path
compared an *early-terminated* overlap scan against a threshold ten times
larger than the one the scan early-exited at, so a placement whose first
offending pair was tiny got accepted even when a later pair overlapped by
tens of percent.
"""

import numpy as np

from pyfracval import particle_generation
from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.fractal import (
    compute_empirical_rg,
    compute_empirical_rg_polydisperse,
)
from pyfracval.pca_agg import PCAggregator
from pyfracval.quality import compute_aggregate_quality, max_self_overlap


class TestMaxSelfOverlap:
    def test_no_overlap_for_touching_spheres(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        max_ov, n_pairs = max_self_overlap(coords, radii)
        assert n_pairs == 0
        assert max_ov == 0.0

    def test_detects_known_overlap_fraction(self):
        # Centres 1.0 apart, radii summing to 2.0 -> overlap 0.5 under the
        # (r_sum - d)/r_sum convention tol_ov uses.
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        max_ov, n_pairs = max_self_overlap(coords, radii)
        assert n_pairs == 1
        assert max_ov == 0.5

    def test_reports_the_maximum_not_the_first(self):
        # A tiny overlap between 0-1 and a large one between 0-2. Anything
        # that early-exits on the first offender would report the tiny one.
        coords = np.array([[0.0, 0.0, 0.0], [1.999, 0.0, 0.0], [0.0, 1.0, 0.0]])
        radii = np.array([1.0, 1.0, 1.0])
        max_ov, n_pairs = max_self_overlap(coords, radii)
        assert n_pairs == 2
        assert max_ov == 0.5

    def test_single_particle_has_no_overlap(self):
        max_ov, n_pairs = max_self_overlap(np.zeros((1, 3)), np.array([1.0]))
        assert (max_ov, n_pairs) == (0.0, 0)


class TestPolydisperseRg:
    def test_exceeds_point_mass_rg_by_particle_gyration_term(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        point_mass = compute_empirical_rg(coords, radii)
        polydisperse = compute_empirical_rg_polydisperse(coords, radii)
        # Rg_poly^2 = Rg_pm^2 + (3/5) r^2 for equal radii
        assert polydisperse**2 == np.float64(point_mass**2 + 0.6 * 1.0**2)
        assert polydisperse > point_mass

    def test_empty_aggregate_is_zero(self):
        assert compute_empirical_rg_polydisperse(np.zeros((0, 3)), np.array([])) == 0.0


class TestComputeAggregateQuality:
    def test_clean_aggregate_passes(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        q = compute_aggregate_quality(coords, radii, df=1.8, kf=1.0, tol_ov=1e-6)
        assert q["overlap_ok"] is True
        assert q["n_overlapping_pairs"] == 0
        assert q["n_particles"] == 2

    def test_overlapping_aggregate_is_flagged(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        q = compute_aggregate_quality(coords, radii, df=1.8, kf=1.0, tol_ov=1e-6)
        assert q["overlap_ok"] is False
        assert q["max_residual_overlap"] == 0.5


class TestPcaProducesOverlapFreeSubclusters:
    """Regression: PCA must never report success for a subcluster whose
    particles interpenetrate.

    Seed 8 at these parameters reproduced the original bug directly - it
    returned a 12-particle subcluster containing a pair overlapping by
    0.43 while the acceptance check believed the worst overlap was 2.6e-6.
    """

    PARAMS = dict(df=2.25, kf=0.95, tol_ov=1e-6, rp_gstd=1.9, rp_g=100.0, n=12)

    def _run(self, seed):
        rng = np.random.default_rng(seed)
        radii = particle_generation.lognormal_pp_radii(
            self.PARAMS["rp_gstd"], self.PARAMS["rp_g"], self.PARAMS["n"], rng=rng
        )
        runner = PCAggregator(
            radii,
            self.PARAMS["df"],
            self.PARAMS["kf"],
            self.PARAMS["tol_ov"],
            rng=rng,
            algorithm_config=OrchestratorAlgorithmConfig(),
        )
        result = runner.run()
        if result is None or runner.not_able_pca:
            return None
        return result

    def test_known_bad_seeds_are_now_clean(self):
        for seed in (8, 116, 124):
            result = self._run(seed)
            if result is None:
                continue  # rejecting outright is also a correct outcome
            max_ov, n_pairs = max_self_overlap(result[:, :3], result[:, 3])
            assert max_ov < 1e-9, (
                f"seed {seed}: PCA returned success with {n_pairs} overlapping "
                f"pairs, worst {max_ov:.3e}"
            )

    def test_no_successful_subcluster_overlaps_across_seed_sweep(self):
        for seed in range(60):
            result = self._run(seed)
            if result is None:
                continue
            max_ov, _ = max_self_overlap(result[:, :3], result[:, 3])
            assert max_ov < 1e-9, f"seed {seed}: worst overlap {max_ov:.3e}"
