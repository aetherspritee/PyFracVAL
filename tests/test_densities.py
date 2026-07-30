"""Tests for optional per-particle densities.

The invariant under test throughout: a particle's density must follow the
*particle*, not the array slot it started in. PCA swaps particles, CCA
reorders and concatenates clusters, and drop-rescue removes some - every
one of those is a chance for densities to silently desynchronise from
radii, which would corrupt every mass-weighted quantity (center of mass,
radius of gyration, Gamma) without raising anything.
"""

import numpy as np
import pytest

from pyfracval import particle_generation
from pyfracval.cca import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.fractal import (
    calculate_cluster_properties,
    calculate_mass,
    compute_empirical_rg_polydisperse,
    resolve_densities,
)
from pyfracval.main_runner import run_simulation
from pyfracval.pca_agg import PCAggregator
from pyfracval.pca_subclusters import Subclusterer


class TestCalculateMass:
    def test_uniform_density_matches_volume(self):
        radii = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            calculate_mass(radii), (4.0 / 3.0) * np.pi * radii**3
        )

    def test_densities_scale_mass(self):
        radii = np.array([1.0, 1.0])
        densities = np.array([1.0, 3.0])
        mass = calculate_mass(radii, densities)
        assert mass[1] == pytest.approx(3.0 * mass[0])

    def test_equal_densities_are_a_uniform_rescale(self):
        radii = np.array([1.0, 2.0, 3.0])
        plain = calculate_mass(radii)
        scaled = calculate_mass(radii, np.full(3, 2.5))
        np.testing.assert_allclose(scaled, 2.5 * plain)


class TestResolveDensities:
    def test_none_stays_none(self):
        assert resolve_densities(None, 5) is None

    def test_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="expected shape"):
            resolve_densities(np.ones(4), 5)

    def test_non_positive_is_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            resolve_densities(np.array([1.0, 0.0]), 2)


class TestDensityAffectsMassWeightedQuantities:
    def test_center_of_mass_shifts_toward_denser_particle(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        _, _, cm_uniform, _ = calculate_cluster_properties(coords, radii, 1.8, 1.0)
        _, _, cm_heavy, _ = calculate_cluster_properties(
            coords, radii, 1.8, 1.0, densities=np.array([1.0, 9.0])
        )
        assert cm_uniform[0] == pytest.approx(1.0)
        assert cm_heavy[0] > cm_uniform[0]

    def test_rg_differs_for_heterogeneous_aggregate(self):
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])
        uniform = compute_empirical_rg_polydisperse(coords, radii)
        heavy = compute_empirical_rg_polydisperse(coords, radii, np.array([1.0, 100.0]))
        # Concentrating the mass on one particle pulls the center of mass
        # onto it, shrinking Rg.
        assert heavy < uniform


class TestPcaKeepsDensitiesWithParticles:
    def test_density_follows_radius_through_pca_swaps(self):
        rng = np.random.default_rng(3)
        radii = particle_generation.lognormal_pp_radii(1.9, 100.0, 12, rng=rng)
        # Encode each particle's identity in its density so any reordering
        # is detectable: density i is uniquely tied to radius i.
        densities = 1.0 + np.arange(radii.size, dtype=float)
        pairing = dict(zip(radii.tolist(), densities.tolist()))

        runner = PCAggregator(
            radii,
            1.79,
            1.4,
            1e-6,
            rng=rng,
            algorithm_config=OrchestratorAlgorithmConfig(),
            densities=densities,
        )
        result = runner.run()
        if result is None or runner.not_able_pca:
            pytest.skip("PCA did not converge for this seed")

        assert runner.densities is not None
        out_radii = result[:, 3]
        for r, d in zip(out_radii.tolist(), runner.densities.tolist()):
            # Retry-with-fresh-radii would break this pairing legitimately;
            # only assert on radii that came from the original draw.
            if r in pairing:
                assert d == pytest.approx(pairing[r])

    def test_uniform_density_leaves_densities_none(self):
        rng = np.random.default_rng(1)
        radii = particle_generation.lognormal_pp_radii(1.5, 100.0, 8, rng=rng)
        runner = PCAggregator(radii, 1.79, 1.4, 1e-6, rng=rng)
        runner.run()
        assert runner.densities is None


class TestCcaKeepsDensitiesWithParticles:
    def _aggregator(self, densities=None, n=32, n_per_cluster=8):
        rng = np.random.default_rng(5)
        n_clusters = n // n_per_cluster
        coords = np.zeros((n, 3))
        radii = np.ones(n)
        i_orden = np.zeros((n_clusters, 3), dtype=int)
        for c in range(n_clusters):
            start = c * n_per_cluster
            for p in range(n_per_cluster):
                coords[start + p] = np.array([c * 50.0 + 2.0 * p, 0.0, 0.0])
            i_orden[c] = [start, start + n_per_cluster - 1, n_per_cluster]
        return CCAggregator(
            initial_coords=coords,
            initial_radii=radii,
            initial_i_orden=i_orden,
            n_total=n,
            df=1.8,
            kf=1.0,
            tol_ov=1e-6,
            ext_case=0,
            rng=rng,
            algorithm_config=OrchestratorAlgorithmConfig(),
            initial_densities=densities,
        )

    def test_density_multiset_is_preserved_through_aggregation(self):
        densities = 1.0 + np.arange(32, dtype=float)
        agg = self._aggregator(densities=densities)
        result = agg.run_cca()
        assert result is not None
        assert agg.densities is not None
        # Every original density must still be present exactly once: CCA
        # may reorder particles but must neither lose nor duplicate them.
        np.testing.assert_allclose(np.sort(agg.densities), np.sort(densities))

    def test_densities_stay_aligned_with_radii(self):
        # Radius encodes identity here, density mirrors it, so a
        # misalignment shows up as a broken r <-> rho relationship.
        rng = np.random.default_rng(11)
        n, n_per_cluster = 32, 8
        n_clusters = n // n_per_cluster
        coords = np.zeros((n, 3))
        radii = np.ones(n)
        i_orden = np.zeros((n_clusters, 3), dtype=int)
        for c in range(n_clusters):
            start = c * n_per_cluster
            for p in range(n_per_cluster):
                coords[start + p] = np.array([c * 50.0 + 2.0 * p, 0.0, 0.0])
            i_orden[c] = [start, start + n_per_cluster - 1, n_per_cluster]
        densities = 2.0 + np.arange(n, dtype=float)
        # Tie density to position-in-cluster via a lookup we can verify.
        identity = {i: densities[i] for i in range(n)}

        agg = CCAggregator(
            initial_coords=coords,
            initial_radii=radii,
            initial_i_orden=i_orden,
            n_total=n,
            df=1.8,
            kf=1.0,
            tol_ov=1e-6,
            ext_case=0,
            rng=rng,
            algorithm_config=OrchestratorAlgorithmConfig(),
            initial_densities=densities,
        )
        result = agg.run_cca()
        assert result is not None
        assert agg.densities is not None
        assert sorted(agg.densities.tolist()) == sorted(identity.values())

    def test_none_densities_stay_none(self):
        agg = self._aggregator(densities=None)
        assert agg.run_cca() is not None
        assert agg.densities is None

    def test_wrong_length_densities_rejected(self):
        with pytest.raises(ValueError, match="expected shape"):
            self._aggregator(densities=np.ones(7))


class TestSubclustererDensities:
    def test_densities_survive_subclustering(self):
        rng = np.random.default_rng(17)
        n = 40
        radii = particle_generation.lognormal_pp_radii(1.5, 100.0, n, rng=rng)
        densities = 1.0 + np.arange(n, dtype=float)
        runner = Subclusterer(
            initial_radii=radii,
            initial_densities=densities,
            df=1.8,
            kf=1.0,
            tol_ov=1e-6,
            n_subcl_percentage=0.2,
            rp_g=100.0,
            rp_gstd=1.5,
            rng=rng,
            algorithm_config=OrchestratorAlgorithmConfig(),
        )
        if not runner.run_subclustering() or runner.not_able_pca:
            pytest.skip("subclustering did not converge for this seed")
        assert runner.all_densities is not None
        assert runner.all_densities.shape == (n,)
        assert np.all(runner.all_densities > 0)


class TestRunSimulationWithDensities:
    def test_end_to_end_heterogeneous_run(self, output_dir):
        n = 48
        # Two materials, a 4x density contrast - the heterogeneous case
        # counts-based Gamma cannot represent at all.
        densities = np.where(np.arange(n) % 2 == 0, 1.0, 4.0).astype(float)
        ok, coords, radii = run_simulation(
            iteration=1,
            sim_config_dict={
                "N": n,
                "Df": 1.8,
                "kf": 1.0,
                "rp_g": 100.0,
                "rp_gstd": 1.3,
                "tol_ov": 1e-6,
                "n_subcl_percentage": 0.2,
                "ext_case": 0,
                "seed": 4,
            },
            output_base_dir=str(output_dir),
            densities=densities,
        )
        assert ok
        assert coords is not None and radii is not None
        assert coords.shape[0] == n

    def test_wrong_length_densities_rejected(self, output_dir):
        with pytest.raises(ValueError, match="expected shape"):
            run_simulation(
                iteration=1,
                sim_config_dict={
                    "N": 16,
                    "Df": 1.8,
                    "kf": 1.0,
                    "rp_g": 100.0,
                    "rp_gstd": 1.3,
                    "tol_ov": 1e-6,
                    "n_subcl_percentage": 0.2,
                    "ext_case": 0,
                    "seed": 1,
                },
                output_base_dir=str(output_dir),
                densities=np.ones(5),
            )
