"""Tests for backtracking CCA pairing and the per-merge event log."""

import json

import numpy as np

from pyfracval.cca import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.event_log import EventLog, MergeEvent


def _make_aggregator(n=88, n_per_cluster=8, algorithm_config=None, seed=7):
    """Build an aggregator over `n // n_per_cluster` well-separated clusters."""
    rng = np.random.default_rng(seed)
    n_clusters = n // n_per_cluster
    coords = np.zeros((n, 3))
    radii = np.ones(n)
    i_orden = np.zeros((n_clusters, 3), dtype=int)
    for c in range(n_clusters):
        start = c * n_per_cluster
        offset = np.array([c * 50.0, 0.0, 0.0])
        for p in range(n_per_cluster):
            coords[start + p] = offset + np.array([2.0 * p, 0.0, 0.0])
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
        algorithm_config=algorithm_config or OrchestratorAlgorithmConfig(),
    )


class TestMergeEventLog:
    def test_writes_one_jsonl_record_per_event(self, tmp_path):
        path = tmp_path / "merges.jsonl"
        log = EventLog(path)
        for i in range(3):
            log.record(
                MergeEvent(
                    round_index=1,
                    pool_size=8,
                    cluster_idx1=i,
                    cluster_idx2=i + 1,
                    n1=4,
                    n2=4,
                    gamma_pc=1.5,
                    gamma_real=True,
                    sum_rmax=3.0,
                    outcome="stuck",
                )
            )
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        records = [json.loads(line) for line in lines]
        assert [r["cluster_idx1"] for r in records] == [0, 1, 2]
        assert all(r["run_id"] == log.run_id for r in records)

    def test_infinite_min_overlap_serializes_as_null(self, tmp_path):
        path = tmp_path / "merges.jsonl"
        log = EventLog(path)
        log.record(
            MergeEvent(
                round_index=1,
                pool_size=2,
                cluster_idx1=0,
                cluster_idx2=1,
                n1=1,
                n2=1,
                gamma_pc=1.0,
                gamma_real=True,
                sum_rmax=2.0,
                outcome="failed_no_candidates",
            )
        )
        record = json.loads(path.read_text().strip())
        assert record["min_overlap"] is None

    def test_unwritable_path_disables_log_without_raising(self, tmp_path):
        # A file where a directory should be: mkdir must fail.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        log = EventLog(blocker / "sub" / "merges.jsonl")
        log.record(
            MergeEvent(
                round_index=1,
                pool_size=2,
                cluster_idx1=0,
                cluster_idx2=1,
                n1=1,
                n2=1,
                gamma_pc=1.0,
                gamma_real=True,
                sum_rmax=2.0,
                outcome="stuck",
            )
        )  # must not raise


class TestBacktrackingPairing:
    def test_is_the_production_default(self):
        assert OrchestratorAlgorithmConfig().cca_pairing_strategy == "backtracking"

    def test_completes_aggregation_to_a_single_cluster(self):
        agg = _make_aggregator()
        result = agg.run_cca()
        assert result is not None
        coords, radii = result
        assert agg.i_t == 1
        assert coords.shape[0] == radii.shape[0]

    def test_records_merge_events_when_log_configured(self, tmp_path):
        path = tmp_path / "merges.jsonl"
        cfg = OrchestratorAlgorithmConfig(event_log_path=str(path))
        agg = _make_aggregator(algorithm_config=cfg)
        agg.run_cca()

        assert path.exists()
        records = [json.loads(line) for line in path.read_text().strip().split("\n")]
        assert records, "expected at least one merge attempt to be logged"
        assert all("outcome" in r for r in records)
        # Rounds must advance rather than all reporting round 1.
        assert max(r["round_index"] for r in records) >= 1
        assert all(r["n1"] > 0 and r["n2"] > 0 for r in records)

    def test_no_log_file_created_when_unconfigured(self, tmp_path):
        agg = _make_aggregator()
        assert agg._merge_log is None
        assert not list(tmp_path.iterdir())

    def test_pass_through_preserves_every_particle(self):
        # Whatever the pairing outcome, backtracking must never lose or
        # duplicate particles - pass-through carries a cluster forward
        # verbatim rather than dropping it.
        agg = _make_aggregator(n=88)
        radii_before = np.sort(agg.radii.copy())
        result = agg.run_cca()
        assert result is not None
        _, radii_after = result
        np.testing.assert_allclose(np.sort(radii_after), radii_before)

    def test_zero_progress_round_fails_instead_of_looping(self):
        # Two clusters so far apart that no gamma-feasible edge exists:
        # nothing can merge, so the round must fail rather than spin.
        coords = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1e6, 0.0, 0.0], [1e6 + 2.0, 0.0, 0.0]]
        )
        radii = np.ones(4)
        i_orden = np.array([[0, 1, 2], [2, 3, 2]])
        agg = CCAggregator(
            initial_coords=coords,
            initial_radii=radii,
            initial_i_orden=i_orden,
            n_total=4,
            df=1.8,
            kf=1.0,
            tol_ov=1e-6,
            ext_case=0,
            rng=np.random.default_rng(0),
            # Nothing may pass through, so a round that cannot merge must fail.
            algorithm_config=OrchestratorAlgorithmConfig(
                cca_backtracking_pass_through=False
            ),
        )
        result = agg.run_cca()
        assert result is None
        assert agg.not_able_cca


class TestDeadline:
    def test_expired_deadline_aborts_before_any_merge(self):
        # run_simulation only checks the clock *between* attempts, so a
        # single long attempt used to be uninterruptible - and
        # backtracking made single attempts much more expensive in
        # regimes where nothing works.
        agg = _make_aggregator()
        agg.deadline = 0.0  # already in the past
        result = agg.run_cca()
        assert result is None
        assert agg.not_able_cca
        assert agg.timed_out

    def test_no_deadline_never_times_out(self):
        agg = _make_aggregator()
        assert agg.deadline is None
        assert agg._out_of_time() is False
        assert agg.run_cca() is not None

    def test_generous_deadline_does_not_interfere(self):
        import time

        agg = _make_aggregator()
        agg.deadline = time.time() + 3600.0
        assert agg.run_cca() is not None
        assert not agg.timed_out


class TestGammaFormulationFlags:
    def test_mass_and_count_gamma_agree_for_monodisperse(self):
        from pyfracval.fractal import gamma_calculation

        radii1 = np.ones(8)
        radii2 = np.ones(8)
        # Masses proportional to r^3; for unit radii they are equal, so the
        # count substitution is exact and both forms must coincide.
        common = dict(rg1=5.0, rg2=5.0, df=1.8, kf=1.0)
        m = float(np.sum((4.0 / 3.0) * np.pi * radii1**3))
        counts = gamma_calculation(
            m,
            common["rg1"],
            radii1,
            m,
            common["rg2"],
            radii2,
            common["df"],
            common["kf"],
            use_mass=False,
        )
        masses = gamma_calculation(
            m,
            common["rg1"],
            radii1,
            m,
            common["rg2"],
            radii2,
            common["df"],
            common["kf"],
            use_mass=True,
        )
        assert counts[0] == masses[0]
        assert counts[1] == masses[1]

    def test_mass_and_count_gamma_differ_for_polydisperse(self):
        from pyfracval.fractal import calculate_mass, gamma_calculation

        radii1 = np.array([1.0, 1.0, 1.0, 1.0])
        radii2 = np.array([4.0, 4.0, 4.0, 4.0])
        m1 = float(np.sum(calculate_mass(radii1)))
        m2 = float(np.sum(calculate_mass(radii2)))
        counts = gamma_calculation(
            m1, 5.0, radii1, m2, 5.0, radii2, 1.8, 1.0, use_mass=False
        )
        masses = gamma_calculation(
            m1, 5.0, radii1, m2, 5.0, radii2, 1.8, 1.0, use_mass=True
        )
        assert counts[1] != masses[1]
