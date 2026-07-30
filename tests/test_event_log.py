"""Tests for the structured generation event log.

The log exists to answer, over a pooled sweep: where does generation
fail, why, and how badly do particles overlap when it does. These tests
pin the properties that answer depends on - that every record carries the
simulation context, that the three record kinds are distinguishable, that
failure attribution is specific rather than "somewhere in the pipeline",
and that overlap is reported under both denominators rather than one
ambiguous "overlap fraction".
"""

import json

import numpy as np
import pytest

from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.event_log import EventLog, MergeEvent, PcaFailureEvent, RunEvent
from pyfracval.main_runner import run_simulation


def _merge(**kw):
    base = dict(
        round_index=1,
        pool_size=4,
        cluster_idx1=0,
        cluster_idx2=1,
        n1=8,
        n2=8,
        gamma_pc=3.0,
        gamma_real=True,
        sum_rmax=6.0,
        outcome="stuck",
    )
    base.update(kw)
    return MergeEvent(**base)


class TestEventLogMechanics:
    def test_each_kind_is_labelled(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        log.record(_merge())
        log.record(
            PcaFailureEvent(
                subcluster_index=0,
                subcluster_size=12,
                particle_index=3,
                reason="no_candidates",
            )
        )
        log.record(RunEvent(outcome="success"))
        kinds = [
            json.loads(x)["kind"]
            for x in (tmp_path / "e.jsonl").read_text().splitlines()
        ]
        assert kinds == ["merge", "pca_failure", "run"]

    def test_context_is_stamped_on_every_record(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl", context={"Df": 2.1, "N": 128})
        log.record(_merge())
        log.record(RunEvent(outcome="failed"))
        for line in (tmp_path / "e.jsonl").read_text().splitlines():
            rec = json.loads(line)
            assert rec["Df"] == 2.1
            assert rec["N"] == 128

    def test_context_never_shadows_a_records_own_field(self, tmp_path):
        # A context key colliding with a record field must not overwrite
        # the record's real value, or the log silently lies.
        log = EventLog(tmp_path / "e.jsonl", context={"n1": 999})
        log.record(_merge(n1=7))
        assert json.loads((tmp_path / "e.jsonl").read_text())["n1"] == 7

    def test_set_context_updates_later_records(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        log.set_context(seed=5)
        log.record(_merge())
        assert json.loads((tmp_path / "e.jsonl").read_text())["seed"] == 5

    def test_non_finite_floats_serialize_as_null(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        log.record(_merge())  # min_overlap defaults to inf
        assert json.loads((tmp_path / "e.jsonl").read_text())["min_overlap"] is None

    def test_unknown_event_type_is_ignored_not_raised(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        log.record(object())
        assert not (tmp_path / "e.jsonl").exists()

    def test_unwritable_path_disables_log_without_raising(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        log = EventLog(blocker / "sub" / "e.jsonl")
        log.record(_merge())  # must not raise

    def test_records_from_one_log_share_a_run_id(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        log.record(_merge())
        log.record(RunEvent(outcome="success"))
        ids = {
            json.loads(x)["run_id"]
            for x in (tmp_path / "e.jsonl").read_text().splitlines()
        }
        assert len(ids) == 1


class TestConfigWiring:
    def test_event_log_auto_enables_the_overlap_census(self):
        # "How many particles overlap" is the question the failure
        # records exist to answer, and the census is what measures it.
        cfg = OrchestratorAlgorithmConfig(event_log_path="/tmp/x.jsonl")
        assert cfg.cca_overlap_census_enabled is True

    def test_census_stays_off_without_a_log(self):
        assert OrchestratorAlgorithmConfig().cca_overlap_census_enabled is False

    def test_drop_rescue_still_auto_enables_the_census(self):
        cfg = OrchestratorAlgorithmConfig(cca_drop_rescue_enabled=True)
        assert cfg.cca_overlap_census_enabled is True


class TestEndToEnd:
    def _run(self, tmp_path, **overrides):
        cfg = {
            "N": 64,
            "Df": 1.8,
            "kf": 1.0,
            "rp_g": 100.0,
            "rp_gstd": 1.4,
            "tol_ov": 1e-6,
            "n_subcl_percentage": 0.2,
            "ext_case": 0,
            "seed": 3,
            "event_log_path": str(tmp_path / "events.jsonl"),
        }
        cfg.update(overrides)
        return run_simulation(
            iteration=1,
            sim_config_dict=cfg,
            output_base_dir=str(tmp_path / "out"),
            max_runtime_seconds=120.0,
        )

    def _records(self, tmp_path):
        path = tmp_path / "events.jsonl"
        assert path.exists(), "expected an event log to be written"
        return [json.loads(x) for x in path.read_text().splitlines()]

    def test_successful_run_emits_a_run_record_with_quality(self, tmp_path):
        ok, _, _ = self._run(tmp_path)
        assert ok
        runs = [r for r in self._records(tmp_path) if r["kind"] == "run"]
        assert len(runs) == 1
        run = runs[0]
        assert run["outcome"] == "success"
        assert run["failure_stage"] is None
        assert run["n_particles_actual"] == 64
        # The quality record must be carried, not just computed.
        assert run["overlap_ok"] is True
        assert run["measured_rg"] is not None
        assert run["rg_error_pct"] is not None

    def test_every_record_carries_the_simulation_parameters(self, tmp_path):
        self._run(tmp_path)
        for rec in self._records(tmp_path):
            for key in ("N", "Df", "kf", "rp_gstd", "seed"):
                assert key in rec, f"{rec['kind']} record missing {key}"

    def test_merge_records_describe_the_attempt(self, tmp_path):
        self._run(tmp_path)
        merges = [r for r in self._records(tmp_path) if r["kind"] == "merge"]
        assert merges
        for m in merges:
            assert m["n1"] > 0 and m["n2"] > 0
            assert m["outcome"]
            assert m["round_index"] >= 1

    def test_failed_run_attributes_a_specific_stage(self, tmp_path):
        # Well past the feasibility boundary, with a tight budget.
        ok, _, _ = self._run(tmp_path, N=128, Df=2.6, kf=1.4, rp_gstd=1.9, seed=11)
        assert not ok
        runs = [r for r in self._records(tmp_path) if r["kind"] == "run"]
        assert runs
        stage = runs[-1]["failure_stage"]
        # The point of the record: not merely "it failed", but where.
        assert stage in {"PCA", "CCA", "TIMEOUT", "RADII_GEN"}, stage

    def test_no_log_written_when_unconfigured(self, tmp_path):
        run_simulation(
            iteration=1,
            sim_config_dict={
                "N": 32,
                "Df": 1.8,
                "kf": 1.0,
                "rp_g": 100.0,
                "rp_gstd": 1.3,
                "tol_ov": 1e-6,
                "n_subcl_percentage": 0.25,
                "ext_case": 0,
                "seed": 1,
            },
            output_base_dir=str(tmp_path / "out"),
        )
        assert not (tmp_path / "events.jsonl").exists()


class TestOverlapReportedUnderBothDenominators:
    def test_census_reports_both_normalizations(self):
        from pyfracval.overlap_statistics import compute_overlap_census

        # Two unit spheres 1.0 apart: r_sum = 2, min(r) = 1, so the
        # overlap is 0.5 of r_sum but 1.0 of min(r). A single "overlap
        # fraction" column cannot mean both.
        c1 = np.array([[0.0, 0.0, 0.0]])
        c2 = np.array([[1.0, 0.0, 0.0]])
        r = np.array([1.0])
        census = compute_overlap_census(c1, r, c2, r)
        assert census.max_overlap_fraction == pytest.approx(1.0)
        assert census.max_overlap_fraction_of_rsum == pytest.approx(0.5)

    def test_the_two_denominators_diverge_for_polydisperse_pairs(self):
        from pyfracval.overlap_statistics import compute_overlap_census

        # A big and a small sphere overlapping: the rsum-normalized value
        # stays modest while the rmin-normalized one is large, which is
        # exactly the confusion the split naming prevents.
        c1 = np.array([[0.0, 0.0, 0.0]])
        c2 = np.array([[9.5, 0.0, 0.0]])
        census = compute_overlap_census(c1, np.array([10.0]), c2, np.array([1.0]))
        assert census.max_overlap_fraction > census.max_overlap_fraction_of_rsum
        assert census.max_overlap_fraction_of_rsum < 0.2
        assert census.max_overlap_fraction > 1.0
