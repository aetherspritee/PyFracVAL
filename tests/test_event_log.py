"""Tests for the structured generation event log.

The log exists to answer, over a pooled sweep: where does generation
fail, why, and how badly do particles overlap when it does. These tests
pin the properties that answer depends on - that every record carries the
simulation context, that the three record kinds are distinguishable, that
failure attribution is specific rather than "somewhere in the pipeline",
and that overlap is reported under both denominators rather than one
ambiguous "overlap fraction".
"""

import gzip
import json
import os
from statistics import median

import numpy as np
import pytest

from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.event_log import (
    EventLog,
    Histogram,
    MergeEvent,
    PcaFailureEvent,
    RunEvent,
)
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


class TestHistogram:
    """The summary fold's median must survive binning."""

    def test_integer_metrics_are_exact(self):
        # width == 1 puts one value per bin, so nothing is approximated -
        # this is why the count-valued metrics use it.
        values = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        h = Histogram(width=1.0)
        for v in values:
            h.add(v)
        assert h.n == len(values)
        assert h.lo == 1 and h.hi == 9
        assert h.median() == median(values)

    def test_even_n_matches_statistics_median(self):
        values = [1, 2, 3, 4]
        h = Histogram(width=1.0)
        for v in values:
            h.add(v)
        assert h.median() == median(values) == 2.5

    def test_float_median_lands_within_half_a_bin(self):
        values = [i * 0.00037 for i in range(1000)]
        h = Histogram(width=0.0002)
        for v in values:
            h.add(v)
        assert abs(h.median() - median(values)) <= 0.0001
        assert h.lo == min(values) and h.hi == max(values)

    def test_log_bins_span_many_decades(self):
        values = [1e-16, 1e-12, 1e-8, 1e-4, 1.0]
        h = Histogram(width=0.0005, log=True)
        for v in values:
            h.add(v)
        assert h.median() == pytest.approx(1e-8, rel=0.01)
        assert h.lo == 1e-16 and h.hi == 1.0

    def test_zero_is_kept_as_the_smallest_observation(self):
        # min_overlap is genuinely 0.0 sometimes; log bins cannot hold it,
        # so it must not be silently dropped from the count.
        h = Histogram(width=0.0005, log=True)
        for v in [0.0, 1e-6, 1e-3]:
            h.add(v)
        assert h.n == 3
        assert h.lo == 0.0
        assert h.median() == pytest.approx(1e-6, rel=0.01)

    def test_merge_is_equivalent_to_one_pass(self):
        left, right = [1, 5, 3, 9], [2, 8, 4]
        a, b, both = Histogram(), Histogram(), Histogram()
        for v in left:
            a.add(v)
        for v in right:
            b.add(v)
        for v in left + right:
            both.add(v)
        a.merge(b)
        assert (a.n, a.lo, a.hi, a.median()) == (
            both.n,
            both.lo,
            both.hi,
            both.median(),
        )
        assert a.median() == median(left + right)

    def test_round_trip_through_json(self):
        h = Histogram(width=0.0005, log=True)
        for v in [0.0, 1e-9, 1e-5, 0.5]:
            h.add(v)
        clone = Histogram.from_dict(json.loads(json.dumps(h.to_dict())))
        assert (clone.n, clone.lo, clone.hi) == (h.n, h.lo, h.hi)
        assert clone.median() == h.median()
        assert clone.n_nonpositive == h.n_nonpositive

    def test_differently_binned_histograms_refuse_to_merge(self):
        with pytest.raises(ValueError):
            Histogram(width=1.0).merge(Histogram(width=0.5))

    def test_non_finite_values_are_skipped(self):
        # MergeEvent.min_overlap defaults to +inf ("never measured"), so
        # the fold meets infinities routinely and must not choke on them.
        h = Histogram(width=0.0005, log=True)
        for v in [float("inf"), float("-inf"), float("nan")]:
            h.add(v)
        assert h.n == 0
        assert h.median() is None
        h.add(0.5)
        assert h.n == 1 and h.median() == pytest.approx(0.5, rel=0.01)


class TestSummaryDetail:
    """Summary mode folds in memory and emits one record per run."""

    def test_only_the_run_record_is_written(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl", detail="summary")
        for _ in range(50):
            log.record(_merge())
        log.record(
            PcaFailureEvent(
                subcluster_index=0,
                subcluster_size=12,
                particle_index=3,
                reason="no_candidates",
            )
        )
        # Nothing on disk yet: the fold is still in memory.
        assert not (tmp_path / "e.jsonl").exists()

        log.record(RunEvent(outcome="success"))
        lines = (tmp_path / "e.jsonl").read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["kind"] == "run_summary"
        assert rec["summary"]["n_merges"] == 50
        assert rec["summary"]["n_pca_failures"] == 1

    def test_summary_is_exposed_in_memory(self, tmp_path):
        # The point of the mode: a worker can hand the fold back rather
        # than the caller re-reading it off disk.
        log = EventLog(tmp_path / "e.jsonl", detail="summary", context={"Df": 2.25})
        assert log.summary is None
        log.record(_merge())
        log.record(RunEvent(outcome="failed", failure_stage="CCA"))
        assert log.summary is not None
        assert log.summary["Df"] == 2.25
        assert log.summary["failure_stage"] == "CCA"
        assert log.summary["summary"]["n_merges"] == 1

    def test_counters_match_the_records_they_replace(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl", detail="summary")
        log.record(_merge(outcome="stuck", round_index=1, candidates_tried=4))
        log.record(_merge(outcome="stuck", round_index=2, attempt_index=1))
        log.record(
            _merge(
                outcome="failed_overlap",
                round_index=1,
                n_offending_particles=2,
                n1=10,
                n2=10,
                max_overlap_of_rsum=0.4,
                max_overlap_of_rmin=1.2,
            )
        )
        log.record(_merge(outcome="failed_no_candidates", round_index=3))
        log.record(RunEvent(outcome="failed"))

        s = log.summary["summary"]
        assert s["n_merges"] == 4
        assert s["n_merge_failures"] == 2
        assert s["n_censused"] == 1
        assert s["n_rescued"] == 1  # the one with attempt_index != 0
        assert s["n_localized"] == 1  # 2/20 = 0.1
        assert s["merge_outcomes"] == {
            "stuck": 2,
            "failed_overlap": 1,
            "failed_no_candidates": 1,
        }
        assert s["by_round"] == {
            "1": {"ok": 1, "bad": 1},
            "2": {"ok": 1, "bad": 0},
            "3": {"ok": 0, "bad": 1},
        }
        assert s["hists"]["overlap_of_rsum"]["n"] == 1

    def test_each_run_folds_independently(self, tmp_path):
        # One EventLog can outlive a run in library use; the second run
        # must not inherit the first one's totals.
        log = EventLog(tmp_path / "e.jsonl", detail="summary")
        log.record(_merge())
        log.record(RunEvent(outcome="success"))
        log.record(_merge())
        log.record(_merge())
        log.record(RunEvent(outcome="success"))
        counts = [
            json.loads(x)["summary"]["n_merges"]
            for x in (tmp_path / "e.jsonl").read_text().splitlines()
        ]
        assert counts == [1, 2]

    def test_summary_mode_never_compresses_or_shards(self, tmp_path):
        # One line per run needs no gzip stream, so the plain atomic-append
        # path applies and no .pid shard appears.
        log = EventLog(tmp_path / "e.jsonl.gz", detail="summary")
        assert not log.compressed
        assert log.path == tmp_path / "e.jsonl.gz"

    def test_rejects_an_unknown_detail(self, tmp_path):
        with pytest.raises(ValueError):
            EventLog(tmp_path / "e.jsonl", detail="terse")


class TestCompressedLog:
    """A ``.gz`` path must round-trip losslessly, shard per process, and
    share one stream between the ``EventLog`` instances a sweep creates."""

    def test_gz_path_is_sharded_by_pid(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl.gz")
        assert log.compressed
        assert log.path.name == f"events.pid{os.getpid()}.jsonl.gz"

    def test_records_round_trip_through_gzip(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl.gz", context={"Df": 2.1})
        log.record(_merge(candidates_tried=7))
        log.record(RunEvent(outcome="success", measured_rg=1.5))
        log.close()

        with gzip.open(log.path, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        assert [r["kind"] for r in records] == ["merge", "run"]
        assert records[0]["candidates_tried"] == 7
        assert records[0]["Df"] == 2.1
        assert records[1]["measured_rg"] == 1.5

    def test_instances_on_one_path_share_a_stream(self, tmp_path):
        # A sweep builds one EventLog per trial. Each opening its own gzip
        # stream on the same file would interleave two buffers into
        # garbage, so they must resolve to a single writer.
        a = EventLog(tmp_path / "e.jsonl.gz", context={"N": 64})
        b = EventLog(tmp_path / "e.jsonl.gz", context={"N": 128})
        assert a.path == b.path
        for _ in range(50):
            a.record(_merge())
            b.record(_merge())
        a.close()

        with gzip.open(a.path, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        assert len(records) == 100
        assert {r["N"] for r in records} == {64, 128}
        assert len({r["run_id"] for r in records}) == 2

    def test_compression_beats_the_plain_text_it_replaces(self, tmp_path):
        plain = EventLog(tmp_path / "p.jsonl")
        packed = EventLog(tmp_path / "p.jsonl.gz")
        for i in range(400):
            plain.record(_merge(candidates_tried=i))
            packed.record(_merge(candidates_tried=i))
        packed.close()
        # The whole point of the suffix: guards against a regression to
        # one gzip member per record, which would come out *larger*.
        assert packed.path.stat().st_size < plain.path.stat().st_size / 5

    def test_plain_path_is_not_sharded_or_compressed(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        assert not log.compressed
        assert log.path == tmp_path / "e.jsonl"

    def test_a_forked_child_does_not_corrupt_the_parents_stream(self, tmp_path):
        # pca_subclusters builds subclusters through a forked Pool, so a
        # child can inherit an open gzip stream mid-record. If it wrote
        # through it - or let atexit close it - the parent's log would be
        # unreadable from that byte on. This is the regression guard.
        log = EventLog(tmp_path / "e.jsonl.gz", context={"where": "parent"})
        for _ in range(20):
            log.record(_merge())  # opens the stream before forking

        pid = os.fork()
        if pid == 0:  # child
            try:
                child = EventLog(tmp_path / "e.jsonl.gz", context={"where": "child"})
                for _ in range(20):
                    child.record(_merge())
                child.close()
                os._exit(0)
            except BaseException:
                os._exit(1)

        assert os.waitpid(pid, 0)[1] == 0, "child failed"
        for _ in range(20):
            log.record(_merge())
        log.close()

        with gzip.open(log.path, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        # Parent's file: complete, readable, and free of the child's rows.
        assert len(records) == 40
        assert {r["where"] for r in records} == {"parent"}
        # The child wrote its own shard rather than nothing.
        shards = sorted(tmp_path.glob("e.pid*.jsonl.gz"))
        assert len(shards) == 2


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
