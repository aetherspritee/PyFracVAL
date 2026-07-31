#!/usr/bin/env python3
"""Aggregate structured event logs into failure statistics.

``pyfracval/event_log.py`` writes one JSONL record per CCA merge attempt,
per PCA subcluster failure, and per completed run, each stamped with the
simulation parameters. This turns a pile of those into the tables a paper
needs:

- **Where** generation fails: run-level attribution across PCA / CCA /
  timeout, and within CCA, which round.
- **Why**: the failure-mode taxonomy (no feasible candidates at all,
  versus candidates that all overlapped, versus a Gamma with no real
  solution). These have different causes and different fixes.
- **How badly**: how many particles overlap at give-up, what fraction of
  the cluster pair that is, and by how much they interpenetrate.

Overlap is reported under both denominators the codebase uses, because
they differ by a large factor for wide size distributions and conflating
them would misstate severity:

- ``_of_rsum`` normalizes by ``r_i + r_j`` and is comparable to ``tol_ov``.
- ``_of_rmin`` normalizes by ``min(r_i, r_j)`` and measures how deeply the
  smaller particle is penetrated.

Usage:
    devenv shell -- uv run python benchmarks/analyze_event_log.py LOG.jsonl [...]
    devenv shell -- uv run python benchmarks/analyze_event_log.py LOG.jsonl --by Df kf
    devenv shell -- uv run python benchmarks/analyze_event_log.py LOG.jsonl --json out.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

SUCCESS_OUTCOMES = {
    "stuck",
    "stuck_relaxed_tol",
    "rescued_soft_relaxation",
    "rescued_drop",
}


def iter_events(paths: list[Path]):
    """Yield records one at a time.

    Streamed rather than materialised: a full boundary sweep writes on the
    order of a million merge records, and holding those as Python dicts
    costs several GB. Everything below accumulates counters and plain
    float arrays instead, which is a couple of orders of magnitude
    cheaper and keeps arbitrarily long logs analysable.
    """
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write can leave one torn final
                    # line; that should not invalidate everything before.
                    print(f"  ! skipping malformed line {path}:{line_no}")


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:5.1f}%" if whole else "    -"


def _stats(values, fmt: str = "{:.3g}") -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    return (
        f"min={fmt.format(min(vals))} "
        f"median={fmt.format(median(vals))} "
        f"max={fmt.format(max(vals))}"
    )


def report_runs(runs: list[dict]) -> None:
    print(f"\n{'=' * 74}\nRUNS: where does generation fail?\n{'=' * 74}")
    if not runs:
        print("  (no run-level records in this log)")
        return
    total = len(runs)
    ok = [r for r in runs if r["outcome"] == "success"]
    bad = [r for r in runs if r["outcome"] != "success"]

    print(f"  total {total}   success {len(ok)} ({_pct(len(ok), total)})")
    for stage, n in Counter(r.get("failure_stage") or "-" for r in bad).most_common():
        print(f"    failed at {stage:14s} {n:6d}  {_pct(n, total)}")

    if ok:
        bad_geom = [r for r in ok if r.get("overlap_ok") is False]
        print(
            f"\n  successful runs with invalid geometry: {len(bad_geom)}"
            f"  ({_pct(len(bad_geom), len(ok))})"
        )
        rg = [r["rg_error_pct"] for r in ok if r.get("rg_error_pct") is not None]
        if rg:
            print(f"  Rg error vs scaling law (%): {_stats(rg, '{:+.2f}')}")
        dropped = sum(r.get("n_particles_dropped", 0) or 0 for r in ok)
        if dropped:
            print(f"  particles dropped across successful runs: {dropped}")
    elapsed = [r.get("elapsed_s") for r in runs if r.get("elapsed_s")]
    if elapsed:
        print(f"  wall time per run (s): {_stats(elapsed, '{:.2f}')}")


def report_pca(pca: list[dict]) -> None:
    print(f"\n{'=' * 74}\nPCA FAILURES: why can a subcluster not be built?\n{'=' * 74}")
    if not pca:
        print("  none recorded")
        return
    print(f"  total {len(pca)}")
    for reason, n in Counter(r["reason"] for r in pca).most_common():
        print(f"    {reason:26s} {n:6d}  {_pct(n, len(pca))}")
    print(
        f"  failed at particle index: "
        f"{_stats([r['particle_index'] for r in pca], '{:.0f}')}"
    )
    print(
        f"  subcluster size         : "
        f"{_stats([r['subcluster_size'] for r in pca], '{:.0f}')}"
    )


def report_merges(merges: list[dict]) -> None:
    print(f"\n{'=' * 74}\nCCA MERGES: why do sticking attempts fail?\n{'=' * 74}")
    if not merges:
        print("  none recorded")
        return
    total = len(merges)
    succ = [m for m in merges if m["outcome"] in SUCCESS_OUTCOMES]
    fail = [m for m in merges if m["outcome"] not in SUCCESS_OUTCOMES]
    print(f"  attempts {total}   stuck {len(succ)} ({_pct(len(succ), total)})")
    for outcome, n in Counter(m["outcome"] for m in merges).most_common():
        print(f"    {outcome:26s} {n:6d}  {_pct(n, total)}")

    by_round: dict[int, Counter] = defaultdict(Counter)
    for m in merges:
        by_round[m["round_index"]][
            "ok" if m["outcome"] in SUCCESS_OUTCOMES else "bad"
        ] += 1
    print(f"\n  {'round':>6} {'stuck':>8} {'failed':>8} {'fail rate':>10}")
    for rnd in sorted(by_round):
        c = by_round[rnd]
        print(
            f"  {rnd:6d} {c['ok']:8d} {c['bad']:8d} "
            f"{_pct(c['bad'], c['ok'] + c['bad']):>10}"
        )

    rescued = sum(1 for m in succ if m.get("attempt_index", 0) > 0)
    print(
        f"\n  merges that needed a later partner (backtracking): {rescued}"
        f"  ({_pct(rescued, max(len(succ), 1))} of stuck)"
    )
    print(
        f"  candidate pairs tried, stuck : "
        f"{_stats([m['candidates_tried'] for m in succ], '{:.0f}')}"
    )
    print(
        f"  candidate pairs tried, failed: "
        f"{_stats([m['candidates_tried'] for m in fail], '{:.0f}')}"
    )


def report_overlap(merges: list[dict]) -> None:
    """The question the log mainly exists to answer."""
    fail = [m for m in merges if m["outcome"] not in SUCCESS_OUTCOMES]
    censused = [m for m in fail if m.get("n_offending_particles") is not None]
    print(
        f"\n{'=' * 74}\nOVERLAP AT FAILURE: how many particles, and how badly?"
        f"\n{'=' * 74}"
    )
    if not censused:
        print("  no censused failures (set event_log_path, which turns the")
        print("  overlap census on automatically)")
        return
    print(
        f"  censused failures: {len(censused)} of {len(fail)}"
        f"  ({_pct(len(censused), len(fail))})"
    )
    print("  (the remainder failed before any geometry existed to census,")
    print("   e.g. no feasible candidate pair at all)")

    offending = [m["n_offending_particles"] for m in censused]
    pair_sizes = [m["n1"] + m["n2"] for m in censused]
    fracs = [o / s for o, s in zip(offending, pair_sizes) if s]
    print(f"\n  offending particles       : {_stats(offending, '{:.0f}')}")
    print(f"  cluster-pair size         : {_stats(pair_sizes, '{:.0f}')}")
    print(f"  offending fraction        : {_stats(fracs, '{:.2f}')}")
    print(
        f"  overlapping pairs         : "
        f"{_stats([m.get('n_pairs_overlapping') for m in censused], '{:.0f}')}"
    )
    print(
        f"\n  worst overlap / (ri+rj)   : "
        f"{_stats([m.get('max_overlap_of_rsum') for m in censused], '{:.3f}')}"
        "   <- comparable to tol_ov"
    )
    print(
        f"  worst overlap / min(ri,rj): "
        f"{_stats([m.get('max_overlap_of_rmin') for m in censused], '{:.3f}')}"
        "   <- penetration of the smaller particle"
    )
    print(
        f"  best overlap reached      : "
        f"{_stats([m.get('min_overlap') for m in censused], '{:.3e}')}"
    )
    print("  (best-reached far above tol_ov means failures are not near-misses)")

    localized = sum(1 for f in fracs if f <= 0.1)
    print(
        f"\n  failures with <=10% of the pair offending: {localized}"
        f"  ({_pct(localized, len(fracs))})"
    )
    print(
        "  (this is the premise drop-rescue rests on; see docs/source/drop_rescue.md)"
    )


def report_sliced(events: list[dict], keys: list[str]) -> None:
    """Failure rate and overlap severity sliced by simulation parameters."""
    runs = [e for e in events if e.get("kind") == "run"]
    merges = [e for e in events if e.get("kind", "merge") == "merge"]
    if not runs:
        return

    def key_of(e):
        return tuple(e.get(k) for k in keys)

    run_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        run_groups[key_of(r)].append(r)
    merge_groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in merges:
        merge_groups[key_of(m)].append(m)

    print(f"\n{'=' * 74}\nSLICED BY {', '.join(keys)}\n{'=' * 74}")
    header = "  " + " ".join(f"{k:>7}" for k in keys)
    print(
        f"{header} {'runs':>6} {'success':>8} {'merge fail':>11} "
        f"{'med offend':>11} {'med frac':>9}"
    )
    for key in sorted(run_groups, key=lambda t: tuple(str(x) for x in t)):
        rs = run_groups[key]
        ok = sum(1 for r in rs if r["outcome"] == "success")
        ms = merge_groups.get(key, [])
        mf = [m for m in ms if m["outcome"] not in SUCCESS_OUTCOMES]
        cens = [m for m in mf if m.get("n_offending_particles") is not None]
        med_off = median([m["n_offending_particles"] for m in cens]) if cens else None
        fracs = [
            m["n_offending_particles"] / (m["n1"] + m["n2"])
            for m in cens
            if (m["n1"] + m["n2"])
        ]
        med_frac = median(fracs) if fracs else None
        cells = " ".join(f"{str(x):>7}" for x in key)
        print(
            f"  {cells} {len(rs):>6} {_pct(ok, len(rs)):>8} "
            f"{(_pct(len(mf), len(ms)) if ms else '-'):>11} "
            f"{(f'{med_off:.0f}' if med_off is not None else '-'):>11} "
            f"{(f'{med_frac:.2f}' if med_frac is not None else '-'):>9}"
        )


class Accumulator:
    """Streaming aggregation of the statistics the reports need.

    Holds counters and float arrays only - never the records themselves -
    so a multi-gigabyte log analyses in a few hundred MB.
    """

    def __init__(self, slice_keys: list[str]):
        self.kinds = Counter()
        self.slice_keys = slice_keys

        # runs
        self.n_runs = 0
        self.n_runs_ok = 0
        self.failure_stages = Counter()
        self.run_rg_err: list[float] = []
        self.run_elapsed: list[float] = []
        self.n_bad_geometry = 0
        self.n_dropped_total = 0

        # pca failures
        self.pca_reasons = Counter()
        self.pca_particle_idx: list[float] = []
        self.pca_sub_size: list[float] = []
        self.n_pca = 0

        # merges
        self.n_merges = 0
        self.merge_outcomes = Counter()
        self.by_round: dict[int, Counter] = defaultdict(Counter)
        self.n_rescued = 0
        self.cand_ok: list[float] = []
        self.cand_bad: list[float] = []

        # overlap census
        self.n_fail = 0
        self.n_censused = 0
        self.offending: list[float] = []
        self.pair_size: list[float] = []
        self.frac: list[float] = []
        self.pairs_ov: list[float] = []
        self.ov_rsum: list[float] = []
        self.ov_rmin: list[float] = []
        self.best_ov: list[float] = []

        # per-slice
        self.slice_runs: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
        self.slice_merges: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
        self.slice_offend: dict[tuple, list[float]] = defaultdict(list)
        self.slice_frac: dict[tuple, list[float]] = defaultdict(list)

    def _key(self, e):
        return tuple(e.get(k) for k in self.slice_keys)

    def add(self, e: dict) -> None:
        kind = e.get("kind", "merge")
        self.kinds[kind] += 1
        if kind == "run":
            self._add_run(e)
        elif kind == "pca_failure":
            self._add_pca(e)
        else:
            self._add_merge(e)

    def _add_run(self, e):
        self.n_runs += 1
        key = self._key(e)
        self.slice_runs[key][0] += 1
        if e.get("outcome") == "success":
            self.n_runs_ok += 1
            self.slice_runs[key][1] += 1
            if e.get("overlap_ok") is False:
                self.n_bad_geometry += 1
            if e.get("rg_error_pct") is not None:
                self.run_rg_err.append(e["rg_error_pct"])
            self.n_dropped_total += e.get("n_particles_dropped", 0) or 0
        else:
            self.failure_stages[e.get("failure_stage") or "-"] += 1
        if e.get("elapsed_s"):
            self.run_elapsed.append(e["elapsed_s"])

    def _add_pca(self, e):
        self.n_pca += 1
        self.pca_reasons[e.get("reason", "unknown")] += 1
        self.pca_particle_idx.append(e.get("particle_index", -1))
        self.pca_sub_size.append(e.get("subcluster_size", 0))

    def _add_merge(self, e):
        self.n_merges += 1
        outcome = e.get("outcome", "?")
        self.merge_outcomes[outcome] += 1
        ok = outcome in SUCCESS_OUTCOMES
        self.by_round[e.get("round_index", 0)]["ok" if ok else "bad"] += 1
        key = self._key(e)
        self.slice_merges[key][0] += 1

        if ok:
            if e.get("attempt_index", 0):
                self.n_rescued += 1
            self.cand_ok.append(e.get("candidates_tried", 0))
            return

        self.n_fail += 1
        self.slice_merges[key][1] += 1
        self.cand_bad.append(e.get("candidates_tried", 0))

        n_off = e.get("n_offending_particles")
        if n_off is None:
            return
        self.n_censused += 1
        size = (e.get("n1", 0) or 0) + (e.get("n2", 0) or 0)
        self.offending.append(n_off)
        self.pair_size.append(size)
        self.slice_offend[key].append(n_off)
        if size:
            self.frac.append(n_off / size)
            self.slice_frac[key].append(n_off / size)
        if e.get("n_pairs_overlapping") is not None:
            self.pairs_ov.append(e["n_pairs_overlapping"])
        if e.get("max_overlap_of_rsum") is not None:
            self.ov_rsum.append(e["max_overlap_of_rsum"])
        if e.get("max_overlap_of_rmin") is not None:
            self.ov_rmin.append(e["max_overlap_of_rmin"])
        if e.get("min_overlap") is not None:
            self.best_ov.append(e["min_overlap"])


def report(a: Accumulator) -> None:
    print("=" * 74)
    total = sum(a.kinds.values())
    print(f"{total} records: " + ", ".join(f"{k}={v}" for k, v in a.kinds.items()))
    print("=" * 74)

    # --- runs ---
    print(f"\n{'=' * 74}\nRUNS: where does generation fail?\n{'=' * 74}")
    if not a.n_runs:
        print("  (no run-level records in this log)")
    else:
        print(
            f"  total {a.n_runs}   success {a.n_runs_ok} ({_pct(a.n_runs_ok, a.n_runs)})"
        )
        for stage, n in a.failure_stages.most_common():
            print(f"    failed at {stage:14s} {n:6d}  {_pct(n, a.n_runs)}")
        print(
            f"\n  successful runs with invalid geometry: {a.n_bad_geometry}"
            f"  ({_pct(a.n_bad_geometry, max(a.n_runs_ok, 1))})"
        )
        if a.run_rg_err:
            print(f"  Rg error vs scaling law (%): {_stats(a.run_rg_err, '{:+.2f}')}")
        if a.n_dropped_total:
            print(f"  particles dropped across successful runs: {a.n_dropped_total}")
        if a.run_elapsed:
            print(f"  wall time per run (s): {_stats(a.run_elapsed, '{:.2f}')}")

    # --- pca ---
    print(f"\n{'=' * 74}\nPCA FAILURES: why can a subcluster not be built?\n{'=' * 74}")
    if not a.n_pca:
        print("  none recorded")
    else:
        print(f"  total {a.n_pca}")
        for reason, n in a.pca_reasons.most_common():
            print(f"    {reason:26s} {n:6d}  {_pct(n, a.n_pca)}")
        print(f"  failed at particle index: {_stats(a.pca_particle_idx, '{:.0f}')}")
        print(f"  subcluster size         : {_stats(a.pca_sub_size, '{:.0f}')}")

    # --- merges ---
    print(f"\n{'=' * 74}\nCCA MERGES: why do sticking attempts fail?\n{'=' * 74}")
    if not a.n_merges:
        print("  none recorded")
    else:
        n_ok = a.n_merges - a.n_fail
        print(f"  attempts {a.n_merges}   stuck {n_ok} ({_pct(n_ok, a.n_merges)})")
        for outcome, n in a.merge_outcomes.most_common():
            print(f"    {outcome:26s} {n:6d}  {_pct(n, a.n_merges)}")
        print(f"\n  {'round':>6} {'stuck':>8} {'failed':>8} {'fail rate':>10}")
        for rnd in sorted(a.by_round):
            c = a.by_round[rnd]
            print(
                f"  {rnd:6d} {c['ok']:8d} {c['bad']:8d} "
                f"{_pct(c['bad'], c['ok'] + c['bad']):>10}"
            )
        print(
            f"\n  merges that needed a later partner (backtracking): {a.n_rescued}"
            f"  ({_pct(a.n_rescued, max(n_ok, 1))} of stuck)"
        )
        print(f"  candidate pairs tried, stuck : {_stats(a.cand_ok, '{:.0f}')}")
        print(f"  candidate pairs tried, failed: {_stats(a.cand_bad, '{:.0f}')}")

    # --- overlap ---
    print(
        f"\n{'=' * 74}\nOVERLAP AT FAILURE: how many particles, and how badly?"
        f"\n{'=' * 74}"
    )
    if not a.n_censused:
        print("  no censused failures (set event_log_path, which turns the")
        print("  overlap census on automatically)")
    else:
        print(
            f"  censused failures: {a.n_censused} of {a.n_fail}"
            f"  ({_pct(a.n_censused, a.n_fail)})"
        )
        print("  (the remainder failed before any geometry existed to census,")
        print("   e.g. no feasible candidate pair at all)")
        print(f"\n  offending particles       : {_stats(a.offending, '{:.0f}')}")
        print(f"  cluster-pair size         : {_stats(a.pair_size, '{:.0f}')}")
        print(f"  offending fraction        : {_stats(a.frac, '{:.2f}')}")
        print(f"  overlapping pairs         : {_stats(a.pairs_ov, '{:.0f}')}")
        print(
            f"\n  worst overlap / (ri+rj)   : {_stats(a.ov_rsum, '{:.3f}')}"
            "   <- comparable to tol_ov"
        )
        print(
            f"  worst overlap / min(ri,rj): {_stats(a.ov_rmin, '{:.3f}')}"
            "   <- penetration of the smaller particle"
        )
        print(f"  best overlap reached      : {_stats(a.best_ov, '{:.3e}')}")
        print("  (best-reached far above tol_ov means failures are not near-misses)")
        localized = sum(1 for f in a.frac if f <= 0.1)
        print(
            f"\n  failures with <=10% of the pair offending: {localized}"
            f"  ({_pct(localized, max(len(a.frac), 1))})"
        )
        print(
            "  (this is the premise drop-rescue rests on; see "
            "docs/source/drop_rescue.md)"
        )

    # --- sliced ---
    if a.slice_runs:
        print(f"\n{'=' * 74}\nSLICED BY {', '.join(a.slice_keys)}\n{'=' * 74}")
        header = "  " + " ".join(f"{k:>7}" for k in a.slice_keys)
        print(
            f"{header} {'runs':>6} {'success':>8} {'merge fail':>11} "
            f"{'med offend':>11} {'med frac':>9}"
        )
        for key in sorted(a.slice_runs, key=lambda t: tuple(str(x) for x in t)):
            n_r, n_ok = a.slice_runs[key]
            m_all, m_bad = a.slice_merges.get(key, [0, 0])
            offs = a.slice_offend.get(key, [])
            fr = a.slice_frac.get(key, [])
            cells = " ".join(f"{str(x):>7}" for x in key)
            print(
                f"  {cells} {n_r:>6} {_pct(n_ok, n_r):>8} "
                f"{(_pct(m_bad, m_all) if m_all else '-'):>11} "
                f"{(f'{median(offs):.0f}' if offs else '-'):>11} "
                f"{(f'{median(fr):.2f}' if fr else '-'):>9}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument(
        "--by",
        nargs="*",
        default=["Df", "kf", "rp_gstd", "N"],
        help="context fields to slice by (default: Df kf rp_gstd N)",
    )
    ap.add_argument("--json", type=Path, help="also write the summary as JSON")
    args = ap.parse_args()

    missing = [p for p in args.logs if not p.exists()]
    if missing:
        raise SystemExit(f"No such log file(s): {', '.join(map(str, missing))}")

    acc = Accumulator(list(args.by or []))
    for event in iter_events(args.logs):
        acc.add(event)
    if not sum(acc.kinds.values()):
        raise SystemExit("No events found - was event_log_path set?")

    report(acc)

    if args.json:
        summary = {
            "n_records": sum(acc.kinds.values()),
            "kinds": dict(acc.kinds),
            "n_runs": acc.n_runs,
            "n_runs_success": acc.n_runs_ok,
            "failure_stages": dict(acc.failure_stages),
            "merge_outcomes": dict(acc.merge_outcomes),
            "pca_failure_reasons": dict(acc.pca_reasons),
            "n_merge_failures": acc.n_fail,
            "n_censused": acc.n_censused,
            "offending_particles": acc.offending,
            "offending_fraction": acc.frac,
            "max_overlap_of_rsum": acc.ov_rsum,
            "max_overlap_of_rmin": acc.ov_rmin,
            "best_overlap_reached": acc.best_ov,
            "by_round": {str(r): dict(c) for r, c in sorted(acc.by_round.items())},
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
