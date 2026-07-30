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


def load(paths: list[Path]) -> list[dict]:
    events: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # A run killed mid-write can leave one torn final
                    # line; that should not invalidate everything before.
                    print(f"  ! skipping malformed line {path}:{line_no}")
    return events


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

    events = load(args.logs)
    if not events:
        raise SystemExit("No events found - was event_log_path set?")

    kinds = Counter(e.get("kind", "merge") for e in events)
    print("=" * 74)
    print(f"{len(events)} records: " + ", ".join(f"{k}={v}" for k, v in kinds.items()))
    print("=" * 74)

    runs = [e for e in events if e.get("kind") == "run"]
    pca = [e for e in events if e.get("kind") == "pca_failure"]
    merges = [e for e in events if e.get("kind", "merge") == "merge"]

    report_runs(runs)
    report_pca(pca)
    report_merges(merges)
    report_overlap(merges)
    if args.by:
        report_sliced(events, list(args.by))

    if args.json:
        fail = [m for m in merges if m["outcome"] not in SUCCESS_OUTCOMES]
        cens = [m for m in fail if m.get("n_offending_particles") is not None]
        summary = {
            "n_records": len(events),
            "kinds": dict(kinds),
            "n_runs": len(runs),
            "n_runs_success": sum(1 for r in runs if r["outcome"] == "success"),
            "failure_stages": dict(
                Counter(
                    r.get("failure_stage") or "-"
                    for r in runs
                    if r["outcome"] != "success"
                )
            ),
            "merge_outcomes": dict(Counter(m["outcome"] for m in merges)),
            "pca_failure_reasons": dict(Counter(r["reason"] for r in pca)),
            "offending_particles": [m["n_offending_particles"] for m in cens],
            "offending_fraction": [
                m["n_offending_particles"] / (m["n1"] + m["n2"])
                for m in cens
                if (m["n1"] + m["n2"])
            ],
            "max_overlap_of_rsum": [m.get("max_overlap_of_rsum") for m in cens],
            "max_overlap_of_rmin": [m.get("max_overlap_of_rmin") for m in cens],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
