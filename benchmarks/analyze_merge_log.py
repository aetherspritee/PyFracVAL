#!/usr/bin/env python3
"""Turn per-merge JSONL event logs into readable statistics.

``pyfracval/merge_log.py`` records one line per CCA merge attempt when
``cca_merge_log_path`` is set. This script aggregates those lines into the
questions people actually ask about a run or a population of runs:

- Not just *whether* merges failed, but **how**: no feasible candidates at
  all, versus candidates that all overlapped, versus a Gamma with no real
  solution. These have completely different fixes.
- **How close** the failures were: best overlap reached, and how many
  particles were actually offending (when the census is enabled).
- **Where** in the hierarchy failures concentrate: round 1 (merging fresh
  PCA subclusters) versus later rounds (merging two already-large
  aggregates). Every hard-regime failure characterised so far has been
  round 1, and that shaped which fixes were worth building.
- **What backtracking bought**: how often a merge only succeeded because
  a later partner was tried (``attempt_index > 0``).
- Whether the Gamma/R_max margin predicts success, which is the standing
  question behind every "can we pre-filter pairs" idea.

Usage:
    devenv shell -- uv run python benchmarks/analyze_merge_log.py LOG.jsonl [...]
    devenv shell -- uv run python benchmarks/analyze_merge_log.py LOG.jsonl --json out.json
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


def load_events(paths: list[Path]) -> list[dict]:
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
                    # A run killed mid-write can leave one torn final line;
                    # that should not invalidate everything before it.
                    print(f"  ! skipping malformed line {path}:{line_no}")
    return events


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:5.1f}%" if whole else "    -"


def _fmt_stats(values: list[float], fmt: str = "{:.3g}") -> str:
    if not values:
        return "n/a"
    return (
        f"min={fmt.format(min(values))} "
        f"median={fmt.format(median(values))} "
        f"max={fmt.format(max(values))}"
    )


def summarize(events: list[dict]) -> dict:
    total = len(events)
    outcomes = Counter(e["outcome"] for e in events)
    successes = [e for e in events if e["outcome"] in SUCCESS_OUTCOMES]
    failures = [e for e in events if e["outcome"] not in SUCCESS_OUTCOMES]

    by_round: dict[int, Counter] = defaultdict(Counter)
    for e in events:
        by_round[e["round_index"]][
            "success" if e["outcome"] in SUCCESS_OUTCOMES else "failure"
        ] += 1

    rescued = [e for e in successes if e.get("attempt_index", 0) > 0]

    # Gamma margin: how much slack the feasibility gate had. Negative means
    # the pair only passed via the relaxation factor.
    def margin(e):
        s = e.get("sum_rmax", 0.0)
        return (s - e.get("gamma_pc", 0.0)) / s if s else 0.0

    return {
        "total_attempts": total,
        "outcomes": dict(outcomes),
        "n_success": len(successes),
        "n_failure": len(failures),
        "success_rate": len(successes) / total if total else 0.0,
        "n_rescued_by_backtracking": len(rescued),
        "by_round": {r: dict(c) for r, c in sorted(by_round.items())},
        "candidates_tried_success": [e.get("candidates_tried", 0) for e in successes],
        "candidates_tried_failure": [e.get("candidates_tried", 0) for e in failures],
        "rotations_success": [e.get("rotations_used", 0) for e in successes],
        "min_overlap_failure": [
            e["min_overlap"] for e in failures if e.get("min_overlap") is not None
        ],
        "offending_failure": [
            e["n_offending_particles"]
            for e in failures
            if e.get("n_offending_particles") is not None
        ],
        "cluster_sizes_failure": [e.get("n1", 0) + e.get("n2", 0) for e in failures],
        "margin_success": [margin(e) for e in successes],
        "margin_failure": [margin(e) for e in failures],
        "particles_dropped": sum(e.get("n_particles_dropped", 0) for e in events),
    }


def report(s: dict) -> None:
    total = s["total_attempts"]
    print("=" * 78)
    print(f"CCA merge attempts: {total}")
    print("=" * 78)

    print(
        f"\nOverall: {s['n_success']} succeeded ({_pct(s['n_success'], total)}), "
        f"{s['n_failure']} failed ({_pct(s['n_failure'], total)})"
    )

    print("\nOutcome breakdown (the 'why', not just the 'whether'):")
    for outcome, count in sorted(s["outcomes"].items(), key=lambda kv: -kv[1]):
        print(f"  {outcome:28s} {count:7d}  {_pct(count, total)}")

    print("\nBy CCA round (round 1 = merging fresh PCA subclusters):")
    print(f"  {'round':>6} {'success':>9} {'failure':>9} {'fail rate':>10}")
    for rnd, counts in s["by_round"].items():
        ok = counts.get("success", 0)
        bad = counts.get("failure", 0)
        print(f"  {rnd:6d} {ok:9d} {bad:9d} {_pct(bad, ok + bad):>10}")

    print(
        f"\nMerges rescued by trying a later partner: {s['n_rescued_by_backtracking']}"
        f"  ({_pct(s['n_rescued_by_backtracking'], max(s['n_success'], 1))} of successes)"
    )
    if s["particles_dropped"]:
        print(f"Particles dropped by drop-rescue: {s['particles_dropped']}")

    print("\nSearch effort:")
    print(
        f"  candidate pairs tried (success): {_fmt_stats(s['candidates_tried_success'], '{:.0f}')}"
    )
    print(
        f"  candidate pairs tried (failure): {_fmt_stats(s['candidates_tried_failure'], '{:.0f}')}"
    )
    print(
        f"  rotations used     (success)   : {_fmt_stats(s['rotations_success'], '{:.0f}')}"
    )

    if s["min_overlap_failure"]:
        print(
            "\nHow close failures came (best overlap reached; tol_ov is typically 1e-6):"
        )
        print(f"  {_fmt_stats(s['min_overlap_failure'], '{:.3e}')}")
    if s["offending_failure"]:
        print(
            "\nOffending particles per failure (census; informs drop-rescue viability):"
        )
        print(f"  {_fmt_stats(s['offending_failure'], '{:.0f}')}")
        print(
            f"  failing cluster-pair sizes: {_fmt_stats(s['cluster_sizes_failure'], '{:.0f}')}"
        )
        fractions = [
            o / c
            for o, c in zip(s["offending_failure"], s["cluster_sizes_failure"])
            if c
        ]
        if fractions:
            print(f"  offending fraction        : {_fmt_stats(fractions, '{:.2f}')}")

    if s["margin_success"] or s["margin_failure"]:
        print("\nGamma feasibility margin, (sum_rmax - gamma)/sum_rmax:")
        print(f"  success: {_fmt_stats(s['margin_success'], '{:.3f}')}")
        print(f"  failure: {_fmt_stats(s['margin_failure'], '{:.3f}')}")
        print("  (overlapping ranges mean the cheap gate cannot predict sticking,")
        print("   which is why backtracking reacts to real outcomes instead.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="merge log JSONL file(s)")
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    args = parser.parse_args()

    missing = [p for p in args.logs if not p.exists()]
    if missing:
        raise SystemExit(f"No such log file(s): {', '.join(str(p) for p in missing)}")

    events = load_events(args.logs)
    if not events:
        raise SystemExit("No merge events found - was cca_merge_log_path set?")

    summary = summarize(events)
    report(summary)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
