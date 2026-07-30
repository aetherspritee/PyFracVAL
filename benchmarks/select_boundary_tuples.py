#!/usr/bin/env python3
"""Select boundary tuples from a stability sweep summary.

The selector reduces a full Step-1 grid to a compact set of tuples for
Step-2 feature testing. Selection is stratified per sigma into stable,
borderline, and unstable classes, while prioritizing combinations close to
classification boundaries and with stronger N-degradation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

STABLE_THRESHOLD = 0.90
BORDERLINE_THRESHOLD = 0.40


def _classify(sr: float, stable: float, borderline: float) -> str:
    if sr >= stable:
        return "stable"
    if sr >= borderline:
        return "borderline"
    return "unstable"


def _fmt_float(value: float) -> str:
    return f"{value:.6g}"


def _parse_sigmas(text: str | None) -> set[float] | None:
    if not text:
        return None
    out: set[float] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.add(float(token))
    return out if out else None


def _load_rows(summary_path: Path) -> list[dict]:
    with summary_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("summary JSON has invalid results payload")
    return rows


def _aggregate(rows: list[dict], allowed_sigmas: set[float] | None) -> list[dict]:
    grouped: dict[tuple[float, float, float], dict[int, float]] = defaultdict(dict)
    for row in rows:
        sigma = float(row["rp_gstd"])
        if allowed_sigmas is not None and sigma not in allowed_sigmas:
            continue
        df = float(row["Df"])
        kf = float(row["kf"])
        n_val = int(row["N"])
        sr = float(row["success_rate"])
        grouped[(sigma, df, kf)][n_val] = sr

    records: list[dict] = []
    for (sigma, df, kf), n_map in grouped.items():
        ns = sorted(n_map)
        if not ns:
            continue
        n_min = ns[0]
        n_max = ns[-1]
        sr_min = n_map[n_min]
        sr_max = n_map[n_max]
        degradation = sr_min - sr_max
        records.append(
            {
                "sigma": sigma,
                "Df": df,
                "kf": kf,
                "n_success": n_map,
                "n_values": ns,
                "n_min": n_min,
                "n_max": n_max,
                "sr_n_min": sr_min,
                "sr_n_max": sr_max,
                "degradation": degradation,
                "df_plus_kf": df + kf,
                "dist_df_plus_kf_3p2": abs((df + kf) - 3.2),
            }
        )
    return records


def _pick_per_sigma(
    records: list[dict],
    per_sigma: int,
    ref_n: int,
    stable: float,
    borderline: float,
) -> list[dict]:
    by_sigma: dict[float, list[dict]] = defaultdict(list)
    for rec in records:
        by_sigma[rec["sigma"]].append(rec)

    selected: list[dict] = []
    for sigma in sorted(by_sigma):
        sigma_recs = by_sigma[sigma]
        prepared: list[dict] = []
        for rec in sigma_recs:
            n_map: dict[int, float] = rec["n_success"]
            if ref_n in n_map:
                sr_ref = n_map[ref_n]
                n_ref = ref_n
            else:
                n_ref = rec["n_max"]
                sr_ref = n_map[n_ref]

            cls = _classify(sr_ref, stable, borderline)
            dist_to_stable = abs(sr_ref - stable)
            dist_to_borderline = abs(sr_ref - borderline)
            boundary_distance = min(dist_to_stable, dist_to_borderline)

            prepared.append(
                {
                    **rec,
                    "sr_ref": sr_ref,
                    "n_ref": n_ref,
                    "class": cls,
                    "dist_to_stable": dist_to_stable,
                    "dist_to_borderline": dist_to_borderline,
                    "boundary_distance": boundary_distance,
                }
            )

        pools: dict[str, list[dict]] = {"stable": [], "borderline": [], "unstable": []}
        for rec in prepared:
            pools[rec["class"]].append(rec)

        q_borderline = (per_sigma * 2) // 5
        q_stable = (per_sigma - q_borderline) // 2
        q_unstable = per_sigma - q_borderline - q_stable
        quota = {"stable": q_stable, "borderline": q_borderline, "unstable": q_unstable}

        def rank_key(rec: dict) -> tuple:
            if rec["class"] == "stable":
                class_distance = rec["dist_to_stable"]
            elif rec["class"] == "unstable":
                class_distance = rec["dist_to_borderline"]
            else:
                class_distance = rec["boundary_distance"]
            return (
                class_distance,
                rec["boundary_distance"],
                -rec["degradation"],
                rec["dist_df_plus_kf_3p2"],
                abs(rec["Df"] - 2.0),
            )

        sigma_pick: list[dict] = []
        for cls_name in ("stable", "borderline", "unstable"):
            pool = sorted(pools[cls_name], key=rank_key)
            sigma_pick.extend(pool[: quota[cls_name]])

        if len(sigma_pick) < per_sigma:
            already = {(r["Df"], r["kf"]) for r in sigma_pick}
            remainder = sorted(prepared, key=rank_key)
            for rec in remainder:
                key = (rec["Df"], rec["kf"])
                if key in already:
                    continue
                sigma_pick.append(rec)
                already.add(key)
                if len(sigma_pick) >= per_sigma:
                    break

        sigma_pick = sigma_pick[:per_sigma]
        for idx, rec in enumerate(sigma_pick, start=1):
            rec["rank_within_sigma"] = idx
            selected.append(rec)

    return selected


def _write_outputs(selected: list[dict], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_selected": len(selected),
        "selected": [
            {
                "sigma": r["sigma"],
                "Df": r["Df"],
                "kf": r["kf"],
                "n_ref": r["n_ref"],
                "sr_ref": r["sr_ref"],
                "class": r["class"],
                "boundary_distance": r["boundary_distance"],
                "degradation": r["degradation"],
                "df_plus_kf": r["df_plus_kf"],
                "dist_df_plus_kf_3p2": r["dist_df_plus_kf_3p2"],
                "rank_within_sigma": r["rank_within_sigma"],
            }
            for r in selected
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sigma",
                "Df",
                "kf",
                "n_ref",
                "sr_ref",
                "class",
                "boundary_distance",
                "degradation",
                "df_plus_kf",
                "dist_df_plus_kf_3p2",
                "rank_within_sigma",
            ],
        )
        writer.writeheader()
        for rec in payload["selected"]:
            writer.writerow(rec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="stability_sweep_summary.json path")
    parser.add_argument("--per-sigma", type=int, default=20)
    parser.add_argument("--reference-n", type=int, default=256)
    parser.add_argument("--sigma-values", type=str, default="1.1,1.5")
    parser.add_argument("--stable-threshold", type=float, default=STABLE_THRESHOLD)
    parser.add_argument(
        "--borderline-threshold", type=float, default=BORDERLINE_THRESHOLD
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/plausibility/step1_boundary_tuples.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_results/plausibility/step1_boundary_tuples.csv"),
    )
    args = parser.parse_args()

    allowed_sigmas = _parse_sigmas(args.sigma_values)
    rows = _load_rows(args.summary)
    records = _aggregate(rows, allowed_sigmas)
    selected = _pick_per_sigma(
        records,
        per_sigma=args.per_sigma,
        ref_n=args.reference_n,
        stable=args.stable_threshold,
        borderline=args.borderline_threshold,
    )

    _write_outputs(selected, args.output_json, args.output_csv)

    print(f"Selected {len(selected)} tuples total")
    by_sigma: dict[float, int] = defaultdict(int)
    by_class: dict[str, int] = defaultdict(int)
    for rec in selected:
        by_sigma[rec["sigma"]] += 1
        by_class[rec["class"]] += 1
    for sigma in sorted(by_sigma):
        print(f"  sigma={_fmt_float(sigma)}: {by_sigma[sigma]} tuples")
    print(
        "  class counts: " + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items()))
    )
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote CSV:  {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
