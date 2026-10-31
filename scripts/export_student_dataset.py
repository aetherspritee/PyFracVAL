#!/usr/bin/env python3
"""Export a set of aggregate ``.dat`` files for the dimensional-analysis project.

Rescales every aggregate to ``rp_g = 1.0`` (the downstream pipeline slices
and renders in absolute lengths, so a file written at ``rp_g = 100`` would
produce zero-thickness slices and 100x images), drops files the catalog
validator flagged, lays them out as ``sigma_X__Df_Y__N_Z/`` folders and
writes a ``cluster_index.csv`` next to them.

Usage:
    devenv shell -- uv run python scripts/export_student_dataset.py \\
        --src benchmark_results/paper_extended_map \\
        --dst /path/to/dimensional-analysis/data/extended_map \\
        --exclude validation_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

INDEX_FIELDS = [
    "config",
    "sigma",
    "Df",
    "N",
    "kf",
    "attempt",
    "seed",
    "success",
    "n_particles",
    "overlap_ok",
    "filepath",
]


def read_dat(path: Path) -> tuple[dict, np.ndarray]:
    """Return the YAML header as a dict and the Nx4 (x, y, z, r) data."""
    header_lines = []
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            header_lines.append(line[2:])
    meta = yaml.safe_load("".join(header_lines))
    data = np.loadtxt(path, comments="#", ndmin=2)
    return meta, data


def rescale(meta: dict, data: np.ndarray) -> tuple[dict, np.ndarray]:
    """Divide every length in header and data by ``rp_g``."""
    sim = meta["simulation_parameters"]
    scale = float(sim["rp_g"])
    if scale == 1.0:
        return meta, data
    agg = meta["aggregate_properties"]
    agg["radius_of_gyration"] = agg["radius_of_gyration"] / scale
    agg["center_of_mass"] = [c / scale for c in agg["center_of_mass"]]
    if agg.get("measured_rg") is not None:
        agg["measured_rg"] = agg["measured_rg"] / scale
    sim["rp_g"] = 1.0
    return meta, data / scale


def write_dat(path: Path, meta: dict, data: np.ndarray) -> None:
    header = yaml.safe_dump(meta, sort_keys=False, default_flow_style=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("".join(f"# {line}\n" for line in header.splitlines()))
        np.savetxt(fh, data, fmt="%24.17e", delimiter=" ")


def fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument(
        "--exclude",
        type=Path,
        help="validation_report.json; its overlapping files are skipped",
    )
    args = ap.parse_args()

    excluded: set[str] = set()
    if args.exclude:
        report = json.loads(args.exclude.read_text())
        excluded = {Path(entry["file"]).name for entry in report["overlapping"]}

    files = sorted(args.src.rglob("fracval_*.dat"))
    rows = []
    skipped = 0
    for i, src in enumerate(files, 1):
        meta, data = read_dat(src)
        if (
            src.name in excluded
            or meta["aggregate_properties"].get("overlap_ok") is False
        ):
            skipped += 1
            continue
        old_rpg = f"rpg{meta['simulation_parameters']['rp_g']:.1f}".replace(".", "p")
        meta, data = rescale(meta, data)
        sim = meta["simulation_parameters"]
        folder = (
            args.dst / f"sigma_{fmt(sim['rp_gstd'])}__Df_{fmt(sim['Df'])}__N_{sim['N']}"
        )
        dst = folder / src.name.replace(old_rpg, "rpg1p0")
        write_dat(dst, meta, data)
        rows.append(
            {
                "config": args.dst.name,
                "sigma": sim["rp_gstd"],
                "Df": sim["Df"],
                "N": sim["N"],
                "kf": sim["kf"],
                "attempt": meta["generation_info"]["iteration"],
                "seed": sim["seed"],
                "success": True,
                "n_particles": meta["aggregate_properties"]["N_particles_actual"],
                "overlap_ok": meta["aggregate_properties"].get("overlap_ok"),
                "filepath": dst.relative_to(args.dst.parent).as_posix(),
            }
        )
        if i % 500 == 0:
            print(f"...{i}/{len(files)}", flush=True)

    with (args.dst / "cluster_index.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} files to {args.dst}, skipped {skipped}")


def _self_check() -> None:
    meta = {
        "simulation_parameters": {"rp_g": 100.0},
        "aggregate_properties": {
            "radius_of_gyration": 250.0,
            "center_of_mass": [100.0, 0.0, -50.0],
            "measured_rg": 240.0,
        },
    }
    meta, data = rescale(meta, np.array([[100.0, 200.0, 300.0, 120.0]]))
    assert meta["simulation_parameters"]["rp_g"] == 1.0
    assert meta["aggregate_properties"]["radius_of_gyration"] == 2.5
    assert meta["aggregate_properties"]["center_of_mass"] == [1.0, 0.0, -0.5]
    assert np.allclose(data, [[1.0, 2.0, 3.0, 1.2]])


if __name__ == "__main__":
    _self_check()
    main()
