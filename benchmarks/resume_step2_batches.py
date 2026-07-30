#!/usr/bin/env python3
"""Resume Step2 feature matrix in manageable Marvin batches.

- Reads boundary tuples from step1 selection JSON
- Detects completed outputs in benchmark_results/plausibility/step2_feature_matrix
- Generates per-batch orchestrator configs for remaining tuples only
- Executes each batch sequentially via benchmarks/run.py unified
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_selected(boundary_json: Path) -> list[dict]:
    data = json.loads(boundary_json.read_text(encoding="utf-8"))
    selected = data.get("selected", [])
    if not isinstance(selected, list):
        raise ValueError("boundary JSON missing selected list")
    return selected


def _tuple_key(index_1_based: int, tup: dict) -> str:
    sigma = str(float(tup["sigma"])).replace(".", "p")
    df = str(float(tup["Df"])).replace(".", "p")
    kf = str(float(tup["kf"])).replace(".", "p")
    cls = str(tup.get("class", "unknown"))
    return f"t{index_1_based:02d}_s{sigma}_df{df}_kf{kf}_{cls}"


def _completed_tuple_ids(output_root: Path) -> set[str]:
    done: set[str] = set()
    if not output_root.exists():
        return done
    for p in output_root.glob("t*_*/unified_N1024_rep1.json"):
        name = p.parent.name
        # format: t01_<mode>_sX_dfY_kfZ_<class>
        if not name.startswith("t"):
            continue
        parts = name.split("_")
        if len(parts) < 6:
            continue
        t_id = parts[0]
        # sigma token is expected as s<digit...>, avoid matching "soft" mode token
        s_idx = None
        for i, tok in enumerate(parts):
            if tok.startswith("s") and len(tok) > 1 and tok[1].isdigit():
                s_idx = i
                break
        if s_idx is None:
            continue
        sig = "_".join([t_id] + parts[s_idx:])
        done.add(sig)
    return done


def _write_batch_config(
    config_path: Path,
    tuples: list[tuple[int, dict]],
    scheduler: str,
    output_root: str,
) -> None:
    feature_modes = [
        (
            "vanilla_single",
            {
                "cca_retry_rotation_mode": "single",
                "cca_sticking_method": "fibonacci",
                "cca_gamma_expansion_enabled": False,
                "cca_pair_feasibility_filter": "none",
                "cca_soft_relaxation_enabled": False,
                "densify_enabled": False,
            },
        ),
        (
            "retry_alternate",
            {
                "cca_retry_rotation_mode": "alternate",
                "cca_sticking_method": "fibonacci",
                "cca_gamma_expansion_enabled": False,
                "cca_pair_feasibility_filter": "none",
                "cca_soft_relaxation_enabled": False,
                "densify_enabled": False,
            },
        ),
        (
            "retry_coarse_to_fine",
            {
                "cca_retry_rotation_mode": "coarse_to_fine",
                "cca_sticking_method": "fibonacci",
                "cca_gamma_expansion_enabled": False,
                "cca_pair_feasibility_filter": "none",
                "cca_soft_relaxation_enabled": False,
                "densify_enabled": False,
            },
        ),
        (
            "gamma_bv",
            {
                "cca_retry_rotation_mode": "single",
                "cca_sticking_method": "fibonacci",
                "cca_gamma_expansion_enabled": True,
                "cca_gamma_expansion_step": 0.02,
                "cca_gamma_expansion_max_factor": 1.05,
                "cca_gamma_expansion_mass_exponent": -0.75,
                "cca_gamma_expansion_max_attempts": 3,
                "cca_pair_feasibility_filter": "bounding_volume",
                "cca_bv_deep_penetration_factor": 0.8,
                "cca_soft_relaxation_enabled": False,
                "densify_enabled": False,
            },
        ),
        (
            "fft_docking",
            {
                "cca_retry_rotation_mode": "single",
                "cca_sticking_method": "fft_docking",
                "cca_fft_grid_size": 64,
                "cca_fft_num_rotations": 70,
                "cca_fft_top_k_peaks": 10,
                "cca_fft_gamma_tolerance": 0.10,
                "cca_gamma_expansion_enabled": False,
                "cca_pair_feasibility_filter": "none",
                "cca_soft_relaxation_enabled": False,
                "densify_enabled": False,
            },
        ),
        (
            "soft_relaxation_fallback",
            {
                "cca_retry_rotation_mode": "single",
                "cca_sticking_method": "fibonacci",
                "cca_gamma_expansion_enabled": False,
                "cca_pair_feasibility_filter": "none",
                "cca_soft_relaxation_enabled": True,
                "cca_soft_relaxation_fallback_only": True,
                "cca_soft_relaxation_k_repulsion": 10.0,
                "cca_soft_relaxation_k_gamma": 1.0,
                "cca_soft_relaxation_gamma_tolerance": 0.05,
                "densify_enabled": False,
            },
        ),
        (
            "densify_radial",
            {
                "cca_retry_rotation_mode": "single",
                "cca_sticking_method": "fibonacci",
                "cca_gamma_expansion_enabled": False,
                "cca_pair_feasibility_filter": "none",
                "cca_soft_relaxation_enabled": False,
                "densify_enabled": True,
                "densify_source_df": 2.0,
                "densify_source_kf": 1.0,
                "densify_method": "radial",
                "densify_rtol": 0.05,
                "densify_max_push_iters": 50,
            },
        ),
    ]

    def tv(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            s = f"{v:.12g}"
            return s if ("." in s or "e" in s or "E" in s) else s + ".0"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(tv(x) for x in v) + "]"
        return '"' + str(v).replace('"', '\\"') + '"'

    lines = []
    lines.append("[defaults]")
    defaults = {
        "execution_mode": "sequential",
        "output_root": output_root,
        "n_values": [512, 1024],
        "repeats": 1,
        "n_aggregates": 1,
        "warmup_tasks": 1,
        "seed_start": 1431354440,
        "trial_timeout": 300,
        "local_workers": 4,
        "profile": False,
    }
    for k in sorted(defaults):
        lines.append(f"{k} = {tv(defaults[k])}")
    lines.append("")
    lines.append("[defaults.simulation]")
    for k, v in {
        "rp_g": 1.0,
        "tol_ov": 1.0e-6,
        "n_subcl_percentage": 0.1,
        "ext_case": 0,
    }.items():
        lines.append(f"{k} = {tv(v)}")
    lines.append("")
    lines.append("[defaults.algorithm]")
    for k, v in {
        "use_cca_incremental_overlap": True,
        "cca_incremental_full_sync_period": 8,
        "cca_candidate_policy": "leaf_hybrid",
        "cca_score_topk_per_class": 32,
        "cca_retry_rotation_mode": "single",
        "cca_retry_escalate_after": 120,
        "cca_dual_jitter_interval": 5,
        "cca_dual_jitter_deg": 8.0,
        "cca_coarse_sweep_steps": 10,
        "cca_coarse_spin_anchor_steps": 6,
        "cca_coarse_spin_moving_steps": 6,
        "cca_coarse_fine_coarse_fraction": 0.67,
        "cca_coarse_fine_spin_deg": 12.0,
        "profile_cca_retry_modes": True,
    }.items():
        lines.append(f"{k} = {tv(v)}")
    lines.append("")

    for idx, tup in tuples:
        sigma = float(tup["sigma"])
        df = float(tup["Df"])
        kf = float(tup["kf"])
        cls = str(tup.get("class", "unknown"))
        for mode_name, mode_cfg in feature_modes:
            run_name = (
                f"t{idx:02d}_{mode_name}_"
                f"s{str(sigma).replace('.', 'p')}_"
                f"df{str(df).replace('.', 'p')}_"
                f"kf{str(kf).replace('.', 'p')}_"
                f"{cls}"
            )
            lines.append("[[runs]]")
            lines.append(f"name = {tv(run_name)}")
            lines.append(f"scheduler = {tv(scheduler)}")
            lines.append("[runs.simulation]")
            lines.append(f"Df = {tv(df)}")
            lines.append(f"kf = {tv(kf)}")
            lines.append(f"rp_gstd = {tv(sigma)}")
            lines.append("[runs.algorithm]")
            for k in sorted(mode_cfg):
                lines.append(f"{k} = {tv(mode_cfg[k])}")
            lines.append("")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary-json",
        type=Path,
        default=Path("benchmark_results/plausibility/step1_boundary_tuples.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark_results/plausibility/step2_feature_matrix"),
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("configs/plausibility_step2_batches"),
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="tcp://marvin.bv.e-technik.tu-dortmund.de:8786",
    )
    parser.add_argument("--max-batches", type=int, default=999)
    args = parser.parse_args()

    selected = _load_selected(args.boundary_json)
    completed = _completed_tuple_ids(args.output_root)

    pending: list[tuple[int, dict]] = []
    for i, tup in enumerate(selected, start=1):
        sig = _tuple_key(i, tup)
        if sig not in completed:
            pending.append((i, tup))

    print(f"Selected tuples: {len(selected)}")
    print(f"Completed tuples: {len(selected) - len(pending)}")
    print(f"Pending tuples: {len(pending)}")
    if not pending:
        print("Nothing to run.")
        return 0

    batches = [
        pending[i : i + args.batch_size]
        for i in range(0, len(pending), args.batch_size)
    ]
    batches = batches[: args.max_batches]

    repo_root = Path(__file__).resolve().parent.parent
    for bidx, bt in enumerate(batches, start=1):
        cfg = args.batch_dir / f"batch_{bidx:03d}.toml"
        _write_batch_config(cfg, bt, args.scheduler, str(args.output_root))
        print(f"\n=== Running batch {bidx}/{len(batches)} with {len(bt)} tuples ===")
        cmd = [
            sys.executable,
            str(repo_root / "benchmarks" / "run.py"),
            "unified",
            "--config",
            str(cfg),
        ]
        result = subprocess.call(cmd, cwd=str(repo_root))
        print(f"Batch {bidx} exit code: {result}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
