#!/usr/bin/env python3
"""Build Step-2 orchestrator config from selected boundary tuples.

Input: JSON produced by benchmarks/select_boundary_tuples.py
Output: TOML orchestrator config consumed by benchmarks/run.py unified
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.12g}"
        if "e" in text or "E" in text:
            return text
        if "." not in text:
            return text + ".0"
        return text
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return '"' + str(value).replace('"', '\\"') + '"'


def _table_lines(prefix: str, mapping: dict) -> list[str]:
    lines = [f"[{prefix}]"]
    for key in sorted(mapping):
        lines.append(f"{key} = {_toml_value(mapping[key])}")
    return lines


def _run_block(name: str, scheduler: str, sim: dict, algo: dict) -> list[str]:
    lines = ["[[runs]]"]
    lines.append(f"name = {_toml_value(name)}")
    lines.append(f"scheduler = {_toml_value(scheduler)}")
    lines.extend(_table_lines("runs.simulation", sim))
    lines.extend(_table_lines("runs.algorithm", algo))
    return lines


def _load_selected(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected", [])
    if not isinstance(selected, list) or not selected:
        raise ValueError("No selected tuples found in boundary JSON")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary-json",
        type=Path,
        default=Path("benchmark_results/plausibility/step1_boundary_tuples.json"),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="tcp://marvin.bv.e-technik.tu-dortmund.de:8786",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-aggregates", type=int, default=8)
    parser.add_argument("--trial-timeout", type=int, default=600)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/plausibility_step2_feature_matrix.toml"),
    )
    args = parser.parse_args()

    selected = _load_selected(args.boundary_json)

    defaults = {
        "execution_mode": "sequential",
        "output_root": "benchmark_results/plausibility/step2_feature_matrix",
        "n_values": [512, 1024],
        "repeats": args.repeats,
        "n_aggregates": args.n_aggregates,
        "warmup_tasks": 1,
        "seed_start": 1431354440,
        "trial_timeout": args.trial_timeout,
        "local_workers": 4,
        "profile": False,
    }

    default_sim = {
        "rp_g": 1.0,
        "tol_ov": 1.0e-6,
        "n_subcl_percentage": 0.1,
        "ext_case": 0,
    }

    default_algo = {
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
    }

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

    lines: list[str] = []
    lines.append("# Step 2 feature matrix generated from boundary tuples")
    lines.append(f"# Source tuples: {args.boundary_json}")
    lines.append(f"# Total tuples: {len(selected)}")
    lines.append("")

    lines.extend(_table_lines("defaults", defaults))
    lines.append("")
    lines.extend(_table_lines("defaults.simulation", default_sim))
    lines.append("")
    lines.extend(_table_lines("defaults.algorithm", default_algo))
    lines.append("")

    for idx, tup in enumerate(selected, start=1):
        sigma = float(tup["sigma"])
        df = float(tup["Df"])
        kf = float(tup["kf"])
        cls = str(tup.get("class", "unknown"))
        for mode_name, mode_algo in feature_modes:
            run_name = (
                f"t{idx:02d}_{mode_name}_"
                f"s{str(sigma).replace('.', 'p')}_"
                f"df{str(df).replace('.', 'p')}_"
                f"kf{str(kf).replace('.', 'p')}_"
                f"{cls}"
            )
            sim = {
                "Df": df,
                "kf": kf,
                "rp_gstd": sigma,
            }
            lines.extend(_run_block(run_name, args.scheduler, sim, mode_algo))
            lines.append("")

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"Generated config: {args.output_config}")
    print(f"Boundary tuples: {len(selected)}")
    print(f"Feature modes per tuple: {len(feature_modes)}")
    print(f"Total runs in config: {len(selected) * len(feature_modes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
