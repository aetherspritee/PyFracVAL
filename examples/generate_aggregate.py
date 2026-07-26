#!/usr/bin/env python3
"""Minimal example of using pyfracval as a library.

This is the simple on-ramp for writing your own custom generation script -
for anything more involved (parameter sweeps, batch generation, comparing
methods), see benchmarks/ instead.

Usage:
    devenv shell -- uv run python examples/generate_aggregate.py
"""

import logging

from pyfracval.config import RunConfig
from pyfracval.logs import create_logger
from pyfracval.main_runner import run_simulation


def main() -> None:
    create_logger(logging.INFO)

    # RunConfig holds everything a generation run needs, with sensible
    # defaults for anything you don't set. You can also load one from a
    # TOML/YAML/JSON file instead of constructing it in code:
    #
    #   config = RunConfig.from_file("my_config.toml")
    #
    config = RunConfig.model_validate(
        {
            "simulation": {
                "N": 128,
                "Df": 1.8,
                "kf": 1.0,
                "rp_gstd": 1.5,
            },
        }
    )

    sim = config.simulation
    sim_config_dict = {
        "N": sim.N,
        "Df": sim.Df,
        "kf": sim.kf,
        "rp_g": sim.rp_g,
        "rp_gstd": sim.rp_gstd,
        "tol_ov": sim.tol_ov,
        "n_subcl_percentage": sim.n_subcl_percentage,
        "ext_case": sim.ext_case,
        **config.algorithm.model_dump(),
    }

    success, coords, radii = run_simulation(
        iteration=1,
        sim_config_dict=sim_config_dict,
        output_base_dir="RESULTS",
        seed=42,
    )

    if success:
        print(f"Generated an aggregate with {coords.shape[0]} particles.")
        print("Saved to RESULTS/")
    else:
        print("Generation failed - try a different (Df, kf, rp_gstd) combination.")
        print("See docs/source/experiments.md for what tends to work.")


if __name__ == "__main__":
    main()
