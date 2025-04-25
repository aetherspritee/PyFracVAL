"""Function to save simulation results."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import config as default_config

logger = logging.getLogger(__name__)

# Define start and end markers for the YAML block
YAML_START_MARKER = "### METADATA START (YAML) ###"
YAML_END_MARKER = "### METADATA END (YAML) ###"


def save_aggregate_data(
    coords: np.ndarray,
    radii: np.ndarray,
    iteration: int,
    metadata_to_save: dict[str, Any],
    sim_config: dict[str, Any],
    output_dir: str = "RESULTS",
):
    """
    Saves the coordinates and radii of the final aggregate.

    Args:
        coords: Nx3 NumPy array of coordinates.
        radii: N NumPy array of radii.
        iteration: The aggregate/iteration number (for filename).
        output_dir: The directory to save the results in.
    """
    n = coords.shape[0]
    if n == 0:
        logger.warning("Attempting to save empty aggregate data.")
        return

    # Create output directory if it doesn't exist
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Format parameters for filename (replace '.' with 'p' for floats)
    n_str = f"{n}"
    df_str = f"{sim_config['Df']:.2f}".replace(".", "p")
    kf_str = f"{sim_config['kf']:.2f}".replace(".", "p")
    rpg_str = f"{sim_config['rp_g']:.1f}".replace(".", "p")
    rpgstd_str = f"{sim_config['rp_gstd']:.2f}".replace(".", "p")
    seed_str = f"{sim_config.get('seed', 'N_A')}"  # Use N_A if no seed
    agg_str = f"{iteration}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    filename = (
        out_path
        / f"fracval_N{n_str}_Df{df_str}_kf{kf_str}_rpg{rpg_str}_rpgstd{rpgstd_str}_seed{seed_str}_agg{agg_str}_{timestamp}.dat"
    )

    # --- Prepare Metadata Header ---
    # metadata = {
    #     "generation_info": {
    #         "script_name": "PyFracVAL",  # Or get dynamically if needed
    #         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    #         "iteration": iteration,
    #     },
    #     "simulation_parameters": sim_config,  # Store the whole config dict
    #     "aggregate_properties": {
    #         "N_particles_actual": n,
    #         # Add calculated properties if available (e.g., final Rg, CM)
    #         # "final_Rg": calculated_rg,
    #     },
    #     "data_columns": {
    #         "column_1": {"name": "X", "unit": "arbitrary"},  # Example unit
    #         "column_2": {"name": "Y", "unit": "arbitrary"},
    #         "column_3": {"name": "Z", "unit": "arbitrary"},
    #         "column_4": {"name": "Radius", "unit": "arbitrary"},
    #     },
    # }

    # --- Convert metadata to YAML string ---
    try:
        # Use sort_keys=False to maintain insertion order (optional)
        # Use default_flow_style=False for block style (more readable)
        yaml_string = yaml.dump(
            metadata_to_save, sort_keys=False, default_flow_style=False, indent=2
        )
    except Exception as e:
        logger.error(f"Error converting metadata to YAML: {e}")
        # Fallback to simpler header or raise error?
        header_string = "# Error embedding YAML metadata\n"
        yaml_string = None  # Flag that YAML failed

    # --- Create Header String ---
    if yaml_string:
        header_string = "".join([f"# {line}\n" for line in yaml_string.splitlines()])
    else:
        # Fallback if YAML conversion failed
        header_string = "# ERROR: Could not generate YAML metadata.\n"

    # Combine data: X, Y, Z, R
    data_to_save = np.hstack((coords, radii.reshape(-1, 1)))

    # Save to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header_string)
            # Use a space delimiter, adjust format if needed
            np.savetxt(f, data_to_save, fmt="%18.10e", delimiter=" ")
        logger.info(f"Successfully saved aggregate data to: {filename}")
    except Exception as e:
        logger.error(f"Error saving results to {filename}: {e}")
