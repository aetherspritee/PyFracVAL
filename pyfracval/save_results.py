"""Function to save simulation results."""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_aggregate_data(
    coords: np.ndarray,
    radii: np.ndarray,
    iteration: int,
    sim_config: dict,
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

    # Combine data: X, Y, Z, R
    data_to_save = np.hstack((coords, radii.reshape(-1, 1)))

    # Save to file
    try:
        # Use a space delimiter, adjust format if needed
        np.savetxt(filename, data_to_save, fmt="%18.10e", delimiter=" ")
        logger.info(f"Successfully saved aggregate data to: {filename}")
    except Exception as e:
        logger.error(f"Error saving results to {filename}: {e}")
