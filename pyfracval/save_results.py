# save_results.py
"""Function to save simulation results."""

import numpy as np
from pathlib import Path
import config

def save_aggregate_data(coords: np.ndarray, radii: np.ndarray, iteration: int, output_dir: str = "RESULTS"):
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
        print("Warning: Attempting to save empty aggregate data.")
        return

    # Create output directory if it doesn't exist
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Format filenames (using 8 digits zero-padding like Fortran example)
    n_str = f"{n:08d}"
    iter_str = f"{iteration:08d}"
    filename = out_path / f"N_{n_str}_Agg_{iter_str}.dat"

    # Combine data: X, Y, Z, R
    data_to_save = np.hstack((coords, radii.reshape(-1, 1)))

    # Save to file
    try:
        # Use a space delimiter, adjust format if needed
        np.savetxt(filename, data_to_save, fmt='%18.10e', delimiter=' ')
        print(f"Successfully saved aggregate data to: {filename}")
    except Exception as e:
        print(f"Error saving results to {filename}: {e}")