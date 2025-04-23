# pyfracval/main_runner.py
"""
Core function to run the FracVAL simulation.
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Import necessary modules from your library
from . import particle_generation, save_results, utils
from .cca_agg import CCAggregator
from .pca_subclusters import Subclusterer


def run_simulation(
    iteration: int,
    sim_config: Dict[str, Any],
    output_base_dir: str = "RESULTS",
    seed: Optional[int] = None,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Runs one full aggregate generation based on the provided configuration.

    Args:
        iteration: The aggregate/iteration number (for filename).
        sim_config: Dictionary containing simulation parameters
                    (N, Df, kf, rp_g, rp_gstd, tol_ov, n_subcl_percentage, ext_case etc.).
        output_base_dir: Base directory to save results.
        seed: Optional random seed.

    Returns:
        Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
            (success_flag, final_coords, final_radii)
            Coords and radii are returned only on success.
    """
    print(f"\n===== Starting Aggregate Generation {iteration} =====")
    print(
        f"Config: N={sim_config['N']}, Df={sim_config['Df']}, kf={sim_config['kf']}, "
        f"rp_g={sim_config['rp_g']}, rp_gstd={sim_config['rp_gstd']}"
    )
    start_time = time.time()

    if seed is not None:
        np.random.seed(seed)  # Use the provided seed directly for this run
        print(f"Using random seed: {seed}")

    # --- Use parameters from sim_config ---
    N = sim_config["N"]
    Df = sim_config["Df"]
    kf = sim_config["kf"]
    rp_g = sim_config["rp_g"]
    rp_gstd = sim_config["rp_gstd"]
    tol_ov = sim_config["tol_ov"]
    n_subcl_percentage = sim_config["n_subcl_percentage"]
    ext_case = sim_config["ext_case"]
    # Add any other required parameters here

    # 1. Generate Initial Particle Radii
    try:
        initial_radii = particle_generation.lognormal_pp_radii(
            rp_gstd,
            rp_g,
            N,  # Seed handled above
        )
    except ValueError as e:
        print(f"Error generating radii: {e}")
        return False, None, None

    # 2. Shuffle Radii
    shuffled_radii = utils.shuffle_array(initial_radii.copy())

    # 3. PCA Subclustering
    print("\n--- Starting PCA Subclustering ---")
    pca_start_time = time.time()
    subcluster_runner = Subclusterer(
        initial_radii=shuffled_radii,
        df=Df,  # Consider passing relaxed df/kf here if needed for PCA stability
        kf=kf,
        tol_ov=tol_ov,
        n_subcl_percentage=n_subcl_percentage,
    )
    pca_success = subcluster_runner.run_subclustering()
    pca_end_time = time.time()
    print(f"PCA Subclustering Time: {pca_end_time - pca_start_time:.2f} seconds")

    if not pca_success:
        print("PCA Subclustering failed.")
        return False, None, None

    num_clusters, not_able_pca, pca_coords_radii, pca_i_orden, _ = (
        subcluster_runner.get_results()
    )

    if not_able_pca or pca_coords_radii is None or pca_i_orden is None:
        print("PCA returned invalid results.")
        return False, None, None

    # 4. Cluster-Cluster Aggregation
    print("\n--- Starting Cluster-Cluster Aggregation ---")
    cca_start_time = time.time()
    cca_runner = CCAggregator(
        initial_coords=pca_coords_radii[:, :3],
        initial_radii=pca_coords_radii[:, 3],
        initial_i_orden=pca_i_orden,
        n_total=N,
        df=Df,  # Use target Df/kf for CCA
        kf=kf,
        tol_ov=tol_ov,
        ext_case=ext_case,
    )
    cca_result = cca_runner.run_cca()
    cca_end_time = time.time()
    print(f"CCA Aggregation Time: {cca_end_time - cca_start_time:.2f} seconds")

    if cca_result is None or cca_runner.not_able_cca:
        print("CCA Aggregation failed.")
        return False, None, None

    # 5. Prepare Results
    final_coords, final_radii = cca_result

    # 6. Save Results (Optional, can be done by caller)
    # Make sure save_results can handle output directory argument
    save_results.save_aggregate_data(
        final_coords, final_radii, iteration, output_dir=output_base_dir
    )

    end_time = time.time()
    print(
        f"===== Aggregate {iteration} Finished Successfully ({end_time - start_time:.2f} seconds) ====="
    )
    return True, final_coords, final_radii
