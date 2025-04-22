# main.py
"""
Main script to run the FracVAL Cluster-Cluster Aggregation simulation.
"""

import time
from typing import Optional

import config  # Import parameters
import numpy as np
import particle_generation
import save_results
import utils
from cca_agg import CCAggregator
from pca_subclusters import Subclusterer


def run_simulation(iteration: int, seed: Optional[int] = None):
    """Runs one full aggregate generation."""
    print(
        f"\n===== Starting Aggregate Generation {iteration}/{config.QUANTITY_AGGREGATES} ====="
    )
    start_time = time.time()

    if seed is not None:
        np.random.seed(
            seed + iteration
        )  # Ensure different seed per iteration if base seed provided
        print(f"Using random seed: {seed + iteration}")

    # 1. Generate Initial Particle Radii
    try:
        initial_radii = particle_generation.lognormal_pp_radii(
            config.RP_GEOMETRIC_STD, config.RP_GEOMETRIC_MEAN, config.N
        )
    except ValueError as e:
        print(f"Error generating radii: {e}")
        return False  # Cannot proceed

    # 2. Shuffle Radii (like Fortran's randsample)
    shuffled_radii = utils.shuffle_array(initial_radii.copy())  # Shuffle a copy

    # 3. PCA Subclustering
    print("\n--- Starting PCA Subclustering ---")
    pca_start_time = time.time()
    subcluster_runner = Subclusterer(
        initial_radii=shuffled_radii,
        df=config.DF,
        kf=config.KF,
        tol_ov=config.TOL_OVERLAP,
        n_subcl_percentage=config.N_SUBCL_PERCENTAGE,
    )
    pca_success = subcluster_runner.run_subclustering()
    pca_end_time = time.time()
    print(f"PCA Subclustering Time: {pca_end_time - pca_start_time:.2f} seconds")

    if not pca_success:
        print("PCA Subclustering failed. Restarting generation...")
        return False  # Indicate failure for restart logic

    num_clusters, not_able_pca, pca_coords_radii, pca_i_orden, pca_radii = (
        subcluster_runner.get_results()
    )

    # Ensure results are valid before proceeding
    if (
        not_able_pca
        or pca_coords_radii is None
        or pca_i_orden is None
        or pca_radii is None
    ):
        print("PCA returned invalid results. Restarting generation...")
        return False

    # 4. Cluster-Cluster Aggregation
    print("\n--- Starting Cluster-Cluster Aggregation ---")
    cca_start_time = time.time()
    cca_runner = CCAggregator(
        initial_coords=pca_coords_radii[:, :3],
        initial_radii=pca_coords_radii[:, 3],  # Radii from PCA output data
        initial_i_orden=pca_i_orden,
        n_total=config.N,
        df=config.DF,
        kf=config.KF,
        tol_ov=config.TOL_OVERLAP,
        ext_case=config.EXT_CASE,
    )
    cca_result = cca_runner.run_cca()
    cca_end_time = time.time()
    print(f"CCA Aggregation Time: {cca_end_time - cca_start_time:.2f} seconds")

    if cca_result is None or cca_runner.not_able_cca:
        print("CCA Aggregation failed. Restarting generation...")
        return False  # Indicate failure

    # 5. Save Results
    final_coords, final_radii = cca_result
    save_results.save_aggregate_data(final_coords, final_radii, iteration)

    end_time = time.time()
    print(
        f"===== Aggregate {iteration} Finished Successfully ({end_time - start_time:.2f} seconds) ====="
    )
    return True  # Indicate success


if __name__ == "__main__":
    total_start_time = time.time()
    aggregates_generated = 0
    attempt = 0
    max_attempts = config.QUANTITY_AGGREGATES * 5  # Allow restarts

    # Optional fixed seed for reproducibility of the entire run sequence
    base_seed = 12345  # Set to None for non-reproducible runs

    while aggregates_generated < config.QUANTITY_AGGREGATES and attempt < max_attempts:
        attempt += 1
        success = run_simulation(aggregates_generated + 1, seed=base_seed)
        if success:
            aggregates_generated += 1
        else:
            print(f"--- Attempt {attempt} failed, trying again ---")
            time.sleep(0.5)  # Small pause before restart

    total_end_time = time.time()
    print("\n--------------------------------------------------")
    if aggregates_generated == config.QUANTITY_AGGREGATES:
        print(f"Finished generating {aggregates_generated} aggregates successfully.")
    else:
        print(f"Failed to generate all aggregates after {max_attempts} attempts.")
    print(f"Total Simulation Time: {total_end_time - total_start_time:.2f} seconds")
    print("--------------------------------------------------")
