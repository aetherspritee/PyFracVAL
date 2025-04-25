# pyfracval/main_runner.py
"""
Core function to run the FracVAL simulation.
"""

import logging
import time
from typing import Any

import numpy as np

# Import necessary modules from your library
from . import particle_generation, utils
from .cca_agg import CCAggregator
from .pca_subclusters import Subclusterer
from .schemas import AggregateProperties, GenerationInfo, Metadata, SimulationParameters

logger = logging.getLogger(__name__)


def run_simulation(
    iteration: int,
    sim_config_dict: dict[str, Any],
    output_base_dir: str = "RESULTS",
    seed: int | None = None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    Runs one full aggregate generation based on the provided configuration.

    Args:
        iteration: The aggregate/iteration number (for filename).
        sim_config: Dictionary containing simulation parameters
                    (N, Df, kf, rp_g, rp_gstd, tol_ov, n_subcl_percentage, ext_case etc.).
        output_base_dir: Base directory to save results.
        seed: Optional random seed.

    Returns:
        tuple[bool, np.ndarray | None, np.ndarray | None]:
            (success_flag, final_coords, final_radii)
            Coords and radii are returned only on success.
    """
    logger.info(f"===== Starting Aggregate Generation {iteration} =====")
    # logger.info(
    #     f"Config: N={sim_config['N']}, Df={sim_config['Df']}, kf={sim_config['kf']}, "
    #     f"rp_g={sim_config['rp_g']}, rp_gstd={sim_config['rp_gstd']}"
    # )

    try:
        # Add seed to the dict if provided separately
        if seed is not None and "seed" not in sim_config_dict:
            sim_config_dict["seed"] = seed
        sim_params = SimulationParameters(**sim_config_dict)
        logger.info(f"Validated Config: {sim_params.model_dump_json(indent=2)}")
    except Exception as e:  # Catch Pydantic validation errors
        logger.error(f"Invalid simulation parameters provided: {e}", exc_info=True)
        return False, None, None

    start_time = time.time()

    if sim_params.seed is not None:
        np.random.seed(sim_params.seed)
        logger.info(f"Using random seed: {sim_params.seed}")

    # --- Use parameters from sim_config ---
    # Df = sim_config["Df"]
    # kf = sim_config["kf"]
    # rp_g = sim_config["rp_g"]
    # rp_gstd = sim_config["rp_gstd"]
    # tol_ov = sim_config["tol_ov"]
    # n_subcl_percentage = sim_config["n_subcl_percentage"]
    # ext_case = sim_config["ext_case"]
    # Add any other required parameters here

    # 1. Generate Initial Particle Radii
    try:
        initial_radii = particle_generation.lognormal_pp_radii(
            sim_params.rp_gstd,
            sim_params.rp_g,
            sim_params.N,
        )
        logger.info(
            f"Generated initial radii (Mean: {np.mean(initial_radii):.2f}, Std: {np.std(initial_radii):.2f})"
        )
    except ValueError as e:
        logger.error(f"Error generating radii: {e}")
        return False, None, None

    # 2. Shuffle Radii
    shuffled_radii = utils.shuffle_array(initial_radii.copy())

    # 3. PCA Subclustering
    logger.info("--- Starting PCA Subclustering ---")
    pca_start_time = time.time()
    subcluster_runner = Subclusterer(
        initial_radii=shuffled_radii,
        df=sim_params.Df,
        kf=sim_params.kf,
        tol_ov=sim_params.tol_ov,
        n_subcl_percentage=sim_params.n_subcl_percentage,
    )
    pca_success = subcluster_runner.run_subclustering()
    pca_end_time = time.time()
    logger.info(f"PCA Subclustering Time: {pca_end_time - pca_start_time:.2f} seconds")

    if not pca_success:
        logger.error(
            f"PCA Subclustering failed (Subcluster {subcluster_runner.number_clusters_processed + 1 if hasattr(subcluster_runner, 'number_clusters_processed') else 'N/A'})."
        )
        logger.error("Potential Causes/Fixes for PCA Failure:")
        logger.error(
            "  - Try reducing subcluster size with '--n-subcl-perc' (e.g., --n-subcl-perc 0.05)."
        )
        logger.error("  - Increase max attempts with '--max-attempts'.")
        logger.error(
            "  - Check simulation parameters (Df, kf, rp_gstd) - may be geometrically constrained."
        )
        logger.error("  - Try a different '--seed' for a new particle configuration.")
        return False, None, None

    num_clusters, not_able_pca, pca_coords_radii, pca_i_orden, _ = (
        subcluster_runner.get_results()
    )

    if not_able_pca or pca_coords_radii is None or pca_i_orden is None:
        logger.error("PCA returned invalid results.")
        return False, None, None

    # 4. Cluster-Cluster Aggregation
    logger.info("--- Starting Cluster-Cluster Aggregation ---")
    cca_start_time = time.time()
    cca_runner = CCAggregator(
        initial_coords=pca_coords_radii[:, :3],
        initial_radii=pca_coords_radii[:, 3],
        initial_i_orden=pca_i_orden,
        n_total=sim_params.N,
        df=sim_params.Df,
        kf=sim_params.kf,
        tol_ov=sim_params.tol_ov,
        ext_case=sim_params.ext_case,
    )
    cca_result = cca_runner.run_cca()
    cca_end_time = time.time()
    logger.info(f"CCA Aggregation Time: {cca_end_time - cca_start_time:.2f} seconds")

    if cca_result is None or cca_runner.not_able_cca:
        logger.error(
            "CCA Aggregation failed."
        )  # CCA errors are often harder to diagnose simply
        logger.error("Potential Causes/Fixes for CCA Failure:")
        logger.error("  - Increase max attempts with '--max-attempts'.")
        logger.error(
            "  - Check simulation parameters (Df, kf) - may be geometrically constrained."
        )
        logger.error("  - Ensure PCA stage produced valid subclusters.")
        logger.error("  - Try a different '--seed'.")
        return False, None, None

    # 5. Prepare Results
    final_coords, final_radii = cca_result
    n_actual = final_coords.shape[0]

    # Calculate final properties including Rg
    final_coords, final_radii = cca_result
    n_actual = final_coords.shape[0]
    final_rg = 0.0
    final_cm = [0.0, 0.0, 0.0]  # Use list default
    # ... (calculate final_rg, final_cm using utils.calculate_cluster_properties) ...
    if n_actual > 0:
        try:
            final_mass, final_rg, final_cm_arr, final_r_max = (
                utils.calculate_cluster_properties(
                    final_coords,
                    final_radii,
                    sim_params.Df,
                    sim_params.kf,
                )
            )
            logger.info(f"Final Aggregate Calculated Rg: {final_rg:.4f}")
            final_cm = final_cm_arr.tolist()  # Convert to list
        except Exception as e:
            logger.warning(f"Could not calculate final aggregate properties: {e}")
            final_rg = None  # Use None if calculation failed
            final_cm = None

    gen_info = GenerationInfo(iteration=iteration)  # Timestamp defaults to now
    agg_props = AggregateProperties(
        N_particles_actual=n_actual,
        radius_of_gyration=final_rg,
        center_of_mass=final_cm,
    )
    metadata_instance = Metadata(
        generation_info=gen_info,
        simulation_parameters=sim_params,  # Pass the validated parameters model
        aggregate_properties=agg_props,
    )

    # 6. Save Results (Optional, can be done by caller)
    # Make sure save_results can handle output directory argument
    metadata_instance.save_to_file(
        folderpath=output_base_dir,
        coords=final_coords,
        radii=final_radii,
    )

    end_time = time.time()
    logger.info(
        f"===== Aggregate {iteration} Finished Successfully ({end_time - start_time:.2f} seconds) ====="
    )
    return True, final_coords, final_radii
