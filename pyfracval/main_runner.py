"""Core function to run the FracVAL simulation."""

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
    Run one full FracVAL aggregate generation (PCA + CCA).

    Orchestrates the simulation pipeline:
    1. Validates input parameters using `SimulationParameters`.
    2. Sets random seed.
    3. Generates initial particle radii (lognormal distribution).
    4. Shuffles radii.
    5. Performs PCA subclustering using `Subclusterer`.
    6. Performs CCA aggregation using `CCAggregator` on the PCA results.
    7. Calculates final aggregate properties (Rg, CM).
    8. Saves results (metadata + data) using `Metadata.save_to_file`.
    9. Provides enhanced error messages and suggestions on failure.

    Parameters
    ----------
    iteration : int
        The iteration number (e.g., for generating multiple aggregates),
        used mainly for output filenames and metadata.
    sim_config_dict : dict[str, Any]
        Dictionary containing simulation parameters conforming to
        `SimulationParameters` schema (N, Df, kf, rp_g, rp_gstd, etc.).
    output_base_dir : str, optional
        Base directory to save the output `.dat` file, by default "RESULTS".
    seed : int | None, optional
        Random seed for reproducibility, by default None (time-based).

    Returns
    -------
    tuple[bool, np.ndarray | None, np.ndarray | None]
        A tuple containing:
            - success_flag (bool): True if the simulation completed successfully,
              False otherwise.
            - final_coords (np.ndarray | None): Nx3 array of coordinates if
              successful, None otherwise.
            - final_radii (np.ndarray | None): N array of radii if successful,
              None otherwise.

    """

    logger.info(f"===== Starting Aggregate Generation {iteration} =====")

    try:
        if seed is not None and "seed" not in sim_config_dict:
            sim_config_dict["seed"] = seed
        sim_params = SimulationParameters(**sim_config_dict)
        logger.info(f"Validated Config: {sim_params.model_dump_json(indent=2)}")
    except Exception as e:
        logger.error(f"Invalid simulation parameters provided: {e}", exc_info=True)
        return False, None, None

    start_time = time.time()

    if sim_params.seed is not None:
        np.random.seed(sim_params.seed)
        logger.info(f"Using random seed: {sim_params.seed}")

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

    # --- Enhanced PCA Failure Handling ---
    if not pca_success or subcluster_runner.not_able_pca:
        failed_subcluster_num = getattr(
            subcluster_runner, "number_clusters_processed", "N/A"
        )
        if failed_subcluster_num != "N/A":
            failed_subcluster_num += 1  # Adjust to 1-based index

        logger.error(
            f"PCA Subclustering failed (Failed on Subcluster {failed_subcluster_num})."
        )
        # logger.error("PCA Failure Diagnosis & Suggestions:")
        # logger.error(
        #     f"  - The current target parameters (Df={sim_params.Df}, kf={sim_params.kf}) might be geometrically challenging during PCA."
        # )
        # logger.error("  - Common Fixes for PCA Failure (Try in order):")
        # logger.error(
        #     f"    1. Increase target kf: Try `--kf {sim_params.kf + 0.1:.1f}` or `--kf {sim_params.kf + 0.2:.1f}`. (Often helps if Gamma or Sticking fails)."
        #     # Decreasing kf is less common for PCA failures seen so far, but possible if Gamma calc itself fails
        #     # logger.error(f"    Alt. Decrease target kf: Try `--kf {max(0.1, sim_params.kf - 0.1):.1f}` (May help if Gamma calculation fails).")
        # )
        # logger.error(
        #     f"    2. Increase target Df: Try `--df {sim_params.Df + 0.05:.2f}` or `--df {sim_params.Df + 0.1:.1f}`."
        # )
        # logger.error(
        #     f"    3. Increase overlap tolerance: Try `--tol-ov 1e-5` or `--tol-ov 1e-4` (If failure is during sticking/rotation)."
        # )
        # logger.error(
        #     f"    4. Reduce subcluster size: Try `--n-subcl-perc {max(0.02, sim_params.n_subcl_percentage * 0.8):.2f}` (e.g., 0.08 if currently 0.1)."
        # )
        # logger.error(
        #     "    5. Try a different random seed: Add `--seed <number>` or change the existing seed."
        # )
        # logger.error(
        #     "    6. Increase max attempts: Use `--max-attempts <number>` (e.g., 10)."
        # )
        return False, None, None
    # --- End Enhanced PCA Handling ---

    # Retrieve results only if PCA succeeded
    num_clusters, not_able_pca_flag, pca_coords_radii, pca_i_orden, _ = (
        subcluster_runner.get_results()
    )
    # Double check, though should be caught above
    if not_able_pca_flag or pca_coords_radii is None or pca_i_orden is None:
        logger.error("PCA returned invalid results despite reporting success.")
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

    # --- Enhanced CCA Failure Handling ---
    if cca_result is None or cca_runner.not_able_cca:
        logger.error("CCA Aggregation failed.")
        # logger.error("CCA Failure Diagnosis & Suggestions:")
        # logger.error(
        #     "  - Failure often occurs during cluster pairing or sticking due to geometric constraints from target Df/kf."
        # )
        # logger.error(
        #     f"  - Check logs for WARNings about 'RELAXED condition' during pairing or 'No initial candidates found' / 'Sticking failed' during sticking."
        # )
        # logger.error("  - Common Fixes for CCA Failure (Try in order):")
        # logger.error(
        #     f"    1. Increase target kf: Try `--kf {sim_params.kf + 0.1:.1f}` or `--kf {sim_params.kf + 0.2:.1f}`. (Often helps relax pairing/sticking)."
        # )
        # logger.error(
        #     f"    2. Increase target Df: Try `--df {sim_params.Df + 0.05:.2f}` or `--df {sim_params.Df + 0.1:.1f}`."
        # )
        # logger.error(
        #     f"    3. Check CCA Pairing Factor: If warnings about RELAXED condition were frequent, the factor in `cca_agg.py` might need adjustment (currently hardcoded)."
        # )
        # logger.error(
        #     "    4. Try a different random seed: Add `--seed <number>` or change the existing seed (affects PCA structure)."
        # )
        # logger.error(
        #     "    5. Increase max attempts: Use `--max-attempts <number>` (e.g., 10)."
        # )
        return False, None, None
    # --- End Enhanced CCA Handling ---

    # 5. Prepare Results (Only if CCA succeeded)
    final_coords, final_radii = cca_result
    n_actual = final_coords.shape[0]

    # Calculate final properties including Rg
    final_rg = 0.0
    final_cm = [0.0, 0.0, 0.0]  # Use list default
    if n_actual > 0:
        try:
            # Pass target Df/kf for final property calculation consistency
            final_mass, final_rg_val, final_cm_arr, final_r_max = (
                utils.calculate_cluster_properties(
                    final_coords,
                    final_radii,
                    sim_params.Df,
                    sim_params.kf,
                )
            )
            # Handle potential None return from calculate_rg inside calculate_cluster_properties
            final_rg = final_rg_val if final_rg_val is not None else 0.0
            final_cm = (
                final_cm_arr.tolist() if final_cm_arr is not None else [0.0, 0.0, 0.0]
            )
            logger.info(f"Final Aggregate Calculated Rg: {final_rg:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate final aggregate properties: {e}")
            final_rg = None  # Use None if calculation failed
            final_cm = None

    # Create Metadata
    gen_info = GenerationInfo(iteration=iteration)
    agg_props = AggregateProperties(
        N_particles_actual=n_actual,
        radius_of_gyration=final_rg,
        center_of_mass=final_cm,
    )
    metadata_instance = Metadata(
        generation_info=gen_info,
        simulation_parameters=sim_params,
        aggregate_properties=agg_props,
    )

    # 6. Save Results
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
