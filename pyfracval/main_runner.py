"""Core function to run the FracVAL simulation."""

import logging
import time
from typing import Any

import numpy as np

# Import necessary modules from your library
from . import config, particle_generation, utils
from .densify import densify_aggregate
from .cca_agg import CCAggregator
from .pca_subclusters import Subclusterer
from .schemas import AggregateProperties, GenerationInfo, Metadata, SimulationParameters

logger = logging.getLogger(__name__)


def run_simulation(
    iteration: int,
    sim_config_dict: dict[str, Any],
    output_base_dir: str = "RESULTS",
    seed: int | None = None,
    max_runtime_seconds: float | None = None,
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
    max_runtime_seconds : float | None, optional
        If set, abort and return (False, None, None) if the total elapsed
        wall-clock time exceeds this value between retry attempts.
        This allows callers to bound the worst-case runtime for parameter
        regions that are difficult or impossible to aggregate.

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
            sim_config_dict = dict(sim_config_dict)  # avoid mutating caller's dict
            sim_config_dict["seed"] = seed
        sim_params = SimulationParameters(**sim_config_dict)
        logger.info(f"Validated Config: {sim_params.model_dump_json(indent=2)}")
    except Exception as e:
        logger.error(f"Invalid simulation parameters provided: {e}", exc_info=True)
        return False, None, None

    start_time = time.time()

    if sim_params.seed is not None:
        rng = np.random.default_rng(sim_params.seed)
        logger.info(f"Using random seed: {sim_params.seed}")
    else:
        rng = np.random.default_rng()

    # Maximum number of PCA+CCA attempts (Fortran restarts on failure)
    # The Fortran re-generates radii from lognormal AND re-shuffles on every restart.
    # We match that behaviour: both steps happen inside the retry loop.
    max_attempts = 20
    pca_coords_radii = None
    pca_i_orden = None
    num_clusters = None
    pca_success = False

    for attempt in range(1, max_attempts + 1):
        # Check wall-clock budget before starting a new attempt
        if max_runtime_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed >= max_runtime_seconds:
                logger.warning(
                    f"run_simulation: wall-clock budget of {max_runtime_seconds}s "
                    f"exhausted after {elapsed:.1f}s (attempt {attempt}). Aborting."
                )
                return False, None, None

        # 1+2. Generate AND shuffle radii every attempt (Fortran does both per restart)
        try:
            initial_radii = particle_generation.lognormal_pp_radii(
                sim_params.rp_gstd,
                sim_params.rp_g,
                sim_params.N,
                rng=rng,
            )
        except ValueError as e:
            logger.error(f"Error generating radii on attempt {attempt}: {e}")
            continue
        shuffled_radii = utils.shuffle_array(initial_radii, rng=rng)

        logger.info(
            f"--- PCA+CCA Attempt {attempt}/{max_attempts} --- "
            f"Radii: mean={np.mean(shuffled_radii):.2f}, std={np.std(shuffled_radii):.2f}"
        )

        # 3. PCA Subclustering
        logger.info("--- Starting PCA Subclustering ---")
        pca_start_time = time.time()
        subcluster_runner = Subclusterer(
            initial_radii=shuffled_radii,
            df=sim_params.Df,
            kf=sim_params.kf,
            tol_ov=sim_params.tol_ov,
            n_subcl_percentage=sim_params.n_subcl_percentage,
            rp_g=sim_params.rp_g,
            rp_gstd=sim_params.rp_gstd,
            rng=rng,
        )
        pca_success = subcluster_runner.run_subclustering()
        pca_end_time = time.time()
        logger.info(
            f"PCA Subclustering Time: {pca_end_time - pca_start_time:.2f} seconds"
        )

        if not pca_success or subcluster_runner.not_able_pca:
            failed_subcluster_num_raw = getattr(
                subcluster_runner, "number_clusters_processed", None
            )
            if isinstance(failed_subcluster_num_raw, int):
                failed_subcluster_num: int | str = failed_subcluster_num_raw + 1
            else:
                failed_subcluster_num = "N/A"
            logger.warning(
                f"PCA Subclustering failed on attempt {attempt} "
                f"(Failed on Subcluster {failed_subcluster_num}). Retrying with new shuffle..."
            )
            continue  # retry with a new shuffle

        # Retrieve PCA results
        num_clusters, not_able_pca_flag, pca_coords_radii, pca_i_orden, _ = (
            subcluster_runner.get_results()
        )
        if not_able_pca_flag or pca_coords_radii is None or pca_i_orden is None:
            logger.warning(
                f"PCA returned invalid results on attempt {attempt} despite reporting success. Retrying..."
            )
            continue

        # 4. Cluster-Cluster Aggregation
        # When densify is enabled, generate at source Df/kf for easier CCA
        densify_enabled_gen = bool(getattr(config, "DENSIFY_ENABLED", False))
        if densify_enabled_gen:
            cca_df = float(getattr(config, "DENSIFY_SOURCE_DF", 2.0))
            cca_kf = float(getattr(config, "DENSIFY_SOURCE_KF", 1.0))
            logger.info(
                f"Densify: generating at source Df/kf={cca_df}/{cca_kf} "
                f"(target: {sim_params.Df}/{sim_params.kf})"
            )
        else:
            cca_df = sim_params.Df
            cca_kf = sim_params.kf

        logger.info("--- Starting Cluster-Cluster Aggregation ---")
        cca_start_time = time.time()
        cca_runner = CCAggregator(
            initial_coords=pca_coords_radii[:, :3],
            initial_radii=pca_coords_radii[:, 3],
            initial_i_orden=pca_i_orden,
            n_total=sim_params.N,
            df=cca_df,
            kf=cca_kf,
            tol_ov=sim_params.tol_ov,
            ext_case=sim_params.ext_case,
            rng=rng,
        )
        cca_result = cca_runner.run_cca()
        cca_end_time = time.time()
        logger.info(
            f"CCA Aggregation Time: {cca_end_time - cca_start_time:.2f} seconds"
        )

        if cca_result is None or cca_runner.not_able_cca:
            logger.warning(
                f"CCA Aggregation failed on attempt {attempt}. Retrying with new shuffle..."
            )
            continue  # retry with a new shuffle

        # Both PCA and CCA succeeded on this attempt
        logger.info(f"PCA+CCA succeeded on attempt {attempt}.")
        break
    else:
        # All attempts exhausted
        logger.error(f"PCA Subclustering failed after {max_attempts} attempts.")
        return False, None, None

    # 5. Prepare Results (Only if CCA succeeded)
    final_coords, final_radii = cca_result
    n_actual = final_coords.shape[0]

    # 5b. Post-aggregation densification (opt-in)
    densify_enabled = bool(getattr(config, "DENSIFY_ENABLED", False))
    if densify_enabled:
        source_df = float(getattr(config, "DENSIFY_SOURCE_DF", 2.0))
        source_kf = float(getattr(config, "DENSIFY_SOURCE_KF", 1.0))
        densify_method = str(getattr(config, "DENSIFY_METHOD", "radial"))
        densify_rtol = float(getattr(config, "DENSIFY_RTOL", 0.02))
        densify_max_push = int(getattr(config, "DENSIFY_MAX_PUSH_ITERS", 50))
        densify_max_iters = int(getattr(config, "DENSIFY_MAX_DENSIFY_ITERS", 20))
        densify_push_frac = float(getattr(config, "DENSIFY_PUSH_FRACTION", 0.5))
        densify_push_pat = int(getattr(config, "DENSIFY_PUSH_PATIENCE", 10))
        logger.info(
            f"Densification enabled: method={densify_method}, "
            f"source Df/kf={source_df}/{source_kf} -> "
            f"target Df/kf={sim_params.Df}/{sim_params.kf}"
        )
        densified_coords, densified_radii, densify_ok = densify_aggregate(
            final_coords,
            final_radii,
            target_df=sim_params.Df,
            target_kf=sim_params.kf,
            tol_ov=sim_params.tol_ov,
            max_push_iters=densify_max_push,
            max_densify_iters=densify_max_iters,
            push_fraction=densify_push_frac,
            push_patience=densify_push_pat,
            rg_rtol=densify_rtol,
            method=densify_method,
        )
        if densify_ok:
            logger.info("Densification succeeded, using densified coordinates.")
            final_coords = densified_coords
            final_radii = densified_radii
            n_actual = final_coords.shape[0]
        else:
            logger.warning("Densification did not fully converge; using best result.")
            final_coords = densified_coords
            final_radii = densified_radii
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
