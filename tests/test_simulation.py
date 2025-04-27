import logging
import time
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from pyfracval import utils
from pyfracval.main_runner import run_simulation
from pyfracval.schemas import Metadata

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def output_dir(tmp_path: Path) -> Path:
    test_output_path = tmp_path / "test_results"
    test_output_path.mkdir()
    # print(f"\nCreated temp output dir: {test_output_path}") # Reduce noise
    return test_output_path


def _run_and_load(
    sim_config: dict, output_dir: Path, iteration: int = 1, seed: int | None = None
):
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    sim_config["seed"] = seed

    logger.info(f"--- Running Test Simulation (Seed: {seed}) ---")
    logger.info(f"Config: {sim_config}")

    # Run simulation with a higher number of attempts for tests
    # Note: The main runner only retries if success=False,
    # it doesn't handle internal retries if PCA/CCA fail within run_simulation
    max_attempts_test = 3  # Allow a few attempts for stochastic failures
    attempt = 0
    success = False
    final_coords = None
    final_radii = None

    # Add a loop within the test helper to allow retries on failure
    while not success and attempt < max_attempts_test:
        attempt += 1
        if attempt > 1:
            logger.warning(
                f"Retrying simulation run (Attempt {attempt}/{max_attempts_test})..."
            )
            # Use a different seed for retry to avoid identical failure
            sim_config["seed"] = seed + attempt
            np.random.seed(sim_config["seed"])  # Reset numpy seed for this attempt

        current_success, current_coords, current_radii = run_simulation(
            iteration=iteration,
            sim_config_dict=sim_config.copy(),  # Pass a copy to avoid mutation issues
            output_base_dir=str(output_dir),
            seed=sim_config["seed"],  # Pass the potentially updated seed
        )
        if current_success:
            success = True
            final_coords = current_coords
            final_radii = current_radii
            break  # Exit retry loop on success

    # Assertions after the retry loop
    assert success, (
        f"run_simulation failed after {max_attempts_test} attempts for config: {sim_config}"
    )
    assert final_coords is not None and final_radii is not None, (
        "run_simulation returned None on success"
    )
    assert final_coords.ndim == 2 and final_coords.shape[1] == 3, (
        "Coordinates shape mismatch"
    )
    assert final_radii.ndim == 1, "Radii shape mismatch"
    assert final_coords.shape[0] == final_radii.shape[0], "Coords/Radii length mismatch"
    n_actual = final_coords.shape[0]
    assert n_actual > 0, "Simulation produced empty aggregate"

    output_files = list(output_dir.glob("*.dat"))
    # Filter potentially multiple files if retries saved them
    # Choose the one matching the *final* successful seed or just the latest one?
    # Finding by seed is more robust if filename includes it
    final_seed_str = f"{sim_config['seed']}"
    target_filename_part = f"seed{final_seed_str}"
    found_file = None
    for f in output_files:
        if target_filename_part in f.name:
            found_file = f
            break
    # Fallback: if seed isn't reliably in name or multiple files exist, take the last modified one
    if not found_file and output_files:
        output_files.sort(key=lambda x: x.stat().st_mtime)
        found_file = output_files[-1]

    assert found_file is not None and found_file.exists(), (
        f"Could not find output file for seed {final_seed_str} in {output_dir}"
    )
    output_file = found_file

    try:
        metadata, data = Metadata.from_file(output_file)
        assert metadata is not None, f"Failed to load metadata from {output_file.name}"
        assert data is not None, f"Failed to load data from {output_file.name}"
        assert data.shape[0] == n_actual, (
            f"Loaded data rows ({data.shape[0]}) mismatch N_actual ({n_actual}) from {output_file.name}"
        )
        assert data.shape[1] == 4, (
            f"Loaded data columns mismatch from {output_file.name}"
        )
        npt.assert_allclose(
            final_coords,
            data[:, :3],
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"Returned coords differ from loaded data in {output_file.name}",
        )
        npt.assert_allclose(
            final_radii,
            data[:, 3],
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"Returned radii differ from loaded data in {output_file.name}",
        )
    except Exception as e:
        pytest.fail(f"Failed to load or validate output file {output_file}: {e}")

    return metadata, data, n_actual, final_radii


# --- Test Cases ---


def test_basic_run(output_dir: Path):
    """Test if the simulation runs without crashing for default-like params."""
    N_test = 64
    sim_config = {
        "N": N_test,
        "Df": 1.8,
        "kf": 1.3,
        "rp_g": 10.0,
        "rp_gstd": 1.2,
        "tol_ov": 1e-4,
        "n_subcl_percentage": 0.15,
        "ext_case": 0,
    }
    # Use a fixed seed for this basic test
    metadata, data, n_actual, _ = _run_and_load(sim_config, output_dir, seed=2001)
    assert abs(n_actual - N_test) <= 1
    assert metadata.aggregate_properties is not None
    assert metadata.aggregate_properties.radius_of_gyration is not None
    logger.info(
        f"Basic run successful. N={n_actual}, Rg={metadata.aggregate_properties.radius_of_gyration:.3f}"
    )


@pytest.mark.parametrize(
    "n_test, df_test, kf_test, gstd_test",
    [
        (64, 1.8, 1.3, 1.0),
        (64, 1.7, 1.1, 1.0),
        (64, 2.1, 1.0, 1.0),
        (64, 1.8, 1.3, 1.2),
        (64, 1.7, 1.1, 1.3),
        (128, 2.0, 1.0, 1.25),
    ],
)
def test_parameter_variations_and_rg_consistency(
    output_dir: Path,
    n_test: int,
    df_test: float,
    kf_test: float,
    gstd_test: float,
    seed: int = 12345,
):
    """Tests various parameter combinations and checks Rg consistency."""
    sim_config = {
        "N": n_test,
        "Df": df_test,
        "kf": kf_test,
        "rp_g": 10.0,
        "rp_gstd": gstd_test,
        "tol_ov": 1e-4,  # Keep relaxed tol for tests
        "n_subcl_percentage": 0.15,
        "ext_case": 0,
    }
    # Use fixed seed based on parameters for reproducibility within parametrize
    metadata, data, n_actual, final_radii = _run_and_load(
        sim_config, output_dir, seed=seed
    )

    assert abs(n_actual - n_test) <= 1
    assert metadata.aggregate_properties is not None
    rg_metadata = metadata.aggregate_properties.radius_of_gyration
    assert rg_metadata is not None

    # geo_mean_r = sim_config["rp_g"]
    # Recalculation for polydisperse doesn't seem necessary for consistency check with internal formula
    # if gstd_test > 1.0: ...

    rg_expected = utils.calculate_rg(final_radii, n_actual, df_test, kf_test)

    logger.info(
        f"Params (N,Df,kf,gstd): ({n_test},{df_test},{kf_test},{gstd_test}) -> "
        f"N_actual={n_actual}, Rg_metadata={rg_metadata:.4f}, Rg_expected={rg_expected:.4f}"
    )

    # Check consistency with a relative tolerance
    assert rg_metadata == pytest.approx(rg_expected, rel=1e-3), (
        f"Rg consistency check failed: Meta={rg_metadata:.4f}, Expected={rg_expected:.4f}"
    )


def test_reproducibility(output_dir: Path):
    """Test if using the same seed produces the same result."""
    N_test = 32
    fixed_seed = 12345
    sim_config = {
        "N": N_test,
        "Df": 1.9,
        "kf": 1.4,
        "rp_g": 10.0,
        "rp_gstd": 1.1,
        "tol_ov": 1e-4,
        "n_subcl_percentage": 0.2,
        "ext_case": 0,
    }

    logger.info("--- Reproducibility Test: Run 1 ---")
    output_dir_run1 = output_dir / "run1"
    output_dir_run1.mkdir()
    metadata1, data1, n1, _ = _run_and_load(
        sim_config, output_dir_run1, seed=fixed_seed
    )

    logger.info("--- Reproducibility Test: Run 2 ---")
    output_dir_run2 = output_dir / "run2"
    output_dir_run2.mkdir()
    metadata2, data2, n2, _ = _run_and_load(
        sim_config, output_dir_run2, seed=fixed_seed
    )

    assert n1 == n2, f"Number of particles differs between runs ({n1} vs {n2})"
    npt.assert_allclose(
        data1,
        data2,
        atol=1e-9,
        err_msg="Particle data differs between runs",
    )
    # Compare relevant parts of metadata, excluding timestamps etc.
    assert metadata1.simulation_parameters.model_dump(
        exclude={"seed"}
    ) == metadata2.simulation_parameters.model_dump(exclude={"seed"})
    assert metadata1.aggregate_properties == metadata2.aggregate_properties
    logger.info("Reproducibility test passed.")
