import logging
import sys
import time  # For seeding if needed
from pathlib import Path

import click
import numpy as np
from click.core import ParameterSource

from pyfracval.config import RunConfig
from pyfracval.logs import TRACE_LEVEL_NUM, create_logger
from pyfracval.main_runner import run_simulation

# --- Default values shown in --help / used when no --config and no flag given ---
_DEFAULTS = RunConfig()
DEFAULT_DF = _DEFAULTS.simulation.Df
DEFAULT_KF = _DEFAULTS.simulation.kf
DEFAULT_N = _DEFAULTS.simulation.N
DEFAULT_R0 = _DEFAULTS.simulation.rp_g
DEFAULT_SIGMA = _DEFAULTS.simulation.rp_gstd  # Note: Sigma here is rp_gstd
DEFAULT_EXT_CASE = _DEFAULTS.simulation.ext_case
DEFAULT_TOL_OV = _DEFAULTS.simulation.tol_ov
DEFAULT_N_SUBCL_PERC = _DEFAULTS.simulation.n_subcl_percentage
DEFAULT_OUTPUT_DIR = _DEFAULTS.output_dir

# Maps click option/kwarg names to RunConfig.simulation field names.
_SIMULATION_FLAG_MAP: dict[str, str] = {
    "df": "Df",
    "kf": "kf",
    "num_particles": "N",
    "rp_g": "rp_g",
    "ext_case": "ext_case",
    "tol_ov": "tol_ov",
    "n_subcl_perc": "n_subcl_percentage",
    "seed": "seed",
}
# Maps click option/kwarg names to top-level RunConfig field names.
_RUN_FLAG_MAP: dict[str, str] = {
    "num_aggregates": "num_aggregates",
    "folder": "output_dir",
    "max_attempts": "max_attempts",
    "plot": "plot",
}


# --- Click Command Group ---
@click.group(
    invoke_without_command=True,
    context_settings=dict(help_option_names=["-h", "--help"]),
    help="Generate fractal particle clusters using the FracVAL algorithm.",
)
@click.version_option(
    package_name="pyfracval"
)  # Add version if you have __version__ in __init__.py
@click.pass_context
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a TOML/YAML/JSON config file (format auto-detected from "
    "the extension). The config file is the source of truth for anything "
    "it sets; any flag below that you explicitly pass overrides the "
    "corresponding config file value. Algorithm-tuning options not exposed "
    "as flags (retry modes, densification, etc.) can only be set via a "
    "config file's [algorithm] table - see docs/source/experiments.md for "
    "what's available.",
)
# --- Options mirroring config.py ---
@click.option(
    "--df",
    type=float,
    default=DEFAULT_DF,
    show_default=True,
    help="Target fractal dimension (Df).",
)
@click.option(
    "--kf",
    type=float,
    default=DEFAULT_KF,
    show_default=True,
    help="Target fractal prefactor (kf).",
)
@click.option(
    "-n",
    "--num-particles",
    type=int,
    default=DEFAULT_N,
    show_default=True,
    help="Total number of primary particles (N).",
)
@click.option(
    "--rp-g",
    type=float,
    default=DEFAULT_R0,
    show_default=True,
    help="Geometric mean radius of primary particles.",
)
@click.option(
    "--rp-gstd",
    type=float,
    default=None,
    show_default=f"Calculated from --rp-std or defaults to {DEFAULT_SIGMA}",  # Show calculated default
    help="Geometric standard deviation of primary particle radii (>= 1.0). "
    "If provided, this value takes precedence over --rp-std.",  # Added precedence info
)
@click.option(
    "--rp-std",
    type=float,
    default=None,
    help="Approximate arithmetic standard deviation of primary particle radii. "
    "If --rp-gstd is NOT provided, this value will be used to estimate "
    "a geometric standard deviation using the heuristic exp(std/mean). "
    "A warning will be shown with the calculated geometric value.",
)
@click.option(
    "--ext-case",
    type=click.IntRange(0, 1),
    default=DEFAULT_EXT_CASE,
    show_default=True,
    help="CCA sticking ext_case (0 or 1). Affects collision geometry check.",
)
@click.option(
    "--tol-ov",
    type=float,
    default=DEFAULT_TOL_OV,
    show_default=True,
    help="Overlap tolerance for particle sticking.",
)
@click.option(
    "--n-subcl-perc",
    type=click.FloatRange(0.01, 0.5),  # Reasonable range
    default=DEFAULT_N_SUBCL_PERC,
    show_default=True,
    help="Target fraction of N for PCA subcluster size (e.g., 0.1 for 10%).",
)
@click.option(
    "--num-aggregates",
    type=int,
    default=1,  # Default to generating 1 aggregate via CLI
    show_default=True,
    help="Number of separate aggregate structures to generate.",
)
@click.option(
    "-p",
    "--plot",
    is_flag=True,
    default=False,
    help="Display the generated aggregate(s) using PyVista interactively.",
)
@click.option(
    "-f",
    "--folder",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory to save the output aggregate data file(s).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible generation.",
)
@click.option(
    "--max-attempts",
    type=int,
    default=5,  # Max retries per aggregate if generation fails
    show_default=True,
    help="Maximum number of attempts to generate each aggregate if it fails.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    default=0,
    help="Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)",
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True, resolve_path=True),  # Path to a file
    default=None,  # Default is None (log to console)
    help="Path to file for logging output instead of console.",
)
def cli(ctx, **kwargs) -> None:
    """Generate fractal particle clusters using the FracVAL algorithm.

    Parameters
    ----------
    ctx : click.Context
        Click command context used to detect whether a subcommand was invoked.
    **kwargs : dict
        Parsed CLI options, including fractal parameters, particle size
        distribution settings, output directory, logging controls, and
        generation limits.

    Returns
    -------
    None
        The command performs side effects only and exits through Click.

    Notes
    -----
    This command implements the Particle-Cluster Aggregation (PCA) followed
    by Cluster-Cluster Aggregation (CCA) approach described by Moran et al.
    (2019). It validates the requested parameters, configures logging, and
    runs one or more aggregate generation attempts.
    """
    if ctx.invoked_subcommand:
        return

    match kwargs["verbose"]:
        case 0:
            log_level = logging.WARNING
        case 1:
            log_level = logging.INFO
        case 2:
            log_level = logging.DEBUG
        case _:
            log_level = TRACE_LEVEL_NUM

    logger = create_logger(log_level, kwargs["log_file"])

    # --- Load config file (source of truth), then layer explicit CLI flags on top ---
    run_cfg = (
        RunConfig.from_file(kwargs["config_path"])
        if kwargs["config_path"]
        else RunConfig()
    )

    def _explicit(name: str) -> bool:
        return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE

    sim_overrides: dict = {
        field: kwargs[flag]
        for flag, field in _SIMULATION_FLAG_MAP.items()
        if _explicit(flag)
    }
    run_overrides: dict = {
        field: kwargs[flag] for flag, field in _RUN_FLAG_MAP.items() if _explicit(flag)
    }

    # rp_gstd / rp_std precedence: --rp-gstd wins, then --rp-std (heuristic),
    # then whatever the config file / defaults already say (left untouched
    # below if neither flag was explicitly given).
    if _explicit("rp_gstd"):
        if kwargs["rp_gstd"] < 1.0:
            raise click.BadParameter(
                "Geometric standard deviation (--rp-gstd) must be >= 1.0.",
                param_hint="--rp-gstd",
            )
        sim_overrides["rp_gstd"] = kwargs["rp_gstd"]
        if _explicit("rp_std"):
            logger.warning("Both --rp-gstd and --rp-std provided. Using --rp-gstd.")
    elif _explicit("rp_std"):
        if kwargs["rp_std"] < 0:
            raise click.BadParameter(
                "Arithmetic standard deviation (--rp-std) cannot be negative.",
                param_hint="--rp-std",
            )
        # Apply heuristic: sigma_g = exp(sigma_a / mu_g)
        rp_g_for_heuristic = sim_overrides.get("rp_g", run_cfg.simulation.rp_g)
        computed_rp_gstd = np.exp(kwargs["rp_std"] / rp_g_for_heuristic)
        sim_overrides["rp_gstd"] = computed_rp_gstd
        logger.warning(
            f"Using heuristic to calculate geometric standard deviation from arithmetic std: "
            f"exp(rp_std / rp_g) = exp({kwargs['rp_std']:.2f} / {rp_g_for_heuristic:.2f}) = {computed_rp_gstd:.3f}. "
            f"Targeting rp_gstd = {computed_rp_gstd:.3f} for generation."
        )

    run_cfg = run_cfg.merged({**run_overrides, "simulation": sim_overrides})
    sim = run_cfg.simulation

    # --- Validate Inputs ---
    if sim.rp_g <= 0:
        raise click.BadParameter(
            "Geometric mean radius (rp_g) must be > 0.", param_hint="--rp-g"
        )
    if sim.N < 2:
        raise click.BadParameter(
            "Number of particles (n) must be at least 2.", param_hint="-n"
        )

    # --- Prepare Configuration for Runner ---
    sim_config = {
        "N": sim.N,
        "Df": sim.Df,
        "kf": sim.kf,
        "rp_g": sim.rp_g,
        "rp_gstd": sim.rp_gstd,
        "tol_ov": sim.tol_ov,
        "n_subcl_percentage": sim.n_subcl_percentage,
        "ext_case": sim.ext_case,
        **run_cfg.algorithm.model_dump(),
    }

    # --- Run Simulation ---
    output_folder = Path(run_cfg.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    global_start_time = time.time()
    base_seed = sim.seed

    # results holds one (success, coords, radii) tuple per aggregate,
    # index i corresponding to aggregate number i+1 - populated by whichever
    # branch below runs.
    results: list[tuple[bool, np.ndarray | None, np.ndarray | None]]

    if run_cfg.dask is not None:
        # A [dask] table is present in the config - dispatch through a Dask
        # cluster instead of running sequentially. No outer retry here (unlike
        # the sequential path's --max-attempts): each aggregate is one Dask
        # task calling run_simulation once, which already retries internally.
        from .batch_runner import generate_aggregates_parallel

        logger.info(
            f"[dask] config present - generating {run_cfg.num_aggregates} "
            f"aggregates via Dask (workers={run_cfg.dask.workers!r}, "
            f"scheduler={run_cfg.dask.scheduler_address or 'local'})."
        )
        results = generate_aggregates_parallel(
            n_aggregates=run_cfg.num_aggregates,
            config=sim_config,
            output_base_dir=str(output_folder),
            seed_start=base_seed if base_seed is not None else 1000,
            n_workers=run_cfg.dask.workers,
            scheduler_address=run_cfg.dask.scheduler_address,
        )
    else:
        results = []
        for i in range(run_cfg.num_aggregates):
            agg_num = i + 1
            attempt = 0
            success = False
            final_coords = None
            final_radii = None

            # Determine seed for this specific aggregate run
            current_seed = (
                base_seed + agg_num
                if base_seed is not None
                else int(time.time() * 1000) % (2**32)
            )

            while not success and attempt < run_cfg.max_attempts:
                attempt += 1
                logger.info(
                    f"--- Generating Aggregate {agg_num}/{run_cfg.num_aggregates}, Attempt {attempt}/{run_cfg.max_attempts} ---"
                )
                success, final_coords, final_radii = run_simulation(
                    iteration=agg_num,
                    sim_config_dict=sim_config,
                    output_base_dir=str(output_folder),
                    seed=current_seed,  # Use specific seed for this run
                )
                if not success:
                    # Log the specific error from the runner first
                    logger.error(
                        f"Aggregate {agg_num} generation failed on attempt {attempt}."
                    )
                    # Provide general retry advice (specific advice logged by runner)
                    logger.info(
                        f"--- Retrying (up to {run_cfg.max_attempts} attempts)... ---"
                    )
                    # time.sleep(0.5)  # Small pause

            if not success:
                logger.critical(
                    f"FATAL: Failed to generate aggregate {agg_num} after {run_cfg.max_attempts} attempts."
                )
                # Optionally exit early on failure
                # ctx.fail(f"Failed to generate aggregate {agg_num}")

            results.append((success, final_coords, final_radii))

    aggregates_generated = sum(1 for success, _, _ in results if success)

    # --- Plotting (both branches share this) ---
    plotters = []  # Store plotters if plotting multiple aggregates
    if run_cfg.plot:
        for i, (success, final_coords, final_radii) in enumerate(results):
            if not (success and final_coords is not None and final_radii is not None):
                continue
            agg_num = i + 1
            try:
                import pyvista as pv  # Import only if needed

                from pyfracval.visualization import plot_particles

                pl = plot_particles(final_coords, final_radii)
                pl.add_text(
                    f"Aggregate {agg_num}/{run_cfg.num_aggregates}\nN={sim_config['N']}, Df={sim_config['Df']:.2f}",
                    position="upper_left",
                    font_size=10,
                )
                plotters.append(pl)
            except ImportError:
                logger.warning(
                    "PyVista not installed. Cannot plot results. Install with 'pip install pyvista'"
                )
                break
            except Exception as e:
                logger.warning(f"Error during plotting: {e}")

    # --- Final Summary ---
    global_end_time = time.time()
    logger.info("--------------------------------------------------")
    logger.info(
        f"Generated {aggregates_generated}/{run_cfg.num_aggregates} aggregates."
    )
    logger.info(f"Results saved to: {output_folder.resolve()}")
    logger.info(
        f"Total Simulation Time: {global_end_time - global_start_time:.2f} seconds"
    )
    logger.info("--------------------------------------------------")

    # --- Show Plots ---
    # Show plots sequentially after all simulations are done
    if plotters:
        logger.info("Displaying plots...")
        first_plotter = plotters[0]
        if len(plotters) > 1:
            # Link cameras if multiple plots exist for consistent view manipulation
            # This might require careful handling depending on PyVista version
            logger.info(f"Linking {len(plotters)} plot windows...")
            # Simple linking (may not work perfectly across separate plotters)
            # for i in range(1, len(plotters)):
            #    plotters[i].link_views(first_plotter) # Try linking to the first one

            # Alternatively, use subplots for multiple aggregates
            # shape = (1, len(plotters)) # Arrange horizontally
            # combined_pl = pv.Plotter(shape=shape)
            # for i, p in enumerate(plotters):
            #    combined_pl.subplot(0, i)
            #    # Add meshes from individual plotters - might need access to glyph mesh
            #    # This requires refactoring plot_particles to return the mesh or data
            # combined_pl.show()
            # --> Showing sequentially is simpler for now <--

        for i, pl in enumerate(plotters):
            logger.info(f"Showing plot for Aggregate {i + 1}...")
            pl.show()  # Blocking call, shows one plot at a time
            logger.info(f"Plot {i + 1} closed.")

    if aggregates_generated < run_cfg.num_aggregates:
        logger.warning(
            f"Only {aggregates_generated}/{kwargs['num_aggregates']} aggregates were generated successfully."
        )
        ctx.exit(1)  # Exit with error code
    else:
        logger.info(
            f"Finished generating {aggregates_generated} aggregates successfully."
        )


# --- Streamlit Command (keep as is if needed) ---
@cli.command(help="""Explore data using Streamlit (Requires separate app.py)""")
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    default=".",  # Look in current directory by default
    help="Path where to look for data files to be displayed by Streamlit app.",
)
def explore(path: str):
    """Launch the Streamlit dashboard to explore saved aggregate data.

    Parameters
    ----------
    path : str
        Directory searched for saved aggregate data files before launching
        the Streamlit app.

    Returns
    -------
    None
        The function launches Streamlit or exits the process on error.

    Notes
    -----
    Requires Streamlit to be installed separately. The app looks for ``.dat``
    files in the provided directory and starts the dashboard with that path.
    """
    logger = create_logger(logging.INFO)

    try:
        from streamlit import runtime
        from streamlit.web import cli as stcli
    except ImportError:
        logger.info(
            "Error: Streamlit is not installed. Please install it: pip install streamlit"
        )
        sys.exit(1)

    if not runtime.exists():
        # app_path = Path(__file__).parent / "pyfracval" / "app.py"
        app_path = Path(__file__).parent / "app.py"
        if not app_path.exists():
            print(f"Error: Streamlit app not found at expected location: {app_path}")
            print("Please ensure app.py exists within the pyfracval directory.")
            sys.exit(1)

        print(f"Launching Streamlit app: {app_path}")
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--",
            f"--path={path}",  # Pass path argument correctly to streamlit
        ]
        sys.exit(stcli.main())
    else:
        print(
            "Streamlit runtime already exists (maybe running from within Streamlit?)."
        )


# --- Main Execution ---
if __name__ == "__main__":
    cli()
