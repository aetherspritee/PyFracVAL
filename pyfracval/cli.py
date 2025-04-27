import logging
import sys
import time  # For seeding if needed
from pathlib import Path

import click

from pyfracval import config as default_config
from pyfracval.logs import TRACE_LEVEL_NUM, create_logger
from pyfracval.main_runner import run_simulation

# --- Default values from config (or override here) ---
# These will be used if the user doesn't provide options
DEFAULT_DF = default_config.DF
DEFAULT_KF = default_config.KF
DEFAULT_N = default_config.N
DEFAULT_R0 = default_config.RP_GEOMETRIC_MEAN
DEFAULT_SIGMA = default_config.RP_GEOMETRIC_STD  # Note: Sigma here is rp_gstd
DEFAULT_EXT_CASE = default_config.EXT_CASE
DEFAULT_TOL_OV = default_config.TOL_OVERLAP
DEFAULT_N_SUBCL_PERC = default_config.N_SUBCL_PERCENTAGE
DEFAULT_OUTPUT_DIR = "RESULTS"  # Default save location


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
    default=DEFAULT_SIGMA,
    show_default=True,
    help="Geometric standard deviation of primary particle radii (>= 1.0).",
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
    """Main CLI entry point."""
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

    # --- Validate Inputs ---
    if kwargs["rp_gstd"] < 1.0:
        raise click.BadParameter(
            "Geometric standard deviation (rp_gstd) must be >= 1.0.",
            param_hint="--rp-gstd",
        )
    if kwargs["rp_g"] <= 0:
        raise click.BadParameter(
            "Geometric mean radius (rp_g) must be > 0.", param_hint="--rp-g"
        )
    if kwargs["num_particles"] < 2:
        raise click.BadParameter(
            "Number of particles (n) must be at least 2.", param_hint="-n"
        )

    # --- Prepare Configuration for Runner ---
    sim_config = {
        "N": kwargs["num_particles"],
        "Df": kwargs["df"],
        "kf": kwargs["kf"],
        "rp_g": kwargs["rp_g"],
        "rp_gstd": kwargs["rp_gstd"],
        "tol_ov": kwargs["tol_ov"],
        "n_subcl_percentage": kwargs["n_subcl_perc"],
        "ext_case": kwargs["ext_case"],
        # Add any other parameters required by run_simulation
    }

    # --- Run Simulation Loop ---
    output_folder = Path(kwargs["folder"])
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    aggregates_generated = 0
    global_start_time = time.time()
    base_seed = kwargs["seed"]
    plotters = []  # Store plotters if plotting multiple aggregates

    for i in range(kwargs["num_aggregates"]):
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

        while not success and attempt < kwargs["max_attempts"]:
            attempt += 1
            logger.info(
                f"--- Generating Aggregate {agg_num}/{kwargs['num_aggregates']}, Attempt {attempt}/{kwargs['max_attempts']} ---"
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
                    f"--- Retrying (up to {kwargs['max_attempts']} attempts)... ---"
                )
                # time.sleep(0.5)  # Small pause

        if success:
            aggregates_generated += 1
            if kwargs["plot"] and final_coords is not None and final_radii is not None:
                try:
                    import pyvista as pv  # Import only if needed

                    pl = plot_particles(final_coords, final_radii)
                    pl.add_text(
                        f"Aggregate {agg_num}/{kwargs['num_aggregates']}\nN={sim_config['N']}, Df={sim_config['Df']:.2f}",
                        position="upper_left",
                        font_size=10,
                    )
                    plotters.append(pl)
                except ImportError:
                    logger.warning(
                        "PyVista not installed. Cannot plot results. Install with 'pip install pyvista'"
                    )
                except Exception as e:
                    logger.warning(f"Error during plotting: {e}")
        else:
            logger.critical(
                f"FATAL: Failed to generate aggregate {agg_num} after {kwargs['max_attempts']} attempts."
            )
            # Optionally exit early on failure
            # ctx.fail(f"Failed to generate aggregate {agg_num}")

    # --- Final Summary ---
    global_end_time = time.time()
    logger.info("--------------------------------------------------")
    logger.info(
        f"Generated {aggregates_generated}/{kwargs['num_aggregates']} aggregates."
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

    if aggregates_generated < kwargs["num_aggregates"]:
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
def explore(path: str):  # pragma: no cover
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
