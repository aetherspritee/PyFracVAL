import sys
from pathlib import Path

import click
import jax
import numpy as np
import pyvista as pv
from streamlit import runtime
from streamlit.web import cli as stcli

from pyfracval.CCA import CCA_subcluster

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


# config
DF = 1.8
Kf = 1.0
N = 1024
R0 = 1
SIGMA = 0
EXT_CASE = 0


@click.group(
    invoke_without_command=True,
    help="Calculate a fractal particle cluster",
)
@click.pass_context
@click.option(
    "-df",
    "--fractal-dimension",
    default=1.8,
    help="Fractal dimension",
)
@click.option(
    "-kf",
    "--fractal-prefactor",
    default=1.0,
    help="Fractal prefactor",
)
@click.option(
    "-n",
    "--number-of-particles",
    default=1024,
    help="Number of particles",
)
@click.option(
    "-r",
    "--mean-radius",
    default=1.0,
    help="Mean radius",
)
@click.option(
    "-s",
    "--std-radius",
    default=0.0,
    help="Standard deviation of the radius",
)
@click.option(
    "-p",
    "--plot",
    is_flag=True,
    help="Display result using pyvista",
)
@click.option(
    "-f",
    "--folder",
    default="results",
    help="Folder to save the results",
)
def cli(
    ctx,
    fractal_dimension: float,
    fractal_prefactor: float,
    number_of_particles: int,
    mean_radius: float,
    std_radius: float,
    plot: bool,
    folder: str,
) -> None:
    if ctx.invoked_subcommand:
        return
    R = np.ones((number_of_particles)) * mean_radius
    isFine = False
    N_subcl_perc = 0.1
    iter = 1
    while not isFine:
        data, CCA_ok, PCA_ok = CCA_subcluster(
            R,
            number_of_particles,
            fractal_dimension,
            fractal_prefactor,
            iter,
            N_subcl_perc,
            EXT_CASE,
            folder=folder,
        )
        isFine = CCA_ok and PCA_ok
        if not isFine:
            print("Restarting, wasnt able to generate aggregate")

    if data is None:
        raise Exception("Failed to generate aggregate. Probably due to PCA.")

    if plot:
        pl = plot_particles(data["x", "y", "z"].to_numpy(), data["r"].to_numpy())
        pl.show()

    print("Successfully generated aggregate")


# def plot_particles(position, radii) -> pv.Plotter:
def plot_particles(position, radii):
    point_cloud = pv.PolyData(position)
    point_cloud["radius"] = [2 * i for i in radii]

    geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
    glyphed = point_cloud.glyph(scale="radius", geom=geom, orient=False)  # type: ignore
    pl = pv.Plotter(window_size=[800, 800])
    pl.add_mesh(glyphed, color="white", smooth_shading=True, pbr=True)
    pl.view_isometric()  # type: ignore
    pl.link_views()
    return pl
    # pl.show()


@cli.command(help="""Explore data using Streamlit""")
@click.option(
    "--path",
    type=str,
    default="results",
    help="Path where to look for data files to be displayed",
)
def explore(path: str):  # pragma: no cover
    if not runtime.exists():
        print(Path(__file__).parent)
        sys.argv = [
            "streamlit",
            "run",
            f"{Path(__file__).parent}/app.py",
            "--",
            "--path",
            path,
        ]
        sys.exit(stcli.main())
