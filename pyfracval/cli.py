import click
import numpy as np

from pyfracval.CCA import CCA_subcluster

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
def cli(
    fractal_dimension: float,
    fractal_prefactor: float,
    number_of_particles: int,
    mean_radius: float,
    std_radius: float,
    plot: bool,
) -> None:
    R = np.ones((number_of_particles)) * mean_radius
    isFine = False
    N_subcl_perc = 0.1
    iter = 1
    while not isFine:
        CCA_ok, PCA_ok = CCA_subcluster(
            R,
            number_of_particles,
            fractal_dimension,
            fractal_prefactor,
            iter,
            N_subcl_perc,
            EXT_CASE,
        )
        isFine = CCA_ok and PCA_ok
        if not isFine:
            print("Restarting, wasnt able to generate aggregate")

    print("Successfully generated aggregate")


# def _download(Src):
#     worker_id = current_process()._identity[0]
#     db = Src()
#     db.download(position=worker_id)


# def download_db(dbs: str):
#     if dbs == "all":
#         download_list = list(databases.values())
#     else:
#         db_list = [item.lower() for item in dbs.split(",")]
#         download_list = [databases[item] for item in db_list]

#     # Use with 3.14: with Pool(processes=2) as pool:
#     # Polars can get deadlock if fork() is used
#     # Using spawn() fixes this for now
#     # Should be fixed in 3.14
#     with get_context("spawn").Pool(processes=2) as pool:
#         for _ in tqdm(
#             pool.imap(_download, download_list),
#             total=len(download_list),
#             desc="TOTAL",
#             position=0,
#         ):
#             pass

#     click.echo("All databases downlaoded!")
#     click.echo("Bye :)")


# @cli.command(help="""Display data from a database.""")
# @click.option(
#     "--db",
#     help="Database to be used.",
# )
# @click.option(
#     "--data",
#     help="Data to be used from the database.",
# )
# @click.option(
#     "--display",
#     default="table",
#     show_default=True,
#     help="How to display the data: table or graph.",
# )
# @click.option(
#     "--bounds",
#     help="Bounds for the graph. Two values separated by a comma, e.g., `1.5,3.56`",
# )
# def show(db, data, display, bounds) -> None:  # pragma: no cover
#     scale = 1e-6
#     df = parse_source(db, data)
#     nk = df.nk.with_columns(pl.col("w").truediv(scale))
#     if bounds is not None:
#         bounds = [float(val) for val in bounds.split(",")]
#         if len(bounds) != 2:
#             raise Exception("Bounds need to have two values separated by a comma.")
#         nk = nk.filter((pl.col("w") > bounds[0]) & (pl.col("w") < bounds[1]))
#     match str.lower(display):
#         case "table":
#             with pl.Config(tbl_rows=1000):
#                 click.echo(nk)
#         case "graph":
#             if "n" in df.nk.columns:
#                 plt.plot(nk["w"], nk["n"], label="n")
#             if "k" in df.nk.columns:
#                 plt.plot(nk["w"], nk["k"], label="k")
#             plt.title("Refractive index values")
#             plt.xlabel(f"Wavelength in {scale}")
#             plt.ylabel("Values")
#             plt.show()
#         case _:
#             raise Exception("Unsupported display option")


# def parse_source(db, data) -> RefIdxDB:  # pragma: no cover
#     match str.lower(db):
#         case "refidx":
#             return RefIdx(path=data)
#         case "aria":
#             return Aria(path=data)
#         case _:
#             raise Exception(f"Provided {db} is not supported!")


# @cli.command(help="""Explore data using Streamlit""")
# def explore():  # pragma: no cover
#     if not runtime.exists():
#         print(Path(__file__).parent)
#         sys.argv = ["streamlit", "run", f"{Path(__file__).parent}/app.py"]
#         sys.exit(stcli.main())
