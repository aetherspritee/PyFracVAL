import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista

from pyfracval.visualization import plot_particles

pv.start_xvfb()

st.set_page_config(layout="wide")
st.title("PyFracVAL")

parser = argparse.ArgumentParser(description="Data explorer for YASF")
parser.add_argument(
    "--path",
    action="append",
    default=[],
    help="Set path to look for data",
)

try:
    args = parser.parse_args()
except SystemExit as e:
    sys.exit(e.code)

files = []
for path in args.path:
    p = Path(path)
    if not p.exists():
        st.warning(f"Path {p.resolve()} does not exist")
        continue
    files.extend([item for item in p.rglob("*.dat") if item.is_file()])
    # files.extend([item for item in p.rglob("*.csv") if item.is_file()])
file = st.selectbox(
    "File",
    files,
    format_func=lambda x: x.stem,
    help="Resize the sidebar if the paths are cut off",
)

match Path(file).suffix:
    case ".csv":
        data = pl.read_csv(file).to_numpy()
    case ".dat":
        # data = pd.read_table(file, delim_whitespace=True, header=None)
        # data = pd.read_csv(file, sep=r"\s+", header=None, names=["x", "y", "z", "r"])
        data = np.loadtxt(file)
    case _:
        st.error("File type not supported")

pl = plot_particles(data[:, :3], data[:, 3])
stpyvista(pl)

with st.expander("Full file path"):
    st.write(file.resolve())

information = re.search(r"N(\d+)-D(\d+_\d+)-K(\d+_\d+)-(\d+)_(\d+)_(\d+)", str(file))
if information is not None:
    n = information.group(1)
    d = information.group(2)
    k = information.group(3)
    date = information.group(4)
    time = information.group(5)
    st.write("Number of particles", int(n))
    st.write("Fractal dimension", float(d.replace("_", ".")))
    st.write("Fractal prefactor", float(k.replace("_", ".")))
    st.write(
        "Date & time",
        datetime.strptime(
            f"{date[0:4]} {date[4:6]} {date[6:8]} {time[0:2]} {time[2:4]} {time[4:6]}",
            "%Y %m %d %H %M %S",
        ),
    )

st.table(data)
