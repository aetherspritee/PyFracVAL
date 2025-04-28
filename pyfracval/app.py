import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# import polars as pl
import pandas as pd
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista

from pyfracval.schemas import Metadata
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
files.sort()
file = st.selectbox(
    "File",
    files,
    format_func=lambda x: x.stem,
    help="Resize the sidebar if the paths are cut off",
)

match Path(file).suffix:
    case ".csv":
        data = pd.read_csv(file).to_numpy()
        information = re.search(
            r"N(\d+)-D(\d+_\d+)-K(\d+_\d+)-(\d+)_(\d+)_(\d+)",
            str(file),
        )
        if information is not None:
            n = information.group(1)
            d = information.group(2)
            k = information.group(3)
            date = information.group(4)
            time = information.group(5)
            metadata = dict(
                N=int(n),
                Df=float(d.replace("_", ".")),
                kf=float(k.replace("_", ".")),
                timestamp=datetime.strptime(
                    f"{date[0:4]} {date[4:6]} {date[6:8]} {time[0:2]} {time[2:4]} {time[4:6]}",
                    "%Y %m %d %H %M %S",
                ),
            )
    case ".dat":
        data = np.loadtxt(file)
        metadata, data = Metadata.from_file(file)
        metadata = metadata.to_dict()
    case _:
        st.error("File type not supported")

plotter = plot_particles(data[:, :3], data[:, 3])
stpyvista(plotter)

gmean = np.exp(np.mean(np.log(data[:, 3])))
gstd = np.exp(np.std(np.log(data[:, 3])))

col1, col2 = st.columns([1, 4])
with col1:
    st.table(
        pd.DataFrame(
            dict(
                Arithemtic=[np.mean(data[:, 3]), np.std(data[:, 3])],
                Geometric=[gmean, gstd],
            ),
            index=["Mean", "STD"],  # pyright: ignore
        )
    )
with col2:
    st.write(
        "Approximate Geometric STD: ",
        np.exp(np.std(data[:, 3]) / np.mean(data[:, 3])),
    )

st.write(metadata)
with st.expander("Full file path"):
    st.write(file.resolve())


with st.expander("Tabulated raw data"):
    st.table(data)
