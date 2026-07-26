"""Streamlit application for exploring saved PyFracVAL aggregate data.

The module initializes the Streamlit UI, discovers ``.dat`` files under the
user-provided paths, and renders aggregate particle data alongside metadata.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista

from pyfracval.schemas import Metadata
from pyfracval.visualization import plot_particles

os.environ["VTK_USE_X"] = "OFF"
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"
pv.OFF_SCREEN = True

st.set_page_config(layout="wide")
st.title("PyFracVAL")

parser = argparse.ArgumentParser(description="Data explorer for PyFracVAL")
parser.add_argument(
    "--path",
    action="append",
    default=[],
    help="Base directory to look for clusters (must contain cluster_index.csv)",
)
parser.add_argument(
    "--index",
    type=Path,
    default=None,
    help="Path to cluster_index.csv (auto-detected from --path if not set)",
)

try:
    args = parser.parse_args()
except SystemExit as e:
    sys.exit(e.code)

# ---------------------------------------------------------------------------
# Load master index
# ---------------------------------------------------------------------------

index_path = args.index
if index_path is None:
    for p in args.path:
        candidate = Path(p) / "cluster_index.csv"
        if candidate.exists():
            index_path = candidate
            break

if index_path is None or not index_path.exists():
    st.error(
        "No cluster_index.csv found. "
        "Provide --path or --index pointing to cluster data."
    )
    st.stop()

df = pd.read_csv(index_path)
df = df[df["success"] == True]

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    configs = sorted(df["config"].unique())
    sel_config = st.multiselect(
        "Configuration",
        configs,
        default=list(configs),
    )

    df_vals = sorted(df["Df"].unique())
    sel_df = st.select_slider(
        "Fractal dimension (Df)",
        options=df_vals,
        value=(df_vals[0], df_vals[-1]),
    )

    n_vals = sorted(df["N"].unique())
    sel_n = st.select_slider(
        "Particles (N)",
        options=n_vals,
        value=(n_vals[0], n_vals[-1]),
    )

    sigma_vals = sorted(df["sigma"].unique())
    sel_sigma = st.select_slider(
        "Polydispersity (σ)",
        options=sigma_vals,
        value=(sigma_vals[0], sigma_vals[-1]),
    )

    kf_vals = sorted(df["kf"].unique())
    sel_kf = st.select_slider(
        "Prefactor (kf)",
        options=kf_vals,
        value=(kf_vals[0], kf_vals[-1]),
    )

    randomize = st.checkbox("Pick random cluster instead")

    st.header("Spacing")

    s_spacing = st.slider(
        "Gap (s × mean radius)",
        min_value=0.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
        help="s = 0 → touching; s = 1 → one radius gap on average",
    )
    show_comparison = False
    if s_spacing > 0.0:
        show_comparison = st.checkbox("Show original side-by-side")

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

mask = (
    df["config"].isin(sel_config)
    & (df["Df"] >= sel_df[0])
    & (df["Df"] <= sel_df[1])
    & (df["N"] >= sel_n[0])
    & (df["N"] <= sel_n[1])
    & (df["sigma"] >= sel_sigma[0])
    & (df["sigma"] <= sel_sigma[1])
    & (df["kf"] >= sel_kf[0])
    & (df["kf"] <= sel_kf[1])
)
filtered = df[mask]
n_matched = len(filtered)
n_total = len(df)

st.caption(f"{n_matched} of {n_total} clusters match filters")

if n_matched == 0:
    st.warning("No clusters match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

if randomize:
    choice = filtered.sample(1).iloc[0]
    file = Path(choice["filepath"])
    st.info(f"Random pick: {file.stem}")
else:
    file_list = [Path(p) for p in filtered["filepath"]]
    file = st.selectbox(
        "Cluster",
        file_list,
        format_func=lambda x: x.stem,
        help="Resize the sidebar if the paths are cut off",
    )

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

metadata_obj, data = Metadata.from_file(file)
metadata = metadata_obj.to_dict()

coords = data[:, :3].copy()
radii = data[:, 3].copy()
rg_orig = metadata.get("aggregate_properties", {}).get("radius_of_gyration") or 0.0

# ---------------------------------------------------------------------------
# Spacing transformation
# ---------------------------------------------------------------------------

T = 1.0 + s_spacing / 2.0
coords_scaled = coords * T

# ---------------------------------------------------------------------------
# Radii statistics (unchanged by spacing)
# ---------------------------------------------------------------------------

mean_r = float(np.mean(radii))
std_r = float(np.std(radii))
gmean_r = float(np.exp(np.mean(np.log(radii))))
gstd_r = float(np.exp(np.std(np.log(radii))))

# ---------------------------------------------------------------------------
# Find touching pairs for gap analysis
# ---------------------------------------------------------------------------


def find_touching_pairs(
    coords: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (i, j, dist_ij, r_i+r_j) for all touching particle pairs."""
    n = len(radii)
    # Process in chunks to limit memory for large N
    chunk = min(n, 512)
    i_list: list[np.ndarray] = []
    j_list: list[np.ndarray] = []
    d_list: list[np.ndarray] = []
    rr_list: list[np.ndarray] = []

    for a in range(0, n, chunk):
        a_end = min(a + chunk, n)
        sub = coords[a:a_end]
        sub_r = radii[a:a_end]
        # sub (C, 3) vs all (N, 3) -> (C, N, 3)
        delta = sub[:, None, :] - coords[None, :, :]
        dists = np.sqrt(np.sum(delta**2, axis=2))  # (C, N)
        rr = sub_r[:, None] + radii[None, :]  # (C, N)

        # Touching: within 1% of contact distance
        rel_err = np.abs(dists - rr) / np.maximum(rr, 1e-12)
        mask = rel_err < 0.01

        # Exclude self-pairs (where delta=0 and rr=2*r_i)
        for k in range(a_end - a):
            mask[k, a + k] = False

        ci, cj = np.where(mask)
        i_list.append(ci + a)
        j_list.append(cj)
        d_list.append(dists[ci, cj])
        rr_list.append(rr[ci, cj])

    if not i_list:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    return (
        np.concatenate(i_list),
        np.concatenate(j_list),
        np.concatenate(d_list),
        np.concatenate(rr_list),
    )


ti, tj, td_orig, trr = find_touching_pairs(coords, radii)
n_pairs = len(ti)
gaps = td_orig * (T - 1.0) if n_pairs > 0 else np.array([])
gap_mean = float(np.mean(gaps)) if n_pairs > 0 else 0.0
gap_std = float(np.std(gaps)) if n_pairs > 0 else 0.0
gap_expected = float(s_spacing * mean_r)

# ---------------------------------------------------------------------------
# 3D view
# ---------------------------------------------------------------------------

if show_comparison and s_spacing > 0.0:
    col_left, col_right = st.columns(2)
    with col_left:
        st.caption("Original (T=1.0)")
        plotter_orig = plot_particles(coords, radii)
        stpyvista(plotter_orig)
    with col_right:
        st.caption(f"Scaling factor T = {T:.3f}")
        plotter_spaced = plot_particles(coords_scaled, radii)
        stpyvista(plotter_spaced)
elif s_spacing > 0.0:
    st.caption(f"Scaling factor T = {T:.3f}")
    plotter = plot_particles(coords_scaled, radii)
    stpyvista(plotter)
else:
    plotter = plot_particles(coords, radii)
    stpyvista(plotter)

# ---------------------------------------------------------------------------
# Gap analysis table
# ---------------------------------------------------------------------------

if s_spacing > 0.0:
    st.subheader("Neighbor Gap Analysis")
    st.caption(f"{n_pairs} touching particle pairs found (contact error < 1% of rᵢ+rⱼ)")

    if n_pairs > 0:
        d_orig_mean = float(np.mean(td_orig))
        d_scaled_mean = float(np.mean(td_orig * T))

        s_from_gap = float(2.0 * gap_mean / np.mean(trr))
        s_from_gap_std = float(2.0 * gap_std / np.std(trr))
        s_from_rg = float(2.0 * (T - 1.0))
        s_from_d_ratio = float(2.0 * (d_scaled_mean / d_orig_mean - 1.0))

        pct = lambda v: (
            f"{abs(v - s_spacing) / s_spacing * 100:.2f}%" if s_spacing > 0 else "—"
        )

        rows = [
            {
                "Metric": "From mean gap",
                "s_eff": f"{s_from_gap:.4f}",
                "Target": f"{s_spacing:.1f}",
                "Δ": pct(s_from_gap),
                "✓": abs(s_from_gap - s_spacing) < 0.02,
            },
            {
                "Metric": "From std of gaps",
                "s_eff": f"{s_from_gap_std:.4f}",
                "Target": f"{s_spacing:.1f}",
                "Δ": pct(s_from_gap_std),
                "✓": abs(s_from_gap_std - s_spacing) < 0.02,
            },
            {
                "Metric": "From Rg scaling",
                "s_eff": f"{s_from_rg:.4f}",
                "Target": f"{s_spacing:.1f}",
                "Δ": pct(s_from_rg),
                "✓": abs(s_from_rg - s_spacing) < 1e-6,
            },
            {
                "Metric": "From d̄ ratio",
                "s_eff": f"{s_from_d_ratio:.4f}",
                "Target": f"{s_spacing:.1f}",
                "Δ": pct(s_from_d_ratio),
                "✓": abs(s_from_d_ratio - s_spacing) < 1e-6,
            },
        ]
    else:
        rows = [
            {
                "Metric": f"No touching pairs found.",
                "s_eff": "—",
                "Target": f"{s_spacing:.1f}",
                "Δ": "—",
                "✓": False,
            },
        ]

    cols = ["✓", "s_eff", "Target", "Δ"]
    for r in rows:
        r["✓"] = "✅" if r["✓"] else "❌"
    st.table(pd.DataFrame(rows).set_index("Metric")[cols])

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

T_val = float(T)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 4])
with col1:
    st.table(
        pd.DataFrame(
            dict(
                Arithmetic=[mean_r, std_r],
                Geometric=[gmean_r, gstd_r],
            ),
            index=["Mean", "STD"],
        )
    )
with col2:
    st.write(
        "Approximate Geometric STD: ",
        np.exp(std_r / mean_r),
    )

st.write(metadata)
with st.expander("Full file path"):
    st.write(file.resolve())

with st.expander("Tabulated raw data"):
    st.table(data)
