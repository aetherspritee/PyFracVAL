from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Sweep configuration — Pydantic models
# ---------------------------------------------------------------------------


class DaskSettings(BaseModel):
    """Settings for the Dask execution backend."""

    enable: bool = False
    scheduler_address: str | None = None
    workers: int | None = None


class SimulationDefaults(BaseModel):
    """Per-trial simulation knobs passed to run_simulation."""

    rp_g: float = 100.0
    tol_ov: float = 1e-6
    n_subcl_percentage: float = 0.1
    ext_case: int = 0
    trial_timeout: int | None = None


class SweepConfig(BaseModel):
    """Full declarative configuration for a stability sweep.

    Load from a TOML file with :meth:`from_toml`, then apply CLI overrides
    with :meth:`merged`.  Top-level keys map to sweep grid and run settings;
    ``[dask]`` and ``[simulation]`` sub-tables map to :class:`DaskSettings`
    and :class:`SimulationDefaults` respectively.

    Example TOML::

        sizes = [64, 128]
        sigmas = [1.0, 1.5]
        df_values = [1.8, 2.0, 2.2]
        kf_values = [0.8, 1.0, 1.2]
        trials = 10
        output_dir = "benchmark_results/my_sweep"
        save_raw = false

        [dask]
        enable = true
        scheduler_address = "tcp://host:8786"
        # workers = 4

        [simulation]
        rp_g = 100.0
        tol_ov = 1.0e-6
        n_subcl_percentage = 0.1
        ext_case = 0
        trial_timeout = 120
    """

    # --- Sweep grid (explicit lists take priority over min/max/step) --------
    sizes: list[int] = Field(default_factory=lambda: [64, 128])
    sigmas: list[float] = Field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])
    df_values: list[float] | None = None
    kf_values: list[float] | None = None
    df_min: float = 1.4
    df_max: float = 2.6
    df_step: float = 0.2
    kf_min: float = 0.6
    kf_max: float = 1.6
    kf_step: float = 0.2

    # --- Run control --------------------------------------------------------
    trials: int = 30
    save_raw: bool = False
    output_dir: str = "benchmark_results"

    # --- Sub-models ---------------------------------------------------------
    dask: DaskSettings = Field(default_factory=DaskSettings)
    simulation: SimulationDefaults = Field(default_factory=SimulationDefaults)

    # --- Validators ---------------------------------------------------------

    @model_validator(mode="after")
    def _validate_grid(self) -> "SweepConfig":
        if self.df_step <= 0:
            raise ValueError("df_step must be positive")
        if self.kf_step <= 0:
            raise ValueError("kf_step must be positive")
        if self.trials < 1:
            raise ValueError("trials must be >= 1")
        return self

    # --- Factory methods ----------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> "SweepConfig":
        """Load a :class:`SweepConfig` from a TOML file."""
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return cls.model_validate(data)

    # --- Merge helpers ------------------------------------------------------

    def merged(self, overrides: dict[str, Any]) -> "SweepConfig":
        """Return a new :class:`SweepConfig` with *overrides* applied.

        Only keys whose value is not ``None`` in *overrides* are applied.
        Nested overrides for ``dask`` and ``simulation`` are handled via
        dedicated sub-dicts (keys ``"dask"`` and ``"simulation"``).

        This is designed to be called with a filtered dict built from
        ``argparse.Namespace``, where unset flags are ``None``.
        """
        top: dict[str, Any] = self.model_dump()

        for key, value in overrides.items():
            if value is None:
                continue
            if key == "dask" and isinstance(value, dict):
                for dk, dv in value.items():
                    if dv is not None:
                        top["dask"][dk] = dv
            elif key == "simulation" and isinstance(value, dict):
                for sk, sv in value.items():
                    if sv is not None:
                        top["simulation"][sk] = sv
            else:
                top[key] = value

        return SweepConfig.model_validate(top)


# ---------------------------------------------------------------------------
# Legacy module-level constants (kept for backward compatibility)
# ---------------------------------------------------------------------------

# --- Simulation Parameters ---
N: int = 128  # Number of Primary Particles (PP)
# N: int = 1024  # Number of Primary Particles (PP)
DF: float = 2.0  # Target Fractal dimension
KF: float = 1.0  # Target Fractal prefactor
QUANTITY_AGGREGATES: int = 1  # Number of aggregates to generate

# --- Primary Particle Properties ---
RP_GEOMETRIC_MEAN: float = 100.0  # Geometric mean radius of PP
RP_GEOMETRIC_STD: float = 1.50  # Geometric standard deviation of PP radii
# RP_GEOMETRIC_STD: float = 1.25  # Geometric standard deviation of PP radii
# RP_GEOMETRIC_STD: float = 1.00  # Geometric standard deviation of PP radii

# --- Algorithm Tuning Parameters ---
EXT_CASE: int = 0  # CCA Sticking: 0 for standard, 1 for 'extreme cases'
N_SUBCL_PERCENTAGE: float = 0.1  # PCA: Target fraction of N for subcluster size
TOL_OVERLAP: float = 1.0e-6  # Overlap tolerance for sticking

# --- Performance Tuning ---
USE_BATCH_ROTATION: bool = (
    False  # Enable batch rotation (slower for N<1000, keep False)
)
ROTATION_BATCH_SIZE: int = (
    32  # Number of rotation angles to evaluate in parallel (if enabled)
)

# --- Constants ---
PI: float = np.pi

# --- Derived Parameters (can be calculated later if needed) ---
# N_SUBCL: int = ... # Calculated in pca_subclusters.py
