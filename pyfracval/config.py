from __future__ import annotations

import tomllib
from math import sqrt
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


class OrchestratorSimulationConfig(BaseModel):
    Df: float = 1.8
    kf: float = 1.0
    rp_g: float = 100.0
    rp_gstd: float = 1.5
    tol_ov: float = 1e-6
    n_subcl_percentage: float = 0.1
    ext_case: int = 0


class OrchestratorAlgorithmConfig(BaseModel):
    use_cca_incremental_overlap: bool = True
    cca_incremental_full_sync_period: int = 20
    cca_candidate_policy: str = "leaf_hybrid"
    cca_score_topk_per_class: int = 32
    cca_retry_rotation_mode: str = "single"
    cca_coarse_fine_coarse_fraction: float = 0.67
    cca_coarse_fine_spin_deg: float = 12.0
    cca_retry_escalate_after: int = 120
    cca_dual_jitter_interval: int = 5
    cca_dual_jitter_deg: float = 8.0
    cca_coarse_sweep_steps: int = 10
    cca_coarse_spin_anchor_steps: int = 6
    cca_coarse_spin_moving_steps: int = 6
    cca_soft_accept_enabled: bool = False
    cca_soft_accept_overlap: float = 5e-5
    cca_repair_max_iters: int = 24
    cca_repair_step_deg: float = 2.0
    cca_repair_step_translation_frac: float = 0.02
    cca_repair_patience: int = 6
    cca_gamma_expansion_enabled: bool = False
    cca_gamma_expansion_step: float = 0.02
    cca_gamma_expansion_max_factor: float = 1.05
    cca_gamma_expansion_mass_exponent: float = -0.75
    cca_gamma_expansion_max_attempts: int = 3
    cca_pair_feasibility_filter: str = "none"
    cca_bv_deep_penetration_factor: float = 0.8
    cca_ssa_min_exposure: float = 0.3
    cca_sticking_method: str = "fibonacci"
    cca_fft_grid_size: int = 64
    cca_fft_num_rotations: int = 70
    cca_fft_top_k_peaks: int = 10
    cca_fft_gamma_tolerance: float = 0.10
    cca_fft_min_peak_distance: int = 3
    profile_cca_retry_modes: bool = False


class OrchestratorDefaultsConfig(BaseModel):
    execution_mode: str = "sequential"
    output_root: str = "benchmark_results/profiles"
    n_values: list[int] = Field(default_factory=lambda: [256])
    repeats: int = 1
    n_aggregates: int = 12
    warmup_tasks: int = 2
    seed_start: int = 1431354440
    trial_timeout: float | None = None
    local_workers: int | None = None
    scheduler: str | None = None
    profile: bool = True
    simulation: OrchestratorSimulationConfig = Field(
        default_factory=OrchestratorSimulationConfig
    )
    algorithm: OrchestratorAlgorithmConfig = Field(
        default_factory=OrchestratorAlgorithmConfig
    )

    @model_validator(mode="after")
    def _validate_defaults(self) -> "OrchestratorDefaultsConfig":
        if self.execution_mode not in {"sequential", "parallel"}:
            raise ValueError(
                "defaults.execution_mode must be 'sequential' or 'parallel'"
            )
        if self.repeats < 1:
            raise ValueError("defaults.repeats must be >= 1")
        if self.n_aggregates < 1:
            raise ValueError("defaults.n_aggregates must be >= 1")
        if self.warmup_tasks < 0:
            raise ValueError("defaults.warmup_tasks must be >= 0")
        if not self.n_values or any(n <= 0 for n in self.n_values):
            raise ValueError("defaults.n_values must contain positive integers")
        if self.local_workers is not None and self.local_workers < 1:
            raise ValueError("defaults.local_workers must be >= 1 when provided")
        if (
            self.scheduler is not None
            and self.scheduler != "local"
            and not self.scheduler.startswith("tcp://")
        ):
            raise ValueError("defaults.scheduler must be 'local' or start with tcp://")
        return self


class OrchestratorRunConfig(BaseModel):
    name: str
    scheduler: str
    workers: int | None = None
    local_workers: int | None = None
    n_values: list[int] | None = None
    repeats: int | None = None
    n_aggregates: int | None = None
    warmup_tasks: int | None = None
    seed_start: int | None = None
    trial_timeout: float | None = None
    profile: bool | None = None
    simulation: OrchestratorSimulationConfig | None = None
    algorithm: OrchestratorAlgorithmConfig | None = None

    @model_validator(mode="after")
    def _validate_run(self) -> "OrchestratorRunConfig":
        if not self.name.strip():
            raise ValueError("runs[].name must be non-empty")
        if self.scheduler != "local" and not self.scheduler.startswith("tcp://"):
            raise ValueError("runs[].scheduler must be 'local' or start with tcp://")
        if self.workers is not None and self.workers < 1:
            raise ValueError("runs[].workers must be >= 1 when provided")
        if self.local_workers is not None and self.local_workers < 1:
            raise ValueError("runs[].local_workers must be >= 1 when provided")
        if self.n_values is not None:
            if not self.n_values or any(n <= 0 for n in self.n_values):
                raise ValueError("runs[].n_values must contain positive integers")
        if self.repeats is not None and self.repeats < 1:
            raise ValueError("runs[].repeats must be >= 1")
        if self.n_aggregates is not None and self.n_aggregates < 1:
            raise ValueError("runs[].n_aggregates must be >= 1")
        if self.warmup_tasks is not None and self.warmup_tasks < 0:
            raise ValueError("runs[].warmup_tasks must be >= 0")
        return self


class OrchestratorConfig(BaseModel):
    defaults: OrchestratorDefaultsConfig = Field(
        default_factory=OrchestratorDefaultsConfig
    )
    runs: list[OrchestratorRunConfig]

    @model_validator(mode="after")
    def _validate_runs(self) -> "OrchestratorConfig":
        if not self.runs:
            raise ValueError("Config must define at least one [[runs]] entry")
        names = [run.name for run in self.runs]
        if len(names) != len(set(names)):
            raise ValueError("run names must be unique")
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> "OrchestratorConfig":
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return cls.model_validate(data)


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
PROFILE_TIMING: bool = False  # Print per-phase timing summary after each run_cca
USE_CCA_INCREMENTAL_OVERLAP: bool = (
    True  # Enable active-set + periodic full-check overlap path in CCA retries
)
CCA_INCREMENTAL_FRONTIER_DELTA: float = (
    1.0e-4  # Reserved for tuning compatibility (currently unused)
)
CCA_INCREMENTAL_FULL_SYNC_PERIOD: int = (
    20  # Force full overlap sync every N retry rotations
)
PROFILE_CCA_LEAF_STATS: bool = (
    False  # Print CCA candidate success/attempt stats by leaf class
)
PROFILE_CCA_CANDIDATE_SCORE: bool = (
    False  # Print candidate quality score statistics and success correlation
)
CCA_CANDIDATE_POLICY: str = (
    "baseline"  # Candidate order: baseline|leaf_soft|leaf_score|leaf_hybrid
)
CCA_SCORE_TOPK_PER_CLASS: int = (
    0  # Optional cap for scored items per class in scored policies (0 = score all)
)
CCA_RETRY_ROTATION_MODE: str = "single"  # Retry rotation mode: single|alternate|dual_jitter|coarse_grid|coarse_to_fine
CCA_RETRY_ESCALATE_AFTER: int = (
    120  # Escalate from single to configured mode after this many retries
)
CCA_DUAL_JITTER_INTERVAL: int = (
    5  # In dual_jitter mode, apply anchor jitter every N retries
)
CCA_DUAL_JITTER_DEG: float = 8.0  # Anchor-cluster jitter amplitude in degrees
CCA_COARSE_SWEEP_STEPS: int = (
    10  # Coarse-grid sweep positions for moving cluster (retry mode coarse_grid)
)
CCA_COARSE_SPIN_ANCHOR_STEPS: int = (
    6  # Coarse-grid spin positions for anchor cluster around CM->cand axis
)
CCA_COARSE_SPIN_MOVING_STEPS: int = (
    6  # Coarse-grid spin positions for moving cluster around CM->cand axis
)
CCA_COARSE_FINE_COARSE_FRACTION: float = (
    0.67  # Fraction of retry budget used for coarse stage in coarse_to_fine mode
)
CCA_COARSE_FINE_SPIN_DEG: float = (
    12.0  # Local spin refinement amplitude in degrees for coarse_to_fine mode
)
CCA_SOFT_ACCEPT_ENABLED: bool = False  # Enable soft-overlap pre-accept before repair
CCA_SOFT_ACCEPT_OVERLAP: float = (
    5.0e-5  # Maximum overlap allowed to trigger rigid repair pass
)
CCA_REPAIR_MAX_ITERS: int = 24  # Max rigid repair iterations for soft-accepted poses
CCA_REPAIR_STEP_DEG: float = (
    2.0  # Base angular step (degrees) for rigid repair local search
)
CCA_REPAIR_STEP_TRANSLATION_FRAC: float = (
    0.02  # Translation step as fraction of mean primary radius during repair
)
CCA_REPAIR_PATIENCE: int = 6  # Stop repair after this many non-improving iterations
CCA_GAMMA_EXPANSION_ENABLED: bool = (
    False  # Enable incremental gamma expansion when sticking fails
)
CCA_GAMMA_EXPANSION_STEP: float = (
    0.02  # Base expansion step as fraction of original gamma_pc
)
CCA_GAMMA_EXPANSION_MAX_FACTOR: float = (
    1.05  # Maximum gamma expansion factor (5% over original)
)
CCA_GAMMA_EXPANSION_MASS_EXPONENT: float = (
    -0.75  # Inverse-mass scaling exponent for expansion step
)
CCA_GAMMA_EXPANSION_MAX_ATTEMPTS: int = (
    3  # Maximum number of expansion increments per pair
)
CCA_PAIR_FEASIBILITY_FILTER: str = (
    "none"  # Pair feasibility pre-filter: none|bounding_volume|ssa
)
CCA_BV_DEEP_PENETRATION_FACTOR: float = (
    0.8  # Safety factor for deep-interpenetration bounding volume reject
)
CCA_SSA_MIN_EXPOSURE: float = (
    0.3  # Minimum exposed fraction for SSA pair filter candidate
)
CCA_STICKING_METHOD: str = "fibonacci"  # "fibonacci" (original) or "fft_docking"
CCA_FFT_GRID_SIZE: int = 64  # Side length of cubic voxel grid for FFT docking
CCA_FFT_NUM_ROTATIONS: int = 70  # Number of SO(3) rotation samples for FFT docking
CCA_FFT_TOP_K_PEAKS: int = 10  # Number of translation candidates per rotation
CCA_FFT_GAMMA_TOLERANCE: float = 0.10  # Relative tolerance on gamma_pc distance check
CCA_FFT_MIN_PEAK_DISTANCE: int = (
    3  # Minimum grid-cell distance between correlation peaks
)
PROFILE_CCA_RETRY_MODES: bool = False  # Print retry-mode usage and success counters
PARALLEL_SUBCLUSTERS: bool = (
    True  # Build independent PCA subclusters in parallel (multiprocessing.Pool)
)
PARALLEL_SUBCLUSTERS_MIN_COUNT: int = (
    4  # Only parallelise when number_clusters >= this threshold
)

# --- Constants ---
PI: float = np.pi
GOLDEN_RATIO: float = (1.0 + sqrt(5.0)) / 2.0  # Fibonacci spiral constant

# --- Derived Parameters (can be calculated later if needed) ---
# N_SUBCL: int = ... # Calculated in pca_subclusters.py
