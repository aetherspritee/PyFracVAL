from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Multi-format config loading
# ---------------------------------------------------------------------------


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Load a config file into a plain dict, auto-detecting format from extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.toml``, ``.yaml``/``.yml``, or ``.json`` config file.

    Returns
    -------
    dict[str, Any]
        The parsed file contents, ready for ``SomeModel.model_validate(...)``.

    Raises
    ------
    ValueError
        If the file extension isn't one of the supported formats.

    Notes
    -----
    TOML is parsed in binary mode via :mod:`tomllib` (its required mode);
    YAML and JSON are parsed in text mode. YAML is loaded with
    :func:`yaml.safe_load` — arbitrary Python object tags are not supported,
    matching TOML/JSON's data-only semantics.
    """
    path = Path(path)
    match path.suffix.lower():
        case ".toml":
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        case ".yaml" | ".yml":
            with path.open() as fh:
                data = yaml.safe_load(fh)
        case ".json":
            with path.open() as fh:
                data = json.load(fh)
        case suffix:
            raise ValueError(
                f"Unrecognized config file extension '{suffix}' for {path} "
                "(supported: .json, .toml, .yaml, .yml)"
            )
    return data if data is not None else {}


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

    # --- Algorithm defaults (optional) ------------------------------------
    algorithm: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Algorithm-level overrides passed through to each simulation trial. "
            "Keys use the OrchestratorAlgorithmConfig field names (lowercase, "
            "e.g. 'densify_enabled', 'cca_retry_rotation_mode')."
        ),
    )

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
        """Load a :class:`SweepConfig` from a TOML file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the TOML file containing sweep configuration values.

        Returns
        -------
        SweepConfig
            A validated configuration instance created from the file data.

        Notes
        -----
        The TOML file is parsed with :mod:`tomllib` and validated through the
        Pydantic model constructor. See :meth:`from_file` for a version that
        also accepts YAML/JSON.
        """
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "SweepConfig":
        """Load a :class:`SweepConfig` from a TOML, YAML, or JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the config file. Format is auto-detected from the file
            extension (``.toml``, ``.yaml``/``.yml``, or ``.json``).

        Returns
        -------
        SweepConfig
            A validated configuration instance created from the file data.
        """
        return cls.model_validate(load_config_dict(path))

    # --- Merge helpers ------------------------------------------------------

    def merged(self, overrides: dict[str, Any]) -> "SweepConfig":
        """Return a new :class:`SweepConfig` with *overrides* applied.

        Parameters
        ----------
        overrides : dict[str, Any]
            Mapping of replacement values. ``None`` values are ignored, while
            nested ``"dask"`` and ``"simulation"`` dictionaries are merged
            into the corresponding sub-models.

        Returns
        -------
        SweepConfig
            A new configuration instance with the requested overrides applied.

        Notes
        -----
        This is intended for filtered CLI or argparse namespaces where unset
        options are represented as ``None``.
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
    N: int = 128
    Df: float = 1.8
    kf: float = 1.0
    rp_g: float = 100.0
    rp_gstd: float = 1.5
    tol_ov: float = 1e-6
    n_subcl_percentage: float = 0.1
    ext_case: int = 0
    seed: int | None = None


class OrchestratorAlgorithmConfig(BaseModel):
    use_cca_incremental_overlap: bool = True
    cca_incremental_full_sync_period: int = 20
    cca_candidate_policy: str = "baseline"
    # --- Pairing strategy (docs/source/matching_pairing.md) ------------------
    # "backtracking" (default, production) retries a cluster against its
    # next feasible partner on a *real* sticking failure instead of
    # aborting the round | "greedy" (the historical first-fit default) |
    # "matching" (exact maximum-cardinality matching over the cheap
    # feasibility graph) | "matching_leaf_weighted" (matching with
    # leaf-class edge weights as a tiebreaker among cardinality-optimal
    # solutions). See docs/source/backtracking_pairing.md and
    # docs/source/matching_pairing.md.
    cca_pairing_strategy: str = "backtracking"
    # Cap on how many partners one cluster may be tried against per round
    # before it is passed through unmerged. Bounds the worst-case cost of
    # backtracking: a failed sticking attempt is expensive (every
    # candidate pair x up to 360 rotations).
    cca_backtracking_max_partners: int = 4
    # When no partner sticks, carry the cluster into the next round
    # unmerged rather than failing the attempt outright. A round that
    # merges nothing at all still fails, so this cannot loop forever.
    cca_backtracking_pass_through: bool = True
    cca_matching_leaf_class_weights: dict[str, float] = Field(
        default_factory=lambda: {"LL": 1.0, "LN": 0.6, "NN": 0.3}
    )
    cca_matching_leaf_class_threshold: float = 0.5
    # --- Overlap-failure census (docs/source/overlap_failure_census.md) ------
    # Strictly opt-in, off the hot path: when a pair's sticking attempt
    # fails, runs one full (non-early-exit) overlap scan to record how
    # many particles overlap and by how much, instead of just the binary
    # success/fail signal pyfracval/overlap.py's scalar checks give.
    cca_overlap_census_enabled: bool = False
    cca_overlap_census_max_pairs: int = 4096
    # --- Gamma formulation (see NOTE.md 1.2) ---------------------------------
    # --- CCA merge equation ---
    # True (default) solves Gamma with true masses - Moran et al. (2019)
    # Eq. 6, the paper's central polydisperse contribution, and what the
    # Fortran CCA does. False substitutes particle *counts* (Filippov
    # et al. 2000 Eq. 7), which is what this port historically did; kept
    # as an opt-in alternative for reproducing that behaviour.
    #
    # The two coincide only for uniform-density monodisperse primary
    # particles. Counts cannot represent a heterogeneous aggregate at all:
    # once particles differ in density, mass stops being a function of
    # radius, so any count- or radius-derived weighting misplaces the
    # center of mass and the radius of gyration.
    gamma_use_mass: bool = True
    # --- PCA monomer-addition equation ---
    # Deliberately defaults to counts, unlike CCA above, matching the
    # Fortran PCA (PCA_cca.f90's Gamma_calculation takes n1, n2, n3).
    #
    # This is not an inconsistency in the original: Eq. 6's mass moments
    # are only consistent with a *count*-based scaling-law Rg when both
    # bodies are aggregates large enough for that law to describe them. In
    # PCA the second body is a single monomer, whose scaling-law Rg is
    # meaningless at n=1 while its mass can rival the whole growing
    # cluster's for a wide size distribution. Mixing the two bases
    # produces Gamma values that admit almost no candidates: measured at
    # sigma=1.9, N=12, 150 seeds, mass-based PCA built 1/150 subclusters
    # against 93/150 for counts. Left configurable for experiments, but
    # turning it on is not advisable.
    pca_gamma_use_mass: bool = False
    # Feed the *measured* radius of gyration of each already-built cluster
    # into the next round's Gamma, instead of re-deriving it from the
    # scaling law. Without this, per-merge deviations (the pairing
    # relaxation factor, the adaptive overlap tolerance) accumulate
    # uncorrected because nothing ever measures what was actually built.
    cca_gamma_measured_rg: bool = False
    # --- Per-merge event log (pyfracval/merge_log.py) ------------------------
    # Path to a JSONL file receiving one record per CCA merge attempt.
    # None disables it entirely (no file opened, no records built).
    cca_merge_log_path: str | None = None
    # --- Drop-a-few-particles rescue (docs/source/drop_rescue.md) ------------
    # Opt-in; requires the overlap census above (auto-enabled when this is
    # True, see _validate_algorithm below). No backfill: a rescued merge
    # yields fewer than the requested N particles.
    cca_drop_rescue_enabled: bool = False
    # Absolute safety cap per side; 0 disables it so the relative cap
    # below scales the budget with cluster size instead (the two combine
    # with min(), so a non-zero absolute cap dominates at large N - see
    # docs/source/drop_rescue.md).
    cca_drop_rescue_max_particles: int = 5
    cca_drop_rescue_max_fraction: float = 0.02  # relative cap, per side
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
    densify_enabled: bool = False
    densify_source_df: float = 2.0
    densify_source_kf: float = 1.0
    densify_max_push_iters: int = 50
    densify_max_densify_iters: int = 20
    densify_push_fraction: float = 0.5
    densify_push_patience: int = 10
    densify_rtol: float = 0.02
    densify_method: str = "radial"
    densify_rtol_multiplier: float = 2.0
    profile_cca_retry_modes: bool = False

    # --- Soft potential relaxation (archived to pyfracval/experimental/,
    # not on the production path - confirmed no improvement over baseline,
    # see docs/source/experiments.md) -------------------------------------
    cca_soft_relaxation_enabled: bool = False
    cca_soft_relaxation_k_repulsion: float = 10.0
    cca_soft_relaxation_k_gamma: float = 1.0
    cca_soft_relaxation_gamma_tolerance: float = 0.05
    cca_soft_relaxation_max_iters: int = 100
    cca_soft_relaxation_learning_rate: float = 0.1
    cca_soft_relaxation_fallback_only: bool = True

    # --- PCA tuning ----------------------------------------------------------
    parallel_subclusters: bool = True
    parallel_subclusters_min_count: int = 4
    use_batch_rotation: bool = False
    rotation_batch_size: int = 32

    # --- Profiling / debug toggles -------------------------------------------
    profile_timing: bool = False
    profile_cca_leaf_stats: bool = False
    profile_cca_candidate_score: bool = False

    @model_validator(mode="after")
    def _drop_rescue_requires_census(self) -> "OrchestratorAlgorithmConfig":
        """cca_drop_rescue_enabled needs a populated overlap census to act
        on - auto-enable it rather than silently no-op-ing if a caller sets
        drop-rescue without also setting the census flag."""
        if self.cca_drop_rescue_enabled and not self.cca_overlap_census_enabled:
            self.cca_overlap_census_enabled = True
        return self


class RunConfig(BaseModel):
    """Top-level config for a single CLI/library simulation run.

    Config files are the source of truth: load one with :meth:`from_file`
    (TOML/YAML/JSON, auto-detected by extension), then apply explicit CLI
    flag overrides with :meth:`merged` — only values the user actually
    typed take precedence over the file; the file takes precedence over
    these built-in defaults.

    Dask is opt-in by *presence*: generation runs sequentially unless the
    config file has a ``[dask]`` table, in which case it dispatches through
    a Dask cluster instead (local by default, or a remote scheduler if
    ``scheduler_address`` is set). There's no separate "enable" flag to
    also remember to flip - writing the table is the switch.

    Example TOML::

        num_aggregates = 5
        output_dir = "RESULTS"

        [simulation]
        N = 256
        Df = 1.8
        kf = 1.0
        rp_gstd = 1.5

        [algorithm]
        cca_retry_rotation_mode = "alternate"
        densify_enabled = true

        # Optional - omit this table entirely to run sequentially.
        [dask]
        workers = 4
        # scheduler_address = "tcp://host:8786"  # omit to use a local cluster
    """

    simulation: OrchestratorSimulationConfig = Field(
        default_factory=lambda: OrchestratorSimulationConfig(Df=2.0)
    )
    algorithm: OrchestratorAlgorithmConfig = Field(
        default_factory=OrchestratorAlgorithmConfig
    )
    num_aggregates: int = 1
    output_dir: str = "RESULTS"
    max_attempts: int = 5
    plot: bool = False
    dask: DaskSettings | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        """Load a :class:`RunConfig` from a TOML, YAML, or JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the config file. Format is auto-detected from the file
            extension (``.toml``, ``.yaml``/``.yml``, or ``.json``).

        Returns
        -------
        RunConfig
            A validated configuration instance created from the file data.
        """
        return cls.model_validate(load_config_dict(path))

    def merged(self, overrides: dict[str, Any]) -> "RunConfig":
        """Return a new :class:`RunConfig` with *overrides* applied.

        Parameters
        ----------
        overrides : dict[str, Any]
            Mapping of replacement values. ``None`` values are ignored, so
            that a dict built from CLI options (unset options represented
            as ``None``) only overrides what the user actually provided.
            Nested ``"simulation"`` and ``"algorithm"`` dicts are merged
            into the corresponding sub-models field-by-field.

        Returns
        -------
        RunConfig
            A new configuration instance with the requested overrides
            applied (flags win over the loaded file, which won over these
            defaults).
        """
        top: dict[str, Any] = self.model_dump()

        for key, value in overrides.items():
            if value is None:
                continue
            if key in ("simulation", "algorithm") and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_value is not None:
                        top[key][sub_key] = sub_value
            else:
                top[key] = value

        return RunConfig.model_validate(top)


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
        """Load an :class:`OrchestratorConfig` from a TOML file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the TOML file containing orchestrator defaults and run
            definitions.

        Returns
        -------
        OrchestratorConfig
            A validated orchestrator configuration created from the file data.

        Notes
        -----
        The TOML file is parsed with :mod:`tomllib` and validated by Pydantic.
        See :meth:`from_file` for a version that also accepts YAML/JSON.
        """
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "OrchestratorConfig":
        """Load an :class:`OrchestratorConfig` from a TOML, YAML, or JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the config file. Format is auto-detected from the file
            extension (``.toml``, ``.yaml``/``.yml``, or ``.json``).

        Returns
        -------
        OrchestratorConfig
            A validated orchestrator configuration created from the file data.
        """
        return cls.model_validate(load_config_dict(path))
