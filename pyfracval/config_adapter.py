"""Backward-compatible config adapter for PyFracVAL.

This module provides :func:`get_config` which returns an
:class:`~pyfracval.config.OrchestratorAlgorithmConfig` instance populated
from Pydantic model defaults, with fallback to legacy module-level constants
in :mod:`pyfracval.config`.

The helper :func:`_getattr_config` it exposes can be used as a
drop-in replacement for ``getattr(config, "UPPER_CASE", fallback)``
patterns scattered throughout the codebase, narrowing the lookup to
Pydantic defaults first and then the legacy constants.

.. deprecated::
    Direct access to ``pyfracval.config.UPPER_CASE`` constants is deprecated.
    Prefer ``from pyfracval.config_adapter import get_config`` and use the
    returned Pydantic model's attributes (lower-case, hyphen-to-underscore).
"""

from __future__ import annotations

import warnings
from typing import Any

from pyfracval.config import OrchestratorAlgorithmConfig

# ---------------------------------------------------------------------------
# Mapping: legacy uppercase constant name -> Pydantic field name
# ---------------------------------------------------------------------------
# This mapping covers every legacy constant in config.py that has a
# corresponding field in OrchestratorAlgorithmConfig.
_LEGACY_TO_PYDANTIC: dict[str, str] = {
    "USE_CCA_INCREMENTAL_OVERLAP": "use_cca_incremental_overlap",
    "CCA_INCREMENTAL_FULL_SYNC_PERIOD": "cca_incremental_full_sync_period",
    "CCA_CANDIDATE_POLICY": "cca_candidate_policy",
    "CCA_SCORE_TOPK_PER_CLASS": "cca_score_topk_per_class",
    "CCA_RETRY_ROTATION_MODE": "cca_retry_rotation_mode",
    "CCA_COARSE_FINE_COARSE_FRACTION": "cca_coarse_fine_coarse_fraction",
    "CCA_COARSE_FINE_SPIN_DEG": "cca_coarse_fine_spin_deg",
    "CCA_RETRY_ESCALATE_AFTER": "cca_retry_escalate_after",
    "CCA_DUAL_JITTER_INTERVAL": "cca_dual_jitter_interval",
    "CCA_DUAL_JITTER_DEG": "cca_dual_jitter_deg",
    "CCA_COARSE_SWEEP_STEPS": "cca_coarse_sweep_steps",
    "CCA_COARSE_SPIN_ANCHOR_STEPS": "cca_coarse_spin_anchor_steps",
    "CCA_COARSE_SPIN_MOVING_STEPS": "cca_coarse_spin_moving_steps",
    "CCA_SOFT_ACCEPT_ENABLED": "cca_soft_accept_enabled",
    "CCA_SOFT_ACCEPT_OVERLAP": "cca_soft_accept_overlap",
    "CCA_REPAIR_MAX_ITERS": "cca_repair_max_iters",
    "CCA_REPAIR_STEP_DEG": "cca_repair_step_deg",
    "CCA_REPAIR_STEP_TRANSLATION_FRAC": "cca_repair_step_translation_frac",
    "CCA_REPAIR_PATIENCE": "cca_repair_patience",
    "CCA_GAMMA_EXPANSION_ENABLED": "cca_gamma_expansion_enabled",
    "CCA_GAMMA_EXPANSION_STEP": "cca_gamma_expansion_step",
    "CCA_GAMMA_EXPANSION_MAX_FACTOR": "cca_gamma_expansion_max_factor",
    "CCA_GAMMA_EXPANSION_MASS_EXPONENT": "cca_gamma_expansion_mass_exponent",
    "CCA_GAMMA_EXPANSION_MAX_ATTEMPTS": "cca_gamma_expansion_max_attempts",
    "CCA_PAIR_FEASIBILITY_FILTER": "cca_pair_feasibility_filter",
    "CCA_BV_DEEP_PENETRATION_FACTOR": "cca_bv_deep_penetration_factor",
    "CCA_SSA_MIN_EXPOSURE": "cca_ssa_min_exposure",
    "CCA_STICKING_METHOD": "cca_sticking_method",
    "CCA_FFT_GRID_SIZE": "cca_fft_grid_size",
    "CCA_FFT_NUM_ROTATIONS": "cca_fft_num_rotations",
    "CCA_FFT_TOP_K_PEAKS": "cca_fft_top_k_peaks",
    "CCA_FFT_GAMMA_TOLERANCE": "cca_fft_gamma_tolerance",
    "CCA_FFT_MIN_PEAK_DISTANCE": "cca_fft_min_peak_distance",
    "DENSIFY_ENABLED": "densify_enabled",
    "DENSIFY_SOURCE_DF": "densify_source_df",
    "DENSIFY_SOURCE_KF": "densify_source_kf",
    "DENSIFY_MAX_PUSH_ITERS": "densify_max_push_iters",
    "DENSIFY_MAX_DENSIFY_ITERS": "densify_max_densify_iters",
    "DENSIFY_PUSH_FRACTION": "densify_push_fraction",
    "DENSIFY_PUSH_PATIENCE": "densify_push_patience",
    "DENSIFY_RTOL": "densify_rtol",
    "DENSIFY_METHOD": "densify_method",
    "DENSIFY_RTOL_MULTIPLIER": "densify_rtol_multiplier",
    "PROFILE_CCA_RETRY_MODES": "profile_cca_retry_modes",
}


def get_config() -> OrchestratorAlgorithmConfig:
    """Return an :class:`OrchestratorAlgorithmConfig` with default values.

    This is the single entry-point for code that wants to read algorithm
    tuning parameters.  The returned model is constructed from Pydantic
    defaults (the canonical source of truth), *not* legacy module-level
    constants.

    Returns
    -------
    OrchestratorAlgorithmConfig
        A fresh configuration instance with all default values.

    Notes
    -----
    The legacy uppercase constants in :mod:`pyfracval.config` (e.g.
    ``config.CCA_STICKING_METHOD``) are still readable but will eventually
    be removed.  Prefer calling :func:`get_config` and using the lower-case
    attribute names on the returned Pydantic model.
    """
    return OrchestratorAlgorithmConfig()


def getattr_config(
    module: Any,
    name: str,
    default: Any = None,
) -> Any:
    """Look up a config value, preferring Pydantic defaults over legacy constants.

    This is a drop-in replacement for the ``getattr(config, "UPPER_CASE", fallback)``
    pattern used throughout the codebase.  It first checks whether *name*
    maps to a field in :class:`OrchestratorAlgorithmConfig`; if so, it returns
    the Pydantic default.  Otherwise it falls back to the legacy module-level
    constant.

    Parameters
    ----------
    module : module
        The :mod:`pyfracval.config` module object (passed explicitly to
        avoid a circular import at import time).
    name : str
        The uppercase legacy constant name, e.g. ``"CCA_STICKING_METHOD"``.
    default : Any, optional
        Fallback value if *name* is neither a Pydantic field nor a module
        constant.

    Returns
    -------
    Any
        The resolved config value.

    Examples
    --------
    >>> from pyfracval import config
    >>> from pyfracval.config_adapter import getattr_config
    >>> getattr_config(config, "CCA_STICKING_METHOD", "fibonacci")
    'fibonacci'
    """
    # 1. Try Pydantic field first (canonical source of truth)
    if name in _LEGACY_TO_PYDANTIC:
        pydantic_field = _LEGACY_TO_PYDANTIC[name]
        return OrchestratorAlgorithmConfig.model_fields[pydantic_field].default

    # 2. Try the module-level constant
    if hasattr(module, name):
        warnings.warn(
            f"Accessing legacy constant config.{name} is deprecated. "
            f"Use get_config().{_LEGACY_TO_PYDANTIC.get(name, name.lower())} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(module, name)

    # 3. Fall back to the caller-provided default
    return default
