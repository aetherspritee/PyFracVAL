"""Archived CCA retry-rotation search strategies.

``alternate``, ``dual_jitter``, ``coarse_grid``, and ``coarse_to_fine`` were
benchmarked against the production default (``single``) at N=256/512 in the
hard regime (see ``docs/source/experiments.md``): all four modes reach the
same success rate and are statistically indistinguishable in timing from
``single``. Broadening the rotation search doesn't help when the underlying
problem is geometric frustration - no orientation is overlap-free at the
required contact distance.

Kept here rather than deleted since they remain reachable via
``cca_retry_rotation_mode`` for anyone who wants to try a different angle
(e.g. a much finer coarse grid) later.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..config import OrchestratorAlgorithmConfig

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

RotateFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
ReintentoFn = Callable[..., tuple[np.ndarray, float]]
NormalizeFn = Callable[[np.ndarray, np.ndarray | None], np.ndarray]


def apply_retry_rotation_mode(
    mode_cfg: str,
    coords1_stick: np.ndarray,
    coords2_current: np.ndarray,
    coords1_base: np.ndarray,
    coords2_base: np.ndarray,
    cm1: np.ndarray,
    cm2_stick: np.ndarray,
    cand2_idx: int,
    vec_0: np.ndarray,
    i_vec: np.ndarray,
    j_vec: np.ndarray,
    axis_anchor: np.ndarray,
    axis_moving: np.ndarray,
    intento: int,
    algorithm_config: OrchestratorAlgorithmConfig,
    reintento_fn: ReintentoFn,
    rotate_fn: RotateFn,
    normalize_axis_fn: NormalizeFn,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Dispatch to one of the archived non-``single`` retry modes."""
    if mode_cfg == "coarse_grid":
        return _apply_coarse_grid(
            coords1_base,
            coords2_base,
            cm1,
            cm2_stick,
            cand2_idx,
            vec_0,
            i_vec,
            j_vec,
            axis_anchor,
            axis_moving,
            intento,
            algorithm_config,
            reintento_fn,
            rotate_fn,
        )

    if mode_cfg == "coarse_to_fine":
        return _apply_coarse_to_fine(
            coords1_stick,
            coords2_current,
            coords1_base,
            coords2_base,
            cm1,
            cm2_stick,
            cand2_idx,
            vec_0,
            i_vec,
            j_vec,
            axis_anchor,
            axis_moving,
            intento,
            algorithm_config,
            reintento_fn,
            rotate_fn,
        )

    escalate_after = int(max(0, algorithm_config.cca_retry_escalate_after))
    use_mode = mode_cfg if intento > escalate_after else "single"

    if use_mode == "alternate":
        return _apply_alternate(
            coords1_stick,
            coords2_current,
            cm1,
            i_vec,
            intento,
            reintento_fn,
            rotate_fn,
            normalize_axis_fn,
            cand2_idx,
            cm2_stick,
            vec_0,
            j_vec,
        )

    if use_mode == "dual_jitter":
        return _apply_dual_jitter(
            coords1_stick,
            coords2_current,
            cm1,
            cm2_stick,
            cand2_idx,
            vec_0,
            i_vec,
            j_vec,
            intento,
            algorithm_config,
            reintento_fn,
            rotate_fn,
            rng,
        )

    coords2_next, _ = reintento_fn(
        coords2_current, cm2_stick, cand2_idx, vec_0, i_vec, j_vec, attempt=intento
    )
    return coords1_stick, coords2_next, "single"


def _apply_coarse_grid(
    coords1_base: np.ndarray,
    coords2_base: np.ndarray,
    cm1: np.ndarray,
    cm2_stick: np.ndarray,
    cand2_idx: int,
    vec_0: np.ndarray,
    i_vec: np.ndarray,
    j_vec: np.ndarray,
    axis_anchor: np.ndarray,
    axis_moving: np.ndarray,
    intento: int,
    algorithm_config: OrchestratorAlgorithmConfig,
    reintento_fn: ReintentoFn,
    rotate_fn: RotateFn,
) -> tuple[np.ndarray, np.ndarray, str]:
    sweep_steps = int(max(1, algorithm_config.cca_coarse_sweep_steps))
    spin_anchor_steps = int(max(1, algorithm_config.cca_coarse_spin_anchor_steps))
    spin_moving_steps = int(max(1, algorithm_config.cca_coarse_spin_moving_steps))
    total = sweep_steps * spin_anchor_steps * spin_moving_steps
    idx = (int(intento) - 1) % total
    block = spin_anchor_steps * spin_moving_steps
    sweep_idx = idx // block
    rem = idx % block
    anchor_idx = rem // spin_moving_steps
    moving_idx = rem % spin_moving_steps

    sweep_attempt = int(round((float(sweep_idx + 1) / float(sweep_steps)) * 360.0))
    sweep_attempt = max(1, min(360, sweep_attempt))
    coords2_swept, _ = reintento_fn(
        coords2_base, cm2_stick, cand2_idx, vec_0, i_vec, j_vec, attempt=sweep_attempt
    )

    anchor_angle = (2.0 * math.pi * float(anchor_idx)) / float(spin_anchor_steps)
    moving_angle = (2.0 * math.pi * float(moving_idx)) / float(spin_moving_steps)

    coords1_next = rotate_fn(coords1_base, cm1, axis_anchor, anchor_angle)
    coords2_next = rotate_fn(coords2_swept, cm2_stick, axis_moving, moving_angle)
    return coords1_next, coords2_next, "coarse_grid"


def _apply_coarse_to_fine(
    coords1_stick: np.ndarray,
    coords2_current: np.ndarray,
    coords1_base: np.ndarray,
    coords2_base: np.ndarray,
    cm1: np.ndarray,
    cm2_stick: np.ndarray,
    cand2_idx: int,
    vec_0: np.ndarray,
    i_vec: np.ndarray,
    j_vec: np.ndarray,
    axis_anchor: np.ndarray,
    axis_moving: np.ndarray,
    intento: int,
    algorithm_config: OrchestratorAlgorithmConfig,
    reintento_fn: ReintentoFn,
    rotate_fn: RotateFn,
) -> tuple[np.ndarray, np.ndarray, str]:
    sweep_steps = int(max(1, algorithm_config.cca_coarse_sweep_steps))
    spin_anchor_steps = int(max(1, algorithm_config.cca_coarse_spin_anchor_steps))
    spin_moving_steps = int(max(1, algorithm_config.cca_coarse_spin_moving_steps))
    total = sweep_steps * spin_anchor_steps * spin_moving_steps
    coarse_fraction = float(algorithm_config.cca_coarse_fine_coarse_fraction)
    coarse_fraction = min(max(coarse_fraction, 0.05), 0.95)
    coarse_budget = max(1, min(total - 1, int(round(total * coarse_fraction))))

    if int(intento) <= coarse_budget:
        if coarse_budget == 1:
            idx = 0
        else:
            idx = int(
                round(((int(intento) - 1) * (total - 1)) / float(coarse_budget - 1))
            )
        block = spin_anchor_steps * spin_moving_steps
        sweep_idx = idx // block
        rem = idx % block
        anchor_idx = rem // spin_moving_steps
        moving_idx = rem % spin_moving_steps

        sweep_attempt = int(round((float(sweep_idx + 1) / float(sweep_steps)) * 360.0))
        sweep_attempt = max(1, min(360, sweep_attempt))
        coords2_swept, _ = reintento_fn(
            coords2_base,
            cm2_stick,
            cand2_idx,
            vec_0,
            i_vec,
            j_vec,
            attempt=sweep_attempt,
        )

        anchor_angle = (2.0 * math.pi * float(anchor_idx)) / float(spin_anchor_steps)
        moving_angle = (2.0 * math.pi * float(moving_idx)) / float(spin_moving_steps)
        coords1_next = rotate_fn(coords1_base, cm1, axis_anchor, anchor_angle)
        coords2_next = rotate_fn(coords2_swept, cm2_stick, axis_moving, moving_angle)
        return coords1_next, coords2_next, "coarse_to_fine_coarse"

    refine_idx = int(intento) - coarse_budget
    refine_deg = float(max(0.0, algorithm_config.cca_coarse_fine_spin_deg))
    refine_rad = np.deg2rad(refine_deg)
    phi = 2.0 * math.pi * float(refine_idx) / float(_GOLDEN_RATIO)
    angle_anchor = refine_rad * float(np.sin(phi))
    angle_moving = refine_rad * float(np.cos(phi))
    coords1_next = rotate_fn(coords1_stick, cm1, axis_anchor, angle_anchor)
    coords2_next = rotate_fn(coords2_current, cm2_stick, axis_moving, angle_moving)
    return coords1_next, coords2_next, "coarse_to_fine_refine"


def _apply_alternate(
    coords1_stick: np.ndarray,
    coords2_current: np.ndarray,
    cm1: np.ndarray,
    i_vec: np.ndarray,
    intento: int,
    reintento_fn: ReintentoFn,
    rotate_fn: RotateFn,
    normalize_axis_fn: NormalizeFn,
    cand2_idx: int,
    cm2_stick: np.ndarray,
    vec_0: np.ndarray,
    j_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if intento % 2 == 0:
        phi = 2.0 * math.pi * float(intento) / float(_GOLDEN_RATIO)
        axis = np.array([i_vec[0], i_vec[1], i_vec[2]], dtype=float)
        axis = normalize_axis_fn(axis, np.array([1.0, 0.0, 0.0]))
        coords1_next = rotate_fn(coords1_stick, cm1, axis, -phi)
        return coords1_next, coords2_current, "alternate_anchor"

    coords2_next, _ = reintento_fn(
        coords2_current, cm2_stick, cand2_idx, vec_0, i_vec, j_vec, attempt=intento
    )
    return coords1_stick, coords2_next, "alternate_moving"


def _apply_dual_jitter(
    coords1_stick: np.ndarray,
    coords2_current: np.ndarray,
    cm1: np.ndarray,
    cm2_stick: np.ndarray,
    cand2_idx: int,
    vec_0: np.ndarray,
    i_vec: np.ndarray,
    j_vec: np.ndarray,
    intento: int,
    algorithm_config: OrchestratorAlgorithmConfig,
    reintento_fn: ReintentoFn,
    rotate_fn: RotateFn,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    coords2_next, _ = reintento_fn(
        coords2_current, cm2_stick, cand2_idx, vec_0, i_vec, j_vec, attempt=intento
    )
    jitter_interval = int(max(1, algorithm_config.cca_dual_jitter_interval))
    if intento % jitter_interval == 0:
        jitter_deg = float(max(0.0, algorithm_config.cca_dual_jitter_deg))
        jitter_rad = np.deg2rad(jitter_deg)
        if jitter_rad > 0.0:
            axis = rng.normal(size=3)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1.0e-12:
                axis = axis / axis_norm
                angle = float(rng.uniform(-jitter_rad, jitter_rad))
                coords1_next = rotate_fn(coords1_stick, cm1, axis, angle)
                return coords1_next, coords2_next, "dual_jitter"
    return coords1_stick, coords2_next, "dual_moving"
