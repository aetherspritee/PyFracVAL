"""Soft potential relaxation for cluster-cluster aggregation.

Implements force-driven relaxation using harmonic repulsion potentials
as an alternative to rigid-body docking for hard regimes (Df≥2.25).

The approach:
1. Initial placement at target gamma_pc distance
2. Compute repulsive forces from overlapping particles
3. Add restoring force to maintain gamma_pc constraint
4. Gradient descent until convergence
5. Return relaxed configuration
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba JIT kernels for force computation
# ---------------------------------------------------------------------------


@jit(nopython=True, fastmath=True, cache=True)
def _compute_forces_kernel(
    coords: np.ndarray,
    radii: np.ndarray,
    k_repulsion: float,
    cm_target: np.ndarray,
    k_gamma: float,
    gamma_target: float,
) -> Tuple[np.ndarray, float, float]:
    """Compute forces on all particles.

    Forces consist of:
    - Repulsive forces from overlapping particles (harmonic potential)
    - Restoring force on cluster CM to maintain gamma_pc constraint

    Returns:
        forces: Nx3 array of forces on each particle
        max_overlap: maximum overlap fraction
        total_energy: total potential energy
    """
    n = coords.shape[0]
    forces = np.zeros((n, 3), dtype=np.float64)
    total_energy = 0.0
    max_overlap = 0.0

    # Compute repulsive forces from overlaps
    for i in prange(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz

            r_sum = radii[i] + radii[j]

            if dist_sq < r_sum * r_sum and dist_sq > 1e-20:
                dist = np.sqrt(dist_sq)
                overlap = r_sum - dist

                # Track max overlap
                ov_frac = overlap / min(radii[i], radii[j])
                if ov_frac > max_overlap:
                    max_overlap = ov_frac

                # Harmonic potential: E = 0.5 * k * overlap^2
                # Force magnitude: F = k * overlap
                force_mag = k_repulsion * overlap

                # Unit vector from j to i
                ux, uy, uz = dx / dist, dy / dist, dz / dist

                # Forces in opposite directions
                forces[i, 0] += force_mag * ux
                forces[i, 1] += force_mag * uy
                forces[i, 2] += force_mag * uz
                forces[j, 0] -= force_mag * ux
                forces[j, 1] -= force_mag * uy
                forces[j, 2] -= force_mag * uz

                total_energy += 0.5 * k_repulsion * overlap * overlap

    # Add restoring force to maintain gamma_pc constraint
    # Compute current CM
    masses = radii**3
    total_mass = np.sum(masses)
    if total_mass > 1e-30:
        cm_current = np.zeros(3, dtype=np.float64)
        for i in range(n):
            cm_current[0] += coords[i, 0] * masses[i]
            cm_current[1] += coords[i, 1] * masses[i]
            cm_current[2] += coords[i, 2] * masses[i]
        cm_current /= total_mass

        # Vector from target to current CM
        d_cm = cm_current - cm_target
        dist_cm = np.sqrt(d_cm[0] ** 2 + d_cm[1] ** 2 + d_cm[2] ** 2)

        if dist_cm > 1e-10:
            # Spring potential: E = 0.5 * k * (dist_cm)^2
            # This pulls CM back to target position
            force_scale = k_gamma * dist_cm / total_mass
            for i in range(n):
                forces[i, 0] -= force_scale * masses[i] * d_cm[0] / dist_cm
                forces[i, 1] -= force_scale * masses[i] * d_cm[1] / dist_cm
                forces[i, 2] -= force_scale * masses[i] * d_cm[2] / dist_cm

            total_energy += 0.5 * k_gamma * dist_cm * dist_cm

    return forces, max_overlap, total_energy


@jit(nopython=True, fastmath=True, cache=True)
def _compute_max_overlap_kernel(
    coords: np.ndarray,
    radii: np.ndarray,
) -> float:
    """Compute maximum overlap fraction (for convergence checking)."""
    n = coords.shape[0]
    max_overlap = 0.0

    for i in prange(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz

            r_sum = radii[i] + radii[j]

            if dist_sq < r_sum * r_sum and dist_sq > 1e-20:
                dist = np.sqrt(dist_sq)
                overlap = r_sum - dist
                ov_frac = overlap / min(radii[i], radii[j])
                if ov_frac > max_overlap:
                    max_overlap = ov_frac

    return max_overlap


# ---------------------------------------------------------------------------
# Python-level functions
# ---------------------------------------------------------------------------


def compute_forces(
    coords: np.ndarray,
    radii: np.ndarray,
    k_repulsion: float = 10.0,
    cm_target: np.ndarray | None = None,
    k_gamma: float = 1.0,
    gamma_target: float = 0.0,
) -> Tuple[np.ndarray, float, float]:
    """Compute forces on all particles.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle coordinates.
    radii : np.ndarray
        N array of particle radii.
    k_repulsion : float
        Spring constant for harmonic repulsion (default 10.0).
    cm_target : np.ndarray, optional
        Target center of mass position for gamma constraint.
    k_gamma : float
        Spring constant for gamma constraint (default 1.0).
    gamma_target : float
        Target gamma distance (not used directly, only for CM constraint).

    Returns
    -------
    forces : np.ndarray
        Nx3 array of forces.
    max_overlap : float
        Maximum overlap fraction.
    total_energy : float
        Total potential energy.
    """
    if cm_target is None:
        # If no target CM, just use current CM (no constraint)
        masses = radii**3
        total_mass = np.sum(masses)
        if total_mass > 1e-30:
            cm_target = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
        else:
            cm_target = np.zeros(3)

    return _compute_forces_kernel(
        coords.astype(np.float64),
        radii.astype(np.float64),
        float(k_repulsion),
        cm_target.astype(np.float64),
        float(k_gamma),
        float(gamma_target),
    )


def soft_relaxation(
    coords: np.ndarray,
    radii: np.ndarray,
    cm_target: np.ndarray,
    k_repulsion: float = 10.0,
    k_gamma: float = 1.0,
    max_iters: int = 100,
    tol_overlap: float = 1e-4,
    tol_force: float = 1e-3,
    learning_rate: float = 0.1,
    verbose: bool = False,
) -> Tuple[np.ndarray, bool, dict]:
    """Relax configuration using gradient descent on soft potential.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of initial coordinates.
    radii : np.ndarray
        N array of particle radii.
    cm_target : np.ndarray
        Target center of mass position.
    k_repulsion : float
        Spring constant for repulsion (default 10.0).
    k_gamma : float
        Spring constant for gamma constraint (default 1.0).
    max_iters : int
        Maximum gradient descent iterations.
    tol_overlap : float
        Convergence tolerance for max overlap.
    tol_force : float
        Convergence tolerance for max force magnitude.
    learning_rate : float
        Gradient descent step size (default 0.1).
    verbose : bool
        Print progress information.

    Returns
    -------
    coords : np.ndarray
        Relaxed coordinates.
    success : bool
        True if converged within tolerances.
    info : dict
        Diagnostics (iterations, final energy, max overlap, etc.).
    """
    current = coords.copy().astype(np.float64)
    n = current.shape[0]

    info = {
        "iterations": 0,
        "final_energy": 0.0,
        "final_max_overlap": 1.0,
        "final_max_force": 0.0,
        "converged": False,
        "convergence_reason": "",
    }

    prev_energy = float("inf")
    patience = 10
    no_improve_count = 0

    for iteration in range(max_iters):
        # Compute forces
        forces, max_overlap, energy = compute_forces(
            current, radii, k_repulsion, cm_target, k_gamma
        )

        # Check convergence
        max_force = np.max(np.linalg.norm(forces, axis=1))

        if max_overlap <= tol_overlap and max_force <= tol_force:
            info["converged"] = True
            info["convergence_reason"] = "tolerance"
            break

        # Check energy improvement (stagnation detection)
        if iteration > 0:
            if energy >= prev_energy - 1e-10:
                no_improve_count += 1
                if no_improve_count >= patience:
                    info["convergence_reason"] = "stagnation"
                    break
            else:
                no_improve_count = 0

        prev_energy = energy

        # Adaptive learning rate
        current_lr = learning_rate * (0.99**iteration)

        # Gradient descent step
        current += current_lr * forces

        if verbose and iteration % 10 == 0:
            logger.debug(
                f"Iter {iteration}: E={energy:.4e}, max_ov={max_overlap:.4e}, "
                f"max_F={max_force:.4e}"
            )

    # Final evaluation
    forces, max_overlap, energy = compute_forces(
        current, radii, k_repulsion, cm_target, k_gamma
    )
    max_force = np.max(np.linalg.norm(forces, axis=1))

    info["iterations"] = iteration + 1
    info["final_energy"] = energy
    info["final_max_overlap"] = max_overlap
    info["final_max_force"] = max_force

    success = max_overlap <= tol_overlap

    return current, success, info


def soft_sticking(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
    gamma_pc: float,
    cm1: np.ndarray,
    cm2: np.ndarray,
    candidate_particle_idx1: int,
    candidate_particle_idx2: int,
    k_repulsion: float = 10.0,
    k_gamma: float = 1.0,
    gamma_tolerance: float = 0.05,
    max_iters: int = 100,
    tol_overlap: float = 1e-4,
    learning_rate: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
    """Stick two clusters using soft potential relaxation.

    This is an alternative to rigid-body docking that allows small
    deviations from exact gamma_pc distance to resolve overlaps.

    Parameters
    ----------
    coords1, coords2 : np.ndarray
        Coordinates of clusters 1 and 2.
    radii1, radii2 : np.ndarray
        Radii of clusters 1 and 2.
    gamma_pc : float
        Target distance between cluster centers.
    cm1, cm2 : np.ndarray
        Current centers of mass.
    candidate_particle_idx1, candidate_particle_idx2 : int
        Indices of particles that should be brought into contact.
    k_repulsion : float
        Repulsion spring constant.
    k_gamma : float
        Gamma constraint spring constant.
    gamma_tolerance : float
        Allowed fractional deviation from gamma_pc (default 5%).
    max_iters : int
        Maximum relaxation iterations.
    tol_overlap : float
        Overlap convergence tolerance.
    learning_rate : float
        Gradient descent step size.

    Returns
    -------
    new_coords1, new_coords2 : np.ndarray
        Relaxed coordinates for both clusters.
    success : bool
        True if relaxation converged within tolerances.
    info : dict
        Diagnostics including final gamma error.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]

    # Combine coordinates and radii
    combined_coords = np.vstack([coords1, coords2]).astype(np.float64)
    combined_radii = np.concatenate([radii1, radii2]).astype(np.float64)

    # Compute target CM positions
    # Place CM2 at distance gamma_pc from CM1 in direction of candidate
    vec = coords1[candidate_particle_idx1] - cm1
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    cm2_target = cm1 + gamma_pc * vec

    # Initial placement: position cluster 2
    displacement = cm2_target - cm2
    combined_coords[n1:, :] += displacement

    # Compute target CM for combined system
    masses = combined_radii**3
    total_mass = np.sum(masses)
    if total_mass > 1e-30:
        # The CM should stay near CM1 (larger cluster)
        cm_target = cm1.copy()
    else:
        cm_target = np.zeros(3)

    # Run relaxation
    relaxed_coords, success, relax_info = soft_relaxation(
        combined_coords,
        combined_radii,
        cm_target,
        k_repulsion=k_repulsion,
        k_gamma=k_gamma,
        max_iters=max_iters,
        tol_overlap=tol_overlap,
        learning_rate=learning_rate,
    )

    # Split back into separate clusters
    new_coords1 = relaxed_coords[:n1, :]
    new_coords2 = relaxed_coords[n1:, :]

    # Compute final gamma and error
    final_cm1 = np.average(new_coords1, axis=0, weights=radii1**3)
    final_cm2 = np.average(new_coords2, axis=0, weights=radii2**3)
    final_gamma = np.linalg.norm(final_cm2 - final_cm1)
    gamma_error = abs(final_gamma - gamma_pc) / gamma_pc if gamma_pc > 0 else 0.0

    # Success requires both low overlap AND gamma within tolerance
    success = success and gamma_error <= gamma_tolerance

    info = {
        **relax_info,
        "target_gamma": gamma_pc,
        "final_gamma": final_gamma,
        "gamma_error": gamma_error,
        "gamma_ok": gamma_error <= gamma_tolerance,
    }

    return new_coords1, new_coords2, success, info
