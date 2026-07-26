"""JAX ports of PyFracVAL's hottest numba kernels, for a head-to-head speed
comparison (see ``jax_vs_numba_benchmark.py`` and
``docs/source/gpu_acceleration.md``).

Ported kernels (identified as the numerical hot path - see
``docs/source/gpu_acceleration.md`` for the survey that picked these):

- ``rodrigues_rotation_jax`` - mirrors ``geometry._rodrigues_rotation_2d``.
  Called once per cluster rotation in PCA/CCA.
- ``max_overlap_pairwise_jax`` - mirrors ``overlap.calculate_max_overlap_cca_fast``.
  Full O(N1*N2) distance matrix + reduction - JAX/GPU has no equivalent to
  the numba fast-path's data-dependent early exit, so this always does the
  full computation.
- ``max_overlap_single_vs_agg_jax`` - mirrors
  ``overlap.calculate_max_overlap_pca_fast`` (one new particle vs an
  existing aggregate).
- ``batch_check_overlaps_pca_jax`` - mirrors
  ``pca_kernels.batch_check_overlaps_pca`` (K candidate positions checked
  against one aggregate in a single call - the batched shape JAX/GPU is
  actually suited for, since it amortizes dispatch overhead over K).

This module is NOT imported by production ``pyfracval`` code - it exists
solely for the benchmark. Requires the ``jax_bench`` dependency group:
``uv sync --group jax_bench``.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

# Persistent compilation cache - without this every fresh process
# recompiles every kernel from scratch, which would dominate any timing
# comparison against numba's on-disk cache=True kernels. Mirrors
# NUMBA_CACHE_DIR (see devenv.nix); JAX needs an explicit opt-in to cache
# *fast*-compiling functions (default threshold is 1s of compile time,
# these tiny kernels compile faster than that).
_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_cache")
)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", True)


@jax.jit
def rodrigues_rotation_jax(
    vectors: jnp.ndarray, axis: jnp.ndarray, angle: jnp.ndarray
) -> jnp.ndarray:
    """Rotate (N,3) vectors around a (3,) axis by ``angle`` radians."""
    axis_n = axis / jnp.linalg.norm(axis)
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    dot_kv = vectors @ axis_n
    cross_kv = jnp.cross(axis_n[None, :], vectors)
    return (
        vectors * cos_a
        + cross_kv * sin_a
        + axis_n[None, :] * dot_kv[:, None] * (1.0 - cos_a)
    )


@jax.jit
def max_overlap_pairwise_jax(
    coords1: jnp.ndarray,
    radii1: jnp.ndarray,
    coords2: jnp.ndarray,
    radii2: jnp.ndarray,
) -> jnp.ndarray:
    """Max overlap between two point sets (mirrors calculate_max_overlap_cca_fast).

    overlap(i,j) = 1 - dist(i,j) / (r1_i + r2_j), clamped to only count
    pairs that actually overlap (dist <= radius_sum), matching the numba
    fast path's semantics.
    """
    diff = coords1[:, None, :] - coords2[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    radius_sum = radii1[:, None] + radii2[None, :]
    overlap = jnp.where(dist <= radius_sum, 1.0 - dist / radius_sum, 0.0)
    return jnp.max(overlap)


@jax.jit
def max_overlap_single_vs_agg_jax(
    coords_agg: jnp.ndarray,
    radii_agg: jnp.ndarray,
    coord_new: jnp.ndarray,
    radius_new: jnp.ndarray,
) -> jnp.ndarray:
    """Max overlap of one new particle vs an aggregate (mirrors
    calculate_max_overlap_pca_fast)."""
    diff = coords_agg - coord_new[None, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    radius_sum = radii_agg + radius_new
    overlap = jnp.where(dist <= radius_sum, 1.0 - dist / radius_sum, 0.0)
    return jnp.max(overlap)


@jax.jit
def batch_check_overlaps_pca_jax(
    coords_agg: jnp.ndarray,
    radii_agg: jnp.ndarray,
    candidate_positions: jnp.ndarray,
    radius_new: jnp.ndarray,
) -> jnp.ndarray:
    """Max overlap for a batch of K candidate positions vs one aggregate
    (mirrors pca_kernels.batch_check_overlaps_pca) - shape (K,)."""
    diff = coords_agg[None, :, :] - candidate_positions[:, None, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    radius_sum = radii_agg[None, :] + radius_new
    overlap = jnp.where(dist <= radius_sum, 1.0 - dist / radius_sum, 0.0)
    return jnp.max(overlap, axis=1)
