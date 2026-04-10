"""Utility functions for vector operations and array manipulation.

.. deprecated::
    This module is being split into domain-specific sub-modules.
    Import from the specific module instead:

    - :mod:`pyfracval.geometry` — Rodrigues rotation, sphere intersection
    - :mod:`pyfracval.fractal` — Fractal metrics and validation
    - :mod:`pyfracval.overlap` — Overlap calculation dispatch
    - :mod:`pyfracval.cca_kernels` — CCA-specific JIT kernels
    - :mod:`pyfracval.pca_kernels` — PCA-specific JIT kernels

All symbols remain importable from this module for backward compatibility.
"""

import logging

import numpy as np

from .cca_kernels import (
    _GOLDEN_RATIO,
    _TWO_PI,
    _cca_reintento_kernel,
    batch_check_overlaps_cca,
    batch_rotate_cluster_cca,
)
from .fractal import (
    calculate_cluster_properties,
    calculate_mass,
    calculate_rg,
    compute_empirical_rg,
    compute_pair_correlation_dimensions,
    gamma_calculation,
    validate_fractal_structure,
)
from .geometry import (
    FLOATING_POINT_ERROR,
    _rodrigues_rotation_2d,
    _two_sphere_intersection_kernel,
    rodrigues_rotation,
    two_sphere_intersection,
)
from .overlap import (
    PARALLEL_OVERLAP_THRESHOLD,
    calculate_max_overlap_cca,
    calculate_max_overlap_cca_auto,
    calculate_max_overlap_cca_fast,
    calculate_max_overlap_cca_parallel,
    calculate_max_overlap_pca,
    calculate_max_overlap_pca_auto,
    calculate_max_overlap_pca_fast,
    calculate_max_overlap_pca_parallel,
)
from .pca_kernels import (
    batch_calculate_positions_pca,
    batch_check_overlaps_pca,
)

logger = logging.getLogger(__name__)


def shuffle_array(
    arr: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Randomly shuffle elements of a 1D array in-place (Fisher-Yates).

    Modifies the input array directly. Mimics Fortran randsample behavior.

    Parameters
    ----------
    arr :
        The 1D NumPy array to shuffle.
    rng : np.random.Generator | None, optional
        A NumPy Generator instance for reproducible randomness. If None,
        a fresh Generator is created.

    Returns
    -------
        The input `arr`, modified in-place.
    """
    _rng = rng if rng is not None else np.random.default_rng()
    n = len(arr)
    for i in range(n - 1):
        # Random index from i to n-1
        j = int(_rng.integers(i, n))
        # Swap elements
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def sort_clusters(i_orden: np.ndarray) -> np.ndarray:
    """Sort cluster information array `i_orden` by cluster size (count).

    Parameters
    ----------
    i_orden : np.ndarray
        The Mx3 NumPy array [start_idx, end_idx, count].

    Returns
    -------
    np.ndarray
        A new Mx3 array sorted by the 'count' column (column index 2).

    Raises
    ------
    ValueError
        If `i_orden` is not an Mx3 array.
    """
    if i_orden.ndim != 2 or i_orden.shape[1] != 3:
        raise ValueError("i_orden must be an Mx3 array")
    if i_orden.shape[0] == 0:
        return i_orden  # Return empty array if input is empty

    # Get the indices that would sort the 'count' column (index 2)
    sort_indices = np.argsort(i_orden[:, 2], kind="stable")
    return i_orden[sort_indices]


# def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     """Calculate the cross product of two 3D vectors.

#     Parameters
#     ----------
#     a : np.ndarray
#         The first 3D vector.
#     b : np.ndarray
#         The second 3D vector.

#     Returns
#     -------
#     np.ndarray
#         The 3D cross product vector.

#     See Also
#     --------
#     numpy.cross : NumPy's implementation.
#     """
#     return np.cross(a, b)


# def normalize(v: np.ndarray) -> np.ndarray:
#     """Normalize a vector to unit length.

#     Returns a zero vector if the input vector's norm is close to zero.

#     Parameters
#     ----------
#     v : np.ndarray
#         The vector (1D array) to normalize.

#     Returns
#     -------
#     np.ndarray
#         The normalized vector, or a zero vector if the norm is negligible.
#     """
#     norm = np.linalg.norm(v)
#     if norm < 1e-12:  # Use tolerance instead of exact zero check
#         return np.zeros_like(v)
#     return v / norm


__all__ = [
    "FLOATING_POINT_ERROR",
    "rodrigues_rotation",
    "_rodrigues_rotation_2d",
    "two_sphere_intersection",
    "_two_sphere_intersection_kernel",
    "calculate_mass",
    "calculate_rg",
    "gamma_calculation",
    "calculate_cluster_properties",
    "compute_empirical_rg",
    "compute_pair_correlation_dimensions",
    "validate_fractal_structure",
    "PARALLEL_OVERLAP_THRESHOLD",
    "calculate_max_overlap_cca",
    "calculate_max_overlap_pca",
    "calculate_max_overlap_pca_fast",
    "calculate_max_overlap_cca_fast",
    "calculate_max_overlap_pca_parallel",
    "calculate_max_overlap_cca_parallel",
    "calculate_max_overlap_pca_auto",
    "calculate_max_overlap_cca_auto",
    "_GOLDEN_RATIO",
    "_TWO_PI",
    "_cca_reintento_kernel",
    "batch_check_overlaps_cca",
    "batch_rotate_cluster_cca",
    "batch_calculate_positions_pca",
    "batch_check_overlaps_pca",
    "shuffle_array",
    "sort_clusters",
]
