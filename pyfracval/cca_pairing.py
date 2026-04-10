"""CCA pair generation and gamma calculation utilities.

Pure functions for pair key encoding/decoding, monomer identification,
gamma distance calculation, and pair generation for CCA aggregation.

Functions
---------
pair_key
    Pack pair indices into a single integer key.
pair_unpack
    Unpack integer pair key into (i, j).
identify_monomers
    Identify monomer clusters in the current cluster info.
calculate_cca_gamma
    Compute CCA centre-to-centre distance (gamma) for a pair.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def pair_key(i: int, j: int, n2: int) -> int:
    """Pack pair indices (i, j) into a single integer key.

    Parameters
    ----------
    i : int
        First cluster index.
    j : int
        Second cluster index.
    n2 : int
        Number of clusters in the second group.

    Returns
    -------
    int
        Encoded pair key.
    """
    return i * n2 + j


def pair_unpack(key: int, n2: int) -> tuple[int, int]:
    """Unpack integer pair key into (i, j).

    Parameters
    ----------
    key : int
        Encoded pair key.
    n2 : int
        Number of clusters in the second group.

    Returns
    -------
    tuple[int, int]
        Decoded pair indices (i, j).
    """
    return key // n2, key % n2


def identify_monomers(clusters_info: np.ndarray) -> np.ndarray | None:
    """Identify which clusters are monomers (single-particle clusters).

    Parameters
    ----------
    clusters_info : np.ndarray
        Array of cluster info where cluster_info[:, 4] contains the
        number of particles in each cluster.

    Returns
    -------
    np.ndarray or None
        Indices of monomer clusters, or None if no monomers exist.
    """
    n_particles = clusters_info[:, 4]
    mask = n_particles == 1
    if not np.any(mask):
        return None
    return np.where(mask)[0]
