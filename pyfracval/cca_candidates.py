"""CCA candidate selection and scoring utilities.

Pure functions for candidate pair selection, leaf mask computation,
surface accessibility, and candidate scoring in CCA aggregation.

Functions
---------
leaf_mask_for_cluster
    Identify leaf-like monomers (degree <= 1) in a cluster.
candidate_leaf_class
    Classify a pair of clusters as LL, LN, or NN based on leaf status.
candidate_score
    Heuristic candidate score in [0, 1] — larger means more promising.
pair_overlap
    Compute overlap fraction between a pair of particles.
select_candidates
    Select candidate particles from cluster 2 for sticking.
pick_candidate_pair
    Pick a candidate pair from the selection matrix.
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def leaf_mask_for_cluster(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Return bool mask of leaf-like monomers (degree <= 1) within a cluster.

    A monomer is a "leaf" if it has at most 1 contact neighbor in the
    aggregate connectivity graph.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of particle centre coordinates.
    radii : np.ndarray
        (N,) array of particle radii.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates a leaf monomer.
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.ones(1, dtype=bool)

    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    d_sq = np.sum(diffs * diffs, axis=2)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    thr_sq = (r_sum + 1.0e-9) * (r_sum + 1.0e-9)
    contact = d_sq <= thr_sq
    np.fill_diagonal(contact, False)
    degree = np.sum(contact, axis=1)
    return degree <= 1


def candidate_leaf_class(leaf1: bool, leaf2: bool) -> str:
    """Classify a candidate pair by leaf status.

    Parameters
    ----------
    leaf1 : bool
        Whether the first cluster is a leaf.
    leaf2 : bool
        Whether the second cluster is a leaf.

    Returns
    -------
    str
        One of ``"LL"``, ``"LN"``, or ``"NN"``.
    """
    if leaf1 and leaf2:
        return "LL"
    if leaf1 or leaf2:
        return "LN"
    return "NN"


def candidate_score(
    coords1: np.ndarray,
    radii1: np.ndarray,
    cm1: np.ndarray,
    cand1_idx: int,
    coords2: np.ndarray,
    radii2: np.ndarray,
    cm2: np.ndarray,
    cand2_idx: int,
    gamma_pc: float,
    leaf_cls: str,
) -> float:
    """Heuristic candidate score in [0,1], larger means more promising.

    Parameters
    ----------
    coords1, coords2 : np.ndarray
        Particle coordinates for clusters 1 and 2.
    radii1, radii2 : np.ndarray
        Particle radii for clusters 1 and 2.
    cm1, cm2 : np.ndarray
        Centre-of-mass vectors for clusters 1 and 2.
    cand1_idx, cand2_idx : int
        Indices of candidate particles in each cluster.
    gamma_pc : float
        Target centre-to-centre distance.
    leaf_cls : str
        Leaf classification (``"LL"``, ``"LN"``, ``"NN"``).

    Returns
    -------
    float
        Candidate score in [0, 1].
    """
    d1 = float(np.linalg.norm(coords1[cand1_idx] - cm1))
    d2 = float(np.linalg.norm(coords2[cand2_idx] - cm2))
    s1 = d1 + float(radii1[cand1_idx])
    s2 = d2 + float(radii2[cand2_idx])

    # Radial compatibility against gamma shell relation.
    err = abs((s1 + s2) - gamma_pc) / max(gamma_pc, 1.0e-12)
    tau_r = 0.08
    radial = math.exp(-err / tau_r)

    # Class prior from observed empirical success tendency.
    if leaf_cls == "LL":
        leaf_prior = 1.0
    elif leaf_cls == "LN":
        leaf_prior = 0.45
    else:
        leaf_prior = 0.10

    score = 0.5 * leaf_prior + 0.5 * radial
    return max(0.0, min(1.0, score))


def pair_overlap(
    coords1: np.ndarray,
    radii1: np.ndarray,
    coords2: np.ndarray,
    radii2: np.ndarray,
) -> float:
    """Compute overlap fraction between a single pair of particles.

    Parameters
    ----------
    coords1 : np.ndarray
        (3,) centre coordinate of particle 1.
    radii1 : np.ndarray
        (1,) radius of particle 1 (or scalar).
    coords2 : np.ndarray
        (3,) centre coordinate of particle 2.
    radii2 : np.ndarray
        (1,) radius of particle 2 (or scalar.

    Returns
    -------
    float
        Overlap fraction (1 - distance/radius_sum), or ``-inf`` if not
        overlapping.
    """
    d_sq = float(np.sum((coords1 - coords2) ** 2))
    i = 0  # single particle
    j = 0  # single particle
    r1 = float(radii1[i]) if radii1.ndim > 0 else float(radii1)
    r2 = float(radii2[j]) if radii2.ndim > 0 else float(radii2)
    radius_sum = r1 + r2
    r_sq = radius_sum * radius_sum
    if d_sq > r_sq:
        return -np.inf
    dist = math.sqrt(d_sq)
    return 1.0 - dist / radius_sum
