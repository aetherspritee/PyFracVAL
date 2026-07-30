r"""Density-density correlation function :math:`f(r)` for fractal aggregates.

The radius of gyration is a single number, and an aggregate can match it
while having quite the wrong internal structure. FracVAL's own validation
(:cite:p:`Moran2019FracVAL` §4.2) therefore rests on :math:`f(r)`, whose
log-log slope should approach :math:`D_f - d` with :math:`d = 3` over the
scaling range. This module implements that estimator so generated
aggregates can be checked against the metric the paper actually used -
notably densified ones, which are reshaped after generation and so have
the most to prove.

Method (paper Eqs. 14-15)
-------------------------
For each of a set of distances :math:`r`, a copy of the aggregate is
displaced by :math:`r` in a random direction, the volume shared by the
aggregate and its copy is computed analytically, and the result is
averaged over orientations and normalized by the aggregate's own volume:

.. math::
    f(r^{(k)}) = \frac{1}{n_{or}}\sum_{n_{or}} V_{int}^{(k)} / V_a

with radii spaced geometrically,
:math:`r^{(k)} = R_{min}(R_{max}/R_{min})^{k/(n_{it}-1)}`, from
:math:`R_{min} = r_{p,geo}/10` to :math:`R_{max} = \delta R_g`
(:math:`\delta = 3.5` is ample).

Two spheres of radii :math:`r_1, r_2` whose centers are :math:`d` apart
share the lens volume

.. math::
    V = \frac{\pi (r_1 + r_2 - d)^2\,(d^2 + 2d r_2 - 3r_2^2
        + 2d r_1 + 6 r_1 r_2 - 3 r_1^2)}{12 d}

which is exact, so no binning or kernel choice enters the estimate.

Cost
----
Naively this is :math:`O(N^2)` per orientation per radius. Since only
pairs closer than the sum of two radii contribute anything, a k-d tree
prunes it to the handful of genuinely overlapping pairs, which is what
makes the paper's :math:`n_{or}=300` affordable at :math:`N=1024`.
"""

import logging

import numpy as np
from scipy.spatial import cKDTree

from . import fractal

logger = logging.getLogger(__name__)


def sphere_intersection_volume(
    r1: np.ndarray, r2: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Exact shared volume of sphere pairs (vectorized).

    Handles all three regimes: disjoint (0), one fully containing the
    other (the smaller sphere's whole volume), and partial overlap (the
    lens formula).

    Parameters
    ----------
    r1, r2, d : np.ndarray
        Broadcastable arrays of the two radii and the center separation.

    Returns
    -------
    np.ndarray
        Intersection volume for each triple.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    d = np.asarray(d, dtype=float)

    out = np.zeros(np.broadcast(r1, r2, d).shape, dtype=float)

    contained = d <= np.abs(r1 - r2)
    overlapping = (~contained) & (d < (r1 + r2))

    if np.any(contained):
        r_small = np.minimum(r1, r2)
        out[contained] = (
            (4.0 / 3.0) * np.pi * np.broadcast_to(r_small, out.shape)[contained] ** 3
        )

    if np.any(overlapping):
        rr1 = np.broadcast_to(r1, out.shape)[overlapping]
        rr2 = np.broadcast_to(r2, out.shape)[overlapping]
        dd = np.broadcast_to(d, out.shape)[overlapping]
        out[overlapping] = (
            np.pi
            * (rr1 + rr2 - dd) ** 2
            * (
                dd**2
                + 2.0 * dd * rr2
                - 3.0 * rr2**2
                + 2.0 * dd * rr1
                + 6.0 * rr1 * rr2
                - 3.0 * rr1**2
            )
            / (12.0 * dd)
        )
    return out


def _intersection_volume_with_shifted_copy(
    coords: np.ndarray,
    radii: np.ndarray,
    tree: cKDTree,
    shift: np.ndarray,
    max_pair_reach: float,
) -> float:
    """Total volume shared between the aggregate and a shifted copy of it."""
    shifted = coords + shift
    shifted_tree = cKDTree(shifted)
    # Only pairs within the largest possible radius sum can contribute.
    pairs = tree.query_ball_tree(shifted_tree, r=max_pair_reach)

    idx_a: list[int] = []
    idx_b: list[int] = []
    for i, partners in enumerate(pairs):
        if partners:
            idx_a.extend([i] * len(partners))
            idx_b.extend(partners)
    if not idx_a:
        return 0.0

    ia = np.asarray(idx_a, dtype=int)
    ib = np.asarray(idx_b, dtype=int)
    d = np.linalg.norm(coords[ia] - shifted[ib], axis=1)
    return float(np.sum(sphere_intersection_volume(radii[ia], radii[ib], d)))


def density_correlation(
    coords: np.ndarray,
    radii: np.ndarray,
    n_orientations: int = 100,
    n_radii: int = 40,
    delta: float = 3.5,
    rng: np.random.Generator | None = None,
    densities: np.ndarray | None = None,
) -> dict:
    """Estimate the density-density correlation function of an aggregate.

    Parameters
    ----------
    coords, radii : np.ndarray
        The aggregate geometry.
    n_orientations : int
        Random displacement directions averaged per radius. The paper uses
        300; 100 is usually enough to see the slope and is 3x cheaper.
    n_radii : int
        Number of geometrically-spaced radii.
    delta : float
        Largest radius as a multiple of Rg.
    densities : np.ndarray, optional
        Only used for the Rg that sets the radius range; f(r) itself is a
        purely geometric (volume) quantity and is unaffected by density.

    Returns
    -------
    dict
        ``r`` (radii), ``f`` (correlation values), ``r_over_rp`` (radii
        normalized by the geometric-mean primary radius), ``rg``, and
        ``rp_geo``.
    """
    _rng = rng if rng is not None else np.random.default_rng()
    n = coords.shape[0]
    if n == 0:
        raise ValueError("density_correlation: empty aggregate")

    rp_geo = float(np.exp(np.mean(np.log(radii[radii > 0]))))
    rg = fractal.compute_empirical_rg_polydisperse(coords, radii, densities)
    if rg <= 0.0:
        raise ValueError("density_correlation: aggregate has non-positive Rg")

    r_min = rp_geo / 10.0
    r_max = delta * rg
    k = np.arange(n_radii)
    r_values = r_min * (r_max / r_min) ** (k / max(n_radii - 1, 1))

    volume_total = float(np.sum((4.0 / 3.0) * np.pi * radii**3))
    tree = cKDTree(coords)
    max_pair_reach = 2.0 * float(np.max(radii))

    f_values = np.zeros(n_radii, dtype=float)
    for ki, r in enumerate(r_values):
        # Uniform random directions on the sphere.
        vecs = _rng.normal(size=(n_orientations, 3))
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        acc = 0.0
        for v in vecs:
            acc += _intersection_volume_with_shifted_copy(
                coords, radii, tree, v * r, max_pair_reach
            )
        f_values[ki] = acc / n_orientations / volume_total

    return {
        "r": r_values,
        "f": f_values,
        "r_over_rp": r_values / rp_geo,
        "rg": rg,
        "rp_geo": rp_geo,
    }


def fit_correlation_slope(
    result: dict,
    fit_lo_over_rp: float = 2.0,
    fit_hi_over_rg: float = 1.0,
) -> dict:
    """Fit the log-log slope of f(r), which should approach ``Df - 3``.

    The fit window matters and is not a free choice. Below roughly
    ``2 r_p`` the curve reflects single-particle overlap rather than
    aggregate structure (the paper notes this explicitly), and beyond
    about ``Rg`` the finite size of the aggregate cuts the power law off.
    Only the range between them carries the fractal signal, and for small
    aggregates that range can be too short to fit meaningfully - which is
    itself worth reporting rather than hiding.

    Returns
    -------
    dict
        ``slope``, ``df_estimate`` (``slope + 3``), ``r_squared``,
        ``n_points`` used, and the window actually used.
    """
    r_over_rp = result["r_over_rp"]
    f = result["f"]
    rg_over_rp = result["rg"] / result["rp_geo"]

    mask = (
        (r_over_rp >= fit_lo_over_rp)
        & (r_over_rp <= fit_hi_over_rg * rg_over_rp)
        & (f > 0.0)
    )
    n_points = int(np.count_nonzero(mask))
    if n_points < 3:
        return {
            "slope": float("nan"),
            "df_estimate": float("nan"),
            "r_squared": float("nan"),
            "n_points": n_points,
            "fit_lo_over_rp": fit_lo_over_rp,
            "fit_hi_over_rp": fit_hi_over_rg * rg_over_rp,
        }

    x = np.log(r_over_rp[mask])
    y = np.log(f[mask])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "slope": float(slope),
        "df_estimate": float(slope) + 3.0,
        "r_squared": r_squared,
        "n_points": n_points,
        "fit_lo_over_rp": fit_lo_over_rp,
        "fit_hi_over_rp": fit_hi_over_rg * rg_over_rp,
    }
