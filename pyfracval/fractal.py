"""Fractal metrics and validation for PyFracVAL.

Functions for computing mass, radius of gyration, gamma values,
cluster properties, and validating fractal structure.

Functions
---------
calculate_mass
    Compute total particle mass (proportional to r^3).
calculate_rg
    Compute theoretical radius of gyration from fractal scaling law.
gamma_calculation
    Compute the centre-to-centre distance (gamma) for PCA/CCA sticking.
calculate_cluster_properties
    Compute total mass, Rg, centre-of-mass, and max radius of a cluster.
compute_empirical_rg
    Compute empirical Rg from actual particle coordinates.
compute_pair_correlation_dimensions
    Estimate fractal dimension from pair-correlation (mass-radius) scaling.
validate_fractal_structure
    Validate that generated aggregate matches target fractal parameters.
"""

import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def calculate_mass(
    radii: np.ndarray, densities: np.ndarray | None = None
) -> np.ndarray:
    """Calculate particle mass from radii and (optionally) per-particle density.

    ``m_i = (4/3) pi r_i^3 rho_i``.

    Parameters
    ----------
    radii : np.ndarray
        Array of particle radii.
    densities : np.ndarray, optional
        Per-particle densities. ``None`` (the default) means uniform
        density, in which case mass is proportional to r^3 and every
        density-aware quantity reduces exactly to its single-material
        form. Supplying densities is what makes *heterogeneous*
        aggregates - different materials, not just different sizes -
        physically meaningful, since the center of mass, radius of
        gyration and the Gamma equation are all mass-weighted.

    Returns
    -------
    np.ndarray
        Array of corresponding particle masses.
    """
    volume = (4.0 / 3.0) * np.pi * (radii**3)
    if densities is None:
        return volume
    return volume * densities


def resolve_densities(
    densities: np.ndarray | None, n: int, context: str = "densities"
) -> np.ndarray | None:
    """Validate an optional per-particle density array against a count.

    Returns the array unchanged (as float) when supplied, or ``None`` for
    the uniform-density case so downstream code can keep taking the
    cheaper ``None`` path rather than carrying an array of ones.
    """
    if densities is None:
        return None
    arr = np.asarray(densities, dtype=float)
    if arr.shape != (n,):
        raise ValueError(
            f"{context}: expected shape ({n},), got {arr.shape}. Densities must "
            f"be one value per particle, aligned with the radii array."
        )
    if np.any(arr <= 0.0):
        raise ValueError(f"{context}: densities must be strictly positive.")
    return arr


def compute_empirical_rg_polydisperse(
    coords: np.ndarray, radii: np.ndarray, densities: np.ndarray | None = None
) -> float:
    """Radius of gyration of actual coordinates, including each primary
    particle's own gyration radius (paper Eq. 4).

    Differs from :func:`compute_empirical_rg`, which treats particles as
    point masses and so omits the :math:`r_{g,i}^2` term below. That
    omission is a constant offset for monodisperse input but grows with
    polydispersity, and it makes the point-mass form **inconsistent with
    the Gamma equation**, whose derivation (paper Appendix A, Eq. A.5)
    carries the term throughout. Anything that has to agree with Gamma -
    the measured-Rg feedback in ``cca/pairing.py``, the per-aggregate
    quality record - must use this function; use
    :func:`compute_empirical_rg` only where a point-mass Rg is what is
    actually wanted.

    The counterpart to :func:`calculate_rg`, which returns the radius of
    gyration the fractal scaling law *prescribes* for a given particle
    count. This one measures what a built aggregate actually has:

    .. math::
        R_g^2 = \\frac{1}{m_a}\\sum_i m_{p,i}[(R_i - R_c)^2 + r_{g,i}^2]

    with :math:`r_{g,i}^2 = \\frac{3}{5}r_{p,i}^2` the primary particle's
    own gyration radius and :math:`R_c` the mass-weighted center of mass
    (Eq. 5). The :math:`r_{g,i}^2` term is what makes this valid for
    polydisperse primary particles :cite:p:`Moran2019FracVAL`; dropping it
    (as monodisperse treatments do) underestimates Rg for wide size
    distributions.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 particle coordinates.
    radii : np.ndarray
        N particle radii.
    densities : np.ndarray, optional
        Per-particle densities; ``None`` means uniform. Both the center of
        mass and the mass weighting below use these, so a heterogeneous
        aggregate's Rg is only correct when they are supplied.

    Returns
    -------
    float
        Measured radius of gyration; 0.0 for an empty aggregate.
    """
    if coords.shape[0] == 0:
        return 0.0

    mass = calculate_mass(radii, densities)
    total_mass = float(np.sum(mass))
    if total_mass <= 1e-12:
        return 0.0

    cm = np.sum(coords * mass[:, np.newaxis], axis=0) / total_mass
    d_sq = np.sum((coords - cm) ** 2, axis=1)
    rg_sq = float(np.sum(mass * (d_sq + 0.6 * radii**2)) / total_mass)
    return float(np.sqrt(max(rg_sq, 0.0)))


def calculate_rg(radii: np.ndarray, npp: int, df: float, kf: float) -> float:
    """Calculate the radius of gyration using the fractal scaling law.

    Implements the formula Rg = a * (N / kf)^(1/Df), where 'a' is the
    geometric mean radius calculated from the input `radii` array.
    See :cite:p:`Moran2019FracVAL` and morphology context
    :cite:p:`Filippov2000Tunable`.

    Parameters
    ----------
    radii : np.ndarray
        Array of radii of particles in the cluster/aggregate.
    npp : int
        Number of primary particles (N) in the cluster.
    df : float
        Fractal dimension (Df).
    kf : float
        Fractal prefactor (kf).

    Returns
    -------
    float
        The calculated radius of gyration (Rg). Returns 0.0 if `npp` is 0,
        `kf` or `df` is zero, or if calculation fails (e.g., log error).
    """
    rg = 0.0
    # TODO: throw an error just in case
    if npp == 0 or kf == 0 or df == 0:
        return 0.0

    # TODO: check radii beforehand
    valid_r = radii[radii > 1e-12]  # Filter near-zero radii
    try:
        if len(valid_r) > 0:
            # Geometric mean radius
            log_r_mean = np.sum(np.log(valid_r)) / len(valid_r)
            geo_mean_r = np.exp(log_r_mean)
            rg = geo_mean_r * (npp / kf) ** (1.0 / df)
    except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
        # Catch potential warnings from log(<=0) as well
        logger.warning(
            f"Could not calculate rg ({e}). npp={npp}, len(valid_r)={len(valid_r)}"
        )

    return max(rg, 0.0)


def gamma_calculation(
    m1: float,
    rg1: float,
    radii1: npt.NDArray,
    m2: float,
    rg2: float,
    radii2: npt.NDArray,
    df: float,
    kf: float,
    use_mass: bool = False,
    all_radii: npt.NDArray | None = None,
) -> tuple[bool, float]:
    """
    Calculates Gamma_pc for adding the next monomer (aggregate 2).

    The Gamma_pc relation follows the FracVAL CCA/PCA formulation
    :cite:p:`Moran2019FracVAL`.

    Parameters
    ----------
    use_mass : bool, default False
        Which form of the Gamma equation to solve. ``False`` substitutes
        particle *counts* for the masses, giving Filippov et al. (2000)
        Eq. 7 - what the Fortran PCA does (``PCA_cca.f90``'s
        ``Gamma_calculation`` takes ``n1, n2, n3``) and what this port has
        historically done everywhere. ``True`` uses the true masses passed
        in as ``m1``/``m2``, giving Moran et al. (2019) Eq. 6 - the
        paper's central polydisperse contribution, and what the Fortran
        *CCA* actually does (``CCA_module.f90:301``). Identical for
        monodisperse primary particles; they diverge as polydispersity
        grows. See NOTE.md 1.2.
    all_radii : np.ndarray, optional
        If provided, the geometric mean radius for rg3 is computed from this
        full set of radii (matching Fortran behaviour where R contains all N
        particles). When None the geometric mean is taken from the local
        combined set (radii1 + radii2).
    rg3_override : float, optional
        Use this radius of gyration for the *combined* aggregate instead of
        deriving it from the scaling law. Only meaningful together with
        measured (rather than scaling-law) rg1/rg2 - see
        ``cca/pairing.py``'s measured-Rg feedback.
    """
    n1 = radii1.size
    n2 = radii2.size

    n3 = n1 + n2
    m3 = m1 + m2

    if not use_mass:
        m1 = n1
        m2 = n2
        m3 = n3

    # Use the full particle set for the geometric-mean radius when available,
    # matching Fortran: rg3 = geomean(R_all) * (n3/kf)^(1/Df)
    rg3_radii = all_radii if all_radii is not None else np.concatenate((radii1, radii2))
    rg3 = calculate_rg(rg3_radii, n3, df, kf)

    # Heuristic from Fortran: ensure rg3 is not smaller than rg1
    # (avoids issues if rg calculation is noisy for small N)
    if n2 == 1 and rg3 < rg1:
        logger.info(f"Gamma calc: Adjusted rg3 from {rg3:.2e} to match rg1 {rg1:.2e}")
        rg3 = rg1

    gamma_pc = 0.0
    gamma_real = False

    term1 = (m3**2) * (rg3**2)
    term2 = m3 * (m1 * rg1**2 + m2 * rg2**2)  # rg2 is for monomer
    denominator = m1 * m2
    radicand = term1 - term2

    # Explicitly check for non-real conditions before calling sqrt.
    # np.sqrt(negative) produces nan + RuntimeWarning (not a Python exception),
    # so a try/except is insufficient — we must guard explicitly.
    if denominator <= 0.0:
        logger.warning(
            f"Gamma_pc calculation: denominator={denominator:.2e} <= 0 "
            f"(n1={n1}, m1={m1:.2e}, n2={n2}, m2={m2:.2e})"
        )
        gamma_real = False
    elif radicand < 0.0:
        logger.debug(
            f"Gamma_pc calculation: radicand={radicand:.2e} < 0 (non-real result). "
            f"n1={n1}, rg1={rg1:.2e}, n2={n2}, rg2={rg2:.2e}, rg3={rg3:.2e}"
        )
        gamma_real = False
    else:
        try:
            val = radicand / denominator
            gamma_pc = float(np.sqrt(val))
            # Sanity check: sqrt should never produce nan/inf here, but guard anyway
            if not np.isfinite(gamma_pc):
                logger.warning(
                    f"Gamma_pc calculation: non-finite result {gamma_pc} "
                    f"(radicand={radicand:.2e}, denominator={denominator:.2e})"
                )
                gamma_pc = 0.0
                gamma_real = False
            else:
                gamma_real = True
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logger.warning(f"Gamma calculation internal failed: {e}")
            gamma_real = False

    return gamma_real, gamma_pc


def calculate_cluster_properties(
    coords: np.ndarray,
    radii: np.ndarray,
    df: float,
    kf: float,
    densities: np.ndarray | None = None,
) -> Tuple[float, float, np.ndarray, float]:
    """Calculate aggregate properties: total mass, Rg, center of mass, Rmax.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle coordinates.
    radii : np.ndarray
        N array of particle radii.
    df : float
        Fractal dimension used for Rg calculation.
    kf : float
        Fractal prefactor used for Rg calculation.
    densities : np.ndarray, optional
        Per-particle densities; ``None`` means uniform. Affects the total
        mass and the center of mass (and hence r_max), which in turn feed
        the Gamma equation.

    Returns
    -------
    tuple[float, float | None, np.ndarray | None, float]
        A tuple containing:
            - total_mass (float): Sum of individual particle masses.
            - rg (float | None): Radius of gyration calculated via `calculate_rg`,
              or None if calculation failed.
            - cm (np.ndarray | None): 3D center of mass coordinates, or None if
              calculation failed.
            - r_max (float): Maximum distance from the center of mass to any
              particle center in the aggregate.

        Returns (0.0, 0.0, np.zeros(3), 0.0) for empty inputs (N=0).
    """
    npp = coords.shape[0]
    if npp == 0:
        return 0.0, 0.0, np.zeros(3), 0.0

    mass_vec = calculate_mass(radii, densities)
    total_mass = np.sum(mass_vec)

    if total_mass < 1e-12:  # Use tolerance
        cm = np.mean(coords, axis=0) if npp > 0 else np.zeros(3)
    else:
        cm = np.sum(coords * mass_vec[:, np.newaxis], axis=0) / total_mass

    rg = calculate_rg(radii, npp, df, kf)

    # Calculate max distance from CM
    if npp > 0:
        dist_from_cm = np.linalg.norm(coords - cm, axis=1)
        r_max = np.max(dist_from_cm)
    else:
        r_max = 0.0

    return total_mass, rg, cm, r_max


def compute_empirical_rg(coords: np.ndarray, radii: np.ndarray) -> float:
    """Compute empirical Rg directly from particle coordinates (mass-weighted).

    Unlike ``calculate_rg`` which uses the fractal scaling law
    Rg = a*(N/kf)^(1/Df), this function measures Rg from the actual
    spatial distribution of particles.

    Treats each particle as a point mass. See
    :func:`compute_empirical_rg_polydisperse` for the form that also
    carries each particle's own gyration radius (paper Eq. 4) - required
    wherever the result has to be consistent with the Gamma equation.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle center coordinates.
    radii : np.ndarray
        N array of particle radii. Mass is proportional to r^3.

    Returns
    -------
    float
        Empirical (mass-weighted) radius of gyration.
    """
    if coords.shape[0] == 0:
        return 0.0
    masses = radii**3
    total_mass = np.sum(masses)
    if total_mass < 1e-30:
        return 0.0
    cm = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    dist_sq = np.sum((coords - cm[np.newaxis, :]) ** 2, axis=1)
    return float(np.sqrt(np.sum(dist_sq * masses) / total_mass))


def compute_pair_correlation_dimensions(
    coords: np.ndarray,
    radii: np.ndarray,
    n_bins: int = 50,
) -> dict[str, np.ndarray]:
    """Estimate fractal dimension from pair-correlation (mass-radius) scaling.

    Computes the cumulative mass M(r) as a function of radial distance
    from the centre of mass. For a fractal aggregate,
    M(r) ~ r^Df, so a log-log fit gives the empirical Df.

    The fit uses raw (un-normalised) cumulative mass because normalised
    mass fractions are in [0,1] whose logs are non-positive, breaking
    the log-linear regression.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle center coordinates.
    radii : np.ndarray
        N array of particle radii.
    n_bins : int
        Number of radial bins (default 50).

    Returns
    -------
    dict with keys:
        r_bins : np.ndarray — bin edge radii (n_bins+1,)
        r_centers : np.ndarray — bin centre radii (n_bins,)
        M_r : np.ndarray — cumulative normalised mass fraction within each radius (n_bins+1,)
        empirical_Df : float — slope of log(M) vs log(r) fit
        fit_r_squared : float — R^2 of the linear fit
        empirical_kf : float — estimated kf from the fit
    """
    n = coords.shape[0]
    if n < 2:
        return {
            "r_bins": np.array([]),
            "r_centers": np.array([]),
            "M_r": np.array([]),
            "empirical_Df": 0.0,
            "fit_r_squared": 0.0,
            "empirical_kf": 0.0,
        }

    masses = radii**3
    total_mass = np.sum(masses)
    cm = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    distances = np.linalg.norm(coords - cm[np.newaxis, :], axis=1)

    r_min = np.min(distances[distances > 0]) * 0.5
    r_max = np.max(distances) * 0.99
    if r_min >= r_max or r_min <= 0:
        return {
            "r_bins": np.array([]),
            "r_centers": np.array([]),
            "M_r": np.array([]),
            "empirical_Df": 0.0,
            "fit_r_squared": 0.0,
            "empirical_kf": 0.0,
        }

    r_bins = np.linspace(r_min, r_max, n_bins + 1)
    cumulative_mass = np.zeros(n_bins + 1)
    for i in range(n_bins + 1):
        mask = distances <= r_bins[i]
        cumulative_mass[i] = np.sum(masses[mask])

    M_r = cumulative_mass / total_mass
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Use raw cumulative mass for log-log fit (normalised values < 1 have negative logs)
    log_r = np.log(r_centers)
    log_M_raw = np.log(cumulative_mass[1:])  # skip bin edge at r_min

    valid = np.isfinite(log_M_raw) & np.isfinite(log_r) & (cumulative_mass[1:] > 0)
    if np.sum(valid) < 3:
        return {
            "r_bins": r_bins,
            "r_centers": r_centers,
            "M_r": M_r,
            "empirical_Df": 0.0,
            "fit_r_squared": 0.0,
            "empirical_kf": 0.0,
        }

    coeffs = np.polyfit(log_r[valid], log_M_raw[valid], 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    ss_res = np.sum((log_M_raw[valid] - (slope * log_r[valid] + intercept)) ** 2)
    ss_tot = np.sum((log_M_raw[valid] - np.mean(log_M_raw[valid])) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Estimate kf from: Rg = a * (N/kf)^(1/Df) => kf = N / (Rg/a)^Df
    # Using empirical fit: M(r) = exp(intercept) * r^slope
    # At r = Rg: M = total_mass, so: total_mass = exp(intercept) * Rg^slope
    # kf from fractal scaling: N = kf * (Rg/a)^Df
    empirical_rg = compute_empirical_rg(coords, radii)
    geo_mean_r = np.exp(np.mean(np.log(radii[radii > 1e-12])))
    if slope > 0 and empirical_rg > 0 and geo_mean_r > 0:
        empirical_kf = n / (empirical_rg / geo_mean_r) ** slope
    else:
        empirical_kf = 0.0

    return {
        "r_bins": r_bins,
        "r_centers": r_centers,
        "M_r": M_r,
        "empirical_Df": float(slope),
        "fit_r_squared": float(r_squared),
        "empirical_kf": float(empirical_kf),
    }


def validate_fractal_structure(
    coords: np.ndarray,
    radii: np.ndarray,
    target_df: float,
    target_kf: float,
    rg_rtol: float = 0.05,
) -> dict[str, float]:
    """Validate that generated aggregate matches target fractal parameters.

    Compares theoretical Rg (from scaling law) vs empirical Rg (from
    coordinates), and estimates the actual fractal dimension from
    mass-radius scaling.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of particle center coordinates.
    radii : np.ndarray
        N array of particle radii.
    target_df : float
        Target fractal dimension.
    target_kf : float
        Target fractal prefactor.
    rg_rtol : float
        Relative tolerance for Rg agreement (default 5%).

    Returns
    -------
    dict with keys:
        N : int — number of particles
        theoretical_rg : float — Rg from scaling law Rg = a*(N/kf)^(1/Df)
        empirical_rg : float — Rg measured from coordinates
        rg_error_pct : float — (empirical - theoretical)/theoretical * 100
        rg_ok : bool — ``rg_error_pct`` < rg_rtol * 100
        empirical_Df : float — Df estimated from mass-radius scaling
        target_Df : float — target fractal dimension
        df_error : float — empirical_Df - target_Df
        fit_r_squared : float — goodness of fit for Df estimation
        empirical_kf : float — estimated fractal prefactor
        target_kf : float — target fractal prefactor
    """
    n = coords.shape[0]
    theoretical_rg = calculate_rg(radii, n, target_df, target_kf)
    empirical_rg = compute_empirical_rg(coords, radii)
    rg_error_pct = (
        (empirical_rg - theoretical_rg) / theoretical_rg * 100
        if theoretical_rg > 0
        else 0.0
    )

    pair_corr = compute_pair_correlation_dimensions(coords, radii)

    return {
        "N": n,
        "theoretical_rg": theoretical_rg,
        "empirical_rg": empirical_rg,
        "rg_error_pct": rg_error_pct,
        "rg_ok": abs(rg_error_pct) < rg_rtol * 100,
        "empirical_Df": pair_corr["empirical_Df"],
        "target_Df": target_df,
        "df_error": pair_corr["empirical_Df"] - target_df,
        "fit_r_squared": pair_corr["fit_r_squared"],
        "empirical_kf": pair_corr["empirical_kf"],
        "target_kf": target_kf,
    }
