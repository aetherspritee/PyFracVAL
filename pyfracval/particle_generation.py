# particle_generation.py
"""Functions for generating primary particle radii."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def random_normal_custom() -> float:
    """
    Generates a normally distributed random number using the Ziggurat method
    as implemented in numpy, which is generally preferred over custom
    acceptance-rejection methods unless specific properties are needed.
    """
    # The custom Fortran implementation is an acceptance-rejection method.
    # For standard normal distribution, numpy's is highly optimized and recommended.
    return np.random.normal(0.0, 1.0)


def lognormal_pp_radii(
    rp_gstd: float,
    rp_g: float,
    n: int,
    seed: int | None = None,
    truncate: bool = True,
) -> np.ndarray:
    """Generate N random radii from a lognormal distribution.

    Parameters
    ----------
    rp_gstd : float
        Geometric standard deviation of the distribution (must be >= 1.0).
        If 1.0, generates monodisperse particles.
    rp_g : float
        Geometric mean radius of the distribution (must be > 0).
    n : int
        Number of radii to generate.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    truncate : bool, optional
        Use the FracVAL 2*sigma truncate version

    Returns
    -------
    np.ndarray
        A 1D NumPy array of N generated radii.

    Raises
    ------
    ValueError
        If `rp_g` is not positive.

    Notes
    -----
    Uses `numpy.random.lognormal`. The underlying normal distribution's
    parameters are mu=log(rp_g) and sigma=log(rp_gstd).
    The original Fortran code included optional truncation at approximately
    +/- 2 geometric standard deviations; this is not enabled by default here.
    """
    if seed is not None:
        np.random.seed(seed)

    if rp_gstd < 1.0:
        logger.warning("Geometric standard deviation should be >= 1.0. Setting to 1.0.")
        rp_gstd = 1.0

    if rp_g <= 0:
        raise ValueError("Geometric mean radius (rp_g) must be positive.")

    if rp_gstd == 1.0:
        # Monodisperse case
        logger.info("Generating monodisperse particles.")
        return np.full(n, rp_g, dtype=float)
    else:
        # Polydisperse case using numpy's lognormal
        # The parameters for np.random.lognormal are mu and sigma of the *underlying* normal distribution.
        # mu = log(geometric_mean)
        # sigma = log(geometric_standard_deviation)
        mu = np.log(rp_g)
        sigma = np.log(rp_gstd)

        if not truncate:
            radii = np.random.lognormal(mean=mu, sigma=sigma, size=n)

            logger.info(
                f"Generated polydisperse particles (mean={np.mean(radii):.2f}, std={np.std(radii):.2f})."
            )
        else:
            # The Fortran code truncates at rp_g / (rp_gstd**2) and rp_g * (rp_gstd**2)
            # This corresponds to approximately +/- 2 sigma in the underlying normal distribution.
            # np.random.lognormal directly generates from the distribution.
            # If strict truncation matching the Fortran code is needed:
            min_val = rp_g / (rp_gstd**2)
            max_val = rp_g * (rp_gstd**2)
            radii = np.zeros(n, dtype=float)
            generated_count = 0
            while generated_count < n:
                # Generate candidates (can generate more than needed for efficiency)
                num_needed = n - generated_count
                candidates = np.random.lognormal(
                    mean=mu, sigma=sigma, size=num_needed * 2
                )  # Generate extras
                valid = candidates[(candidates >= min_val) & (candidates <= max_val)]
                num_valid = len(valid)
                take = min(num_valid, num_needed)
                if take > 0:
                    radii[generated_count : generated_count + take] = valid[:take]
                    generated_count += take
            logger.info(
                f"Generated polydisperse particles (truncated between {min_val:.2f} and {max_val:.2f})."
            )
        return radii
