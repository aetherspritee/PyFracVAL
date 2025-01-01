import numpy as np
import numpy.typing as npt
from numba import jit


@jit(cache=True)
def sphere_sphere_intersection(
    p1: npt.NDArray[np.float64],
    r1: float,
    p2: npt.NDArray[np.float64],
    r2: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Calculate the intersection between two spheres.
    Equations taken from appendix D in the paper.

    abs(arg) should be smaller than 1! If it isn't, the spheres do not intersect!
    Source: https://mathworld.wolfram.com/Sphere-SphereIntersection.html
    """
    dp = p2 - p1
    abc = 2 * dp
    a, b, c = abc
    d = r1**2 - r2**2 + np.sum(p2**2) - np.sum(p1**2)
    t = (np.dot(abc, p1) - d) / (np.dot(abc, -dp))
    # print(f"{t=}")

    p0 = p1 + t * dp

    distance = np.sqrt(np.sum((p2 - p1) ** 2))
    arg = (r1**2 + distance**2 - r2**2) / (2 * r1 * distance)
    # FIXME: does not work, urgh...
    # assert np.abs(arg) <= 1, f"{arg=} is greater than 1"
    r0 = r1 * np.sqrt(1 - arg**2)

    k_vec = abc
    k_vec /= np.linalg.norm(k_vec)
    i_vec = np.array([-c, -c, a + b])
    i_vec /= np.linalg.norm(i_vec)
    j_vec = np.cross(k_vec, i_vec)

    theta = np.pi * 2 * np.random.rand()
    p = p0 + r0 * (np.cos(theta) * i_vec + np.sin(theta) * j_vec)
    if np.any(np.isnan(p)):
        print("""
              Seems like these two spheres do no intersect!
              Open up an issue and inform us about it :)
              """)

    return p, p0, r0, i_vec, j_vec
