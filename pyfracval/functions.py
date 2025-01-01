import jax
import jax.numpy as jnp
from jax import Array

PRNG_KEY = jax.random.PRNGKey(42)


def random_theta(key: Array | None = None) -> tuple[Array, Array]:
    if key is None:
        key = jax.random.key(42)
    new_key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey)
    return 2 * jnp.pi * u, new_key
    # theta = 2 * jnp.pi * u
    # return theta


# @jit(cache=True)
# def sphere_sphere_intersection(
#     p1: npt.NDArray[np.float64],
#     r1: float,
#     p2: npt.NDArray[np.float64],
#     r2: float,
# ) -> tuple[
#     npt.NDArray[np.float64],
#     npt.NDArray[np.float64],
#     float,
#     npt.NDArray[np.float64],
#     npt.NDArray[np.float64],
# ]:
@jax.jit
def sphere_sphere_intersection(
    p1: Array,
    r1: float,
    p2: Array,
    r2: float,
) -> tuple[
    Array,
    Array,
    Array,
    # float,
    Array,
    Array,
]:
    """
    Calculate the intersection between two spheres.
    Equations taken from appendix D in the paper.

    abs(arg) should be smaller than 1! If it isn't, the spheres do not intersect!
    Source: https://mathworld.wolfram.com/Sphere-SphereIntersection.html

    Args:
        p1 (npt.NDArray[np.float64]): coordinates of the particle
        r1 (float): radius of the particle
        p2 (npt.NDArray[np.float64]): coordinates of the center of mass
        r2 (float): radius of the particle cluster
        gamma_pc (float): gamma value for the clusters

    Returns:
        p (jax.Array): coordinates of the new particle
        r0 (float): radius of the new particle
        p0 (jax.Array): coordinates of the new particle
        i_vec (jax.Array): i vector
        j_vec (jax.Array): j vector
    """
    dp = p2 - p1
    abc = 2 * dp
    a, b, c = abc
    d = r1**2 - r2**2 + jnp.sum(p2**2) - jnp.sum(p1**2)
    t = (jnp.dot(abc, p1) - d) / (jnp.dot(abc, -dp))

    p0 = p1 + t * dp

    distance = jnp.sqrt(jnp.sum((p2 - p1) ** 2))
    arg = (r1**2 + distance**2 - r2**2) / (2 * r1 * distance)
    # FIXME: does not work, urgh...
    # assert jnp.abs(arg) <= 1, f"{arg=} is greater than 1"
    r0 = r1 * jnp.sqrt(1 - arg**2)

    k_vec = abc
    k_vec /= jnp.linalg.norm(k_vec)
    i_vec = jnp.array([-c, -c, a + b])
    i_vec /= jnp.linalg.norm(i_vec)
    j_vec = jnp.cross(k_vec, i_vec)

    theta, _ = random_theta()
    p = p0 + r0 * (jnp.cos(theta) * i_vec + jnp.sin(theta) * j_vec)
    # if jnp.any(jnp.isnan(p)):
    #     print("""
    #           Seems like these two spheres do no intersect!
    #           Open up an issue and inform us about it :)
    #           """)

    return p, p0, r0, i_vec, j_vec
