from time import sleep

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

# from jax.experimental import checkify
from jaxtyping import Array, Bool, Float, Integer, Key, PRNGKeyArray, Scalar
from numba import jit
from tqdm import trange

from pyfracval.functions import random_theta, sphere_sphere_intersection

# TODO: make a PCA class with internal values to be passed down
# This would cut down on copying the data around
# and make the function a bit more readable
# TODO: when starting to create the class, use pydantic,
# and don't forget to use numba jitclass ;)

PRNG_KEY = jax.random.key(42)


# Particle-Cluster Aggregation
# @jit(cache=True)
def PCA(
    number: int,
    mass: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
    Df: float,
    kf: float,
    tolerance: float,
) -> tuple[bool, np.ndarray]:
    PCA_ok = True
    # n1, m1, rg1, x_cm, y_cm, z_cm, X, Y, Z = first_two_monomers(r, mass, number, Df, kf)
    n1, m1, rg1, p_cm, p = first_two_monomers(r, mass, number, Df, kf)

    if number <= 2:
        return True, np.column_stack((p, r))

    # x_cm, y_cm, z_cm = p_cm
    # X, Y, Z = p.transpose()

    k = 3
    # while k <= number:
    # TODO: k - 1 -> k
    for k_temp in range(2, number):
        k = k_temp + 1
        n2 = 1
        m2 = mass[k - 1]

        rg2 = np.sqrt(0.6) * r[k - 1]

        n3 = n1 + n2
        m3 = m1 + m2
        rg3 = gmean(r) * np.power(n3 / kf, 1 / Df)

        gamma_ok, gamma_pc = gamma_calc(rg1, rg2, rg3, n1, n2, n3)
        monomer_candidates = np.zeros(
            (number - k + 1), dtype=np.bool_
        )  # 0 = not considered
        monomer_candidates[0] = 1  # first one has been considered

        candidates, _ = random_list_selection(
            gamma_ok, gamma_pc, p, r, n1, p_cm
        )

        list_sum = 0
        while list_sum == 0:
            while (
                np.sum(candidates) == 0
                and np.sum(monomer_candidates) <= number - k
            ):
                candidates, _ = search_monomer_candidates(
                    r,
                    mass,
                    monomer_candidates,
                    number,
                    k,
                    n3,
                    Df,
                    kf,
                    rg1,
                    n1,
                    # np.column_stack((X, Y, Z)),
                    # np.array([x_cm, y_cm, z_cm]),
                    p,
                    p_cm,
                )

            previous_candidate = -1

            if np.sum(candidates) > 0:
                candidates, selected_real = random_list_selection_one(
                    candidates, previous_candidate
                )
                previous_candidate = selected_real
            elif np.sum(candidates) == number - k + 1:
                PCA_ok = False
                return False, np.zeros((number, 4))

            # p_sel = p[selected_real, :]
            # X_sel, Y_sel, Z_sel = p_sel
            # X_sel = X[selected_real]
            # Y_sel = Y[selected_real]
            # Z_sel = Z[selected_real]
            # R_sel = r[selected_real]
            # r_k = r[k - 1]

            # print(f"modified {k-1 = }")
            p[k - 1, :], r0, p0, i_vec, j_vec = sticking_process(
                p[selected_real, :], r[selected_real], r[k - 1], p_cm, gamma_pc
            )
            # X[k - 1], Y[k - 1], Z[k - 1] = p_k
            # x0, y0, z0 = p0

            cov_max = overlap_check(
                # np.column_stack((X[0:k], Y[0:k], Z[0:k])),
                p[0:k, :],
                r[0:k],
                k,
            )

            # curr_try = 1
            # while cov_max > tolerance and curr_try < 360:
            for curr_try in range(1, 360):
                if cov_max <= tolerance:
                    break

                p[k - 1, :], _ = sticking_process2(
                    # np.array([x0, y0, z0]),
                    p0,
                    r0,
                    i_vec,
                    j_vec,
                )
                # p[k - 1, :] = p_k
                # X[k - 1], Y[k - 1], Z[k - 1] = p_k
                # x_k, y_k, z_k = p_k

                # X[k - 1] = x_k
                # Y[k - 1] = y_k
                # Z[k - 1] = z_k
                cov_max = overlap_check(
                    # np.column_stack((X[0:k], Y[0:k], Z[0:k])),
                    p[0:k, :],
                    r[0:k],
                    k,
                )

                if np.mod(curr_try, 359) == 0 and np.sum(candidates) > 1:
                    candidates, selected_real = random_list_selection_one(
                        candidates, previous_candidate
                    )
                    # if np.sum(candidates) == 0:
                    #     print("FOURTH, candidates all ZERO")
                    # X_sel = X[selected_real]
                    # Y_sel = Y[selected_real]
                    # Z_sel = Z[selected_real]
                    # R_sel = r[selected_real]
                    # r_k = r[k - 1]
                    p[k - 1, :], r0, _, i_vec, j_vec = sticking_process(
                        # np.array([X_sel, Y_sel, Z_sel]),
                        # R_sel,
                        # r_k,
                        # np.array([x_cm, y_cm, z_cm]),
                        p[selected_real, :],
                        r[selected_real],
                        r[k - 1],
                        p_cm,
                        gamma_pc,
                    )
                    # p[k - 1, :] = p_k
                    # X[k - 1], Y[k - 1], Z[k - 1] = p_k
                    # X[k - 1] = x_k
                    # Y[k - 1] = y_k
                    # Z[k - 1] = z_k
                    previous_candidate = selected_real
                    curr_try += 1

                    # print("huh1")
                    cov_max = overlap_check(
                        # np.column_stack((X[0:k], Y[0:k], Z[0:k])),
                        p[0:k, :],
                        r[0:k],
                        k,
                    )
                    # print("huh2")

            # FIXME: candidates may be full of zeros at some point which causes the program to get stuck
            list_sum = np.sum(candidates)
            if np.sum(candidates) == 0:
                print("FIFTH, candidates all ZERO")
                # sleep(2)

            # FIXME: why does this happen?
            # maybe it can be forced/corrected not to happen
            if cov_max > tolerance:
                list_sum = 0
                # candidates *= 0
                candidates |= False
                # print("SIXTH, set all candidates to ZERO")
                if number == k:
                    print("Failure -- restarting PCA routine")
                    return False, np.zeros((number, 4))

        # x_cm, y_cm, z_cm = p_cm
        # x_cm = (x_cm * m1 + p[k - 1, 0] * m2) / (m1 + m2)
        # y_cm = (y_cm * m1 + p[k - 1, 1] * m2) / (m1 + m2)
        # z_cm = (z_cm * m1 + p[k - 1, 2] * m2) / (m1 + m2)
        # p_cm = np.array([x_cm, y_cm, z_cm])
        p_cm = (p_cm * m1 + p[k - 1, :] * m2) / (m1 + m2)

        n1 = n3
        m1 = m3
        rg1 = gmean(r) * (np.power(n1 / kf, 1 / Df))

        # switch to for loop
        # k = k + 1
        # FIXME: this gets stuck for some reason (no apparent connection to the arcos issue)
        # print("hi")

    return PCA_ok, np.column_stack((p, r))
    # return PCA_ok, np.column_stack((X, Y, Z, r))


# @jit(cache=True)
def PCA_subcluster(
    N: int,
    N_subcluster: int,
    R: np.ndarray,
    DF: float,
    kf: float,
    tolerance: float,
) -> tuple[bool, np.ndarray, int, np.ndarray]:
    PCA_OK = True
    N_clusters = int(np.floor(N / N_subcluster))
    if int(np.mod(N, N_subcluster)) != 0:
        N_clusters = N_clusters + 1
        N_subcluster_m = np.ones((N_clusters)) * N_subcluster
        N_subcluster_m[-1] = N - N_subcluster * (N_clusters - 1)
    else:
        N_subcluster_m = np.ones((int(N_clusters))) * N_subcluster

    Na = 1
    acum = 0

    i_orden = np.zeros((N_clusters, 3))
    data = np.zeros((N, 4))
    for i in trange(1, N_clusters + 1, desc="PCA Loop"):
        number = int(N_subcluster_m[i - 1])
        radius = R[Na - 1 : Na + number - 1]
        mass = np.zeros((number))

        for j in range(radius.size):
            mass[j] = 4 / 3 * np.pi * np.power(R[j], 3)

        PCA_OK = False
        while not PCA_OK:
            PCA_OK, data_new = PCA(number, mass, radius, DF, kf, tolerance)
        # print(f"PCA LOOP {i} DONE!")
        # time.sleep(1)

        if i == 0:
            acum = number
            for ii in range(number + 1):
                data[ii, :] = data_new[ii, :]
            i_orden[0, 0:2] = np.array([1, acum])
            i_orden[0, 2] = acum
        else:
            for ii in range(acum, acum + number):
                data[ii, :] = data_new[ii - acum, :]
            i_orden[i - 1, 0:2] = np.array([acum + 1, acum + number])
            i_orden[i - 1, 2] = number
            acum += number

        Na += number
    return PCA_OK, data, N_clusters, i_orden


# @jit(cache=True)
def first_two_monomers(
    r: npt.NDArray[np.float64],
    m: npt.NDArray[np.float64],
    n: int,
    df: float,
    kf: float,
) -> tuple[
    int, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    # ) -> tuple[int, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes two particles in space next to each other

    Args:
        r (np.ndarray): radius of particles
        m (np.ndarray): mass of particles
        n (int): number of particles
        df (float): fractal dimension
        kf (float): fractal prefactor

    Returns:
        n1 (int): number of particles in cluster, i.e., 2
        m1 (float): mass of cluster
        rg1 (float): radius of gyration of cluster
        x_cm (float): x coordinate of center of mass
        y_cm (float): y coordinate of center of mass
        z_cm (float): z coordinate of center of mass
        x (np.ndarray): x coordinates of particles
        y (np.ndarray): y coordinates of particles
        z (np.ndarray): z coordinates of particles
    """
    # TODO: why keep them in separate arrays?
    p = np.zeros((n, 3))

    u = np.random.rand()
    v = np.random.rand()

    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    # Place p[1] at distance R[0]+R[1] from p[0]
    # at angles theta, phi
    p[1, :] = p[0, :] + (r[0] + r[1]) * np.array(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )

    m1 = m[0] + m[1]
    n1 = 2

    # eq 2a for two elements and eq 3
    rg1 = np.sqrt(r[0] * r[1]) * np.power(n1 / kf, 1 / df)
    # print(f"{rg1 = }")

    # Center of mass
    p_cm = (p[0, :] * m[0] + p[1, :] * m[1]) / m1
    # x_cm, y_cm, z_cm = p_cm
    # x, y, z = p.transpose()

    return n1, m1, rg1, p_cm, p
    # return n1, m1, rg1, x_cm, y_cm, z_cm, x, y, z


@jax.jit
def gamma_calc(
    rg1: Float,
    rg2: Float,
    rg: Float,
    n1: Integer,
    n2: Integer,
    n: Integer,
) -> tuple[Bool, Float]:
    """
    Check if the gamma value is correct for two given clusters
    and a potential aggregation of those clusters.
    Return the gamma value for the clusters.
    Uses equation 7 to compute the gamma value for the given values

    Args:
        rg1 (float): radius of gyration of cluster 1
        rg2 (float): radius of gyration of cluster 2
        rg (float): radius of gyration of the combined cluster
        n1 (int): number of particles in cluster 1
        n2 (int): number of particles in cluster 2
        n (int): number of particles in the combined cluster

    Return:
        gamma_ok (bool): True if the gamma value is valid
        gamma_pc (float): gamma value for the clusters
    """
    # TODO: create assert message out of this
    if n1 + n2 != n:
        print("Number of particles do not add up!")

    gamma_pc = 0.0

    rg_aux = rg
    # TODO: explain why or reference!
    if rg < rg1:
        rg_aux = rg1

    # gamma squared from eq 7
    gamma_pc_2 = ((n * rg_aux) ** 2 - n * (n1 * rg1**2 + n2 * rg2**2)) / (
        n1 * n2
    )
    gamma_ok = gamma_pc_2 > 0
    if gamma_ok:
        gamma_pc = jnp.sqrt(gamma_pc_2)

    return gamma_ok, gamma_pc


# @jit(fastmath=True, cache=True)
def random_list_selection(
    gamma_ok: bool,
    gamma_pc: float,
    p: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
    n1: int,
    p_cm: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.bool_], float]:
    """
    Args:
        gamma_ok (bool): True if the gamma value was valid
        gamma_pc (float): gamma value for the clusters
        p (npt.NDArray): coordinates of particles
        r (npt.NDArray): radii of particles
        n1 (int): number of particles in the cluster
        p_cm (npt.NDArray): coordinates of center of mass

    Returns:
        candidates (np.ndarray): list of candidate indices
        rmax (float): maximum distance from center of mass
    """
    if not gamma_ok:
        return np.zeros((n1), dtype=np.bool_), 0.0

    dists = np.sqrt(np.sum((p - p_cm) ** 2, axis=1))
    rmax = np.max(dists)

    candidates = (dists[:n1] > gamma_pc - r[n1] - r[:n1]) & (
        dists[:n1] < gamma_pc + r[n1] + r[:n1]
    )
    candidates &= r[n1] + r[:n1] <= gamma_pc

    return candidates, rmax


@jax.jit
def search_monomer_candidates(
    r: Array,
    m: Array,
    monomer_candidates: Array,
    n: int,
    k: int,
    n3: int,
    df: float,
    kf: float,
    rg1: float,
    n1: int,
    p: Array,
    p_cm: Array,
    key: Array | None = None,
) -> tuple[Array, Array]:
    """
    Search for possible monomer candidates
    """
    # vector_search = np.arange(n - k + 1) + k - 2
    # vector_search = np.zeros((n - k + 1))
    # for i in range(vector_search.size):
    #     vector_search[i] = i + k - 2

    # for i in range(monomer_candidates.size):
    #     if monomer_candidates[i] == 1:
    #         vector_search[i] = 0
    # TODO: assert that the monomer_candidates array has the size n-k+1
    # vector_search *= monomer_candidates != 1
    # vector_search2 = vector_search[vector_search != 0]

    # if vector_search2.size > 1:
    #     u = np.random.rand()
    #     rs_1 = int(vector_search2[int(u * (vector_search.size - 1))])
    # else:
    #     rs_1 = int(vector_search2[0])
    # Random sample from vector_search
    # TODO: monomer_candidates is full with 0 and a leading 1, why?
    # vector_search = np.arange(n - k + 1) + k - 2
    if key is None:
        key = jax.random.key(42)
    rs_1 = jnp.random.choice(
        key,
        (np.arange(n - k + 1) + k - 2)[monomer_candidates == 1],
        1,
    )[0]

    r_sl = r
    m_sl = m
    # TODO: why?
    r[rs_1 - 1] = r_sl[k]
    r[k] = r_sl[rs_1 - 1]
    m[rs_1 - 1] = m_sl[k]
    m_sl[k] = m[rs_1 - 1]

    # TODO: not needed?
    # m2 = M[k - 1]
    # m3 = np.sum(M[0 : k - 1])
    # TODO: why?
    rg2 = np.sqrt(0.6 * r[k])
    rg3 = gmean(r) * jnp.power(n3 / kf, 1.0 / df)

    gamma_ok, gamma_pc = gamma_calc(rg1, rg2, rg3, n1, 1, n3)

    candidates, rmax = random_list_selection(
        gamma_ok, gamma_pc, p, r, n1, p_cm
    )

    return candidates, rmax


def random_list_selection_one(
    candidates: Array,
    previous_candidate: int,
    key: PRNGKeyArray = None,
) -> tuple[Array, int]:
    """
    Get and index for a candidate.

    Args:
        candidates (npt.NDArray[np.bool_]): list of candidate indices
        previous_candidate (int): index of the previous candidate

    Returns:
        candidates (npt.NDArray[np.bool_]): updated list of candidate indices
        selected_real (int): index of the selected candidate

    """
    if key is None:
        key = jax.random.key(42)
    candidates_filtered = jax.lax.cond(
        previous_candidate > -1,
        lambda x: x.at[previous_candidate].set(0),
        lambda x: x,
        candidates,
    )
    # print(f"{jnp.sum(candidates_filtered > 0) = }")
    selected = jax.random.randint(
        key, (1,), 0, jnp.sum(candidates_filtered > 0)
    )[0]
    # print(f"{selected = }")
    # selected = 1 + int(n * (jnp.sum(candidates > 0) - 1))

    # selected_real = jnp.argmax(jnp.cumsum(candidates_filtered > 0) == selected)
    selected_real = jnp.argmax(
        jnp.cumsum(candidates_filtered > 0) - 1 == selected
    )
    selected_real = int(selected_real)
    # print(f"{selected_real = }")
    return candidates_filtered, selected_real


@jax.jit
# @checkify.checkify
def sticking_process(
    p: Float[Array],
    r: float,
    r_k: float,
    p_cm: Float[Array],
    gamma_pc: float,
) -> tuple[
    Float[Array],
    Float[Array],
    Float[Array],
    Float[Array],
    Float[Array],
]:
    r1 = r + r_k
    r2 = gamma_pc

    p_k, r0, p0, i_vec, j_vec = sphere_sphere_intersection(p, r1, p_cm, r2)
    return p_k, r0, p0, i_vec, j_vec


@jax.jit
def sticking_process2(
    p0: Float[Array],
    r0: Float,
    i_vec: Float[Array],
    j_vec: Float[Array],
) -> tuple[Float[Array], Float[Array]]:
    theta, _ = random_theta()

    p_k = p0 + r0 * (jnp.cos(theta) * i_vec + jnp.sin(theta) * j_vec)
    return p_k, theta


@jax.jit
def overlap_check(
    p: Float[Array],
    r: Float[Array],
    k: int,
) -> Float[Array]:
    distance_k = jnp.sqrt(jnp.sum((p - p[k - 1, :]) ** 2, axis=1))
    c = ((r[k - 1] + r) - distance_k) / (r[k - 1] + r)
    c = c.at[-1].set(0.0)
    return jnp.max(c)


@jax.jit
def gmean(a: Float[Array, "n"]) -> Float[Scalar, ""]:
    """
    Calculate the geometric mean of an array of numbers
    """
    log_a = jnp.log(a)
    return jnp.exp(jnp.mean(log_a))
