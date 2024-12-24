from time import sleep

import numpy as np
import numpy.typing as npt
from numba import jit, prange
from tqdm import trange


# Particle-Cluster Aggregation
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
            gamma_ok,
            gamma_pc,
            # np.column_stack((X, Y, Z)),
            p,
            r,
            n1,
            # np.array([x_cm, y_cm, z_cm]),
            p_cm,
        )

        list_sum = 0
        while list_sum == 0:
            while np.sum(candidates) == 0 and np.sum(monomer_candidates) <= number - k:
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
                exit(-1)

            # p_sel = p[selected_real, :]
            # X_sel, Y_sel, Z_sel = p_sel
            # X_sel = X[selected_real]
            # Y_sel = Y[selected_real]
            # Z_sel = Z[selected_real]
            # R_sel = r[selected_real]
            # r_k = r[k - 1]

            # print(f"modified {k-1 = }")
            p[k - 1, :], r0, p0, i_vec, j_vec = sticking_process(
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
                sleep(2)

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


@jit()
def first_two_monomers(
    r: npt.NDArray[np.float64],
    m: npt.NDArray[np.float64],
    n: int,
    df: float,
    kf: float,
) -> tuple[int, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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


@jit()
def gamma_calc(
    rg1: float, rg2: float, rg: float, n1: int, n2: int, n: int
) -> tuple[bool, float]:
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
    gamma_pc_2 = (
        np.power(n * rg_aux, 2) - n * (n1 * np.power(rg1, 2) + n2 * np.power(rg2, 2))
    ) / (n1 * n2)
    gamma_ok = gamma_pc_2 > 0
    if gamma_ok:
        gamma_pc = np.sqrt(gamma_pc_2)

    return gamma_ok, gamma_pc


@jit(nopython=True, parallel=True)
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
    candidates = np.zeros((n1), dtype=np.bool_)
    if not gamma_ok:
        return candidates, 0.0

    rmax = 0.0
    dists = np.sqrt(np.sum((p - p_cm) ** 2, axis=1))
    for ii in prange(n1):
        if dists[ii] > rmax:
            rmax = dists[ii]
        # Condition similar to triangle inequality
        # with sides dist, gamma_pc, and r[n1] + r[ii] (one inequality is missing)
        # TODO: condense the conditions in a single expression
        # and skip the for loop
        if (dists[ii] > gamma_pc - r[n1] - r[ii]) and (
            dists[ii] < gamma_pc + r[n1] + r[ii]
        ):
            candidates[ii] = True
        if r[n1] + r[ii] > gamma_pc:
            candidates[ii] = False

    return candidates, rmax


@jit()
def search_monomer_candidates(
    r: npt.NDArray[np.float64],
    m: npt.NDArray[np.float64],
    monomer_candidates: npt.NDArray[np.bool_],
    n: int,
    k: int,
    n3: int,
    df: float,
    kf: float,
    rg1: float,
    n1: int,
    p: npt.NDArray[np.float64],
    p_cm: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.bool_], float]:
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
    rs_1 = np.random.choice(
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
    rg3 = gmean(r) * np.power(n3 / kf, 1.0 / df)

    gamma_ok, gamma_pc = gamma_calc(rg1, rg2, rg3, n1, 1, n3)

    candidates, rmax = random_list_selection(
        gamma_ok,
        gamma_pc,
        # np.column_stack((x, y, z)),
        p,
        r,
        n1,
        # np.array([x_cm, y_cm, z_cm]),
        p_cm,
    )

    return candidates, rmax


@jit()
def random_list_selection_one(
    candidates: npt.NDArray[np.bool_],
    previous_candidate: int,
) -> tuple[npt.NDArray[np.bool_], int]:
    """
    Get and index for a candidate.

    Args:
        candidates (npt.NDArray[np.bool_]): list of candidate indices
        previous_candidate (int): index of the previous candidate

    Returns:
        candidates (npt.NDArray[np.bool_]): updated list of candidate indices
        selected_real (int): index of the selected candidate

    """
    if previous_candidate > -1:
        candidates[previous_candidate] = 0
    # candidates2 = candidates[candidates > 0]
    # n = np.random.rand()
    # print("random list selection one")
    # print(f"{n = }")
    # n = 0.5
    # selected = 1 + int(n * (candidates2.size - 1))
    # # Tweak:
    # selected = 1 + np.random.randint(np.sum(candidates > 0) - 1)

    # selected_real = 0
    # j = 0
    # for i in range(candidates.size):
    #     if candidates[i] > 0:
    #         j += 1
    #     if j == selected:
    #         selected_real = i
    #         break
    selected_real = np.random.choice(
        np.arange(candidates.size, dtype=np.int64)[candidates > 0], 1
    )[0]
    return candidates, selected_real


# Can be enabled when the arg problem has been solved
# @jit()
def sticking_process(
    # x: float,
    # y: float,
    # z: float,
    p: npt.NDArray[np.float64],
    r: float,
    r_k: float,
    # x_cm: float,
    # y_cm: float,
    # z_cm: float,
    p_cm: npt.NDArray[np.float64],
    gamma_pc: float,
) -> tuple[
    npt.NDArray[np.float64],
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Handles the intersection/collision of two spheres.
    The `_k` coordinates represent a point on the intersection of the two spheres.

    Args:
        p (npt.NDArray[np.float64]): coordinates of the particle
        r (float): radius of the particle
        r_k (float): radius of the particle cluster
        p_cm (npt.NDArray[np.float64]): coordinates of the center of mass
        gamma_pc (float): gamma value for the clusters

    Returns:
        p_k (npt.NDArray[np.float64]): coordinates of the new particle
        r0 (float): radius of the new particle
        p0 (npt.NDArray[np.float64]): coordinates of the new particle
        i_vec (npt.NDArray[np.float64]): i vector
        j_vec (npt.NDArray[np.float64]): j vector
    """
    # TODO: Find some sources and explanation on the math!
    # p1 = np.array([x, y, z])
    r1 = r + r_k

    # p2 = np.array([x_cm, y_cm, z_cm])
    r2 = gamma_pc

    dp = p_cm - p
    abc = 2 * dp
    d = np.sum(p**2) - r1**2 - np.sum(p_cm**2) + r2**2

    distance = np.sqrt(np.sum(dp**2))

    t_sp = (np.dot(p, abc) + d) / (-2 * distance**2)

    p0 = p + t_sp * dp

    arg = (r1**2 + distance**2 - r2**2) / (2 * r1 * distance)
    # FIXME: this should never happen!
    if abs(arg) > 1:
        print(f"{arg=}")
        exit(-1)
        # return
    r0 = r1 * np.sqrt(1 - arg**2)

    # k points from point 1 to point 2
    # i is orthogonal to k (dot product is zero)
    # j is orthogonal the plane spanned by k and i
    # TODO: find a nicer i vector to create?
    a = abc[0]
    b = abc[1]
    c = abc[2]
    k_vec = abc / np.sqrt(np.sum(abc**2))
    i_vec = np.array([-c, -c, a + b]) / np.sqrt(2 * c**2 + (a + b) ** 2)
    j_vec = np.cross(k_vec, i_vec)

    theta = 2.0 * np.pi * np.random.rand()

    # Point on the cross-section of the two spheres
    p_k = p0 + r0 * (np.cos(theta) * i_vec + np.sin(theta) * j_vec)

    # x_k, y_k, z_k = p_k
    # x0, y0, z0 = p0

    return p_k, r0, p0, i_vec, j_vec


@jit()
def sticking_process2(
    p0: npt.NDArray[np.float64],
    r0: float,
    i_vec: npt.NDArray[np.float64],
    j_vec: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float]:
    u = np.random.rand()
    theta = 2 * np.pi * u

    p_k = p0 + r0 * (np.cos(theta) * i_vec + np.sin(theta) * j_vec)
    return p_k, theta


@jit()
def overlap_check(
    p: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
    k: int,
) -> np.float64:
    # TODO: do we need k here?
    # the arrays already have k+1 length
    # -> could be infered!

    # C = np.zeros((k - 1))
    # for i in range(k - 1):
    #     distance_kj = np.sqrt(
    #         np.power(x[k - 1] - x[i], 2)
    #         + np.power(y[k - 1] - y[i], 2)
    #         + np.power(z[k - 1] - z[i], 2)
    #     )

    #     if distance_kj < (r[k - 1] + r[i]):
    #         C[i] = ((r[k - 1] + r[i]) - distance_kj) / (r[k - 1] + r[i])
    #     else:
    #         C[i] = 0

    distance_k = np.sqrt(np.sum((p - p[k - 1, :]) ** 2, axis=1))
    c = ((r[k - 1] + r) - distance_k) / (r[k - 1] + r)
    c[-1] = 0.0
    return np.max(c)


@jit(fastmath=True)
def gmean(a: npt.NDArray[np.float64]) -> float:
    """
    Calculate the geometric mean of an array of numbers
    """

    # with np.errstate(divide="ignore"):
    #     log_a = np.log(a)
    log_a = np.log(a)
    return np.exp(np.mean(log_a))
