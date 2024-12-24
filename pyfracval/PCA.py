import time
from time import sleep

import numba
import numpy as np
import pyvista as pv
from scipy.stats import gmean
from tqdm import trange


# Particle-Cluster Aggregation
def PCA(
    number: int,
    mass: np.ndarray,
    r: np.ndarray,
    Df: float,
    kf: float,
    tolerance: float,
) -> tuple[bool, np.ndarray]:
    PCA_ok = True
    n1, m1, rg1, x_cm, y_cm, z_cm, X, Y, Z = first_two_monomers(r, mass, number, Df, kf)

    if number > 2:
        k = 3
        while k <= number:
            n2 = 1
            m2 = mass[k - 1]

            rg2 = np.sqrt(0.6) * r[k - 1]

            n3 = n1 + n2
            m3 = m1 + m2
            rg3 = gmean(r) * np.power(n3 / kf, 1 / Df)

            gamma_ok, gamma_pc = gamma_calc(rg1, rg2, rg3, n1, n2, n3)
            monomer_candidates = np.zeros((number - k + 1))  # 0 = not considered
            monomer_candidates[0] = 1  # first one has been considered

            candidates, rmax = random_list_selection(
                gamma_ok, gamma_pc, X, Y, Z, r, n1, x_cm, y_cm, z_cm
            )

            list_sum = 0
            # print("here")
            while list_sum == 0:
                # print(f"{np.sum(candidates) == 0 = }")
                # print(f"{np.sum(monomer_candidates) <= number-k = }")
                # if not np.sum(monomer_candidates) <= number-k-1:
                #     print(f"{number = }")
                #     print(f"{k = }")
                #     print(f"{number-k = }")
                #     print(f"{list_sum = }")
                #     print(f"{np.sum(candidates) = }")
                #     print(f"{monomer_candidates = }")
                while (
                    np.sum(candidates) == 0 and np.sum(monomer_candidates) <= number - k
                ):
                    candidates, rmax = search_monomer_candidates(
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
                        X,
                        Y,
                        Z,
                        x_cm,
                        y_cm,
                        z_cm,
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

                curr_try = 1
                X_sel = X[selected_real]
                Y_sel = Y[selected_real]
                Z_sel = Z[selected_real]
                R_sel = r[selected_real]
                r_k = r[k - 1]

                # print(f"modified {k-1 = }")
                x_k, y_k, z_k, r0, x0, y0, z0, i_vec, j_vec = sticking_process(
                    X_sel, Y_sel, Z_sel, R_sel, r_k, x_cm, y_cm, z_cm, gamma_pc
                )
                X[k - 1] = x_k
                Y[k - 1] = y_k
                Z[k - 1] = z_k

                cov_max = overlap_check(X[0:k], Y[0:k], Z[0:k], r[0:k], k)

                positions = [[X[i], Y[i], Z[i]] for i in range(X.shape[0])]
                # print(f"{len(positions) = }")
                # print(f"{positions}")
                # point_cloud = pv.PolyData(positions)
                # point_cloud["radius"] = [2 for i in range(X.shape[0])]

                # geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
                # glyphed = point_cloud.glyph(scale="radius", geom=geom,)
                #     # p = pv.Plotter(notebook=False, off_screen=True, window_size=(2000,2000))
                # p = pv.Plotter(notebook=False)
                # p.add_mesh(glyphed, color='white',smooth_shading=True)
                # p.show()
                # if cov_max > tolerance:
                #     print("Need to readjust")
                while cov_max > tolerance and curr_try < 360:
                    x_k, y_k, z_k, _ = sticking_process2(x0, y0, z0, r0, i_vec, j_vec)

                    X[k - 1] = x_k
                    Y[k - 1] = y_k
                    Z[k - 1] = z_k
                    cov_max = overlap_check(X[0:k], Y[0:k], Z[0:k], r[0:k], k)
                    curr_try += 1

                    if np.mod(curr_try, 359) == 0 and np.sum(candidates) > 1:
                        candidates, selected_real = random_list_selection_one(
                            candidates, previous_candidate
                        )
                        # if np.sum(candidates) == 0:
                        #     print("FOURTH, candidates all ZERO")
                        X_sel = X[selected_real]
                        Y_sel = Y[selected_real]
                        Z_sel = Z[selected_real]
                        R_sel = r[selected_real]
                        r_k = r[k - 1]
                        x_k, y_k, z_k, r0, x0, y0, z0, i_vec, j_vec = sticking_process(
                            X_sel, Y_sel, Z_sel, R_sel, r_k, x_cm, y_cm, z_cm, gamma_pc
                        )
                        X[k - 1] = x_k
                        Y[k - 1] = y_k
                        Z[k - 1] = z_k
                        previous_candidate = selected_real
                        curr_try += 1

                        # print("huh1")
                        cov_max = overlap_check(X[0:k], Y[0:k], Z[0:k], r[0:k], k)
                        # print("huh2")

                # FIXME: candidates may be full of zeros at some point which causes the program to get stuck
                list_sum = np.sum(candidates)
                if np.sum(candidates) == 0:
                    print("FIFTH, candidates all ZERO")
                    sleep(2)

                if cov_max > tolerance:
                    list_sum = 0
                    candidates *= 0
                    # print("SIXTH, set all candidates to ZERO")
                    if number == k:
                        print("Failure -- restarting PCA routine")
                        return False, np.zeros((number, 4))

            # print("bruh")
            x_cm = (x_cm * m1 + X[k - 1] * m2) / (m1 + m2)
            y_cm = (y_cm * m1 + Y[k - 1] * m2) / (m1 + m2)
            z_cm = (z_cm * m1 + Z[k - 1] * m2) / (m1 + m2)

            n1 = n3
            m1 = m3
            rg1 = gmean(r) * (np.power(n1 / kf, 1 / Df))
            k = k + 1
            # FIXME: this gets stuck for some reason (no apparent connection to the arcos issue)
            # print("hi")

    data_new = np.zeros((number, 4))
    for i in range(number):
        data_new[i, :] = np.array([X[i], Y[i], Z[i], r[i]])

    # point_cloud = pv.PolyData(data_new[:,:-1])
    # point_cloud["radius"] = [2 for i in range(X.shape[0])]

    # geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
    # glyphed = point_cloud.glyph(scale="radius", geom=geom,)
    #     # p = pv.Plotter(notebook=False, off_screen=True, window_size=(2000,2000))
    # p = pv.Plotter(notebook=False)
    # p.add_mesh(glyphed, color='white',smooth_shading=True)
    # p.show()
    return PCA_ok, data_new


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


@numba.jit()
def first_two_monomers(
    r: np.ndarray, m: np.ndarray, n: int, df: float, kf: float
) -> tuple[int, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
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
    x = np.zeros((n))
    y = np.zeros((n))
    z = np.zeros((n))

    u = np.random.rand()
    v = np.random.rand()

    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    # Place p[1] at distance R[0]+R[1] from p[0]
    # at angles theta, phi
    x[1] = x[0] + (r[0] + r[1]) * np.cos(theta) * np.sin(phi)
    y[1] = y[0] + (r[0] + r[1]) * np.sin(theta) * np.sin(phi)
    z[1] = z[0] + (r[0] + r[1]) * np.cos(phi)

    m1 = m[0] + m[1]
    n1 = 2

    # eq 2a for two elements and eq 3
    rg1 = np.sqrt(r[0] * r[1]) * np.power(n1 / kf, 1 / df)
    # print(f"{rg1 = }")

    # Center of mass
    x_cm = (x[0] * m[0] + x[1] * m[1]) / m1
    y_cm = (y[0] * m[0] + y[1] * m[1]) / m1
    z_cm = (z[0] * m[0] + z[1] * m[1]) / m1

    return n1, m1, rg1, x_cm, y_cm, z_cm, x, y, z


@numba.jit()
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


@numba.jit(nopython=True, parallel=True)
def random_list_selection(
    gamma_ok: bool,
    gamma_pc: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    n1: int,
    x_cm: float,
    y_cm: float,
    z_cm: float,
) -> tuple[np.ndarray, float]:
    """
    Args:
        gamma_ok (bool): True if the gamma value was valid
        gamma_pc (float): gamma value for the clusters
        x (np.ndarray): x coordinates of particles
        y (np.ndarray): y coordinates of particles
        z (np.ndarray): z coordinates of particles
        n1 (int): number of particles in the cluster
        x_cm (float): x coordinate of center of mass
        y_cm (float): y coordinate of center of mass
        z_cm (float): z coordinate of center of mass

    Returns:
        candidates (np.ndarray): list of candidate indices
        rmax (float): maximum distance from center of mass
    """
    # TODO: candidates should be a boolean array
    candidates = np.zeros((n1))
    rmax = 0.0
    dists = np.sqrt(
        np.power(x - x_cm, 2) + np.power(y - y_cm, 2) + np.power(z - z_cm, 2)
    )
    if gamma_ok:
        for ii in numba.prange(n1):
            if dists[ii] > rmax:
                rmax = dists[ii]
            # Condition similar to triangle inequality
            # with sides dist, gamma_pc, and r[n1] + r[ii] (one inequality is missing)
            # TODO: condense the conditions in a single expression
            # and skip the for loop
            if (dists[ii] > gamma_pc - r[n1] - r[ii]) and (
                dists[ii] < gamma_pc + r[n1] + r[ii]
            ):
                candidates[ii] = 1
            if r[n1] + r[ii] > gamma_pc:
                candidates[ii] = 0

    return candidates, rmax


def search_monomer_candidates(
    r: np.ndarray,
    m: np.ndarray,
    monomer_candidates: np.ndarray,
    n: int,
    k: int,
    n3: int,
    df: float,
    kf: float,
    rg1: float,
    n1: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_cm: float,
    y_cm: float,
    z_cm: float,
):
    r_sl = r
    m_sl = m
    vector_search = np.zeros((n - k + 1))
    # TODO: this loop can be avoided
    # by directly initializing the values with np.arange
    # vector_search = np.arange(n - k + 1) + k - 2
    for i in range(vector_search.size):
        vector_search[i] = i + k - 2
        # print(f"{i+k-2 = }")

    # TODO: this loop can be avoided
    # by multiplication with a logical array
    for i in range(monomer_candidates.size):
        if monomer_candidates[i] == 1:
            vector_search[i] = 0
    vector_search2 = vector_search[vector_search != 0]

    if vector_search2.size > 1:
        u = np.random.rand()
        # print("search monomer candidates")
        # print(f"{u = }")
        rs_1 = int(vector_search2[int(u * (vector_search.size - 1))])
    else:
        rs_1 = int(vector_search2[0])

    r[rs_1 - 1] = r_sl[k]
    r[k] = r_sl[rs_1 - 1]
    m[rs_1 - 1] = m_sl[k]
    m_sl[k] = m[rs_1 - 1]

    # TODO: not needed?
    # m2 = M[k - 1]
    # m3 = np.sum(M[0 : k - 1])
    rg2 = np.sqrt(0.6 * r[k])
    rg3 = gmean(r) * np.power(n3 / kf, 1.0 / df)
    # rg3 = (np.exp(np.sum(np.log(r)) / np.log(r).size)) * np.power(n3 / kf, 1.0 / df)

    gamma_ok, gamma_pc = gamma_calc(rg1, rg2, rg3, n1, 1, n3)

    candidates, rmax = random_list_selection(
        gamma_ok, gamma_pc, x, y, z, r, n1, x_cm, y_cm, z_cm
    )

    return candidates, rmax


def random_list_selection_one(candidates: np.ndarray, previous_candidate: int):
    if previous_candidate > -1:
        candidates[previous_candidate] = 0
    candidates2 = candidates[candidates > 0]
    n = np.random.rand()
    # print("random list selection one")
    # print(f"{n = }")
    # n = 0.5
    selected = 1 + int(n * (candidates2.size - 1))

    selected_real = 0
    j = 0
    for i in range(candidates.size):
        if candidates[i] > 0:
            j += 1
        if j == selected:
            selected_real = i
            break
    return candidates, selected_real


def sticking_process(
    x: float,
    y: float,
    z: float,
    r: float,
    r_k: float,
    x_cm: float,
    y_cm: float,
    z_cm: float,
    gamma_pc: float,
):
    x1 = x
    y1 = y
    z1 = z
    r1 = r + r_k
    x2 = x_cm
    y2 = y_cm
    z2 = z_cm
    r2 = gamma_pc

    # print(f"{x1 = }, {y1 = }, {z1 = }, {r1 = }")
    # print(f"{x2 = }, {y2 = }, {z2 = }, {r2 = }")

    a = 2 * (x2 - x1)
    b = 2 * (y2 - y1)
    c = 2 * (z2 - z1)
    d = (
        np.power(x1, 2)
        - np.power(x2, 2)
        + np.power(y1, 2)
        - np.power(y2, 2)
        + np.power(z1, 2)
        - np.power(z2, 2)
        - np.power(r1, 2)
        + np.power(r2, 2)
    )

    t_sp = (x1 * a + y1 * b + z1 * c + d) / (
        a * (x1 - x2) + b * (y1 - y2) + c * (z1 - z2)
    )

    x0 = x1 + t_sp * (x2 - x1)
    y0 = y1 + t_sp * (y2 - y1)
    z0 = z1 + t_sp * (z2 - z1)

    distance = np.sqrt(
        np.power(x2 - x1, 2) + np.power(y2 - y1, 2) + np.power(z2 - z1, 2)
    )

    # print(f"{distance = }")

    alpha = np.arccos(
        (np.power(r1, 2) + np.power(distance, 2) - np.power(r2, 2))
        / (2 * r1 * distance)
    )
    # print(f"{alpha = }")
    if (
        abs(
            (np.power(r1, 2) + np.power(distance, 2) - np.power(r2, 2))
            / (2 * r1 * distance)
        )
        > 1
    ):
        # FIXME: this should never happen!

        pos = [[x1, y1, z1], [x2, y2, z2]]
        point_cloud = pv.PolyData(pos)
        point_cloud["radius"] = [2, 2]

        geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
        glyphed = point_cloud.glyph(
            scale="radius",
            geom=geom,
        )
        # p = pv.Plotter(notebook=False, off_screen=True, window_size=(2000,2000))
        p = pv.Plotter(notebook=False)
        p.add_mesh(glyphed, color="white", smooth_shading=True)
        p.show()
        print(
            f"{(np.power(r1,2) + np.power(distance,2) - np.power(r2,2))/(2*r1*distance) = }"
        )
        exit(-1)
    r0 = r1 * np.sin(alpha)

    # AmBdC = (A+B)/C
    AmBdC = (a + b) / c

    k_vec = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
    i_vec = np.array([1, 1, -AmBdC]) / np.sqrt(1 + 1 + AmBdC**2)
    j_vec = np.cross(k_vec, i_vec)

    u = np.random.rand()
    v = np.random.rand()
    # print("sticking process")
    # print(f"{u = }, {v = }")

    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)

    x_k = x0 + r0 * np.cos(theta) * i_vec[0] + r0 * np.sin(theta) * j_vec[0]
    y_k = y0 + r0 * np.cos(theta) * i_vec[1] + r0 * np.sin(theta) * j_vec[1]
    z_k = z0 + r0 * np.cos(theta) * i_vec[2] + r0 * np.sin(theta) * j_vec[2]

    return x_k, y_k, z_k, r0, x0, y0, z0, i_vec, j_vec


def sticking_process2(x0, y0, z0, r0, i_vec, j_vec):
    u = np.random.rand()
    # print("sticking process2")
    # print(f"{u = }")
    theta = 2 * np.pi * u

    x_k = x0 + r0 * np.cos(theta) * i_vec[0] + r0 * np.sin(theta) * j_vec[0]
    y_k = y0 + r0 * np.cos(theta) * i_vec[1] + r0 * np.sin(theta) * j_vec[1]
    z_k = z0 + r0 * np.cos(theta) * i_vec[2] + r0 * np.sin(theta) * j_vec[2]
    return x_k, y_k, z_k, theta


def overlap_check(x: np.ndarray, y: np.ndarray, z: np.ndarray, r: np.ndarray, k: int):
    C = np.zeros((k - 1))
    for i in range(k - 1):
        distance_kj = np.sqrt(
            np.power(x[k - 1] - x[i], 2)
            + np.power(y[k - 1] - y[i], 2)
            + np.power(z[k - 1] - z[i], 2)
        )

        if distance_kj < (r[k - 1] + r[i]):
            C[i] = ((r[k - 1] + r[i]) - distance_kj) / (r[k - 1] + r[i])
        else:
            C[i] = 0

    return np.max(C)
