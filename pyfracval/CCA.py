import copy
import re
from datetime import datetime
from pathlib import Path

import numba
import numpy as np
import polars as pl
from tqdm import tqdm

from .PCA import PCA_subcluster


def CCA_subcluster(
    R: np.ndarray,
    N: int,
    DF: float,
    kf: float,
    iter: int,
    N_subcl_perc: float,
    ext_case: int,
    tolerance: float = 1e-7,
    folder: str = "results",
) -> tuple[pl.DataFrame | None, bool, bool]:
    CCA_OK = True

    if N < 50:
        N_subcluster = 5
    elif N > 500:
        N_subcluster = 50
    else:
        N_subcluster = int(N_subcl_perc * N)

    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)

    PCA_OK, data, n_clusters, i_orden = PCA_subcluster(
        N, N_subcluster, R, DF, kf, tolerance
    )

    if not PCA_OK:
        print("PCA failed")
        return None, CCA_OK, PCA_OK

    I_total = int(n_clusters)

    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    R = data[:, 3]

    iteration = 1
    fill_xnew = 0
    progress_bar = tqdm(
        total=np.ceil(np.log2(n_clusters)).astype(int),
        desc="I_total",
    )
    # TODO: Precalculate the I_total beforehand
    # and iterate over them in a foor loop
    while I_total > 1:
        i_orden = sort_rows(i_orden)

        ID_agglom, CCA_OK = generate_CCA_pairs(I_total, i_orden, X, Y, Z, R, DF, kf)

        ID_mon = CCA_identify_monomers(i_orden)

        number_pairs = (I_total + 1) // 2

        considered = np.zeros((I_total))
        X_next = np.zeros((N))
        Y_next = np.zeros((N))
        Z_next = np.zeros((N))
        R_next = np.zeros((N))
        k = 1
        u = 1
        acum = 1

        other = 0
        i_orden = np.zeros((int(number_pairs), 3))
        # TODO: why not for loop? More secure!
        while k <= I_total:
            for i in range(ID_agglom[k - 1, :].size):
                if ID_agglom[k - 1, i] == 1:
                    other = i + 1
                    break
            IS_EMPTY = True

            for i in range(int(np.sum(considered))):
                if i == other - 1 and considered[i] == 1:
                    IS_EMPTY = False
                    break

            if k != other and IS_EMPTY:
                Xn, Yn, Zn, Rn, CCA_OK = CCA(
                    X, Y, Z, R, N, ID_mon, k, other, DF, kf, ext_case, tolerance
                )

                considered[k - 1] = 1
                considered[other - 1] = 1

                if np.sum(considered) == 2:
                    for i in range(Xn.size):
                        X_next[i] = Xn[i]
                        Y_next[i] = Yn[i]
                        Z_next[i] = Zn[i]
                        R_next[i] = Rn[i]
                    fill_xnew = Xn.size
                else:
                    for i in range(fill_xnew, fill_xnew + Xn.size):
                        X_next[i] = Xn[i - fill_xnew]
                        Y_next[i] = Yn[i - fill_xnew]
                        Z_next[i] = Zn[i - fill_xnew]
                        R_next[i] = Rn[i - fill_xnew]
                    fill_xnew += Xn.size

                count_k = 0
                for j in range(ID_mon.size):
                    if ID_mon[j] + 1 == k:
                        count_k += 1
                count_other = 0
                for j in range(ID_mon.size):
                    if ID_mon[j] + 1 == other:
                        count_other += 1

                i_orden[u - 1, 0:3] = np.array(
                    [acum, acum + count_k + count_other - 1, count_k + count_other]
                )

                acum = acum + count_k + count_other
                u += 1
            k += 1

        if np.sum(considered) < I_total:
            considered[other - 1] = 1
            count_other = 0

            for j in range(ID_mon.size):
                if ID_mon[j] + 1 == other:
                    count_other += 1

            i_orden[u - 1, 0:3] = np.array([acum, acum + count_other - 1, count_other])
            Xn = np.zeros((count_other))
            Yn = np.zeros((count_other))
            Zn = np.zeros((count_other))
            Rn = np.zeros((count_other))

            count_other = 0
            for j in range(ID_mon.size):
                if ID_mon[j] + 1 == other:
                    Xn[count_other] = X[j]
                    Yn[count_other] = Y[j]
                    Zn[count_other] = Z[j]
                    Rn[count_other] = R[j]
                    count_other += 1

            for i in range(fill_xnew, fill_xnew + Xn.size):
                X_next[i] = Xn[i - fill_xnew]
                Y_next[i] = Yn[i - fill_xnew]
                Z_next[i] = Zn[i - fill_xnew]
                R_next[i] = Rn[i - fill_xnew]

        I_total = (I_total + 1) // 2

        X = X_next
        Y = Y_next
        Z = Z_next
        R = R_next

        iteration += 1
        progress_bar.update(1)

    result = pl.DataFrame(
        {
            "x": pl.Series(X),
            "y": pl.Series(Y),
            "z": pl.Series(Z),
            "r": pl.Series(R),
        }
    )

    CCA_OK = np.logical_not(np.isnan(result.to_numpy())).any().astype(bool)

    filename = filename_generate(N, DF, kf)
    # save results
    if CCA_OK and PCA_OK:
        save_results(result, iter, filename=filename, folder=folder)

    return result, CCA_OK, PCA_OK


def generate_CCA_pairs(
    I_t: int,
    i_orden: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    Df: float,
    kf: float,
) -> tuple[np.ndarray, bool]:
    I_t = int(I_t)
    ID_agglom = np.zeros((I_t, I_t))
    CCA_ok = True
    gamma_real = True

    for i in range(I_t - 1):
        size_vectors = int(i_orden[i, 1] - i_orden[i, 0] + 1)
        X1 = np.zeros((size_vectors))
        Y1 = np.zeros((size_vectors))
        Z1 = np.zeros((size_vectors))
        R1 = np.zeros((size_vectors))

        jjj = 0
        for jj in range(int(i_orden[i, 0]), int(i_orden[i, 1] + 1)):
            X1[jjj] = X[jj - 1]
            Y1[jjj] = Y[jj - 1]
            Z1[jjj] = Z[jj - 1]
            R1[jjj] = R[jj - 1]
            jjj += 1
        rg1, r1_max, m1, _, _, _ = CCA_agg_properties(X1, Y1, Z1, R1, jjj, Df, kf)

        cntr = 0
        j = 0

        while cntr == 0:
            if j + 1 > i_orden[:, 0].size:
                CCA_ok = False
                print("CCA NOT OK")

            if np.sum(ID_agglom[i, :]) < 1 and np.sum(ID_agglom[:, i]) < 1:
                if i != j and np.sum(ID_agglom[:, j]) < 1:
                    size_vectors = int(i_orden[j, 1] - i_orden[j, 0] + 1)
                    X2 = np.zeros((size_vectors))
                    Y2 = np.zeros((size_vectors))
                    Z2 = np.zeros((size_vectors))
                    R2 = np.zeros((size_vectors))

                    jjj = 0
                    for jj in range(int(i_orden[j, 0]), int(i_orden[j, 1] + 1)):
                        X2[jjj] = X[jj - 1]
                        Y2[jjj] = Y[jj - 1]
                        Z2[jjj] = Z[jj - 1]
                        R2[jjj] = R[jj - 1]
                        jjj += 1

                    rg2, r2_max, m2, _, _, _ = CCA_agg_properties(
                        X2, Y2, Z2, R2, jjj, Df, kf
                    )

                    m3 = m1 + m2
                    r_com = np.hstack((R1, R2))
                    rg3 = (np.exp(np.sum(np.log(r_com)) / (np.log(r_com).size))) * (
                        (R1.size + R2.size) / kf
                    ) ** (1.0 / Df)
                    if np.power(m3, 2) * np.power(rg3, 2) > m3 * (
                        m1 * np.power(rg1, 2) + m2 * np.power(rg2, 2)
                    ):
                        gamma_pc = np.sqrt(
                            (
                                np.power(m3, 2) * np.power(rg3, 2)
                                - m3 * (m1 * np.power(rg1, 2) + m2 * np.power(rg2, 2))
                            )
                            / (m1 * m2)
                        )
                        gamma_real = True
                    else:
                        gamma_pc = np.inf
                        gamma_real = False

                    if gamma_pc < r1_max + r2_max and gamma_real:
                        ID_agglom[i, j] = 1
                        ID_agglom[j, i] = 1
                        cntr = 1

            else:
                cntr = 1
            j += 1

    if int(np.mod(I_t, 2)) != 0:
        for i in range(ID_agglom[0, :].size):
            if np.sum(ID_agglom[:, i]) == 0:
                loc = i

        ID_agglom[loc, loc] = 1
    return ID_agglom, CCA_ok


def CCA_agg_properties(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    npp: int,
    Df: float,
    kf: float,
) -> tuple[float, float, float, float, float, float]:
    m_vec = np.zeros((npp))
    R_i = np.zeros((npp))
    sum_xm: float = 0
    sum_ym: float = 0
    sum_zm: float = 0
    for i in range(npp):
        m_vec[i] = 4 / 3 * np.pi * np.power(R[i], 3)
        sum_xm += X[i] * m_vec[i]
        sum_ym += Y[i] * m_vec[i]
        sum_zm += Z[i] * m_vec[i]

    m = float(np.sum(m_vec))
    X_cm = sum_xm / m
    Y_cm = sum_ym / m
    Z_cm = sum_zm / m

    rg = float(
        (np.exp(np.sum(np.log(R) / np.log(R).size))) * np.power(npp / kf, 1 / Df)
    )
    for i in range(npp):
        R_i[i] = np.sqrt(
            np.power(X_cm - X[i], 2)
            + np.power(Y_cm - Y[i], 2)
            + np.power(Z_cm - Z[i], 2)
        )

    R_max = float(np.max(R_i))
    return rg, R_max, m, X_cm, Y_cm, Z_cm


def CCA_identify_monomers(i_orden: np.ndarray):
    ID_mon = np.zeros((int(np.sum(i_orden[:, 2]))))

    for i in range(i_orden[:, 0].size):
        for j in range(int(i_orden[i, 0] - 1), int(i_orden[i, 1])):
            ID_mon[j] = i
    return ID_mon


@numba.jit()
def CCA_random_select_list(
    X1: np.ndarray,
    Y1: np.ndarray,
    Z1: np.ndarray,
    R1: np.ndarray,
    X_cm1: float,
    Y_cm1: float,
    Z_cm1: float,
    X2: np.ndarray,
    Y2: np.ndarray,
    Z2: np.ndarray,
    R2: np.ndarray,
    X_cm2: float,
    Y_cm2: float,
    Z_cm2: float,
    curr_list: np.ndarray,
    gamma_pc: float,
    gamma_real: bool,
    ext_case: int,
):
    if gamma_real and ext_case == 1:
        for i in range(curr_list.shape[0] - 1):
            d_i_min = (
                np.sqrt(
                    np.power(X1[i] - X_cm1, 2)
                    + np.power(Y1[i] - Y_cm1, 2)
                    + np.power(Z1[i] - Z_cm1, 2)
                )
                - R1[i]
            )
            d_i_max = (
                np.sqrt(
                    np.power(X1[i] - X_cm1, 2)
                    + np.power(Y1[i] - Y_cm1, 2)
                    + np.power(Z1[i] - Z_cm1, 2)
                )
                + R1[i]
            )
            for j in range(curr_list.shape[1] - 1):
                d_j_min = (
                    np.sqrt(
                        np.power(X2[j] - X_cm2, 2)
                        + np.power(Y2[j] - Y_cm2, 2)
                        + np.power(Z2[j] - Z_cm2, 2)
                    )
                    - R2[j]
                )
                d_j_max = (
                    np.sqrt(
                        np.power(X2[j] - X_cm2, 2)
                        + np.power(Y2[j] - Y_cm2, 2)
                        + np.power(Z2[j] - Z_cm2, 2)
                    )
                    + R2[j]
                )
                if d_i_max + d_j_max > gamma_pc:
                    if np.abs(d_j_max - d_i_max) < gamma_pc:
                        curr_list[i, j] = 1
                    elif d_j_max - d_i_max > gamma_pc and d_j_min - d_i_max < gamma_pc:
                        curr_list[i, j] = 1
                    elif d_i_max - d_j_max > gamma_pc and d_i_min - d_j_max < gamma_pc:
                        curr_list[i, j] = 1
    elif gamma_real and ext_case == 0:
        for i in range(curr_list.shape[0]):
            d_i_max = (
                np.sqrt(
                    np.power(X1[i] - X_cm1, 2)
                    + np.power(Y1[i] - Y_cm1, 2)
                    + np.power(Z1[i] - Z_cm1, 2)
                )
                + R1[i]
            )
            for j in range(curr_list.shape[1]):
                d_j_max = (
                    np.sqrt(
                        np.power(X2[j] - X_cm2, 2)
                        + np.power(Y2[j] - Y_cm2, 2)
                        + np.power(Z2[j] - Z_cm2, 2)
                    )
                    + R2[j]
                )
                if (
                    d_i_max + d_j_max > gamma_pc
                    and np.abs(d_j_max - d_i_max) < gamma_pc
                ):
                    curr_list[i, j] = 1

    return curr_list


def CCA(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    N: int,
    ID_mon: np.ndarray,
    k: int,
    other: int,
    Df: float,
    kf: float,
    ext_case,
    tolerance: float,
):
    CCA_ok = True

    monomers_1 = 0
    for i in range(N):
        if ID_mon[i] + 1 == k:
            monomers_1 += 1

    X1 = np.zeros((monomers_1))
    Y1 = np.zeros((monomers_1))
    Z1 = np.zeros((monomers_1))
    R1 = np.zeros((monomers_1))

    monomers_1 = 0

    for i in range(N):
        if ID_mon[i] + 1 == k:
            X1[monomers_1] = X[i]
            Y1[monomers_1] = Y[i]
            Z1[monomers_1] = Z[i]
            R1[monomers_1] = R[i]
            monomers_1 += 1

    n1 = X1.size

    rg1, r1_max, m1, X_cm1, Y_cm1, Z_cm1 = CCA_agg_properties(
        X1, Y1, Z1, R1, n1, Df, kf
    )

    monomers_2 = 0
    for i in range(N):
        if ID_mon[i] + 1 == other:
            monomers_2 += 1

    X2 = np.zeros((monomers_2))
    Y2 = np.zeros((monomers_2))
    Z2 = np.zeros((monomers_2))
    R2 = np.zeros((monomers_2))

    monomers_2 = 0

    for i in range(N):
        if ID_mon[i] + 1 == other:
            X2[monomers_2] = X[i]
            Y2[monomers_2] = Y[i]
            Z2[monomers_2] = Z[i]
            R2[monomers_2] = R[i]
            monomers_2 += 1

    n2 = X2.size

    rg2, r2_max, m2, X_cm2, Y_cm2, Z_cm2 = CCA_agg_properties(
        X2, Y2, Z2, R2, n2, Df, kf
    )

    m3 = m1 + m2
    n3 = n1 + n2

    r_com = np.hstack((R1, R2))
    rg3 = (np.exp(np.sum(np.log(r_com)) / (np.log(r_com).size))) * (
        np.power((n3) / kf, (1.0 / Df))
    )

    if np.power(m3, 2) * np.power(rg3, 2) > m3 * (
        m1 * np.power(rg1, 2) + m2 * np.power(rg2, 2)
    ):
        gamma_pc = np.sqrt(
            (
                np.power(m3, 2) * np.power(rg3, 2)
                - m3 * (m1 * np.power(rg1, 2) + m2 * np.power(rg2, 2))
            )
            / (m1 * m2)
        )
        gamma_real = True
    else:
        gamma_real = False

    CM1 = np.zeros((3))
    CM2 = np.zeros((3))

    CM1[0] = X_cm1
    CM1[1] = Y_cm1
    CM1[2] = Z_cm1

    CM2[0] = X_cm2
    CM2[1] = Y_cm2
    CM2[2] = Z_cm2

    curr_list = np.zeros((n1, n2))
    curr_list = CCA_random_select_list(
        X1,
        Y1,
        Z1,
        R1,
        X_cm1,
        Y_cm1,
        Z_cm1,
        X2,
        Y2,
        Z2,
        R2,
        X_cm2,
        Y_cm2,
        Z_cm2,
        curr_list,
        gamma_pc,
        gamma_real,
        ext_case,
    )

    COR1 = np.zeros((n1, 4))
    COR2 = np.zeros((n2, 4))

    list_sum = 0

    while list_sum == 0:
        prev_cand1: int = 0
        prev_cand2 = 0
        cov_max = 1

        # TODO: this here should go into its own function. run different tries in parallel, check if it fits, then break out.
        while cov_max > tolerance:
            if np.sum(curr_list) > 1:
                prev_cand1 = CCA_random_pick(curr_list, prev_cand1)
                prev_cand2 = 0
                prev_cand2 = CCA_random_pick(curr_list, prev_cand1, prev_cand2)
            else:
                CCA_ok = False

            curr_try = 1

            COR1[:, 0] = X1
            COR1[:, 1] = Y1
            COR1[:, 2] = Z1
            COR1[:, 3] = R1

            COR2[:, 0] = X2
            COR2[:, 1] = Y2
            COR2[:, 2] = Z2
            COR2[:, 3] = R2

            COR1, COR2, CM2, vec0, i_vec, j_vec = CCA_sticking_process(
                gamma_real,
                gamma_pc,
                COR1,
                COR2,
                CM1,
                CM2,
                prev_cand1,
                prev_cand2,
                ext_case,
                n1,
                n2,
            )

            X1 = COR1[:, 0]
            Y1 = COR1[:, 1]
            Z1 = COR1[:, 2]

            X2 = COR2[:, 0]
            Y2 = COR2[:, 1]
            Z2 = COR2[:, 2]

            X_cm2 = CM2[0]
            Y_cm2 = CM2[1]
            Z_cm2 = CM2[2]

            cov_max = CCA_overlap_check(n1, n2, X1, X2, Y1, Y2, Z1, Z2, R1, R2)

            while cov_max > tolerance and curr_try < 360:
                X2, Y2, Z2 = CCA_sticking_process_v2(
                    CM2, vec0, X2, Y2, Z2, i_vec, j_vec, prev_cand2
                )
                cov_max = CCA_overlap_check(n1, n2, X1, X2, Y1, Y2, Z1, Z2, R1, R2)
                curr_try += 1

                if (
                    int(np.mod(curr_try, 359)) == 0
                    and np.sum(curr_list[prev_cand1, :]) > 1
                ):
                    prev_cand2 = CCA_random_pick(curr_list, prev_cand1, prev_cand2)

                    COR1[:, 0] = X1
                    COR2[:, 0] = X2

                    COR1[:, 1] = Y1
                    COR2[:, 1] = Y2

                    COR1[:, 2] = Z1
                    COR2[:, 2] = Z2

                    COR1, COR2, CM2, vec0, i_vec, j_vec = CCA_sticking_process(
                        gamma_real,
                        gamma_pc,
                        COR1,
                        COR2,
                        CM1,
                        CM2,
                        prev_cand1,
                        prev_cand2,
                        ext_case,
                        n1,
                        n2,
                    )
                    cov_max = CCA_overlap_check(n1, n2, X1, X2, Y1, Y2, Z1, Z2, R1, R2)
                    curr_try = 1
            list_sum = np.sum(curr_list[prev_cand1, :])

    X1 = COR1[:, 0]
    Y1 = COR1[:, 1]
    Z1 = COR1[:, 2]

    X2 = COR2[:, 0]
    Y2 = COR2[:, 1]
    Z2 = COR2[:, 2]

    X_cm2 = CM2[0]
    Y_cm2 = CM2[1]
    Z_cm2 = CM2[2]

    monomers_1 = 0
    for i in range(N - 1):
        if ID_mon[i] + 1 == k:
            X[i] = X1[monomers_1]
            Y[i] = Y1[monomers_1]
            Z[i] = Z1[monomers_1]
            R[i] = R1[monomers_1]
            monomers_1 += 1

    monomers_2 = 0
    for i in range(N - 1):
        if ID_mon[i] + 1 == other:
            X[i] = X2[monomers_2]
            Y[i] = Y2[monomers_2]
            Z[i] = Z2[monomers_2]
            R[i] = R2[monomers_2]
            monomers_2 += 1

    Xn = np.zeros((n1 + n2))
    Yn = np.zeros((n1 + n2))
    Zn = np.zeros((n1 + n2))
    Rn = np.zeros((n1 + n2))

    Xn[0:n1] = X1
    Xn[n1 : n1 + n2] = X2

    Yn[0:n1] = Y1
    Yn[n1 : n1 + n2] = Y2

    Zn[0:n1] = Z1
    Zn[n1 : n2 + n1] = Z2

    Rn[0:n1] = R1
    Rn[n1 : n2 + n1] = R2

    if cov_max > tolerance:
        CCA_ok = False

    return Xn, Yn, Zn, Rn, CCA_ok


def CCA_random_pick(curr_list: np.ndarray, prev_cand1: int, prev_cand2=None):
    selected_real = 0
    if prev_cand2 is None:
        if prev_cand1 > 0:
            curr_list[prev_cand1, :] = curr_list[prev_cand1, :] * 0
        list_sum = np.array(
            [np.sum(curr_list[i, :]) for i in range(curr_list.shape[0])]
        )
        curr_list2 = list_sum[list_sum > 0]

        uu = np.random.rand()
        selected = int(uu * (curr_list2.size - 1)) + 1
        sel = 0
        jj = 0

        for i in range(list_sum.size):
            if list_sum[i] > 0:
                jj += 1
                sel = jj
            if sel == selected:
                selected_real = i
                break
        prev_cand1 = selected_real
        return prev_cand1
    else:
        if prev_cand2 > 0:
            curr_list[prev_cand1, prev_cand2] = curr_list[prev_cand1, prev_cand2] * 0
        list_sum = curr_list[prev_cand1, :]
        curr_list2 = list_sum[list_sum > 0]

        uu = np.random.rand()
        selected = 1 + int(uu * (curr_list2.size - 1))
        sel = 0
        jj = 0

        for i in range(list_sum.size):
            if list_sum[i] > 0:
                jj += 1
                sel = jj
            if sel == selected:
                selected_real = i
                break
        prev_cand2 = selected_real
        return prev_cand2


def CCA_sticking_process(
    gamma_real: bool,
    gamma_pc: float,
    COR1,
    COR2,
    CM1,
    CM2,
    prev_cand1: int,
    prev_cand2: int,
    ext_case: int,
    n1: int,
    n2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if gamma_real:
        X1 = COR1[:, 0]
        Y1 = COR1[:, 1]
        Z1 = COR1[:, 2]
        R1 = COR1[:, 3]

        X2 = COR2[:, 0]
        Y2 = COR2[:, 1]
        Z2 = COR2[:, 2]
        R2 = COR2[:, 3]

        X_cm1 = CM1[0]
        Y_cm1 = CM1[1]
        Z_cm1 = CM1[2]

        X_cm2 = CM2[0]
        Y_cm2 = CM2[1]
        Z_cm2 = CM2[2]

        vect_x = X1[prev_cand1] - X_cm1
        vect_y = Y1[prev_cand1] - Y_cm1
        vect_z = Z1[prev_cand1] - Z_cm1
        vect_mag = np.sqrt(
            np.power(vect_x, 2) + np.power(vect_y, 2) + np.power(vect_z, 2)
        )

        vect_x /= vect_mag
        vect_y /= vect_mag
        vect_z /= vect_mag

        x_cm22 = X_cm1 + gamma_pc * vect_x
        y_cm22 = Y_cm1 + gamma_pc * vect_y
        z_cm22 = Z_cm1 + gamma_pc * vect_z

        # displacement of aggregate 2
        disp_x = x_cm22 - X_cm2
        disp_y = y_cm22 - Y_cm2
        disp_z = z_cm22 - Z_cm2

        X2_new = X2 + disp_x
        Y2_new = Y2 + disp_y
        Z2_new = Z2 + disp_z

        # sphere 1
        x1_sph1 = X_cm1
        y1_sph1 = Y_cm1
        z1_sph1 = Z_cm1

        d1_min = (
            np.sqrt(
                np.power(X1[prev_cand1] - X_cm1, 2)
                + np.power(Y1[prev_cand1] - Y_cm1, 2)
                + np.power(Z1[prev_cand1] - Z_cm1, 2)
            )
            - R1[prev_cand1]
        )
        d1_max = (
            np.sqrt(
                np.power(X1[prev_cand1] - X_cm1, 2)
                + np.power(Y1[prev_cand1] - Y_cm1, 2)
                + np.power(Z1[prev_cand1] - Z_cm1, 2)
            )
            + R1[prev_cand1]
        )

        # sphere 2
        x2_sph2 = x_cm22
        y2_sph2 = y_cm22
        z2_sph2 = z_cm22
        d2_min = (
            np.sqrt(
                np.power(X2_new[prev_cand2] - x_cm22, 2)
                + np.power(Y2_new[prev_cand2] - y_cm22, 2)
                + np.power(Z2_new[prev_cand2] - z_cm22, 2)
            )
            - R2[prev_cand2]
        )
        d2_max = (
            np.sqrt(
                np.power(X2_new[prev_cand2] - x_cm22, 2)
                + np.power(Y2_new[prev_cand2] - y_cm22, 2)
                + np.power(Z2_new[prev_cand2] - z_cm22, 2)
            )
            + R2[prev_cand2]
        )

        sphere1 = np.array([x1_sph1, y1_sph1, z1_sph1, d1_min, d1_max])
        sphere2 = np.array([x2_sph2, y2_sph2, z2_sph2, d2_min, d2_max])

        u_s1_cm1 = np.array([X_cm1, Y_cm1, Z_cm1])
        if ext_case == 1:
            if np.abs(d2_max - d1_max) < gamma_pc:
                case = 1
                x, y, z, sph1_r = random_point_SC(case, sphere1, sphere2)
                u_s1_cm1 = np.array([(X_cm1 - x), (Y_cm1 - y), (Z_cm1 - z)])
            elif d2_max - d1_max > gamma_pc and d2_min - d1_max < gamma_pc:
                case = 2
                x, y, z, sph1_r = random_point_SC(case, sphere1, sphere2)
                u_s1_cm1 = np.array([(X_cm1 - x), (Y_cm1 - y), (Z_cm1 - z)])
            elif d1_max - d2_max > gamma_pc and d1_min - d2_max < gamma_pc:
                case = 3
                x, y, z, sph1_r = random_point_SC(case, sphere1, sphere2)
                u_s1_cm1 = np.array([(x - X_cm1), (y - Y_cm1), (z - Z_cm1)])
        elif ext_case == 0:
            sph1_r = d1_max
            sph2_r = d2_max

            sphere1 = np.array([x1_sph1, y1_sph1, z1_sph1, sph1_r])
            sphere2 = np.array([x2_sph2, y2_sph2, z2_sph2, sph2_r])
            x, y, z, _, _, _ = CCA_2_sphere_intersec(sphere1, sphere2)
            u_s1_cm1 = np.array([X_cm1 - x, Y_cm1 - y, Z_cm1 - z])

        u_s1_cm1 = u_s1_cm1 / np.linalg.norm(u_s1_cm1)
        disp_s1 = R1[prev_cand1]

        x = x + disp_s1 * u_s1_cm1[0]
        y = y + disp_s1 * u_s1_cm1[1]
        z = z + disp_s1 * u_s1_cm1[2]

        # update coordinates of monomers of aggregate 1
        x_cm11 = X_cm1
        y_cm11 = Y_cm1
        z_cm11 = Z_cm1

        Z1_new = Z1
        Y1_new = Y1
        X1_new = X1

        v1 = np.array(
            [
                X1_new[prev_cand1] - x_cm11,
                Y1_new[prev_cand1] - y_cm11,
                Z1_new[prev_cand1] - z_cm11,
            ]
        )
        v2 = np.array([x - x_cm11, y - y_cm11, z - z_cm11])
        s_vec = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        As = np.array(
            [
                [0, -s_vec[2], s_vec[1]],
                [s_vec[2], 0, -s_vec[0]],
                [-s_vec[1], s_vec[0], 0],
            ]
        )
        rot = (
            np.identity(3)
            + np.sin(angle) * As
            + (1 - np.cos(angle)) * np.matmul(As, As)
        )

        for i in range(n1):
            new_c = np.matmul(
                rot,
                np.array([X1_new[i] - x_cm11, Y1_new[i] - y_cm11, Z1_new[i] - z_cm11]),
            )
            X1_new[i] = x_cm11 + new_c[0]
            Y1_new[i] = y_cm11 + new_c[1]
            Z1_new[i] = z_cm11 + new_c[2]

        sph2_r = np.sqrt(
            np.power(X2_new[prev_cand2] - x_cm22, 2)
            + np.power(Y2_new[prev_cand2] - y_cm22, 2)
            + np.power(Z2_new[prev_cand2] - z_cm22, 2)
        )
        sph2_x = x_cm22
        sph2_y = y_cm22
        sph2_z = z_cm22

        sph1_r = R1[prev_cand1] + R2[prev_cand2]
        sph1_x = X1_new[prev_cand1]
        sph1_y = Y1_new[prev_cand1]
        sph1_z = Z1_new[prev_cand1]

        sphere1 = np.array([sph1_x, sph1_y, sph1_z, sph1_r])
        sphere2 = np.array([sph2_x, sph2_y, sph2_z, sph2_r])

        x, y, z, vec0, i_vec, j_vec = CCA_2_sphere_intersec(sphere1, sphere2)

        v1 = np.array(
            [
                X2_new[prev_cand2] - x_cm22,
                Y2_new[prev_cand2] - y_cm22,
                Z2_new[prev_cand2] - z_cm22,
            ]
        )
        v2 = np.array([x - x_cm22, y - y_cm22, z - z_cm22])
        s_vec = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        As = np.array(
            [
                [0, -s_vec[2], s_vec[1]],
                [s_vec[2], 0, -s_vec[0]],
                [-s_vec[1], s_vec[0], 0],
            ]
        )
        rot = (
            np.identity(3)
            + np.sin(angle) * As
            + (1 - np.cos(angle)) * np.matmul(As, As)
        )

        for i in range(n2):
            new_c = np.matmul(
                rot,
                np.array([X2_new[i] - x_cm22, Y2_new[i] - y_cm22, Z2_new[i] - z_cm22]),
            )
            X2_new[i] = x_cm22 + new_c[0]
            Y2_new[i] = y_cm22 + new_c[1]
            Z2_new[i] = z_cm22 + new_c[2]

        CM2 = np.array([x_cm22, y_cm22, z_cm22])
        COR1[:, 0] = X1_new
        COR1[:, 1] = Y1_new
        COR1[:, 2] = Z1_new

        COR2[:, 0] = X2_new
        COR2[:, 1] = Y2_new
        COR2[:, 2] = Z2_new

        return COR1, COR2, CM2, vec0, i_vec, j_vec
    else:
        vec0 = np.zeros(())
        i_vec = np.zeros(())
        j_vec = np.zeros(())
        return COR1, COR2, CM2, vec0, i_vec, j_vec


def random_point_SC(case: int, sphere1: np.ndarray, sphere2: np.ndarray):
    phi_crit_max = 0
    phi_crit_min = 0
    sph1_r_max = sphere1[3]
    sph1_r_min = sphere1[4]

    sph2_r_max = sphere2[3]
    sph2_r_min = sphere2[4]
    if case == 1:
        sphere1[3] = sphere1[4]
        sphere2[3] = sphere2[4]
        phi_crit_max = spherical_cap_angle(sphere1, sphere2)
        norm12 = np.linalg.norm(sphere1 - sphere2)

        if sph1_r_max + sph2_r_min > norm12:
            sphere1[3] = sph1_r_max
            sphere2[3] = sph2_r_min
            phi_crit_min = spherical_cap_angle(sphere1, sphere2)
        else:
            phi_crit_min = 0
        sph1_r = sph1_r_max
    elif case == 2:
        sphere1[3] = sphere1[4]
        sphere2[3] = sphere2[3]
        phi_crit_max = spherical_cap_angle(sphere1, sphere2)
        phi_crit_min = 0
        sph1_r = sph1_r_max
    elif case == 3:
        sphere1[3] = sphere1[3]
        sphere2[3] = sphere2[4]
        phi_crit_max = spherical_cap_angle(sphere1, sphere2)
        phi_crit_min = 0
        sph1_r = sph1_r_min
    uu = np.random.rand()
    theta_r = 2 * np.pi * uu
    uu = np.random.rand()
    phi_r = phi_crit_min + (phi_crit_max - phi_crit_min) * uu

    sph1_x = sphere1[0]
    sph1_y = sphere1[1]
    sph1_z = sphere1[2]

    x = sph1_x + sph1_r * np.cos(theta_r) * np.sin(phi_r)
    y = sph1_y + sph1_r * np.sin(theta_r) * np.sin(phi_r)
    z = sph1_z + sph1_r * np.cos(phi_r)

    # rotate point to return to original coordinate system
    r12 = sphere2 - sphere1
    r12 = r12 / np.linalg.norm(r12)

    v1 = np.array([0, 0, 1])
    v2 = copy.deepcopy(r12)
    s_vec = np.cross(v1, v2)
    s_vec /= np.linalg.norm(s_vec)
    angle = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    As = np.array(
        [[0, -s_vec[2], s_vec[1]], [s_vec[2], 0, -s_vec[0]], -s_vec[1], s_vec[2], 0]
    )
    Rot = np.identity(3) + np.sin(angle) * As + (1 - np.cos(angle)) * np.matmul(As, As)

    new_c = np.matmul(Rot, np.array([x - sph1_x, y - sph1_y, z - sph1_z]))

    x = sph1_x + new_c[0]
    y = sph1_y + new_c[1]
    z = sph1_z + new_c[2]

    return x, y, z, sph1_r


def spherical_cap_angle(sphere1: np.ndarray, sphere2: np.ndarray):
    A = 2 * (sphere2[0] - sphere1[0])
    B = 2 * (sphere2[1] - sphere1[1])
    C = 2 * (sphere2[2] - sphere1[2])
    D = (
        np.power(sphere1[0], 2)
        - np.power(sphere2[0], 2)
        + np.power(sphere1[1], 2)
        - np.power(sphere2[1], 2)
        + np.power(sphere1[2], 2)
        - np.power(sphere2[2], 2)
        - np.power(sphere1[3], 2)
        + np.power(sphere2[3], 2)
    )

    t = (sphere1[0] * A + sphere1[1] * B + sphere1[2] * C + D) / (
        A * (sphere1[0] - sphere2[0])
        + B * (sphere1[1] - sphere2[1])
        + C * (sphere1[2] - sphere2[2])
    )

    x0 = sphere1[0] + t * (sphere2[0] - sphere1[0])
    y0 = sphere1[1] + t * (sphere2[1] - sphere1[1])
    z0 = sphere1[2] + t * (sphere2[2] - sphere1[2])

    distance = np.sqrt(
        np.power(sphere2[0] - sphere1[0], 2)
        + np.power(sphere2[1] - sphere1[1], 2)
        + np.power(sphere2[2] - sphere1[2], 2)
    )
    alpha_0 = np.arccos(
        (np.power(sphere1[3], 2) + np.power(distance, 2) - np.power(sphere2[3], 2))
        / (2 * sphere1[3] * distance)
    )
    r0 = sphere1[3] * np.sin(alpha_0)
    lc_cm1 = np.sqrt(
        np.power(t * (sphere2[0] - sphere1[0]), 2)
        + np.power(t * (sphere2[1] - sphere1[1]), 2)
        + np.power(t * (sphere2[2] - sphere1[2]), 2)
    )
    lp_cm1 = np.sqrt(np.power(lc_cm1, 2) + np.power(r0, 2))
    if t < 0:
        lp_cm1 *= -1
    phi_crit = np.arccos(lc_cm1 / lp_cm1)

    return phi_crit


def CCA_2_sphere_intersec(sphere1: np.ndarray, sphere2: np.ndarray):
    A = 2 * (sphere2[0] - sphere1[0])
    B = 2 * (sphere2[1] - sphere1[1])
    C = 2 * (sphere2[2] - sphere1[2])
    D = (
        np.power(sphere1[0], 2)
        - np.power(sphere2[0], 2)
        + np.power(sphere1[1], 2)
        - np.power(sphere2[1], 2)
        + np.power(sphere1[2], 2)
        - np.power(sphere2[2], 2)
        - np.power(sphere1[3], 2)
        + np.power(sphere2[3], 2)
    )
    t = (sphere1[0] * A + sphere1[1] * B + sphere1[2] * C + D) / (
        A * (sphere1[0] - sphere2[0])
        + B * (sphere1[1] - sphere2[1])
        + C * (sphere1[2] - sphere2[2])
    )

    x0 = sphere1[0] + t * (sphere2[0] - sphere1[0])
    y0 = sphere1[1] + t * (sphere2[1] - sphere1[1])
    z0 = sphere1[2] + t * (sphere2[2] - sphere1[2])

    distance = np.sqrt(
        np.power(sphere2[0] - sphere1[0], 2)
        + np.power(sphere2[1] - sphere1[1], 2)
        + np.power(sphere2[2] - sphere1[2], 2)
    )

    alpha_0 = np.arccos(
        (np.power(sphere1[3], 2) + np.power(distance, 2) - np.power(sphere2[3], 2))
        / (2 * sphere1[3] * distance)
    )
    r0 = sphere1[3] * np.sin(alpha_0)

    AmBdC = -(A + B) / C

    k_vec = np.array([A, B, C]) / np.sqrt(
        np.power(A, 2) + np.power(B, 2) + np.power(C, 2)
    )
    i_vec = np.array([1, 1, AmBdC]) / np.sqrt(1 + 1 + np.power(AmBdC, 2))
    j_vec = np.cross(k_vec, i_vec)

    uu = np.random.rand()
    theta = np.pi * 2 * uu

    x = x0 + r0 * np.cos(theta) * i_vec[0] + r0 * np.sin(theta) * j_vec[0]
    y = y0 + r0 * np.cos(theta) * i_vec[1] + r0 * np.sin(theta) * j_vec[1]
    z = z0 + r0 * np.cos(theta) * i_vec[2] + r0 * np.sin(theta) * j_vec[2]
    vec0 = np.array([x0, y0, z0, r0])
    return x, y, z, vec0, i_vec, j_vec


@numba.jit()
def CCA_overlap_check(
    n1: int,
    n2: int,
    X1: np.ndarray,
    X2: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
):
    cov_max = 0

    for i in range(n1):
        for j in range(n2):
            d_ij = np.sqrt(
                np.power(X1[i] - X2[j], 2)
                + np.power(Y1[i] - Y2[j], 2)
                + np.power(Z1[i] - Z2[j], 2)
            )
            if d_ij < R1[i] + R2[j]:
                c_ij = (R1[i] + R2[j] - d_ij) / (R1[i] + R2[j])
                if c_ij > cov_max:
                    cov_max = c_ij
    return cov_max


@numba.jit()
def CCA_sticking_process_v2(
    CM2: np.ndarray,
    vec0: np.ndarray,
    X2_new: np.ndarray,
    Y2_new: np.ndarray,
    Z2_new: np.ndarray,
    i_vec: np.ndarray,
    j_vec: np.ndarray,
    prev_cand: int,
):
    uu = np.random.rand()
    theta_a = 2 * np.pi * uu

    x = (
        vec0[0]
        + vec0[3] * np.cos(theta_a) * i_vec[0]
        + vec0[3] * np.sin(theta_a) * j_vec[0]
    )
    y = (
        vec0[1]
        + vec0[3] * np.cos(theta_a) * i_vec[1]
        + vec0[3] * np.sin(theta_a) * j_vec[1]
    )
    z = (
        vec0[2]
        + vec0[3] * np.cos(theta_a) * i_vec[2]
        + vec0[3] * np.sin(theta_a) * j_vec[2]
    )

    v1 = np.array(
        [
            X2_new[prev_cand] - CM2[0],
            Y2_new[prev_cand] - CM2[1],
            Z2_new[prev_cand] - CM2[2],
        ]
    )
    v2 = np.array([x - CM2[0], y - CM2[1], z - CM2[2]])
    s_vec = np.cross(v1, v2) / my_norm(np.cross(v1, v2))
    # print(f"{np.linalg.norm(np.cross(v1,v2)) = }")
    # print(f"{my_norm(np.cross(v1,v2)) = }")

    if (
        np.dot(v1, v2) / abs(np.dot(v1, v2)) > 1
        or np.dot(v1, v2) / abs(np.dot(v1, v2)) < -1
    ):
        angle = np.arccos(1)
    else:
        angle = np.arccos(np.dot(v1, v2) / (my_norm(v1) * my_norm(v2)))

    As = np.array(
        [[0, -s_vec[2], s_vec[1]], [s_vec[2], 0, -s_vec[0]], [-s_vec[1], s_vec[0], 0]]
    )
    # rot = np.identity(3) + np.sin(angle)*As + (1-np.cos(angle)) * np.matmul(As,As)
    rot = np.identity(3) + np.sin(angle) * As + (1 - np.cos(angle)) * (As @ As)

    for i in range(X2_new.shape[0]):
        # new_c = np.matmul(rot, np.array([X2_new[i]-CM2[0], Y2_new[i]-CM2[1], Z2_new[i]-CM2[2]]))
        new_c = rot @ np.array(
            [X2_new[i] - CM2[0], Y2_new[i] - CM2[1], Z2_new[i] - CM2[2]]
        )
        X2_new[i] = CM2[0] + new_c[0]
        Y2_new[i] = CM2[1] + new_c[1]
        Z2_new[i] = CM2[2] + new_c[2]

    return X2_new, Y2_new, Z2_new


def filename_generate(n: int, df: float, kf: float) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return re.sub(r"[\.\,\s]", "_", f"N{n}-D{df}-K{kf}-{now}")


def save_results(
    data: pl.DataFrame,
    iteration: int,
    folder: str = "",
    filename: str = "test",
):
    path = Path(folder)
    print(path)
    if not path.exists():
        print(f"`{path}` does not exist. Making the directory for you!")
        path.mkdir(parents=True, exist_ok=True)
    path /= f"{filename}_{iteration}.csv"
    data.write_csv(path)


def sort_rows(i_orden: np.ndarray):
    c_sort = 2
    for irow in range(i_orden.shape[0]):
        krow = np.argmin(i_orden[irow : i_orden.shape[0], c_sort]) + irow

        temp = copy.deepcopy(i_orden[irow, :])
        i_orden[irow, :] = i_orden[krow, :]
        i_orden[krow, :] = temp

    return i_orden


@numba.jit()
def my_norm(a: np.ndarray) -> float:
    n = np.sqrt(np.power(a[0], 2) + np.power(a[1], 2) + np.power(a[2], 2))
    return n
