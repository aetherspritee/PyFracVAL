import numpy as np
import time

def PCA(number: int, mass: np.ndarray, r: np.ndarray, Df: float, kf: float, tolerance: float) -> tuple[bool, np.ndarray]:
    PCA_ok = True
    n1,m1,rg1,x_cm,y_cm,z_cm, X,Y,Z = First_two_monomers(r, mass, number, Df, kf)

    if number > 2:
        k=3
        while k < number+1:
            n2 = 1
            m2 = mass[k-1]

            rg2 = np.sqrt(0.6)*r[k-1]

            n3 = n1+n2
            m3 = m1+m2

            rg3 = (np.exp(np.sum(np.log(r))/r.size))*np.power(n3/kf,1/Df)
            gamma_ok, gamma_pc = gamma_calc(rg1,rg2,rg3,n1,n2,n3)
            monomer_candidates = np.zeros((number-k+1)) # 0 = not considered
            monomer_candidates[0] = 1 # first one has been considered
            candidates, rmax = random_list_selection(gamma_ok, gamma_pc, X,Y,Z,r,n1,x_cm,y_cm,z_cm)
            list_sum = 0

            while list_sum == 0:
                while np.sum(candidates) == 0 and np.sum(monomer_candidates) < number-k:
                    candidates, rmax = search_monomer_candidates(r,mass,monomer_candidates,number,k,n3,Df,kf,rg1,n1,X,Y,Z,x_cm,y_cm,z_cm)

                previous_candidate = 0 

                if np.sum(candidates) > 0:
                    candidates, selected_real = random_list_selection_one(candidates, previous_candidate)
                    previous_candidate = selected_real
                elif np.sum(candidates) == number-k+1:
                    PCA_ok = False

                curr_try = 1
                X_sel = X[selected_real]
                Y_sel = Y[selected_real]
                Z_sel = Z[selected_real]
                R_sel = r[selected_real]
                r_k = r[k-1]

                x_k, y_k, z_k, r0,x0,y0,z0,i_vec,j_vec = sticking_process(X_sel,Y_sel,Z_sel,R_sel,r_k,x_cm,y_cm,z_cm,gamma_pc)
                X[k-1] = x_k
                Y[k-1] = y_k
                Z[k-1] = z_k

                cov_max = overlap_check(X[0:k], Y[0:k], Z[0:k], r[0:k],k)

                while cov_max > tolerance and curr_try < 360:
                    x_k, y_k, z_k,_ = sticking_process2(x0,y0,z0,r0,i_vec,j_vec)

                    X[k-1] = x_k
                    Y[k-1] = y_k
                    Z[k-1] = z_k
                    cov_max = overlap_check(X[0:k], Y[0:k], Z[0:k], r[0:k],k)
                    curr_try += 1

                    if np.mod(curr_try,359) == 0 and np.sum(candidates) > 1:
                        candidates, selected_real = random_list_selection_one(candidates, previous_candidate)
                        X_sel = X[selected_real]
                        Y_sel = Y[selected_real]
                        Z_sel = Z[selected_real]
                        R_sel = r[selected_real]
                        r_k = r[k-1]
                        x_k, y_k, z_k, r0,x0,y0,z0,i_vec,j_vec = sticking_process(X_sel,Y_sel,Z_sel,R_sel,r_k,x_cm,y_cm,z_cm,gamma_pc)
                        X[k-1] = x_k
                        Y[k-1] = y_k
                        Z[k-1] = z_k
                        previous_candidate = selected_real
                        curr_try += 1

                        cov_max = overlap_check(X[0:k], Y[0:k], Z[0:k], r[0:k],k)

                list_sum = np.sum(candidates)

                if cov_max > tolerance:
                    list_sum = 0
                    candidates *= 0


            x_cm = (x_cm*m1 + X[k-1]*m2)/(m1+m2)
            y_cm = (y_cm*m1 + Y[k-1]*m2)/(m1+m2)
            z_cm = (z_cm*m1 + Z[k-1]*m2)/(m1+m2)

            n1 = n3
            m1 = m3
            rg1 = (np.exp(np.sum(np.log(r))/np.log(r).size))*(np.power(n1/kf, 1/Df))
            k = k + 1
    
    data_new = np.zeros((number,4))
    for i in range(number):
        data_new[i,:] = np.array([X[i], Y[i], Z[i], r[i]])
    return PCA_ok, data_new

def PCA_subcluster(N: int, N_subcluster: int, R: np.ndarray, DF: float, kf: float, tolerance: float) -> tuple[bool, np.ndarray, int, np.ndarray]:
    PCA_OK = True
    N_clusters = int(np.floor(N/N_subcluster))
    if int(np.mod(N,N_subcluster)) != 0:
        N_clusters = N_clusters + 1
        N_subcluster_m = np.ones((N_clusters)) * N_subcluster
        N_subcluster_m[-1] = N - N_subcluster*(N_clusters-1)
    else:
        N_subcluster_m = np.ones((int(N_clusters)))*N_subcluster

    Na = 1
    acum = 0

    i_orden = np.zeros((N_clusters,3))
    data = np.zeros((N,4))
    for i in range(1,N_clusters+1):
        number = int(N_subcluster_m[i-1])
        radius = R[Na-1:Na+number-1]
        mass = np.zeros((number))

        for j in range(radius.size):
            mass[j] = 4/3 * np.pi * np.power(R[j],3)

        PCA_OK, data_new = PCA(number,mass,radius,DF,kf,tolerance)

        if i == 0:
            acum = number
            for ii in range(number+1):
                data[ii,:] = data_new[ii,:]
            i_orden[0,0:2] =  np.array([1, acum])
            i_orden[0,2] = acum
        else:
            for ii in range(acum,acum+number):
                data[ii,:] = data_new[ii-acum,:]
            i_orden[i-1,0:2] = np.array([acum+1, acum+number])
            i_orden[i-1,2] = number
            acum += number

        Na += number
    return PCA_OK, data, N_clusters, i_orden

def First_two_monomers(R: np.ndarray,M: np.ndarray,N: int,DF: float,kf:float) -> tuple[int,float,float,float,float,float, np.ndarray, np.ndarray, np.ndarray]:
    X = np.zeros((N))
    Y = np.zeros((N))
    Z = np.zeros((N))

    u = np.random.rand()
    v = np.random.rand()
    theta = 2*np.pi*u
    phi = np.arccos(2*v-1)
    theta = 1
    phi = 1

    X[1] = X[0] + (R[0]+R[1])*np.cos(theta)*np.sin(phi)
    Y[1] = Y[0] + (R[0]+R[1])*np.sin(theta)*np.sin(phi)
    Z[1] = Z[0] + (R[0]+R[1])*np.cos(phi)

    m1 = M[0] + M[1]
    n1 = 2

    rg1 = (np.exp(np.sum(np.log(R[:2]))/2))*np.power(n1/kf,1/DF)

    x_cm = (X[0]*M[0]+X[1]*M[1])/(M[0] + M[1])
    y_cm = (Y[0]*M[0]+Y[1]*M[1])/(M[0] + M[1])
    z_cm = (Z[0]*M[0]+Z[1]*M[1])/(M[0] + M[1])

    return n1,m1,rg1,x_cm,y_cm,z_cm, X,Y,Z

def gamma_calc(rg1: float,rg2: float,rg3: float,n1: int,n2: int,n3: int) -> tuple[bool,float]:
    gamma_ok = True
    gamma_pc = 0.0

    rg3_aux = rg3
    if rg3 < rg1:
        rg3_aux = rg1

    
    if np.power(n3,2)*np.power(rg3_aux,2) > n3*(n1*np.power(rg1,2)+ n2*np.power(rg2,2)):
        gamma_pc = np.sqrt((np.power(n3,2)*np.power(rg3_aux,2)-n3*(n1*np.power(rg1,2)+n2*np.power(rg2,2)))/(n1*n2))
    else:
        gamma_ok = False
    
    return gamma_ok, gamma_pc

def random_list_selection(gamma_ok: bool, gamma_pc: float,X: np.ndarray, Y: np.ndarray, Z: np.ndarray,R: np.ndarray, n1: int, x_cm: float, y_cm: float, z_cm: float) -> tuple[np.ndarray, float]:
    candidates = np.zeros((n1))
    rmax = 0.0
    if gamma_ok:
        for ii in range(n1):
            dist = np.sqrt(np.power(X[ii]-x_cm, 2) + np.power(Y[ii]-y_cm, 2) + np.power(Z[ii]-z_cm, 2))
            if dist > rmax:
                rmax = dist
            if dist > gamma_pc-R[n1]-R[ii] and dist < gamma_pc+R[n1]+R[ii]:
                candidates[ii] = 1
            if R[n1]+R[ii] > gamma_pc:
                candidates[ii] = 0

    return candidates, rmax

def search_monomer_candidates(R: np.ndarray, M: np.ndarray, monomer_candidates: np.ndarray,N: int, k: int, n3: int, Df: float,kf: float, rg1: float, n1: int, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, x_cm: float, y_cm: float, z_cm: float):
    R_sl = R
    M_sl = M 
    vector_search = np.zeros((N-k+1))
    for i in range(vector_search.size):
        vector_search[i] = i+k

    for i in range(monomer_candidates.size):
        if monomer_candidates[i] == 1:
            vector_search[i] = 0
    vector_search2 = vector_search[vector_search != 0] 

    if vector_search2.size > 1:
        u = np.random.rand()
        RS_1 = int(vector_search2[int(u*(vector_search.size-1))])
    else:
        RS_1 = int(vector_search2[0])

    R[RS_1-1] = R_sl[k]
    R[k] = R_sl[RS_1-1]
    M[RS_1-1] = M_sl[k]
    M_sl[k] = M[RS_1-1]

    m2 = M[k-1]
    rg2 = np.sqrt(0.6*R[k])
    m3 = np.sum(M[0:k-1])
    rg3 = (np.exp(np.sum(np.log(R))/np.log(R).size))*np.power(n3/kf,1./Df)

    gamma_ok, gamma_pc = gamma_calc(rg1,rg2,rg3,n1,1,n3)

    candidates, rmax = random_list_selection(gamma_ok, gamma_pc, X,Y,Z,R,n1,x_cm,y_cm,z_cm)

    candidates[RS_1-k-1] = 1
    return candidates, rmax

def random_list_selection_one(candidates: np.ndarray, previous_candidate: int):
    if previous_candidate > 0:
        candidates[previous_candidate] = 0
    candidates2 = candidates[candidates > 0]
    n = np.random.rand()
    selected = 1+int(n*(candidates2.size-1))

    selected_real = 0
    j = 0
    for i in range(candidates.size):
        if candidates[i] > 0:
            j += 1
        if j == selected:
            selected_real = i
            break
    return candidates, selected_real

def sticking_process(x: float,y: float,z: float,r: float,r_k: float, x_cm: float, y_cm: float, z_cm: float, gamma_pc: float):
    x1 = x
    y1 = y
    z1 = z
    r1 = r + r_k
    x2 = x_cm
    y2 = y_cm
    z2 = z_cm
    r2 = gamma_pc

    a = 2*(x2-x1)
    b = 2*(y2-y1)
    c = 2*(z2-z1)
    d = np.power(x1,2)-np.power(x2,2) + np.power(y1,2)-np.power(y2,2) + np.power(z1,2)-np.power(z2,2) - np.power(r1,2)+np.power(r2,2)

    t_sp = (x1*a + y1*b + z1*c + d)/(a*(x1-x2) + b*(y1-y2) + c*(z1-z2))

    x0 = x1 + t_sp*(x2-x1)
    y0 = y1 + t_sp*(y2-y1)
    z0 = z1 + t_sp*(z2-z1)

    distance = np.sqrt(np.power(x2-x1,2) + np.power(y2-y1,2) + np.power(z2-z1,2))

    alpha = np.arccos((np.power(r1,2) + np.power(distance,2) - np.power(r2,2))/(2*r1*distance))
    r0 = r1*np.sin(alpha)

    # AmBdC = (A+B)/C
    AmBdC = (a+b)/c

    k_vec = np.array([a,b,c])/np.sqrt(a**2+b**2+c**2)
    i_vec = np.array([1,1,-AmBdC])/np.sqrt(1+1+AmBdC**2)
    j_vec = np.cross(k_vec,i_vec)

    u = np.random.rand()
    v = np.random.rand()
    theta = 2.*np.pi*u
    phi = np.arccos(2.*v-1.)

    x_k = x0 + r0*np.cos(theta)*i_vec[0]+r0*np.sin(theta)*j_vec[0]
    y_k = y0 + r0*np.cos(theta)*i_vec[1]+r0*np.sin(theta)*j_vec[1]
    z_k = z0 + r0*np.cos(theta)*i_vec[2]+r0*np.sin(theta)*j_vec[2]

    return x_k, y_k, z_k, r0, x0,y0,z0,i_vec, j_vec

def sticking_process2(x0, y0, z0, r0,i_vec,j_vec):
    u = np.random.rand()
    theta = 2 * np.pi * u 

    x_k = x0 + r0*np.cos(theta)*i_vec[0]+r0*np.sin(theta)*j_vec[0]
    y_k = y0 + r0*np.cos(theta)*i_vec[1]+r0*np.sin(theta)*j_vec[1]
    z_k = z0 + r0*np.cos(theta)*i_vec[2]+r0*np.sin(theta)*j_vec[2]
    return x_k, y_k, z_k, theta

def overlap_check(x: np.ndarray, y: np.ndarray, z: np.ndarray, r: np.ndarray, k: int):
    C = np.zeros((k-1))
    for i in range(k-1):
        distance_kj = np.sqrt(np.power(x[k-1]-x[i],2) + np.power(y[k-1]-y[i],2) + np.power(z[k-1]-z[i],2))

        if distance_kj < (r[k-1]+r[i]):
            C[i] = ((r[k-1]+r[i]) - distance_kj)/(r[k-1]+r[i])
        else:
            C[i] = 0

    return np.max(C)
