import numpy as np

def PCA(number: int, mass: np.ndarray, r: np.ndarray, Df: float, kf: float, tolerance: float) -> None:
    # Continue here
    n1,m1,rg1,x_cm,y_cm,z_cm = First_two_monomers(r, mass, number, Df, kf)

    if number > 2:
        k=3
    while k < number:
        n2 = 1
        m2 = mass[k-1]

        rg2 = np.sqrt(0.6*r[k-1])

        n3 = n1+n2
        m3 = m1+m3

    rg3 = (np.exp(np.sum(np.log(r))/r.size))*np.power(n3/kf,1/Df)



    print(tolerance)

def PCA_subcluster(N: int, N_subcluster: int, R: np.ndarray, DF: float, kf: float, tolerance: float) -> bool:
    print(R,DF,kf,tolerance)
    PCA_OK = True
    N_clusters = np.floor(N/N_subcluster)

    # if (int(mod(real(N),real(N_subcl))) .NE. 0) then
    #     Number_clusters = Number_clusters +1
    #     allocate(N_subcl_m(Number_clusters))
    #     N_subcl_m(1:(Number_clusters-1)) = N_subcl
    #     N_subcl_m(Number_clusters) = N - N_subcl*(Number_clusters-1)
    # else
    #     allocate(N_subcl_m(Number_clusters))
    #     N_subcl_m(1:Number_clusters) = N_subcl
    # end if

    if int(np.mod(N,N_subcluster)) != 0:
        N_clusters = N_clusters + 1
        N_subcluster_m = np.ones((N_clusters)) * N_subcluster
        N_subcluster_m[-1] = N - N_subcluster*(N_clusters-1)
    else:
        N_subcluster_m = np.ones(N_clusters)*N_subcluster

    Na = 1
    acum = 0

    i_orden = np.zeros((N_clusters,3))

    for i in range(1,N_clusters):
        number = N_subcluster_m[i]
        radius = R[Na-1:Na+number-2]
        mass = np.zeros((number))

        for j in range(1,radius.size):
            mass[j] = 4/3 * np.pi * np.power(R[j],3)

        # TODO: Continue here
        PCA(number,mass,radius,DF,kf,tolerance)
    return PCA_OK

# FIXME: Check this in its entirety
def First_two_monomers(R: np.ndarray,M: np.ndarray,N: int,DF: float,kf:float) -> tuple[float,float,float,float,float,float]:
    X,Y,Z = np.zeros((N))

    u = np.random.rand()
    v = np.random.rand()
    theta = 2*np.pi*u
    phi = np.arccos(2*v-1)

    X[1] = X[0] + (R[0]+R[1])*np.cos(theta)*np.sin(phi)
    Y[1] = Y[0] + (R[0]+R[1])*np.sin(theta)*np.sin(phi)
    Z[1] = Z[0] + (R[0]+R[1])*np.cos(phi)

    m1 = M[0] + M[1]
    n1 = 2

    rg1 = (np.exp(np.sum(np.log(R[:2])))*np.power(n1/kf,1/DF))

    x_cm = (X[1]*M[1]+X[2]*M[2])/(M[1] + M[2])
    y_cm = (Y[1]*M[1]+Y[2]*M[2])/(M[1] + M[2])
    z_cm = (Z[1]*M[1]+Z[2]*M[2])/(M[1] + M[2])

    return n1,m1,rg1,x_cm,y_cm,z_cm
