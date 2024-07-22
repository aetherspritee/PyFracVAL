import numpy as np
from PCA import PCA, PCA_subcluster

# call CCA_sub(not_able_cca,not_able_pca,N,rp_g, Df,kf, R, tol_ov)
def CCA(R: np.ndarray, N: int, DF: float, kf: float, tolerance: float=1e-7):
    CCA_OK = True
    print(R,DF,kf,tolerance)
    
    if N <= 50:
        N_subcluster = 5
    else:
        N_subcluster = 50

    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    
    # TODO: Continue here!
    PCA_OK = PCA_subcluster(N, N_subcluster, R, DF, kf, tolerance)

    if not PCA_OK:
        return X,Y,Z,R ,CCA_OK, PCA_OK

    I_total = 10

    while I_total > 1:
        ID_agglom = generate_CCA_pairs(I_total)


def generate_CCA_pairs(I: int) -> np.ndarray:
    ID_agglom = np.zeros((I,I))
    return ID_agglom
