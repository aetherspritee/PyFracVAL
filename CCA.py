import numpy as np

# call CCA_sub(not_able_cca,not_able_pca,N,rp_g, Df,kf, R, tol_ov)
def CCA(R: np.ndarray, N: int, DF: float, kf: float, tolerance: float=1e-7):
    CCA_OK = True
    PCA_OK = True
    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    return X,Y,Z ,CCA_OK, PCA_OK
