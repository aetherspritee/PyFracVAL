'''
hi!
'''

import numpy as np
from CCA import CCA

# config
DF = 2.0
Kf = 1.0
N = 100
R0 = 1
SIGMA = 0

def shuffle(arr: np.ndarray) -> np.ndarray:
    return arr

def main():
    R = np.zeros((N))
    R = shuffle(R)
    isFine = True

    while isFine:
        X,Y,Z, CCA_ok, PCA_ok = CCA(R,N,DF,Kf)
        isFine = CCA_ok and PCA_ok
