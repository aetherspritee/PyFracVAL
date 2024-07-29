'''
hi!
'''

import numpy as np
from CCA import CCA_subcluster

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

    iter = 1
    while isFine:
        CCA_ok, PCA_ok = CCA_subcluster(R,N,DF,Kf, iter)
        isFine = CCA_ok and PCA_ok
        if not isFine:
            print("Restarting, wasnt able to generate aggregate")

    
    print("Successfully generated aggregate")
