'''
hi!
'''

import numpy as np
from CCA import CCA_subcluster

# config
DF = 2.0
Kf = 1.0
N = 64
R0 = 0.01
SIGMA = 0
EXT_CASE = 0

def shuffle(arr: np.ndarray) -> np.ndarray:
    return arr

def main():
    R = np.ones((N))*R0
    R = shuffle(R)
    isFine = True
    N_subcl_perc = 0.1
    iter = 1
    while isFine:
        CCA_ok, PCA_ok = CCA_subcluster(R,N,DF,Kf, iter, N_subcl_perc, EXT_CASE)
        isFine = CCA_ok and PCA_ok
        if not isFine:
            print("Restarting, wasnt able to generate aggregate")

    
    print("Successfully generated aggregate")

if __name__ == "__main__":
    main()
