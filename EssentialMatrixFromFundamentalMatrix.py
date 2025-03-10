import numpy as np

def getEssentialMatrix(F, K):
    E = K.T @ F @ K
    U,S,V = np.linalg.svd(E)
    S = [1,1,0]
    return U @ np.diag(S) @ V