import numpy as np

def LinearPnP(imagePoints, worldPoints, K):
    A = []
    for i in range(imagePoints.shape[0]):
        X, Y, Z, _ = worldPoints[i]
        u, v, _ = imagePoints[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(A)
    _, _, vTemp = np.linalg.svd(A)
    P = vTemp[-1, :].reshape(3, 4)
    temp = np.linalg.inv(K) @ P[:, :3]
    U, D, V = np.linalg.svd(temp)
    R = U @ V
    C = np.linalg.inv(K) @ P[:, 3]/D[0]
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    C = -R.T@C
    return R, C
