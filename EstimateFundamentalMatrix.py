import numpy as np
import cv2
def normalizePoints(points):
    pts = points[:,:2]
    mean = np.mean(pts, axis=0)
    centeroid = pts - mean
    rms = np.sqrt(np.mean(np.sum(centeroid**2, axis=1)))
    if rms == 0:
        return points, np.eye(3)
    scale = np.sqrt(2) / rms
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    if points.shape[1] == 2:
        homogeneous_points = np.hstack((points, np.ones((len(points), 1))))
    else:
        homogeneous_points = points
    normalized_pts = T @ homogeneous_points.T
    return normalized_pts.T, T


def getFundamentalMatrix(featureMatches):
    refPoints = np.array([match['image1_points'] + (1,) for match in featureMatches])
    targetPoints = np.array([match['image2_points'] + (1,) for match in featureMatches])
    
    normRef, T1 = normalizePoints(refPoints)
    normTarget, T2 = normalizePoints(targetPoints)
    A = np.zeros((normRef.shape[0], 9))
    for i in range(normRef.shape[0]):
        x_1, y_1 = normRef[i][0], normRef[i][1]
        x_2, y_2 = normTarget[i][0], normTarget[i][1]
        A[i] = np.array([x_1*x_2, x_2*y_1, 
                         x_2, y_2*x_1, y_2*y_1, 
                         y_2, x_1, y_1, 1])
    
    _, _, V = np.linalg.svd(A)
    F = V.T[:, -1].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F = np.dot(T2.T, np.dot(F, T1))
    F = F / F[2, 2]
    
    return F

