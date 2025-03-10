import numpy as np
from LinearPnP import LinearPnP

def reprojectionErrorPnP(P, imagePoint, worldPoint, verbose=False):
    
    x0 = worldPoint
    row1Proj, row2Proj, row3Proj = P 
    row1Proj, row2Proj, row3Proj = row1Proj.reshape(1,-1), row2Proj.reshape(1,-1),row3Proj.reshape(1,-1)
    
    uObs,vObs = imagePoint[0], imagePoint[1]
    uProj = np.divide(row1Proj.dot(x0) , row3Proj.dot(x0))
    vProj =  np.divide(row2Proj.dot(x0) , row3Proj.dot(x0))
    
    error = np.square(vObs - vProj) + np.square(uObs - uProj)
    
    if verbose:
        print(f"Observed : ({uObs:.2f}, {vObs:.2f}), "
              f"Projected: ({uProj[0][0]:.2f}, {vProj[0][0]:.2f}), "
              f"Error: {error[0][0]:.2f}")
    return error


def PnPRANSAC(imagePoints, worldPoints, K, maxIterations=1000, flag=False):
    bestInliers = []
    bestPose = None
    threshold = 30
    imagePoints = np.array(imagePoints)
    worldPoints = np.array(worldPoints)
    
    for _ in range(maxIterations):
        indices = np.random.choice(len(imagePoints), 6, replace=False)
        sampledImagePoints = imagePoints[indices]
        sampledWorldPoints = worldPoints[indices]
        R, C = LinearPnP(sampledImagePoints, sampledWorldPoints, K)
        C_ = np.reshape(C, (3, 1))        
        I_ = np.identity(3)
        P = np.dot(K, np.dot(R, np.hstack((I_, -C_))))
        inliers = []
        for i in range(len(imagePoints)):
            reprojectionError = reprojectionErrorPnP(P, imagePoints[i], worldPoints[i], flag)
            if reprojectionError < threshold:
                inliers.append(i)
        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            bestPose = (R, C)
    
    return bestInliers, bestPose[0], bestPose[1]


    