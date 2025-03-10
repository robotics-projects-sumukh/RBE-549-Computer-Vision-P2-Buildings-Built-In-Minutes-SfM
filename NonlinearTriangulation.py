import scipy
import numpy as np

def reprojectionError(X0, P1, P2, imagePoint1, imagePoint2):
    # Extract rows from first projection matrix
    row1Proj1, row2Proj1, row3Proj1 = P1
    row1Proj1 = row1Proj1.reshape(1, -1)
    row2Proj1 = row2Proj1.reshape(1, -1)
    row3Proj1 = row3Proj1.reshape(1, -1)

    # Extract rows from second projection matrix
    row1Proj2, row2Proj2, row3Proj2 = P2
    row1Proj2 = row1Proj2.reshape(1, -1)
    row2Proj2 = row2Proj2.reshape(1, -1)
    row3Proj2 = row3Proj2.reshape(1, -1)

    # Calculate projections for first camera
    u_obs1, v_obs1 = imagePoint1[0], imagePoint1[1]
    uProj1 = np.divide(row1Proj1.dot(X0), row3Proj1.dot(X0))
    vProj1 = np.divide(row2Proj1.dot(X0), row3Proj1.dot(X0))
    error1 = np.square(v_obs1 - vProj1) + np.square(u_obs1 - uProj1)

    # Calculate projections for second camera
    u_obs2, v_obs2 = imagePoint2[0], imagePoint2[1]
    uProj2 = np.divide(row1Proj2.dot(X0), row3Proj2.dot(X0))
    vProj2 = np.divide(row2Proj2.dot(X0), row3Proj2.dot(X0))
    error2 = np.square(v_obs2 - vProj2) + np.square(u_obs2 - uProj2)

    totalError = error1 + error2
    return totalError

def nonLinearTriangulation(K, R1, C1, R2, C2, worldPoints, pts1, pts2):
    
    I = np.identity(3)
    C1_ = np.reshape(C1, (3, 1))        
    C2_ = np.reshape(C2, (3, 1))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1_))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2_))))

    refinedPoints = []
    for i in range(len(worldPoints)):
        optimizedParams = scipy.optimize.least_squares(
            fun=reprojectionError, 
            x0=worldPoints[i], 
            method="trf", 
            args=[P1, P2, pts1[i], pts2[i]]
        )
        
        X1 = optimizedParams.x
        refinedPoints.append(X1)
    refinedPoints = np.array(refinedPoints)
    refinedPoints = refinedPoints / refinedPoints[:,3].reshape(-1,1)
    return refinedPoints