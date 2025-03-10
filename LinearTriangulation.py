import numpy as np

def getTriangulatedPoints(R1, C1, R2, C2, matches, K):
    P1 = K @ np.hstack((R1, -R1 @ C1.reshape(-1,1)))
    P2 = K @ np.hstack((R2, -R2 @ C2.reshape(-1,1)))
    
    worldPoints = []
    for m in matches:
        pt1 = np.array(m['image1_points'] + (1,))
        pt2 = np.array(m['image2_points'] + (1,))
        
        A = np.vstack([
            pt1[1]*P1[2] - P1[1],
            P1[0] - pt1[0]*P1[2],
            pt2[1]*P2[2] - P2[1],
            P2[0] - pt2[0]*P2[2]
        ])
        
        _,_,Vt = np.linalg.svd(A)
        X = Vt[-1]
        worldPoints.append(X/X[3])
    
    return np.array(worldPoints)