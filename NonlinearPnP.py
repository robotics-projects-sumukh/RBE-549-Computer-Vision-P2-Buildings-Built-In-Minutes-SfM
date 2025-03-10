from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from PnPRANSAC import *

def reprojectionLossPnP(x0, imagePoints, worldPoints, K):
    C = x0[:3]
    q = x0[3:]
    R = Rotation.from_quat(q).as_matrix()
    C_ = np.reshape(C, (3, 1))        
    I_ = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I_, -C_))))
    error = []
    for i in range(len(imagePoints)):
        reprojection_error = reprojectionErrorPnP(P, imagePoints[i], worldPoints[i])
        error.append(reprojection_error)    
    return np.mean(error)
    

def NonLinearPnP(imagePoints, worldPoints, R, C, K):
    quat = Rotation.from_matrix(R).as_quat()
    optim_params = least_squares(
        fun=reprojectionLossPnP, 
        x0=np.hstack([C, quat]), 
        method="trf", 
        args=[imagePoints, worldPoints, K], 
        max_nfev=5000,
        verbose=0)
    params = optim_params.x
    C = params[:3]
    R = Rotation.from_quat(params[3:]).as_matrix()
    return R, C.reshape(-1,1)

    