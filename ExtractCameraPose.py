import numpy as np

def getCameraPoses(E):
    W = np.array([[0, -1, 0], 
                  [1, 0, 0], 
                  [0, 0, 1]])
    
    U, _, Vt = np.linalg.svd(E)

    R1 = np.dot(np.dot(U, W), Vt)
    C1 = U[:, 2].reshape(3, 1)

    R2 = np.dot(np.dot(U, W), Vt)
    C2 = -U[:, 2].reshape(3, 1)
    
    R3 = np.dot(np.dot(U, W.T), Vt)
    C3 = U[:, 2].reshape(3, 1)
    
    R4 = np.dot(np.dot(U, W.T), Vt)
    C4 = -U[:, 2].reshape(3, 1)
    
    if np.linalg.det(R1) < 0:
        print("C1 and R1 are corrected")
        R1 = -R1
        C1 = -C1
        # print (R1)
        # print (C1)
    if np.linalg.det(R2) < 0:
        print("C2 and R2 are corrected")
        R2 = -R2
        C2 = -C2
        # print (R2)
        # print (C2)
    if np.linalg.det(R3) < 0:
        print("C3 and R3 are corrected")
        R3 = -R3
        C3 = -C3
        # print (R3) 
        # print (C3)
    if np.linalg.det(R4) < 0:
        print("C4 and R4 are corrected")
        R4 = -R4
        C4 = -C4
        # print (R4)
        # print (C4)

    return [(R1, C1), (R2, C2), (R3, C3), (R4, C4)]