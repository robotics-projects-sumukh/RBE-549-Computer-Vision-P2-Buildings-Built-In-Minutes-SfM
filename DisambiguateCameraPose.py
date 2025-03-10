import numpy as np

def countFrontPoints(R, C, points):
    count = 0
    r3 = R[2]
    C = C.reshape(3,1)
    for pt in points:
        X = pt[:3].reshape(3,1)
        if r3.dot(X - C) > 0 and X[2] > 0:
            count += 1
    return count

def disambiguateCameraPose(poses, triangulatedPoints):
    bestScore = -1
    selectedPose = None
    worldPoints = None
    
    for pose, cloud in zip(poses, triangulatedPoints):
        score = countFrontPoints(pose[0], pose[1], cloud)
        if score > bestScore:
            bestScore = score
            selectedPose = pose
            worldPoints = cloud
            
    return selectedPose, worldPoints