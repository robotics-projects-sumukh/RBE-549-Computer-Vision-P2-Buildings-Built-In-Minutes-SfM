import numpy as np
from EstimateFundamentalMatrix import *

def getBestInliers(matches, threshold):
    bestInliers = set()
    maxInliers = 0
    image1_points = np.array([match['image1_points'] + (1,) for match in matches])
    image2_points = np.array([match['image2_points'] + (1,) for match in matches])
    for i in range(100):
        sampleIndices = np.random.choice(matches, size=8, replace=False)
        # Estimate the fundamental matrix
        F = getFundamentalMatrix(sampleIndices)
        currentInliers = set()
        for j in range(len(matches)):
            points1 = np.array(matches[j]['image1_points'] + (1,))
            points2 = np.array(matches[j]['image2_points'] + (1,))
            error = np.abs(np.dot(np.dot(points1.T, F), points2)) 
            if error < threshold:
                currentInliers.add(j)
        if len(currentInliers) > maxInliers:
            maxInliers = len(currentInliers)
            bestInliers = currentInliers
            bestMatches = np.array(matches)[list(bestInliers)]
    _, mask = cv2.findFundamentalMat(image1_points, image2_points, cv2.FM_RANSAC)
    bestMatches = np.array(matches)[mask.ravel() == 1]
    return bestMatches
