from scipy.spatial.transform import Rotation
from GetInlierRANSANC import *
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from LinearPnP import *
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *
import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def loadImageSet(dir):
    imageSet = []
    for i in range(5):
        imageSet.append(cv2.imread(f"{dir}{i+1}.png"))
    return imageSet

def getIntrinsicParams(calibPath):
    with open(calibPath, 'r') as f:
        return np.array([[float(v) for v in line.strip().split()] for line in f])

def getFeatureMatches(filePath, viewPair):
    featureData = []
    seenPoints = set()
    
    with open(filePath, 'r') as f:
        nFeatures = int(f.readline().split(":")[1].strip())
        for line in f:
            parts = line.strip().split()
            nMatches = int(parts[0])
            color = tuple(map(int, parts[1:4]))
            currentUV = tuple(map(float, parts[4:6]))
            
            if currentUV in seenPoints:
                continue
            seenPoints.add(currentUV)

            foundMatch = False
            offset = 6
            while offset < len(parts):
                if int(parts[offset]) == viewPair[1]:
                    pairUV = (float(parts[offset+1]), float(parts[offset+2]))
                    foundMatch = True
                    break
                offset += 3
                
            if foundMatch:    
                featureData.append({
                    'color': color,
                    'image1_points': currentUV,
                    'image2_points': pairUV
                })
    return featureData

def findUniqueFeatures(featureList, viewIndices):
    uniqueFeatures = []
    uniqueIndices = []
    targetKey = f'image{viewIndices[-1]}_points'
    originKey = f'image{viewIndices[0]}_points'
    comparisonKeys = [f'image{num}_points' for num in viewIndices[1:-1]]
    
    for idx, feature in enumerate(featureList):
        if all(key not in feature for key in comparisonKeys):
            if targetKey in feature and originKey in feature:
                uniqueFeatures.append({
                    'image2_points': feature[targetKey],
                    'image1_points': feature[originKey]
                })
                uniqueIndices.append(idx)
    return uniqueFeatures, uniqueIndices

def mergeFeatureMatches(viewPairs, matchLists):
    mergedFeatures = []
    for index, matches in enumerate(matchLists):
        img1, img2 = viewPairs[index]  
        for match in matches:
            newMatch = {
                f'image{img1}_points': match['image1_points'],
                f'image{img2}_points': match['image2_points']
            }
            found = False
            for existingMatch in mergedFeatures:
                commonImages = set(existingMatch.keys()) & set(newMatch.keys())
                if commonImages and all(existingMatch[img] == newMatch[img] for img in commonImages):
                    existingMatch.update(newMatch)
                    found = True
                    break
            if not found:
                mergedFeatures.append(newMatch)
    return mergedFeatures

def findCommonFeatures(featureList, viewIndices):
    commonFeatures = []
    commonIndices = []
    targetKeys = [f'image{idx}_points' for idx in viewIndices]
    
    for idx, feature in enumerate(featureList):
        if all(key in feature for key in targetKeys):
            commonFeatures.append({k: feature[k] for k in targetKeys})
            commonIndices.append(idx)
    return commonFeatures, commonIndices

def getProjectionMatrix(K,R,C):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    projectionMatrix = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return projectionMatrix

def visualizeReprojection(img, projMatrix, worldPoints, imagePoints, title):
    displayImg = img.copy()
    for i in range(len(worldPoints)):
        pt = imagePoints[i]
        projUV = projMatrix @ worldPoints[i]
        projUV /= projUV[2]
        cv2.circle(displayImg, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1)
        cv2.circle(displayImg, (int(projUV[0]), int(projUV[1])), 3, (0,255,0), -1)
    
    os.makedirs('IntermediateOutputImages', exist_ok=True)
    cv2.imwrite(f'IntermediateOutputImages/{title}.png', displayImg)


def drawEpipolarLines(img1, img2, lines, pts1, pts2):
    r,c,_ = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    for line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -line[2]/line[1]])
        x1,y1 = map(int, [c, -(line[2]+line[0]*c)/line[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, color, -1)
        img2 = cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, color, -1)
    return img1, img2


def main(args):
    dir = args.path
    images = loadImageSet(dir)
    instrinsicParameters = getIntrinsicParams(dir + 'calibration.txt')
    imagePairs = []
    for i in range(1, 6):
        for j in range(i+1, 6):
            imagePairs.append((i,j))    
    featureMatches = []
    viewsNum = 5
    for pair in imagePairs:
        featureMatches.append(getFeatureMatches(dir + f'matching{pair[0]}.txt', pair))
        
    Inliers = []
    
    for i in range(10):
        inliers = getBestInliers(featureMatches[i], 0.1)
        Inliers.append(inliers)
    
    mergedFeatures = mergeFeatureMatches(imagePairs[0:viewsNum-1], Inliers[0:viewsNum-1])

    
    bestInliers = []
    bestIndices = []
    for pair in imagePairs[0:viewsNum-1]:
        common, indices = findCommonFeatures(mergedFeatures, pair)
        bestInliers.append(common)
        bestIndices.append(indices)
        
    
    print("="*50)
    print("Calculating Fundamental Matrix")
    print("="*50)
    print("")
    F = getFundamentalMatrix(bestInliers[0])
    print(f"Fundamental Matrix:")
    print("")
    print(F)
    print("")

    pts1 = np.array([match['image1_points'] for match in bestInliers[0]])
    pts2 = np.array([match['image2_points'] for match in bestInliers[0]])
    
    
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    os.makedirs('IntermediateOutputImages', exist_ok=True)

    img5,img6 = drawEpipolarLines(images[0],images[1],lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawEpipolarLines(images[1],images[0],lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.savefig('IntermediateOutputImages/EpipolarLines.png')
    plt.close()
    
    print("="*50)
    print("Calculating Essential Matrix")
    print("="*50)
    print("")
    E = getEssentialMatrix(F, instrinsicParameters)
    print(f"Essential Matrix:")
    print("")
    print(E)
    print("")

    print("="*50)
    print("Extracting Camera Poses")
    print("="*50)
    cameraPoses = getCameraPoses(E)
    
    print("="*50)
    print("Linear Triangulation")
    print("="*50)
    R1 = np.eye(3)
    C1 = np.zeros((3,1))
    triangulatedPoints = []

    for pose in cameraPoses:
        points = getTriangulatedPoints(R1, C1, pose[0], pose[1],
                                        bestInliers[0], instrinsicParameters)
        triangulatedPoints.append(points)
    
    triangulatedPoints = np.array(triangulatedPoints)
    color = ['y', 'g', 'b', 'r']
    for i in range(4):
        plt.axis([-20, 20, -20, 20])
        plt.scatter(triangulatedPoints[i,:,0], triangulatedPoints[i,:,2], s=1, c=color[i])
        
    plt.savefig('IntermediateOutputImages/LinearTriangulation.png')
    plt.close()
        
    
    # Disambiguate the camera poses
    print("="*50)
    print("Disambiguating Camera Pose")
    print("="*50)
    bestCameraPose, correctWorldPoints = disambiguateCameraPose(cameraPoses, triangulatedPoints)
    print(f"Selected Camera Pose:\nR:{bestCameraPose[0]}\nC:{bestCameraPose[1]}")
    
    
    pts1 = np.array([match['image1_points'] + (1,) for match in bestInliers[0]])
    pts2 = np.array([match['image2_points'] + (1,) for match in bestInliers[0]])
    
    print("="*50)
    print("Performing Non Linear Triangulation ")
    print("="*50)
    # Non Linear Triangulation
    refinedPoints = nonLinearTriangulation(instrinsicParameters, np.eye(3), np.zeros((3,1)), 
                                                bestCameraPose[0], bestCameraPose[1], 
                                                correctWorldPoints, pts1, pts2)
    
    P1 = getProjectionMatrix(instrinsicParameters, np.eye(3), np.zeros((3,1)))
    P2 = getProjectionMatrix(instrinsicParameters, bestCameraPose[0], bestCameraPose[1])
    
    error = []
    # Linear Reprojection error
    for i in range(len(correctWorldPoints)):
        error.append(reprojectionError(correctWorldPoints[i], P1, P2, pts1[i], pts2[i]))
    print(f"Linear Reprojection Error: {np.mean(error)}")
    
    error = []
    # Non Linear Reprojection error
    for i in range(len(refinedPoints)):
        error.append(reprojectionError(refinedPoints[i], P1, P2, pts1[i], pts2[i]))

    print(f"Non Linear Reprojection Error: {np.mean(error)}")
    
    visualizeReprojection(images[0], P1, correctWorldPoints, pts1, 'Linear Reprojection')
    visualizeReprojection(images[0], P1, refinedPoints, pts1, 'Non-Linear Reprojection')
    
    
    fig, ax = plt.subplots()
    # Plot the reprojected points
    plt.scatter(correctWorldPoints[:,0], correctWorldPoints[:,2], s=1, c='r', label="Linear Triangulation")
    plt.scatter(refinedPoints[:,0], refinedPoints[:,2], s=1, c='b', label="Non-Linear Triangulation")   
    marker_color = ['y', 'g', 'b', 'r', 'c'] 
    for rotation, position, label in  [(R1, C1, "1"), (bestCameraPose[0], bestCameraPose[1], "2")]:
        angles = Rotation.from_matrix(rotation).as_euler('XYZ')
        angles_deg = np.rad2deg(angles)        
        ax.plot(position[0], position[2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None', label=f'Camera {label}', color=marker_color[int(label)-1])
        correction = -0.1
        ax.annotate(label, xy=(position[0] + correction, position[2] + correction))

    plt.axis([-15, 15, -5, 25])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.savefig('IntermediateOutputImages/Linear_NonLinear_Triangulation.png')
    plt.close()
    
    R_All = []
    C_All = []
    R_All.append(np.eye(3))
    C_All.append(np.zeros((3,1)))
    R_All.append(bestCameraPose[0])
    C_All.append(bestCameraPose[1])

    # Linear PnP and Pnp Ransac
    activeImages = [1,2]
    pnpError = []

    worldPoints = np.zeros((len(mergedFeatures), 4))
    
    for idx in range(refinedPoints.shape[0]):
        worldPoints[bestIndices[0][idx]] = refinedPoints[idx]
        
    for viewIdx in range(viewsNum-2):
        print("="*50)
        print(f"Performing PnP for Image {viewIdx+3}")
        print("="*50)
        activeImages.append(viewIdx+3)
        commonFeatures, origIndices = findCommonFeatures(mergedFeatures, activeImages)
        indices = np.array(origIndices)
        relevantPoints = worldPoints[indices]        
        commonPoints = [match[f'image{viewIdx+3}_points'] + (1,) for match in commonFeatures]
        pts1 = [match['image1_points'] + (1,) for match in commonFeatures]
        P1 = getProjectionMatrix(instrinsicParameters, np.eye(3), np.zeros((3,1)))
        # PnP Ransac
        inliers, R_new, C_new  = PnPRANSAC(commonPoints, relevantPoints, instrinsicParameters)
        P = getProjectionMatrix(instrinsicParameters, R_new, C_new)
        # Linear PnP error
        pnpError = []
        for idx in range(len(relevantPoints)):
            pnpError.append(reprojectionErrorPnP(P, commonPoints[idx], relevantPoints[idx]))

        print(f"Linear PnP Error: {np.mean(pnpError)}")
        visualizeReprojection(images[viewIdx+2], P, relevantPoints, commonPoints, f"Linear PnP Image {viewIdx+3}")
        
        R_new, C_new = NonLinearPnP(commonPoints, relevantPoints, R_new, C_new, instrinsicParameters)
        print("Camera Pose after Non Linear PnP:")
        print("R: ")
        print(R_new)
        print("C: ")
        print(C_new)
        P = getProjectionMatrix(instrinsicParameters, R_new, C_new)
        
        pnpError = []
        for idx in range(len(relevantPoints)):
            pnpError.append(reprojectionErrorPnP(P, commonPoints[idx], relevantPoints[idx]))
            
        print(f"NonLinear PnP Error: {np.mean(pnpError)}")
        visualizeReprojection(images[viewIdx+2], P, relevantPoints, commonPoints, f"Non Linear PnP Image {viewIdx+3}")
        
        uniqueFeatures, uniqueIndices = findUniqueFeatures(mergedFeatures, activeImages)
        pts1 = np.array([match['image1_points'] + (1,) for match in uniqueFeatures])
        pts2 = np.array([match[f'image2_points'] + (1,) for match in uniqueFeatures])
        
        points = getTriangulatedPoints(R1, C1, R_new, C_new, uniqueFeatures, instrinsicParameters)
        refinedNewPoints = nonLinearTriangulation(instrinsicParameters, R1, C1, R_new, C_new, points, pts1, pts2)
        
        for idx in range(refinedNewPoints.shape[0]):
            worldPoints[uniqueIndices[idx]] = refinedNewPoints[idx]
        
        R_All.append(R_new)
        C_All.append(C_new)         

        fig, ax = plt.subplots()
        plt.scatter(worldPoints[:,0], worldPoints[:,2], s=1, c='r', label="After PnP")
        variable = []
        for idx in range(viewIdx+3):
            variable.append((R_All[idx], C_All[idx], str(idx+1)))

        for rotation, position, label in variable:
            # Convert rotation matrix to Euler angles
            angles = Rotation.from_matrix(rotation).as_euler('XYZ')
            angles_deg = np.rad2deg(angles)
            # print(f"Camera {label} position: {position}, orientation: {angles_deg}")
            
            ax.plot(position[0], position[2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None', label=f'Camera {label}', color=marker_color[int(label)-1])
            
            # Annotate camera with label
            correction = -0.1
            ax.annotate(label, xy=(position[0] + correction, position[2] + correction))

        # Setting the plot axis limits
        plt.axis([-15, 15, -5, 25])

        # Adding labels and legend
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.legend()
        img_title = f"EstimatedCameraPosesAfterPnPforImage{viewIdx+3}"
        plt.savefig('IntermediateOutputImages/' + img_title + '.png')
        plt.close()
    
    print("="*50)
    print("Building Visibility Matrix and Performing Bundle Adjustment")
    print("="*50)

    number_of_cameras = viewsNum
    visibility_matrix = build_visibility_matrix(number_of_cameras, mergedFeatures)
    R_News, C_News, X = bundleAdjustment(mergedFeatures, np.array(worldPoints[:, :3]), visibility_matrix, np.array(R_All), np.array(C_All), number_of_cameras, instrinsicParameters)
    
    print("Final R :")
    print(R_News)
    print("Final C :")
    print(C_News)
    
    # plot the camera positions and orientations
    fig, ax = plt.subplots()
    plt.scatter(worldPoints[:,0], worldPoints[:,2], s=1, c='r', label="Before Bundle Adjustment")
    plt.scatter(X[:,0], X[:,2], s=1, c='b', label="After Bundle Adjustment")
    for i in range(viewsNum):
        # Convert rotation matrix to Euler angles
        angles = Rotation.from_matrix(R_News[i]).as_euler('XYZ')
        angles_deg = np.rad2deg(angles)        
        ax.plot(C_News[i][0], C_News[i][2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None', label=f'Camera {i+1}', color=marker_color[i])
        correction = -0.1
        ax.annotate(i+1, xy=(C_News[i][0] + correction, C_News[i][2] + correction))
    
    plt.axis([-20, 20, -10, 25])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.savefig('IntermediateOutputImages/BundleAdjustment.png')
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the images", default='./P2Data/')
    args = parser.parse_args()
    main(args)
