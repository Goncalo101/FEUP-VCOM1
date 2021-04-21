import numpy as np
import cv2
import glob

# Camera calibration using chess board 

# Intrinsic Calibration ------------------------------------------------------
# termination criteria for cornerSubPix
# terminates if accuracy (0.001) or maximum number of iterations (30) is reached
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)*2.44 # convert scale to centimeters

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('assets/internal/*')
scale_percent = 25
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine corner locations with a window size of 11*2+1=23 and no zerozone value
        # zerozone is the size of the region in the middle of the search zone over which 
        # the search algorithm is not run
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        imgResized = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)))
        cv2.imshow('img', imgResized)
        cv2.waitKey(1)

cv2.destroyAllWindows()


# Calibrate the camera taking into account the object and image points (chessboard corners) calculated above
# mtx - 3x3 floating point camera intrinsic matrix
# dist - distortion coefficients (k1, k2, p1, p2, k3)
# rvecs - rotation vectors estimated for each pattern view
# tvecs - translation vectors for each pattern view
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Total error calculation (estimation of how exact the found parameters are)
# The lower the better
mean_error = 0
for i in range(len(objpoints)):
    # Transform object point into image point
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # Norm between the transformed value and the value found by the corner finding algorithm
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

# Arithmetical mean of the errors calculated for all the calibration images
print("total error: ", mean_error/len(objpoints))


#Undisort image -------------------------------------------------------------

#img = cv2.imread('./assets/images/i/IMG_0922.JPG')
#h,  w = img.shape[:2]

# Return the new camera intrinsic matrix based on the free scaling parameter (1)
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# Method 1 - shortest path (use the ROI to crop the result)
#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('./assets/images/i/calibresult.png',dst)

# Method 2 - curved path (find mapping function from distorted image to undistorted image and remap)
# undistort
# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)

# they should be preserved through the program execution, for external calibration and measures
# internal values never change, are related to the camera hardware

# Extrinsic calibration ---------------------------------------------------------------

# Find the rotation and translation vectors. -> with solvePnPRansac using values from before
# ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
# but corners2 should contain the information In simple words, we find the points on image plane corresponding to each of (3,0,0),(0,3,0),(0,0,3) in 3D space.

#funtion to draw the axis on the chessboard
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    # Draw the axis (X axis in blue, Y in green and Z in red)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# termination criteria for cornerSubPix
# terminates if accuracy (0.001) or maximum number of iterations (30) is reached
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)*2.44 # convert scale to centimeters

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# vectors for the X, Y and Z axis in this order
# Z is negative because it points to the camera 
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for fname in glob.glob('assets/images/i/IMG_0927.JPG'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        
        objpoints.append(objp)
        imgpoints.append(corners2)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        # Find the rotation and translation vectors.
        # Finds an object pose from 3D-2D point correspondences using the RANSAC scheme. 
        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
       
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # draw the axis and show image
        img = draw(img,corners2,imgpts)
        mean_error2 = 0
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite(fname[:6]+'.png', img)
cv2.destroyAllWindows()


# Reprojection Error -------------------------------------------------------------
mean_error = 0
for i in range(len(objpoints)):
    # Transform object point into image point
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs, tvecs, mtx, dist)

    # Norm between the transformed value and the value found by the corner finding algorithm
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

# Arithmetical mean of the errors calculated for all the calibration images
print("reprojection error: ", mean_error/len(objp))


# Projection Matrix------------------------------------------------------------- 

rotation_mat = np.zeros(shape=(3, 3))
R = cv2.Rodrigues(rvecs, rotation_mat)[0] # Use Rodrigues Transform to calculate the rotation matrix
RT = np.column_stack((R, tvecs))

# Calculate Projection Matrix
P = np.matmul(mtx, RT)

# Save the Projection Matrix in a file to be used for the lineDetection script
np.savez("matrix.npz", P)

print(P)