import numpy as np
import cv2
import glob

# Camera calibration using chess board 

# Intrinsic Calibration ------------------------------------------------------
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)*2.44

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

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        imgResized = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)))
        cv2.imshow('img', imgResized)
        cv2.waitKey(1)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Values printed just to see them
#print('ret: ')
#print(ret)
#print('mtx: ')
#print(mtx)
#print('dist: ')
#print(dist)
#print('rvecs: ')
#print(rvecs)
#print('tvecs: ')
#print(tvecs)

# Total error calculation -------------------------------------------------------------
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))

#Undisort image -------------------------------------------------------------

#img = cv2.imread('./assets/images/i/IMG_0922.JPG')
#h,  w = img.shape[:2]
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# Method 1 - shortest path
# undistort
#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('./assets/images/i/calibresult.png',dst)

# Method 2 - curved path
# undistort
#mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibresult.png',dst)

# they should be preserved through the program execution, for external calibration and measures
# internal values never change, are related to the camera hardware

# Extrinsic calibration ---------------------------------------------------------------

# Find the rotation and translation vectors. -> with solvePnPRansac using values from before
#ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
# but corners2 should contain the information In simple words, we find the points on image plane corresponding to each of (3,0,0),(0,3,0),(0,0,3) in 3D space.

#funtion to draw the axis on the chessboard
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5) # B -> X
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5) # G -> Y
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5) # R -> Z
    return img

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)*2.44

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

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
       
        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
       
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        mean_error2 = 0
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite(fname[:6]+'.png', img)
cv2.destroyAllWindows()


#Reprojection Error -------------------------------------------------------------
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs, tvecs, mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("reprojection error: ", mean_error/len(objp))


#Projection Matrix------------------------------------------------------------- 

rotation_mat = np.zeros(shape=(3, 3))
R = cv2.Rodrigues(rvecs, rotation_mat)[0]
RT = np.column_stack((R, tvecs))
#Projectioin Matrix
P = np.matmul(mtx,RT)

np.savez("matrix.npz", P)

print(P)