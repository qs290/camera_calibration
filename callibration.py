import numpy as np
import cv2 as cv
import imagingcontrol4 as ic4
import time
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('E:\\IUT\\2e_annee\\4_semestre\\robotique_vision\\image_cam_81\\*.png')
print(f"Found {len(images)} images for calibration.")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(f"Chessboard corners found in {fname}")
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("\n=== MATRICE CAMERA ===")
print(mtx)

print("\n=== DISTORSION ===")
print(dist)

cv.destroyAllWindows()

# -----  Indistorsion de l'image -----
img = cv.imread('E:\\IUT\\2e_annee\\4_semestre\\robotique_vision\\image_cam_81\\makima_0.png')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# recadrer l'image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('E:\\IUT\\2e_annee\\4_semestre\\robotique_vision\\calibresult2.png', dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2SQR) / len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(np.sqrt(mean_error/len(objpoints))) )