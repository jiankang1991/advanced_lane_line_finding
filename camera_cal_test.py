
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

def camera_cal(cal_dir, nx, ny):
    """ camera calibration based on multiple chessboard images
    https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    input:
        cal_dir: input image directory 
        nx: number of corners in a row 
        ny: number of corners in a column

    return:
        mtx: camera matrix
        dist: distortion coefficients

    """
    img_paths = glob(cal_dir + '/calibration*.jpg')

    # print(img_paths)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    objp = np.zeros((nx * ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for path in img_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret == True:
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            objpoints.append(objp)
            imgpoints.append(corners)

            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return (ret, mtx, dist, rvecs, tvecs)








if __name__ == "__main__":
    
    img_dir = './camera_cal'

    cal_img = './camera_cal/calibration1.jpg'

    img = cv2.imread(cal_img)

    _, mtx, dist, _, _ = camera_cal(img_dir, nx=9, ny=6)

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # cv2.imshow('img', dst)
    # cv2.waitKey(0)
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.set_title('Distorted image')
    ax1.imshow(img[...,::-1])

    ax2.set_title('Undistorted image')
    ax2.imshow(dst[...,::-1])

    plt.tight_layout()
    plt.savefig('./output_images/cameral_calibration.png')
    plt.show()














