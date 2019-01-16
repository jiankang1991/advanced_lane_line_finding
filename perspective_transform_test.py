
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import linear_model

from camera_cal_test import camera_cal
import gradients_colors_thresholding_test as grad_color_thres




def src_points_perspective(img, lines, top_y):
    """source points determination based on detected hough lines
    Input:
        img: input image
        lines: lines created by hough transformation
        top_y: y coordinate of top point
    """

    
    left_points_x = []
    left_points_y = []
    
    right_points_x = []
    right_points_y = []
    
    # min_y = 1000

    for line in lines:
    
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            
            if slope < 0:

                left_points_x += [x1,x2]
                left_points_y += [y1,y2]
            else:

                right_points_x += [x1,x2]
                right_points_y += [y1,y2]
                
            
            # if min(y1,y2) < min_y:
            #     min_y = min(y1,y2)
    
    ransac_left = linear_model.RANSACRegressor()
    ransac_right = linear_model.RANSACRegressor()
    
    ransac_left.fit(np.asarray(left_points_y)[:,np.newaxis], np.asarray(left_points_x)[:,np.newaxis])
    ransac_right.fit(np.asarray(right_points_y)[:,np.newaxis], np.asarray(right_points_x)[:,np.newaxis])
    
    predicted_points_left = ransac_left.predict(np.asarray([img.shape[0], top_y])[:,np.newaxis])
    predicted_points_right = ransac_right.predict(np.asarray([img.shape[0], top_y])[:,np.newaxis])
    
    # cv2.line(img, (int(predicted_points_left[0]), img.shape[0]), (int(predicted_points_left[1]), min_y), color, thickness)
    # cv2.line(img, (int(predicted_points_right[0]), img.shape[0]), (int(predicted_points_right[1]), min_y), color, thickness)

    return {'left_bottom': np.float32([int(predicted_points_left[0]), img.shape[0]]),
            'left_top': np.float32([int(predicted_points_left[1]), top_y]),
            'right_bottom': np.float32([int(predicted_points_right[0]), img.shape[0]]),
            'right_top': np.float32([int(predicted_points_right[1]), top_y])}

def perspective_transform(src, dst, img, gray_shape, draw_line=False):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, gray_shape[::-1])
    
    if draw_line:
        cv2.line(warped, tuple(dst[0].astype(int)), tuple(dst[1].astype(int)), color=[0,0,255], thickness=5)
        cv2.line(warped, tuple(dst[2].astype(int)), tuple(dst[3].astype(int)), color=[0,0,255], thickness=5)
        
    return warped, M

if __name__ == "__main__":
    
    img_dir = './camera_cal'
    _, mtx, dist, _, _ = camera_cal(img_dir, nx=9, ny=6)

    rho = 2
    theta = np.pi/180
    threshold = 100
    min_line_len = 50
    max_line_gap = 100

    offset = 300
    top_y = 450

    plt.figure(figsize = (20,7))
    gs1 = gridspec.GridSpec(2, 4)
    gs1.update(left=0.02, right=0.93, top=0.92, bottom=0.03, hspace=0., wspace=0.05) # set the spacing between axes. 

    for idx, img_nm in enumerate(os.listdir('./test_images')):
        img_path = os.path.join('./test_images', img_nm)

        img = cv2.imread(img_path)

        img = cv2.undistort(img, mtx, dist, None, mtx)

        grad_x_binary_map = grad_color_thres.abs_sobel_thresh(img, thresh=(20,100))
        s_binary_map, _ = grad_color_thres.hls_thres(img, thresh=(170,255))

        dir_binary_map = grad_color_thres.dir_threshold(img, sobel_kernel=9, thresh=(0.5,np.pi/2))

        color_binary = np.dstack(( dir_binary_map, grad_x_binary_map, s_binary_map)) * 255

        combined_binary = np.zeros_like(s_binary_map)
        combined_binary[(s_binary_map == 1) | (grad_x_binary_map == 1) & (dir_binary_map == 1)] = 1

        vertices = np.asarray([[(150,img.shape[0]),(600,450),(750,450),(img.shape[1],img.shape[0])]])

        masked_img = grad_color_thres.region_of_interest(combined_binary, vertices)

        hough_lines = grad_color_thres.hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap, straight_line=True, draw_line=False)

        src_points = src_points_perspective(img, hough_lines, top_y)

        dst_points = np.float32([[offset, img.shape[0]], [offset, 0], [img.shape[1]-offset, 0], [img.shape[1]-offset, img.shape[0]]])

        warped_img, _ = perspective_transform(np.float32([src_points['left_bottom'], 
                                                       src_points['left_top'],
                                                       src_points['right_top'],
                                                       src_points['right_bottom']]), 
                                           dst_points, img, img.shape[:2], draw_line=True)

        ax = plt.subplot(gs1[idx])
        ax.set_axis_off()
        ax.imshow(warped_img[...,::-1], cmap='gray')
        ax.set_title(img_nm.split('.')[0])
    plt.savefig('./output_images/perspective_test_image.png')
    plt.show()
