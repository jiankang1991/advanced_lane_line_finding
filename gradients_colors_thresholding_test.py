
import cv2
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from camera_cal_test import camera_cal


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """ thresholding method based on gradients of image in x or y direction
    input:
        img: BGR img read by cv2
        orient: 'x' or 'y'
        sobel_kernel: size of sobel kernel
        thresh: low and upper thresholds for absolute values of derivatives
    
    output:
        binary_output: binary map of preserved values in-between the thresholds
    """


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        img_gray_x_deriv = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
        img_gray_x_deriv = np.uint8(255*np.absolute(img_gray_x_deriv)/np.max(np.absolute(img_gray_x_deriv)))
        binary_output = np.zeros_like(img_gray_x_deriv) # Remove this line
        binary_output[(img_gray_x_deriv>thresh[0])&(img_gray_x_deriv<thresh[1])] = 1
    else:
        img_gray_y_deriv = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)
        img_gray_y_deriv = np.uint8(255*np.absolute(img_gray_y_deriv)/np.max(np.absolute(img_gray_y_deriv)))
        binary_output = np.zeros_like(img_gray_y_deriv) # Remove this line
        binary_output[(img_gray_y_deriv>thresh[0])&(img_gray_y_deriv<thresh[1])] = 1

    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    """ thresholding method absed on magnitude of derivatives
    Input:
        image: BGR image read by cv2
        sobel_kernel: size of sobel kernel
        mag_thresh: low and upper thresholds for magnitudes of derivatives
    Output:
        binary_output: binary map of preserved values in-between the thresholds
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_gray_x_deriv = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    img_gray_y_deriv = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    img_gray_xy_deriv = np.sqrt(img_gray_x_deriv**2 + img_gray_y_deriv**2)
    
    img_gray_xy_deriv = np.uint8(255*np.absolute(img_gray_xy_deriv)/np.max(np.absolute(img_gray_xy_deriv)))
    
    binary_output = np.zeros_like(img_gray_xy_deriv)
    
    
    binary_output[(img_gray_xy_deriv>mag_thresh[0])&(img_gray_xy_deriv<mag_thresh[1])] = 1
    
    return binary_output
    
def hls_thres(img, thresh=(0, 255)):
    """ thresholding based on S channel value of HLS image 
    Input:
        img: BGR image read by cv2
        thresh: low and upper thresholds for values of S channel
    Output:
        binary_output: binary map of preserved values in-between the thresholds
    """

    
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S_channel = img_HLS[...,-1]
    binary_output = np.zeros_like(S_channel)
    binary_output[(S_channel>thresh[0])&(S_channel<=thresh[1])] = 1
    
    return binary_output, S_channel

def hls_thres_L(img, thresh=(0, 255)):
    """ thresholding based on L channel value of HLS image 
    Input:
        img: BGR image read by cv2
        thresh: low and upper thresholds for values of S channel
    Output:
        binary_output: binary map of preserved values in-between the thresholds
    """

    
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L_channel = img_HLS[...,-2]
    binary_output = np.zeros_like(L_channel)
    binary_output[(L_channel>thresh[0])&(L_channel<=thresh[1])] = 1
    
    return binary_output, L_channel

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ thresholding method based on direction of derivatives
    Input: 
        image: BGR image read by cv2
        sobel_kernel: size of sobel kernel
        thresh: low and upper thresholds for direction of derivatives
    Output:
        binary_output: binary map of preserved values in-between the thresholds
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_gray_x_deriv = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    img_gray_y_deriv = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    grad_dir = np.arctan2(img_gray_y_deriv, img_gray_x_deriv)
    
    binary_output = np.zeros_like(grad_dir)
    
    binary_output[(grad_dir>thresh[0])&(grad_dir<thresh[1])] = 1
    

    return binary_output


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, straight_line = False, draw_line = True):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    if draw_line:
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, lines, straight_line=straight_line)
        return line_img
    else:
        return lines

def draw_lines(img, lines, color=[0, 0, 255], thickness=10, straight_line = False):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    if not straight_line:

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:

        left_points_x = []
        left_points_y = []
        
        right_points_x = []
        right_points_y = []
        
        min_y = 1000

        for line in lines:
        
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1)
                
                if slope < 0:

                    left_points_x += [x1,x2]
                    left_points_y += [y1,y2]
                else:

                    right_points_x += [x1,x2]
                    right_points_y += [y1,y2]
                    
                
                if min(y1,y2) < min_y:
                    min_y = min(y1,y2)
        
        ransac_left = linear_model.RANSACRegressor()
        ransac_right = linear_model.RANSACRegressor()
        
        ransac_left.fit(np.asarray(left_points_y)[:,np.newaxis], np.asarray(left_points_x)[:,np.newaxis])
        ransac_right.fit(np.asarray(right_points_y)[:,np.newaxis], np.asarray(right_points_x)[:,np.newaxis])
        
        predicted_points_left = ransac_left.predict(np.asarray([img.shape[0], min_y])[:,np.newaxis])
        predicted_points_right = ransac_right.predict(np.asarray([img.shape[0], min_y])[:,np.newaxis])
        
        cv2.line(img, (int(predicted_points_left[0]), img.shape[0]), (int(predicted_points_left[1]), min_y), color, thickness)
        cv2.line(img, (int(predicted_points_right[0]), img.shape[0]), (int(predicted_points_right[1]), min_y), color, thickness)


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


if __name__ == "__main__":
    
    img_dir = './camera_cal'
    _, mtx, dist, _, _ = camera_cal(img_dir, nx=9, ny=6)

    # img_path = './test_images/test2.jpg'
    # img_path = './test_images/straight_lines2.jpg'

    rho = 2
    theta = np.pi/180
    threshold = 100
    min_line_len = 50
    max_line_gap = 100

    plt.figure(figsize = (20,7))
    gs1 = gridspec.GridSpec(2, 4)
    gs1.update(left=0.02, right=0.93, top=0.92, bottom=0.03, hspace=0., wspace=0.05) # set the spacing between axes. 


    for idx, img_nm in enumerate(os.listdir('./test_images')):

        img_path = os.path.join('./test_images', img_nm)


        img = cv2.imread(img_path)

        #### image undistortion
        
        

        img = cv2.undistort(img, mtx, dist, None, mtx)

        # cv2.imshow('img', dst)
        # cv2.waitKey(0)

        #### combine gradients and color thresholding

        grad_x_binary_map = abs_sobel_thresh(img, thresh=(20,100))
        s_binary_map, _ = hls_thres(img, thresh=(170,255))

        dir_binary_map = dir_threshold(img, sobel_kernel=9, thresh=(0.5,np.pi/2))

        color_binary = np.dstack(( dir_binary_map, grad_x_binary_map, s_binary_map)) * 255

        combined_binary = np.zeros_like(s_binary_map)
        combined_binary[(s_binary_map == 1) | (grad_x_binary_map == 1) & (dir_binary_map == 1)] = 1

        # kernel = np.ones((1,1),np.uint8)

        # opening_combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel)

        # Plotting thresholded images
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.set_title('Stacked thresholds')
        # ax1.imshow(color_binary)

        # ax2.set_title('Combined S channel and gradient thresholds')
        # ax2.imshow(combined_binary, cmap='gray')

        # ax3.set_title('denoising')
        # ax3.imshow(opening_combined_binary, cmap='gray')
        # plt.show()


        

        
        vertices = np.asarray([[(150,img.shape[0]),(600,450),(730,450),(img.shape[1],img.shape[0])]])

        masked_img = region_of_interest(combined_binary, vertices)

        line_img = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap, straight_line=True)

        added_line_img = weighted_img(line_img, img)

        ax = plt.subplot(gs1[idx])
        ax.set_axis_off()
        ax.imshow(added_line_img[...,::-1])
        # ax.imshow(combined_binary, cmap='gray')
        ax.set_title(img_nm.split('.')[0])
        # plt.tight_layout()
    
    plt.savefig('./output_images/src_points_test_image.png')
    plt.show()



