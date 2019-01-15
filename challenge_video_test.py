
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import linear_model
from scipy import stats
from collections import deque

from camera_cal_test import camera_cal
import gradients_colors_thresholding_test as grad_color_thres

from perspective_transform_test import perspective_transform



# Define a class to receive the characteristics of each line detection
class Line():
    """ line class for lane detection 
    wraped_shape: shape of input image
    N: number of frames to track
    ym_per_pix: how many meters one pixel represent along y direction
    """
    def __init__(self, wraped_shape, N, ym_per_pix, xm_per_pix):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_x_bottom = deque(maxlen=N)
        self.recent_x_top = deque(maxlen=N)
        self.best_x_bottom = None
        self.best_x_top = None
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.recent_fit = deque(maxlen=N)
        #polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = np.linspace(0, wraped_shape[0]-1, wraped_shape[0])  

        self.curv_y_eval = wraped_shape[0]
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix

    def update_recent_fit(self, fit):
        """ update the polynomial coefficients"""
        self.recent_fit.append(fit)

    def update_recent_x_bottom(self, x):
        """ update the bottom x coordinate """
        self.recent_x_bottom.append(x)

    def update_recent_x_top(self, x):
        """ update the top x coordinate """
        self.recent_x_top.append(x)
    
    def calc_best_x_bottom(self):
        """ average the bottom x coordinate """
        x_points = np.asarray(list(self.recent_x_bottom))
        # self.best_x_bottom = np.mean(x_points)
        self.best_x_bottom = stats.trim_mean(x_points, proportiontocut=0.3)
    
    def calc_best_x_top(self):
        """ average the top x coordinate """
        x_points = np.asarray(list(self.recent_x_top))
        # self.best_x_top = np.mean(x_points)
        self.best_x_top = stats.trim_mean(x_points, proportiontocut=0.3)

    def calc_best_fit(self):
        """ average the polynomial coefficients """
        fits = np.concatenate(tuple(self.recent_fit), axis=1)
        # self.best_fit = np.mean(fits,axis=1)
        self.best_fit = stats.trim_mean(fits, proportiontocut=0.3, axis=1)

    def calc_radius_of_curvature(self):
        """ calculate the curvature """
        # self.radius_of_curvature = (1 + (2 * self.best_fit[0] * (self.curv_y_eval*self.ym_per_pix) + self.best_fit[1])**2)**(3/2)/(2*abs(self.best_fit[0]))
        # self.radius_of_curvature = (1 + (2 * (self.best_fit[0]*self.xm_per_pix/self.ym_per_pix**2) * 
        # (self.curv_y_eval) + self.best_fit[1]*self.xm_per_pix/self.ym_per_pix)**2)**(3/2)/(2*abs(self.best_fit[0]*self.xm_per_pix/self.ym_per_pix**2))

        real_a = self.best_fit[0] * self.xm_per_pix/(self.ym_per_pix**2)
        real_b = self.best_fit[1] * self.xm_per_pix/self.ym_per_pix

        numerator = (1 + (2*real_a*(self.curv_y_eval*self.ym_per_pix)+real_b)**2)**1.5
        denominator = 2*abs(real_a)

        self.radius_of_curvature = numerator/denominator


    def calc_line_base_pos(self):
        """ update the current position of line """
        # self.line_base_pos = self.best_x_bottom
        self.line_base_pos = np.poly1d(self.best_fit)(self.curv_y_eval)

    def calc_allx(self):
        """ calculate the x according to the fitted polynomial  """
        self.allx = self.best_fit[0]*self.ally**2 + self.best_fit[1]*self.ally + self.best_fit[2]



def find_lane_pixels_scratch(binary_warped, part = 'left'):
    """ modification of find lane pixels based on sliding window method for video processing 
    Input: 
    binary_warped: binary warped image
    part: left or right lane
    """

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)

    if part == 'left':
        base = np.argmax(histogram[:midpoint])
    else:
        base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if len(nonzero) == 0:
        print('no active pixels found in the warped binary img')
        return None
    else:
        # Current positions to be updated later for each window in nwindows
        x_current = base
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_x_low = int(x_current - margin)  # Update this
            win_x_high = int(x_current + margin)  # Update this
            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_inds = ((nonzerox >= win_x_low) & (nonzerox < win_x_high) & 
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            # x_p = x_current
            if len(good_inds) > minpix:
                x_current = np.mean(nonzerox[good_inds])
        
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
        
        if lane_inds.size == 0:
            print('no active points found in the interest areas')
            return None
        else:
            # Extract left and right line pixel positions
            x = nonzerox[lane_inds]
            y = nonzeroy[lane_inds] 

            return (x, y)

def fit_polynomial(binary_warped, line, part = 'left'):
    """ fit polynomial method for video processing
    Input:
        binary_warped: warped image
        line: left lane or right lane object
        part: left lane or right lane
        """
    # Find our lane pixels first
    # leftx, lefty, rightx, righty, out_img = find_lane_pixels_scratch(binary_warped, part = 'left')
    
    points = find_lane_pixels_scratch(binary_warped, part = part)
    
    ### check whether the sliding window method find the lane
    if points is not None:
        x, y = points

        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        fit = np.polyfit(y, x, deg=2)
        x_bottom = np.poly1d(fit)(binary_warped.shape[0])
        x_top = np.poly1d(fit)(0) 

        line.detected = True
        line.update_recent_fit(fit[:,np.newaxis])

        line.update_recent_x_bottom(x_bottom)
        line.update_recent_x_top(x_top)
        
        line.calc_best_fit()
        line.calc_best_x_bottom()
        line.calc_best_x_top()
        line.calc_allx()
        line.calc_line_base_pos()
        line.calc_radius_of_curvature()

    else:
        line.detected = False
    
def search_around_poly(binary_warped, margin, line, part='left'):
    """ modified search around the exist polynomial fitted line for video processing
    Input:
        binary_warped: input binary warped image
        margin: designed margin area to search
        line: left or right lane line object
        part: left or right lane
        """
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    # margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    if len(nonzero) == 0:
        print('no active pixels found in the warped binary img')
        line.detected = False
        # return None
    else:
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        lane_inds = ((nonzerox >= (np.poly1d(line.best_fit)(nonzeroy) - margin)) & (nonzerox < (np.poly1d(line.best_fit)(nonzeroy) + margin)))

        # Again, extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        if len(lane_inds) == 0:
            print('previous polynomial fit cannot help find the current active points')
            print('create the polinomial from scratch again')
            line.detected = False
            # fit_polynomial(binary_warped, line, part = 'left')
            # if line.detected == False:
                # print('cannot find polinomial from scratch, use recorded parameter to draw')
                # return None
        else:
            fit = np.polyfit(y, x, deg=2)
            x_bottom = np.poly1d(fit)(binary_warped.shape[0])
            x_top = np.poly1d(fit)(0)

            line.detected = True
            line.update_recent_fit(fit[:,np.newaxis])

            line.update_recent_x_bottom(x_bottom)
            line.update_recent_x_top(x_top)
        
            line.calc_best_fit()
            line.calc_best_x_bottom()
            line.calc_best_x_top()
            line.calc_allx()
            line.calc_line_base_pos()
            line.calc_radius_of_curvature()

            # return 1

def drawing_lane_line_area(warped_size, left_lane, right_lane, image, Minv):
    """ modified function for video processing """
    # Create an image to draw the lines on
    warp_zero = np.zeros(warped_size).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.allx, left_lane.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.allx, right_lane.ally])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

def drawing_poly_warped_img(warped_img, left_lane, right_lane):

    out_img = np.dstack((warped_img, warped_img, warped_img))*255
    
    left_pts = np.array(list(zip(left_lane.allx, left_lane.ally)), np.int32)
    left_pts = left_pts.reshape((-1,1,2))

    right_pts = np.array(list(zip(right_lane.allx, right_lane.ally)), np.int32)
    right_pts = right_pts.reshape((-1,1,2))

    cv2.polylines(out_img,[left_pts],False,(0,255,0),5)
    cv2.polylines(out_img,[right_pts],False,(0,255,255),5)

    return out_img

if __name__ == "__main__":
    
    # video_pth = './challenge_video.mp4'
    video_pth = './project_video.mp4'
    # video_pth = './harder_challenge_video.mp4'
    
    

    video_capture = cv2.VideoCapture(video_pth)

    ret, frame = video_capture.read()
    # [h, w] = frame.shape[:2]
    # print(frame.shape)
    img_dir = './camera_cal'
    _, mtx, dist, _, _ = camera_cal(img_dir, nx=9, ny=6)

    #### challenge video
    # src = np.float32([[520, 482],[800, 482],
    #                   [1250, 720],[40, 720]])

    #### project video
    src = np.float32([[520, 450],[730, 450],
                      [1250, 720],[40, 720]])

    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    frame_num = 0
    margin = 60

    left_lane = Line(frame.shape, N=15, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)
    right_lane = Line(frame.shape, N=15, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)

    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output_images/project_output.mp4',fourcc, 20.0, (frame.shape[1],int(frame.shape[0]*1.5)))


    while True:

        print(frame_num)

        ret, frame = video_capture.read()

        if ret==True:
            img = cv2.undistort(frame, mtx, dist, None, mtx)

            grad_x_binary_map = grad_color_thres.abs_sobel_thresh(img, thresh=(5,100))
            s_binary_map, S_channel = grad_color_thres.hls_thres(img, thresh=(50,255))
            l_binary_map, L_channel = grad_color_thres.hls_thres_L(img, thresh=(160,255))
            dir_binary_map = grad_color_thres.dir_threshold(img, sobel_kernel=9, thresh=(0.5,1.3))

            color_binary = np.dstack(( dir_binary_map, grad_x_binary_map, s_binary_map)) * 255

            combined_binary = np.zeros_like(s_binary_map)

            combined_binary[((s_binary_map == 1) | (l_binary_map == 1)) & ((dir_binary_map==1) & (grad_x_binary_map==1))] = 1

            warped_img, _ = perspective_transform(src, dst, combined_binary, img.shape[:2], draw_line=False)

            # search around the margin area according to the fitted polynomial
            if left_lane.detected and (frame_num != 0):
                search_around_poly(warped_img, margin, left_lane, part='left')
            if right_lane.detected and (frame_num != 0):
                search_around_poly(warped_img, margin, right_lane, part='right')
            

            # detect from scratch if lane line is not detected in last frame or frame mumber is 0
            if not left_lane.detected or (frame_num == 0):
                fit_polynomial(warped_img, left_lane, part='left')
            
            if not right_lane.detected or (frame_num == 0):
                fit_polynomial(warped_img, right_lane, part='right')
            
            out_img = drawing_poly_warped_img(warped_img, left_lane, right_lane)
            birdview_img, _ = perspective_transform(src, dst, img, img.shape[:2], draw_line=False)

            combined_birdview_warped_bin_img = np.hstack((cv2.resize(birdview_img, (int(img.shape[1]/2), int(img.shape[0]/2))), 
                                                          cv2.resize(out_img, (int(img.shape[1]/2), int(img.shape[0]/2)))))

            result = drawing_lane_line_area(img.shape[:2], left_lane, right_lane, img, Minv)

            vehicle_offset = ((left_lane.line_base_pos + right_lane.line_base_pos)/2 - (img.shape[1]/2)) * xm_per_pix
            
            if vehicle_offset >= 0:
                text = 'vehicle is {0:.2f}(m) right of center'.format(abs(vehicle_offset))
            else:
                text = 'vehicle is {0:.2f}(m) left of center'.format(abs(vehicle_offset))
            
            cv2.putText(result, 'Radius of curvature (left): {}(m)'.format(int(left_lane.radius_of_curvature)),
                        (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.putText(result, 'Radius of curvature (right): {}(m)'.format(int(right_lane.radius_of_curvature)),
                        (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.putText(result, text,
                        (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            final_combined = np.vstack((combined_birdview_warped_bin_img, result))

            cv2.imshow("video", final_combined)

            # write the flipped frame
            out.write(final_combined)

            frame_num += 1

            k = cv2.waitKey(1) & 0xff

            if k == ord('q') or k == 27:
                break
        else:
            break

# Release everything if job is finished
video_capture.release()
out.release()
cv2.destroyAllWindows()








