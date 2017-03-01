import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
global count
global left_fit
global right_fit
left_fit = 0
right_fit = 0
count = 0
#imageio.plugins.ffmpeg.download()
#%matplotlib inline


def CalibrateCamera():
    calibrationImage = mpimg.imread("camera_cal/calibration1.jpg")
    # real undistorted corners. known object vorrdinates. (x,y,z)
    objpoints = []
    # these are the corners
    imgpoints = []
    objp = np.zeros((9 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)
    gray = cv2.cvtColor(calibrationImage, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 5), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(gray, (9, 5), corners, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def Undistort(image, mtx, dist ):
    return cv2.undistort(image, mtx, dist, None, mtx)

def GetWarpImageData(img_size):
    src = np.float32([[585, 460],
                      [203, 720],
                      [1127, 720],
                      [695, 460]
                     ])
    dst = np.float32([[320, 0], [320, 720], [960, 720],[960, 0]])
    m = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return m, Minv

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absSobel = np.absolute(sobel)
    scaledSobel = np.uint8(255*absSobel/np.max(absSobel))
    sxBinary = np.zeros_like(scaledSobel)
    sxBinary[(scaledSobel >= thresh[0]) & (scaledSobel <= thresh[1])] = 1
    return sxBinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    absSobelX = np.absolute(sobelx)
    absSobelY = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradDir = np.arctan2(absSobelY, absSobelX)
    # 5) Create a binary mask where direction thresholds are met
    sxBinary = np.zeros_like(gradDir)
    sxBinary[(gradDir > thresh[0]) & (gradDir <= thresh[1])]=1
    # 6) Return this mask as your binary_output image
    return sxBinary
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    absMag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    #scaledSobel = np.uint8(255*absMag/np.max(absMag))
    scale_factor = np.max(absMag)/255
    absMag = (absMag/scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    #sxBinary = np.zeros_like(scaledSobel)
    #sxBinary[(scaledSobel >= mag_thresh[0]) & (scaledSobel <= mag_thresh[0])] = 1
    binary_output = np.zeros_like(absMag)
    binary_output[(absMag >= mag_thresh[0]) & (absMag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
# Pipeline for color and gradient work

def ProcessImage(img):

    ksize=31
    min_gradx=24
    max_gradx=188
    min_grady=20
    max_grady=255
    min_mag=6
    max_mag=170
    min_dir=0
    max_dir=1.0
    gradx=abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(min_gradx, max_gradx))
    #grady=abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(min_grady, max_grady))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(min_mag, max_mag))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(min_dir, max_dir))
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def FindLinesInIntialImage(img):
    binary_warped = img

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    """
    return leftx, rightx, lefty, righty, left_fit, right_fit, left_fitx, right_fitx

def ProcessSubsequantImages(binary_warped, left_fit, right_fit ):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return leftx, rightx, lefty, righty, left_fit, right_fit, left_fitx, right_fitx

def DetermineCurve(left_fit, right_fit, leftx, rightx, lefty, righty, ploty):
    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    center = (((left_fit[0]*720**2+left_fit[1]*720+left_fit[2]) +(right_fit[0]*720**2+right_fit[1]*720+right_fit[2]) ) /2 - 640)*xm_per_pix
    return left_curverad,right_curverad, center
    # Example values: 632.1 m    626.2 m
def DrawRegion(imageIn, warpedIn, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    # workingWarped =  cv2.warpPerspective(imageIn, m, (imageIn.shape[1], imageIn.shape[0]), flags=cv2.INTER_LINEAR)
    warp_zero = np.zeros_like(warpedIn).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (imageIn.shape[1], imageIn.shape[0]))
    # Combine the result with the original image
    return cv2.addWeighted(imageIn, 1, newwarp, 0.3, 0)
mtx, dist = CalibrateCamera()

# used to hold a collection so we can average it out
leftFitXCollection = []
rightFitXCollection = []


def RunPipeline(image):
    global count
    global left_fit
    global right_fit
    # print(count)
    img_size = (image.shape[1], image.shape[0])
    # undistort the image using the calibrated values
    undistortedImage = Undistort(image, mtx, dist)
    m, Minv = GetWarpImageData(img_size)
    # warp the image into an overhead view
    warped = cv2.warpPerspective(undistortedImage, m, img_size, flags=cv2.INTER_LINEAR)
    # work with the S channel
    hlsImage = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    S = hlsImage[:, :, 2]

    # perform the gradient and threshold work and get a binary image
    combined = ProcessImage(S)
    leftx, rightx, lefty, righty, left_fit, right_fit, left_fitx, right_fitx = FindLinesInIntialImage(combined)
    # if this is the first image through, then we are on the first image.
    """
    if count < 1:
        #print("firstImage")
        leftx, rightx, lefty, righty, left_fit, right_fit, left_fitx, right_fitx = FindLinesInIntialImage(combined)
    else:
        leftx, rightx, lefty, righty, left_fit, right_fit, left_fitx, right_fitx = ProcessSubsequantImages(combined, left_fit, right_fit)
        # this will smooth out the lines over the last 5 frames
        #
"""
    # left_fitx, right_fitx = SmoothLines(left_fitx, right_fitx)
    ploty = np.linspace(0, combined.shape[0] - 1, combined.shape[0])

    # Deterimne the radius of the curve
    left_curve, right_curve, center = DetermineCurve(left_fit, right_fit, leftx, rightx, lefty, righty, ploty)

    # Tell how far left or right of center the vehicle is
    if center < 0:
        center = "left of center: " + str(np.abs(center))
    else:
        center = "right of center: " + str(np.abs(center))

    # Draw the region on the image.
    finalImage = DrawRegion(image, combined, left_fitx, right_fitx, ploty, Minv)
    count += 1
    return finalImage
    # plt.imshow(finalImage, cmap='gray')
    # print(left_curve, right_curve, center, count)
image = mpimg.imread("trimmed_Moment.jpg")

result = RunPipeline(image)
output = 'test_out.mp4'
clip1 = VideoFileClip("trimmed.mp4")
clip = clip1.fl_image(RunPipeline)
"""
plt.imshow(result)
plt.interactive(False)
plt.show()
"""