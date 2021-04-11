import cv2
import numpy as np
import math

# Auxiliary functions
def show_image(title, image, scale_percent):
    imgResized = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)))
    cv2.imshow(title, imgResized)

# Opening an image
imgOriginal = cv2.imread('./assets/images/b/IMG_0869.jpg')
imgGrey = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

#------------------TESTING ZONE-------------------------------------

# Gaussian Blur
img = cv2.GaussianBlur(imgGrey,(7,7),0)

#-------------------------------------------------------

# Using a Canny Filter
imgWithCanny = cv2.Canny(imgGrey, 60, 100, None, 3)

# Copy edges to the images that will display the results in BGR
cdst = imgOriginal.copy()

# Standard Hough Line Transform
lines = cv2.HoughLinesP(imgWithCanny, 1, np.pi / 180, 50, maxLineGap=50)

# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)

# Print Original Image
scale_percent = 20
#show_image('Img', imgOriginal, scale_percent)

# Print Grey Image
show_image('Img Grey', imgGrey, scale_percent)

# Print Gaussian Image
show_image('Img Gaussian', img, scale_percent)

# Print Final Image
#show_image('Img Canny', imgWithCanny, scale_percent)
show_image('Detected Lines (in red) - Standard Hough Line Transform', cdst, scale_percent)
cv2.waitKey(0)