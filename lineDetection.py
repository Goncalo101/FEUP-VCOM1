import cv2
import numpy as np
import math

# Opening an image
img = cv2.imread('./assets/images/b/IMG_0869.jpg', cv2.COLOR_BGR2GRAY)

# Using a Canny Filter
imgWithCanny = cv2.Canny(img, 60, 100, None, 3)

# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(imgWithCanny, cv2.COLOR_GRAY2BGR)

# Standard Hough Line Transform
lines = cv2.HoughLinesP(imgWithCanny, 1, np.pi / 180, 50, maxLineGap=50)

# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

# Print Original Grey Image
scale_percent = 50
imgResized = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)))
cv2.imshow('Img', imgResized)

# Print Final Image
imgResizedCanny = cv2.resize(imgWithCanny, (int(imgWithCanny.shape[1] * scale_percent / 100), int(imgWithCanny.shape[0] * scale_percent / 100)))
imgResizedCannyLines = cv2.resize(cdst, (int(cdst.shape[1] * scale_percent / 100), int(cdst.shape[0] * scale_percent / 100)))
cv2.imshow("Img Canny", imgResizedCanny)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", imgResizedCannyLines)
cv2.waitKey(0)