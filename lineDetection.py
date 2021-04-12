import cv2
import numpy as np
import math

# Auxiliary functions
def show_image(title, image, scale_percent):
    imgResized = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)))
    cv2.imshow(title, imgResized)

# Opening an image
imgOriginal = cv2.imread('./assets/images/g/test.jpg')
imgGrey = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

#------------------TESTING ZONE-------------------------------------

# Gaussian Blur
#imgGaussian = cv2.GaussianBlur(imgGrey,(45,45),0)

# Bilateral Filter
imgBilateral = cv2.bilateralFilter(imgGrey,9,75,75)

# Using a Canny Filter
imgWithCanny = cv2.Canny(imgBilateral, 60, 60, None, 3)

# Dilate
kernel = np.ones((5, 5), np.uint8)
imgDilate = cv2.dilate(imgWithCanny, kernel, iterations=3)

# Erode
imgErode = cv2.erode(imgDilate, kernel, iterations=3)

#-------------------------------------------------------

# Copy edges to the images that will display the results in BGR
cdst = imgOriginal.copy()

# Standard Hough Line Transform
lines = cv2.HoughLinesP(imgErode, 1, np.pi / 180, 50, maxLineGap=50)

# Draw the lines
counter = 0
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        #if (l[1] > 1800 or l[1] < 900):
            #continue
        if (l[1] > 280 or l[1] < 50 or l[3] > 280 or l[3] < 50):
            continue
        counter = counter + 1
        cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,255,0), 5, cv2.LINE_AA)
print(counter)

# Print Original Image
scale_percent = 50
#show_image('Img', imgOriginal, scale_percent)

# Print Grey Image
#show_image('Img Grey', imgGrey, scale_percent)

# Print Gaussian Image
#show_image('Img Gaussian', imgGaussian, scale_percent)

# Print Bilateral Image
#show_image('Img Bilateral', imgBilateral, scale_percent)

# Print Canny Image
show_image('Img Canny', imgWithCanny, scale_percent)

# Print Erode Image
show_image('Img Erode', imgErode, scale_percent)

# Print Final Image
show_image('Detected Lines (in green) - Standard Hough Line Transform', cdst, scale_percent)
cv2.waitKey(0)