import cv2
import numpy as np
import math

# Auxiliary functions ------------------------------------------------------------------------------------------------------
# Print image
def show_image(title, image, scale_percent):
    imgResized = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)))
    cv2.imshow(title, imgResized)

# Draw circle
def draw_circle(image, coordinates):
    cv2.circle(image, coordinates, 30, (255,0,0), 20, cv2.LINE_AA)

# Main -----------------------------------------------------------------------------------------------------------------------
# Opening an image
imgOriginal = cv2.imread('./assets/images/i/IMG_0922.JPG')
imgGrey = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

# Bilateral Filter
imgBilateral = cv2.bilateralFilter(imgGrey, 11, 75, 75)

# Using a Canny Filter
imgWithCanny = cv2.Canny(imgBilateral, 60, 60, None, 3)

# Dilate
kernel = np.ones((5, 5), np.uint8)
imgDilate = cv2.dilate(imgWithCanny, kernel, iterations=7)

# Erode
imgErode = cv2.erode(imgDilate, kernel, iterations=7)

# Copy edges to the images that will display the results in BGR
cdst = imgOriginal.copy()

# Probabilistic Hough Line Transform
lines = cv2.HoughLinesP(imgErode, 1, np.pi / 180, 50, minLineLength=85, maxLineGap=50)

# Draw the lines
leftMostX, leftMostY = imgOriginal.shape[:2]
rightMostX, rightMostY = (0,0)
upLeftMostX, upLeftMostY = imgOriginal.shape[:2]
counter = 0
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        # Specify the area of interest
        if (l[1] > 2000 or l[1] < 900 or l[3] > 2000 or l[3] < 900):
            continue
        # Left most vertex
        if (l[0] < leftMostX):
            leftMostX = l[0]
            leftMostY = l[1]
        if (l[2] < leftMostX):
            leftMostX = l[2]
            leftMostY = l[3]
        # Right most vertex
        if (l[2] > rightMostX):
            rightMostX = l[2]
            rightMostY = l[1]
        if (l[0] > rightMostX):
            rightMostX = l[0]
            rightMostY = l[3]
        # Up Left most vertex
        if (l[0] < upLeftMostX and l[1] <= upLeftMostY):
            upLeftMostX = l[0]
            upLeftMostY = l[1]
        if (l[2] < upLeftMostX and l[3] <= upLeftMostY):
            upLeftMostX = l[2]
            upLeftMostY = l[3]
        # Draw line
        counter = counter + 1
        cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,255,0), 5, cv2.LINE_AA)

# Draw important vertices
print(counter)
draw_circle(cdst, (leftMostX, leftMostY))
draw_circle(cdst, (rightMostX, rightMostY))
draw_circle(cdst, (upLeftMostX, upLeftMostY))

# Print Original Image
scale_percent = 20
scale_percent2 = 10
#show_image('Img', imgOriginal, scale_percent2)

# Print Grey Image
#show_image('Img Grey', imgGrey, scale_percent2)

# Print Bilateral Image
#show_image('Img Bilateral', imgBilateral, scale_percent)

# Print Canny Image
#show_image('Img Canny', imgWithCanny, scale_percent)

# Print Dilate Image
#show_image('Img Dilate', imgDilate, scale_percent)

# Print Erode Image
show_image('Img Erode', imgErode, scale_percent)

# Print Final Image
show_image('Detected Lines (in green) - Standard Hough Line Transform', cdst, scale_percent)
cv2.waitKey(0)