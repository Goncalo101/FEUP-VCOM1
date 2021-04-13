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
leftMostLeftShadowY, leftMostLeftShadowX = imgOriginal.shape[:2]
rightMostLeftShadowX, rightMostLeftShadowY = (0,0)

leftMostRightShadowY, leftMostRightShadowX = imgOriginal.shape[:2]
rightMostRightShadowX, rightMostRightShadowY = (0,0)

upLeftMostY, upLeftMostX = imgOriginal.shape[:2]
upRightMostX, upRightMostY = (0,0)

counter = 0

if lines is not None:
    #draw lines on the image 
    for i in range(0, len(lines)):
        l = lines[i][0]
        # Specify the area of interest
        if (l[1] > 2000 or l[1] < 900 or l[3] > 2000 or l[3] < 900):
            continue
        # Draw line
        counter = counter + 1
        cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,255,0), 5, cv2.LINE_AA)
    #get points on the upmost line left shadow
    for i in range(0, len(lines)):
        l = lines[i][0]
        # Specify the area of interest
        if (l[1] > 1900 or l[1] < 1600 or l[3] > 1900 or l[3] < 1600 and l[0] > 1200 or l[2] > 1200 ):
            continue
         # Left most vertex
        if (l[0] < leftMostLeftShadowX):
            leftMostLeftShadowX = l[0]
            leftMostLeftShadowY = l[1]
        if (l[2] < leftMostLeftShadowX):
            leftMostLeftShadowX = l[2]
            leftMostLeftShadowY = l[3]
        # Right most vertex
        if (l[2] > rightMostLeftShadowX):
            rightMostLeftShadowX = l[2]
            rightMostLeftShadowY = l[1]
        if (l[0] > rightMostLeftShadowX):
            rightMostLeftShadowX = l[0]
            rightMostLeftShadowY = l[3]

    #get points on the upmostline right shadow
    for i in range(0, len(lines)):
        l = lines[i][0]
        # Specify the area of interest
        if (l[1] > 1900 or l[1] < 1600 or l[3] > 1900 or l[3] < 1600 and l[0] < 1200 or l[2] < 1200 ):
            continue
          # Left most vertex
        if (l[0] < leftMostRightShadowX):
            leftMostRightShadowX = l[0]
            leftMostRightShadowY = l[1]
        if (l[2] < leftMostRightShadowX):
            leftMostRightShadowX = l[2]
            leftMostRightShadowY = l[3]
        # Right most vertex
        if (l[2] > rightMostRightShadowX):
            rightMostRightShadowX = l[2]
            rightMostRightShadowY = l[1]
        if (l[0] > rightMostRightShadowX):
            rightMostRightShadowX = l[0]
            rightMostRightShadowY = l[3]
    #get points on the upmostline top shadow 
    for i in range(0, len(lines)):
        l = lines[i][0]
        # Specify the area of interest
        if (l[1] > 1300 or l[1] < 900 or l[3] > 1300 or l[3] < 900):
            continue
         # Up Left most vertex
        if (l[0] < upLeftMostX and l[1] <= upLeftMostY):
            upLeftMostX = l[0]
            upLeftMostY = l[1]
        if (l[2] < upLeftMostX and l[3] <= upLeftMostY):
            upLeftMostX = l[2]
            upLeftMostY = l[3]
        # Right most vertex
        if (l[2] > upRightMostX):
            upRightMostX = l[2]
            upRightMostY = l[1]
        if (l[0] > upRightMostX):
            upRightMostX = l[0]
            upRightMostY = l[3]
    
# Draw important vertices
print(counter)

draw_circle(cdst, (leftMostLeftShadowX, leftMostLeftShadowY))
draw_circle(cdst, (rightMostLeftShadowX, rightMostLeftShadowY))

draw_circle(cdst, (leftMostRightShadowX, leftMostRightShadowY))
draw_circle(cdst, (rightMostRightShadowX, rightMostRightShadowY))

draw_circle(cdst, (upLeftMostX, upLeftMostY))
draw_circle(cdst, (upRightMostX, upRightMostY))


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