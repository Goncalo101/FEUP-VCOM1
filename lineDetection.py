import cv2
import numpy as np
import math

# Auxiliary functions ------------------------------------------------------------------------------------------------------
# Print image
def show_image(title, image, scale_percent):
    imgResized = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)))
    cv2.imshow(title, imgResized)

# Draw circle
def draw_vertex(image, coordinates):
    cv2.circle(image, coordinates, 30, (255,0,0), 20, cv2.LINE_AA)
    x = coordinates[0] + 100
    y = coordinates[1]
    cv2.putText(image, "(" + str(coordinates) + ")", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 3)

# Get line limits
def get_line_limits_x(lines, minX, minY, maxX, maxY):
    leftMostX = maxX
    leftMostY = maxY
    rightMostX = minX
    rightMostY = minY
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            # Specify the area of interest
            if (l[1] >= maxY or l[1] <= minY or l[3] >= maxY or l[3] <= minY or l[0] >= maxX or l[0] <= minX or l[2] >= maxX or l[2] <= minX):
                continue
            # Left most vertex
            if (l[0] <= leftMostX):
                leftMostX = l[0]
                leftMostY = l[1]
            if (l[2] <= leftMostX):
                leftMostX = l[2]
                leftMostY = l[3]
            # Right most vertex
            if (l[2] >= rightMostX):
                rightMostX = l[2]
                rightMostY = l[3]
            if (l[0] >= rightMostX):
                rightMostX = l[0]
                rightMostY = l[1]
    return (leftMostX, leftMostY, rightMostX, rightMostY)

def get_bigger(n1, n2):
    if (n1 > n2):
        return n1
    else: return n2

def get_smaller(n1, n2):
    if (n1 < n2):
        return n1
    else: return n2

def line_shadow_plane():
    # Draw the lines
    maxY = 2000
    minY = 900
    maxX = imgOriginal.shape[1]
    minX = 0
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            # Specify the area of interest
            if (l[1] > maxY or l[1] < minY or l[3] > maxY or l[3] < minY):
                continue
            # Draw line
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,255,0), 5, cv2.LINE_AA)

    # Get points #1 and #2
    (point1X, point1Y, point2X, point2Y) = get_line_limits_x(lines, minX, minY, maxX, int(round(maxY / 1.2)))
    # Get points #3 and #4
    (point3X, point3Y, point4X, point4Y) = get_line_limits_x(lines, minX, get_bigger(point1Y, point2Y), get_smaller(point1X, point2X), int(round(maxY / 1.05)))
    # Get points #5 and #6
    (point5X, point5Y, point6X, point6Y) = get_line_limits_x(lines, get_bigger(point1X, point2X), get_bigger(point1Y, point2Y), maxX, maxY)
    # Get points #7 and #8
    (point7X, point7Y, point8X, point8Y) = get_line_limits_x(lines, minX, minY, maxX, 1300)
    # Get points #9 and #10
    (point9X, point9Y, point10X, point10Y) = get_line_limits_x(lines, get_bigger(point1X, point2X), get_bigger(point5Y, point6Y), maxX, maxY)
    # Get points #11 and #12
    (point11X, point11Y, point12X, point12Y) = get_line_limits_x(lines, minX, 1900, get_smaller(point1X, point2X), maxY)

    # Draw important vertices
    draw_vertex(cdst, (point1X, point1Y))
    draw_vertex(cdst, (point2X, point2Y))
    draw_vertex(cdst, (point3X, point3Y))
    draw_vertex(cdst, (point4X, point4Y))
    draw_vertex(cdst, (point5X, point5Y))
    draw_vertex(cdst, (point6X, point6Y))
    draw_vertex(cdst, (point7X, point7Y))
    draw_vertex(cdst, (point8X, point8Y))
    draw_vertex(cdst, (point9X, point9Y))
    draw_vertex(cdst, (point10X, point10Y))
    draw_vertex(cdst, (point11X, point11Y))
    draw_vertex(cdst, (point12X, point12Y))

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

# Shadow
line_shadow_plane()

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
#show_image('Img Erode', imgErode, scale_percent)

# Print Final Image
show_image('Detected Lines (in green) - Standard Hough Line Transform', cdst, scale_percent)
cv2.waitKey(0)