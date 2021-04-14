import cv2
import numpy as np
import math

# Auxiliary functions ------------------------------------------------------------------------------------------------------
# Print image
def show_image(title, image, scale_percent):
    imgResized = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)))
    cv2.imshow(title, imgResized)

# Draw circle
def draw_vertex(image, place, plane):
    cv2.circle(image, place, 30, (255,0,0), 20, cv2.LINE_AA)
    x = place[0] + 100
    y = place[1]
    coordinates = get_3D_coordinates(place, "matrix.npz", plane)
    draw_coordinates(image, (x,y), coordinates)

def draw_coordinates(image, place, coordinates):
    print(coordinates)
    cv2.putText(image, str(coordinates), place, cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 3)

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
    draw_vertex(cdst, (point1X, point1Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point2X, point2Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point3X, point3Y), (0, 1, 0, 0))
    draw_vertex(cdst, (point4X, point4Y), (0, 1, 0, 0))
    draw_vertex(cdst, (point5X, point5Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point6X, point6Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point7X, point7Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point8X, point8Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point9X, point9Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point10X, point10Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point11X, point11Y), (0, 1, 0, 0))
    #draw_vertex(cdst, (point12X, point12Y), (0, 1, 0, 0))

def get_3D_coordinates(vertex, matrix, plane):
    npzfile = np.load(matrix)
    P = npzfile['arr_0']
    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = plane[3]
    i = vertex[0]
    j = vertex[1]
    
    x = -((B*(P[2][2]*i - P[1][2]) - C*(i*P[1][1] - P[0][1]))*(-C*(P[2][3]*i - P[0][3]) - D*(P[2][2]*i - P[0][2])) - (B*(P[2][2]*i - P[0][2]) - C*(P[2][1]*i - P[0][1]))*(-C*(P[2][3]*j - P[1][3]) - D*(P[2][2]*i - P[1][2])))/((A*(P[2][2]*i - P[0][2]) - C*(P[2][0]*i - P[0][0]))*(B*(P[2][2]*i - P[1][2]) - C*(i*P[1][1] - P[0][1])) - (A*(P[2][2]*i - P[1][2]) - C*(P[2][0]*i - P[1][0]))*(B*(P[2][2]*i - P[0][2]) - C*(P[2][1]*i - P[0][1])))
    y = -(P[2][0]*C*P[0][3]*i - P[2][0]*C*P[2][3]*i*i + P[2][0]*C*P[2][3]*i*j - P[2][0]*C*i*P[1][3] + P[2][0]*D*P[0][2]*i - P[2][0]*D*i*P[1][2] - A*P[2][2]*P[0][3]*i + A*P[2][2]*P[2][3]*i*i - A*P[2][2]*P[2][3]*i*j + A*P[2][2]*i*P[1][3] + A*P[0][2]*P[2][3]*j - A*P[0][2]*P[1][3] + A*P[0][3]*P[1][2] - A*P[2][3]*i*P[1][2] - P[0][0]*C*P[2][3]*j + P[0][0]*C*P[1][3] - P[0][0]*D*P[2][2]*i + P[0][0]*D*P[1][2] - C*P[0][3]*P[1][0] + C*P[2][3]*i*P[1][0] + D*P[2][2]*i*P[1][0] - D*P[0][2]*P[1][0])/(-P[2][0]*B*P[0][2]*i + P[2][0]*B*i*P[1][2] - P[2][0]*P[2][1]*C*i*i + P[2][0]*C*i*i*P[1][1] + A*P[2][1]*P[2][2]*i*i - A*P[2][1]*i*P[1][2] - A*P[0][1]*P[0][2] + A*P[0][1]*P[1][2] - A*P[2][2]*i*i*P[1][1] + A*P[0][2]*i*P[1][1] + P[0][0]*B*P[2][2]*i - P[0][0]*B*P[1][2] + P[0][0]*C*P[0][1] - P[0][0]*C*i*P[1][1] - B*P[2][2]*i*P[1][0] + B*P[0][2]*P[1][0] + P[2][1]*C*i*P[1][0] - C*P[0][1]*P[1][0])
    z = -(-P[2][0]*B*P[0][3]*i + P[2][0]*B*P[2][3]*i*i - P[2][0]*B*P[2][3]*i*j + P[2][0]*B*i*P[1][3] + P[2][0]*P[2][1]*D*i*i - P[2][0]*D*i*i*P[1][1] + A*P[2][1]*P[2][3]*i*j - A*P[2][1]*i*P[1][3] - A*P[0][1]*P[0][3] + A*P[0][1]*P[2][3]*i - A*P[0][1]*P[2][3]*j + A*P[0][1]*P[1][3] + A*P[0][3]*i*P[1][1] - A*P[2][3]*i*i*P[1][1] + P[0][0]*B*P[2][3]*j - P[0][0]*B*P[1][3] - P[0][0]*P[0][1]*D + P[0][0]*D*i*P[1][1] + B*P[0][3]*P[1][0] - B*P[2][3]*i*P[1][0] - P[2][1]*D*i*P[1][0] + P[0][1]*D*P[1][0])/(-P[2][0]*B*P[0][2]*i + P[2][0]*B*i*P[1][2] - P[2][0]*P[2][1]*C*i*i + P[2][0]*C*i*i*P[1][1] + A*P[2][1]*P[2][2]*i*i - A*P[2][1]*i*P[1][2] - A*P[0][1]*P[0][2] + A*P[0][1]*P[1][2] - A*P[2][2]*i*i*P[1][1] + A*P[0][2]*i*P[1][1] + P[0][0]*B*P[2][2]*i - P[0][0]*B*P[1][2] + P[0][0]*C*P[0][1] - P[0][0]*C*i*P[1][1] - B*P[2][2]*i*P[1][0] + B*P[0][2]*P[1][0] + P[2][1]*C*i*P[1][0] - C*P[0][1]*P[1][0])
    
    decimal = 4
    return (round(x, decimal), round(y, decimal), round(z, decimal))
    #with the points (i,J)
    #(P[3][1]*i-P[1][1])*x+(P[3][2]*i-P[1][2])*y+(P[3][3]*i-P[1][3])*z = P[1][4]-P[3][4]*i
    #(P[3][1]*i-P[2][1])*x+(P[2][2]*i-P[1][2])*y+(P[3][3]*i-P[2][3])*z = P[2][4]-P[3][4]*j
    #A*x+B*y+C*z=D
    

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