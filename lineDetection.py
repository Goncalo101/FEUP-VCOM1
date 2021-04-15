import cv2
import numpy as np
import math

# Auxiliary functions ------------------------------------------------------------------------------------------------------

# Print image
def show_image(title, image, scale_percent):
    imgResized = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)))
    cv2.imshow(title, imgResized)

# Draw vertex
def draw_vertex(image, place2D, place3D):
    cv2.circle(image, place2D, 30, (255,0,0), 20, cv2.LINE_AA)
    x = place2D[0] + 100
    y = place2D[1]
    draw_coordinates(image, (x,y), place3D)

# Draw coordinates
def draw_coordinates(image, place, coordinates):
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
    return ((leftMostX, leftMostY), (rightMostX, rightMostY))

# Get bigger between two numbers
def get_bigger(n1, n2):
    if (n1 > n2):
        return n1
    else: return n2

# Get smaller between two numbers
def get_smaller(n1, n2):
    if (n1 < n2):
        return n1
    else: return n2

# Handle shadow lines and vertices
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
    (point1, point2) = get_line_limits_x(lines, minX, minY, maxX, int(round(maxY / 1.2)))
    # Get points #3 and #4
    (point3, point4) = get_line_limits_x(lines, minX, get_bigger(point1[1], point2[1]), get_smaller(point1[0], point2[0]), int(round(maxY / 1.05)))
    # Get points #5 and #6
    (point5, point6) = get_line_limits_x(lines, get_bigger(point1[0], point2[0]), get_bigger(point1[1], point2[1]), maxX, maxY)
    # Get points #7 and #8
    (point7, point8) = get_line_limits_x(lines, minX, minY, maxX, 1300)
    # Get points #9 and #10
    (point9, point10) = get_line_limits_x(lines, get_bigger(point1[0], point2[0]), get_bigger(point5[1], point6[1]), maxX, maxY)
    # Get points #11 and #12
    (point11, point12) = get_line_limits_x(lines, minX, 1900, get_smaller(point1[0], point2[0]), maxY)

    # Draw important vertices
    matrix = "matrix.npz"
    plane = (0, 1, 0, 0)
    draw_vertex(cdst, point1, get_3D_coordinates(point1, matrix, plane))
    draw_vertex(cdst, point2, get_3D_coordinates(point2, matrix, plane))
    draw_vertex(cdst, point3, get_3D_coordinates(point3, matrix, plane))
    draw_vertex(cdst, point4, get_3D_coordinates(point4, matrix, plane))
    draw_vertex(cdst, point5, get_3D_coordinates(point5, matrix, plane))
    draw_vertex(cdst, point6, get_3D_coordinates(point6, matrix, plane))
    draw_vertex(cdst, point7, get_3D_coordinates(point7, matrix, plane))
    draw_vertex(cdst, point8, get_3D_coordinates(point8, matrix, plane))
    draw_vertex(cdst, point9, get_3D_coordinates(point9, matrix, plane))
    draw_vertex(cdst, point10, get_3D_coordinates(point10, matrix, plane))
    draw_vertex(cdst, point11, get_3D_coordinates(point11, matrix, plane))
    draw_vertex(cdst, point12, get_3D_coordinates(point12, matrix, plane))

# Get 3D coordinates from a 2D vertex, a matrix and a plane in which the vertex is
def get_3D_coordinates(vertex, matrix, plane):
    #with the points (i,J)
    #(P[3][1]*i-P[1][1])*x+(P[3][2]*i-P[1][2])*y+(P[3][3]*i-P[1][3])*z = P[1][4]-P[3][4]*i
    #(P[3][1]*i-P[2][1])*x+(P[2][2]*i-P[1][2])*y+(P[3][3]*i-P[2][3])*z = P[2][4]-P[3][4]*j
    #A*x+B*y+C*z=D

    npzfile = np.load(matrix)
    P = npzfile['arr_0']

    C11 = P[0][0]
    C12 = P[0][1]
    C13 = P[0][2]
    C14 = P[0][3]

    C21 = P[1][0]
    C22 = P[1][1]
    C23 = P[1][2]
    C24 = P[1][3]
    
    C31 = P[2][0]
    C32 = P[2][1]
    C33 = P[2][2]
    C34 = P[2][3]

    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = plane[3]
    i = vertex[0]
    j = vertex[1]

    operands = np.array([
        [(C31*i-C11), (C32*i-C12), (C33*i-C13)], 
        [(C31*j-C21), (C32*j-C22), (C33*j-C23)],
        [A, B, C]])

    results = np.array([(C14-C34*i), (C24-C34*j), D])

    x = np.linalg.solve(operands, results)
    print(x)
    
    decimal = 4
    return (round(x[0], decimal), round(x[1], decimal), round(x[2], decimal))
    

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