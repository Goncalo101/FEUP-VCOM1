import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Auxiliary functions ------------------------------------------------------------------------------------------------------

# Print image
def show_image(title, image, scale_percent):
    """
    Opens a window with the image
    @param
    title - Title of the window
    image - Image to be printed
    scale_percent - Resize scale
    """
    # Resize image
    imgResized = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)))
    # Print image
    cv2.imshow(title, imgResized)

# Draw vertex
def draw_vertex(image, place2D, place3D, showLeft = False):
    """
    Draws a blue circle and 3D coordinates text, representing a vertex, in given 2D coordinates of the image
    @param
    image - Image where to print vertex
    place2D - Place on the image where to print coordinates in (x, y) format
    place3D - 3D coordinates to be printed, corresponding to the real world coordinates of the vertex
    showLeft - Whether to print the 3D coordinates on the left or on the right
    """
    # Draw circle
    cv2.circle(image, place2D, 30, (255,0,0), 20, cv2.LINE_AA)
    if (showLeft):
        x = place2D[0] - 900
        y = place2D[1] - 50
    else:
        x = place2D[0] + 50
        y = place2D[1] - 50
    # Draw coordinates
    draw_coordinates(image, (x,y), place3D)

# Draw coordinates
def draw_coordinates(image, place, coordinates):
    """
    Draws 3D coordinates text, representing a vertex, in given 2D coordinates of the image
    @param
    image - Image where to print coordinates
    place - Place on the image where to print coordinates in (x, y) format
    coordinates - 3D coordinates to be printed, corresponding to the real world coordinates of the vertex
    """
    # Draw coordinates text
    cv2.putText(image, str(coordinates), place, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)

# Get line limits
def get_line_limits_x(lines, minX, minY, maxX, maxY):
    """
    Calculates line left and right limits vertices, for a specified range
    @param
    lines - Previously defined lines in the image
    minX - Minimum value in x-axis
    minY - Minimum value in y-axis
    maxX - Maximum value in x-axis
    maxY - Maximum value in y-axis
    @return Returns in a pair the most left and the most right vertices of the line, respectively
    """
    leftMostX = maxX
    leftMostY = maxY
    rightMostX = minX
    rightMostY = minY
    if lines is not None:
        # For each line in range
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

# Get higher between two numbers
def get_bigger(n1, n2):
    """
    Calculates the highest number between two given numbers
    @param
    n1 - Number 1
    n2 - Number 2
    @return Returns n1 if it is higher than n2, n2 otherwise 
    """
    if (n1 > n2):
        return n1
    else: return n2

# Get smaller between two numbers
def get_smaller(n1, n2):
    """
    Calculates the smallest number between two given numbers
    @param
    n1 - Number 1
    n2 - Number 2
    @return Returns n1 if it is smaller than n2, n2 otherwise 
    """
    if (n1 < n2):
        return n1
    else: return n2

# Handle shadow lines and vertices
def line_shadow_plane(lines):
    """
    Draws the previously defined lines in the image and their limits vertices
    @param
    lines - Lines to be drawn
    @return Return points of interest to draw Z graph, return only the points form the line on top 
    """
    maxY = 2000
    minY = 900
    maxX = imgOriginal.shape[1]
    minX = 0

    # Draw the lines
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
    plane = (0, 1, 0, 0) # plane y = 0
    point1Coordinates = get_3D_coordinates(point1, matrix, plane)
    draw_vertex(cdst, point1, point1Coordinates)
    point2Coordinates = get_3D_coordinates(point2, matrix, plane)
    draw_vertex(cdst, point2, point2Coordinates)

    point9Coordinates = get_3D_coordinates(point9, matrix, plane)
    draw_vertex(cdst, point9, point9Coordinates, True)
    point10Coordinates = get_3D_coordinates(point10, matrix, plane)
    draw_vertex(cdst, point10, point10Coordinates, True)

    point11Coordinates = get_3D_coordinates(point11, matrix, plane)
    draw_vertex(cdst, point11, point11Coordinates)
    point12Coordinates = get_3D_coordinates(point12, matrix, plane)
    draw_vertex(cdst, point12, point12Coordinates)

    point3Coordinates = get_3D_coordinates(point3, matrix, (0,0,1,point11Coordinates[2]))
    draw_vertex(cdst, point3, point3Coordinates)
    point4Coordinates = get_3D_coordinates(point4, matrix, (0,0,1,point12Coordinates[2]))
    draw_vertex(cdst, point4, point4Coordinates)

    point5Coordinates = get_3D_coordinates(point5, matrix, (0,0,1,point9Coordinates[2]))
    draw_vertex(cdst, point5, point5Coordinates, True)
    point6Coordinates = get_3D_coordinates(point6, matrix, (0,0,1,point10Coordinates[2]))
    draw_vertex(cdst, point6, point6Coordinates, True)

    point7Coordinates = get_3D_coordinates(point7, matrix, (0,0,1,point1Coordinates[2]))
    draw_vertex(cdst, point7, point7Coordinates)
    point8Coordinates = get_3D_coordinates(point8, matrix, (0,0,1,point2Coordinates[2]))
    draw_vertex(cdst, point8, point8Coordinates)

    # Return points of interest to draw Z graph, return only the points form the line on top 
    return np.asarray([list(point1Coordinates),list(point2Coordinates),list(point3Coordinates),list(point4Coordinates),list(point5Coordinates),list(point6Coordinates)])

# Get 3D coordinates from a 2D vertex, a matrix and a plane in which the vertex is
def get_3D_coordinates(vertex, matrix, plane):
    """
    Get 3D real world coordinates from a 2D vertex, a matrix and a plane in which the vertex is
    @param
    vertex - 2d coordinates of a vertex in the image in format (x, y)
    matrix - Perspective projection matrix file name
    plane - Plane from which the vertex belongs in 3D coordinates
    """
    # With the points (i,J)
    # (P[3][1]*i-P[1][1])*x+(P[3][2]*i-P[1][2])*y+(P[3][3]*i-P[1][3])*z = P[1][4]-P[3][4]*i
    # (P[3][1]*j-P[2][1])*x+(P[3][2]*j-P[2][2])*y+(P[3][3]*j-P[2][3])*z = P[2][4]-P[3][4]*j
    # A*x+B*y+C*z=D

    # Load previously exported perspective projection matrix file
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

    # Plane coefficients
    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = plane[3]

    # Vertex 2D coordinates
    i = vertex[0]
    j = vertex[1]

    # Solve system
    operands = np.array([
        [(C31*i-C11), (C32*i-C12), (C33*i-C13)], 
        [(C31*j-C21), (C32*j-C22), (C33*j-C23)],
        [A, B, C]])

    results = np.array([(C14-C34*i), (C24-C34*j), D])
    x = np.linalg.solve(operands, results)
    # Print results on the console
    print(x)
    
    # Return coordinates with a rounded value
    decimal = 4
    
    return (round(x[0], decimal), round(x[1], decimal), round(x[2], decimal))
    
def plotGraph(points):
    """
    Plot a graph to show high Z variation along X 
    @param
    points - 3d matrix with the coordinates of the points to be ploted
    """
    # Sort Points by X in ascending order
    sorted = points[points[:,0].argsort()]
    # Select only X coordenates from the matrix 
    x = sorted[...,0]
    # Select only X coordenates from the matrix 
    z = sorted[...,2]
    # Ploting graph 
    plt.figure()
    plt.plot(x,z)
    plt.show()

# Main -----------------------------------------------------------------------------------------------------------------------

# Opening an image
imgOriginal = cv2.imread('./assets/images/i/IMG_0922.JPG')


# Convert to grayscale
imgGrey = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

# Apply Bilateral Filter to remove noise
# cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
# d - Diameter of each pixel neighborhood used during filtering
# sigmaColor - Filter sigma in the color space: represents the amount of colors within the pixel neighborhood that will be mixed together, resulting in larger areas of semi-equal color
# sigmaSpace - Filter sigma in the coordinate space: larger values mean that farther pixels will influence each other as long as their colors are close enough
imgBilateral = cv2.bilateralFilter(imgGrey, 11, 75, 75)


# Apply Canny Filter to detect edges
# cv2.Canny(image, threshold1, threshold2, edges, apertureSize)
# threshold1 - first threshold for the hysteresis procedure
# threshold1 - second threshold for the hysteresis procedure
# apertureSize - aperture size for the Sobel operator
imgWithCanny = cv2.Canny(imgBilateral, 60, 60, None, 3)


# Fill the lines using Dilation and Erosion
# Dilate
# kernel - matrix 5x5 filled with 1s used to convolve the image
kernel = np.ones((5, 5), np.uint8)
imgDilate = cv2.dilate(imgWithCanny, kernel, iterations=7)
# Erode
imgErode = cv2.erode(imgDilate, kernel, iterations=7)


# Copy edges to the images that will display the results in BGR
cdst = imgOriginal.copy()


# Probabilistic Hough Line Transform - lines in green color
# cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
# rho - Distance resolution of the accumulator in pixels
# theta - Angle resolution of the accumulator in radians
# threshold - Accumulator threshold; only lines with votes > threshold are returned
# minLineLength - Minimum length of line
# maxLineGap - Maximum allowed gap between line segments to treat them as single line
lines = cv2.HoughLinesP(imgErode, 1, np.pi / 180, 50, minLineLength=85, maxLineGap=50)
# Shadow
points = line_shadow_plane(lines)

#Plot High Graph
plotGraph(points)

# Print Original Image
scale_percent = 20
#show_image('Img', imgOriginal, scale_percent)
# Print Grey Image
#show_image('Img Grey', imgGrey, scale_percent)
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
#plot Z graph 
cv2.waitKey(0)