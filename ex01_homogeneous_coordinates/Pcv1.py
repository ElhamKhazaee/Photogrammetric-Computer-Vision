# ============================================================
# File        : main.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Institution : Technische Universität Berlin
# Description : Entry point for Exercise 01 (Homogeneous Coordinates).
#               Runs processing and validation routines.
# ============================================================
import numpy as np
import cv2 as cv

GEOM_TYPE_POINT = 0
GEOM_TYPE_LINE  = 1


def eucl2hom_point_2D(p):
    """
    Convert a 2D point from Euclidean to homogeneous coordinates
    @param p: The point to convert (in Euclidean coordinates)
    @return: The same point in homogeneous coordinates
    """
    # TO DO !!!
    return np.array([p[0], p[1], 1.0], dtype=np.float32)


def hom2eucl_point_2D(p):
    """
    Convert a 2D point from homogeneous to Euclidean coordinates
    @param p: The point to convert in homogeneous coordinates
    @return: The same point in Euclidean coordinates
    """
    # TO DO !!!
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float32)


def getConnectingLine_2D(p1, p2):
    """
    Calculates the joining line between two points (in 2D)
    @param p1: First of the two points in homogeneous coordinates
    @param p2: Second of the two points in homogeneous coordinates
    @return: The joining line in homogeneous coordinates
    """
    # TO DO !!!
    return np.cross(p1, p2)


def getTranslationMatrix_2D(dx, dy):
    T = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0, 1]], dtype=np.float32)
    return T



def getRotationMatrix_2D(phi):
    phi_rad = np.deg2rad(phi)
    R = np.array([[np.cos(phi_rad), -np.sin(phi_rad), 0],
                  [np.sin(phi_rad),  np.cos(phi_rad), 0],
                  [0, 0, 1]], dtype=np.float32)
    return R



def getScalingMatrix_2D(lambda_):
    S = np.array([[lambda_, 0, 0],
                  [0, lambda_, 0],
                  [0, 0, 1]], dtype=np.float32)
    return S



def getH_2D(T, R, S):
    """
    Combines translation-, rotation-, and scaling-matrices to a single transformation matrix H.
    The returned transformation behaves as if objects were first transformed by T, then by R, and finally by S.
    """
    H = S @ R @ T   # apply T, then R, then S to column vectors
    return H.astype(np.float32)





def applyH_2D(geomObjects, H, gtype):
    """
    Applies a 2D transformation to an array of points or lines
    @param geomObjects: Array of input objects, each in homogeneous coordinates
    @param H: Matrix representing the transformation
    @param gtype: The type of the geometric objects, point or line. All are the same type.
    @return: Array of transformed objects.
    """
    result = []
    
    #******* Small list/array cheat sheet ************************************/
    #
    #   Number of elements in array:                  len(a)
    #   Access i-th element (reading or writing):     a[i]
    #   Append an element to a list:                  a.append(element)
    #
    #**************************************************************************/

    # TO DO !!!

    if gtype == GEOM_TYPE_POINT:
        for p in geomObjects:
            p_new = H @ p
            p_new /= p_new[2]
            result.append(p_new)
    elif gtype == GEOM_TYPE_LINE:
        H_inv_T = np.linalg.inv(H).T
        for l in geomObjects:
            l_new = H_inv_T @ l
            result.append(l_new)
    return result


def isPointOnLine_2D(point, line, eps=1e-5):
    """
    Checks if a point is on a line
    @param point: The given point in homogeneous coordinates
    @param line: The given line in homogeneous coordinates
    @param eps: The used accuracy (allowed distance to still be considered on the line)
    @return: Returns True if the point is on the line
    """
    # TO DO !!!
    return abs(line.T @ point) < eps


def run(filename):
    """
    Function loads input image, calls processing function and saves result (usually)
    @param filename: Path to input image
    """
    # window names
    win1 = "Image"

    # load image as gray-scale
    print("Load image: start")
    inputImage = None

    inputImage = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    if inputImage is None:
        print(f"ERROR: image could not be loaded from {filename}")
        print("Press enter to continue...")
        input()
    else:
        print(f"Load image: done ( {inputImage.shape[0]} x {inputImage.shape[1]} )")
    
    # show input image
    cv.namedWindow(win1, cv.WINDOW_AUTOSIZE)
    cv.imshow(win1, inputImage)
    cv.imwrite("output_mandrill.png", inputImage)
    cv.waitKey(0)

    # the two given points as numpy arrays
    x = np.array([2.0, 3.0], dtype=np.float32)
    y = np.array([-4.0, 5.0], dtype=np.float32)

    # same points in homogeneous coordinates
    v1 = None
    v2 = None
    # TO DO !!!
    # define v1 as homogeneous version of x
    # define v2 as homogeneous version of y
    v1 = eucl2hom_point_2D(x)
    v2 = eucl2hom_point_2D(y)

    # print points
    print(f"point 1: {v1}^T")
    print(f"point 2: {v2}^T")
    print()
    
    # connecting line between those points in homogeneous coordinates
    line = getConnectingLine_2D(v1, v2)
    
    # print line
    print(f"joining line: {line}^T")
    print()
    
    # the parameters of the transformation
    dx = 6              # translation in x
    dy = -7             # translation in y
    phi = 15            # rotation angle in degree
    lambda_ = 8         # scaling factor

    # matrices for transformation
    # calculate translation matrix
    T = getTranslationMatrix_2D(dx, dy)
    # calculate rotation matrix
    R = getRotationMatrix_2D(phi)
    # calculate scale matrix
    S = getScalingMatrix_2D(lambda_)
    # combine individual transformations to a homography
    H = getH_2D(T, R, S)
    
    # print calculated matrices
    print("Translation matrix:")
    print(T)
    print()
    print("Rotation matrix:")
    print(R)
    print()
    print("Scaling matrix:")
    print(S)
    print()
    print("Homography:")
    print(H)
    print()

    # transform first point x (and print it)
    v1_new = applyH_2D([v1], H, GEOM_TYPE_POINT)[0]
    print(f"new point 1: {v1_new}^T")
    print(f"new point 1 (eucl): {hom2eucl_point_2D(v1_new)}^T")
    # transform second point y (and print it)
    v2_new = applyH_2D([v2], H, GEOM_TYPE_POINT)[0]
    print(f"new point 2: {v2_new}^T")
    print(f"new point 2 (eucl): {hom2eucl_point_2D(v2_new)}^T")
    print()
    # transform joining line (and print it)
    line_new = applyH_2D([line], H, GEOM_TYPE_LINE)[0]
    print(f"new line: {line_new}^T")
    print()

    # check if transformed points are still on transformed line
    xOnLine = isPointOnLine_2D(v1_new, line_new)
    yOnLine = isPointOnLine_2D(v2_new, line_new)
    if xOnLine:
        print("first point lies still on the line *yay*")
    else:
        print("first point does not lie on the line *oh oh*")
    if yOnLine:
        print("second point lies still on the line *yay*")
    else:
        print("second point does not lie on the line *oh oh*")
if __name__ == "__main__":
    run("mandrill.png")
