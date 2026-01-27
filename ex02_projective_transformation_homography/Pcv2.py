# ============================================================
# File        : Pcv2.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Core implementation for Exercise 02.
#               Homography estimation and projective image warping
#               utilities used for rectification and panorama creation.
# ============================================================
import numpy as np
import cv2 as cv

GEOM_TYPE_POINT = 0
GEOM_TYPE_LINE  = 1


def getCondition2D(points):
    """
    Get the conditioning matrix of given points
    @param points: the points as matrix
    @return: the condition matrix (already allocated)
    """
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]

    # centroid
    tx = np.mean(x)
    ty = np.mean(y)

    # mean absolute distance to centroid
    sx = np.mean(np.abs(x - tx))
    sy = np.mean(np.abs(y - ty))

    # build T
    T = np.array([
        [1.0 / sx, 0.0, -tx / sx],
        [0.0, 1.0 / sy, -ty / sy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return T
    


def getDesignMatrix_homography2D(conditioned_base, conditioned_attach):
    """
    Define the design matrix as needed to compute 2D-homography
    @param conditioned_base: first set of conditioned points x' --> x' = H * x
    @param conditioned_attach: second set of conditioned points x --> x' = H * x
    @return: the design matrix to be computed
    """
    base = np.asarray(conditioned_base, dtype=np.float32)
    attach = np.asarray(conditioned_attach, dtype=np.float32)

    N = base.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float32)

    for i in range(N):
        u_p, v_p, w_p = base[i]      # x' = (u', v', w')
        u, v, w = attach[i]          # x  = (u,  v,  w)

        # first row
        A[2 * i, 0:3] = -w_p * np.array([u, v, w], dtype=np.float32)
        A[2 * i, 3:6] = 0.0
        A[2 * i, 6:9] =  u_p * np.array([u, v, w], dtype=np.float32)

        # second row
        A[2 * i + 1, 0:3] = 0.0
        A[2 * i + 1, 3:6] = -w_p * np.array([u, v, w], dtype=np.float32)
        A[2 * i + 1, 6:9] =  v_p * np.array([u, v, w], dtype=np.float32)

    return A


def solve_dlt_homography2D(A):
    """
    Solve homogeneous equation system by usage of SVD
    @param A: the design matrix
    @return: solution of the homogeneous equation system
    """
    A = np.asarray(A, dtype=np.float32) 
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]          
    H = h.reshape((3, 3))

    return H.astype(np.float32)


def decondition_homography2D(T_base, T_attach, H):
    """
    Decondition a homography that was estimated from conditioned point clouds
    @param T_base: conditioning matrix T' of first set of points x'
    @param T_attach: conditioning matrix T of second set of points x
    @param H: conditioned homography that has to be un-conditioned (in-place)
    @return: deconditioned homography
    """
    T_base = np.asarray(T_base, dtype=np.float32)
    T_attach = np.asarray(T_attach, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32)
    H_decond = np.linalg.inv(T_base) @ H @ T_attach

    return H_decond.astype(np.float32)


def homography2D(base, attach):
    """
    Compute the homography
    @param base: first set of points x'
    @param attach: second set of points x
    @return: homography H, so that x' = Hx
    """
    base = np.asarray(base, dtype=np.float32)
    attach = np.asarray(attach, dtype=np.float32)

    # 1) conditioning matrices
    T_base = getCondition2D(base)
    T_attach = getCondition2D(attach)

    # 2) conditioned points
    conditioned_base   = (T_base @ base.T).T
    conditioned_attach = (T_attach @ attach.T).T

    # 3) design matrix
    A = getDesignMatrix_homography2D(conditioned_base, conditioned_attach)

    # 4) DLT in conditioned space
    H_tilde = solve_dlt_homography2D(A)

    # 5) decondition
    H = decondition_homography2D(T_base, T_attach, H_tilde)

    return H.astype(np.float32)




# Functions from exercise 1
# Reuse your solutions from the last exercise here

def applyH_2D(geomObjects, H, gtype):
    """
    Applies a 2D transformation to an array of points or lines
    @param geomObjects: Array of input objects, each in homogeneous coordinates
    @param H: Matrix representing the transformation
    @param gtype: The type of the geometric objects, point or line. All are the same type.
    @return: Array of transformed objects.
    """
    
    #******* Small list/array cheat sheet ************************************/
    #
    #   Number of elements in array:                  len(a)
    #   Access i-th element (reading or writing):     a[i]
    #   Append an element to a list:                  a.append(element)
    #
    #**************************************************************************/

    result = []

    if gtype == GEOM_TYPE_POINT:
        for p in geomObjects:
            p_new = H @ p      # transform
            p_new = p_new / p_new[2]   # normalize
            result.append(p_new)

    elif gtype == GEOM_TYPE_LINE:
        H_inv_T = np.linalg.inv(H).T
        for l in geomObjects:
            l_new = H_inv_T @ l
            result.append(l_new)

    else:
        raise RuntimeError("Unhandled geometry type!")

    return result
    
def eucl2hom_point_2D(p):
    """
    Convert a 2D point from Euclidean to homogeneous coordinates
    @param p: The point to convert (in Euclidean coordinates)
    @return: The same point in homogeneous coordinates
    """
    p = np.asarray(p, dtype=np.float32)
    return np.array([p[0], p[1], 1.0], dtype=np.float32)
