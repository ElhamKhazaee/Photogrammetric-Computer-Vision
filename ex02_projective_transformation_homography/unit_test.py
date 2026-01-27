
# ============================================================
# File        : unit_test.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Unit tests for Exercise 02 (homography + warping).
# ============================================================

import numpy as np
from Pcv2 import (
    getCondition2D, getDesignMatrix_homography2D, solve_dlt_homography2D,
    decondition_homography2D, homography2D
)

def almost_equal_mat(A, B, eps=1e-3):
    return np.sum(np.abs(A - B)) <= eps

def test_getCondition2D():
    p = np.array([
        [93.0, 617.0, 1.0],
        [729.0, 742.0, 1.0],
        [703.0, 1233.0, 1.0],
        [152.0, 1103.0, 1.0],
    ], dtype=np.float32)

    Ttrue = np.array([
        [1./296.75, 0, -419.25/296.75],
        [0, 1./244.25, -923.75/244.25],
        [0, 0, 1]
    ], dtype=np.float32)

    Test = getCondition2D(p)
    Test = Test / Test[2,2]
    assert almost_equal_mat(Test, Ttrue), "getCondition2D failed"

def test_solve_and_decondition():
    # Matrix from C++ test - note the pattern matches the DLT formulation
    A = np.array([
        [ 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1],
        [ 0, 0, 0, 1.0994103, 1.2558856, -1, 1.0994103, 1.2558856, -1],
        [-1.0438079, 0.74411488, -1, 0, 0, 0, 1.0438079, -0.74411488, 1],
        [ 0, 0, 0, -1.0438079, 0.74411488, -1, -1.0438079, 0.74411488, -1],
        [-0.9561919, -1.2661204, -1, 0, 0, 0, 0.9561919, 1.2661204, 1],
        [ 0, 0, 0, -0.9561919, -1.2661204, -1, 0.9561919, 1.2661204, 1],
        [ 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1],
        [ 0, 0, 0, 0.90058976, -0.73387909, -1, -0.90058976, 0.73387909, 1],
    ], dtype=np.float32)

    Hest = solve_dlt_homography2D(A)
    Hest = Hest / Hest[2,2]
    Htrue = np.array([
        [0.57111752, -0.017852778, 0.013727478],
        [-0.15091757, 0.57065326, -0.04098846],
        [0.024604173, -0.041672569, 0.56645769]
    ], dtype=np.float32)
    Htrue /= Htrue[2,2]
    assert almost_equal_mat(Hest, Htrue), "solve_dlt_homography2D failed"

    T1 = np.array([[1./319.5,0,-1],[0,1./319.5,-1],[0,0,1]],dtype=np.float32)
    T2 = np.array([[1./296.75,0,-419.25/296.75],[0,1./244.25,-923.75/244.25],[0,0,1]],dtype=np.float32)
    H = decondition_homography2D(T1, T2, Hest)
    H /= H[2,2]
    Htrue2 = np.array([
        [0.9304952,-0.11296108,-16.839279],
        [-0.19729686,1.003845,-601.02362],
        [0.00012028422,-0.00024751772,1]
    ],dtype=np.float32)
    assert almost_equal_mat(H, Htrue2), "decondition_homography2D failed"

def test_homography2D():
    p1 = np.array([
        [0.0,0.0,1.0],
        [639.0,0.0,1.0],
        [639.0,639.0,1.0],
        [0.0,639.0,1.0],
    ],dtype=np.float32)
    p2 = np.array([
        [93.0,617.0,1.0],
        [729.0,742.0,1.0],
        [703.0,1233.0,1.0],
        [152.0,1103.0,1.0],
    ],dtype=np.float32)
    H = homography2D(p1,p2)
    H /= H[2,2]
    Htrue = np.array([
        [0.9304952,-0.11296108,-16.839279],
        [-0.19729686,1.003845,-601.02362],
        [0.00012028422,-0.00024751772,1]
    ],dtype=np.float32)
    assert almost_equal_mat(H,Htrue),"homography2D failed"

if __name__=="__main__":
    test_getCondition2D()
    test_solve_and_decondition()
    test_homography2D()
    print("Finished basic testing: Everything seems to be fine.")
