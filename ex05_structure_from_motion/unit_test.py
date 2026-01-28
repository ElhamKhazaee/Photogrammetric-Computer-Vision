# ==============================================================================
# File        : unit_test.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Unit tests for Exercise 05 –
# Structure from Motion (SfM) and bundle adjustment.
# ==============================================================================

import sys
import os
import math
import numpy as np


from Helper import OptimizationProblem, LevenbergMarquardt, hom2eucl

import Pcv5 as pcv5

DEBUG = True


def dbg(msg):
    if DEBUG:
        print(msg)

def test_MatSize(casename, a, rows, cols, channels):
    got_channels = 1 if a.ndim == 2 else a.shape[2]
    if (a.shape[0] != rows) or (a.shape[1] != cols) or (got_channels != channels):
        print('\n' + casename + ': fail')
        print('Expected:')
        print('({},{},{})'.format(rows, cols, channels))
        print('Given:')
        print('({},{},{})'.format(a.shape[0], a.shape[1], got_channels))
        return False
    return True

def test_closeMat(casename, a, b):
    eps = 1e-3
    if np.sum(np.abs(a - b)) > eps:
        print('Wrong or inaccurate calculations!')
        if casename:
            print('In matrix ' + casename + '!')
        print('Expected:')
        print(a)
        print('Given:')
        print(b)
        return False
    return True

def getFakePoints():
    p_fst = np.array([
        67, 215, 294, 100, 187, 281, 303, 162,
        18, 22, 74, 85, 115, 153, 194, 225,
        1, 1, 1, 1, 1, 1, 1, 1
    ], dtype=np.float32).reshape((3, 8))
    p_snd = np.array([
        5, 161, 227, 41, 100, 220, 237, 74,
        18, 22, 74, 85, 116, 152, 195, 225,
        1, 1, 1, 1, 1, 1, 1, 1
    ], dtype=np.float32).reshape((3, 8))
    return p_fst, p_snd

def getFakePointsWithOutliers():
    p_fst = np.array([
        314, 86, 346, 91, 330, 332, 97, 303,
        97, 96, 356, 367, 304, 343, 326, 86,
        341, 93, 338, 94, 366, 337, 96, 332,
        351, 369, 85, 342, 346, 354, 352, 727,
        360, 331, 85, 336, 341, 86, 335, 344,
        94, 420, 499, 374, 393, 328, 354, 345.6,
        315.6, 360, 350.4, 327.6, 330, 337.2, 345.6, 326.4,
        342, 360, 332.4, 366, 368.4, 313.2, 346.8, 304.8,
        319.2, 331.2, 339.6, 97.2, 85.2, 86.4, 96, 93.6,
        94.8, 86.4, 98.4, 88.8, 93.6, 344.4, 327.6, 338.4,
        426, 338.4, 378, 332.4, 346.8, 340.8, 303.6, 334.8,
        331.2, 320.4, 324, 499.2, 718.56, 335.52, 718.56, 498.24,
        345.6, 96.48, 367.2, 90.72, 154, 286, 246, 292,
        304, 302, 277, 87, 302, 297, 237, 260,
        90, 209, 250, 297, 240, 289, 240, 300,
        257, 169, 286, 240, 211, 256, 300, 251,
        255, 236, 254, 195, 253, 237, 279, 237,
        214, 281, 171, 214, 279, 84, 308, 309,
        259, 186, 140, 255.6, 254.4, 256.8, 211.2, 240,
        241.2, 169.2, 246, 249.6, 208.8, 253.2, 255.6, 256.8,
        256.8, 153.6, 226.8, 87.6, 249.6, 236.4, 248.4, 277.2,
        278.4, 282, 288, 289.2, 296.4, 297.6, 298.8, 300,
        300, 194.4, 140.4, 304.8, 306, 183.6, 109.2, 240,
        195.6, 213.6, 90, 236.4, 234, 146.4, 142.8, 308.4,
        416.16, 168.48, 410.4, 308.16, 230.4, 277.92, 256.32, 285.12,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1
    ], dtype=np.float32).reshape((3, 100))
    p_snd = np.array([
        327, 57.6, 366, 62.4, 352.8, 354, 70.8, 320.4,
        70.8, 68.4, 372, 386, 320.4, 358, 345.6, 57.6,
        352.8, 66.24, 352, 67.2, 386, 353, 69.6, 346.8,
        367, 388, 56.16, 360, 367, 372, 373, 338.4,
        380, 344, 55.2, 350, 358, 57.6, 350, 362,
        68, 436.8, 523, 398, 411, 343, 370, 367,
        331.2, 378.72, 367, 342, 346.8, 352.8, 366, 345.6,
        357.6, 379.2, 354.24, 385.2, 385.2, 326.4, 362.4, 320.4,
        332, 344.4, 360, 70.8, 56.16, 57.6, 67.392, 66.24,
        67.68, 57.6, 72, 60.48, 66.24, 360, 342, 360,
        449.28, 352.512, 398.4, 346.8, 361, 358, 320.4, 349.92,
        344.16, 334.8, 338.4, 523.2, 712.8, 350.784, 712.8, 522.72,
        360, 70.848, 385.2, 63.936, 148, 288, 246, 295.2,
        306, 303, 278.4, 79.2, 306, 301.2, 236, 260,
        82.8, 207, 249.12, 301.2, 238.8, 292.32, 239, 303.6,
        257, 166, 288, 240, 209, 256, 303.84, 251,
        255, 236, 254, 138, 253, 236, 280.8, 236,
        212, 283.2, 167, 212, 280, 82.8, 308, 310,
        259, 183, 136, 255, 253.44, 256.32, 209, 238.8,
        240, 165.6, 246, 249.12, 206.4, 253.2, 254.88, 256.8,
        256.8, 148.8, 225.6, 79.2, 249, 235.2, 247.68, 278.4,
        279.36, 283.68, 290.304, 292.32, 300.96, 300.96, 302.4, 303.84,
        303.84, 192, 135.6, 306, 306.72, 179.712, 106.8, 240,
        193, 212, 82.8, 234.72, 233.28, 141.6, 138, 308.4,
        406.8, 164.16, 400.8, 308.16, 229.2, 278.208, 256.8, 286.848,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1
    ], dtype=np.float32).reshape((3, 100))
    return p_fst, p_snd

def convertPointsOld2New(pset):
    return [pset[:, i].astype(np.float32) for i in range(pset.shape[1])]

def test_getCondition2D():
    correct = True
    return correct

def test_getFundamentalMatrix():
    print('===============================\nPcv5::getFundamentalMatrix(..):')
    try:
        correct = True
        Ftrue = np.array([
        6.4590546e-07, -0.00014758465, 0.015314385, 0.00015971341, -2.0858946e-05, -0.039460059, -0.016546328, 0.031596929,
        1
    ], dtype=np.float32).reshape((3, 3))
        p_fst, p_snd = getFakePoints()
        F = pcv5.getFundamentalMatrix(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd))
        dbg('Raw F from getFundamentalMatrix:\n{}'.format(F))
        if abs(F[2,2]) < 1e-4:
            print('Warning: There seems to be a problem with getFundamentalMatrix(..)!')
            print('	==> Expected F(2,2) to be nonzero!')
            print('	==> Got:')
            print(F)
            return False
        F = F / F[2,2]
        Ftrue = Ftrue / Ftrue[2,2]
        dbg('Normalized F:\n{}'.format(F))
        correct = test_closeMat('', Ftrue, F) and correct
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_getDesignMatrix_fundamental():
    print('===============================\nPcv5::getDesignMatrix_fundamental(..):')
    try:
        correct = True
        p1 = np.array([
        -1.8596188, 0.19237423, 1.2876947, -1.4020798, -0.1958406, 1.1074524, 1.4124782, -0.54246116,
        -1.5204918, -1.4549181, -0.60245907, -0.4221313, 0.069671988, 0.69262278, 1.3647538, 1.8729507,
        1, 1, 1, 1, 1, 1, 1, 1
    ], dtype=np.float32).reshape((3, 8))
        p2 = np.array([
        -1.64, 0.35679984, 1.2015998, -1.1791999, -0.42400002, 1.112, 1.3295999, -0.7568,
        -1.5194274, -1.4539878, -0.60327208, -0.4233129, 0.083844543, 0.67280149, 1.3762779, 1.8670754,
        1, 1, 1, 1, 1, 1, 1, 1
    ], dtype=np.float32).reshape((3, 8))
        Fest = pcv5.getDesignMatrix_fundamental(convertPointsOld2New(p1), convertPointsOld2New(p2))
        dbg('Design matrix shape: {}'.format(Fest.shape))
        dbg('Design matrix first row: {}'.format(Fest[0]))
        correct = (test_MatSize('Wrong dimensions!', Fest, 8, 9, 1) or test_MatSize('Wrong dimensions!', Fest, 9, 9, 1)) and correct
        Ftrue8 = np.array([
        3.0497749, 2.4936066, -1.64, 2.8255558, 2.310277, -1.5194274, -1.8596188, -1.5204918,
        1, 0.068639092, -0.51911455, 0.35679984, -0.27970979, 2.1154332, -1.4539878, 0.19237423,
        -1.4549181, 1, 1.5472938, -0.72391474, 1.2015998, -0.77683026, 0.36344674, -0.60327208,
        1.2876947, -0.60245907, 1, 1.6533325, 0.49777719, -1.1791999, 0.5935185, 0.17869362,
        -0.4233129, -1.4020798, -0.4221313, 1, 0.083036415, -0.029540924, -0.42400002, -0.016420165,
        0.0058416161, 0.083844543, -0.1958406, 0.069671988, 1, 1.231487, 0.7701965, 1.112,
        0.74509561, 0.46599764, 0.67280149, 1.1074524, 0.69262278, 1, 1.8780308, 1.8145765,
        1.3295999, 1.9439626, 1.8782806, 1.3762779, 1.4124782, 1.3647538, 1, 0.41053459,
        -1.4174491, -0.7568, -1.012816, 3.4969401, 1.8670754, -0.54246116, 1.8729507, 1
    ], dtype=np.float32).reshape((8, 9))
        Ftrue9 = np.array([
        3.0497749, 2.4936066, -1.64, 2.8255558, 2.310277, -1.5194274, -1.8596188, -1.5204918,
        1, 0.068639092, -0.51911455, 0.35679984, -0.27970979, 2.1154332, -1.4539878, 0.19237423,
        -1.4549181, 1, 1.5472938, -0.72391474, 1.2015998, -0.77683026, 0.36344674, -0.60327208,
        1.2876947, -0.60245907, 1, 1.6533325, 0.49777719, -1.1791999, 0.5935185, 0.17869362,
        -0.4233129, -1.4020798, -0.4221313, 1, 0.083036415, -0.029540924, -0.42400002, -0.016420165,
        0.0058416161, 0.083844543, -0.1958406, 0.069671988, 1, 1.231487, 0.7701965, 1.112,
        0.74509561, 0.46599764, 0.67280149, 1.1074524, 0.69262278, 1, 1.8780308, 1.8145765,
        1.3295999, 1.9439626, 1.8782806, 1.3762779, 1.4124782, 1.3647538, 1, 0.41053459,
        -1.4174491, -0.7568, -1.012816, 3.4969401, 1.8670754, -0.54246116, 1.8729507, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0
    ], dtype=np.float32).reshape((9, 9))
        if Fest.shape[0] == 8:
            correct = test_closeMat('', Ftrue8, Fest) and correct
        else:
            correct = test_closeMat('', Ftrue9, Fest) and correct
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_solve_dlt_F():
    print('===============================\nPcv5::solve_dlt_fundamental(..):')
    try:
        correct = True
        A = np.array([
        3.0497749, 2.4936066, -1.64, 2.8255558, 2.310277, -1.5194274, -1.8596188, -1.5204918,
        1, 0.068639092, -0.51911455, 0.35679984, -0.27970979, 2.1154332, -1.4539878, 0.19237423,
        -1.4549181, 1, 1.5472938, -0.72391474, 1.2015998, -0.77683026, 0.36344674, -0.60327208,
        1.2876947, -0.60245907, 1, 1.6533325, 0.49777719, -1.1791999, 0.5935185, 0.17869362,
        -0.4233129, -1.4020798, -0.4221313, 1, 0.083036415, -0.029540924, -0.42400002, -0.016420165,
        0.0058416161, 0.083844543, -0.1958406, 0.069671988, 1, 1.231487, 0.7701965, 1.112,
        0.74509561, 0.46599764, 0.67280149, 1.1074524, 0.69262278, 1, 1.8780308, 1.8145765,
        1.3295999, 1.9439626, 1.8782806, 1.3762779, 1.4124782, 1.3647538, 1, 0.41053459,
        -1.4174491, -0.7568, -1.012816, 3.4969401, 1.8670754, -0.54246116, 1.8729507, 1
    ], dtype=np.float32).reshape((8, 9))
        Fest = pcv5.solve_dlt_fundamental(A)
        dbg('solve_dlt_fundamental raw F:\n{}'.format(Fest))
        if abs(Fest[2,2]) < 1e-4:
            print('Warning: There seems to be a problem with solve_dlt_fundamental(..)!')
            print('	==> Expected F(2,2) to be nonzero!')
            print('	==> Got:')
            print(Fest)
            return False
        Fest = Fest / Fest[2,2]
        Ftrue = np.array([
        0.0083019603, -0.53950614, -0.047245972, 0.53861266, -0.059489254, -0.45286086, 0.075440452, 0.44964278,
        -0.0060508098
    ], dtype=np.float32).reshape((3, 3))
        Ftrue = Ftrue / Ftrue[2,2]
        dbg('solve_dlt_fundamental normalized F:\n{}'.format(Fest))
        correct = test_closeMat('', Ftrue, Fest) and correct
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_decondition_F():
    print('===============================\nPcv5::decondition(..):')
    try:
        correct = True
        H = np.array([
        0.0027884692, -0.53886771, -0.053913236, 0.53946984, -0.059588462, -0.45182425, 0.068957359, 0.45039368,
        -0.01389052
    ], dtype=np.float32).reshape((3, 3))
        T1 = np.array([
        0.013864818, 0, -2.7885616, 0, 0.016393442, -1.8155738, 0, 0,
        1
    ], dtype=np.float32).reshape((3, 3))
        T2 = np.array([
        0.0128, 0, -1.704, 0, 0.016359918, -1.813906, 0, 0,
        1
    ], dtype=np.float32).reshape((3, 3))
        H = pcv5.decondition_fundamental(T1, T2, H)
        dbg('Deconditioned F (raw):\n{}'.format(H))
        if abs(H[2,2]) < 1e-4:
            print('Warning: There seems to be a problem with decondition_fundamental(..)!')
            print('	==> Expected F(2,2) to be nonzero!')
            return False
        H = H / H[2,2]
        Htrue = np.array([
        6.4590546e-07, -0.00014758465, 0.015314385, 0.00015971341, -2.0858946e-05, -0.039460059, -0.016546328, 0.031596929,
        1
    ], dtype=np.float32).reshape((3, 3))
        dbg('Deconditioned F (normalized):\n{}'.format(H))
        correct = test_closeMat('', Htrue, H) and correct
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_forceSingularity():
    print('===============================\nPcv5::forceSingularity(..):')
    try:
        correct = True
        Fsest = np.array([
        0.0083019603, -0.53950614, -0.047245972, 0.53861266, -0.059489254, -0.45286086, 0.075440452, 0.44964278,
        -0.0060508098
    ], dtype=np.float32).reshape((3, 3))
        dbg('Input F before enforcing singularity:\n{}'.format(Fsest))
        Fsest = pcv5.forceSingularity(Fsest)
        dbg('Output F after enforcing singularity:\n{}'.format(Fsest))
        Fstrue = np.array([
        0.0027884692, -0.53886771, -0.053913236, 0.53946984, -0.059588462, -0.45182425, 0.068957359, 0.45039368,
        -0.01389052
    ], dtype=np.float32).reshape((3, 3))
        correct = test_closeMat('', Fstrue, Fsest) and correct
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_getError():
    print('===============================\nPcv5::getError(..):')
    try:
        correct = True
        Ftrue = np.array([
        0.18009815, 0.84612828, -124.47226, 0.51897198, 0.75658411, -182.07408, 0.00088265416, 0.0073684035,
        -0.94836563
    ], dtype=np.float32).reshape((3, 3))
        erTrue = 3983.8915033623125
        p_fst, p_snd = getFakePoints()
        erEst = pcv5.getError_set(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd), Ftrue)
        dbg('Computed error: {}, expected: {}'.format(erEst, erTrue))
        if abs(erEst - erTrue) > 1e-3:
            print('Wrong or inaccurate calculations!')
            print('In value "Error"!')
            print('Expected:')
            print(erTrue)
            print('Given:')
            print(erEst)
            return False
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_countInliers():
    print('===============================\nPcv5::countInliers(..):')
    try:
        correct = True
        F = np.array([
        6.4590546e-07, -0.00014758465, 0.015314385, 0.00015971341, -2.0858946e-05, -0.039460059, -0.016546328, 0.031596929,
        1
    ], dtype=np.float32).reshape((3, 3))
        p_fst, p_snd = getFakePoints()
        numInliers = pcv5.countInliers(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd), F, 1.0)
        dbg('countInliers returned {} (threshold=1.0)'.format(numInliers))
        true_numInliers = 5
        if numInliers != true_numInliers:
            print('Wrong or inaccurate calculations!')
            print('In value "Error"!')
            print('Expected:')
            print(true_numInliers)
            print('Given:')
            print(numInliers)
            return False
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_RANSAC():
    print('===============================\nPcv5::estimateFundamentalRANSAC(..):')
    try:
        correct = True
        p_fst, p_snd = getFakePointsWithOutliers()
        F = pcv5.estimateFundamentalRANSAC(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd), 2000, 1.0)
        numInliers = pcv5.countInliers(convertPointsOld2New(p_fst), convertPointsOld2New(p_snd), F, 1.0)
        dbg('RANSAC found F:\n{}'.format(F))
        dbg('RANSAC inliers: {}'.format(numInliers))
        if numInliers < 70:
            print('The solution that RANSAC finds is not very good (has few inliers)')
            print('Got {} inliers but expected around 100'.format(numInliers))
            correct = False
        if numInliers > 150:
            print('Something weird is going on with the test')
            correct = False
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_computeCameraPose():
    print('===============================\nPcv5::computeCameraPose(..):')
    try:
        correct = True
        return correct
    except Exception as exc:
        print(exc)
        return False


class RosenbrockMinimum(OptimizationProblem):
    class J(OptimizationProblem.JacobiMatrix):
        def __init__(self):
            self.j = np.zeros((2, 2), dtype=np.float32)

        def multiply(self, dst, src):
            dst[0] = src[0] * self.j[0, 0] + src[1] * self.j[0, 1]
            dst[1] = src[0] * self.j[1, 0] + src[1] * self.j[1, 1]

        def transposedMultiply(self, dst, src):
            dst[0] = src[0] * self.j[0, 0] + src[1] * self.j[1, 0]
            dst[1] = src[0] * self.j[0, 1] + src[1] * self.j[1, 1]

        def computeDiagJtJ(self, dst):
            dst[0] = self.j[0, 0] * self.j[0, 0] + self.j[0, 1] * self.j[1, 0]
            dst[1] = self.j[1, 0] * self.j[0, 1] + self.j[1, 1] * self.j[1, 1]

    class S(OptimizationProblem.State):
        def __init__(self):
            self.x = -0.5
            self.y = 0.5

        def clone(self):
            return RosenbrockMinimum.S()

        def computeResiduals(self, residuals):
            residuals[0] = 1.0 - self.x
            residuals[1] = 0.0 - 100.0 * (self.y - self.x * self.x)

        def computeJacobiMatrix(self, dst):
            dst.j[0, 0] = 1.0
            dst.j[0, 1] = 0.0
            dst.j[1, 0] = -200.0 * self.x
            dst.j[1, 1] = 100.0

        def update(self, update, dst):
            dst.x = self.x + update[0]
            dst.y = self.y + update[1]

    def __init__(self):
        super().__init__()
        self.m_numUpdateParameters = 2
        self.m_numResiduals = 2

    def createJacobiMatrix(self):
        return RosenbrockMinimum.J()

def test_Jacobi(name, optimizationProblem, state):
    print('===============================')
    print(name + ':')
    try:
        correct = True
        jacobi = optimizationProblem.createJacobiMatrix()
        state.computeJacobiMatrix(jacobi)
        posState = state.clone()
        negState = state.clone()
        update = np.zeros((optimizationProblem.getNumUpdateParameters(),), dtype=np.float32)
        posResiduals = np.zeros((optimizationProblem.getNumResiduals(),), dtype=np.float32)
        negResiduals = np.zeros((optimizationProblem.getNumResiduals(),), dtype=np.float32)
        residualGrad = np.zeros((optimizationProblem.getNumResiduals(),), dtype=np.float32)
        maskVector = np.zeros((optimizationProblem.getNumUpdateParameters(),), dtype=np.float32)
        derivativeVector = np.zeros((optimizationProblem.getNumResiduals(),), dtype=np.float32)
        delta = 0.01
        maxDiff = 1e-3
        for paramIdx in range(optimizationProblem.getNumUpdateParameters()):
            update[paramIdx] = delta
            state.update(update, posState)
            posState.computeResiduals(posResiduals)
            update[paramIdx] = -delta
            state.update(update, negState)
            negState.computeResiduals(negResiduals)
            update[paramIdx] = 0.0
            residualGrad = (negResiduals - posResiduals) / (2.0 * delta)
            maskVector.fill(0.0)
            derivativeVector.fill(0.0)
            maskVector[paramIdx] = 1.0
            jacobi.multiply(derivativeVector, maskVector)
            for r in range(optimizationProblem.getNumResiduals()):
                diff = derivativeVector[r] - residualGrad[r]
                if abs(diff) > maxDiff:
                    print('Error in JacobiMatrix::multiply for parameter {} and residual {}'.format(paramIdx, r))
                    print('   Expected: {} but got {}'.format(residualGrad[r], derivativeVector[r]))
                    correct = False
            for r in range(optimizationProblem.getNumResiduals()):
                maskVector.fill(0.0)
                derivativeVector.fill(0.0)
                derivativeVector[r] = 1.0
                jacobi.transposedMultiply(maskVector, derivativeVector)
                diff = maskVector[paramIdx] - residualGrad[r]
                if abs(diff) > maxDiff:
                    print('Error in JacobiMatrix::transposedMultiply for parameter {} and residual {}'.format(paramIdx, r))
                    print('   Expected: {} but got {}'.format(residualGrad[r], maskVector[paramIdx]))
                    correct = False
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_LevenbergMarquardt():
    print('===============================\nLevenberg Marquardt(..):')
    try:
        correct = True
        rosenbrock = RosenbrockMinimum()
        correct &= test_Jacobi('Rosenbrock jacobi computations', rosenbrock, RosenbrockMinimum.S())
        lm = LevenbergMarquardt(rosenbrock, RosenbrockMinimum.S())
        for _ in range(200):
            lm.iterate()
            if lm.getLastError() < 1e-5:
                break
        state = lm.getState()
        dbg('Levenberg-Marquardt final error: {}'.format(lm.getLastError()))
        dbg('Levenberg-Marquardt final state: x={}, y={}'.format(state.x, state.y))
        if abs(state.x - 1.0) > 1e-1:
            print('Error in LevenbergMarquardt implementation!')
            print('Final x should be 1.0 but is {}'.format(state.x))
            correct = False
        if abs(state.y - 1.0) > 1e-1:
            print('Error in LevenbergMarquardt implementation!')
            print('Final y should be 1.0 but is {}'.format(state.y))
            correct = False
        return correct
    except Exception as exc:
        print(exc)
        return False

def test_BundleAdjustment():
    print('===============================\nBundleAdjustment:')
    try:
        correct = True
        C1Pose = np.eye(4, dtype=np.float32)
        C2Pose = np.eye(4, dtype=np.float32)
        C2Pose[0, 3] = 1.0
        C3Pose = np.eye(4, dtype=np.float32)
        C3Pose[0, 2] = 1.0
        K = np.eye(3, dtype=np.float32)
        tracks = [np.zeros(4, dtype=np.float32) for _ in range(10)]
        for j in range(2):
            for i in range(5):
                idx = j * 5 + i
                tracks[idx][0] = i - 2.5
                tracks[idx][1] = j - 0.5
                tracks[idx][2] = 5.0 + 0.02 * tracks[idx][0] * tracks[idx][0]
                tracks[idx][3] = 1.0
        baScene = pcv5.Scene(cameras=[pcv5.Camera(0, []) for _ in range(3)], numTracks=len(tracks), numInternalCalibs=1)
        for i in range(5):
            p = K @ np.eye(3, 4, dtype=np.float32) @ C1Pose @ tracks[i]
            baScene.cameras[0].keypoints.append(pcv5.KeyPoint(hom2eucl(p), i, 0.5 + i * 0.1))
        for i in range(3, 8):
            p = K @ np.eye(3, 4, dtype=np.float32) @ C2Pose @ tracks[i]
            baScene.cameras[1].keypoints.append(pcv5.KeyPoint(hom2eucl(p), i, 0.5 + (i-3) * 0.1))
        for i in range(5, 10):
            p = K @ np.eye(3, 4, dtype=np.float32) @ C3Pose @ tracks[i]
            baScene.cameras[2].keypoints.append(pcv5.KeyPoint(hom2eucl(p), i, 0.5 + (i-5) * 0.1))
        rng = np.random.default_rng(1337)
        def smallDeviation():
            return float(rng.normal(0.0, 0.2))
        ba = pcv5.BundleAdjustment(baScene)
        state = pcv5.BundleAdjustment.BAState(baScene)
        state.m_internalCalibs[0].K = K.copy()
        state.m_internalCalibs[0].K[0, 0] = state.m_internalCalibs[0].K[0, 0] + smallDeviation()
        state.m_internalCalibs[0].K[1, 1] = state.m_internalCalibs[0].K[0, 0]
        for i in range(len(tracks)):
            state.m_tracks[i].location = tracks[i].copy()
            state.m_tracks[i].location[0] += smallDeviation()
            state.m_tracks[i].location[1] += smallDeviation()
            state.m_tracks[i].location[2] += smallDeviation()
            state.m_tracks[i].location = state.m_tracks[i].location / np.linalg.norm(state.m_tracks[i].location)
        state.m_cameras[0].H = C1Pose.copy()
        state.m_cameras[0].H[0, 3] += smallDeviation()
        state.m_cameras[0].H[1, 3] += smallDeviation()
        state.m_cameras[0].H[2, 3] += smallDeviation()
        state.m_cameras[1].H = C2Pose.copy()
        state.m_cameras[1].H[0, 3] += smallDeviation()
        state.m_cameras[1].H[1, 3] += smallDeviation()
        state.m_cameras[1].H[2, 3] += smallDeviation()
        state.m_cameras[2].H = C3Pose.copy()
        state.m_cameras[2].H[0, 3] += smallDeviation()
        state.m_cameras[2].H[1, 3] += smallDeviation()
        state.m_cameras[2].H[2, 3] += smallDeviation()
        correct &= test_Jacobi('BundleAdjustment jacobi computations', ba, state)
        if not correct:
            return False
        res_init = np.zeros((ba.getNumResiduals(),), dtype=np.float32)
        state.computeResiduals(res_init)
        dbg('Initial BA error: {}'.format(float(np.dot(res_init, res_init))))
        lm = LevenbergMarquardt(ba, state)
        for i in range(200):
            lm.iterate()
            if lm.getLastError() < 1e-5:
                break
        dbg('Final BA error after LM: {}'.format(lm.getLastError()))
        if lm.getLastError() > 1e-3:
            print('Bundleadjustment does not converge to a small error.')
            print('Expected error < 1e-3, but got {}'.format(lm.getLastError()))
            correct = False
        return correct
    except Exception as exc:
        print(exc)
        return False

def main():
    print('\n********************')
    print('Testing: Start')
    correct = True
    correct &= test_getFundamentalMatrix()
    correct &= test_getCondition2D()
    correct &= test_getDesignMatrix_fundamental()
    correct &= test_solve_dlt_F()
    correct &= test_decondition_F()
    correct &= test_forceSingularity()
    correct &= test_getError()
    correct &= test_countInliers()
    correct &= test_RANSAC()
    correct &= test_computeCameraPose()
    correct &= test_LevenbergMarquardt()
    correct &= test_BundleAdjustment()
    print('Testing: Done')
    if correct:
        print('Everything seems (!) to be correct.')
        return 0
    print('There seem to be problems.')
    return -1

if __name__ == '__main__':
    raise SystemExit(main())
