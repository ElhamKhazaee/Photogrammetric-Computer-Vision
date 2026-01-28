# ==============================================================================
# File        : unit_test.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Unit tests for Exercise 04 –
#               Fundamental Matrix Estimation.
# ==============================================================================

import os
import sys
import random

import numpy as np

student_dir = os.environ.get("STUDENT_PCV4_DIR")
if student_dir and student_dir not in sys.path:
    sys.path.insert(0, student_dir)

if os.environ.get("PCV4_NON_INTERACTIVE") == "1":
    def input(prompt=""):
        if prompt:
            print(prompt)
        return ""

from Pcv4 import (
    get_condition_2d,
    get_design_matrix_fundamental,
    solve_dlt_fundamental,
    force_singularity,
    decondition_fundamental,
    get_fundamental_matrix,
    get_error_multiple,
    get_error_single,
    count_inliers,
    estimate_fundamental_ransac,
)


def convert_points_old_to_new(pset: np.ndarray) -> list:
    """Convert from column-based matrix to list of vectors."""
    result = []
    for i in range(pset.shape[1]):
        result.append(np.array([pset[0, i], pset[1, i], pset[2, i]], dtype=np.float32))
    return result


def get_fake_points():
    """Generate fake point correspondences for testing."""
    p_fst = np.zeros((3, 8), dtype=np.float32)
    p_snd = np.zeros((3, 8), dtype=np.float32)

    p_fst[0, 0] = 67;  p_fst[1, 0] = 18;  p_fst[2, 0] = 1
    p_snd[0, 0] = 5;   p_snd[1, 0] = 18;  p_snd[2, 0] = 1

    p_fst[0, 1] = 215; p_fst[1, 1] = 22;  p_fst[2, 1] = 1
    p_snd[0, 1] = 161; p_snd[1, 1] = 22;  p_snd[2, 1] = 1

    p_fst[0, 2] = 294; p_fst[1, 2] = 74;  p_fst[2, 2] = 1
    p_snd[0, 2] = 227; p_snd[1, 2] = 74;  p_snd[2, 2] = 1

    p_fst[0, 3] = 100; p_fst[1, 3] = 85;  p_fst[2, 3] = 1
    p_snd[0, 3] = 41;  p_snd[1, 3] = 85;  p_snd[2, 3] = 1

    p_fst[0, 4] = 187; p_fst[1, 4] = 115; p_fst[2, 4] = 1
    p_snd[0, 4] = 100; p_snd[1, 4] = 116; p_snd[2, 4] = 1

    p_fst[0, 5] = 281; p_fst[1, 5] = 153; p_fst[2, 5] = 1
    p_snd[0, 5] = 220; p_snd[1, 5] = 152; p_snd[2, 5] = 1

    p_fst[0, 6] = 303; p_fst[1, 6] = 194; p_fst[2, 6] = 1
    p_snd[0, 6] = 237; p_snd[1, 6] = 195; p_snd[2, 6] = 1

    p_fst[0, 7] = 162; p_fst[1, 7] = 225; p_fst[2, 7] = 1
    p_snd[0, 7] = 74;  p_snd[1, 7] = 225; p_snd[2, 7] = 1

    return p_fst, p_snd


def get_fake_points_with_outliers():
    """Generate fake point correspondences with outliers for RANSAC testing."""
    num_outliers = 100
    num_inliers = 100

    p_fst = np.zeros((3, num_inliers + num_outliers), dtype=np.float32)
    p_snd = np.zeros((3, num_inliers + num_outliers), dtype=np.float32)

    # Inlier data (truncated for brevity - same as C++ version)
    inlier_data = [
        (314, 154, 327, 148), (86, 286, 57.6, 288), (346, 246, 366, 246),
        (91, 292, 62.4, 295.2), (330, 304, 352.8, 306), (332, 302, 354, 303),
        (97, 277, 70.8, 278.4), (303, 87, 320.4, 79.2), (97, 302, 70.8, 306),
        (96, 297, 68.4, 301.2), (356, 237, 372, 236), (367, 260, 386, 260),
        (304, 90, 320.4, 82.8), (343, 209, 358, 207), (326, 250, 345.6, 249.12),
        (86, 297, 57.6, 301.2), (341, 240, 352.8, 238.8), (93, 289, 66.24, 292.32),
        (338, 240, 352, 239), (94, 300, 67.2, 303.6), (366, 257, 386, 257),
        (337, 169, 353, 166), (96, 286, 69.6, 288), (332, 240, 346.8, 240),
        (351, 211, 367, 209), (369, 256, 388, 256), (85, 300, 56.16, 303.84),
        (342, 251, 360, 251), (346, 255, 367, 255), (354, 236, 372, 236),
        (352, 254, 373, 254), (727, 195, 338.4, 138), (360, 253, 380, 253),
        (331, 237, 344, 236), (85, 279, 55.2, 280.8), (336, 237, 350, 236),
        (341, 214, 358, 212), (86, 281, 57.6, 283.2), (335, 171, 350, 167),
        (344, 214, 362, 212), (94, 279, 68, 280), (420, 84, 436.8, 82.8),
        (499, 308, 523, 308), (374, 309, 398, 310), (393, 259, 411, 259),
        (328, 186, 343, 183), (354, 140, 370, 136), (345.6, 255.6, 367, 255),
        (315.6, 254.4, 331.2, 253.44), (360, 256.8, 378.72, 256.32),
        (350.4, 211.2, 367, 209), (327.6, 240, 342, 238.8), (330, 241.2, 346.8, 240),
        (337.2, 169.2, 352.8, 165.6), (345.6, 246, 366, 246), (326.4, 249.6, 345.6, 249.12),
        (342, 208.8, 357.6, 206.4), (360, 253.2, 379.2, 253.2), (332.4, 255.6, 354.24, 254.88),
        (366, 256.8, 385.2, 256.8), (368.4, 256.8, 385.2, 256.8), (313.2, 153.6, 326.4, 148.8),
        (346.8, 226.8, 362.4, 225.6), (304.8, 87.6, 320.4, 79.2), (319.2, 249.6, 332, 249),
        (331.2, 236.4, 344.4, 235.2), (339.6, 248.4, 360, 247.68), (97.2, 277.2, 70.8, 278.4),
        (85.2, 278.4, 56.16, 279.36), (86.4, 282, 57.6, 283.68), (96, 288, 67.392, 290.304),
        (93.6, 289.2, 66.24, 292.32), (94.8, 296.4, 67.68, 300.96), (86.4, 297.6, 57.6, 300.96),
        (98.4, 298.8, 72, 302.4), (88.8, 300, 60.48, 303.84), (93.6, 300, 66.24, 303.84),
        (344.4, 194.4, 360, 192), (327.6, 140.4, 342, 135.6), (338.4, 304.8, 360, 306),
        (426, 306, 449.28, 306.72), (338.4, 183.6, 352.512, 179.712), (378, 109.2, 398.4, 106.8),
        (332.4, 240, 346.8, 240), (346.8, 195.6, 361, 193), (340.8, 213.6, 358, 212),
        (303.6, 90, 320.4, 82.8), (334.8, 236.4, 349.92, 234.72), (331.2, 234, 344.16, 233.28),
        (320.4, 146.4, 334.8, 141.6), (324, 142.8, 338.4, 138), (499.2, 308.4, 523.2, 308.4),
        (718.56, 416.16, 712.8, 406.8), (335.52, 168.48, 350.784, 164.16),
        (718.56, 410.4, 712.8, 400.8), (498.24, 308.16, 522.72, 308.16),
        (345.6, 230.4, 360, 229.2), (96.48, 277.92, 70.848, 278.208),
        (367.2, 256.32, 385.2, 256.8), (90.72, 285.12, 63.936, 286.848),
    ]

    for i, (x1, y1, x2, y2) in enumerate(inlier_data):
        p_fst[0, i] = x1
        p_fst[1, i] = y1
        p_fst[2, i] = 1.0
        p_snd[0, i] = x2
        p_snd[1, i] = y2
        p_snd[2, i] = 1.0

    # Add random outliers
    random.seed(42)  # For reproducibility
    for i in range(num_outliers):
        idx = num_inliers + i
        p_fst[0, idx] = random.uniform(0, 500)
        p_fst[1, idx] = random.uniform(0, 500)
        p_fst[2, idx] = 1.0
        p_snd[0, idx] = random.uniform(0, 500)
        p_snd[1, idx] = random.uniform(0, 500)
        p_snd[2, idx] = 1.0

    return p_fst, p_snd


def test_close_mat(casename: str, expected: np.ndarray, actual: np.ndarray, eps: float = 1e-3) -> bool:
    """Test if two matrices are close."""
    if np.sum(np.abs(expected - actual)) > eps:
        print("Wrong or inaccurate calculations!")
        if casename:
            print(f"In matrix {casename}!")
        print(f"\nExpected:\n{expected}\nGiven:\n{actual}")
        return False
    return True


def test_mat_size(casename: str, mat: np.ndarray, rows: int, cols: int) -> bool:
    """Test matrix dimensions."""
    if mat.shape[0] != rows or mat.shape[1] != cols:
        print(f"\n{casename}: fail")
        print(f"\nExpected: ({rows}, {cols})\nGiven: {mat.shape}")
        return False
    return True


def test_get_condition_2d() -> bool:
    """Test getCondition2D function."""
    print("===============================")
    print("Pcv4::get_condition_2d(..):")
    
    try:
        p = np.array([
            [18.5, 99.1, 13.8, 242.1, 151.1, 243.1],
            [46.8, 146.5, 221.8, 52.5, 147.1, 224.5],
            [1, 1, 1, 1, 1, 1]
        ], dtype=np.float32)

        Ttrue = np.array([
            [0.011883541, 0, -1.5204991],
            [0, 0.016626639, -2.3255126],
            [0, 0, 1]
        ], dtype=np.float32)

        Test = get_condition_2d(convert_points_old_to_new(p))
        
        if abs(Test[2, 2]) < 1e-4:
            print("Warning: There seems to be a problem with get_condition_2d(..)!")
            print("\t==> Expected T(2,2) to be nonzero!")
            return False
        
        Test = Test / Test[2, 2]
        
        if not test_close_mat("", Ttrue, Test):
            print("Warning: There seems to be a problem with get_condition_2d(..)!")
            input("Press enter to continue...")
            return False
            
        return True
        
    except Exception as exc:
        print(exc)
        return False


def test_get_fundamental_matrix() -> bool:
    """Test getFundamentalMatrix function."""
    print("===============================")
    print("Pcv4::get_fundamental_matrix(..):")
    
    try:
        Ftrue = np.array([
            [6.4590546e-07, -0.00014758465, 0.015314385],
            [0.00015971341, -2.0858946e-05, -0.039460059],
            [-0.016546328, 0.031596929, 1]
        ], dtype=np.float32)

        p_fst, p_snd = get_fake_points()
        F = get_fundamental_matrix(convert_points_old_to_new(p_fst), convert_points_old_to_new(p_snd))
        
        if abs(F[2, 2]) < 1e-4:
            print("Warning: There seems to be a problem with get_fundamental_matrix(..)!")
            print("\t==> Expected F(2,2) to be nonzero!")
            return False
        
        F = F / F[2, 2]
        Ftrue = Ftrue / Ftrue[2, 2]
        
        return test_close_mat("", Ftrue, F)
        
    except Exception as exc:
        print(exc)
        return False


def test_get_design_matrix_fundamental() -> bool:
    """Test getDesignMatrix_fundamental function."""
    print("===============================")
    print("Pcv4::get_design_matrix_fundamental(..):")
    
    try:
        p1 = np.array([
            [-1.8596188, 0.19237423, 1.2876947, -1.4020798, -0.1958406, 1.1074524, 1.4124782, -0.54246116],
            [-1.5204918, -1.4549181, -0.60245907, -0.4221313, 0.069671988, 0.69262278, 1.3647538, 1.8729507],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ], dtype=np.float32)
        
        p2 = np.array([
            [-1.64, 0.35679984, 1.2015998, -1.1791999, -0.42400002, 1.112, 1.3295999, -0.7568],
            [-1.5194274, -1.4539878, -0.60327208, -0.4233129, 0.083844543, 0.67280149, 1.3762779, 1.8670754],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ], dtype=np.float32)

        Fest = get_design_matrix_fundamental(convert_points_old_to_new(p1), convert_points_old_to_new(p2))
        
        # Check dimensions (can be 8x9 or 9x9)
        if not (test_mat_size("Wrong dimensions!", Fest, 8, 9) or test_mat_size("Wrong dimensions!", Fest, 9, 9)):
            return False

        if Fest.shape[0] == 8:
            Ftrue = np.array([
                [3.0497749, 2.4936066, -1.64, 2.8255558, 2.310277, -1.5194274, -1.8596188, -1.5204918, 1],
                [0.068639092, -0.51911455, 0.35679984, -0.27970979, 2.1154332, -1.4539878, 0.19237423, -1.4549181, 1],
                [1.5472938, -0.72391474, 1.2015998, -0.77683026, 0.36344674, -0.60327208, 1.2876947, -0.60245907, 1],
                [1.6533325, 0.49777719, -1.1791999, 0.5935185, 0.17869362, -0.4233129, -1.4020798, -0.4221313, 1],
                [0.083036415, -0.029540924, -0.42400002, -0.016420165, 0.0058416161, 0.083844543, -0.1958406, 0.069671988, 1],
                [1.231487, 0.7701965, 1.112, 0.74509561, 0.46599764, 0.67280149, 1.1074524, 0.69262278, 1],
                [1.8780308, 1.8145765, 1.3295999, 1.9439626, 1.8782806, 1.3762779, 1.4124782, 1.3647538, 1],
                [0.41053459, -1.4174491, -0.7568, -1.012816, 3.4969401, 1.8670754, -0.54246116, 1.8729507, 1]
            ], dtype=np.float32)
        else:
            Ftrue = np.array([
                [3.0497749, 2.4936066, -1.64, 2.8255558, 2.310277, -1.5194274, -1.8596188, -1.5204918, 1],
                [0.068639092, -0.51911455, 0.35679984, -0.27970979, 2.1154332, -1.4539878, 0.19237423, -1.4549181, 1],
                [1.5472938, -0.72391474, 1.2015998, -0.77683026, 0.36344674, -0.60327208, 1.2876947, -0.60245907, 1],
                [1.6533325, 0.49777719, -1.1791999, 0.5935185, 0.17869362, -0.4233129, -1.4020798, -0.4221313, 1],
                [0.083036415, -0.029540924, -0.42400002, -0.016420165, 0.0058416161, 0.083844543, -0.1958406, 0.069671988, 1],
                [1.231487, 0.7701965, 1.112, 0.74509561, 0.46599764, 0.67280149, 1.1074524, 0.69262278, 1],
                [1.8780308, 1.8145765, 1.3295999, 1.9439626, 1.8782806, 1.3762779, 1.4124782, 1.3647538, 1],
                [0.41053459, -1.4174491, -0.7568, -1.012816, 3.4969401, 1.8670754, -0.54246116, 1.8729507, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.float32)

        return test_close_mat("", Ftrue, Fest)
        
    except Exception as exc:
        print(exc)
        return False


def test_solve_dlt() -> bool:
    """Test solve_dlt_fundamental function."""
    print("===============================")
    print("Pcv4::solve_dlt_fundamental(..):")
    
    try:
        # Note: The C++ test uses Mat_<float> A(8, 9) which creates 8 rows x 9 cols
        # The 9th row of data in C++ is ignored - we match that behavior here
        A = np.array([
            [0.55613172, 0.36151901, -0.11716299, 0.42143318, 0.50170219, -0.081145406, -0.14141218, -0.081358112, 0.28945163],
            [0.037880164, -0.20165631, 0.27597162, -0.17037115, 0.45962757, 0.49632433, 0.33699739, 0.49706647, 0.18567577],
            [-0.15503874, -0.42786688, -0.46612, -0.3501814, 0.46457446, -0.081916817, -0.46723419, -0.081672095, 0.074027866],
            [0.13710688, -0.26806444, 0.39586496, -0.27092993, 0.16518487, -0.46980408, 0.37809196, -0.46927336, 0.2608797],
            [0.59673834, -0.2240283, -0.15800957, -0.28392494, -0.5223974, 0.13491097, -0.12557232, 0.13693732, 0.40313032],
            [0.53824687, -0.15117355, 0.070677012, -0.15590209, 0.12148597, -0.008281284, 0.0014170756, -0.00044638285, -0.80206388],
            [-0.021527633, -0.058972765, 0.70984244, 0.040031876, -0.014713437, -0.0036024868, -0.69528753, 0.067997277, 0.047980171],
            [-0.010421497, 0.45400813, -0.008048675, -0.44800973, 0.050085664, -0.54432148, -0.016960399, 0.54209197, 0.0066823587],
        ], dtype=np.float32)

        Fest = solve_dlt_fundamental(A)
        
        if abs(Fest[2, 2]) < 1e-4:
            print("Warning: There seems to be a problem with solve_dlt_fundamental(..)!")
            print("\t==> Expected F(2,2) to be nonzero!")
            print(Fest)
            return False
        
        Fest = Fest / Fest[2, 2]
        
        Ftrue = np.array([
            [0.0083019603, -0.53950614, -0.047245972],
            [0.53861266, -0.059489254, -0.45286086],
            [0.075440452, 0.44964278, -0.0060508098]
        ], dtype=np.float32)
        Ftrue = Ftrue / Ftrue[2, 2]

        return test_close_mat("", Ftrue, Fest)
        
    except Exception as exc:
        print(exc)
        return False


def test_decondition() -> bool:
    """Test decondition_fundamental function."""
    print("===============================")
    print("Pcv4::decondition_fundamental(..):")
    
    try:
        H = np.array([
            [0.0027884692, -0.53886771, -0.053913236],
            [0.53946984, -0.059588462, -0.45182425],
            [0.068957359, 0.45039368, -0.01389052]
        ], dtype=np.float32)
        
        T1 = np.array([
            [0.013864818, 0, -2.7885616],
            [0, 0.016393442, -1.8155738],
            [0, 0, 1]
        ], dtype=np.float32)
        
        T2 = np.array([
            [0.0128, 0, -1.704],
            [0, 0.016359918, -1.813906],
            [0, 0, 1]
        ], dtype=np.float32)

        H_decond = decondition_fundamental(T1, T2, H)
        
        if abs(H_decond[2, 2]) < 1e-4:
            print("Warning: There seems to be a problem with decondition_fundamental(..)!")
            print("\t==> Expected F(2,2) to be nonzero!")
            return False
        
        H_decond = H_decond / H_decond[2, 2]
        
        Htrue = np.array([
            [6.4590546e-07, -0.00014758465, 0.015314385],
            [0.00015971341, -2.0858946e-05, -0.039460059],
            [-0.016546328, 0.031596929, 1]
        ], dtype=np.float32)

        return test_close_mat("", Htrue, H_decond)
        
    except Exception as exc:
        print(exc)
        return False


def test_force_singularity() -> bool:
    """Test forceSingularity function."""
    print("===============================")
    print("Pcv4::force_singularity(..):")
    
    try:
        Fsest = np.array([
            [0.0083019603, -0.53950614, -0.047245972],
            [0.53861266, -0.059489254, -0.45286086],
            [0.075440452, 0.44964278, -0.0060508098]
        ], dtype=np.float32)

        Fsest = force_singularity(Fsest)
        
        Fstrue = np.array([
            [0.0027884692, -0.53886771, -0.053913236],
            [0.53946984, -0.059588462, -0.45182425],
            [0.068957359, 0.45039368, -0.01389052]
        ], dtype=np.float32)

        return test_close_mat("", Fstrue, Fsest)
        
    except Exception as exc:
        print(exc)
        return False


def test_get_error() -> bool:
    """Test getError function."""
    print("===============================")
    print("Pcv4::get_error(..):")
    
    try:
        Ftrue = np.array([
            [0.18009815, 0.84612828, -124.47226],
            [0.51897198, 0.75658411, -182.07408],
            [0.00088265416, 0.0073684035, -0.94836563]
        ], dtype=np.float32)
        
        er_true = 3983.8915033623125

        p_fst, p_snd = get_fake_points()
        er_est = get_error_multiple(convert_points_old_to_new(p_fst), convert_points_old_to_new(p_snd), Ftrue)

        eps = 1e-3
        if abs(er_est - er_true) > eps:
            print("Wrong or inaccurate calculations!")
            print("In value \"Error\"!")
            print(f"\nExpected:\n{er_true}\nGiven:\n{er_est}")
            return False
            
        return True
        
    except Exception as exc:
        print(exc)
        return False


def test_count_inliers() -> bool:
    """Test countInliers function."""
    print("===============================")
    print("Pcv4::count_inliers(..):")
    
    try:
        F = np.array([
            [6.4590546e-07, -0.00014758465, 0.015314385],
            [0.00015971341, -2.0858946e-05, -0.039460059],
            [-0.016546328, 0.031596929, 1]
        ], dtype=np.float32)

        p_fst, p_snd = get_fake_points()
        num_inliers = count_inliers(convert_points_old_to_new(p_fst), convert_points_old_to_new(p_snd), F, 1.0)
        true_num_inliers = 5

        if num_inliers != true_num_inliers:
            print("Wrong or inaccurate calculations!")
            print("In value \"Inliers\"!")
            print(f"\nExpected:\n{true_num_inliers}\nGiven:\n{num_inliers}")
            return False

        return True
        
    except Exception as exc:
        print(exc)
        return False


def test_ransac() -> bool:
    """Test estimateFundamentalRANSAC function."""
    print("===============================")
    print("Pcv4::estimate_fundamental_ransac(..):")
    
    try:
        p_fst, p_snd = get_fake_points_with_outliers()
        F = estimate_fundamental_ransac(convert_points_old_to_new(p_fst), convert_points_old_to_new(p_snd), 2000, 1.0)
        num_inliers = count_inliers(convert_points_old_to_new(p_fst), convert_points_old_to_new(p_snd), F, 1.0)
        
        if num_inliers < 70:
            print("The solution that RANSAC finds is not very good (has few inliers)")
            print(f"Got {num_inliers} inliers but expected around 100")
            return False
            
        if num_inliers > 150:
            print("Something weird is going on with the test")
            return False

        return True
        
    except Exception as exc:
        print(exc)
        return False


def main():
    print()
    print("********************")
    print("Testing: Start")

    correct = True
    correct &= test_get_fundamental_matrix()
    correct &= test_get_condition_2d()
    correct &= test_get_design_matrix_fundamental()
    correct &= test_solve_dlt()
    correct &= test_decondition()
    correct &= test_force_singularity()
    correct &= test_get_error()
    correct &= test_count_inliers()
    correct &= test_ransac()

    print("Testing: Done")
    if correct:
        print("Everything seems (!) to be correct.")
    else:
        print("There seem to be problems.")
    print()
    print("********************")
    print()

    print("Press enter to continue...")
    input()

    return 0


if __name__ == "__main__":
    sys.exit(main())
