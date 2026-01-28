# ==============================================================================
# File        : Pcv4.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Estimation of the fundamental matrix from
#               image point correspondences (Exercise 04).
# ==============================================================================

from __future__ import annotations

import random
from typing import List, Tuple

import cv2
import numpy as np

# Import from the local Helper module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Helper import RawOrbMatches, draw_epi_line, extract_raw_orb_matches


# Geometry type constants
GEOM_TYPE_POINT = 0
GEOM_TYPE_LINE = 1


def apply_h_2d(geom_objects: List[np.ndarray], H: np.ndarray, gtype: int) -> List[np.ndarray]:
    """
    Applies a 2D transformation to an array of points or lines.

    @param geom_objects: Array of input objects, each in homogeneous coordinates
    @param H: Matrix representing the transformation (3x3)
    @param gtype: The type of the geometric objects, point or line. All are the same type.
    @return: Array of transformed objects.
    """
    out: List[np.ndarray] = []

    if gtype == GEOM_TYPE_POINT:
        for x in geom_objects:
            x_t = H @ x
            out.append(x_t.astype(np.float32))

    elif gtype == GEOM_TYPE_LINE:
        # lines transform with inverse-transpose
        H_inv_T = np.linalg.inv(H).T
        for l in geom_objects:
            l_t = H_inv_T @ l
            out.append(l_t.astype(np.float32))

    else:
        raise ValueError("Unknown geometry type")

    return out


def get_condition_2d(points_2d: List[np.ndarray]) -> np.ndarray:
    """
    Get the conditioning matrix of given points.

    @param points_2d: The points as list of homogeneous coordinate arrays
    @return: The condition matrix (3x3)
    """
    xs = np.array([p[0] / p[2] for p in points_2d], dtype=np.float32)
    ys = np.array([p[1] / p[2] for p in points_2d], dtype=np.float32)

    cx = float(xs.mean())
    cy = float(ys.mean())

    mean_abs_x = float(np.mean(np.abs(xs - cx)))
    mean_abs_y = float(np.mean(np.abs(ys - cy)))

    # avoid division by zero
    if mean_abs_x < 1e-12:
        mean_abs_x = 1.0
    if mean_abs_y < 1e-12:
        mean_abs_y = 1.0

    sx = 1.0 / mean_abs_x
    sy = 1.0 / mean_abs_y

    T = np.array([
        [sx, 0.0, -sx * cx],
        [0.0, sy, -sy * cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return T


def get_design_matrix_fundamental(p1_conditioned: List[np.ndarray],
                                   p2_conditioned: List[np.ndarray]) -> np.ndarray:
    """
    Define the design matrix as needed to compute fundamental matrix.

    @param p1_conditioned: First set of conditioned points
    @param p2_conditioned: Second set of conditioned points
    @return: The design matrix to be computed
    """
    if len(p1_conditioned) != len(p2_conditioned):
        raise RuntimeError("Point lists must have same length")

    n = len(p1_conditioned)
    A = np.zeros((n, 9), dtype=np.float32)

    for i, (x1, x2) in enumerate(zip(p1_conditioned, p2_conditioned)):
        x = float(x1[0] / x1[2])
        y = float(x1[1] / x1[2])
        xp = float(x2[0] / x2[2])
        yp = float(x2[1] / x2[2])

        # row corresponding to x'^T F x = 0
        A[i, :] = np.array([
            xp * x, xp * y, xp,
            yp * x, yp * y, yp,
            x, y, 1.0
        ], dtype=np.float32)

    return A


def solve_dlt_fundamental(A: np.ndarray) -> np.ndarray:
    """
    Solve homogeneous equation system by usage of SVD.

    @param A: The design matrix
    @return: The estimated fundamental matrix (3x3)
    """
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1, :]  # eigenvector for smallest singular value
    F = f.reshape((3, 3)).astype(np.float32)
    return F


def force_singularity(F: np.ndarray) -> np.ndarray:
    """
    Enforce rank of 2 on fundamental matrix.

    @param F: The matrix to be changed (3x3)
    @return: The modified fundamental matrix
    """
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    F2 = (U @ np.diag(S) @ Vt).astype(np.float32)
    return F2


def decondition_fundamental(T1: np.ndarray, T2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Decondition a fundamental matrix that was estimated from conditioned points.

    @param T1: Conditioning matrix of set of 2D image points (3x3)
    @param T2: Conditioning matrix of set of 2D image points (3x3)
    @param F: Conditioned fundamental matrix that has to be un-conditioned (3x3)
    @return: Un-conditioned fundamental matrix
    """
    # F = T2^T * Fc * T1
    return (T2.T @ F @ T1).astype(np.float32)


def get_fundamental_matrix(p1: List[np.ndarray], p2: List[np.ndarray]) -> np.ndarray:
    """
    Compute the fundamental matrix.

    @param p1: First set of points
    @param p2: Second set of points
    @return: The estimated fundamental matrix (3x3)
    """
    if len(p1) < 8 or len(p2) < 8:
        raise RuntimeError("Need at least 8 point correspondences")

    T1 = get_condition_2d(p1)
    T2 = get_condition_2d(p2)

    p1c = apply_h_2d(p1, T1, GEOM_TYPE_POINT)
    p2c = apply_h_2d(p2, T2, GEOM_TYPE_POINT)

    A = get_design_matrix_fundamental(p1c, p2c)
    Fc = solve_dlt_fundamental(A)
    Fc = force_singularity(Fc)

    F = decondition_fundamental(T1, T2, Fc)
    return F.astype(np.float32)


def get_error_single(p1: np.ndarray, p2: np.ndarray, F: np.ndarray) -> float:
    """
    Calculate geometric error of estimated fundamental matrix for a single point pair.
    Implement the "Sampson distance".

    @param p1: First point
    @param p2: Second point
    @param F: Fundamental matrix (3x3)
    @return: Geometric error (Sampson distance)
    """
    x1 = p1.astype(np.float64)
    x2 = p2.astype(np.float64)
    Fd = F.astype(np.float64)

    Fx1 = Fd @ x1
    Ftx2 = Fd.T @ x2

    num = (x2.T @ Fd @ x1) ** 2
    denom = (Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2)

    if denom < 1e-12:
        return float("inf")

    return float(num / denom)


def get_error_multiple(p1: List[np.ndarray], p2: List[np.ndarray], F: np.ndarray) -> float:
    """
    Calculate geometric error of estimated fundamental matrix for a set of point pairs.
    Implement the mean "Sampson distance".

    @param p1: First set of points
    @param p2: Second set of points
    @param F: Fundamental matrix (3x3)
    @return: Mean geometric error
    """
    if len(p1) == 0:
        return 0.0
    errs = [get_error_single(a, b, F) for a, b in zip(p1, p2)]
    return float(np.mean(errs))


def count_inliers(p1: List[np.ndarray], p2: List[np.ndarray],
                  F: np.ndarray, threshold: float) -> int:
    """
    Count the number of inliers of an estimated fundamental matrix.

    @param p1: First set of points
    @param p2: Second set of points
    @param F: Fundamental matrix (3x3)
    @param threshold: Maximal "Sampson distance" to still be counted as an inlier
    @return: Number of inliers
    """
    cnt = 0
    for a, b in zip(p1, p2):
        if get_error_single(a, b, F) < threshold:
            cnt += 1
    return cnt


def estimate_fundamental_ransac(p1: List[np.ndarray], p2: List[np.ndarray],
                                 num_iterations: int, threshold: float) -> np.ndarray:
    """
    Estimate the fundamental matrix robustly using RANSAC.
    Use the number of inliers as the score.

    @param p1: First set of points
    @param p2: Second set of points
    @param num_iterations: How many subsets are to be evaluated
    @param threshold: Maximal "Sampson distance" to still be counted as an inlier
    @return: The fundamental matrix (3x3)
    """
    subset_size = 8

    n = len(p1)
    if n < subset_size:
        raise RuntimeError("Need at least 8 matches for RANSAC")

    best_F = np.eye(3, dtype=np.float32)
    best_inliers = -1
    best_inlier_mask = None

    for _ in range(num_iterations):
        idxs = random.sample(range(n), subset_size)
        p1_sub = [p1[i] for i in idxs]
        p2_sub = [p2[i] for i in idxs]

        try:
            F_try = get_fundamental_matrix(p1_sub, p2_sub)
        except Exception:
            continue

        inl = 0
        mask = []
        for i in range(n):
            e = get_error_single(p1[i], p2[i], F_try)
            ok = (e < threshold)
            mask.append(ok)
            if ok:
                inl += 1

        if inl > best_inliers:
            best_inliers = inl
            best_F = F_try
            best_inlier_mask = mask

    # refine with all inliers of best model
    if best_inlier_mask is not None and best_inliers >= 8:
        p1_in = [p1[i] for i in range(n) if best_inlier_mask[i]]
        p2_in = [p2[i] for i in range(n) if best_inlier_mask[i]]
        try:
            best_F = get_fundamental_matrix(p1_in, p2_in)
        except Exception:
            pass

    return best_F.astype(np.float32)


def visualize(img1: np.ndarray, img2: np.ndarray,
              p1: List[np.ndarray], p2: List[np.ndarray], F: np.ndarray) -> None:
    """
    Draw points and corresponding epipolar lines into both images.

    @param img1: Structure containing first image
    @param img2: Structure containing second image
    @param p1: First point set (points in first image)
    @param p2: Second point set (points in second image)
    @param F: Fundamental matrix (mapping from point in img1 to lines in img2)
    """
    img1_copy = img1.copy()
    img2_copy = img2.copy()

    for x1, x2 in zip(p1, p2):
        u1 = int(x1[0] / x1[2]); v1 = int(x1[1] / x1[2])
        u2 = int(x2[0] / x2[2]); v2 = int(x2[1] / x2[2])

        cv2.circle(img1_copy, (u1, v1), 2, (0, 255, 0), 2)
        cv2.circle(img2_copy, (u2, v2), 2, (0, 255, 0), 2)

        # epiline in img2 from point in img1: l2 = F x1
        l2 = F @ x1
        draw_epi_line(img2_copy, float(l2[0]), float(l2[1]), float(l2[2]))

        # epiline in img1 from point in img2: l1 = F^T x2
        l1 = F.T @ x2
        draw_epi_line(img1_copy, float(l1[0]), float(l1[1]), float(l1[2]))

    cv2.imshow("Epilines img1", img1_copy)
    cv2.imshow("Epilines img2", img2_copy)
    cv2.waitKey(0)


def filter_matches(raw_orb_matches: RawOrbMatches) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Filters the raw matches.
    Applies cross consistency check and ratio test (ratio of 0.75) and returns the point pairs that pass both.

    @param raw_orb_matches: Structure containing keypoints and raw matches obtained from comparing feature descriptors
    @return: Tuple of (p1, p2) - Points within the first and second image
    """
    p1: List[np.ndarray] = []
    p2: List[np.ndarray] = []

    ratio = 0.75

    for idx1, match in raw_orb_matches.matches_1_2.items():

        # Ratio test
        if match.second_closest_distance <= 1e-12:
            continue
        if match.closest_distance >= ratio * match.second_closest_distance:
            continue

        idx2 = match.closest

        # Cross consistency check
        if idx2 not in raw_orb_matches.matches_2_1:
            continue
        back = raw_orb_matches.matches_2_1[idx2].closest
        if back != idx1:
            continue

        p1.append(raw_orb_matches.keypoints1[idx1])
        p2.append(raw_orb_matches.keypoints2[idx2])

    return p1, p2


def get_points_automatic(img1: np.ndarray, img2: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Computes matches automatically.
    Points will be in homogeneous coordinates.

    @param img1: The first image
    @param img2: The second image
    @return: Tuple of (p1, p2) - Points within the first and second image
    """
    raw = extract_raw_orb_matches(img1, img2)
    p1, p2 = filter_matches(raw)
    return p1, p2


# Export all public functions
__all__ = [
    "GEOM_TYPE_POINT",
    "GEOM_TYPE_LINE",
    "apply_h_2d",
    "get_condition_2d",
    "get_design_matrix_fundamental",
    "solve_dlt_fundamental",
    "force_singularity",
    "decondition_fundamental",
    "get_fundamental_matrix",
    "get_error_single",
    "get_error_multiple",
    "count_inliers",
    "estimate_fundamental_ransac",
    "visualize",
    "filter_matches",
    "get_points_automatic",
]
