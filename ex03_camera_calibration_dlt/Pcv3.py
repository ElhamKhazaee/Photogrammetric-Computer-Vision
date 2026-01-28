# ============================================================
# File        : Pcv3.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Exercise 03 – Camera Calibration (DLT).
#               Core implementation of DLT-based camera calibration
#               (projection matrix estimation) and related routines.
# ============================================================

from __future__ import annotations

from enum import Enum, auto
from typing import List, Sequence

import numpy as np

from Helper import ProjectionMatrixInterpretation


class GeometryType(Enum):
    GEOM_TYPE_POINT = auto()
    GEOM_TYPE_LINE = auto()


def get_condition_2d(points2d: Sequence[np.ndarray]) -> np.ndarray:
    """Get the conditioning matrix for given 2D homogeneous points."""
   
    pts = np.asarray(points2d, dtype=np.float64)  # shape (N, 3)
    # convert to Euclidean (in case w != 1)
    u = pts[:, 0] / pts[:, 2]
    v = pts[:, 1] / pts[:, 2]

    # centroid
    tx = np.mean(u)
    ty = np.mean(v)

    # mean absolute deviation along each axis
    sx = np.mean(np.abs(u - tx))
    sy = np.mean(np.abs(v - ty))

    # avoid division by zero (degenerate case)
    if sx == 0:
        sx = 1.0
    if sy == 0:
        sy = 1.0

    T = np.array(
        [
            [1.0 / sx, 0.0, -tx / sx],
            [0.0, 1.0 / sy, -ty / sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return T


def get_condition_3d(points3d: Sequence[np.ndarray]) -> np.ndarray:
    """Get the conditioning matrix for given 3D homogeneous points."""
   
    pts = np.asarray(points3d, dtype=np.float64)  # shape (N, 4)
    X = pts[:, 0] / pts[:, 3]
    Y = pts[:, 1] / pts[:, 3]
    Z = pts[:, 2] / pts[:, 3]

    tx = np.mean(X)
    ty = np.mean(Y)
    tz = np.mean(Z)

    sx = np.mean(np.abs(X - tx))
    sy = np.mean(np.abs(Y - ty))
    sz = np.mean(np.abs(Z - tz))

    if sx == 0:
        sx = 1.0
    if sy == 0:
        sy = 1.0
    if sz == 0:
        sz = 1.0

    T = np.array(
        [
            [1.0 / sx, 0.0, 0.0, -tx / sx],
            [0.0, 1.0 / sy, 0.0, -ty / sy],
            [0.0, 0.0, 1.0 / sz, -tz / sz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return T


def apply_h_2d(geom_objects: Sequence[np.ndarray], H: np.ndarray, geom_type: GeometryType) -> List[np.ndarray]:
    """Apply a 2D transformation to an array of points or lines."""
    H = np.asarray(H, dtype=np.float64)
    result: List[np.ndarray] = []

    if geom_type == GeometryType.GEOM_TYPE_POINT:
        for p in geom_objects:
            p_arr = np.asarray(p, dtype=np.float64)
            p_new = H @ p_arr
            result.append(p_new)
    elif geom_type == GeometryType.GEOM_TYPE_LINE:
        H_inv_T = np.linalg.inv(H).T
        for l in geom_objects:
            l_arr = np.asarray(l, dtype=np.float64)
            l_new = H_inv_T @ l_arr
            result.append(l_new)
    else:
        raise RuntimeError("Unhandled geometry type in apply_h_2d")

    return result


def apply_h_3d_points(points: Sequence[np.ndarray], H: np.ndarray) -> List[np.ndarray]:
    """Apply a 3D transformation to an array of points."""

    H = np.asarray(H, dtype=np.float64)
    result: List[np.ndarray] = []
    for p in points:
        p_arr = np.asarray(p, dtype=np.float64)
        p_new = H @ p_arr
        result.append(p_new)
    return result

def get_design_matrix_camera(points2d: Sequence[np.ndarray], points3d: Sequence[np.ndarray]) -> np.ndarray:
    """Create the design matrix used for DLT calibration."""

    pts2 = np.asarray(points2d, dtype=np.float64)  # (N, 3)
    pts3 = np.asarray(points3d, dtype=np.float64)  # (N, 4)
    n = pts2.shape[0]

    A = np.zeros((2 * n, 12), dtype=np.float64)

    for i in range(n):
        u, v, w = pts2[i]
        X = pts3[i]  # (4,)

        # first row
        A[2 * i, 0:4] = -w * X
        A[2 * i, 4:8] = 0.0
        A[2 * i, 8:12] = u * X

        # second row
        A[2 * i + 1, 0:4] = 0.0
        A[2 * i + 1, 4:8] = -w * X
        A[2 * i + 1, 8:12] = v * X

    return A


def solve_dlt_camera(A: np.ndarray) -> np.ndarray:
    """Solve the homogeneous system using SVD."""

    A = np.asarray(A, dtype=np.float64)
    # Perform SVD
    U, S, Vt = np.linalg.svd(A)

    p = Vt[-1, :]  # last row of V^T → last column of V
    P = p.reshape(3, 4)
    return P


def decondition_camera(T_2d: np.ndarray, T_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Undo conditioning on the estimated projection matrix."""

    # Inverse of 2D conditioning matrix
    T2D_inv = np.linalg.inv(T_2d)

    # Decondition
    P_real  = T2D_inv @ P @ T_3d

    return P_real 

def calibrate(points2d: Sequence[np.ndarray], points3d: Sequence[np.ndarray]) -> np.ndarray:
    """Estimate the projection matrix from 2D/3D correspondences."""
   
    # 1. conditioning matrices
    T2 = get_condition_2d(points2d)
    T3 = get_condition_3d(points3d)

    # 2. conditioned points
    pts2_norm = apply_h_2d(points2d, T2, GeometryType.GEOM_TYPE_POINT)
    pts3_norm = apply_h_3d_points(points3d, T3)

    # 3. design matrix
    A = get_design_matrix_camera(pts2_norm, pts3_norm)

    # 4. DLT
    P_tilde = solve_dlt_camera(A)

    # 5. decondition
    P = decondition_camera(T2, T3, P_tilde)
    return P


def interprete(P: np.ndarray, info: ProjectionMatrixInterpretation | None = None) -> tuple[np.ndarray, np.ndarray, ProjectionMatrixInterpretation]:
    """Extract internal/external parameters from P."""
   
    if info is None:
        info = ProjectionMatrixInterpretation()
        P = np.asarray(P, dtype=np.float64)

    # Split
    M = P[:, :3]
    p4 = P[:, 3]

    # Camera center C = - M^{-1} p4
    C = -np.linalg.inv(M) @ p4
    info.cameraLocation = C.astype(np.float64)

    # ---- RQ decomposition  ----
   
    # 1) Flip M upside down
    M_flip = np.flipud(M)

    # 2) QR decomposition of transpose
    Q, R = np.linalg.qr(M_flip.T)

    # 3) Undo flips
    K = np.flipud(R.T)
    K = np.fliplr(K)

    Rmat = np.flipud(Q.T)

    # ---- Fix signs so diag(K) > 0 ----
    for i in range(3):
        if K[i, i] < 0:
            K[:, i] *= -1
            Rmat[i, :] *= -1

    # Normalize so K[2,2] = 1
    if K[2, 2] != 0:
        K /= K[2, 2]

    # ---- Intrinsic parameters ----
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    skew_term = K[0, 1]

    info.principalDistance = float(fx)
    info.aspectRatio = float(fy / fx)
    info.principalPoint = np.array([cx, cy], dtype=np.float64)

    if skew_term == 0:
        info.skew = 90.0
    else:
        info.skew = float(abs(np.degrees(np.arctan(fx / skew_term))))

    # ---- Rotation angles (photogrammetry convention) ----
    r11, r12, r13 = Rmat[0, :]
    r21, r22, r23 = Rmat[1, :]
    r31, r32, r33 = Rmat[2, :]

    phi = np.degrees(np.arcsin(Rmat[2, 0]))
    omega = np.degrees(np.arctan2(-Rmat[2, 1], Rmat[2, 2]))
    kappa = np.degrees(np.arctan2(-Rmat[1, 0], Rmat[0, 0]))

    info.omega = float(omega)
    info.phi = float(phi)
    info.kappa = float(kappa)

    return K, Rmat, info



__all__ = [
    "GeometryType",
    "ProjectionMatrixInterpretation",
    "get_condition_2d",
    "get_condition_3d",
    "apply_h_2d",
    "apply_h_3d_points",
    "get_design_matrix_camera",
    "solve_dlt_camera",
    "decondition_camera",
    "calibrate",
    "interprete",
]
