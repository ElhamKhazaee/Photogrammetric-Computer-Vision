# ==============================================================================
# File        : Pcv5.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Exercise 05 – Structure from Motion (SfM).
# Core implementation of scene reconstruction, camera pose estimation,
# track handling, and bundle adjustment.
# ==============================================================================

import math
import random
from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy as np

from Helper import (
    OptimizationProblem,
    hom2eucl,
    rotationMatrixX,
    rotationMatrixY,
    rotationMatrixZ,
    translationMatrix,
)


GEOM_TYPE_POINT = 0
GEOM_TYPE_LINE = 1


@dataclass
class ProjectionMatrixInterpretation:
    principalDistance: float = 0.0
    skew: float = 0.0
    aspectRatio: float = 0.0
    principalPoint: np.ndarray | None = None
    omega: float = 0.0
    phi: float = 0.0
    kappa: float = 0.0
    cameraLocation: np.ndarray | None = None


# functions to be implemented
# --> please edit ONLY these functions!


def applyH_2D(geomObjects, H, gtype):
    out = []
    H = np.asarray(H, dtype=np.float32)

    if gtype == GEOM_TYPE_POINT:
        for x in geomObjects:
            x = np.asarray(x, dtype=np.float32).reshape(3)
            out.append((H @ x).astype(np.float32))
    elif gtype == GEOM_TYPE_LINE:
        H_inv_T = np.linalg.inv(H).T
        for l in geomObjects:
            l = np.asarray(l, dtype=np.float32).reshape(3)
            out.append((H_inv_T @ l).astype(np.float32))
    else:
        raise ValueError("Unknown geometry type")

    return out


def getCondition2D(points2D):
    pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in points2D]
    xs = np.array([p[0] / p[2] for p in pts], dtype=np.float64)
    ys = np.array([p[1] / p[2] for p in pts], dtype=np.float64)

    cx = float(xs.mean())
    cy = float(ys.mean())

    mean_abs_x = float(np.mean(np.abs(xs - cx)))
    mean_abs_y = float(np.mean(np.abs(ys - cy)))

    if mean_abs_x < 1e-12:
        mean_abs_x = 1.0
    if mean_abs_y < 1e-12:
        mean_abs_y = 1.0

    sx = 1.0 / mean_abs_x
    sy = 1.0 / mean_abs_y

    T = np.array(
        [[sx, 0.0, -sx * cx],
         [0.0, sy, -sy * cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return T


def applyH_3D_points(geomObjects, H):
    out = []
    H = np.asarray(H, dtype=np.float32)
    for X in geomObjects:
        X = np.asarray(X, dtype=np.float32).reshape(4)
        out.append((H @ X).astype(np.float32))
    return out


def getCondition3D(points3D):
    pts = [np.asarray(p, dtype=np.float64).reshape(4) for p in points3D]
    xs = np.array([p[0] / p[3] for p in pts], dtype=np.float64)
    ys = np.array([p[1] / p[3] for p in pts], dtype=np.float64)
    zs = np.array([p[2] / p[3] for p in pts], dtype=np.float64)

    cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())

    mean_abs_x = float(np.mean(np.abs(xs - cx)))
    mean_abs_y = float(np.mean(np.abs(ys - cy)))
    mean_abs_z = float(np.mean(np.abs(zs - cz)))

    if mean_abs_x < 1e-12:
        mean_abs_x = 1.0
    if mean_abs_y < 1e-12:
        mean_abs_y = 1.0
    if mean_abs_z < 1e-12:
        mean_abs_z = 1.0

    sx = 1.0 / mean_abs_x
    sy = 1.0 / mean_abs_y
    sz = 1.0 / mean_abs_z

    T = np.array(
        [[sx, 0.0, 0.0, -sx * cx],
         [0.0, sy, 0.0, -sy * cy],
         [0.0, 0.0, sz, -sz * cz],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return T



def getDesignMatrix_camera(points2D, points3D):
    if len(points2D) != len(points3D):
        raise RuntimeError("Point lists must have same length")

    n = len(points2D)
    A = np.zeros((2 * n, 12), dtype=np.float32)

    for i, (x, X) in enumerate(zip(points2D, points3D)):
        x = np.asarray(x, dtype=np.float64).reshape(3)
        X = np.asarray(X, dtype=np.float64).reshape(4)

        u = float(x[0] / x[2])
        v = float(x[1] / x[2])

        X1, X2, X3, X4 = X.tolist()

        # Two rows from x × (P X) = 0
        A[2 * i + 0, :] = np.array(
            [X1, X2, X3, X4, 0, 0, 0, 0, -u * X1, -u * X2, -u * X3, -u * X4],
            dtype=np.float32,
        )
        A[2 * i + 1, :] = np.array(
            [0, 0, 0, 0, X1, X2, X3, X4, -v * X1, -v * X2, -v * X3, -v * X4],
            dtype=np.float32,
        )

    return A


def solve_dlt_camera(A):
    _, _, vt = np.linalg.svd(A)
    p = vt[-1, :]
    P = p.reshape((3, 4)).astype(np.float32)
    return P


def decondition_camera(T_2D, T_3D, P):
    T_2D = np.asarray(T_2D, dtype=np.float32)
    T_3D = np.asarray(T_3D, dtype=np.float32)
    P = np.asarray(P, dtype=np.float32)
    return (np.linalg.inv(T_2D) @ P @ T_3D).astype(np.float32)



def calibrate(points2D, points3D):
    T2 = getCondition2D(points2D)
    T3 = getCondition3D(points3D)

    p2c = applyH_2D(points2D, T2, GEOM_TYPE_POINT)
    p3c = applyH_3D_points(points3D, T3)

    A = getDesignMatrix_camera(p2c, p3c)
    Pc = solve_dlt_camera(A)
    P = decondition_camera(T2, T3, Pc)

    return P.astype(np.float32)



def interprete(P, K, R, info):
    P = np.asarray(P, dtype=np.float32)

    # OpenCV decomposeProjectionMatrix
    _K, _R, _t, _, _, _, _ = cv.decomposeProjectionMatrix(P)
    _K = np.asarray(_K, dtype=np.float32)
    _R = np.asarray(_R, dtype=np.float32)
    _t = np.asarray(_t, dtype=np.float32).reshape(4)

    if abs(_K[2, 2]) > 1e-12:
        _K = _K / _K[2, 2]

    K[:, :] = _K
    R[:, :] = _R

    # Camera center in world coordinates
    C = (_t[:3] / _t[3]).astype(np.float32)

    # Euler angles (photogrammetry convention approx): R = Rz(kappa) Ry(phi) Rx(omega)
    r = _R
    phi = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    cphi = math.cos(phi)
    if abs(cphi) < 1e-8:
        omega = 0.0
        kappa = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    else:
        omega = math.atan2(float(r[2, 1]), float(r[2, 2]))
        kappa = math.atan2(float(r[1, 0]), float(r[0, 0]))

    info.principalDistance = float(_K[0, 0])
    info.skew = float(_K[0, 1])
    info.aspectRatio = float(_K[1, 1] / _K[0, 0]) if abs(_K[0, 0]) > 1e-12 else 0.0
    info.principalPoint = np.array([_K[0, 2], _K[1, 2]], dtype=np.float32)
    info.omega = float(omega)
    info.phi = float(phi)
    info.kappa = float(kappa)
    info.cameraLocation = C



def getDesignMatrix_fundamental(p1_conditioned, p2_conditioned):
    if len(p1_conditioned) != len(p2_conditioned):
        raise RuntimeError("Point lists must have same length")
    n = len(p1_conditioned)
    A = np.zeros((n, 9), dtype=np.float32)

    for i, (x1, x2) in enumerate(zip(p1_conditioned, p2_conditioned)):
        x1 = np.asarray(x1, dtype=np.float64).reshape(3)
        x2 = np.asarray(x2, dtype=np.float64).reshape(3)

        x = float(x1[0] / x1[2])
        y = float(x1[1] / x1[2])
        xp = float(x2[0] / x2[2])
        yp = float(x2[1] / x2[2])

        A[i, :] = np.array(
            [xp * x, xp * y, xp,
             yp * x, yp * y, yp,
             x, y, 1.0],
            dtype=np.float32,
        )
    return A


def solve_dlt_fundamental(A):
    _, _, vt = np.linalg.svd(A)
    f = vt[-1, :]
    return f.reshape((3, 3)).astype(np.float32)


def forceSingularity(F):
    F = np.asarray(F, dtype=np.float32)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    return (U @ np.diag(S) @ Vt).astype(np.float32)


def decondition_fundamental(T1, T2, F):
    T1 = np.asarray(T1, dtype=np.float32)
    T2 = np.asarray(T2, dtype=np.float32)
    F = np.asarray(F, dtype=np.float32)
    return (T2.T @ F @ T1).astype(np.float32)




def getFundamentalMatrix(p1, p2):
    if len(p1) < 8:
        raise RuntimeError("Need at least 8 point correspondences")

    T1 = getCondition2D(p1)
    T2 = getCondition2D(p2)

    p1c = applyH_2D(p1, T1, GEOM_TYPE_POINT)
    p2c = applyH_2D(p2, T2, GEOM_TYPE_POINT)

    A = getDesignMatrix_fundamental(p1c, p2c)
    Fc = solve_dlt_fundamental(A)
    Fc = forceSingularity(Fc)
    F = decondition_fundamental(T1, T2, Fc)

    return F.astype(np.float32)



def getError(p1, p2, F):
    x1 = np.asarray(p1, dtype=np.float64).reshape(3)
    x2 = np.asarray(p2, dtype=np.float64).reshape(3)
    Fd = np.asarray(F, dtype=np.float64).reshape(3, 3)

    Fx1 = Fd @ x1
    Ftx2 = Fd.T @ x2

    num = float((x2.T @ Fd @ x1) ** 2)
    den = float(Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2)
    if den < 1e-12:
        return float("inf")
    return float(num / den)


def getError_set(p1, p2, F):
    if len(p1) == 0:
        return 0.0
    errs = [getError(a, b, F) for a, b in zip(p1, p2)]
    return float(np.mean(errs))


def countInliers(p1, p2, F, threshold):
    cnt = 0
    for a, b in zip(p1, p2):
        if getError(a, b, F) < threshold:
            cnt += 1
    return cnt


def estimateFundamentalRANSAC(p1, p2, numIterations, threshold):
    subsetSize = 8
    n = len(p1)
    if n < subsetSize:
        raise RuntimeError("Need at least 8 matches for RANSAC")

    bestF = np.eye(3, dtype=np.float32)
    bestInliers = -1
    bestMask = None

    for _ in range(int(numIterations)):
        idxs = random.sample(range(n), subsetSize)
        p1s = [p1[i] for i in idxs]
        p2s = [p2[i] for i in idxs]

        try:
            Ftry = getFundamentalMatrix(p1s, p2s)
        except Exception:
            continue

        inl = 0
        mask = []
        for i in range(n):
            e = getError(p1[i], p2[i], Ftry)
            ok = (e < threshold)
            mask.append(ok)
            if ok:
                inl += 1

        if inl > bestInliers:
            bestInliers = inl
            bestF = Ftry
            bestMask = mask

    # refine with all inliers
    if bestMask is not None and bestInliers >= 8:
        p1in = [p1[i] for i in range(n) if bestMask[i]]
        p2in = [p2[i] for i in range(n) if bestMask[i]]
        try:
            bestF = getFundamentalMatrix(p1in, p2in)
        except Exception:
            pass

    return bestF.astype(np.float32)


def computeCameraPose(K, p1, p2):
    # --- make everything float64 + contiguous (OpenCV likes this) ---
    K = np.ascontiguousarray(np.asarray(K, dtype=np.float64))

    F = getFundamentalMatrix(p1, p2)
    F = np.ascontiguousarray(np.asarray(F, dtype=np.float64))

    E = K.T @ F @ K

    # enforce essential constraints (two equal singular values, rank 2)
    U, S, Vt = np.linalg.svd(E)
    s = (S[0] + S[1]) * 0.5
    E = U @ np.diag([s, s, 0.0]) @ Vt
    E = np.ascontiguousarray(np.asarray(E, dtype=np.float64))

    pts1 = np.ascontiguousarray(
        np.array([[float(x[0] / x[2]), float(x[1] / x[2])] for x in p1], dtype=np.float64)
    )
    pts2 = np.ascontiguousarray(
        np.array([[float(x[0] / x[2]), float(x[1] / x[2])] for x in p2], dtype=np.float64)
    )

    # recover pose
    _, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

    H = np.eye(4, dtype=np.float32)
    H[:3, :3] = np.asarray(R, dtype=np.float32)
    H[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    return H



# Non-TODO implementations (port from C++)

def estimateProjectionRANSAC(points2D, points3D, numIterations, threshold):
    subsetSize = 6
    rng = random.Random()
    bestP = np.zeros((3, 4), dtype=np.float32)
    bestInliers = 0

    points2D_subset = [None] * subsetSize
    points3D_subset = [None] * subsetSize
    for _ in range(numIterations):
        for j in range(subsetSize):
            index = rng.randrange(len(points2D))
            points2D_subset[j] = points2D[index]
            points3D_subset[j] = points3D[index]

        P = calibrate(points2D_subset, points3D_subset)

        numInliers = 0
        for i in range(len(points2D)):
            projected = P @ points3D[i]
            if projected[2] > 0.0:
                if (abs(points2D[i][0] - projected[0] / projected[2]) < threshold and
                    abs(points2D[i][1] - projected[1] / projected[2]) < threshold):
                    numInliers += 1

        if numInliers > bestInliers:
            bestInliers = numInliers
            bestP = P

    return bestP


def linearTriangulation(P1, P2, x1, x2):
    A = np.zeros((4, 4), dtype=np.float32)
    A[0, :] = x1[0] * P1[2, :] - P1[0, :]
    A[1, :] = x1[1] * P1[2, :] - P1[1, :]
    A[2, :] = x2[0] * P2[2, :] - P2[0, :]
    A[3, :] = x2[1] * P2[2, :] - P2[1, :]

    _, _, vt = np.linalg.svd(A)
    tmp = vt[-1, :]
    return tmp.astype(np.float32)


def linearTriangulation_set(P1, P2, x1, x2):
    result = []
    for i in range(len(x1)):
        result.append(linearTriangulation(P1, P2, x1[i], x2[i]))
    return result


class NumUpdateParams:
    TRACK = 4
    CAMERA = 6
    INTERNAL_CALIB = 3


@dataclass
class KeyPoint:
    location: np.ndarray
    trackIdx: int
    weight: float


@dataclass
class Camera:
    internalCalibIdx: int
    keypoints: List[KeyPoint]


@dataclass
class Scene:
    cameras: List[Camera]
    numTracks: int
    numInternalCalibs: int

def _project_point(K, H, X4):
    """Project homogeneous 3D point X4 with P = K [I|0] H, return (u,v) or None."""
    P = (K @ (np.eye(3, 4, dtype=np.float32) @ H)).astype(np.float32)
    x = P @ X4
    z = float(x[2])
    if abs(z) < 1e-12:
        return None
    return np.array([float(x[0] / z), float(x[1] / z)], dtype=np.float32)


def _apply_internal_update(K, du):
    """du = [df, dcx, dcy]"""
    K2 = np.array(K, dtype=np.float32, copy=True)
    df, dcx, dcy = float(du[0]), float(du[1]), float(du[2])
    K2[0, 0] += df
    K2[1, 1] += df
    K2[0, 2] += dcx
    K2[1, 2] += dcy
    return K2


def _apply_camera_update(H, du):
    """du = [domega, dphi, dkappa, dtx, dty, dtz] with small increments."""
    domega, dphi, dkappa = float(du[0]), float(du[1]), float(du[2])
    dtx, dty, dtz = float(du[3]), float(du[4]), float(du[5])

    Rinc = (rotationMatrixZ(dkappa) @ rotationMatrixY(dphi) @ rotationMatrixX(domega)).astype(np.float32)
    Tinc = translationMatrix(dtx, dty, dtz).astype(np.float32)

    Hinc = (Rinc @ Tinc).astype(np.float32)
    return (Hinc @ H).astype(np.float32)

class BundleAdjustment(OptimizationProblem):
    class BAJacobiMatrix(OptimizationProblem.JacobiMatrix):
        class RowBlock:
            def __init__(self):
                self.internalCalibIdx = 0
                self.cameraIdx = 0
                self.keypointIdx = 0
                self.trackIdx = 0
                self.J_internalCalib = np.zeros((2, NumUpdateParams.INTERNAL_CALIB), dtype=np.float32)
                self.J_camera = np.zeros((2, NumUpdateParams.CAMERA), dtype=np.float32)
                self.J_track = np.zeros((2, NumUpdateParams.TRACK), dtype=np.float32)

        def __init__(self, scene):
            numResidualPairs = 0
            for camera in scene.cameras:
                numResidualPairs += len(camera.keypoints)

            self.m_rows = []
            for camIdx, camera in enumerate(scene.cameras):
                for kpIdx, kp in enumerate(camera.keypoints):
                    row = BundleAdjustment.BAJacobiMatrix.RowBlock()
                    row.internalCalibIdx = camera.internalCalibIdx
                    row.cameraIdx = camIdx
                    row.keypointIdx = kpIdx
                    row.trackIdx = kp.trackIdx
                    self.m_rows.append(row)

            self.m_internalCalibOffset = 0
            self.m_cameraOffset = self.m_internalCalibOffset + scene.numInternalCalibs * NumUpdateParams.INTERNAL_CALIB
            self.m_trackOffset = self.m_cameraOffset + len(scene.cameras) * NumUpdateParams.CAMERA
            self.m_totalUpdateParams = self.m_trackOffset + scene.numTracks * NumUpdateParams.TRACK

        def multiply(self, dst, src):
            for r, row in enumerate(self.m_rows):
                sumX = 0.0
                sumY = 0.0
                for i in range(NumUpdateParams.INTERNAL_CALIB):
                    idx = self.m_internalCalibOffset + row.internalCalibIdx * NumUpdateParams.INTERNAL_CALIB + i
                    sumX += src[idx] * row.J_internalCalib[0, i]
                    sumY += src[idx] * row.J_internalCalib[1, i]
                for i in range(NumUpdateParams.CAMERA):
                    idx = self.m_cameraOffset + row.cameraIdx * NumUpdateParams.CAMERA + i
                    sumX += src[idx] * row.J_camera[0, i]
                    sumY += src[idx] * row.J_camera[1, i]
                for i in range(NumUpdateParams.TRACK):
                    idx = self.m_trackOffset + row.trackIdx * NumUpdateParams.TRACK + i
                    sumX += src[idx] * row.J_track[0, i]
                    sumY += src[idx] * row.J_track[1, i]
                dst[r * 2 + 0] = sumX
                dst[r * 2 + 1] = sumY

        def transposedMultiply(self, dst, src):
            dst[:] = 0.0
            for r, row in enumerate(self.m_rows):
                for i in range(NumUpdateParams.INTERNAL_CALIB):
                    idx = self.m_internalCalibOffset + row.internalCalibIdx * NumUpdateParams.INTERNAL_CALIB + i
                    dst[idx] += src[r * 2 + 0] * row.J_internalCalib[0, i]
                    dst[idx] += src[r * 2 + 1] * row.J_internalCalib[1, i]
                for i in range(NumUpdateParams.CAMERA):
                    idx = self.m_cameraOffset + row.cameraIdx * NumUpdateParams.CAMERA + i
                    dst[idx] += src[r * 2 + 0] * row.J_camera[0, i]
                    dst[idx] += src[r * 2 + 1] * row.J_camera[1, i]
                for i in range(NumUpdateParams.TRACK):
                    idx = self.m_trackOffset + row.trackIdx * NumUpdateParams.TRACK + i
                    dst[idx] += src[r * 2 + 0] * row.J_track[0, i]
                    dst[idx] += src[r * 2 + 1] * row.J_track[1, i]

        def computeDiagJtJ(self, dst):
            dst[:] = 0.0
            for r, row in enumerate(self.m_rows):
                for i in range(NumUpdateParams.INTERNAL_CALIB):
                    idx = self.m_internalCalibOffset + row.internalCalibIdx * NumUpdateParams.INTERNAL_CALIB + i
                    dst[idx] += row.J_internalCalib[0, i] ** 2 + row.J_internalCalib[1, i] ** 2
                for i in range(NumUpdateParams.CAMERA):
                    idx = self.m_cameraOffset + row.cameraIdx * NumUpdateParams.CAMERA + i
                    dst[idx] += row.J_camera[0, i] ** 2 + row.J_camera[1, i] ** 2
                for i in range(NumUpdateParams.TRACK):
                    idx = self.m_trackOffset + row.trackIdx * NumUpdateParams.TRACK + i
                    dst[idx] += row.J_track[0, i] ** 2 + row.J_track[1, i] ** 2

    class BAState(OptimizationProblem.State):
        @dataclass
        class TrackState:
            location: np.ndarray

        @dataclass
        class CameraState:
            H: np.ndarray

        @dataclass
        class InternalCalibrationState:
            K: np.ndarray

        def __init__(self, scene):
            self.m_scene = scene
            self.m_tracks = [BundleAdjustment.BAState.TrackState(np.zeros(4, dtype=np.float32)) for _ in range(scene.numTracks)]
            self.m_cameras = [BundleAdjustment.BAState.CameraState(np.eye(4, dtype=np.float32)) for _ in range(len(scene.cameras))]
            self.m_internalCalibs = [BundleAdjustment.BAState.InternalCalibrationState(np.eye(3, dtype=np.float32)) for _ in range(scene.numInternalCalibs)]

        def clone(self):
            return BundleAdjustment.BAState(self.m_scene)

        def computeResiduals(self, residuals):
            residuals[:] = 0.0
            ridx = 0

            for camIdx, cam in enumerate(self.m_scene.cameras):
                K = self.m_internalCalibs[cam.internalCalibIdx].K
                H = self.m_cameras[camIdx].H

                for kp in cam.keypoints:
                    X = self.m_tracks[kp.trackIdx].location
                    uv = _project_point(K, H, X)
                    if uv is None:
                        residuals[ridx + 0] = 0.0
                        residuals[ridx + 1] = 0.0
                    else:
                        residuals[ridx + 0] = (kp.location[0] - uv[0]) * kp.weight
                        residuals[ridx + 1] = (kp.location[1] - uv[1]) * kp.weight
                    ridx += 2

        def computeJacobiMatrix(self, dst):
            # Numerical Jacobian (finite differences) per residual block
            eps = 5e-4

            for r, row in enumerate(dst.m_rows):
                camIdx = row.cameraIdx
                kpIdx = row.keypointIdx
                trackIdx = row.trackIdx
                icIdx = row.internalCalibIdx

                cam = self.m_scene.cameras[camIdx]
                kp = cam.keypoints[kpIdx]

                K0 = self.m_internalCalibs[icIdx].K
                H0 = self.m_cameras[camIdx].H
                X0 = self.m_tracks[trackIdx].location

                base = _project_point(K0, H0, X0)
                if base is None:
                    base_uv = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    base_uv = base

                # residual is (proj - obs) * weight
                def residual_uv(K, H, X):
                    uv = _project_point(K, H, X)
                    if uv is None:
                        uv = np.array([0.0, 0.0], dtype=np.float32)
                    res = (uv - kp.location) * kp.weight
                    return res

                r0 = residual_uv(K0, H0, X0)

                # internal calib params (3)
                for i in range(NumUpdateParams.INTERNAL_CALIB):
                    du = np.zeros((3,), dtype=np.float32)
                    du[i] = eps
                    Kp = _apply_internal_update(K0, du)
                    rp = residual_uv(Kp, H0, X0)
                    row.J_internalCalib[:, i] = ((rp - r0) / eps).astype(np.float32)

                # camera params (6)
                for i in range(NumUpdateParams.CAMERA):
                    du = np.zeros((6,), dtype=np.float32)
                    du[i] = eps
                    Hp = _apply_camera_update(H0, du)
                    rp = residual_uv(K0, Hp, X0)
                    row.J_camera[:, i] = ((rp - r0) / eps).astype(np.float32)

                # track params (4)
                for i in range(NumUpdateParams.TRACK):
                    du = np.zeros((4,), dtype=np.float32)
                    du[i] = eps
                    Xp = (X0 + du).astype(np.float32)
                    rp = residual_uv(K0, H0, Xp)
                    row.J_track[:, i] = ((rp - r0) / eps).astype(np.float32)

        def update(self, update, dst):
            # Copy current state to dst then apply increments
            dst.m_internalCalibs = [BundleAdjustment.BAState.InternalCalibrationState(np.array(s.K, copy=True)) for s in
                                    self.m_internalCalibs]
            dst.m_cameras = [BundleAdjustment.BAState.CameraState(np.array(s.H, copy=True)) for s in self.m_cameras]
            dst.m_tracks = [BundleAdjustment.BAState.TrackState(np.array(s.location, copy=True)) for s in self.m_tracks]

            u = np.asarray(update, dtype=np.float32).reshape(-1)

            # Offsets: must match BAJacobiMatrix layout
            internal_off = 0
            camera_off = internal_off + self.m_scene.numInternalCalibs * NumUpdateParams.INTERNAL_CALIB
            track_off = camera_off + len(self.m_scene.cameras) * NumUpdateParams.CAMERA

            # internal calib updates
            for ic in range(self.m_scene.numInternalCalibs):
                du = u[internal_off + ic * NumUpdateParams.INTERNAL_CALIB: internal_off + (
                            ic + 1) * NumUpdateParams.INTERNAL_CALIB]
                dst.m_internalCalibs[ic].K = _apply_internal_update(dst.m_internalCalibs[ic].K, du)

            # camera updates
            for camIdx in range(len(self.m_scene.cameras)):
                du = u[camera_off + camIdx * NumUpdateParams.CAMERA: camera_off + (camIdx + 1) * NumUpdateParams.CAMERA]
                dst.m_cameras[camIdx].H = _apply_camera_update(dst.m_cameras[camIdx].H, du)

            # track updates
            for t in range(self.m_scene.numTracks):
                du = u[track_off + t * NumUpdateParams.TRACK: track_off + (t + 1) * NumUpdateParams.TRACK]
                dst.m_tracks[t].location = (dst.m_tracks[t].location + du).astype(np.float32)

        def weighDownOutliers(self):
            pass

    def __init__(self, scene):
        super().__init__()
        self.m_scene = scene
        self.m_numResiduals = 0
        for camera in self.m_scene.cameras:
            self.m_numResiduals += len(camera.keypoints) * 2

        self.m_numUpdateParameters = (
            self.m_scene.numInternalCalibs * NumUpdateParams.INTERNAL_CALIB
            + len(self.m_scene.cameras) * NumUpdateParams.CAMERA
            + self.m_scene.numTracks * NumUpdateParams.TRACK
        )

    def createJacobiMatrix(self):
        return BundleAdjustment.BAJacobiMatrix(self.m_scene)

    def downweightOutlierKeypoints(self, state):
        residuals = np.zeros((self.m_numResiduals,), dtype=np.float32)
        state.computeResiduals(residuals)

        distances = np.zeros((self.m_numResiduals // 2,), dtype=np.float32)
        residualIdx = 0
        for c in self.m_scene.cameras:
            for _kp in c.keypoints:
                distances[residualIdx // 2] = math.sqrt(
                    residuals[residualIdx] ** 2 + residuals[residualIdx + 1] ** 2
                )
                residualIdx += 2

        sortedDistances = np.sort(distances)
        if sortedDistances.size == 0:
            return
        thresh = sortedDistances[sortedDistances.size * 2 // 3] * 2.0

        residualIdx = 0
        for c in self.m_scene.cameras:
            for kp in c.keypoints:
                if distances[residualIdx] > thresh:
                    kp.weight *= 0.5
                residualIdx += 1

def buildScene(imagesFilenames):
    """Build a multi-view Scene from a list of image filenames.

    The returned Scene contains:
    - one internal calibration (index 0)
    - one Camera per image
    - tracks constructed by connected components over robust pairwise matches
    """
    # Performance knobs: the Python BA implementation is very slow with many tracks
    # (numerical Jacobian), so keep the problem size bounded for student testing.
    threshold = 20.0
    orb_max_features = 3000
    ransac_iterations_F = 250
    min_matches = 150
    max_tracks_for_ba = 400
    min_views_per_track = 2

    class _Image:
        def __init__(self):
            self.keypoints = []
            self.descriptors = None
            self.matches = []  # per keypoint: list[(otherImageIdx, otherKpIdx)]

    all_images = [_Image() for _ in range(len(imagesFilenames))]

    orb = cv.ORB_create()
    orb.setMaxFeatures(int(orb_max_features))
    for i, fn in enumerate(imagesFilenames):
        img = cv.imread(fn)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        kps, desc = orb.detectAndCompute(img, None)
        all_images[i].keypoints = kps
        all_images[i].descriptors = desc
        all_images[i].matches = [[] for _ in range(len(kps))]

    matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
    for i in range(len(all_images)):
        for j in range(i + 1, len(all_images)):
            if all_images[i].descriptors is None or all_images[j].descriptors is None:
                continue
            raw = matcher.knnMatch(all_images[i].descriptors, all_images[j].descriptors, k=2)
            good = []
            for m in raw:
                if len(m) < 2:
                    continue
                if m[0].distance <= m[1].distance * 0.75:
                    good.append(m[0])

            if not good:
                continue

            p1 = [
                np.array(
                    [
                        all_images[i].keypoints[m.queryIdx].pt[0],
                        all_images[i].keypoints[m.queryIdx].pt[1],
                        1.0,
                    ],
                    dtype=np.float32,
                )
                for m in good
            ]
            p2 = [
                np.array(
                    [
                        all_images[j].keypoints[m.trainIdx].pt[0],
                        all_images[j].keypoints[m.trainIdx].pt[1],
                        1.0,
                    ],
                    dtype=np.float32,
                )
                for m in good
            ]

            F = estimateFundamentalRANSAC(p1, p2, int(ransac_iterations_F), threshold)

            inlier_matches = []
            for idx, m in enumerate(good):
                if getError(p1[idx], p2[idx], F) < threshold:
                    inlier_matches.append((m.queryIdx, m.trainIdx))

            if len(inlier_matches) >= min_matches:
                for q, t in inlier_matches:
                    all_images[i].matches[q].append((j, t))
                    all_images[j].matches[t].append((i, q))

    scene = Scene(
        cameras=[Camera(0, []) for _ in range(len(imagesFilenames))],
        numTracks=0,
        numInternalCalibs=1,
    )

    handled = set()  # (imgIdx, kpIdx)
    kp_stack = []
    kp_list = []
    images_spanned = set()

    for i in range(len(all_images)):
        for kp_idx in range(len(all_images[i].keypoints)):
            if not all_images[i].matches[kp_idx]:
                continue
            if (i, kp_idx) in handled:
                continue

            valid = True
            kp_stack.append((i, kp_idx))
            while kp_stack:
                img_kp = kp_stack.pop()

                if img_kp[0] in images_spanned:
                    valid = False

                handled.add(img_kp)
                kp_list.append(img_kp)
                images_spanned.add(img_kp[0])

                for matched in all_images[img_kp[0]].matches[img_kp[1]]:
                    if matched not in handled:
                        kp_stack.append(matched)

            if valid:
                track_idx = scene.numTracks
                for img_id, kp_id in kp_list:
                    pt = all_images[img_id].keypoints[kp_id].pt
                    scene.cameras[img_id].keypoints.append(
                        KeyPoint(
                            location=np.array([pt[0], pt[1]], dtype=np.float32),
                            trackIdx=track_idx,
                            weight=1.0,
                        )
                    )
                scene.numTracks += 1

            kp_list.clear()
            images_spanned.clear()

    for c in scene.cameras:
        if len(c.keypoints) < 100:
            print(
                f"Warning: One camera is connected with only {len(c.keypoints)} keypoints, this might be too unstable!"
            )

    # Prune tracks to keep BA tractable in Python.
    if scene.numTracks > max_tracks_for_ba:
        track_counts = np.zeros((scene.numTracks,), dtype=np.int32)
        for cam in scene.cameras:
            for kp in cam.keypoints:
                if 0 <= kp.trackIdx < scene.numTracks:
                    track_counts[kp.trackIdx] += 1

        keep = [i for i, c in enumerate(track_counts.tolist()) if c >= min_views_per_track]
        keep.sort(key=lambda tid: track_counts[tid], reverse=True)
        keep = keep[:max_tracks_for_ba]

        remap = {old: new for new, old in enumerate(keep)}
        for cam in scene.cameras:
            new_kps = []
            for kp in cam.keypoints:
                new_idx = remap.get(kp.trackIdx)
                if new_idx is None:
                    continue
                new_kps.append(
                    KeyPoint(
                        location=kp.location,
                        trackIdx=int(new_idx),
                        weight=kp.weight,
                    )
                )
            cam.keypoints = new_kps

        old_num = scene.numTracks
        scene.numTracks = len(keep)
        print(f"Pruned tracks for BA: {old_num} -> {scene.numTracks}")

    return scene


def produceInitialState(scene, initialInternalCalib, state):
    """Produce an initial BA state (camera poses + triangulated tracks).

    """
    threshold = 20.0
    ransac_iterations_P = 250
    state.m_internalCalibs[0].K = np.asarray(initialInternalCalib, dtype=np.float32)

    triangulated_points = set()

    image1 = 0
    image2 = 1
    if len(scene.cameras) < 2:
        raise ValueError("Need at least 2 images/cameras to initialize")

    # Find stereo pose of first two images
    track2keypoint = {}
    for kp in scene.cameras[image1].keypoints:
        track2keypoint[kp.trackIdx] = kp.location

    matches = []
    matches2track = []
    for kp in scene.cameras[image2].keypoints:
        if kp.trackIdx in track2keypoint:
            matches.append((track2keypoint[kp.trackIdx], kp.location))
            matches2track.append(kp.trackIdx)

    print(f"Initial pair has {len(matches)} matches")
    p1 = [np.array([m[0][0], m[0][1], 1.0], dtype=np.float32) for m in matches]
    p2 = [np.array([m[1][0], m[1][1], 1.0], dtype=np.float32) for m in matches]

    K = state.m_internalCalibs[0].K
    state.m_cameras[image1].H = np.eye(4, dtype=np.float32)
    state.m_cameras[image2].H = computeCameraPose(K, p1, p2)

    P1 = (K @ np.eye(3, 4, dtype=np.float32)).astype(np.float32)
    P2 = (K @ (np.eye(3, 4, dtype=np.float32) @ state.m_cameras[image2].H)).astype(
        np.float32
    )

    Xs = linearTriangulation_set(P1, P2, p1, p2)
    for i, X in enumerate(Xs):
        t = X.astype(np.float32)
        n = float(np.linalg.norm(t))
        if n > 1e-12:
            t = t / n
        state.m_tracks[matches2track[i]].location = t
        triangulated_points.add(matches2track[i])

    # Estimate remaining cameras using already triangulated tracks
    for c in range(len(scene.cameras)):
        if c in (image1, image2):
            continue

        triangulated_keypoints = [
            kp for kp in scene.cameras[c].keypoints if kp.trackIdx in triangulated_points
        ]
        if len(triangulated_keypoints) < 100:
            print(
                f"Warning: Camera {c} is only estimated from {len(triangulated_keypoints)} keypoints"
            )

        points2D = [
            np.array([kp.location[0], kp.location[1], 1.0], dtype=np.float32)
            for kp in triangulated_keypoints
        ]
        points3D = [state.m_tracks[kp.trackIdx].location for kp in triangulated_keypoints]

        print(f"Estimating camera {c} from {len(triangulated_keypoints)} keypoints")
        P = estimateProjectionRANSAC(points2D, points3D, int(ransac_iterations_P), threshold)

        Ktmp = np.eye(3, dtype=np.float32)
        Rtmp = np.eye(3, dtype=np.float32)
        info = ProjectionMatrixInterpretation()
        interprete(P, Ktmp, Rtmp, info)

        H = np.eye(4, dtype=np.float32)
        H[:3, :3] = Rtmp
        if info.cameraLocation is None:
            cam_loc = np.zeros((3,), dtype=np.float32)
        else:
            cam_loc = np.asarray(info.cameraLocation, dtype=np.float32).reshape(3)
        H = (
            H
            @ translationMatrix(-float(cam_loc[0]), -float(cam_loc[1]), -float(cam_loc[2]))
        ).astype(np.float32)
        state.m_cameras[c].H = H

    # Triangulate remaining tracks
    for c in range(len(scene.cameras)):
        P1 = (
            state.m_internalCalibs[scene.cameras[c].internalCalibIdx].K
            @ (np.eye(3, 4, dtype=np.float32) @ state.m_cameras[c].H)
        ).astype(np.float32)

        for other_c in range(c):
            P2 = (
                state.m_internalCalibs[scene.cameras[other_c].internalCalibIdx].K
                @ (np.eye(3, 4, dtype=np.float32) @ state.m_cameras[other_c].H)
            ).astype(np.float32)
            for kp in scene.cameras[c].keypoints:
                if kp.trackIdx in triangulated_points:
                    continue
                for other_kp in scene.cameras[other_c].keypoints:
                    if kp.trackIdx == other_kp.trackIdx:
                        X = linearTriangulation(
                            P1,
                            P2,
                            np.array([kp.location[0], kp.location[1], 1.0], dtype=np.float32),
                            np.array(
                                [other_kp.location[0], other_kp.location[1], 1.0],
                                dtype=np.float32,
                            ),
                        )
                        n = float(np.linalg.norm(X))
                        if n > 1e-12:
                            X = X / n
                        state.m_tracks[kp.trackIdx].location = X.astype(np.float32)
                        triangulated_points.add(kp.trackIdx)
                        break

    if len(triangulated_points) != len(state.m_tracks):
        print("Warning: Some tracks were not triangulated. This should not happen!")
