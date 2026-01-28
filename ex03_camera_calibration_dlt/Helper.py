# ============================================================
# File        : Helper.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Exercise    : 03 – Camera Calibration (DLT)
# Description : Helper utilities for camera calibration using
#               Direct Linear Transformation (DLT), including
#               data handling and projection matrix interpretation.
# ============================================================

"""Helper utilities mirroring the C++ Helper module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ProjectionMatrixInterpretation:
    """Container mirroring the C++ struct for camera parameters."""

    principalDistance: float = 0.0
    skew: float = 90.0
    aspectRatio: float = 1.0
    principalPoint: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    omega: float = 0.0
    phi: float = 0.0
    kappa: float = 0.0
    cameraLocation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __str__(self) -> str:
        return (
            f"principal distance: {self.principalDistance:.6f}\n"
            f"skew (deg): {self.skew:.6f}\n"
            f"aspect ratio: {self.aspectRatio:.6f}\n"
            f"principal point: ({self.principalPoint[0]:.6f}, {self.principalPoint[1]:.6f})\n"
            f"omega (deg): {self.omega:.6f}\n"
            f"phi (deg): {self.phi:.6f}\n"
            f"kappa (deg): {self.kappa:.6f}\n"
            f"camera location: ({self.cameraLocation[0]:.6f}, {self.cameraLocation[1]:.6f}, {self.cameraLocation[2]:.6f})\n"
        )


def _load_object_points(path: Path) -> List[np.ndarray]:
    """Read 3D points from a whitespace-separated file."""
    pts: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = [float(v) for v in line.strip().split()[:3]]
            pts.append(np.array([parts[0], parts[1], parts[2], 1.0], dtype=np.float32))
    if not pts:
        raise ValueError(f"No object points found in {path}")
    return pts


def _mouse_callback(event: int, x: int, y: int, _flags: int, data: List[np.ndarray]) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data[0], (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("Calibration Image", data[0])
        data[1].append(np.array([x + 0.5, y + 0.5, 1.0], dtype=np.float32))


def get_points(calib_img: np.ndarray, object_point_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Interactively collect 2D image points and load matching 3D object points."""
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for interactive point collection")
    pts3d = _load_object_points(Path(object_point_file))
    collected: List[np.ndarray] = []
    display = calib_img.copy()
    cv2.namedWindow("Calibration Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibration Image", display)
    cv2.setMouseCallback("Calibration Image", _mouse_callback, [display, collected])
    print(
        "Click the image points in the same order as the 3D points listed in the file.\n"
        "Press ESC when you are done."
    )
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()
    if len(collected) != len(pts3d):
        raise ValueError(
            f"Collected {len(collected)} image points but {len(pts3d)} object points are defined."
        )
    return collected, pts3d


__all__ = [
    "ProjectionMatrixInterpretation",
    "get_points",
]
