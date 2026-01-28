# ============================================================
# File        : main.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Exercise 03 – Camera Calibration (DLT).
#               Demonstrates estimation of a camera projection matrix
#               from 2D–3D correspondences and evaluation of reprojection.
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from Helper import ProjectionMatrixInterpretation, get_points
from Pcv3 import calibrate, interprete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCV3 calibration pipeline")
    parser.add_argument("image", type=Path, help="Calibration image path")
    parser.add_argument("points", type=Path, help="Text file containing 3D object points")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not open {args.image}")
    points2d, points3d = get_points(image, str(args.points))
    P = calibrate(points2d, points3d)
    K, R, info = interprete(P, ProjectionMatrixInterpretation())
    print("Calibration matrix K:\n", K)
    print("Rotation matrix R:\n", R)
    print("\nInterpretation:\n", info)


if __name__ == "__main__":
    main()
