# ============================================================
# File        : main.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Exercise 02 – Projective Transformation (Homography).
#               Demonstrates image rectification and panorama stitching
#               using homography-based projective geometry.
# ============================================================
import sys
import cv2 as cv
import numpy as np
from Pcv2 import homography2D
from Helper import getPoints, stitch


def run(fnameBase: str, fnameLeft: str, fnameRight: str):
    baseImage = cv.imread(fnameBase)
    attachImage = cv.imread(fnameLeft)
    if baseImage is None:
        print(f"ERROR: Cannot read image ({fnameBase})")
        sys.exit(1)
    if attachImage is None:
        print(f"ERROR: Cannot read image ({fnameLeft})")
        sys.exit(1)

    n, p_basis, p_attach = getPoints(baseImage, attachImage)

    print(f"Number of defined point pairs: {n}")
    print("\nPoints in base image:")
    for p in p_basis:
        print(p)
    print("\nPoints in second image:")
    for p in p_attach:
        print(p)

    H = homography2D(p_basis, p_attach)
    panorama = stitch(baseImage, attachImage, H)

    windowName = "Panorama"
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.imshow(windowName, panorama)
    cv.waitKey(0)
    try:
        cv.destroyWindow(windowName)
    except cv.error:
        pass

    baseImage = panorama
    attachImage = cv.imread(fnameRight)
    if attachImage is None:
        print(f"ERROR: Cannot read image ({fnameRight})")
        sys.exit(1)

    n, p_basis, p_attach = getPoints(baseImage, attachImage)

    print(f"Number of defined point pairs: {n}")
    print("\nPoints in base image:")
    for p in p_basis:
        print(p)
    print("\nPoints in second image:")
    for p in p_attach:
        print(p)

    H = homography2D(p_basis, p_attach)
    panorama = stitch(baseImage, attachImage, H)

    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.imshow(windowName, panorama)
    cv.waitKey(0)
    try:
        cv.destroyWindow(windowName)
    except cv.error:
        pass
    cv.imwrite("panorama.png", panorama)


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <path to base image> <path to 2nd image> <path to 3rd image>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
