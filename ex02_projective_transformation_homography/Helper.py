# ============================================================
# File        : Helper.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Helper functions for Exercise 02
#               (I/O, visualization, and supporting math utilities).
# ============================================================
from dataclasses import dataclass, field
import numpy as np
import cv2 as cv
from Pcv2 import eucl2hom_point_2D


@dataclass
class WinInfo:
    img: np.ndarray
    name: str
    pointList: list = field(default_factory=list)


def _getPointsCB(event, x, y, flags, param):
    win: WinInfo = param
    if event == cv.EVENT_LBUTTONDOWN:
        p = (x, y)
        cv.circle(win.img, p, 2, (0, 255, 0), 2)
        cv.circle(win.img, p, 15, (0, 255, 0), 2)
        cv.imshow(win.name, win.img)
        win.pointList.append(eucl2hom_point_2D((x + 0.5, y + 0.5)))


def getPoints(baseImg: np.ndarray, attachImg: np.ndarray):
    print()
    print("Please select at least four points by clicking at the corresponding image positions:")
    print("Firstly click at the point that shall be transformed (image to attach), then the corresponding point within the base image")
    print("Continue until you collected as many point pairs as you wish")
    print("Stop the point selection by pressing any key\n")

    winBase = WinInfo(baseImg.copy(), "Base image")
    winAttach = WinInfo(attachImg.copy(), "Image to attach")

    cv.namedWindow(winBase.name, cv.WINDOW_NORMAL)
    cv.imshow(winBase.name, winBase.img)
    cv.setMouseCallback(winBase.name, _getPointsCB, winBase)

    cv.namedWindow(winAttach.name, cv.WINDOW_NORMAL)
    cv.imshow(winAttach.name, winAttach.img)
    cv.setMouseCallback(winAttach.name, _getPointsCB, winAttach)

    cv.waitKey(0)

    cv.destroyWindow(winBase.name)
    cv.destroyWindow(winAttach.name)

    points_base = winBase.pointList
    points_attach = winAttach.pointList
    return len(points_base), np.array(points_base, dtype=np.float32), np.array(points_attach, dtype=np.float32)


def stitch(baseImg: np.ndarray, attachImg: np.ndarray, H: np.ndarray) -> np.ndarray:
    # Compute corners of warped image
    corners = np.array([[[0, 0], [0, attachImg.shape[0]], [attachImg.shape[1], 0], [attachImg.shape[1], attachImg.shape[0]]]], dtype=np.float32)
    corners = cv.perspectiveTransform(corners, H.astype(np.float32))

    x_start = min(corners[0, 0, 0], corners[0, 1, 0], 0.0)
    x_end = max(corners[0, 2, 0], corners[0, 3, 0], float(baseImg.shape[1]))
    y_start = min(corners[0, 0, 1], corners[0, 2, 1], 0.0)
    y_end = max(corners[0, 1, 1], corners[0, 3, 1], float(baseImg.shape[0]))

    T = np.array([[1, 0, -x_start], [0, 1, -y_start], [0, 0, 1]], dtype=np.float32)
    T = T @ H.astype(np.float32)

    panorama = cv.warpPerspective(attachImg, T, (int(x_end - x_start + 1), int(y_end - y_start + 1)), flags=cv.INTER_LINEAR)

    roi = panorama[int(-y_start):int(-y_start + baseImg.shape[0]), int(-x_start):int(-x_start + baseImg.shape[1])]
    roi[...] = baseImg

    return panorama
