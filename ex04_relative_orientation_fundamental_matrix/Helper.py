# ==============================================================================
# File        : Helper.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Helper utilities for feature matching,
#               visualization, and epipolar geometry (Exercise 04).
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class Match:
    """Structure containing match information between keypoints."""
    # Index of the closest match
    closest: int = 0
    # Feature distance of closest match
    closest_distance: float = 0.0
    # Index of the second closest match
    second_closest: int = 0
    # Feature distance of second closest match
    second_closest_distance: float = 0.0


@dataclass
class RawOrbMatches:
    """Structure containing keypoints and raw matches obtained from comparing feature descriptors."""
    # Location of keypoints in first image (homogeneous coordinates)
    keypoints1: List[np.ndarray] = field(default_factory=list)
    # Location of keypoints in second image (homogeneous coordinates)
    keypoints2: List[np.ndarray] = field(default_factory=list)
    # For each keypoint in first image, which keypoints are similar in second image
    # Key is index of keypoint in first image, value is Match struct
    matches_1_2: Dict[int, Match] = field(default_factory=dict)
    # For each keypoint in second image, which keypoints are similar in first image
    matches_2_1: Dict[int, Match] = field(default_factory=dict)


class WinInfo:
    """Structure for window callback data."""
    def __init__(self, img: np.ndarray, name: str):
        self.img = img.copy()
        self.name = name
        self.point_list: List[np.ndarray] = []


def _get_points_callback(event: int, x: int, y: int, flags: int, param: WinInfo) -> None:
    """Mouse callback to get points and draw circles."""
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw green point
        cv2.circle(param.img, (x, y), 2, (0, 255, 0), 2)
        # Draw green circle
        cv2.circle(param.img, (x, y), 15, (0, 255, 0), 2)
        # Update image
        cv2.imshow(param.name, param.img)
        # Add point to point list (homogeneous coordinates)
        param.point_list.append(np.array([float(x), float(y), 1.0], dtype=np.float32))


def get_points_manual(img1: np.ndarray, img2: np.ndarray) -> Tuple[int, List[np.ndarray], List[np.ndarray]]:
    """
    Displays two images and catches the point pairs marked by left mouse clicks.
    Points will be in homogeneous coordinates.
    
    @param img1: The first image
    @param img2: The second image
    @return: Tuple of (number_of_points, p1, p2) where p1 and p2 are lists of points
    """
    print()
    print("Please select at least four points by clicking at the corresponding image positions:")
    print("Firstly click at the point that shall be transformed (within the image to be attached),")
    print("followed by a click on the corresponding point within the base image")
    print("Continue until you have collected as many point pairs as you wish")
    print("Stop the point selection by pressing any key")
    print()

    window_info_base = WinInfo(img1, "Image 1")
    window_info_attach = WinInfo(img2, "Image 2")

    # Show input images and install mouse callback
    cv2.namedWindow(window_info_base.name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_info_base.name, window_info_base.img)
    cv2.setMouseCallback(window_info_base.name, _get_points_callback, window_info_base)

    cv2.namedWindow(window_info_attach.name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_info_attach.name, window_info_attach.img)
    cv2.setMouseCallback(window_info_attach.name, _get_points_callback, window_info_attach)

    # Wait until any key was pressed
    cv2.waitKey(0)

    cv2.destroyWindow(window_info_base.name)
    cv2.destroyWindow(window_info_attach.name)

    num_of_points = len(window_info_base.point_list)
    p1 = window_info_base.point_list
    p2 = window_info_attach.point_list

    return num_of_points, p1, p2


def draw_epi_line(img: np.ndarray, a: float, b: float, c: float) -> None:
    """
    Draws line given in homogeneous representation into image.
    Line equation: ax + by + c = 0
    
    @param img: The image to draw into
    @param a: Line parameter
    @param b: Line parameter
    @param c: Line parameter
    """
    rows, cols = img.shape[:2]

    # Calculate intersection with image borders
    # Intersection with bottom edge (y=0): ax + c = 0 => x = -c/a
    p1 = (int(-c / a) if abs(a) > 1e-10 else 0, 0)
    # Intersection with left edge (x=0): by + c = 0 => y = -c/b
    p2 = (0, int(-c / b) if abs(b) > 1e-10 else 0)
    # Intersection with top edge (y=rows-1)
    p3 = (int((-b * (rows - 1) - c) / a) if abs(a) > 1e-10 else 0, rows - 1)
    # Intersection with right edge (x=cols-1)
    p4 = (cols - 1, int((-a * (cols - 1) - c) / b) if abs(b) > 1e-10 else 0)

    # Check start and end points
    start_point = None
    end_point = None

    for cur_p in [p1, p2, p3, p4]:
        if 0 <= cur_p[0] < cols and 0 <= cur_p[1] < rows:
            if start_point is None:
                start_point = cur_p
            else:
                end_point = cur_p

    # Draw line
    if start_point is not None and end_point is not None:
        cv2.line(img, start_point, end_point, (0, 0, 255), 1)


def draw_matches(img1: np.ndarray, img2: np.ndarray, 
                 p1: List[np.ndarray], p2: List[np.ndarray]) -> np.ndarray:
    """
    Concatenates two images and draws the matches between them as lines.
    
    @param img1: First image
    @param img2: Second image
    @param p1: List of keypoints in first image (homogeneous coordinates)
    @param p2: List of corresponding keypoints in second image (homogeneous coordinates)
    @return: Image showing matches
    """
    # Convert images to 8-bit color if necessary
    img1_converted = img1.astype(np.uint8) if img1.dtype != np.uint8 else img1.copy()
    img2_converted = img2.astype(np.uint8) if img2.dtype != np.uint8 else img2.copy()
    
    if len(img1_converted.shape) == 2:
        img1_converted = cv2.cvtColor(img1_converted, cv2.COLOR_GRAY2BGR)
    if len(img2_converted.shape) == 2:
        img2_converted = cv2.cvtColor(img2_converted, cv2.COLOR_GRAY2BGR)

    # Create combined image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_width = max(w1, w2)
    combined = np.zeros((h1 + h2, max_width, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1_converted
    combined[h1:h1 + h2, :w2] = img2_converted

    if len(p1) != len(p2):
        raise RuntimeError("Mismatched sizes for matched point arrays!")

    # Draw lines
    for i in range(len(p1)):
        x1 = int(p1[i][0] / p1[i][2])
        y1 = int(p1[i][1] / p1[i][2])
        x2 = int(p2[i][0] / p2[i][2])
        y2 = int(p2[i][1] / p2[i][2]) + h1
        cv2.line(combined, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Draw circles on top
    for i in range(len(p1)):
        x1 = int(p1[i][0] / p1[i][2])
        y1 = int(p1[i][1] / p1[i][2])
        x2 = int(p2[i][0] / p2[i][2])
        y2 = int(p2[i][1] / p2[i][2]) + h1
        cv2.circle(combined, (x1, y1), 2, (0, 255, 0), 2)
        cv2.circle(combined, (x2, y2), 2, (0, 255, 0), 2)

    return combined


def extract_raw_orb_matches(img1: np.ndarray, img2: np.ndarray) -> RawOrbMatches:
    """
    Computes and matches ORB feature points in two images.
    
    @param img1: First image
    @param img2: Second image
    @return: Structure containing keypoint locations and matches
    """
    orb = cv2.ORB_create(nfeatures=30000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    result = RawOrbMatches()

    # Convert keypoints to homogeneous coordinates
    result.keypoints1 = [
        np.array([kp.pt[0], kp.pt[1], 1.0], dtype=np.float32)
        for kp in keypoints1
    ]
    result.keypoints2 = [
        np.array([kp.pt[0], kp.pt[1], 1.0], dtype=np.float32)
        for kp in keypoints2
    ]

    # Match from image 1 to image 2
    if descriptors1 is not None and descriptors2 is not None:
        matches_1_2 = matcher.knnMatch(descriptors1, descriptors2, k=2)
        for m in matches_1_2:
            if len(m) >= 2:
                match = Match()
                match.closest = m[0].trainIdx
                match.closest_distance = m[0].distance
                match.second_closest = m[1].trainIdx
                match.second_closest_distance = m[1].distance
                result.matches_1_2[m[0].queryIdx] = match

        # Match from image 2 to image 1
        matches_2_1 = matcher.knnMatch(descriptors2, descriptors1, k=2)
        for m in matches_2_1:
            if len(m) >= 2:
                match = Match()
                match.closest = m[0].trainIdx
                match.closest_distance = m[0].distance
                match.second_closest = m[1].trainIdx
                match.second_closest_distance = m[1].distance
                result.matches_2_1[m[0].queryIdx] = match

    return result


__all__ = [
    "Match",
    "RawOrbMatches",
    "get_points_manual",
    "draw_epi_line",
    "draw_matches",
    "extract_raw_orb_matches",
]
