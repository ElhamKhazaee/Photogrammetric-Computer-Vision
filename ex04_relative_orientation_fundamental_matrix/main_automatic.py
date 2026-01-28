# ==============================================================================
# File        : main_automatic.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Automatic point matching using ORB features for
#               fundamental matrix estimation (Exercise 04).
# ==============================================================================

import sys

import cv2
import numpy as np

from Helper import draw_matches
from Pcv4 import (
    get_fundamental_matrix,
    get_error_multiple,
    count_inliers,
    visualize,
    get_points_automatic,
    estimate_fundamental_ransac,
)


def main():
    """
    Main function. Loads images and processes fundamental matrix estimation.
    Usage: python main_automatic.py <path_to_1st_image> <path_to_2nd_image>
    """
    # Check if image paths were defined
    if len(sys.argv) != 3:
        print("Usage: python main_automatic.py <path_to_1st_image> <path_to_2nd_image>")
        print("Press enter to continue...")
        input()
        return -1

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]

    # Load images
    fst_image = cv2.imread(img1_path)
    snd_image = cv2.imread(img2_path)

    if fst_image is None or snd_image is None:
        print("ERROR: Could not load images")
        print("Press enter to continue...")
        input()
        return -2

    # Get corresponding points within the two images
    p_fst, p_snd = get_points_automatic(fst_image, snd_image)

    matches_image = draw_matches(fst_image, snd_image, p_fst, p_snd)
    cv2.imwrite("matches.png", matches_image)

    # Just some output
    print(f"Number of defined point pairs: {len(p_fst)}")

    inlier_threshold = 2.0

    print("=== Direct estimation: ===")

    # Calculate fundamental matrix
    F = get_fundamental_matrix(p_fst, p_snd)

    # Calculate geometric error
    err = get_error_multiple(p_fst, p_snd, F)
    print(f"Geometric error: {err}")
    num_inlier = count_inliers(p_fst, p_snd, F, inlier_threshold)
    print(f"Number of inliers: {num_inlier} of {len(p_fst)}")

    # Visualize epipolar lines
    visualize(fst_image, snd_image, p_fst, p_snd, F)

    print("=== Robust estimation: ===")

    F = estimate_fundamental_ransac(p_fst, p_snd, 10000, inlier_threshold)

    # Calculate geometric error
    err = get_error_multiple(p_fst, p_snd, F)
    print(f"Geometric error: {err}")
    num_inlier = count_inliers(p_fst, p_snd, F, inlier_threshold)
    print(f"Number of inliers: {num_inlier} of {len(p_fst)}")

    # Visualize epipolar lines
    visualize(fst_image, snd_image, p_fst, p_snd, F)

    print("Press enter to continue...")
    input()

    return 0


if __name__ == "__main__":
    sys.exit(main())
