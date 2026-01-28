# ==============================================================================
# File        : main_manual.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Manual point selection for fundamental matrix
#               estimation (Exercise 04).
# ==============================================================================

import sys

import cv2
import numpy as np

from Helper import get_points_manual
from Pcv4 import (
    get_fundamental_matrix,
    get_error_multiple,
    count_inliers,
    visualize,
)


def main():
    """
    Main function. Loads images and processes fundamental matrix estimation.
    Usage: python main_manual.py <path_to_1st_image> <path_to_2nd_image>
    """
    # Check if image paths were defined
    if len(sys.argv) != 3:
        print("Usage: python main_manual.py <path_to_1st_image> <path_to_2nd_image>")
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

    inlier_threshold = 2.0

    # Get corresponding points within the two images
    # Start with one point within the first image, then click on corresponding point in second image
    number_of_point_pairs, p_fst, p_snd = get_points_manual(fst_image, snd_image)

    # Just some output
    print(f"Number of defined point pairs: {number_of_point_pairs}")
    print()
    print("Points in first image:")
    for p in p_fst:
        print(p)
    print()
    print("Points in second image:")
    for p in p_snd:
        print(p)

    # Calculate fundamental matrix
    F = get_fundamental_matrix(p_fst, p_snd)

    # Visualize epipolar lines
    visualize(fst_image, snd_image, p_fst, p_snd, F)

    # Calculate geometric error
    err = get_error_multiple(p_fst, p_snd, F)
    print(f"Geometric error: {err}")
    num_inlier = count_inliers(p_fst, p_snd, F, inlier_threshold)
    print(f"Number of inliers: {num_inlier} of {len(p_fst)}")

    print("Press enter to continue...")
    input()

    return 0


if __name__ == "__main__":
    sys.exit(main())
