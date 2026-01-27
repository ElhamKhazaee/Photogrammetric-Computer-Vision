# ============================================================
# File        : main.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Institution : Technische Universität Berlin
# Description : Entry point for Exercise 01 (Homogeneous Coordinates).
#               Runs processing and validation routines.
# ============================================================
import sys
from Pcv1 import run


def main():
    """
    Main function. Loads and processes image
    Usage: python main.py <path_to_image>
    """
    # check if image path was defined
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_image>")
        print("Press enter to continue...")
        input()
        return -1
    
    # get filename from command line argument
    fname = sys.argv[1]
    
    # start processing
    run(fname)

    return 0


if __name__ == "__main__":
    main()
