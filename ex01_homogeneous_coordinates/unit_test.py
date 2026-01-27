# ============================================================
# File        : main.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Institution : Technische Universität Berlin
# Description : Entry point for Exercise 01 (Homogeneous Coordinates).
#               Runs processing and validation routines.
# ============================================================
import sys
import numpy as np
import cv2 as cv
from Pcv1 import (
    getConnectingLine_2D, getScalingMatrix_2D, getRotationMatrix_2D,
    getTranslationMatrix_2D, getH_2D, applyH_2D, isPointOnLine_2D,
    eucl2hom_point_2D, hom2eucl_point_2D, GEOM_TYPE_POINT, GEOM_TYPE_LINE
)


def test_getConnectingLine():
    v1 = np.array([0, 0, 1], dtype=np.float32)
    v2 = np.array([1, 1, 1], dtype=np.float32)
    lt = np.array([-1, 1, 0], dtype=np.float32)
    lc = getConnectingLine_2D(v1, v2)
    
    if not np.allclose(lc, lt):
        print("There seems to be a problem with getConnectingLine_2D(..)!")
        return False
    return True


def test_getScaleMatrix():
    St = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=np.float32)
    Sc = getScalingMatrix_2D(3)
    
    if not np.allclose(Sc, St):
        print("There seems to be a problem with getScalingMatrix_2D(..)!")
        print("Press enter to continue...")
        input()
        sys.exit(-1)
    return True


def test_getRotMatrix():
    Rt = np.array([[1./np.sqrt(2), -1./np.sqrt(2), 0], 
                   [1./np.sqrt(2), 1./np.sqrt(2), 0], 
                   [0, 0, 1]], dtype=np.float32)
    Rc = getRotationMatrix_2D(45)
    
    if not np.allclose(Rc, Rt):
        print("There seems to be a problem with getRotationMatrix_2D(..)!")
        print("Press enter to continue...")
        input()
        sys.exit(-1)
    return True


def test_getTranslMatrix():
    Tt = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]], dtype=np.float32)
    Tc = getTranslationMatrix_2D(-1, -1)
    
    if not np.allclose(Tc, Tt):
        print("There seems to be a problem with getTranslationMatrix_2D(..)!")
        print("Press enter to continue...")
        input()
        sys.exit(-1)
    return True


def test_getH():
    St = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=np.float32)
    Rt = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    Tt = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]], dtype=np.float32)
    Ht = np.array([[0, -3, 3], [3, 0, -3], [0, 0, 1]], dtype=np.float32)
    Hc = getH_2D(Tt, Rt, St)

    if not np.allclose(Hc, Ht):
        print("There seems to be a problem with getH_2D(..)!")
        print("Press enter to continue...")
        input()
        sys.exit(-1)
    return True


def test_applyH():
    H = np.array([[0, -3, 3], 
                  [3, 0, -3], 
                  [0, 0, 1]], dtype=np.float32)

    # Test points
    v = []
    vnt = []
    for i in range(10):
        v.append(np.array([1.0*i, 1.0, 1.0], dtype=np.float32))
        vnt.append(np.array([0.0, 3.0*i - 3.0, 1.0], dtype=np.float32))

    vnc = applyH_2D(v, H, GEOM_TYPE_POINT)

    if len(v) != len(vnc):
        print("There seems to be a problem with applyH_2D(..) for points! The number of returned points does not match the number of given points.")
        print("Press enter to continue...")
        input()
        sys.exit(-1)

    for i in range(len(v)):
        if not np.allclose(vnc[i], vnt[i]):
            print(f"There seems to be a problem with applyH_2D(..) for points for the {i}th element in the array!")
            print(vnc[i])
            print()
            print(vnt[i])
            print("Press enter to continue...")
            input()
            sys.exit(-1)

    # Test lines
    v = []
    vnt = []
    for i in range(10):
        v.append(np.array([-1.0*i, 1.0, 0.0], dtype=np.float32))
        vnt.append(np.array([-1.0/3.0, -1.0/3.0*i, 1.0 - i], dtype=np.float32))

    vnc = applyH_2D(v, H, GEOM_TYPE_LINE)

    if len(v) != len(vnc):
        print("There seems to be a problem with applyH_2D(..) for lines! The number of returned lines does not match the number of given lines.")
        print("Press enter to continue...")
        input()
        sys.exit(-1)

    for i in range(len(v)):
        if not np.allclose(vnc[i], vnt[i]):
            print(f"There seems to be a problem with applyH_2D(..) for lines for the {i}th element in the array!")
            print("Press enter to continue...")
            input()
            sys.exit(-1)
    
    return True


def test_isPointOnLine():
    try:
        v = np.array([1, 1, 1], dtype=np.float32)
        l = np.array([-1, 1, 0], dtype=np.float32)

        t = True
        c = isPointOnLine_2D(v, l)
        
        if t != c:
            print(f"isPointOnLine_2D: fail\nExpected:\n{t}\nGiven:\n{c}")
            print("Press enter to continue...")
            input()
            sys.exit(-1)
        
        v = np.array([0, 1, 1], dtype=np.float32)
        l = np.array([-1, 1, 1], dtype=np.float32)

        t = False
        c = isPointOnLine_2D(v, l)
        if t != c:
            print(f"isPointOnLine_2D: fail\nExpected:\n{t}\nGiven:\n{c}")
            print("Press enter to continue...")
            input()
            sys.exit(-1)
        
        v = np.array([0, 1, 1], dtype=np.float32)
        l = np.array([-1, -1, -1], dtype=np.float32)

        t = False
        c = isPointOnLine_2D(v, l)
        if t != c:
            print(f"isPointOnLine_2D - negative dot product: fail\nExpected:\n{t}\nGiven:\n{c}")
            print("Press enter to continue...")
            input()
            sys.exit(-1)
        
        v = np.array([0, 1e-6, 1], dtype=np.float32)
        l = np.array([-1, -1, 0], dtype=np.float32)

        t = True
        c = isPointOnLine_2D(v, l)
        if t != c:
            print(f"isPointOnLine_2D - eps: fail\nExpected:\n{t}\nGiven:\n{c}")
            print("Press enter to continue...")
            input()
            sys.exit(-1)
        
        v = np.array([5, -2, 0], dtype=np.float32)
        l = np.array([0, 0, 1], dtype=np.float32)

        t = True
        c = isPointOnLine_2D(v, l)
        if t != c:
            print(f"isPointOnLine_2D - ideal line: fail\nExpected:\n{t}\nGiven:\n{c}")
            print("Press enter to continue...")
            input()
            sys.exit(-1)
            
    except Exception as exc:
        print(exc)
        return False
    
    return True


def test_eucl2hom():
    a = np.array([1, 2], dtype=np.float32)
    b = eucl2hom_point_2D(a)
    if (a[0] * b[2] != b[0]) or (a[1] * b[2] != b[1]):
        print("There seems to be a problem with eucl2hom_point_2D(..)!")
        print("Press enter to continue...")
        input()
        sys.exit(-1)
    return True


def test_hom2eucl():
    a = np.array([8, 4, 2], dtype=np.float32)
    b = hom2eucl_point_2D(a)
    if (b[0] != 4.0) or (b[1] != 2.0):
        print("There seems to be a problem with hom2eucl_point_2D(..)!")
        print("Press enter to continue...")
        input()
        sys.exit(-1)
    return True


def main():
    test_getConnectingLine()
    test_getScaleMatrix()
    test_getRotMatrix()
    test_getTranslMatrix()
    test_getH()
    test_applyH()
    test_isPointOnLine()
    test_eucl2hom()
    test_hom2eucl()
    
    print("Finished basic testing: Everything seems to be fine.")
    input()

    return 0


if __name__ == "__main__":
    sys.exit(main())
