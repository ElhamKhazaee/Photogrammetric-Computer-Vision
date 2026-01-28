# ==============================================================================
# File        : main.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Main script for Exercise 05 –
# Structure from Motion (SfM). Executes the SfM pipeline,
# bundle adjustment, and result export.
# ==============================================================================

import sys
import math
import numpy as np

from Pcv5 import buildScene, produceInitialState, BundleAdjustment
from Helper import LevenbergMarquardt, hom2eucl


MAX_ITERS = 50


def writeStateToPly(filename, state):
    num_vertices_tracks = len(state.m_tracks) * 4
    num_vertices_cameras = len(state.m_cameras) * 5
    num_vertices = num_vertices_tracks + num_vertices_cameras
    num_triangles = len(state.m_tracks) * 4 + len(state.m_cameras) * 4

    with open(filename, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {num_triangles}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for track in state.m_tracks:
            center = hom2eucl(track.location)
            center = np.clip(center, -50.0, 50.0)
            color = "255 255 255"
            size = 0.01
            s2 = size * math.sqrt(0.5)
            f.write(f"{center[0]-size} {center[1]} {center[2]-s2} {color}\n")
            f.write(f"{center[0]+size} {center[1]} {center[2]-s2} {color}\n")
            f.write(f"{center[0]} {center[1]-size} {center[2]+s2} {color}\n")
            f.write(f"{center[0]} {center[1]+size} {center[2]+s2} {color}\n")

        for cam in state.m_cameras:
            color = "255 0 0"
            H = cam.H
            Hinv = np.linalg.inv(H)
            size = 0.2
            corners = [
                Hinv @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                Hinv @ np.array([+size, +size, size, 1.0], dtype=np.float32),
                Hinv @ np.array([-size, +size, size, 1.0], dtype=np.float32),
                Hinv @ np.array([+size, -size, size, 1.0], dtype=np.float32),
                Hinv @ np.array([-size, -size, size, 1.0], dtype=np.float32),
            ]
            for c in corners:
                c = c / c[3]
                f.write(f"{c[0]} {c[1]} {c[2]} {color}\n")

        for i in range(len(state.m_tracks)):
            f.write(f"3 {i*4 + 0} {i*4 + 1} {i*4 + 3}\n")
            f.write(f"3 {i*4 + 1} {i*4 + 2} {i*4 + 3}\n")
            f.write(f"3 {i*4 + 2} {i*4 + 0} {i*4 + 3}\n")
            f.write(f"3 {i*4 + 0} {i*4 + 1} {i*4 + 2}\n")

        for i in range(len(state.m_cameras)):
            o = num_vertices_tracks + i * 5
            f.write(f"3 {o + 0} {o + 1} {o + 2}\n")
            f.write(f"3 {o + 0} {o + 2} {o + 4}\n")
            f.write(f"3 {o + 0} {o + 4} {o + 3}\n")
            f.write(f"3 {o + 0} {o + 3} {o + 1}\n")


def main(argv):
    if len(argv) < 5:
        print("Usage: main <focal length> <principal point X> <principal point y> <path to 1st image> <path to 2nd image> <path to 3rd image> ...")
        return -1

    images_filenames = argv[4:]

    K = np.eye(3, dtype=np.float32)
    K[0, 0] = float(argv[1])
    K[1, 1] = float(argv[1])
    K[0, 2] = float(argv[2])
    K[1, 2] = float(argv[3])

    scene = buildScene(images_filenames)
    state = BundleAdjustment.BAState(scene)
    produceInitialState(scene, K, state)
    writeStateToPly("beforeBA.ply", state)

    bundle_adjustment = BundleAdjustment(scene)
    lm = LevenbergMarquardt(bundle_adjustment, state)

    for i in range(MAX_ITERS):
        lm.iterate()

        sum_weights = 0.0
        for c in scene.cameras:
            for kp in c.keypoints:
                sum_weights += kp.weight * kp.weight

        if sum_weights > 0.0:
            err = np.sqrt(lm.getLastError() / sum_weights)
        else:
            err = 0.0
        print(f"iter {i} error: {err} (reprojection stddev in pixels)")

        if i % 10 == 9:
            bundle_adjustment.downweightOutlierKeypoints(lm.getState())

        if lm.getDamping() > 1e6:
            break

    writeStateToPly("afterBA.ply", lm.getState())
    state = lm.getState()
    # =========================
    # Normalize reconstruction
    # =========================

    # Collect all 3D points
    Xs = np.stack([t.location[:3] / t.location[3] for t in state.m_tracks], axis=0)

    # 1) Compute centroid
    centroid = Xs.mean(axis=0)

    # 2) Compute scale (max distance from centroid)
    scale = np.max(np.linalg.norm(Xs - centroid, axis=1))

    # Avoid division by zero
    if scale < 1e-9:
        scale = 1.0

    # 3) Normalize 3D points
    for t in state.m_tracks:
        X = t.location[:3] / t.location[3]
        Xn = (X - centroid) / scale
        t.location = np.array([Xn[0], Xn[1], Xn[2], 1.0], dtype=np.float32)

    # 4) Normalize camera poses
    for cam in state.m_cameras:
        Hinv = np.linalg.inv(cam.H)
        C = Hinv[:3, 3]  # camera center
        Cn = (C - centroid) / scale
        Hinv[:3, 3] = Cn
        cam.H = np.linalg.inv(Hinv)

    np.savez(
        "pcv5_result.npz",
        K=state.m_internalCalibs[0].K,
        Hs=np.stack([c.H for c in state.m_cameras], axis=0),
        Xs=np.stack([t.location for t in state.m_tracks], axis=0),
    )

    print("Saved results to pcv5_result.npz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
