"""
I-24 3D dataset helpers for the Kocur baseline: per-camera projection,
ground-truth cuboids and road-plane back-projection.

I-24 conventions (same as src/speed_lstm/data.py): annotation (x, y) is the
rear ground-center in feet, d = direction in {-1, +1} (EB = +1, WB = -1),
l/w/h in feet, flat road z = 0. hg.json holds, per side and camera, a 3x4
P (feet -> pixels at the calibration resolution, 1920x1080) and the three
vanishing points `vps` (along road, across road, vertical).

The z axis of these P matrices points DOWN: the camera centers come out at
z = -100 to -150 ft (the I-24 MOTION poles are that tall). So a vehicle's
roof is at z = -h. `load_camera` returns the up sign so callers don't
hard-code it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FT_TO_M = 0.3048
FT_S_TO_KMH = FT_TO_M * 3.6
SIDES = {1: "EB", -1: "WB"}
CALIB_WIDTH = 1920  # hg.json pixel coordinates are for 1920x1080


def load_camera(hg_path: str | Path, camera: str, video_width: int) -> dict:
    """{'P': {+1: P, -1: P}, 'vps': (3, 2), 'scale': s} with everything in video pixels."""
    with open(hg_path) as f:
        hg = json.load(f)
    s = video_width / CALIB_WIDTH
    S = np.diag([s, s, 1.0])
    P = {d: S @ np.asarray(hg[k][camera]["P"], dtype=np.float64) for d, k in SIDES.items() if camera in hg[k]}
    # Some cameras (e.g. scene1 p2c6, p3c5) store P with the opposite overall sign, so visible
    # points get w < 0; flip those so a road point the camera sees projects with w > 0.
    for d, k in SIDES.items():
        if d in P and hg[k][camera].get("space_pts"):
            x, y = hg[k][camera]["space_pts"][0][:2]
            if (P[d] @ np.array([x, y, 0.0, 1.0]))[2] < 0:
                P[d] = -P[d]
    vps = np.asarray(hg["EB"][camera]["vps"], dtype=np.float64) * s
    up = {d: float(np.sign(camera_center(p)[2])) for d, p in P.items()}  # camera is above the road
    return {"P": P, "vps": vps, "scale": s, "up": up}


def load_gt(ann_csv: str | Path, ts_csv: str | Path, camera: str) -> pd.DataFrame:
    """Annotation rows seen by `camera`, with that camera's timestamp per frame."""
    ann = pd.read_csv(ann_csv)
    ann = ann[ann["camera"] == camera].copy()
    ts = pd.read_csv(ts_csv)
    if camera in ts.columns:
        ann = ann.merge(ts[["frame", camera]].rename(columns={camera: "timestamp"}), on="frame", how="left")
    else:
        ann["timestamp"] = np.nan
    ann["direction"] = ann["direction"].astype(int)
    return ann.reset_index(drop=True)


def gt_corners(row, up: float = -1.0) -> np.ndarray:
    """(8, 3) cuboid corners in feet; first 4 on the ground, roof at z = up * h."""
    x, y, l, w, h, d = (float(row[k]) for k in ("x", "y", "l", "w", "h", "direction"))
    h = up * abs(h)
    xs = (x, x + d * l)
    ys = (y - w / 2, y + w / 2)
    ground = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
    return np.array([(a, b, 0.0) for a, b in ground] + [(a, b, h) for a, b in ground])


def gt_footprint(row) -> np.ndarray:
    return gt_corners(row)[:4, :2]


def project(P: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N, 3) feet -> (N, 2) pixels."""
    q = (P @ np.c_[pts, np.ones(len(pts))].T).T
    return q[:, :2] / q[:, 2:3]


def ground_homography_inv(P: np.ndarray) -> np.ndarray:
    """Pixels -> road plane (z = 0) feet."""
    return np.linalg.inv(P[:, [0, 1, 3]])


def to_road(Hinv: np.ndarray, uv: np.ndarray) -> np.ndarray:
    uv = np.atleast_2d(uv)
    q = (Hinv @ np.c_[uv, np.ones(len(uv))].T).T
    return q[:, :2] / q[:, 2:3]


def camera_center(P: np.ndarray) -> np.ndarray:
    """Camera position in feet (right null vector of P)."""
    _, _, vt = np.linalg.svd(P)
    c = vt[-1]
    return c[:3] / c[3]


def near_face_center(row, cam_x: float) -> np.ndarray:
    """Ground center of the vehicle face closer to the camera along the road."""
    x, y, l, d = (float(row[k]) for k in ("x", "y", "l", "direction"))
    faces = (x, x + d * l)
    fx = min(faces, key=lambda v: abs(v - cam_x))
    return np.array([fx, y])


def height_from_edge(P: np.ndarray, ground_xy: np.ndarray, top_uv: np.ndarray) -> float:
    """|z| (feet) of the point above ground_xy that projects closest to top_uv (linear least squares)."""
    X0 = np.array([ground_xy[0], ground_xy[1], 0.0, 1.0])
    ez = np.array([0.0, 0.0, 1.0, 0.0])
    A, b = [], []
    for r, t in ((0, top_uv[0]), (1, top_uv[1])):
        A.append(t * (P[2] @ ez) - P[r] @ ez)
        b.append(P[r] @ X0 - t * (P[2] @ X0))
    A, b = np.array(A), np.array(b)
    return abs(float(A @ b / (A @ A)))
