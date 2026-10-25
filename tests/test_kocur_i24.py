"""Geometry helpers of the Kocur-on-I-24 baseline (baselines/kocur_i24/i24.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "kocur_i24"))
import i24  # noqa: E402


def _pole_camera(height_ft=120.0):
    """Pinhole camera above the road looking down the +x axis, I-24 style z-down world."""
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    tilt = np.deg2rad(35)
    # camera axes in world coords; +z world is down, so image y (down) has a positive z component
    z_c = np.array([np.cos(tilt), 0, np.sin(tilt)])
    x_c = np.array([0.0, 1.0, 0.0])
    y_c = np.cross(z_c, x_c)
    R = np.stack([x_c, y_c, z_c])
    C = np.array([0.0, 0.0, -height_ft])
    return K @ np.c_[R, -R @ C]


def _row(**kw):
    base = dict(x=300.0, y=6.0, l=16.0, w=6.5, h=5.5, direction=1)
    base.update(kw)
    return pd.Series(base)


def test_camera_center_is_above_the_road_in_z_down_frame():
    P = _pole_camera(120.0)
    assert np.allclose(i24.camera_center(P), [0, 0, -120.0], atol=1e-6)


def test_ground_homography_round_trip():
    P = _pole_camera()
    pts = np.array([[250.0, -10.0, 0.0], [400.0, 12.0, 0.0]])
    uv = i24.project(P, pts)
    assert np.allclose(i24.to_road(i24.ground_homography_inv(P), uv), pts[:, :2], atol=1e-6)


def test_roof_is_drawn_above_the_ground_and_height_is_recovered():
    P = _pole_camera()
    row = _row()
    corners = i24.gt_corners(row, up=float(np.sign(i24.camera_center(P)[2])))
    uv = i24.project(P, corners)
    assert np.all(uv[4:, 1] < uv[:4, 1])  # roof higher in the image than the wheels
    for k in range(4):
        assert abs(i24.height_from_edge(P, corners[k, :2], uv[4 + k]) - 5.5) < 1e-6


def test_near_face_is_the_one_facing_the_camera():
    # camera at x = 0 looking toward +x: an EB car's rear (x) faces it, a WB car's front (x - l) does
    assert np.allclose(i24.near_face_center(_row(direction=1), 0.0), [300.0, 6.0])
    assert np.allclose(i24.near_face_center(_row(direction=-1), 0.0), [284.0, 6.0])
