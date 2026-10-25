"""Kocur boxes -> lift3d-format 3D boxes (baselines/kocur_brno/kocur_boxes3d.py), on a synthetic Brno camera."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "kocur_brno"))
sys.path.insert(0, str(ROOT / "src"))
import kocur_boxes3d as kb  # noqa: E402
from speed_lstm import brno  # noqa: E402
from speed_lstm.data import cuboid_corners, project_points, project_to_bbox  # noqa: E402

CALIB = {"vp1": [950.0, -400.0], "vp2": [5200.0, 600.0], "pp": [960.5, 540.5], "scale": 0.05}


def _camera():
    P = brno.projection_from_calibration(CALIB["vp1"], CALIB["vp2"], CALIB["pp"], CALIB["scale"])
    vp2 = P[:, 1] / P[2, 1]
    vp3 = P[:, 2] / P[2, 2]
    # A warp like Kocur's: VP2 lines -> horizontal, VP3 lines -> vertical (sends both VPs to infinity).
    M = np.linalg.inv(np.column_stack([vp2, vp3, [960.0, 540.0, 1.0]]))
    return P, M


def _kocur_row(frame, center, dims, P):
    # Kocur's corner order is case-dependent; shuffle to make sure nothing relies on it.
    corners = project_points(cuboid_corners(np.asarray(center), np.asarray(dims)), P)[[5, 2, 7, 0, 3, 6, 1, 4]]
    return {"frame": frame, **dict(zip(kb.CORNER_COLS, corners.ravel()))}


def test_box_to_3d_recovers_the_cuboid():
    P, M = _camera()
    center, dims = np.array([25.0, -1.5, 0.75]), np.array([4.6, 1.8, 1.5])
    row = _kocur_row(0, center, dims, P)
    corners = np.array([row[c] for c in kb.CORNER_COLS]).reshape(8, 2)
    c, d = kb.box_to_3d(corners, P, kb.i24.ground_homography_inv(P), M, P[:2, 2] / P[2, 2])
    assert np.allclose(c, center, atol=1e-6) and np.allclose(d, dims, atol=1e-6)


def test_kocur_lifted_matches_tracks_and_fixes_dims():
    P, M = _camera()
    dims = np.array([4.6, 1.8, 1.5])
    dets, kdet = [], []
    for f in range(20):
        center = np.array([15.0 + 0.6 * f, 1.0, 0.75])  # moving away from the camera: +x, direction +1
        noisy = dims * (1 + 0.05 * np.sin(f))  # Kocur's per-frame size wobbles
        kdet.append(_kocur_row(f, [*center[:2], noisy[2] / 2], noisy, P))
        box = project_to_bbox(center, dims, P)
        for tid, cls in ((7, "car"), (8, "truck")):  # YOLO's duplicate track of the same car
            dets.append({"frame": f, "track_id": tid, "cls": cls, "xmin": box[0], "ymin": box[1],
                         "xmax": box[2], "ymax": box[3]})
    kdet.append(_kocur_row(5, np.array([60.0, -3.0, 0.75]), dims, P))  # a car with no track
    out = kb.kocur_lifted(pd.DataFrame(dets), pd.DataFrame(kdet), P, M, np.arange(20) * 0.04, smooth_window=1)
    assert sorted(out["track_id"].unique()) == [7, 8] and len(out) == 40
    t7 = out[out["track_id"] == 7]
    assert t7[["length", "width", "height"]].nunique().eq(1).all()  # one size per track
    assert np.allclose(t7[["length", "width", "height"]].iloc[0], dims * (1 + 0.05 * np.median(np.sin(range(20)))),
                       rtol=1e-6)
    assert (t7["direction"] == 1).all()
    assert np.allclose(t7["cz"], t7["height"] / 2)
    assert np.allclose(t7["cx"].diff().dropna(), 0.6, atol=1e-6)
