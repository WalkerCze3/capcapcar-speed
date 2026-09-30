import numpy as np
import pandas as pd

from speed_lstm.brno import (compute_camera_calibration, compute_matches, calculate_speeds, line_crossing,
                             prefilter, projection_from_calibration, road_plane_point, tracks_to_cars)
from speed_lstm.lift3d import ground_point

# A plausible Brno-style calibration (1920x1080, vp1 up the road, vp2 far to the side).
CALIB = {"vp1": [1150.0, -700.0], "vp2": [-9000.0, 1500.0], "pp": [960.5, 540.5], "scale": 0.042}


def test_projection_matches_official_road_distances():
    P = projection_from_calibration(**CALIB)
    road_plane, focal = compute_camera_calibration(CALIB["vp1"], CALIB["vp2"], CALIB["pp"])
    pixels = [(900, 900), (1000, 700), (700, 1000), (1300, 650), (960, 800)]
    ours = [ground_point(u, v, P) for u, v in pixels]
    official = [road_plane_point([u, v], focal, road_plane, CALIB["pp"]) for u, v in pixels]
    for i in range(len(pixels)):
        for j in range(i + 1, len(pixels)):
            d_ours = np.linalg.norm(ours[i] - ours[j])
            d_off = CALIB["scale"] * np.linalg.norm(official[i] - official[j])
            np.testing.assert_allclose(d_ours, d_off, rtol=1e-9)
    # Ground points reproject onto their pixels.
    for (u, v), xy in zip(pixels, ours):
        uvw = P @ np.array([xy[0], xy[1], 0.0, 1.0])
        np.testing.assert_allclose(uvw[:2] / uvw[2], [u, v], atol=1e-6)


def test_road_axes_map_to_vanishing_points():
    P = projection_from_calibration(**CALIB)
    for axis, vp in ((0, CALIB["vp1"]), (1, CALIB["vp2"])):
        d = P[:, axis]
        np.testing.assert_allclose(d[:2] / d[2], vp, rtol=1e-6)
    # Camera sits above the road: its center is (0, 0, h > 0).
    _, _, vt = np.linalg.svd(P)
    c = vt[-1] / vt[-1][3]
    assert abs(c[0]) < 1e-6 and abs(c[1]) < 1e-6 and c[2] > 1.0


def _synthetic_gt_and_track(speed_mps=25.0, fps=50.0):
    """One car driving along x at constant speed through three measurement lines."""
    P = projection_from_calibration(**CALIB)
    xs_line = [25.0, 35.0, 45.0]                    # line positions along the road (m), all in view
    def img_line(x):                                # image of the road line {x = const}
        a = P @ np.array([x, -20.0, 0.0, 1.0]); b = P @ np.array([x, 20.0, 0.0, 1.0])
        l = np.cross(a, b)
        return l / np.linalg.norm(l[:2])
    def lane_line(y):
        a = P @ np.array([0.0, y, 0.0, 1.0]); b = P @ np.array([40.0, y, 0.0, 1.0])
        return np.cross(a, b)
    # Lines listed far -> near like the dataset (car crosses the last line first).
    lines = [img_line(x) for x in xs_line[::-1]]
    t = np.arange(0, 2.0, 1 / fps)
    x = 15.0 + speed_mps * t
    gt = {"fps": fps, "measurementLines": lines, "laneDivLines": [lane_line(y) for y in (-3.0, 0.0, 3.0)],
          "invalidLanes": set(),
          # Dataset order: intersections[-1] is the last line in time.
          "cars": [{"carId": 0, "valid": True, "speed": speed_mps * 3.6, "laneIndex": {0},
                    "intersections": [{"measurementLineId": i, "videoTime": (xl - 15.0) / speed_mps}
                                      for i, xl in enumerate(xs_line)]}]}
    lifted = pd.DataFrame({"track_id": 7, "timestamp": t, "cx": x, "cy": -1.5})
    return P, gt, lifted


def test_official_matching_recovers_speed_on_synthetic_track():
    P, gt, lifted = _synthetic_gt_and_track()
    cars = prefilter(tracks_to_cars(lifted, P, gt["fps"]), gt)
    assert len(cars) == 1
    assert calculate_speeds(cars, gt, CALIB) == 0
    np.testing.assert_allclose(cars[0]["speed"], 90.0, rtol=1e-3)
    m = compute_matches(gt, cars)
    assert m[0]["matched"] and m[0]["track_id"] == 7
    np.testing.assert_allclose(m[0]["full_kmh"], 90.0, rtol=1e-3)


def test_label_windows_keeps_matched_pass_only():
    from speed_lstm.brno import label_windows
    matches = pd.DataFrame([
        {"matched": True, "valid": True, "track_id": 1, "gt_id": 10, "gt_kmh": 72.0, "track_t_first": 2.0, "track_t_last": 3.0},
        {"matched": True, "valid": False, "track_id": 2, "gt_id": 11, "gt_kmh": 50.0, "track_t_first": 2.0, "track_t_last": 3.0},
        {"matched": False, "valid": True, "track_id": np.nan, "gt_id": 12, "gt_kmh": 60.0, "track_t_first": np.nan, "track_t_last": np.nan},
    ])
    w = lambda tid, t0: {"track_id": tid, "timestamps": [t0, t0 + 0.6]}
    out = label_windows([w(1, 0.5), w(1, 2.2), w(1, 3.2), w(1, 5.0), w(2, 2.2), w(3, 2.2)], matches, "s9")
    assert [o["timestamps"][0] for o in out] == [2.2, 3.2]
    assert all(o["target_speed"] == 20.0 and o["group"] == "s9:10" for o in out)


def test_finetune_runs_and_selects_by_val(tmp_path):
    import torch
    from speed_lstm import features_v2 as fv2
    from speed_lstm.finetune import finetune, group_split
    from speed_lstm.model import SpeedLSTM
    n = fv2.n_features("3d")
    m = SpeedLSTM(input_size=n)
    torch.save({"model_state": m.state_dict(), "mode": "3d", "input_size": n, "hidden_size": m.hidden_size,
                "n_observations": 16, "max_timestamp_gap": 0.2, "feature_mean": np.zeros(n), "feature_std": np.ones(n),
                "target_mean": 20.0, "target_std": 5.0}, tmp_path / "init.pt")
    rng = np.random.default_rng(0)
    windows = []
    for car in range(20):
        v = rng.uniform(15, 35)
        for k in range(4):
            t = np.arange(16) * 0.04 + k
            b3 = np.column_stack([v * t, np.zeros(16), np.full(16, 0.75), np.full((16, 3), [4.5, 1.8, 1.5])])
            windows.append({"timestamps": t.tolist(), "boxes3d": b3.tolist(), "boxes2d": None,
                            "target_speed": v, "group": f"c{car}"})
    train, val = group_split(windows, 0.25, 0)
    assert not {w["group"] for w in train} & {w["group"] for w in val}
    res = finetune(tmp_path / "init.pt", train, val, tmp_path / "out", epochs=15, lr=3e-3, renorm="all", log=lambda *_: None)
    assert (tmp_path / "out" / "best.pt").exists()
    assert res["best"]["mae"] <= res["history"][0]["mae"]


def test_experiment_accepts_long_inline_config(tmp_path):
    import json, subprocess, sys
    from pathlib import Path
    cfg = {"name": "x" * 400, "test": [], "finetune": None}
    (tmp_path / "runs/v2/3d").mkdir(parents=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "scripts/brno_experiment.py"),
                        "--config", json.dumps(cfg), "--project", str(tmp_path), "--out-dir", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
