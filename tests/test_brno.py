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
