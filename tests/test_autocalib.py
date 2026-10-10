"""Automatic calibration on synthetic traffic seen by a known camera."""

import numpy as np
import pandas as pd
import pytest

from speed_lstm import autocalib, brno
from speed_lstm.data import project_to_bbox
from speed_lstm.lift3d import DIM_PRIORS

W, H = 1920, 1080
PP = np.array([W / 2.0, H / 2.0])
VP1 = np.array([1000.0, -600.0])   # along the road, above the image
VP2 = np.array([-4000.0, 1500.0])  # across the road, far left
SCALE = 0.05


def true_P():
    return brno.projection_from_calibration(VP1, VP2, PP, SCALE)


def synthetic_dets(P, n_cars=40, seed=0, px_noise=0.5, dims_jitter=0.05):
    """Cars of roughly prior size driving along the road in 3 lanes, both directions, boxes in pixels."""
    rng = np.random.default_rng(seed)
    prior = np.asarray(DIM_PRIORS["car"][0])
    rows = []
    for tid in range(n_cars):
        lane = rng.choice([-3.5, 0.0, 3.5])
        dims = prior * np.exp(rng.normal(0.0, dims_jitter, 3))
        direction = 1 if lane >= 0 else -1
        frame = 0
        for x in np.arange(-150.0, 150.0, 0.6)[::direction]:
            box = project_to_bbox(np.array([x, lane, dims[2] / 2]), dims, P)
            if box is None:
                continue
            box = box + rng.normal(0.0, px_noise, 4)
            if box[0] < 5 or box[1] < 5 or box[2] > W - 5 or box[3] > H - 5 or box[2] - box[0] < 12:
                continue
            rows.append({"frame": frame, "track_id": tid, "cls": "car", "conf": 0.9,
                         "xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3]})
            frame += 1
    return pd.DataFrame(rows)


def noisy_vp_candidates(vp, n, deg, outlier_frac, seed):
    """Homogeneous VPs scattered `deg` degrees around `vp` on the Gaussian sphere, plus uniform outliers."""
    rng = np.random.default_rng(seed)
    f0 = float(np.hypot(W, H))
    d = autocalib.to_direction(np.append(vp, 1.0), PP, f0)[0]
    out = []
    for i in range(n):
        if rng.random() < outlier_frac:
            v = rng.normal(size=3)
        else:
            v = d + rng.normal(0.0, np.radians(deg), 3)
        out.append(autocalib.from_direction(v / np.linalg.norm(v), PP, f0))
    return np.array(out)


def test_direction_round_trip_handles_infinity():
    f0 = 2000.0
    for vp in (np.array([300.0, 200.0, 1.0]), np.array([1.0, 0.2, 0.0])):
        back = autocalib.from_direction(autocalib.to_direction(vp, PP, f0)[0], PP, f0)
        assert np.allclose(np.cross(back, vp), 0.0, atol=1e-9 * np.linalg.norm(vp) * np.linalg.norm(back))


def test_aggregate_directions_ignores_sign_and_outliers():
    rng = np.random.default_rng(1)
    true = np.array([0.6, 0.0, 0.8])
    good = true + rng.normal(0, 0.01, (80, 3))
    good *= rng.choice([-1, 1], size=(80, 1))          # antipodes are the same VP
    bad = rng.normal(size=(20, 3))
    dirs = np.vstack([good, bad])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    est, _ = autocalib.aggregate_directions(dirs)
    assert abs(est @ true) > 0.999


def test_vp1_from_track_lines():
    dets = synthetic_dets(true_P())
    lines = autocalib.track_lines(dets)
    assert len(lines) >= 30
    vp, frac = autocalib.vp_from_lines(lines)
    assert frac > 0.8
    assert np.linalg.norm(autocalib.to_point(vp) - VP1) < 25.0


def test_scale_recovered_from_car_size():
    dets = synthetic_dets(true_P(), dims_jitter=0.0, px_noise=0.0)
    boxes = autocalib.sample_car_boxes(dets, (W, H), max_boxes=150)[autocalib.BOX2D_COLS].to_numpy()
    scale, resid = autocalib.estimate_scale(boxes, VP1, VP2, PP)
    assert scale == pytest.approx(SCALE, rel=0.01)
    assert resid < 1.0


def test_calibrate_end_to_end():
    P = true_P()
    dets = synthetic_dets(P)
    vp2s = noisy_vp_candidates(VP2, 200, deg=1.0, outlier_frac=0.2, seed=3)
    cal = autocalib.calibrate(dets, (W, H), vp2_candidates=vp2s, max_boxes=150)

    assert cal.quality["vp1_source"] == "tracks"
    assert np.linalg.norm(cal.vp1 - VP1) < 25.0
    true_focal = np.sqrt(-np.dot(VP1 - PP, VP2 - PP))
    assert cal.focal == pytest.approx(true_focal, rel=0.03)
    assert cal.scale == pytest.approx(SCALE, rel=0.05)
    assert cal.reliable

    # Ground distances under the estimated P match the true ones: what speed depends on.
    from speed_lstm.lift3d import ground_point
    a, b = np.array([0.0, 0.0, 0.0, 1.0]), np.array([20.0, 0.0, 0.0, 1.0])
    ua, ub = (P @ a), (P @ b)
    ga = ground_point(ua[0] / ua[2], ua[1] / ua[2], cal.P)
    gb = ground_point(ub[0] / ub[2], ub[1] / ub[2], cal.P)
    assert np.linalg.norm(gb - ga) == pytest.approx(20.0, rel=0.05)


def test_calibration_json_is_a_brno_system_file(tmp_path):
    dets = synthetic_dets(true_P())
    cal = autocalib.calibrate(dets, (W, H), vp2_candidates=noisy_vp_candidates(VP2, 100, 1.0, 0.0, 4),
                              max_boxes=60)
    path = tmp_path / "auto_calib.json"
    cal.save(path)
    calib, cars = brno.load_system(path)
    assert cars == []
    P = brno.projection_from_calibration(calib["vp1"], calib["vp2"], calib["pp"], calib["scale"])
    assert np.allclose(P, cal.P)


def test_missing_vp2_is_an_error():
    with pytest.raises(ValueError, match="VP2"):
        autocalib.calibrate(synthetic_dets(true_P(), n_cars=15), (W, H))
