import json

import numpy as np
import pandas as pd
import torch

from speed_lstm import features_v2 as fv2
from speed_lstm.data import FEET_TO_METERS, project_to_bbox
from speed_lstm.lift3d import DIM_PRIORS, fit_cuboid, fit_track, ground_point, load_projections, scale_projection
from speed_lstm.model import Predictor, SpeedLSTM
from speed_lstm.video import build_video_windows, lift_tracks, predict_windows

IMG_W, IMG_H = 1920, 1080
CAR_DIMS = np.array(DIM_PRIORS["car"][0])


def pole_camera() -> np.ndarray:
    """Pinhole camera 10 m up beside the road, looking down-road. Metre-space (3, 4) P."""
    K = np.array([[1400.0, 0, IMG_W / 2], [0, 1400.0, IMG_H / 2], [0, 0, 1]])
    C = np.array([-15.0, -12.0, 10.0])
    target = np.array([25.0, 2.0, 0.0])
    fwd = (target - C) / np.linalg.norm(target - C)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # camera x=right, y=down, z=forward
    return K @ np.hstack([R, (-R @ C)[:, None]])


def car_center(x, y=0.0):
    return np.array([x, y, CAR_DIMS[2] / 2])


def test_ground_point_roundtrip():
    P = pole_camera()
    for xy in ([5.0, 0.0], [20.0, 3.5], [35.0, -2.0]):
        uvw = P @ np.array([*xy, 0.0, 1.0])
        np.testing.assert_allclose(ground_point(uvw[0] / uvw[2], uvw[1] / uvw[2], P), xy, atol=1e-6)


def test_fit_cuboid_fixed_dims_recovers_center():
    P = pole_camera()
    true = car_center(18.0, 1.5)
    bbox = project_to_bbox(true, CAR_DIMS, P)
    fit = fit_cuboid(bbox, P, "car", fixed_dims=CAR_DIMS)
    np.testing.assert_allclose(fit.center, true, atol=1e-3)
    assert fit.residual_px < 0.05


def test_fit_track_recovers_positions_with_free_dims():
    # True dims differ from the prior; two-pass fit should still place the car close to truth.
    P = pole_camera()
    dims = np.array([4.9, 1.9, 1.45])
    xs = np.linspace(2.0, 30.0, 40)
    centers = [np.array([x, 1.0, dims[2] / 2]) for x in xs]
    bboxes = np.stack([project_to_bbox(c, dims, P) for c in centers])
    fits = fit_track(bboxes, P, "car")
    err = np.array([np.linalg.norm(f.center[:2] - c[:2]) for f, c in zip(fits, centers)])
    assert err.max() < 0.5
    # Distance travelled (what speed depends on) is recovered far better than absolute position.
    travelled = fits[-1].center[0] - fits[0].center[0]
    assert abs(travelled - (xs[-1] - xs[0])) / (xs[-1] - xs[0]) < 0.03


def test_scale_projection_feet():
    P_ft = pole_camera()  # pretend this P expects feet
    P_m = scale_projection(P_ft, "ft")
    pt_m = np.array([3.0, 1.0, 0.5])
    a = P_m @ np.append(pt_m, 1.0)
    b = P_ft @ np.append(pt_m / FEET_TO_METERS, 1.0)
    np.testing.assert_allclose(a, b)


def test_load_projections_hg_json(tmp_path):
    P = pole_camera().tolist()
    path = tmp_path / "hg.json"
    path.write_text(json.dumps({"EB": {"p1c1": {"P": P}}, "WB": {"p1c1": {"P": P}, "p1c2": {"P": P}}}))
    assert set(load_projections(path, "p1c1", "m")) == {1, -1}
    assert set(load_projections(path, "p1c2", "m")) == {-1}

    single = tmp_path / "cam.json"
    single.write_text(json.dumps({"P": P}))
    np.testing.assert_allclose(load_projections(single, units="m")[1], P)


def synthetic_detections(P, speed_mps=20.0, fps=30.0, n_frames=40, direction=1, drop_frames=()):
    rows = []
    for f in range(n_frames):
        if f in drop_frames:
            continue
        x = 5.0 + direction * speed_mps * f / fps if direction == 1 else 30.0 - speed_mps * f / fps
        b = project_to_bbox(car_center(x, 1.0), CAR_DIMS, P)
        rows.append({"frame": f, "track_id": 7, "cls": "car", "conf": 0.9,
                     "xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]})
    return pd.DataFrame(rows)


def test_lift_and_window_end_to_end(tmp_path):
    P = pole_camera()
    fps, speed = 30.0, 20.0
    dets = synthetic_detections(P, speed, fps, n_frames=40)
    ts = np.arange(40) / fps
    lifted = lift_tracks(dets, ts, {1: P, -1: P}, (IMG_W, IMG_H), smooth_window=1)

    assert len(lifted) == 40
    assert (lifted["direction"] == 1).all()  # x increases -> EB

    windows = build_video_windows(lifted, n_observations=16, stride=8)
    assert len(windows) == 4  # starts 0, 8, 16, 24
    assert np.asarray(windows[0]["boxes3d"]).shape == (16, 6)
    assert np.asarray(windows[0]["boxes2d"]).shape == (16, 4)

    # Model-ready: a random-init checkpoint accepts these windows end to end.
    mode = "combined"
    n = fv2.n_features(mode)
    model = SpeedLSTM(input_size=n)
    torch.save({"model_state": model.state_dict(), "mode": mode, "input_size": n, "hidden_size": model.hidden_size,
                "feature_version": "v2", "n_observations": 16, "max_timestamp_gap": 0.2,
                "feature_mean": np.zeros(n), "feature_std": np.ones(n), "target_mean": 20.0, "target_std": 5.0},
               tmp_path / "best.pt")
    preds = predict_windows(Predictor(tmp_path / "best.pt", device="cpu"), windows)
    assert len(preds) == 4 and (preds["speed_mps"] >= 0).all()
    np.testing.assert_allclose(preds["geometric_mps"], speed, rtol=0.03)


def test_westbound_track_picks_wb_projection():
    P = pole_camera()
    dets = synthetic_detections(P, direction=-1)
    lifted = lift_tracks(dets, np.arange(40) / 30.0, {1: P, -1: P}, (IMG_W, IMG_H))
    assert (lifted["direction"] == -1).all()


def test_frame_gap_and_truncation_break_windows():
    P = pole_camera()
    dets = synthetic_detections(P, n_frames=40, drop_frames={20})
    lifted = lift_tracks(dets, np.arange(40) / 30.0, {1: P, -1: P}, (IMG_W, IMG_H))
    windows = build_video_windows(lifted, n_observations=16, stride=8)
    for w in windows:
        assert np.all(np.diff(w["frames"]) == 1)
    assert len(windows) == 2  # [0..15] and [21..36]; everything spanning frame 20 is rejected

    edge = dets.copy()
    edge.loc[edge["frame"] < 10, "xmin"] = 0.0  # clipped at the left image edge
    lifted = lift_tracks(edge, np.arange(40) / 30.0, {1: P, -1: P}, (IMG_W, IMG_H))
    assert lifted["frame"].min() == 10


def test_calibration_check_picks_units_and_matches_tracks(tmp_path):
    from speed_lstm.data import load_annotations, load_homography, metric_center_and_dims
    from speed_lstm.video_eval import annotation_boxes, best_ious, calibration_check, camera_projections, match_tracks
    from tests.helpers import default_vehicle, make_scene

    P_m = pole_camera()
    P_ft = P_m @ np.diag([FEET_TO_METERS] * 3 + [1.0])  # I-24 style: P expects feet
    make_scene(tmp_path, "scene1", [default_vehicle(0, n_frames=30, x0_ft=10, y_ft=5),
                                    default_vehicle(1, n_frames=30, x0_ft=30, y_ft=-5)],
               P_by_camera={"p1c1": P_ft.tolist()})
    ann = load_annotations(tmp_path, "scene1")
    rows = []
    for _, r in ann.iterrows():
        b = project_to_bbox(*metric_center_and_dims(r), P_m)
        rows.append({"frame": r["frame"], "track_id": 50 + r["id"], "xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]})
    dets = pd.DataFrame(rows)
    hg = load_homography(tmp_path, "scene1")

    calib = calibration_check(dets, ann, hg, "p1c1")
    assert (calib.iloc[0]["units"], calib.iloc[0]["image_scale"]) == ("ft", 1.0)
    assert calib.iloc[0]["median_iou"] > 0.99

    matches = best_ious(dets, annotation_boxes(ann, camera_projections(hg, "p1c1", "ft", 1.0)))
    assert match_tracks(matches) == {50: 0, 51: 1}
