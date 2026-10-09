import numpy as np
import pandas as pd
import torch

from speed_lstm import features_v2 as fv2
from speed_lstm.model import Predictor, SpeedLSTM
from speed_lstm.video import (TRAJECTORY_FEATURE_COLS, build_video_windows, predict_windows, summarize_tracks,
                              track_trajectories, trajectory_features)
from tests.test_lift3d import IMG_H, IMG_W, pole_camera, synthetic_detections


def test_track_trajectories_keeps_raw_boxes_and_drops_truncated():
    dets = synthetic_detections(pole_camera(), n_frames=40)
    dets.loc[dets["frame"] < 5, "xmin"] = 0.0  # clipped at the left image edge
    traj = track_trajectories(dets, np.arange(40) / 30.0, (IMG_W, IMG_H))
    assert traj["frame"].tolist() == list(range(5, 40))
    np.testing.assert_allclose(traj["timestamp"], np.arange(5, 40) / 30.0)
    kept = dets[dets["frame"] >= 5].reset_index(drop=True)
    np.testing.assert_allclose(traj[["xmin", "ymin", "xmax", "ymax"]], kept[["xmin", "ymin", "xmax", "ymax"]])


def test_trajectory_features_values():
    # Box grows 10% per frame while its center moves 6 px right and its bottom 4 px down, at 10 fps.
    rows = []
    for f in range(3):
        w, h = 50 * 1.1 ** f, 40 * 1.1 ** f
        cx, bottom = 200 + 6 * f, 300 + 4 * f
        rows.append({"track_id": 1, "frame": f, "timestamp": f / 10, "cls": "car",
                     "xmin": cx - w / 2, "ymin": bottom - h, "xmax": cx + w / 2, "ymax": bottom})
    rows.append({"track_id": 2, "frame": 0, "timestamp": 0.0, "cls": "car",
                 "xmin": 10.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0})
    feats = trajectory_features(pd.DataFrame(rows), (1000, 500))

    assert list(feats.columns[-len(TRAJECTORY_FEATURE_COLS):]) == TRAJECTORY_FEATURE_COLS
    first, second, other = feats.iloc[0], feats.iloc[1], feats.iloc[3]
    assert np.isnan(first["dx_px"]) and np.isnan(other["dx_px"])  # diffs never cross tracks
    np.testing.assert_allclose([first["pos_x"], first["pos_y"], first["rel_height"]], [0.2, 0.6, 0.08])
    np.testing.assert_allclose([second["dx_px"], second["dbottom_px"], second["frame_gap"]], [6, 4, 1])
    np.testing.assert_allclose([second["vx_px_s"], second["vbottom_px_s"]], [60, 40])
    np.testing.assert_allclose(second["vx_per_width_s"], 60 / 55)
    np.testing.assert_allclose([second["dlog_width_s"], second["dlog_height_s"], second["dlog_area_s"]],
                               [10 * np.log(1.1)] * 2 + [20 * np.log(1.1)])


def test_calibration_free_windows_run_a_2d_checkpoint(tmp_path):
    dets = synthetic_detections(pole_camera(), n_frames=40)
    traj = track_trajectories(dets, np.arange(40) / 30.0, (IMG_W, IMG_H))
    windows = build_video_windows(traj, n_observations=16, stride=8)
    assert len(windows) == 4 and "boxes3d" not in windows[0]

    n = fv2.n_features("2d")
    model = SpeedLSTM(input_size=n)
    torch.save({"model_state": model.state_dict(), "mode": "2d", "input_size": n, "hidden_size": model.hidden_size,
                "feature_version": "v2", "n_observations": 16, "max_timestamp_gap": 0.2,
                "feature_mean": np.zeros(n), "feature_std": np.ones(n), "target_mean": 20.0, "target_std": 5.0},
               tmp_path / "best.pt")
    preds = predict_windows(Predictor(tmp_path / "best.pt", device="cpu"), windows)
    assert len(preds) == 4 and (preds["speed_mps"] >= 0).all() and preds["geometric_mps"].isna().all()
    summary = summarize_tracks(preds, traj)
    assert summary["track_id"].tolist() == [7] and summary["cls"].tolist() == ["car"]
