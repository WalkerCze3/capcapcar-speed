import numpy as np

from speed_lstm.data import load_all_windows, project_to_bbox
from tests.helpers import default_vehicle, make_scene


def test_timestamp_join_and_windowing(tmp_path):
    make_scene(tmp_path, "scene1", [default_vehicle(0, n_frames=32)])
    windows = load_all_windows(tmp_path, ["scene1"], require_2d=True)
    assert len(windows) > 0
    w = windows[0]
    assert w.timestamps.shape == (16,)
    assert np.all(np.diff(w.timestamps) > 0)
    # ts.csv values were 1.7e9 + i*0.033 for this single camera -> spot check magnitude
    assert 1_700_000_000.0 <= w.timestamps[0] < 1_700_000_010.0


def test_window_stride_and_count(tmp_path):
    # 32 frames, window 16, stride 8 -> starts at 0,8,16 -> 3 windows for one vehicle/camera
    make_scene(tmp_path, "scene1", [default_vehicle(0, n_frames=32)])
    windows = load_all_windows(tmp_path, ["scene1"], require_2d=True)
    assert len(windows) == 3


def test_missing_frame_indices_reject_window(tmp_path):
    make_scene(tmp_path, "scene1", [default_vehicle(0, n_frames=16)])
    ann_path = tmp_path / "obj" / "scene1_annotations.csv"
    import pandas as pd
    df = pd.read_csv(ann_path)
    df = df[df["frame"] != 5]  # punch a hole in the middle of the only window
    df.to_csv(ann_path, index=False)

    windows = load_all_windows(tmp_path, ["scene1"], require_2d=True)
    assert len(windows) == 0


def test_timestamp_gap_rejects_window(tmp_path):
    make_scene(tmp_path, "scene1", [default_vehicle(0, n_frames=16)])
    ts_path = tmp_path / "ts" / "scene1_ts.csv"
    import pandas as pd
    ts = pd.read_csv(ts_path)
    ts.loc[ts["frame"] >= 8, "p1c1"] += 1.0  # blow a > 0.2s gap into the middle
    ts.to_csv(ts_path, index=False)

    windows = load_all_windows(tmp_path, ["scene1"], require_2d=True)
    assert len(windows) == 0


def test_vehicle_and_camera_isolation(tmp_path):
    make_scene(
        tmp_path, "scene1",
        [default_vehicle(0, camera="p1c1", n_frames=16, x0_ft=0),
         default_vehicle(1, camera="p1c1", n_frames=16, x0_ft=300),
         default_vehicle(2, camera="p1c2", n_frames=16, x0_ft=0)],
        cameras=["p1c1", "p1c2"],
    )
    windows = load_all_windows(tmp_path, ["scene1"], require_2d=True)
    keys = {(w.camera, w.vehicle_id) for w in windows}
    assert keys == {("p1c1", 0), ("p1c1", 1), ("p1c2", 2)}
    for w in windows:
        assert len(set(zip([w.camera] * 16, [w.vehicle_id] * 16))) == 1  # trivially true, real check is `keys` above


def test_horizon_crossing_rejected():
    # X's w-column coefficient is -1, so w = 50 - X: positive for small X, negative for large X.
    P = np.array([[1, 0, -1], [0, 1, 0], [0, 0, 0], [0, 0, 50]], dtype=np.float64)
    near_center = np.array([1.0, 1.0, 1.0])
    near_dims = np.array([4.0, 2.0, 1.5])
    far_center = np.array([200.0, 1.0, 1.0])  # X=200 -> w = 50-200 < 0 for corners near this X
    far_dims = np.array([4.0, 2.0, 1.5])

    assert project_to_bbox(near_center, near_dims, P) is not None
    assert project_to_bbox(far_center, far_dims, P) is None
