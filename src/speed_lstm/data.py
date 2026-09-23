"""
Read I-24 annotations, per-camera corrected timestamps, and camera
homography/projection files; convert annotated boxes to metric
(rear-ground-center, flat-road) coordinates; project cuboids into image
space for the 2D feature path; and build fixed-length observation windows
with their speed targets.

Coordinate convention (see ARCHITECTURE.md section 2): the annotated (x, y)
is the rear ground-center of the vehicle, direction d in {-1, +1} indicates
which way the vehicle points, and the road is assumed flat at z = 0:

    center_x = (x + d * length / 2) * FEET_TO_METERS
    center_y = y * FEET_TO_METERS
    center_z = (height / 2) * FEET_TO_METERS

A window is 16 consecutive same-(scene, camera, vehicle) observations
(stride 8 between window starts). It is rejected for non-consecutive frame
indices, non-increasing timestamps, a timestamp gap > MAX_TIMESTAMP_GAP,
non-finite or non-positive geometry, or (2D-requiring modes only) a corner
projecting past the camera's horizon at any observation in the window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEET_TO_METERS = 0.3048
WINDOW_LEN = 16
WINDOW_STRIDE = 8
MAX_TIMESTAMP_GAP = 0.2  # seconds

REQUIRED_ANNOTATION_COLS = ["frame", "camera", "id", "x", "y", "l", "w", "h", "direction"]

# 8 corner sign combinations for a box centered at `center` with half-extents
# `half`; z in {-1, +1} maps to {0, height} because center_z is already
# height/2 above the road.
_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float64,
)


@dataclass
class Window:
    scene: str
    camera: str
    vehicle_id: int
    frames: np.ndarray          # (16,) int
    timestamps: np.ndarray      # (16,) float64 seconds
    metric_center: np.ndarray   # (16, 3) float64 meters [x, y, z]
    metric_dims: np.ndarray     # (16, 3) float64 meters [length, width, height]
    direction: int
    box2d: np.ndarray | None    # (16, 4) float32 [xmin, ymin, xmax, ymax], or None
    target_speed: float         # m/s, mean path speed over the full 16-point window


def load_annotations(data_dir: Path, scene: str) -> pd.DataFrame:
    path = Path(data_dir) / "obj" / f"{scene}_annotations.csv"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_ANNOTATION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    return df


def load_timestamps(data_dir: Path, scene: str) -> pd.DataFrame:
    """Long-format (frame, camera, timestamp), melted from the per-camera-column ts csv."""
    path = Path(data_dir) / "ts" / f"{scene}_ts.csv"
    ts_wide = pd.read_csv(path)
    camera_cols = [c for c in ts_wide.columns if c != "frame"]
    return ts_wide.melt(id_vars="frame", value_vars=camera_cols, var_name="camera", value_name="timestamp")


def load_homography(data_dir: Path, scene: str) -> dict:
    path = Path(data_dir) / "hg" / f"{scene}_hg.json"
    with open(path) as f:
        return json.load(f)


def metric_center_and_dims(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    d = float(row["direction"])
    length, width, height = float(row["l"]), float(row["w"]), float(row["h"])
    center = np.array([
        (row["x"] + d * length / 2.0) * FEET_TO_METERS,
        row["y"] * FEET_TO_METERS,
        (height / 2.0) * FEET_TO_METERS,
    ], dtype=np.float64)
    dims = np.array([length, width, height], dtype=np.float64) * FEET_TO_METERS
    return center, dims


def cuboid_corners(center: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """8 world-space corners (meters), axes aligned to the road frame (x=along, y=across, z=up)."""
    half = dims / 2.0
    return center[None, :] + _CORNER_SIGNS * half[None, :]


def project_points(points_xyz: np.ndarray, P: np.ndarray) -> np.ndarray | None:
    """
    points_xyz: (N, 3) world points, meters.
    P: (4, 3) space->image projection matrix, row-vector convention:
       image_homog = [X, Y, Z, 1] @ P.
    Returns (N, 2) image-plane points, or None if any point is at/behind the
    camera (homogeneous w <= epsilon) — i.e. crosses the horizon.
    """
    n = points_xyz.shape[0]
    homog = np.concatenate([points_xyz, np.ones((n, 1))], axis=1)  # (N, 4)
    img_homog = homog @ P  # (N, 3)
    w = img_homog[:, 2]
    if np.any(w <= 1e-6):
        return None
    return img_homog[:, :2] / w[:, None]


def project_to_bbox(center: np.ndarray, dims: np.ndarray, P: np.ndarray) -> np.ndarray | None:
    """Axis-aligned xyxy box from the min/max of the 8 projected cuboid corners, or None on horizon crossing."""
    img_pts = project_points(cuboid_corners(center, dims), P)
    if img_pts is None:
        return None
    xmin, ymin = img_pts.min(axis=0)
    xmax, ymax = img_pts.max(axis=0)
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)


def build_vehicle_tracks(ann: pd.DataFrame, ts_long: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    """Group by (camera, vehicle id), timestamp-joined, sorted by frame, deduplicated by frame."""
    merged = ann.merge(ts_long, on=["frame", "camera"], how="left")
    merged = merged.dropna(subset=["timestamp"])
    merged = merged.sort_values("frame").drop_duplicates(subset=["camera", "id", "frame"], keep="first")

    tracks = {}
    for (camera, vid), g in merged.groupby(["camera", "id"], sort=False):
        tracks[(camera, vid)] = g.sort_values("frame").reset_index(drop=True)
    return tracks


def _is_valid_window(g: pd.DataFrame) -> bool:
    if len(g) != WINDOW_LEN:
        return False

    frames = g["frame"].to_numpy()
    if not np.all(np.diff(frames) == 1):
        return False  # missing frame indices within the window

    ts = g["timestamp"].to_numpy(dtype=np.float64)
    if not np.all(np.diff(ts) > 0):
        return False  # non-increasing timestamps
    if np.any(np.diff(ts) > MAX_TIMESTAMP_GAP):
        return False

    for col in ("x", "y", "l", "w", "h"):
        vals = g[col].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            return False
        if col in ("l", "w", "h") and np.any(vals <= 0):
            return False

    return True


def build_windows(tracks: dict, scene: str, hg: dict, need_2d: bool) -> list[Window]:
    windows: list[Window] = []
    for (camera, vid), g in tracks.items():
        P = None
        if need_2d:
            # hg.json is keyed by direction group first ('EB'/'WB', matching
            # the annotations' direction column: +1=EB, -1=WB), then by
            # camera name -- each direction group has its own per-camera
            # homography/projection, not a single shared one.
            direction_key = "EB" if int(g["direction"].iloc[0]) == 1 else "WB"
            cam_hg = hg.get(direction_key, {}).get(camera)
            if cam_hg is None:
                continue
            P = np.array(cam_hg["P"], dtype=np.float64)

        n = len(g)
        for start in range(0, max(n - WINDOW_LEN + 1, 0), WINDOW_STRIDE):
            win_df = g.iloc[start:start + WINDOW_LEN]
            if not _is_valid_window(win_df):
                continue

            centers = np.zeros((WINDOW_LEN, 3), dtype=np.float64)
            dims = np.zeros((WINDOW_LEN, 3), dtype=np.float64)
            box2d = np.zeros((WINDOW_LEN, 4), dtype=np.float32) if need_2d else None

            ok = True
            for i in range(WINDOW_LEN):
                row = win_df.iloc[i]
                c, d = metric_center_and_dims(row)
                centers[i], dims[i] = c, d
                if need_2d:
                    bbox = project_to_bbox(c, d, P)
                    if bbox is None:
                        # A corner crosses the horizon for this observation.
                        # We reject the WINDOW (not the whole camera/vehicle
                        # track) so one bad frame doesn't discard otherwise
                        # valid data elsewhere in a long track.
                        ok = False
                        break
                    box2d[i] = bbox
            if not ok:
                continue

            ts = win_df["timestamp"].to_numpy(dtype=np.float64)
            xy = centers[:, :2]
            step_dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            target_speed = float(step_dist.sum() / (ts[-1] - ts[0]))

            windows.append(Window(
                scene=scene, camera=camera, vehicle_id=int(vid),
                frames=win_df["frame"].to_numpy(),
                timestamps=ts,
                metric_center=centers, metric_dims=dims,
                direction=int(win_df["direction"].iloc[0]),
                box2d=box2d,
                target_speed=target_speed,
            ))
    return windows


def load_scene_windows(data_dir: Path, scene: str, need_2d: bool) -> list[Window]:
    ann = load_annotations(data_dir, scene)
    ts_long = load_timestamps(data_dir, scene)
    hg = load_homography(data_dir, scene) if need_2d else {}
    tracks = build_vehicle_tracks(ann, ts_long)
    return build_windows(tracks, scene, hg, need_2d)


def load_all_windows(data_dir: str | Path, scenes: list[str], require_2d: bool = True) -> list[Window]:
    """
    require_2d=True (the default) always applies the 2D-projection filter,
    even when training a 3D-only model — this keeps the window population
    (and therefore the speed-balanced split) identical across 2D/3D/combined
    training runs, so their test metrics are comparable. Only pass False if
    you specifically want a larger, 3D-only window population that isn't
    comparable to the 2D/combined runs.
    """
    windows: list[Window] = []
    for scene in scenes:
        windows.extend(load_scene_windows(Path(data_dir), scene, require_2d))
    return windows
