"""
BrnoCompSpeed adapter: calibration -> metric P, and the dataset's official
speed evaluation (ported from github.com/JakubSochor/BrnoCompSpeed, Python 2)
applied to video_speed tracks.

Calibration: BrnoCompSpeed calibrations are two vanishing points (vp1 along
the traffic, vp2 across it), the principal point and a `scale`. The official
code puts the camera at the principal point with the image plane `focal`
pixels in front of it, intersects pixel rays with the road plane
n . X + 10 = 0, and multiplies distances on that plane by `scale` to get
metres. `projection_from_calibration` builds the equivalent 3x4 P for this
repo's road frame (x along vp1, y across, z up, metres, camera above the
origin), so lift3d / the v2 model work on Brno video unchanged.

Evaluation (same rules as the official eval.py):
  - a track's reference point is its fitted cuboid's ground center, projected
    into the image (the official format wants a point on the road plane);
  - its crossing time and spot for each measurement line come from local
    linear fits around the line;
  - it is matched to the ground-truth car whose last-line crossing is within
    MAX_TIME_DIFF seconds, in the same lane;
  - "full" speed = road distance between the first and last line crossings /
    time, which is how Brno systems are scored. The v2 model's speed for the
    same car is the median of its windows between those crossings.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

WIDTH, HEIGHT = 1920, 1080
SAFE_BORDER_OFFSET = 10
MAX_TIME_DIFF = 0.2  # seconds
ROAD_PLANE_OFFSET = 10.0  # the official code's arbitrary road-plane distance, in its own units


# ------------------------------------------------------------------ calibration

def _h(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return p if p.shape[-1] == 3 else np.append(p, 1.0)


def compute_camera_calibration(vp1, vp2, pp) -> tuple[np.ndarray, float]:
    """Official computeCameraCalibration: (roadPlane [n, 10], focal)."""
    vp1, vp2, pp = (np.asarray(v, dtype=np.float64)[:2] for v in (vp1, vp2, pp))
    focal = float(np.sqrt(-np.dot(vp1 - pp, vp2 - pp)))
    pp_w = np.append(pp, 0.0)
    vp3_w = np.cross(np.append(vp1, focal) - pp_w, np.append(vp2, focal) - pp_w)
    vp3 = vp3_w[:2] / vp3_w[2] * focal + pp
    vp3_dir = np.append(vp3, focal) - pp_w
    return np.append(vp3_dir / np.linalg.norm(vp3_dir), ROAD_PLANE_OFFSET), focal


def road_plane_point(p, focal: float, road_plane: np.ndarray, pp) -> np.ndarray:
    """Official getWorldCoordinagesOnRoadPlane (unscaled units)."""
    p = _h(p)
    p = p / p[2]
    pp_w = np.append(np.asarray(pp, dtype=np.float64)[:2], 0.0)
    direction = np.append(p[:2], focal) - pp_w
    t = -np.dot(road_plane, np.append(pp_w, 1.0)) / np.dot(road_plane[:3], direction)
    return pp_w + t * direction


def projection_from_calibration(vp1, vp2, pp, scale: float) -> np.ndarray:
    """Metre-space 3x4 P in this repo's road frame (x toward vp1, z up, camera at (0, 0, h))."""
    road_plane, focal = compute_camera_calibration(vp1, vp2, pp)
    pp = np.asarray(pp, dtype=np.float64)[:2]
    K = np.array([[focal, 0.0, pp[0]], [0.0, focal, pp[1]], [0.0, 0.0, 1.0]])

    # Official camera coords are (pixel - pp, focal) with the camera at pp_w; shift to camera-centred.
    # Its plane n . Xc + d = 0 always has n pointing forward (n_z > 0) and is the mirror image of the road
    # through the camera centre (same distances, behind the camera). For a camera tilted down at the road,
    # the physical "down" also points forward, so the real road is at the same distance along n: up = -n.
    n = road_plane[:3]
    d = float(np.dot(n, np.append(pp, 0.0)) + road_plane[3])
    up = -n
    height_m = scale * abs(d)

    along = np.append(np.asarray(vp1, dtype=np.float64)[:2] - pp, focal)
    along -= np.dot(along, up) * up
    along /= np.linalg.norm(along)
    across = np.cross(up, along)
    R = np.stack([along, across, up], axis=1)  # world axes in camera coords
    return K @ np.hstack([R, (-height_m * up)[:, None]])


def load_system(path: str | Path) -> tuple[dict, list[dict]]:
    """A result json from the dataset's results/ dir: (camera_calibration, that system's own tracked cars)."""
    import json

    with open(path) as f:
        data = json.load(f)
    return data.get("camera_calibration", data), data.get("cars", [])


# ------------------------------------------------------------------ ground truth

def load_gt(path: str | Path) -> dict:
    """gt_data.pkl is a Python 2 pickle with numpy arrays."""
    with open(path, "rb") as f:
        gt = pickle.load(f, encoding="latin1")
    gt["measurementLines"] = [np.asarray(l, dtype=np.float64) for l in gt["measurementLines"]]
    gt["laneDivLines"] = [np.asarray(l, dtype=np.float64) for l in gt["laneDivLines"]]
    gt["invalidLanes"] = set(gt.get("invalidLanes", set()))
    return gt


def distance_check(gt: dict, calib: dict) -> pd.DataFrame:
    """Official calibration check: road distance of each hand-measured segment vs. its true length."""
    road_plane, focal = compute_camera_calibration(calib["vp1"], calib["vp2"], calib["pp"])
    rows = []
    for m in gt["distanceMeasurement"]:
        a, b = (road_plane_point(m[k], focal, road_plane, calib["pp"]) for k in ("p1", "p2"))
        rows.append({"toVP1": bool(m["toVP1"]), "true_m": float(m["distance"]),
                     "measured_m": float(calib["scale"] * np.linalg.norm(a - b))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ official helpers

def _point_line_distance(p, l) -> float:
    p = _h(p)
    return abs(np.dot(l, p / p[2])) / np.linalg.norm(l[:2])


def _point_line_projection(l, p) -> np.ndarray:
    p = _h(p)
    p = p / p[-1]
    c = p[0] * l[1] - p[1] * l[0]
    x = np.cross(l, np.array([-l[1], l[0], c]))
    return x / x[-1]


def _between_lines(p, l1, l2) -> bool:
    p = _h(p)
    return np.dot(p, l1) * np.dot(p, l2) * np.dot(l1[:2], l2[:2]) <= 0


def lane_for_point(p, lines) -> int | None:
    for i in range(len(lines) - 1):
        if _between_lines(p, lines[i], lines[i + 1]):
            return i
    return None


def _linregress(x, y) -> tuple[float, float]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    vx = np.var(x)
    if vx == 0:
        return np.nan, np.nan
    slope = np.cov(x, y, bias=True)[0, 1] / vx
    return float(slope), float(y.mean() - slope * x.mean())


def line_crossing(line, pos_x, pos_y, frames, around: int = 6) -> tuple[np.ndarray, float]:
    """Official getCarTimeAndSpatialIntersection: (image point, frame) where the track crosses `line`."""
    pts = sorted(zip(pos_x, pos_y, frames), key=lambda i: _point_line_distance([i[0], i[1], 1.0], line))
    pts = pts[:min(around, len(pts))]
    slope, intercept = _linregress([p[0] for p in pts], [p[1] for p in pts])
    spatial_line = np.array([slope, -1.0, intercept])
    with np.errstate(all="ignore"):
        spatial = np.cross(spatial_line, line)
        spatial = spatial / spatial[-1]
        norm_pt = np.cross(spatial_line, [0.0, 1.0, 0.0])
        norm_pt = norm_pt / norm_pt[-1]
        dists = [np.linalg.norm(norm_pt - _point_line_projection(spatial_line, [p[0], p[1], 1.0])) for p in pts]
    t_slope, t_intercept = _linregress(dists, [p[2] for p in pts])
    return spatial, t_slope * np.linalg.norm(spatial - norm_pt) + t_intercept


# -------------------------------------------------------------- our tracks -> cars

def tracks_to_cars(lifted: pd.DataFrame, P: np.ndarray, gt_fps: float) -> list[dict]:
    """
    Official result-JSON cars from lifted tracks: the cuboid's ground center
    projected into the image, with frames in ground-truth fps units
    (lifted["timestamp"] is seconds since the start of the video).
    """
    cars = []
    for tid, g in lifted.groupby("track_id", sort=True):
        g = g.sort_values("timestamp")
        ground = np.column_stack([g["cx"], g["cy"], np.zeros(len(g)), np.ones(len(g))])
        uvw = ground @ P.T
        ok = uvw[:, 2] > 1e-6
        if ok.sum() == 0:
            continue
        uv = uvw[ok, :2] / uvw[ok, 2:]
        cars.append({"id": int(tid), "posX": uv[:, 0].tolist(), "posY": uv[:, 1].tolist(),
                     "frames": (g["timestamp"].to_numpy()[ok] * gt_fps).tolist()})
    return cars


def prefilter(cars: list[dict], gt: dict) -> list[dict]:
    """Official prefilterData."""
    fps = gt["fps"]
    last_gt = max(c["intersections"][-1]["videoTime"] for c in gt["cars"])
    out = []
    for car in cars:
        if not isinstance(car.get("posX"), list) or not car["frames"] or car["frames"][0] / fps >= last_gt:
            continue
        keep = [i for i, (x, y) in enumerate(zip(car["posX"], car["posY"]))
                if SAFE_BORDER_OFFSET < x < WIDTH - SAFE_BORDER_OFFSET and SAFE_BORDER_OFFSET < y < HEIGHT - SAFE_BORDER_OFFSET]
        car = {**car, **{k: [car[k][i] for i in keep] for k in ("posX", "posY", "frames")}}
        if len(car["frames"]) <= 5:
            continue
        lanes = [lane_for_point([x, y, 1.0], gt["laneDivLines"]) for x, y in zip(car["posX"], car["posY"])]
        if None in lanes or (gt["invalidLanes"] and any(l in gt["invalidLanes"] for l in lanes)):
            continue
        between = sum(1 for x, y in zip(car["posX"], car["posY"])
                      if 0 <= x <= WIDTH and 0 <= y <= HEIGHT
                      and _between_lines([x, y, 1.0], gt["measurementLines"][0], gt["measurementLines"][-1]))
        if between >= 6:
            out.append(car)
    return out


def calculate_speeds(cars: list[dict], gt: dict, calib: dict) -> int:
    """Official calculateSpeeds ("full" and "median" speeds, km/h). Returns the number of failed cars."""
    road_plane, focal = compute_camera_calibration(calib["vp1"], calib["vp2"], calib["pp"])
    project = lambda p: road_plane_point(p, focal, road_plane, calib["pp"])
    lines, fps, scale = gt["measurementLines"], gt["fps"], calib["scale"]
    errors = 0
    for car in cars:
        crossings = [line_crossing(l, car["posX"], car["posY"], car["frames"]) for l in lines]
        (start_sp, start_f), (end_sp, end_f) = crossings[-1], crossings[0]
        test = np.concatenate([end_sp, start_sp, [start_f, end_f]])
        if (not np.all(np.isfinite(test)) or lane_for_point(end_sp, gt["laneDivLines"]) is None
                or lane_for_point(start_sp, gt["laneDivLines"]) is None or end_f == start_f):
            errors += 1
            continue
        elapsed = abs(end_f - start_f) / fps
        car["speed"] = scale * np.linalg.norm(project(start_sp) - project(end_sp)) / elapsed * 3.6
        car["laneIndex"] = lane_for_point(end_sp, gt["laneDivLines"])
        car["timeIntersectionLast"] = end_f / fps
        car["timeIntersectionFirst"] = start_f / fps
        pts = [project([x, y, 1.0]) for x, y in zip(car["posX"], car["posY"])]
        k = 5
        per_frame = [scale * np.linalg.norm(pts[i] - pts[i + k]) / (abs(car["frames"][i] - car["frames"][i + k]) / fps) * 3.6
                     for i in range(len(pts) - k) if car["frames"][i] != car["frames"][i + k]]
        car["medianSpeed"] = float(np.median(per_frame)) if per_frame else np.nan
    return errors


def compute_matches(gt: dict, cars: list[dict], t_max: float | None = None) -> list[dict]:
    """
    Official computeMatches, keeping both measurement modes. t_max limits the
    ground truth to cars that finished crossing within the processed video.
    """
    measured = [c for c in cars if "laneIndex" in c]
    matches = []
    for g in gt["cars"]:
        t_last = g["intersections"][-1]["videoTime"]
        if t_max is not None and t_last > t_max:
            continue
        lanes = set(g["laneIndex"])
        cands = sorted((c for c in measured if c["laneIndex"] in lanes),
                       key=lambda c: abs(c["timeIntersectionLast"] - t_last))
        row = {"gt_id": g["carId"], "valid": bool(g["valid"]), "gt_kmh": float(g["speed"]),
               "t_first": g["intersections"][0]["videoTime"], "t_last": t_last, "matched": False}
        if cands and abs(cands[0]["timeIntersectionLast"] - t_last) < MAX_TIME_DIFF:
            c = cands[0]
            row.update(matched=True, track_id=c["id"], full_kmh=c["speed"], median_kmh=c["medianSpeed"],
                       track_t_first=c["timeIntersectionFirst"], track_t_last=c["timeIntersectionLast"])
        matches.append(row)
    return matches


def model_speed_for_matches(matches: pd.DataFrame, preds: pd.DataFrame) -> pd.Series:
    """Median model speed (km/h) of the matched track's windows that overlap its line-to-line pass."""
    out = pd.Series(np.nan, index=matches.index)
    by_track = {tid: g for tid, g in preds.groupby("track_id")}
    for i, r in matches[matches["matched"]].iterrows():
        g = by_track.get(r["track_id"])
        if g is None:
            continue
        lo, hi = sorted((r["track_t_first"], r["track_t_last"]))
        inside = g[(g["t_end"] >= lo) & (g["t_start"] <= hi)]
        out[i] = (inside if len(inside) else g)["speed_kmh"].median()
    return out


def error_stats(errors: np.ndarray) -> dict:
    e = np.abs(np.asarray(errors, dtype=np.float64))
    e = e[np.isfinite(e)]
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "mean": float(e.mean()), "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)), "worst": float(e.max())}
