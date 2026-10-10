"""
Automatic camera calibration from passing traffic: tracked vehicles -> P, no manual input.

    tracks (ByteTrack boxes) --> VP1 from the lines the box centers travel along (Gaussian-sphere fit)
    vehicle crops --(VP CNN, speed_lstm.vp_cnn)--> per-vehicle VP1 / VP2 candidates --> robust aggregate
    VP1, VP2, principal point (image center) --> focal length + road plane (brno.compute_camera_calibration)
    scale: the one value at which the far cars' detector boxes are as high, relative to the box of a
           prior-sized car cuboid (lift3d.DIM_PRIORS) at the same spot, as on calibrated cameras
    --> brno.projection_from_calibration(vp1, vp2, pp, scale) = metre-space P

The result is saved in the BrnoCompSpeed results format ({"camera_calibration": {vp1, vp2, pp,
scale}}) plus "P" in metres, so the same file works as `--calib` for scripts/brno_eval_cli.py
(scored like any other calibration) and for video_speed_cli.py (with --calib-units m).

Vanishing points are handled as homogeneous image points so a VP at infinity (lines parallel in the
image) is no special case. For averaging they are turned into unit directions on the Gaussian sphere,
(u - ppx, v - ppy, f0) normalized, where a point and its antipode are the same VP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from speed_lstm import brno
from speed_lstm.lift3d import DIM_PRIORS, fit_cuboid

BOX2D_COLS = ["xmin", "ymin", "xmax", "ymax"]

# Quality gates for `AutoCalibration.reliable`.
MIN_VP1_LINES = 10
MIN_VP1_INLIER_FRAC = 0.5
MAX_VP1_COND_DEG = 5.0        # the track lines must fan out enough (vp1_conditioning_deg)
MAX_SPEED_DRIFT_PCT = 5.0     # cars must keep their speed (far half vs near half of each track) within 5%
MIN_VP2_SAMPLES = 30
MAX_SCALE_SPLIT_DIFF = 0.05   # scale from either half of the cars must agree within 5%
MAX_MEDIAN_RESIDUAL_PX = 4.0
MIN_SCALE_TRACKS = 20         # the size sample must come from many vehicles ...
MAX_SCALE_TRACK_SHARE = 0.2   # ... none of which (e.g. a parked car) supplies more than 20% of it
VP1_AGREE_DEG = 2.0           # track and CNN VP1 closer than this are averaged, else the speed check picks one
VP1_HEIGHT_FRAC = 0.7         # track point (see track_lines); fitted on sessions 1-2 (0.5-0.7 is flat)
VP1_SPREAD_FLOOR_DEG = 0.5    # floor on the split-half spread of either VP1 when averaging (bias is not in it)
MAX_SCALE_NEAR_FAR_DIFF = 0.25  # scale from the near and the far half of the boxes must agree within 25%

# Scale cue: a detector box is lower than the box of a prior-sized car cuboid at the same spot (a car is
# not a box: rounded roof, glass). Median log(detected height / prior-cuboid box height) over the far
# half of the car boxes, at the reference calibration of the training recordings (estimate_scale).
BOX_HEIGHT_LOG_RATIO = -0.1569  # fitted on BrnoCompSpeed sessions 1-2 (6 recordings)
FAR_FRACTION = 0.5


# ------------------------------------------------------------- VP geometry

def to_direction(vps_h: np.ndarray, pp, f0: float) -> np.ndarray:
    """Homogeneous image points (N, 3) -> unit directions (N, 3) on the Gaussian sphere."""
    vps_h = np.atleast_2d(np.asarray(vps_h, dtype=np.float64))
    d = np.stack([vps_h[:, 0] - pp[0] * vps_h[:, 2], vps_h[:, 1] - pp[1] * vps_h[:, 2], f0 * vps_h[:, 2]], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def from_direction(d: np.ndarray, pp, f0: float) -> np.ndarray:
    """Unit direction (3,) -> homogeneous image point (3,)."""
    d = np.asarray(d, dtype=np.float64)
    return np.array([d[0] + pp[0] * d[2] / f0, d[1] + pp[1] * d[2] / f0, d[2] / f0])


def to_point(vp_h: np.ndarray, far: float = 1e7) -> np.ndarray:
    """Homogeneous VP -> finite (u, v); a VP at infinity is pushed `far` pixels out along its direction."""
    vp_h = np.asarray(vp_h, dtype=np.float64)
    if abs(vp_h[2]) > 1e-12 * np.linalg.norm(vp_h[:2]):
        return vp_h[:2] / vp_h[2]
    return vp_h[:2] / np.linalg.norm(vp_h[:2]) * far


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two sign-ambiguous unit directions, degrees."""
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def aggregate_directions(dirs: np.ndarray, weights: np.ndarray | None = None, keep: float = 0.7,
                         n_iter: int = 5) -> tuple[np.ndarray, float]:
    """
    Robust mean of sign-ambiguous unit directions: the principal axis of sum(w d d^T), refit
    `n_iter` times on the `keep` fraction closest to the current estimate.
    Returns (direction, median angular deviation of the kept samples in degrees).
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    w = np.ones(len(dirs)) if weights is None else np.asarray(weights, dtype=np.float64)
    mask = np.ones(len(dirs), dtype=bool)
    est = None
    for _ in range(n_iter + 1):
        M = (dirs[mask] * w[mask, None]).T @ dirs[mask]
        est = np.linalg.eigh(M)[1][:, -1]
        ang = np.degrees(np.arccos(np.clip(np.abs(dirs @ est), 0.0, 1.0)))
        cut = np.quantile(ang, keep)
        mask = ang <= cut
    return est, float(np.median(ang[mask]))


def track_lines(dets: pd.DataFrame, img_size: tuple[int, int] | None = None, min_points: int = 8,
                min_travel_px: float = 40.0, max_rms_px: float = 2.0, height_frac: float = VP1_HEIGHT_FRAC,
                border_margin: float = 3.0) -> pd.DataFrame:
    """
    One homogeneous line per track through a fixed relative point of its boxes, (box center u,
    ymax - height_frac * box height), total least squares; kept when the vehicle travelled far enough
    in a straight enough line. With `img_size`, boxes clipped by the image border are dropped first
    (a box cut off at the bottom edge bends the track). height_frac = 0.5 (box center) averages the
    near and far extremal corners in both coordinates, so it stays closer to one 3D point than the
    bottom-center (u from two side corners, v from the nearest bottom corner), whose lines miss VP1.
    Columns: track_id, a, b, c (line a u + b v + c = 0, (a, b) unit), mu, mv (midpoint), travel_px, rms_px.
    """
    if img_size is not None:
        from speed_lstm.video import _drop_truncated

        dets = _drop_truncated(dets, img_size[0], img_size[1], border_margin)
    rows = []
    for tid, g in dets.groupby("track_id", sort=True):
        if len(g) < min_points:
            continue
        v = g["ymax"].to_numpy() - height_frac * (g["ymax"] - g["ymin"]).to_numpy()
        pts = np.stack([(g["xmin"] + g["xmax"]).to_numpy() / 2.0, v], axis=1).astype(np.float64)
        mid = pts.mean(axis=0)
        _, s, vt = np.linalg.svd(pts - mid, full_matrices=False)
        along, normal = vt[0], vt[1]
        travel = float(np.ptp((pts - mid) @ along))
        rms = float(s[1] / np.sqrt(len(pts)))
        if travel < min_travel_px or rms > max_rms_px:
            continue
        rows.append({"track_id": tid, "a": normal[0], "b": normal[1], "c": -float(normal @ mid),
                     "mu": mid[0], "mv": mid[1], "travel_px": travel, "rms_px": rms})
    return pd.DataFrame(rows, columns=["track_id", "a", "b", "c", "mu", "mv", "travel_px", "rms_px"])


def line_angle_errors(vp_h: np.ndarray, lines: pd.DataFrame) -> np.ndarray:
    """Degrees between each track's direction and the direction from its midpoint to the VP."""
    vp_h = np.asarray(vp_h, dtype=np.float64)
    to_vp = np.stack([vp_h[0] - lines["mu"].to_numpy() * vp_h[2], vp_h[1] - lines["mv"].to_numpy() * vp_h[2]], axis=1)
    to_vp /= np.linalg.norm(to_vp, axis=1, keepdims=True) + 1e-300
    line_dir = np.stack([-lines["b"].to_numpy(), lines["a"].to_numpy()], axis=1)
    cos = np.abs(np.sum(to_vp * line_dir, axis=1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def _sphere_frame(lines: pd.DataFrame, pp, f0):
    if pp is None:  # standalone use: centre the lines' own extent
        pp = lines[["mu", "mv"]].to_numpy(dtype=np.float64).mean(axis=0)
    if f0 is None:
        f0 = max(1000.0, 2.0 * float(np.ptp(lines[["mu", "mv"]].to_numpy(dtype=np.float64), axis=0).max()))
    return np.asarray(pp, dtype=np.float64), float(f0)


def line_normals(lines: pd.DataFrame, pp, f0: float) -> np.ndarray:
    """Unit normals (N, 3) of the planes through the camera centre and each image line (Gaussian sphere of to_direction)."""
    K = np.array([[f0, 0.0, pp[0]], [0.0, f0, pp[1]], [0.0, 0.0, 1.0]])
    n = lines[["a", "b", "c"]].to_numpy(dtype=np.float64) @ K
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _sphere_residual_deg(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(np.abs(n @ d), 0.0, 1.0)))


def vp_from_lines(lines: pd.DataFrame, pp=None, f0: float | None = None, scale_deg: float = 0.5,
                  n_trials: int = 300, n_iter: int = 10, seed: int = 0) -> tuple[np.ndarray | None, float]:
    """
    VP1 as the direction closest to all track lines on the Gaussian sphere: minimizes the Cauchy loss
    (scale `scale_deg`) of the angle between the VP direction and each line's plane, weights
    sqrt(travel), started from the best RANSAC line pair and refined by reweighted least squares.
    An angle on the sphere treats a VP far outside the image like any other; the old image-angle
    inlier test let the intersection slide along a dominant lane (session2_left, 17.8 deg off).
    pp / f0 default to the lines' own centre and extent. Returns (homogeneous VP, fraction of lines
    whose image-angle error is <= 2 deg).
    """
    if len(lines) < 2:
        return None, 0.0
    pp, f0 = _sphere_frame(lines, pp, f0)
    n = line_normals(lines, pp, f0)
    w = np.sqrt(lines["travel_px"].to_numpy(dtype=np.float64))

    def cost(d):
        return float(np.sum(w * np.log1p((_sphere_residual_deg(n, d) / scale_deg) ** 2)))

    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(n_trials):
        i, j = rng.choice(len(n), size=2, replace=False)
        d = np.cross(n[i], n[j])
        if np.linalg.norm(d) < 1e-9:
            continue
        d /= np.linalg.norm(d)
        c = cost(d)
        if c < best_cost:
            best, best_cost = d, c
    if best is None:
        return None, 0.0
    d = best
    for _ in range(n_iter):
        ww = w / (1.0 + (_sphere_residual_deg(n, d) / scale_deg) ** 2)
        d = np.linalg.eigh((n * ww[:, None]).T @ n)[1][:, 0]
    vp = from_direction(d, pp, f0)
    return vp, float((line_angle_errors(vp, lines) <= 2.0).mean())


def vp1_conditioning_deg(lines: pd.DataFrame, vp_h: np.ndarray, pp, f0: float, scale_deg: float = 0.5) -> float:
    """
    sqrt(lambda1 / lambda2) (degrees) of the robust-weighted moment matrix of the line normals at the VP:
    the RMS line residual amplified by how little the lines fan out, i.e. roughly how far the VP could
    move along its weakest direction if the residuals were systematic. Large when one lane dominates.
    """
    n = line_normals(lines, pp, f0)
    d = to_direction(vp_h, pp, f0)[0]
    w = np.sqrt(lines["travel_px"].to_numpy(dtype=np.float64)) / (1.0 + (_sphere_residual_deg(n, d) / scale_deg) ** 2)
    ev = np.linalg.eigvalsh((n * w[:, None]).T @ n)
    return float(np.degrees(np.sqrt(max(ev[0], 0.0) / max(ev[1], 1e-300))))


def vp1_split_deg(lines: pd.DataFrame, pp, f0: float, seed: int = 0) -> float:
    """Angle between the VP1s of alternate halves of the tracks: how stable the intersection is."""
    a, _ = vp_from_lines(lines.iloc[::2], pp, f0, seed=seed)
    b, _ = vp_from_lines(lines.iloc[1::2], pp, f0, seed=seed)
    if a is None or b is None:
        return float("inf")
    return _angle_deg(*to_direction(np.stack([a, b]), pp, f0))


def speed_drift_pct(dets: pd.DataFrame, track_ids, P: np.ndarray, img_size: tuple[int, int],
                    min_points: int = 10, border_margin: float = 3.0) -> tuple[float, int]:
    """
    Median over tracks of log(speed over the far half / speed over the near half) in %, positions
    from the box bottom-centers on the road plane of P (scale-free). Traffic keeps its speed over a
    few tens of metres, so this is ~0 under the right perspective; a wrong VP1 / focal length
    stretches the far part of the road against the near part. Returns (drift %, tracks used).
    """
    from speed_lstm.video import _drop_truncated

    Hinv = np.linalg.inv(P[:, [0, 1, 3]])
    d = _drop_truncated(dets[dets["track_id"].isin(set(track_ids))], img_size[0], img_size[1], border_margin)
    logs = []
    for _, g in d.groupby("track_id"):
        if len(g) < min_points:
            continue
        q = np.stack([(g["xmin"] + g["xmax"]).to_numpy() / 2.0, g["ymax"].to_numpy(), np.ones(len(g))], axis=1) @ Hinv.T
        x, f = q[:, 0] / q[:, 2], g["frame"].to_numpy(dtype=np.float64)
        order = np.argsort(x)
        x, f = x[order], f[order]
        h = len(x) // 2
        near, far = np.polyfit(f[:h], x[:h], 1)[0], np.polyfit(f[h:], x[h:], 1)[0]
        if near * far > 0:
            logs.append(np.log(far / near))
    return (100.0 * float(np.median(logs)) if logs else float("nan")), len(logs)


# ------------------------------------------------------------------- scale

def camera_height_unit(vp1, vp2, pp) -> float:
    """Camera height above the road in the calibration's own units (scale = 1)."""
    road_plane, _ = brno.compute_camera_calibration(vp1, vp2, pp)
    return abs(float(np.dot(road_plane[:3], np.append(np.asarray(pp, dtype=np.float64)[:2], 0.0)) + road_plane[3]))


def sample_car_boxes(dets: pd.DataFrame, img_size: tuple[int, int], per_track: int = 5, max_boxes: int = 400,
                     min_travel_px: float = 40.0, border_margin: float = 3.0, seed: int = 0) -> pd.DataFrame:
    """
    Unclipped car boxes from moving tracks: up to `per_track` boxes spread evenly over each track
    whose bottom-center travelled at least `min_travel_px` (a parked car would otherwise supply most
    of the sample), then at most `max_boxes` (random subset).
    """
    from speed_lstm.video import _drop_truncated

    cars = _drop_truncated(dets[dets["cls"] == "car"], img_size[0], img_size[1], border_margin)
    cars = cars.sort_values(["track_id", "frame"])
    u = (cars["xmin"] + cars["xmax"]) / 2.0
    g = pd.DataFrame({"track_id": cars["track_id"], "u": u, "v": cars["ymax"]}).groupby("track_id")
    travel = np.hypot(g["u"].max() - g["u"].min(), g["v"].max() - g["v"].min())
    cars = cars[cars["track_id"].isin(travel.index[travel >= min_travel_px])]
    picks = []
    for _, t in cars.groupby("track_id", sort=False):
        idx = np.unique(np.linspace(0, len(t) - 1, min(per_track, len(t))).round().astype(int))
        picks.append(t.iloc[idx])
    cars = pd.concat(picks) if picks else cars.iloc[:0]
    if len(cars) > max_boxes:
        cars = cars.sample(max_boxes, random_state=seed).sort_values(["track_id", "frame"])
    return cars


def size_log_ratio(boxes: np.ndarray, P: np.ndarray, prior_sigma_scale: float = 3.0) -> tuple[float, float]:
    """
    Median over boxes of mean(log(fitted dims / car prior)) under P, and the median fit residual (px).
    Zero when cars lifted with P come out exactly prior-sized.
    """
    prior = np.asarray(DIM_PRIORS["car"][0], dtype=np.float64)
    logs, resid = [], []
    for b in boxes:
        f = fit_cuboid(b, P, "car", prior_sigma_scale=prior_sigma_scale)
        if f is None:
            continue
        logs.append(float(np.mean(np.log(f.dims / prior))))
        resid.append(f.residual_px)
    if not logs:
        return float("nan"), float("nan")
    return float(np.median(logs)), float(np.median(resid))


def prior_cuboid_boxes(boxes: np.ndarray, P: np.ndarray, dims=None, n_iter: int = 10) -> np.ndarray:
    """
    (N, 4) xyxy boxes of a road-aligned cuboid of `dims` (default the car prior) placed on the road so
    that its projected box has each detection's bottom edge and horizontal centre (2 equations, 2
    unknowns: Newton on the ground position, all boxes at once). NaN rows where that fails.
    """
    from speed_lstm.data import _CORNER_SIGNS

    boxes = np.asarray(boxes, dtype=np.float64)
    dims = np.asarray(DIM_PRIORS["car"][0] if dims is None else dims, dtype=np.float64)
    goal = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
    offs = _CORNER_SIGNS * dims / 2.0 + np.array([0.0, 0.0, dims[2] / 2.0])

    def project(xy):
        pts = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)[:, None, :] + offs[None]
        h = pts @ P[:, :3].T + P[:, 3]
        front = (h[..., 2] > 1e-6).all(axis=1)
        uv = h[..., :2] / np.where(h[..., 2:] > 1e-6, h[..., 2:], np.nan)
        return np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1), front

    def resid(xy):
        b, front = project(xy)
        return np.stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]], axis=1) - goal, front

    g = np.linalg.solve(P[:, [0, 1, 3]], np.stack([goal[:, 0], goal[:, 1], np.ones(len(goal))]))
    xy = (g[:2] / g[2]).T  # bottom-centre on the road: a start just in front of the cuboid's centre
    eps = 1e-3
    for _ in range(n_iter):
        r, _ = resid(xy)
        J = np.stack([(resid(xy + [eps, 0.0])[0] - r) / eps, (resid(xy + [0.0, eps])[0] - r) / eps], axis=2)
        ok = np.isfinite(J).all(axis=(1, 2)) & np.isfinite(r).all(axis=1) & (np.abs(np.linalg.det(J)) > 1e-12)
        step = np.zeros_like(xy)
        step[ok] = -np.linalg.solve(J[ok], r[ok][:, :, None])[:, :, 0]
        xy = xy + np.clip(step, -5.0, 5.0)
    r, front = resid(xy)
    out, _ = project(xy)
    out[~(front & (np.abs(r) < 0.5).all(axis=1))] = np.nan
    return out


def box_height_log_ratio(boxes: np.ndarray, P: np.ndarray) -> float:
    """Median over boxes of log(detected box height / height of the prior-cuboid box at the same spot)."""
    prior = prior_cuboid_boxes(boxes, P)
    lr = np.log((boxes[:, 3] - boxes[:, 1]) / (prior[:, 3] - prior[:, 1]))
    lr = lr[np.isfinite(lr)]
    return float(np.median(lr)) if len(lr) else float("nan")


def box_part(boxes: np.ndarray, part: str = "far", far_fraction: float = FAR_FRACTION) -> np.ndarray:
    """Boolean mask: the `far_fraction` of boxes with the smallest height ("far"), the rest ("near"), or "all"."""
    h = boxes[:, 3] - boxes[:, 1]
    if part == "all":
        return np.ones(len(boxes), dtype=bool)
    far = h <= np.quantile(h, far_fraction)
    return far if part == "far" else ~far


def fit_box_height_log_ratio(recordings, img_size: tuple[int, int], part: str = "far") -> float:
    """
    BOX_HEIGHT_LOG_RATIO from calibrated training recordings: mean over `recordings` ((dets, calib) pairs,
    calib with vp1, vp2, pp, scale, e.g. a Brno system_*.json) of box_height_log_ratio of the `part`
    boxes of sample_car_boxes at that calibration. dets must come from the same detector and tracker
    as at run time (video.detect_and_track stores ByteTrack state boxes, not raw detections).
    """
    vals = []
    for dets, cal in recordings:
        boxes = sample_car_boxes(dets, img_size)[BOX2D_COLS].to_numpy(dtype=np.float64)
        P = brno.projection_from_calibration(cal["vp1"], cal["vp2"], np.asarray(cal["pp"], dtype=np.float64), cal["scale"])
        vals.append(box_height_log_ratio(boxes[box_part(boxes, part)], P))
    return float(np.mean(vals))


def estimate_scale(boxes: np.ndarray, vp1, vp2, pp, part: str = "far", target: float = BOX_HEIGHT_LOG_RATIO,
                   init_height_m: float = 8.0, tol: float = 1e-3, max_iter: int = 12) -> tuple[float, int]:
    """
    Scale (metres per calibration unit) at which the detected car boxes of `part` (far half by default,
    see box_part) are, in the median, exp(target) times as high as the box of a prior-sized car cuboid at
    the same spot (box_height_log_ratio == target): a secant search on log(scale), started from the scale
    that puts the camera `init_height_m` above the road.
    Returns (scale, number of boxes used).
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    boxes = boxes[box_part(boxes, part)]
    x0 = np.log(init_height_m / camera_height_unit(vp1, vp2, pp))

    def g(x):
        return box_height_log_ratio(boxes, brno.projection_from_calibration(vp1, vp2, pp, float(np.exp(x)))) - target

    x1 = x0 + 0.2
    g0, g1 = g(x0), g(x1)
    for _ in range(max_iter):
        if not (np.isfinite(g0) and np.isfinite(g1)) or abs(g1 - g0) < 1e-12:
            break
        x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
        x2 = float(np.clip(x2, x1 - 1.0, x1 + 1.0))  # at most e^1 per step
        x0, g0 = x1, g1
        x1, g1 = x2, g(x2)
        if abs(g1) < tol:
            break
    return float(np.exp(x1)), int(len(boxes))


# ------------------------------------------------------------- calibration

@dataclass
class AutoCalibration:
    vp1: np.ndarray
    vp2: np.ndarray
    pp: np.ndarray
    focal: float
    scale: float
    P: np.ndarray
    quality: dict = field(default_factory=dict)

    @property
    def reliable(self) -> bool:
        q = self.quality
        vp1_ok = q.get("vp1_source") == "cnn" or q.get("vp1_cond_deg", np.inf) <= MAX_VP1_COND_DEG
        return bool(q.get("vp1_lines", 0) >= MIN_VP1_LINES and q.get("vp1_inlier_frac", 0.0) >= MIN_VP1_INLIER_FRAC
                    and vp1_ok and abs(q.get("speed_drift_pct", np.inf)) <= MAX_SPEED_DRIFT_PCT
                    and q.get("vp2_samples", 0) >= MIN_VP2_SAMPLES
                    and q.get("scale_split_diff", np.inf) <= MAX_SCALE_SPLIT_DIFF
                    and q.get("median_residual_px", np.inf) <= MAX_MEDIAN_RESIDUAL_PX
                    and q.get("scale_near_far_diff", np.inf) <= MAX_SCALE_NEAR_FAR_DIFF
                    and q.get("scale_tracks", 0) >= MIN_SCALE_TRACKS
                    and q.get("scale_top_track_share", 1.0) <= MAX_SCALE_TRACK_SHARE)

    def to_json(self) -> dict:
        return {"camera_calibration": {"vp1": self.vp1.tolist(), "vp2": self.vp2.tolist(), "pp": self.pp.tolist(),
                                       "scale": self.scale},
                "P": self.P.tolist(), "units": "m", "focal": self.focal, "reliable": self.reliable,
                "quality": self.quality, "source": "speed_lstm.autocalib"}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))


def _drift_for(vp1_h, vp2_h, pp, dets, track_ids, img_size) -> float:
    vp1, vp2 = to_point(vp1_h), to_point(vp2_h)
    if len(track_ids) == 0 or -float(np.dot(vp1 - pp, vp2 - pp)) <= 0:
        return float("nan")
    return speed_drift_pct(dets, track_ids, brno.projection_from_calibration(vp1, vp2, pp, 1.0), img_size)[0]


def choose_vp1(choices: dict, spreads: dict, drifts: dict, pp, f0: float) -> tuple[np.ndarray, str, dict]:
    """
    VP1 from the track lines and / or the CNN. When both exist and agree within VP1_AGREE_DEG they are
    averaged on the sphere, weighted by 1 / (split-half disagreement^2 + VP1_SPREAD_FLOOR_DEG^2); when they disagree,
    the one under which cars keep their speed (smaller |speed drift|) wins: a check neither VP was
    fitted to, unlike the median line error, which the track VP minimizes by construction.
    """
    info = {}
    if len(choices) == 1:
        (src, vp), = choices.items()
        return vp, src, info
    dt, dc = (to_direction(choices[k], pp, f0)[0] for k in ("tracks", "cnn"))
    info["vp1_tracks_cnn_deg"] = _angle_deg(dt, dc)
    if info["vp1_tracks_cnn_deg"] <= VP1_AGREE_DEG:
        wt, wc = (1.0 / (min(spreads.get(k, 10.0), 10.0) ** 2 + VP1_SPREAD_FLOOR_DEG ** 2) for k in ("tracks", "cnn"))
        d = wt * dt + wc * np.sign(dt @ dc) * dc
        return from_direction(d / np.linalg.norm(d), pp, f0), "combined", info
    if not all(np.isfinite(drifts.get(k, np.nan)) for k in choices):
        return choices["cnn"], "cnn", info
    src = min(choices, key=lambda k: abs(drifts[k]))
    return choices[src], src, info


def calibrate(dets: pd.DataFrame, img_size: tuple[int, int], vp1_candidates: np.ndarray | None = None,
              vp2_candidates: np.ndarray | None = None, pp=None, max_boxes: int = 400,
              seed: int = 0, height_log_ratio: float = BOX_HEIGHT_LOG_RATIO) -> AutoCalibration:
    """
    dets: detect_and_track output (frame, track_id, cls, xmin, ymin, xmax, ymax).
    vp1_candidates / vp2_candidates: (N, 3) homogeneous per-vehicle VPs from the CNN
    (speed_lstm.vp_cnn); VP2 is required, VP1 candidates are optional. VP1 is taken from the
    track lines, the CNN, or both (choose_vp1).
    pp: principal point, default the image center.
    max_boxes: car boxes sampled for the scale search (estimate_scale uses the far half).
    height_log_ratio: estimate_scale's target (0 for boxes that are exact cuboid projections, e.g. synthetic).
    """
    w, h = img_size
    pp = np.array([w / 2.0, h / 2.0]) if pp is None else np.asarray(pp, dtype=np.float64)
    f0 = float(np.hypot(w, h))
    quality: dict = {}

    if vp2_candidates is None or not len(vp2_candidates):
        raise ValueError("VP2 needs CNN candidates (speed_lstm.vp_cnn); track lines only give VP1")
    d2s = to_direction(vp2_candidates, pp, f0)
    d2, spread = aggregate_directions(d2s)
    quality["vp2_samples"] = int(len(d2s))
    quality["vp2_spread_deg"] = spread
    vp2_h = from_direction(d2, pp, f0)

    lines = track_lines(dets, img_size, height_frac=VP1_HEIGHT_FRAC)
    quality["vp1_lines"] = int(len(lines))
    choices, spreads = {}, {}
    vp1_lines, _ = vp_from_lines(lines, pp, f0, seed=seed)
    if vp1_lines is not None:
        choices["tracks"] = vp1_lines
        spreads["tracks"] = quality["vp1_split_deg"] = vp1_split_deg(lines, pp, f0, seed=seed)
        quality["vp1_cond_deg"] = vp1_conditioning_deg(lines, vp1_lines, pp, f0)
    if vp1_candidates is not None and len(vp1_candidates):
        dc = to_direction(vp1_candidates, pp, f0)
        d, _ = aggregate_directions(dc)
        choices["cnn"] = from_direction(d, pp, f0)
        if len(dc) >= 4:
            spreads["cnn"] = _angle_deg(aggregate_directions(dc[::2])[0], aggregate_directions(dc[1::2])[0])
    if not choices:
        raise ValueError("No VP1: too few straight tracks and no CNN VP1 candidates")
    drifts = {k: _drift_for(v, vp2_h, pp, dets, lines["track_id"], img_size) for k, v in choices.items()}
    vp1_h, src, info = choose_vp1(choices, spreads, drifts, pp, f0)
    quality.update(info)
    quality["vp1_source"] = src
    if len(lines):
        errs = line_angle_errors(vp1_h, lines)
        quality["vp1_median_line_err_deg"] = float(np.median(errs))
        quality["vp1_inlier_frac"] = float((errs <= 2.0).mean())
    else:
        quality["vp1_inlier_frac"] = 0.0
    quality["speed_drift_pct"] = drifts[src] if src in drifts else _drift_for(vp1_h, vp2_h, pp, dets,
                                                                               lines["track_id"], img_size)

    d1 = to_direction(vp1_h, pp, f0)[0]
    vp1, vp2 = to_point(vp1_h), to_point(vp2_h)
    f2 = -float(np.dot(vp1 - pp, vp2 - pp))
    if f2 <= 0:
        raise ValueError(f"VP1 {vp1} and VP2 {vp2} give an imaginary focal length (not orthogonal directions)")
    focal = float(np.sqrt(f2))
    quality["vp1_vp2_angle_deg"] = _angle_deg(d1, d2)

    boxes_df = sample_car_boxes(dets, img_size, max_boxes=max_boxes, seed=seed)
    boxes = boxes_df[BOX2D_COLS].to_numpy(dtype=np.float64)
    if len(boxes) < 4:
        raise ValueError(f"Only {len(boxes)} unclipped car boxes: not enough to estimate the scale")
    scale, n_used = estimate_scale(boxes, vp1, vp2, pp, target=height_log_ratio)
    tracks = boxes_df["track_id"].to_numpy()
    uniq, counts = np.unique(tracks, return_counts=True)
    quality["scale_tracks"] = int(len(uniq))
    quality["scale_top_track_share"] = float(counts.max() / len(tracks))
    half = np.isin(tracks, uniq[::2])
    if half.sum() >= 4 and (~half).sum() >= 4:
        sa, _ = estimate_scale(boxes[half], vp1, vp2, pp, target=height_log_ratio)
        sb, _ = estimate_scale(boxes[~half], vp1, vp2, pp, target=height_log_ratio)
        quality["scale_split_diff"] = float(abs(sa - sb) / scale)
    # a distorted road geometry (wrong VP1 or focal) makes near and far cars disagree about the scale
    s_near, _ = estimate_scale(boxes, vp1, vp2, pp, part="near", target=height_log_ratio)
    quality["scale_near_far_diff"] = float(abs(s_near - scale) / scale)
    _, resid = size_log_ratio(boxes, brno.projection_from_calibration(vp1, vp2, pp, scale))
    quality.update({"scale_boxes": n_used, "median_residual_px": resid,
                    "camera_height_m": float(scale * camera_height_unit(vp1, vp2, pp)), "focal_px": focal})

    P = brno.projection_from_calibration(vp1, vp2, pp, scale)
    return AutoCalibration(vp1=vp1, vp2=vp2, pp=pp, focal=focal, scale=scale, P=P, quality=quality)
