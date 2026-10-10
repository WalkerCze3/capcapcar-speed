"""
Automatic camera calibration from passing traffic: tracked vehicles -> P, no manual input.

    tracks (ByteTrack boxes) --> VP1 from the lines the box bottom-centers travel along
    vehicle crops --(VP CNN, speed_lstm.vp_cnn)--> per-vehicle VP1 / VP2 candidates --> robust aggregate
    VP1, VP2, principal point (image center) --> focal length + road plane (brno.compute_camera_calibration)
    scale: the one value that makes lifted cars as big as a typical car (lift3d.DIM_PRIORS)
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
MIN_VP2_SAMPLES = 30
MAX_SCALE_SPLIT_DIFF = 0.05   # scale from either half of the cars must agree within 5%
MAX_MEDIAN_RESIDUAL_PX = 4.0


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


def track_lines(dets: pd.DataFrame, min_points: int = 8, min_travel_px: float = 40.0,
                max_rms_px: float = 2.0) -> pd.DataFrame:
    """
    One homogeneous line per track through its box bottom-centers (total least squares), kept when
    the vehicle travelled far enough in a straight enough line. Columns: track_id, a, b, c (line
    a u + b v + c = 0, (a, b) unit), mu, mv (midpoint), travel_px, rms_px.
    """
    rows = []
    for tid, g in dets.groupby("track_id", sort=True):
        if len(g) < min_points:
            continue
        pts = np.stack([(g["xmin"] + g["xmax"]).to_numpy() / 2.0, g["ymax"].to_numpy()], axis=1).astype(np.float64)
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


def vp_from_lines(lines: pd.DataFrame, inlier_deg: float = 2.0, n_trials: int = 500,
                  seed: int = 0) -> tuple[np.ndarray | None, float]:
    """
    VP1 as the common intersection of the track lines: RANSAC over line pairs, then a weighted
    least-squares refit (smallest singular vector) on the inliers. Returns (homogeneous VP, inlier fraction).
    """
    if len(lines) < 2:
        return None, 0.0
    L = lines[["a", "b", "c"]].to_numpy(dtype=np.float64)
    w = np.sqrt(lines["travel_px"].to_numpy(dtype=np.float64))
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    for _ in range(n_trials):
        i, j = rng.choice(len(L), size=2, replace=False)
        vp = np.cross(L[i], L[j])
        if not np.any(vp):
            continue
        inl = line_angle_errors(vp, lines) <= inlier_deg
        score = float(w[inl].sum())
        if score > best_score:
            best, best_score = vp, score
    if best is None:
        return None, 0.0
    for _ in range(3):  # refit, then re-pick inliers around the refit
        inl = line_angle_errors(best, lines) <= inlier_deg
        if inl.sum() < 2:
            break
        best = np.linalg.svd(L[inl] * w[inl, None])[2][-1]
    return best, float((line_angle_errors(best, lines) <= inlier_deg).mean())


# ------------------------------------------------------------------- scale

def camera_height_unit(vp1, vp2, pp) -> float:
    """Camera height above the road in the calibration's own units (scale = 1)."""
    road_plane, _ = brno.compute_camera_calibration(vp1, vp2, pp)
    return abs(float(np.dot(road_plane[:3], np.append(np.asarray(pp, dtype=np.float64)[:2], 0.0)) + road_plane[3]))


def sample_car_boxes(dets: pd.DataFrame, img_size: tuple[int, int], every: int = 5, max_boxes: int = 400,
                     border_margin: float = 3.0, seed: int = 0) -> pd.DataFrame:
    """Unclipped car boxes, every `every`-th frame of each track, at most `max_boxes` (random subset)."""
    from speed_lstm.video import _drop_truncated

    cars = _drop_truncated(dets[dets["cls"] == "car"], img_size[0], img_size[1], border_margin)
    cars = cars.sort_values(["track_id", "frame"])
    cars = cars[cars.groupby("track_id").cumcount() % every == 0]
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


def estimate_scale(boxes: np.ndarray, vp1, vp2, pp, init_height_m: float = 8.0, tol: float = 1e-3,
                   max_iter: int = 12) -> tuple[float, float]:
    """
    Scale (metres per calibration unit) at which the median lifted car matches the car prior:
    a secant search on log(scale) for size_log_ratio == 0. At that point the prior pulls neither way,
    so its strength only changes how fast the search converges, not where it lands.
    Starts from the scale that puts the camera `init_height_m` above the road.
    Returns (scale, median residual px at the final scale).
    """
    x0 = np.log(init_height_m / camera_height_unit(vp1, vp2, pp))

    def g(x):
        return size_log_ratio(boxes, brno.projection_from_calibration(vp1, vp2, pp, float(np.exp(x))))

    x1 = x0 + 0.2
    (g0, _), (g1, r1) = g(x0), g(x1)
    for _ in range(max_iter):
        if not (np.isfinite(g0) and np.isfinite(g1)) or abs(g1 - g0) < 1e-12:
            break
        x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
        x2 = float(np.clip(x2, x1 - 1.0, x1 + 1.0))  # at most e^1 per step
        x0, g0 = x1, g1
        x1, (g1, r1) = x2, g(x2)
        if abs(g1) < tol:
            break
    return float(np.exp(x1)), r1


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
        return bool(q.get("vp1_lines", 0) >= MIN_VP1_LINES and q.get("vp1_inlier_frac", 0.0) >= MIN_VP1_INLIER_FRAC
                    and q.get("vp2_samples", 0) >= MIN_VP2_SAMPLES
                    and q.get("scale_split_diff", np.inf) <= MAX_SCALE_SPLIT_DIFF
                    and q.get("median_residual_px", np.inf) <= MAX_MEDIAN_RESIDUAL_PX)

    def to_json(self) -> dict:
        return {"camera_calibration": {"vp1": self.vp1.tolist(), "vp2": self.vp2.tolist(), "pp": self.pp.tolist(),
                                       "scale": self.scale},
                "P": self.P.tolist(), "units": "m", "focal": self.focal, "reliable": self.reliable,
                "quality": self.quality, "source": "speed_lstm.autocalib"}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))


def calibrate(dets: pd.DataFrame, img_size: tuple[int, int], vp1_candidates: np.ndarray | None = None,
              vp2_candidates: np.ndarray | None = None, pp=None, max_boxes: int = 400,
              seed: int = 0) -> AutoCalibration:
    """
    dets: detect_and_track output (frame, track_id, cls, xmin, ymin, xmax, ymax).
    vp1_candidates / vp2_candidates: (N, 3) homogeneous per-vehicle VPs from the CNN
    (speed_lstm.vp_cnn); VP2 is required, VP1 candidates are optional. VP1 is taken from the
    track lines or from the CNN, whichever agrees better with the track lines.
    pp: principal point, default the image center.
    max_boxes: car boxes used by the scale search (each search step lifts all of them).
    """
    w, h = img_size
    pp = np.array([w / 2.0, h / 2.0]) if pp is None else np.asarray(pp, dtype=np.float64)
    f0 = float(np.hypot(w, h))
    quality: dict = {}

    lines = track_lines(dets)
    quality["vp1_lines"] = int(len(lines))
    vp1_lines, inlier_frac = vp_from_lines(lines, seed=seed)
    choices = {}
    if vp1_lines is not None:
        choices["tracks"] = vp1_lines
    if vp1_candidates is not None and len(vp1_candidates):
        d, _ = aggregate_directions(to_direction(vp1_candidates, pp, f0))
        choices["cnn"] = from_direction(d, pp, f0)
    if not choices:
        raise ValueError("No VP1: too few straight tracks and no CNN VP1 candidates")
    if len(lines):
        errs = {k: float(np.median(line_angle_errors(v, lines))) for k, v in choices.items()}
        src = min(errs, key=errs.get)
        quality["vp1_median_line_err_deg"] = errs[src]
        quality["vp1_inlier_frac"] = float((line_angle_errors(choices[src], lines) <= 2.0).mean())
    else:
        src = "cnn"
        quality["vp1_inlier_frac"] = 0.0
    vp1_h = choices[src]
    quality["vp1_source"] = src

    if vp2_candidates is None or not len(vp2_candidates):
        raise ValueError("VP2 needs CNN candidates (speed_lstm.vp_cnn); track lines only give VP1")
    d1 = to_direction(vp1_h, pp, f0)[0]
    d2s = to_direction(vp2_candidates, pp, f0)
    d2, spread = aggregate_directions(d2s)
    quality["vp2_samples"] = int(len(d2s))
    quality["vp2_spread_deg"] = spread
    vp2_h = from_direction(d2, pp, f0)

    vp1, vp2 = to_point(vp1_h), to_point(vp2_h)
    f2 = -float(np.dot(vp1 - pp, vp2 - pp))
    if f2 <= 0:
        raise ValueError(f"VP1 {vp1} and VP2 {vp2} give an imaginary focal length (not orthogonal directions)")
    focal = float(np.sqrt(f2))
    quality["vp1_vp2_angle_deg"] = float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), 0.0, 1.0))))

    boxes_df = sample_car_boxes(dets, img_size, max_boxes=max_boxes, seed=seed)
    boxes = boxes_df[BOX2D_COLS].to_numpy(dtype=np.float64)
    if len(boxes) < 4:
        raise ValueError(f"Only {len(boxes)} unclipped car boxes: not enough to estimate the scale")
    scale, resid = estimate_scale(boxes, vp1, vp2, pp)
    tracks = boxes_df["track_id"].to_numpy()
    uniq = np.unique(tracks)
    half = np.isin(tracks, uniq[::2])
    if half.sum() >= 2 and (~half).sum() >= 2:
        sa, _ = estimate_scale(boxes[half], vp1, vp2, pp)
        sb, _ = estimate_scale(boxes[~half], vp1, vp2, pp)
        quality["scale_split_diff"] = float(abs(sa - sb) / scale)
    quality.update({"scale_boxes": int(len(boxes)), "median_residual_px": resid,
                    "camera_height_m": float(scale * camera_height_unit(vp1, vp2, pp)), "focal_px": focal})

    P = brno.projection_from_calibration(vp1, vp2, pp, scale)
    return AutoCalibration(vp1=vp1, vp2=vp2, pp=pp, focal=focal, scale=scale, P=P, quality=quality)
