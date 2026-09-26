"""
Lift a 2D detector box to a road-aligned 3D cuboid using a known camera
projection matrix (monocular, geometry-only — no learned 3D head).

The cuboid parameterization matches what the v2 model was trained on (see
speed_lstm.data): center [x, y, z] in metres in the road frame (x along the
road, y across, z up), flat road at z = 0 so center_z = height / 2, dims
[length, width, height] in metres, axes aligned with the road.

Fitting: the 8 cuboid corners are projected through P and their min/max
box is matched to the detected xyxy box with Levenberg-Marquardt. A single
2D box has only 4 constraints, so dimensions are regularized toward a
per-class prior (in log space). For a whole track, `fit_track` first fits
position + dims per frame, then fixes dims to the track median (a vehicle
doesn't change size) and refits position only — 2 unknowns against 4
constraints, which is well-posed and much steadier frame to frame.

Calibration units: I-24 hg.json projection matrices take world coordinates
in FEET. `scale_projection` folds the unit conversion into P so everything
here works in metres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from speed_lstm.data import FEET_TO_METERS, cuboid_corners, project_points, project_to_bbox

# (length, width, height) in metres, and log-space sigma for each.
DIM_PRIORS = {
    "car": ((4.6, 1.85, 1.55), (0.15, 0.08, 0.12)),
    "motorcycle": ((2.2, 0.8, 1.4), (0.2, 0.2, 0.2)),
    "bus": ((12.0, 2.55, 3.2), (0.2, 0.05, 0.1)),
    "truck": ((8.0, 2.5, 3.3), (0.5, 0.08, 0.2)),
}
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Direction convention from speed_lstm.data / hg.json: +1 = EB (x increasing), -1 = WB.
DIRECTION_KEYS = {1: "EB", -1: "WB"}

_HORIZON_PENALTY = 1e3
# 12 cuboid edges as index pairs into data._CORNER_SIGNS order (sx, sy, sz nested loops).
CUBOID_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def scale_projection(P: np.ndarray, units: str) -> np.ndarray:
    """Return P' such that P' @ [X_m, Y_m, Z_m, 1] == P @ [X_u, Y_u, Z_u, 1] for P expecting `units`."""
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (3, 4):
        raise ValueError(f"Projection matrix must be (3, 4), got {P.shape}")
    if units == "m":
        return P.copy()
    if units == "ft":
        s = 1.0 / FEET_TO_METERS
        return P @ np.diag([s, s, s, 1.0])
    raise ValueError(f"Unknown calibration units: {units!r} (expected 'ft' or 'm')")


def load_projections(path: str | Path, camera: str | None = None, units: str = "ft",
                     image_scale: float = 1.0) -> dict[int, np.ndarray]:
    """
    Metre-space projection matrix per direction {+1: P, -1: P}.

    Accepts either an I-24 hg.json ({"EB": {camera: {"P": ...}}, "WB": {...}},
    needs `camera`) or a single-camera file {"P": [[...], [...], [...]]}, in
    which case the same P is used for both directions.

    image_scale: video resolution / calibration resolution (e.g. 0.5 for a
    1080p video with a 4K calibration).
    """
    with open(path) as f:
        calib = json.load(f)
    S = np.diag([image_scale, image_scale, 1.0])

    if "P" in calib:
        P = S @ scale_projection(calib["P"], units)
        return {1: P, -1: P}

    if camera is None:
        cams = sorted({c for key in DIRECTION_KEYS.values() for c in calib.get(key, {})})
        raise ValueError(f"{path} is a per-camera hg.json; pass a camera name (available: {cams})")
    projections = {}
    for d, key in DIRECTION_KEYS.items():
        cam_hg = calib.get(key, {}).get(camera)
        if cam_hg is not None:
            projections[d] = S @ scale_projection(cam_hg["P"], units)
    if not projections:
        raise ValueError(f"Camera {camera!r} not found under EB or WB in {path}")
    return projections


def ground_point(u: float, v: float, P: np.ndarray) -> np.ndarray | None:
    """Back-project image point (u, v) onto the road plane z = 0 -> (x, y) metres, or None above the horizon."""
    H = P[:, [0, 1, 3]]  # z = 0 plane homography: [x, y, 1] -> image
    g = np.linalg.solve(H, np.array([u, v, 1.0]))
    if abs(g[2]) < 1e-12:
        return None
    xy = g[:2] / g[2]
    # Reject solutions behind the camera (projecting back gives negative depth).
    if (P @ np.array([xy[0], xy[1], 0.0, 1.0]))[2] <= 1e-6:
        return None
    return xy


def _levenberg_marquardt(fun, x0: np.ndarray, max_iter: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Minimize ||fun(x)||^2. Forward-difference Jacobian; small problems only (<= 5 params)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    r = fun(x)
    cost = r @ r
    lam = 1e-3
    for _ in range(max_iter):
        J = np.empty((r.size, x.size))
        for i in range(x.size):
            step = 1e-6 * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += step
            J[:, i] = (fun(xp) - r) / step
        A = J.T @ J
        g = J.T @ r
        improved = False
        while lam < 1e10:
            dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_new = fun(x + dx)
            cost_new = r_new @ r_new
            if cost_new < cost:
                x, r, cost = x + dx, r_new, cost_new
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 4.0
        if not improved or np.linalg.norm(dx) < 1e-9 * (1.0 + np.linalg.norm(x)):
            break
    return x, r


@dataclass
class CuboidFit:
    center: np.ndarray      # (3,) metres
    dims: np.ndarray        # (3,) metres [length, width, height]
    box2d: np.ndarray       # (4,) reprojected xyxy of the fitted cuboid
    residual_px: float      # RMS pixel error between reprojected and detected box


def _bbox_residual(center: np.ndarray, dims: np.ndarray, bbox: np.ndarray, P: np.ndarray, px_sigma: float) -> np.ndarray:
    proj = project_to_bbox(center, dims, P)
    if proj is None:
        return np.full(4, _HORIZON_PENALTY)
    return (proj.astype(np.float64) - bbox) / px_sigma


def fit_cuboid(bbox, P: np.ndarray, cls: str = "car", fixed_dims=None, init_xy=None) -> CuboidFit | None:
    """
    Fit a road-aligned cuboid whose projected box matches `bbox` (xyxy pixels).

    fixed_dims: if given, only the ground position (x, y) is solved.
    init_xy: warm start (e.g. previous frame's solution); defaults to the
             back-projected bottom-center of the box.
    Returns None if the box can't be placed on the road plane (above horizon).
    """
    bbox = np.asarray(bbox, dtype=np.float64)
    prior_dims, prior_sigma = (np.asarray(a, dtype=np.float64) for a in DIM_PRIORS.get(cls, DIM_PRIORS["car"]))
    # Pixel noise scales with apparent size so near and far vehicles are weighted alike.
    px_sigma = max(1.0, 0.02 * float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])))

    if init_xy is None:
        init_xy = ground_point((bbox[0] + bbox[2]) / 2.0, bbox[3], P)
        if init_xy is None:
            return None
    init_xy = np.asarray(init_xy, dtype=np.float64)

    if fixed_dims is not None:
        dims = np.asarray(fixed_dims, dtype=np.float64)

        def residual(p):
            return _bbox_residual(np.array([p[0], p[1], dims[2] / 2.0]), dims, bbox, P, px_sigma)

        sol, _ = _levenberg_marquardt(residual, init_xy)
        center = np.array([sol[0], sol[1], dims[2] / 2.0])
    else:
        log_prior = np.log(prior_dims)

        def residual(p):
            d = np.exp(p[2:])
            return np.concatenate([
                _bbox_residual(np.array([p[0], p[1], d[2] / 2.0]), d, bbox, P, px_sigma),
                (p[2:] - log_prior) / prior_sigma,
            ])

        sol, _ = _levenberg_marquardt(residual, np.concatenate([init_xy, log_prior]))
        dims = np.exp(sol[2:])
        center = np.array([sol[0], sol[1], dims[2] / 2.0])

    proj = project_to_bbox(center, dims, P)
    if proj is None:
        return None
    rms = float(np.sqrt(np.mean((proj.astype(np.float64) - bbox) ** 2)))
    return CuboidFit(center=center, dims=dims, box2d=proj, residual_px=rms)


def fit_track(bboxes: np.ndarray, P: np.ndarray, cls: str = "car") -> list[CuboidFit | None]:
    """
    Two-pass fit for one tracked vehicle (bboxes: (N, 4), time-ordered).
    Pass 1 solves position + dims per frame; pass 2 fixes dims to the
    track's median and re-solves position only. Warm-starts each frame
    from the previous one.
    """
    first: list[CuboidFit | None] = []
    prev_xy = None
    for bbox in bboxes:
        fit = fit_cuboid(bbox, P, cls, init_xy=prev_xy)
        first.append(fit)
        prev_xy = fit.center[:2] if fit is not None else None

    ok = [f for f in first if f is not None]
    if not ok:
        return first
    track_dims = np.median(np.stack([f.dims for f in ok]), axis=0)

    second: list[CuboidFit | None] = []
    for bbox, f1 in zip(bboxes, first):
        init = f1.center[:2] if f1 is not None else None
        second.append(fit_cuboid(bbox, P, cls, fixed_dims=track_dims, init_xy=init))
    return second


def cuboid_image_corners(center: np.ndarray, dims: np.ndarray, P: np.ndarray) -> np.ndarray | None:
    """(8, 2) projected cuboid corners, for drawing (pair with CUBOID_EDGES)."""
    return project_points(cuboid_corners(np.asarray(center), np.asarray(dims)), P)
