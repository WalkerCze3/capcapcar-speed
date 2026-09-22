"""
v2 feature engineering.

A window holds 16 raw observations, but every feature here is defined at a
*transition* between two consecutive raw observations, so a 16-observation
window yields 15 feature time steps (matching the model's [batch, 15, F]
input) — the very first raw observation only ever serves as history (for
the one-step rate/velocity of the second observation, and as the anchor for
relative position), it never appears as an output row by itself.

Output row j (0-indexed, j = 0..14) corresponds to raw observation (j+1):
  - static/raw geometry at row j uses raw observation j+1
  - one-step rate / log-size-rate at row j uses raw observations j and j+1
    (always defined for every row: every row has a "previous" raw obs)
  - relative position at row j is raw position (j+1) minus raw position 0
    (the window's first observation)
  - lag-k velocity at row j uses raw observations (j+1) and (j+1-k); zero
    with mask 0 until j+1-k >= 0, i.e. until j >= k-1
  - acceleration at row j is the change in lag-1 velocity between rows j
    and j-1; zero with mask 0 only at j=0 (both lag-1 velocities it needs
    are otherwise always defined for j>=1)
  - elapsed time at row j is (t[j+1] - t[j])

All of the above only reads raw observations <= j+1, so nothing at output
row j depends on any observation after it (causal).

Feature column order (matches the counts in ARCHITECTURE.md section 3.3):
  raw geometry, one-step rates, log-size rates, relative positions,
  lag velocities (grouped by position type, each lag in order),
  lag validity masks (grouped the same way), acceleration,
  acceleration validity masks, elapsed time.
"""

from __future__ import annotations

import numpy as np

LAGS = (1, 3, 5, 10)

N_FEATURES_2D = 53
N_FEATURES_3D = 41
N_FEATURES_COMBINED = 93


def _rate(raw: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """raw: (16, D) -> (15, D). Always defined (every output row has a previous raw obs)."""
    return (raw[1:] - raw[:-1]) / dt[:, None]


def _log_rate(raw: np.ndarray, dt: np.ndarray) -> np.ndarray:
    log_raw = np.log(raw)
    return (log_raw[1:] - log_raw[:-1]) / dt[:, None]


def _relative(raw: np.ndarray) -> np.ndarray:
    """raw: (16, D) -> (15, D), relative to raw[0] (the window's first observation)."""
    return raw[1:] - raw[0:1]


def _lag_velocity(pos: np.ndarray, t: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """pos: (16, D), t: (16,) -> vel (15, D), mask (15,)."""
    n_out = pos.shape[0] - 1
    vel = np.zeros((n_out, pos.shape[1]), dtype=np.float64)
    mask = np.zeros(n_out, dtype=np.float32)
    for j in range(n_out):
        r = j + 1
        if r - lag >= 0:
            vel[j] = (pos[r] - pos[r - lag]) / (t[r] - t[r - lag])
            mask[j] = 1.0
    return vel, mask


def _acceleration(vel1: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """vel1: (15, D) lag-1 velocity. dt: (15,) per-row elapsed time (dt[j] = t[j+1]-t[j])."""
    n_out, d = vel1.shape
    accel = np.zeros((n_out, d), dtype=np.float64)
    mask = np.zeros(n_out, dtype=np.float32)
    for j in range(1, n_out):
        mid_dt = (dt[j] + dt[j - 1]) / 2.0
        accel[j] = (vel1[j] - vel1[j - 1]) / mid_dt
        mask[j] = 1.0
    return accel, mask


def _position_group(pos: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    pos: (16, D) raw positions for one position type.
    Returns (velocities (15, D*len(LAGS)), masks (15, len(LAGS)), accel (15, D), accel_mask (15,)).
    """
    dt = t[1:] - t[:-1]
    vel_blocks, mask_blocks = [], []
    vel1 = None
    for lag in LAGS:
        vel, mask = _lag_velocity(pos, t, lag)
        if lag == 1:
            vel1 = vel
        vel_blocks.append(vel)
        mask_blocks.append(mask[:, None])
    velocities = np.concatenate(vel_blocks, axis=1)
    masks = np.concatenate(mask_blocks, axis=1)
    accel, accel_mask = _acceleration(vel1, dt)
    return velocities, masks, accel, accel_mask


def compute_features_2d(box2d: np.ndarray, t: np.ndarray) -> np.ndarray:
    """box2d: (16, 4) [xmin, ymin, xmax, ymax]. t: (16,) seconds. Returns (15, 53) float32."""
    dt = t[1:] - t[:-1]

    xmin, ymin, xmax, ymax = box2d[:, 0], box2d[:, 1], box2d[:, 2], box2d[:, 3]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    aspect = width / height

    # center(2), bottom_center(2, bcx==cx by construction), width, height, area, aspect => 8
    raw = np.stack([cx, cy, cx, ymax, width, height, area, aspect], axis=1)

    raw_out = raw[1:]                                    # (15, 8)
    rate_out = _rate(raw, dt)                             # (15, 8)
    log_rate_out = _log_rate(raw[:, [4, 5]], dt)           # (15, 2) width, height

    center_pos, bcenter_pos = raw[:, 0:2], raw[:, 2:4]
    rel_center = _relative(center_pos)                     # (15, 2)
    rel_bcenter = _relative(bcenter_pos)                    # (15, 2)

    vel_center, mask_center, accel_center, accel_mask_center = _position_group(center_pos, t)
    vel_bcenter, mask_bcenter, accel_bcenter, accel_mask_bcenter = _position_group(bcenter_pos, t)

    elapsed = dt[:, None]                                   # (15, 1)

    feats = np.concatenate([
        raw_out, rate_out, log_rate_out,
        rel_center, rel_bcenter,
        vel_center, vel_bcenter,
        mask_center, mask_bcenter,
        accel_center, accel_bcenter,
        accel_mask_center[:, None], accel_mask_bcenter[:, None],
        elapsed,
    ], axis=1)
    assert feats.shape[1] == N_FEATURES_2D, feats.shape
    return feats.astype(np.float32)


def compute_features_3d(center3d: np.ndarray, dims3d: np.ndarray, t: np.ndarray) -> np.ndarray:
    """center3d: (16, 3) [x,y,z] meters. dims3d: (16, 3) [length,width,height] meters. Returns (15, 41) float32."""
    dt = t[1:] - t[:-1]

    volume = dims3d[:, 0] * dims3d[:, 1] * dims3d[:, 2]
    raw = np.concatenate([center3d, dims3d, volume[:, None]], axis=1)  # (16, 7)

    raw_out = raw[1:]
    rate_out = _rate(raw, dt)
    log_rate_out = _log_rate(dims3d, dt)                    # (15, 3) length, width, height

    rel_center = _relative(center3d)                         # (15, 3)

    vel_center, mask_center, accel_center, accel_mask_center = _position_group(center3d, t)

    elapsed = dt[:, None]

    feats = np.concatenate([
        raw_out, rate_out, log_rate_out,
        rel_center,
        vel_center,
        mask_center,
        accel_center,
        accel_mask_center[:, None],
        elapsed,
    ], axis=1)
    assert feats.shape[1] == N_FEATURES_3D, feats.shape
    return feats.astype(np.float32)


def compute_features_combined(box2d: np.ndarray, center3d: np.ndarray, dims3d: np.ndarray, t: np.ndarray) -> np.ndarray:
    """2D features concatenated with 3D features, dropping 3D's elapsed-time column (already in 2D's)."""
    f2d = compute_features_2d(box2d, t)
    f3d = compute_features_3d(center3d, dims3d, t)
    feats = np.concatenate([f2d, f3d[:, :-1]], axis=1)
    assert feats.shape[1] == N_FEATURES_COMBINED, feats.shape
    return feats


def compute_features(mode: str, box2d: np.ndarray | None, center3d: np.ndarray, dims3d: np.ndarray,
                      t: np.ndarray) -> np.ndarray:
    if mode == "2d":
        return compute_features_2d(box2d, t)
    if mode == "3d":
        return compute_features_3d(center3d, dims3d, t)
    if mode == "combined":
        return compute_features_combined(box2d, center3d, dims3d, t)
    raise ValueError(f"Unknown mode: {mode!r}")


def n_features(mode: str) -> int:
    return {"2d": N_FEATURES_2D, "3d": N_FEATURES_3D, "combined": N_FEATURES_COMBINED}[mode]
