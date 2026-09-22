"""
Turn a single track's raw per-frame box sequence into a per-frame feature
sequence the model consumes.

Three feature modes (set in configs/*.yaml under features.mode):

  self_normalized (2D bbox, recommended default for pixel-space input):
      dx  = (x_t - x_{t-1}) / w_t
      dy  = (y_t - y_{t-1}) / h_t
      dw  = (w_t - w_{t-1}) / w_{t-1}
      dh  = (h_t - h_{t-1}) / h_{t-1}
      ar  = w_t / h_t
    Dividing displacement/size-change by the box's OWN current size cancels
    most of the near/far-from-camera scale difference in pixel-space input.

  raw (2D bbox):
      Plain pixel differences, matching the original paper's feature choice.
      Kept only for an A/B comparison against self_normalized.

  metric_3d (3D box in real-world coordinates, e.g. I24-3D-style annotations
             where x/y are already in feet along/across the road):
      vx  = (x_t - x_{t-1}) / dt      -- already ~ speed along road
      vy  = (y_t - y_{t-1}) / dt      -- lateral velocity component
      len_norm = length_t / median(length over track)   -- flags occlusion/partial box
      wid_norm = width_t / median(width over track)
      direction = heading, as given
    IMPORTANT: because x/y are already metric here, vx/vy computed this way
    are themselves a noisy speed estimate — the model's job in this mode is
    DENOISING that noisy per-frame signal into one stable value (handling
    occlusion, box jitter, missed frames), not "recovering scale" the way it
    is in the 2D pixel-space modes. Don't expect it to beat a plain
    median-filtered Δx/Δt by a huge margin unless occlusion/noise in your
    annotations is substantial — that gap IS the thing worth measuring.

First frame of every track has no t-1 to diff against; its delta/velocity
features are zero for that frame in all three modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_FEATURES = 5  # kept identical across modes so model.py needs no changes


def compute_features(track_df: pd.DataFrame, mode: str = "self_normalized") -> np.ndarray:
    """
    track_df: rows for ONE track, already sorted ascending by frame/time.

      self_normalized / raw  expect columns: x, y, w, h
      metric_3d               expects columns: x, y, length, width, direction, timestamp

    (Standard names — already renamed via the config's column map before
    this is called; see dataset.required_columns().)

    Returns: float32 array of shape (T, N_FEATURES), T = len(track_df).
    """
    if mode in ("self_normalized", "raw"):
        return _compute_features_2d(track_df, mode)
    elif mode == "metric_3d":
        return _compute_features_3d(track_df)
    else:
        raise ValueError(f"Unknown feature mode: {mode!r}")


def _compute_features_2d(track_df: pd.DataFrame, mode: str) -> np.ndarray:
    x = track_df["x"].to_numpy(dtype=np.float64)
    y = track_df["y"].to_numpy(dtype=np.float64)
    w = track_df["w"].to_numpy(dtype=np.float64)
    h = track_df["h"].to_numpy(dtype=np.float64)

    T = len(track_df)
    feats = np.zeros((T, N_FEATURES), dtype=np.float32)
    if T == 0:
        return feats

    dx_raw = np.diff(x, prepend=x[0])
    dy_raw = np.diff(y, prepend=y[0])
    dw_raw = np.diff(w, prepend=w[0])
    dh_raw = np.diff(h, prepend=h[0])
    dx_raw[0] = dy_raw[0] = dw_raw[0] = dh_raw[0] = 0.0  # no t-1 for the first frame

    eps = 1e-6
    if mode == "self_normalized":
        dx = dx_raw / (w + eps)
        dy = dy_raw / (h + eps)
        dw = dw_raw / (np.roll(w, 1) + eps)
        dh = dh_raw / (np.roll(h, 1) + eps)
        dw[0] = dh[0] = 0.0
    else:  # raw
        dx, dy, dw, dh = dx_raw, dy_raw, dw_raw, dh_raw

    aspect = w / (h + eps)

    feats[:, 0] = dx
    feats[:, 1] = dy
    feats[:, 2] = dw
    feats[:, 3] = dh
    feats[:, 4] = aspect
    return feats


def _compute_features_3d(track_df: pd.DataFrame) -> np.ndarray:
    x = track_df["x"].to_numpy(dtype=np.float64)
    y = track_df["y"].to_numpy(dtype=np.float64)
    length = track_df["length"].to_numpy(dtype=np.float64)
    width = track_df["width"].to_numpy(dtype=np.float64)
    direction = track_df["direction"].to_numpy(dtype=np.float64)
    t = track_df["timestamp"].to_numpy(dtype=np.float64)

    T = len(track_df)
    feats = np.zeros((T, N_FEATURES), dtype=np.float32)
    if T == 0:
        return feats

    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0

    dx_raw = np.diff(x, prepend=x[0])
    dy_raw = np.diff(y, prepend=y[0])
    dx_raw[0] = dy_raw[0] = 0.0

    vx = np.zeros(T, dtype=np.float64)
    vy = np.zeros(T, dtype=np.float64)
    valid_dt = dt > 0  # frame 0 has dt==0 and stays zero; guards duplicate timestamps too
    vx[valid_dt] = dx_raw[valid_dt] / dt[valid_dt]
    vy[valid_dt] = dy_raw[valid_dt] / dt[valid_dt]

    eps = 1e-6
    med_len = np.median(length) if T > 0 else eps
    med_wid = np.median(width) if T > 0 else eps
    len_norm = length / (med_len + eps)
    wid_norm = width / (med_wid + eps)

    feats[:, 0] = vx
    feats[:, 1] = vy
    feats[:, 2] = len_norm
    feats[:, 3] = wid_norm
    feats[:, 4] = direction
    return feats


def pad_or_truncate(feats: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a (T, F) feature array to exactly max_len time steps.

    Returns (padded_feats, mask) where padded_feats has shape (max_len, F)
    and mask has shape (max_len,) with 1.0 for real frames and 0.0 for
    padding — the model must not attend to padded positions.

    Truncation keeps the LAST max_len frames (closest to when speed is
    reported), not the first — usually more relevant for the label.
    """
    T, F = feats.shape
    out = np.zeros((max_len, F), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)

    if T == 0:
        return out, mask

    if T >= max_len:
        out[:, :] = feats[-max_len:, :]
        mask[:] = 1.0
    else:
        out[:T, :] = feats
        mask[:T] = 1.0

    return out, mask
