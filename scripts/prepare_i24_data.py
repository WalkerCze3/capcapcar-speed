#!/usr/bin/env python3
"""
Convert I-24 dataset files (obj/sceneX_annotations.csv + ts/sceneX_ts.csv)
into the trajectory CSV schema configs/metric_3d.yaml expects:
    track_id, frame, timestamp, x, y, length, width, direction, speed

Each (object id, camera) pair becomes its own track — the I-24 dataset
observes the same physical object from multiple cameras at once, and this
keeps the conversion simple (no cross-camera fusion) at the cost of some
redundant, highly-correlated tracks in the training set.

`speed` is not provided by I-24 directly, so it's derived per track as the
average speed over the whole trajectory: total displacement (feet) divided
by total elapsed time (seconds, from ts.csv), converted to mph. This is a
DERIVED label, not an independently measured ground truth — read the
"3D / metric mode" section of the README before trusting comparisons that
use it.

Usage:
    python scripts/prepare_i24_data.py --data-dir /path/to/i24_dataset \
        --scenes 1 2 3 --out data/i24_tracks.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FT_S_TO_MPH = 0.6818182

REQUIRED_ANNOTATION_COLS = ["frame", "camera", "id", "x", "y", "l", "w", "direction"]


def load_scene(data_dir: Path, scene: int) -> pd.DataFrame:
    ann_path = data_dir / "obj" / f"scene{scene}_annotations.csv"
    ts_path = data_dir / "ts" / f"scene{scene}_ts.csv"

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {ann_path}")
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timestamps file: {ts_path}")

    ann = pd.read_csv(ann_path)
    missing = [c for c in REQUIRED_ANNOTATION_COLS if c not in ann.columns]
    if missing:
        raise ValueError(f"{ann_path} is missing expected column(s): {missing}")

    ts_wide = pd.read_csv(ts_path)
    camera_cols = [c for c in ts_wide.columns if c != "frame"]
    ts_long = ts_wide.melt(id_vars="frame", value_vars=camera_cols,
                            var_name="camera", value_name="timestamp")

    merged = ann.merge(ts_long, on=["frame", "camera"], how="left")
    n_unmatched = merged["timestamp"].isna().sum()
    if n_unmatched:
        print(f"[scene{scene}] warning: {n_unmatched} row(s) had no matching "
              f"timestamp (frame/camera not found in {ts_path.name}) — dropped")
        merged = merged.dropna(subset=["timestamp"])

    merged["track_id"] = (
        f"scene{scene}_" + merged["id"].astype(str) + "_" + merged["camera"].astype(str)
    )
    merged = merged.rename(columns={"l": "length", "w": "width"})
    return merged[["track_id", "frame", "timestamp", "x", "y", "length", "width", "direction"]]


def compute_speed_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Average speed per track = total displacement / total elapsed time (mph)."""
    rows = []
    for track_id, g in df.groupby("track_id", sort=False):
        g = g.sort_values("timestamp")
        if len(g) < 2:
            continue
        dt = g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]
        if dt <= 0:
            continue
        dx = g["x"].iloc[-1] - g["x"].iloc[0]
        dy = g["y"].iloc[-1] - g["y"].iloc[0]
        speed_ft_s = np.hypot(dx, dy) / dt
        rows.append((track_id, speed_ft_s * FT_S_TO_MPH))
    return pd.DataFrame(rows, columns=["track_id", "speed"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                    help="Root of the I-24 dataset (contains obj/, ts/, video/, ...).")
    p.add_argument("--scenes", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--out", required=True, help="Output trajectory CSV path.")
    p.add_argument("--min-track-len", type=int, default=2,
                    help="Drop tracks with fewer than this many rows.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    scene_dfs = [load_scene(data_dir, s) for s in args.scenes]
    all_rows = pd.concat(scene_dfs, ignore_index=True)

    counts = all_rows.groupby("track_id").size()
    keep_ids = counts[counts >= args.min_track_len].index
    dropped = len(counts) - len(keep_ids)
    if dropped:
        print(f"Dropping {dropped} track(s) with < {args.min_track_len} rows")
    all_rows = all_rows[all_rows["track_id"].isin(keep_ids)]

    labels = compute_speed_labels(all_rows)
    out_df = all_rows.merge(labels, on="track_id", how="inner")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_tracks = out_df["track_id"].nunique()
    print(f"Wrote {len(out_df)} rows, {n_tracks} tracks -> {out_path}")
    print(f"Speed label range: {out_df.groupby('track_id')['speed'].first().min():.1f} - "
          f"{out_df.groupby('track_id')['speed'].first().max():.1f} mph")


if __name__ == "__main__":
    main()
