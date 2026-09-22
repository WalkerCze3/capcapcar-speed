"""
Load a trajectory CSV (VS13/I-24-style or your own extracted tracks),
rename columns per config, group rows into per-track sequences, and expose
them as a PyTorch Dataset of (features, mask, label).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from speedmodel.features import compute_features, pad_or_truncate


def required_columns(feature_mode: str, need_speed: bool) -> list[str]:
    """
    The standard (post-rename) column set this codebase needs, which depends
    on which feature extractor will run — 2D pixel-space modes need bbox
    w/h, metric_3d needs real-world dims + a timestamp for dt.

    `frame` is always required (used as the sort key within a track, via
    build_track_groups) even in metric_3d mode — if your data only has a
    timestamp, map BOTH `frame` and `timestamp` to that same source column
    in configs/*.yaml.
    """
    if feature_mode in ("self_normalized", "raw"):
        cols = ["track_id", "frame", "x", "y", "w", "h"]
    elif feature_mode == "metric_3d":
        cols = ["track_id", "frame", "timestamp", "x", "y", "length", "width", "direction"]
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode!r}")

    if need_speed:
        cols.append("speed")
    return cols


def load_and_rename(csv_path: str | Path, column_map: dict, feature_mode: str, need_speed: bool) -> pd.DataFrame:
    """
    Read the raw CSV and rename its columns to the standard names this
    codebase uses internally. Which columns are required depends on
    feature_mode — see required_columns().
    """
    df = pd.read_csv(csv_path)
    required = required_columns(feature_mode, need_speed)

    rename = {}
    missing = []
    for std_name in required:
        raw_name = column_map.get(std_name, std_name)
        if raw_name not in df.columns:
            missing.append(f"{std_name} (expected column '{raw_name}')")
        else:
            rename[raw_name] = std_name

    if missing:
        raise ValueError(
            "CSV is missing required column(s): " + "; ".join(missing) +
            f"\nAvailable columns: {list(df.columns)}\n"
            "Fix configs/*.yaml under `columns:` to match your actual header."
        )

    df = df.rename(columns=rename)
    return df[required]


def load_labels_from_file(labels_path: str | Path) -> dict:
    """Expects a csv with columns: track_id, speed."""
    labels_df = pd.read_csv(labels_path)
    if not {"track_id", "speed"}.issubset(labels_df.columns):
        raise ValueError(f"Labels file {labels_path} must have columns: track_id, speed")
    return dict(zip(labels_df["track_id"], labels_df["speed"]))


def build_track_groups(df: pd.DataFrame) -> dict[object, pd.DataFrame]:
    """Group rows by track_id, sorted ascending by frame within each group."""
    groups = {}
    for track_id, g in df.groupby("track_id", sort=False):
        groups[track_id] = g.sort_values("frame").reset_index(drop=True)
    return groups


class TrajectoryDataset(Dataset):
    """
    One item = one track's full trajectory, turned into a fixed-length
    (max_seq_len, N_FEATURES) tensor + an attention mask + (optionally) the
    speed label.
    """

    def __init__(self, track_groups: dict, track_ids: list, feature_mode: str,
                 max_seq_len: int, labels: dict | None = None):
        self.track_groups = track_groups
        self.track_ids = list(track_ids)
        self.feature_mode = feature_mode
        self.max_seq_len = max_seq_len
        self.labels = labels  # None at pure-inference time with no ground truth

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int):
        track_id = self.track_ids[idx]
        track_df = self.track_groups[track_id]

        feats = compute_features(track_df, mode=self.feature_mode)
        feats, mask = pad_or_truncate(feats, self.max_seq_len)

        item = {
            "track_id": track_id,
            "features": torch.from_numpy(feats),        # (max_seq_len, N_FEATURES)
            "mask": torch.from_numpy(mask),              # (max_seq_len,)
        }

        speed = self.labels.get(track_id) if self.labels is not None else None
        if speed is None and "speed" in track_df.columns:
            # Fall back to a constant-speed column already present in the track rows
            speed = float(track_df["speed"].iloc[0])
        if speed is not None:
            item["speed"] = torch.tensor(float(speed), dtype=torch.float32)

        return item


def split_track_ids(track_ids: list, val_frac: float, test_frac: float, seed: int
                     ) -> tuple[list, list, list]:
    """
    Split at the TRACK level, never the frame level — splitting frames
    randomly would leak parts of the same vehicle's trajectory across
    train/val/test and inflate reported accuracy.
    """
    rng = np.random.default_rng(seed)
    ids = list(track_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    val_ids = ids[:n_val]
    test_ids = ids[n_val:n_val + n_test]
    train_ids = ids[n_val + n_test:]
    return train_ids, val_ids, test_ids
