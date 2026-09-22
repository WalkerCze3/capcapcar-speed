"""
Run a trained checkpoint on a trajectory CSV that has no speed labels,
producing one predicted speed per track_id.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from speedmodel.dataset import load_and_rename, build_track_groups, TrajectoryDataset
from speedmodel.model import SpeedRegressor


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[SpeedRegressor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]

    model = SpeedRegressor(
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        bidirectional=model_cfg["bidirectional"],
        dropout=model_cfg["dropout"],
        rnn_type=model_cfg["rnn_type"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg


def predict(csv_path: str, checkpoint_path: str, out_csv: str | None = None) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(checkpoint_path, device)

    col_map = cfg["columns"]
    feat_cfg = cfg["features"]

    df = load_and_rename(csv_path, col_map, feat_cfg["mode"], need_speed=False)
    track_groups = build_track_groups(df)
    track_ids = list(track_groups.keys())

    ds = TrajectoryDataset(track_groups, track_ids, feat_cfg["mode"], feat_cfg["max_seq_len"], labels=None)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    results = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            pred = model(features, mask).cpu()
            for tid, speed in zip(batch["track_id"], pred.tolist()):
                results.append({"track_id": tid, "predicted_speed": speed})

    out_df = pd.DataFrame(results)
    if out_csv:
        out_df.to_csv(out_csv, index=False)
        print(f"[predict] wrote {len(out_df)} predictions -> {out_csv}")
    return out_df
