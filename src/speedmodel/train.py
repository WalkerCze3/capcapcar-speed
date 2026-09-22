"""
Train SpeedRegressor on a trajectory CSV. Splits by TRACK (never by frame,
see dataset.split_track_ids for why), trains with Huber/SmoothL1 loss
(robust to the occasional mislabeled or noisy track), and saves the
best-on-validation checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from speedmodel.dataset import (
    load_and_rename, load_labels_from_file, build_track_groups,
    TrajectoryDataset, split_track_ids,
)
from speedmodel.model import SpeedRegressor


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, all_preds, all_targets = 0.0, [], []
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        speed = batch["speed"].to(device)

        with torch.set_grad_enabled(is_train):
            pred = model(features, mask)
            loss = criterion(pred, speed)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(speed)
        all_preds.append(pred.detach())
        all_targets.append(speed.detach())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return {
        "loss": total_loss / len(all_preds),
        "mae": mae(all_preds, all_targets),
        "rmse": rmse(all_preds, all_targets),
    }


def train(csv_path: str, cfg: dict, out_dir: str | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    col_map = cfg["columns"]
    label_cfg = cfg["labels"]
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    need_speed_in_csv = label_cfg["source"] == "column"
    df = load_and_rename(csv_path, col_map, feat_cfg["mode"], need_speed=need_speed_in_csv)
    track_groups = build_track_groups(df)

    labels = None
    if label_cfg["source"] == "file":
        labels = load_labels_from_file(label_cfg["path"])

    track_ids = list(track_groups.keys())
    train_ids, val_ids, test_ids = split_track_ids(
        track_ids, train_cfg["val_frac"], train_cfg["test_frac"], train_cfg["seed"]
    )
    print(f"[train] tracks: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    def make_loader(ids, shuffle):
        ds = TrajectoryDataset(track_groups, ids, feat_cfg["mode"], feat_cfg["max_seq_len"], labels)
        return DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=shuffle)

    train_loader = make_loader(train_ids, shuffle=True)
    val_loader = make_loader(val_ids, shuffle=False)
    test_loader = make_loader(test_ids, shuffle=False)

    model = SpeedRegressor(
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        bidirectional=model_cfg["bidirectional"],
        dropout=model_cfg["dropout"],
        rnn_type=model_cfg["rnn_type"],
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    out_dir = Path(out_dir or train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(f"[train] epoch {epoch:3d} | "
              f"train MAE {train_metrics['mae']:.3f} RMSE {train_metrics['rmse']:.3f} | "
              f"val MAE {val_metrics['mae']:.3f} RMSE {val_metrics['rmse']:.3f}")

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "config": cfg}, out_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stop_patience"]:
                print(f"[train] early stopping at epoch {epoch} (no val improvement for "
                      f"{train_cfg['early_stop_patience']} epochs)")
                break

    # Evaluate the BEST checkpoint (not necessarily the last epoch) on the held-out test split.
    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, test_loader, criterion, device, optimizer=None)
    print(f"[train] FINAL test MAE {test_metrics['mae']:.3f} RMSE {test_metrics['rmse']:.3f}")

    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history, "test": test_metrics, "best_val_mae": best_val_mae}, f, indent=2)

    return {"best_val_mae": best_val_mae, "test": test_metrics, "checkpoint": str(out_dir / "best_model.pt")}
