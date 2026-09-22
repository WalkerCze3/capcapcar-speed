"""
Train SpeedLSTM on windows built by speed_lstm.data, split by
speed_lstm.splitting (speed-balanced, whole-vehicle-group). Every window is
a fixed 15 feature time steps (from its 16 raw observations), so there's no
padding/masking to manage.

Feature and target normalization use only the TRAINING split's statistics
(mean/std over windows and time steps, float64 accumulation, std floor
1e-6), applied identically to train/val/test. The checkpoint with the
lowest validation RMSE (in physical m/s, not standardized units) is kept;
there's no LR scheduler or early stopping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from speed_lstm import features_v2 as fv2
from speed_lstm.data import MAX_TIMESTAMP_GAP, WINDOW_LEN, Window, load_all_windows
from speed_lstm.model import SpeedLSTM
from speed_lstm.splitting import speed_balanced_split

DEFAULT_SCENES = ["scene1", "scene2", "scene3"]


def compute_window_features(window: Window, mode: str) -> np.ndarray:
    box2d = window.box2d if mode in ("2d", "combined") else None
    return fv2.compute_features(mode, box2d, window.metric_center, window.metric_dims, window.timestamps)


class SpeedWindowDataset(Dataset):
    def __init__(self, windows: list[Window], mode: str,
                 feature_mean: np.ndarray, feature_std: np.ndarray,
                 target_mean: float, target_std: float):
        self.windows = windows
        self.mode = mode
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        feats = compute_window_features(w, self.mode)
        feats = (feats - self.feature_mean) / self.feature_std
        target_std = (w.target_speed - self.target_mean) / self.target_std
        return {
            "features": torch.from_numpy(feats.astype(np.float32)),
            "target": torch.tensor(target_std, dtype=torch.float32),
            "target_raw": torch.tensor(w.target_speed, dtype=torch.float32),
        }


def fit_normalization(windows: list[Window], mode: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Feature mean/std over all training windows & time steps, and target mean/std, float64."""
    all_feats = [compute_window_features(w, mode).astype(np.float64) for w in windows]
    stacked = np.concatenate(all_feats, axis=0)  # (n_windows * 15, F)
    feature_mean = stacked.mean(axis=0)
    feature_std = stacked.std(axis=0)
    feature_std = np.maximum(feature_std, 1e-6)

    targets = np.array([w.target_speed for w in windows], dtype=np.float64)
    target_mean = float(targets.mean())
    target_std = max(float(targets.std()), 1e-6)

    return feature_mean, feature_std, target_mean, target_std


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def run_epoch(model, loader, device, optimizer=None) -> dict:
    """Returns total loss plus standardized predictions / raw targets for the caller to denormalize."""
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.MSELoss()

    total_loss, n = 0.0, 0
    all_pred_std, all_target_raw = [], []

    for batch in loader:
        features = batch["features"].to(device)
        target_std = batch["target"].to(device)

        with torch.set_grad_enabled(is_train):
            pred_std = model(features)
            loss = criterion(pred_std, target_std)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = features.shape[0]
        total_loss += loss.item() * bs
        n += bs

        all_pred_std.append(pred_std.detach())
        all_target_raw.append(batch["target_raw"].to(device))

    return {
        "loss": total_loss / n,
        "_pred_std": torch.cat(all_pred_std),
        "_target_raw": torch.cat(all_target_raw),
    }


def _denorm(pred_std: torch.Tensor, target_mean: float, target_std: float) -> torch.Tensor:
    return pred_std * target_std + target_mean


def train(data_dir: str, mode: str, out_dir: str, scenes: list[str] | None = None,
          epochs: int = 20, batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 0.01,
          seed: int = 42) -> dict:
    scenes = scenes or DEFAULT_SCENES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    windows = load_all_windows(data_dir, scenes, require_2d=True)
    print(f"[train] loaded {len(windows)} windows from scenes {scenes}")

    split = speed_balanced_split(windows, seed=seed)
    train_windows, val_windows, test_windows = split["train"], split["val"], split["test"]
    print(f"[train] windows: {len(train_windows)} train / {len(val_windows)} val / {len(test_windows)} test")

    feature_mean, feature_std, target_mean, target_std = fit_normalization(train_windows, mode)

    def make_loader(ws, shuffle):
        ds = SpeedWindowDataset(ws, mode, feature_mean, feature_std, target_mean, target_std)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_windows, shuffle=True)
    val_loader = make_loader(val_windows, shuffle=False)
    test_loader = make_loader(test_windows, shuffle=False)

    input_size = fv2.n_features(mode)
    model = SpeedLSTM(input_size=input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_val_rmse = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)

        train_pred_phys = _denorm(train_metrics["_pred_std"], target_mean, target_std).clamp(min=0)
        val_pred_phys = _denorm(val_metrics["_pred_std"], target_mean, target_std).clamp(min=0)

        train_rmse = rmse(train_pred_phys, train_metrics["_target_raw"])
        train_mae = mae(train_pred_phys, train_metrics["_target_raw"])
        val_rmse = rmse(val_pred_phys, val_metrics["_target_raw"])
        val_mae = mae(val_pred_phys, val_metrics["_target_raw"])

        history.append({
            "epoch": epoch,
            "train": {"loss": train_metrics["loss"], "mae": train_mae, "rmse": train_rmse},
            "val": {"loss": val_metrics["loss"], "mae": val_mae, "rmse": val_rmse},
        })
        print(f"[train] epoch {epoch:3d} | train MAE {train_mae:.3f} RMSE {train_rmse:.3f} "
              f"| val MAE {val_mae:.3f} RMSE {val_rmse:.3f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                "model_state": model.state_dict(),
                "mode": mode,
                "input_size": input_size,
                "hidden_size": model.hidden_size,
                "feature_version": fv2.__name__,
                "n_observations": WINDOW_LEN,
                "max_timestamp_gap": MAX_TIMESTAMP_GAP,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "output_units": "m/s",
                "seed": seed,
                "split_name": "speed_balanced",
            }, out_path / "best.pt")

    checkpoint = torch.load(out_path / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, test_loader, device, optimizer=None)
    test_pred_phys = _denorm(test_metrics["_pred_std"], target_mean, target_std).clamp(min=0)
    test_rmse = rmse(test_pred_phys, test_metrics["_target_raw"])
    test_mae = mae(test_pred_phys, test_metrics["_target_raw"])
    print(f"[train] FINAL test MAE {test_mae:.3f} RMSE {test_rmse:.3f}")

    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    manifest = {
        "mode": mode,
        "scenes": scenes,
        "seed": seed,
        "train_groups": sorted({f"{w.scene}:{w.vehicle_id}" for w in train_windows}),
        "val_groups": sorted({f"{w.scene}:{w.vehicle_id}" for w in val_windows}),
        "test_groups": sorted({f"{w.scene}:{w.vehicle_id}" for w in test_windows}),
        "test_window_keys": sorted(
            f"{w.scene}:{w.camera}:{w.vehicle_id}:{w.frames[0]}" for w in test_windows
        ),
    }
    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "checkpoint": str(out_path / "best.pt"),
        "best_val_rmse": best_val_rmse,
        "test": {"mae": test_mae, "rmse": test_rmse},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--mode", choices=["2d", "3d", "combined"], required=True)
    p.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    result = train(args.data_dir, args.mode, args.out, scenes=args.scenes,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    print(f"\n[train] best checkpoint: {result['checkpoint']}")
    print(f"[train] best val RMSE: {result['best_val_rmse']:.3f}")
    print(f"[train] test MAE: {result['test']['mae']:.3f}  RMSE: {result['test']['rmse']:.3f}")


if __name__ == "__main__":
    main()
