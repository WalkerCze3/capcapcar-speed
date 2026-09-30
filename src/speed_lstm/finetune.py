"""
Fine-tune a v2 checkpoint on video-lifted windows labeled with measured speeds
(e.g. BrnoCompSpeed, via speed_lstm.brno.label_windows), and score checkpoints
on cached test windows without re-running detection or lifting.

A labeled window is a predict.py-style dict (timestamps, boxes2d, boxes3d)
plus target_speed (m/s) and group (one id per real vehicle). Validation and
model selection use the per-vehicle error in km/h — the median prediction of
a vehicle's windows vs. its measured speed — which is what the Brno
evaluation scores, rather than the per-window error.

Options:
  freeze:  none | lstm (train attention + head only) | head_only (train the
           last Linear only — an affine recalibration of the output)
  renorm:  keep (the checkpoint's feature/target normalization)
           | features (refit feature mean/std on the new training windows)
           | all (refit features and target)
  loss:    mse | huber (on standardized targets)
  scratch: ignore the checkpoint's weights (same architecture), for comparison
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from speed_lstm import features_v2 as fv2
from speed_lstm.model import SpeedLSTM


def window_features(w: dict, mode: str) -> np.ndarray:
    b3 = np.asarray(w["boxes3d"], dtype=np.float64)
    b2 = np.asarray(w["boxes2d"], dtype=np.float64) if mode in ("2d", "combined") else None
    return fv2.compute_features(mode, b2, b3[:, :3], b3[:, 3:6], np.asarray(w["timestamps"], dtype=np.float64))


def load_json_windows(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            out.extend(json.load(f))
    return out


def load_checkpoint(path, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def model_from_checkpoint(ckpt: dict, device) -> SpeedLSTM:
    model = SpeedLSTM(input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def predict_batch(model: nn.Module, ckpt: dict, feats: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    """feats: (N, 15, F) raw features -> (N,) m/s, using ckpt's normalization."""
    model.eval()
    x = (feats - ckpt["feature_mean"]) / ckpt["feature_std"]
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size].astype(np.float32)).to(device)
            out.append(model(xb).cpu().numpy())
    pred = np.concatenate(out) * ckpt["target_std"] + ckpt["target_mean"] if out else np.zeros(0)
    return np.maximum(pred, 0.0)


def group_errors_kmh(pred_mps: np.ndarray, target_mps: np.ndarray, groups: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"group": groups, "pred": pred_mps * 3.6, "target": target_mps * 3.6})
    g = df.groupby("group").agg(pred=("pred", "median"), target=("target", "first"))
    g["err"] = g["pred"] - g["target"]
    return g


def _stats(err: np.ndarray) -> dict:
    e = np.asarray(err, dtype=np.float64)
    e = e[np.isfinite(e)]
    if not len(e):
        return {"n": 0}
    a = np.abs(e)
    return {"n": int(len(e)), "mae": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "bias": float(e.mean())}


def group_split(windows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups = sorted({w["group"] for w in windows})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_groups = set(groups[:max(1, int(round(len(groups) * val_frac)))])
    return [w for w in windows if w["group"] not in val_groups], [w for w in windows if w["group"] in val_groups]


def finetune(init_checkpoint: str | Path, train_windows: list[dict], val_windows: list[dict], out_dir: str | Path,
             epochs: int = 30, lr: float = 3e-4, weight_decay: float = 0.01, batch_size: int = 256,
             freeze: str = "none", renorm: str = "keep", loss: str = "mse", scratch: bool = False,
             seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = copy.deepcopy(load_checkpoint(init_checkpoint))
    mode = ckpt["mode"]

    Xtr = np.stack([window_features(w, mode) for w in train_windows]).astype(np.float64)
    ytr = np.array([w["target_speed"] for w in train_windows], dtype=np.float64)
    Xva = np.stack([window_features(w, mode) for w in val_windows]).astype(np.float64)
    yva = np.array([w["target_speed"] for w in val_windows], dtype=np.float64)
    gva = [w["group"] for w in val_windows]

    if renorm in ("features", "all"):
        flat = Xtr.reshape(-1, Xtr.shape[-1])
        ckpt["feature_mean"] = flat.mean(axis=0)
        ckpt["feature_std"] = np.maximum(flat.std(axis=0), 1e-6)
    if renorm == "all":
        ckpt["target_mean"] = float(ytr.mean())
        ckpt["target_std"] = max(float(ytr.std()), 1e-6)

    model = SpeedLSTM(input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"]).to(device)
    if not scratch:
        model.load_state_dict(ckpt["model_state"])
    if freeze == "lstm":
        for p in model.lstm.parameters():
            p.requires_grad = False
    elif freeze == "head_only":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("head.2.")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    crit = nn.HuberLoss(delta=1.0) if loss == "huber" else nn.MSELoss()

    xtr_t = torch.from_numpy(((Xtr - ckpt["feature_mean"]) / ckpt["feature_std"]).astype(np.float32))
    ytr_t = torch.from_numpy(((ytr - ckpt["target_mean"]) / ckpt["target_std"]).astype(np.float32))

    def val_score():
        g = group_errors_kmh(predict_batch(model, ckpt, Xva, device), yva, gva)
        return _stats(g["err"].to_numpy())

    best = {"epoch": 0, **val_score()}
    log(f"[finetune] {len(train_windows)} train windows ({len({w['group'] for w in train_windows})} cars), "
        f"{len(val_windows)} val windows ({len(set(gva))} cars); mode {mode}, freeze {freeze}, renorm {renorm}, "
        f"loss {loss}, lr {lr}, scratch {scratch}")
    log(f"[finetune] epoch   0 | val car MAE {best.get('mae', float('nan')):.2f} km/h, bias {best.get('bias', float('nan')):+.2f}")
    best_state = copy.deepcopy(model.state_dict())
    history = [best]
    n = len(xtr_t)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xtr_t[idx].to(device), ytr_t[idx].to(device)
            l = crit(model(xb), yb)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            total += l.item() * len(idx)
        s = {"epoch": epoch, "train_loss": total / n, **val_score()}
        history.append(s)
        log(f"[finetune] epoch {epoch:3d} | train loss {s['train_loss']:.4f} | val car MAE {s['mae']:.2f} km/h, "
            f"bias {s['bias']:+.2f}")
        if s["mae"] < best["mae"]:
            best = s
            best_state = copy.deepcopy(model.state_dict())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt["model_state"] = best_state
    ckpt["finetuned_from"] = str(init_checkpoint)
    ckpt["finetune"] = {"epochs": epochs, "lr": lr, "freeze": freeze, "renorm": renorm, "loss": loss,
                        "scratch": scratch, "seed": seed, "best_epoch": best["epoch"]}
    torch.save(ckpt, out_dir / "best.pt")
    (out_dir / "finetune_history.json").write_text(json.dumps(history, indent=2))
    log(f"[finetune] best epoch {best['epoch']}: val car MAE {best['mae']:.2f} km/h -> {out_dir / 'best.pt'}")
    return {"checkpoint": str(out_dir / "best.pt"), "best": best, "history": history}


def evaluate_recording(checkpoint: str | Path, run_dir: str | Path) -> pd.DataFrame:
    """
    Re-predict a prepared Brno run's cached windows (windows.json) with `checkpoint`
    and return its car_eval rows (valid, matched cars) with a fresh model_kmh.
    """
    from speed_lstm.brno import model_speed_for_matches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint, device)
    model = model_from_checkpoint(ckpt, device)
    run_dir = Path(run_dir)
    windows = json.loads((run_dir / "windows.json").read_text())
    matches = pd.read_csv(run_dir / "car_eval.csv")
    if windows:
        feats = np.stack([window_features(w, ckpt["mode"]) for w in windows]).astype(np.float64)
        speed = predict_batch(model, ckpt, feats, device)
    else:
        speed = np.zeros(0)
    preds = pd.DataFrame({"track_id": [w["track_id"] for w in windows],
                          "t_start": [w["timestamps"][0] for w in windows],
                          "t_end": [w["timestamps"][-1] for w in windows],
                          "speed_kmh": speed * 3.6})
    matches["model_kmh"] = model_speed_for_matches(matches, preds)
    return matches[matches["valid"] & matches["matched"]].copy()


def error_summary(rows: pd.DataFrame, col: str = "model_kmh") -> dict:
    err = (rows[col] - rows["gt_kmh"]).to_numpy()
    s = _stats(err)
    if s["n"]:
        s["mean_rel_pct"] = float(np.nanmean(np.abs(err) / rows["gt_kmh"].to_numpy()) * 100)
    return s
