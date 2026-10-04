#!/usr/bin/env python3
"""
One fine-tuning experiment on prepared BrnoCompSpeed runs (see brno_eval_cli.py).

Config (json):
    {
      "name": "ft_all_lr3e-4",
      "base_checkpoint": "runs/v2/3d/best.pt",       # relative to --project, or absolute
      "train": ["session1_center", ...],              # recordings with labeled_windows.json
      "val": ["session3_right"],                      # optional; default: a group split of train
      "val_frac": 0.15,
      "test": ["session4_center", ...],               # recordings with windows.json + car_eval.csv
      "finetune": {"epochs": 30, "lr": 3e-4, "freeze": "none", "renorm": "keep", "loss": "mse",
                   "scratch": false, "seed": 0}       # null: evaluate base_checkpoint as is
    }

Writes results.json, test_cars.csv and (if fine-tuned) best.pt to --out-dir,
and appends a row to <project>/runs/experiments/leaderboard.csv.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm import finetune as ft  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="Path to a json config, or the json itself")
    p.add_argument("--project", required=True, help="Final-Project-CHULA root (holds runs/)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    # Inline json, or a path to a json file (a long json string would overflow a path check).
    cfg = json.loads(args.config) if args.config.lstrip().startswith("{") else json.loads(Path(args.config).read_text())
    project = Path(args.project)
    runs_root = project / cfg.get("runs_root", "runs/brno")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    base = Path(cfg.get("base_checkpoint", "runs/v2/3d/best.pt"))
    base = base if base.is_absolute() else project / base

    def available(recs, needed):
        ok = [r for r in recs if (runs_root / r / needed).exists()]
        missing = sorted(set(recs) - set(ok))
        if missing:
            print(f"[exp] skipping {missing}: no {needed} yet")
        return ok

    ckpt = base
    if cfg.get("ensemble_of"):  # average the best.pt of earlier experiments (no training here)
        ckpt = [str(project / "runs/experiments" / e / "best.pt") for e in cfg["ensemble_of"]]
    result = {"name": cfg.get("name"), "config": cfg, "time": datetime.datetime.now().isoformat(timespec="seconds")}
    if cfg.get("finetune") is not None and not cfg.get("ensemble_of"):
        train_recs = available(cfg.get("train", []), "labeled_windows.json")
        train = ft.load_json_windows([runs_root / r / "labeled_windows.json" for r in train_recs])
        if cfg.get("val"):
            val_recs = available(cfg["val"], "labeled_windows.json")
            val = ft.load_json_windows([runs_root / r / "labeled_windows.json" for r in val_recs])
        else:
            train, val = ft.group_split(train, cfg.get("val_frac", 0.15), cfg["finetune"].get("seed", 0))
        if cfg["finetune"].get("target") == "residual":  # corrections to each car's geometric speed
            train, val = ft.attach_base_speed(train, runs_root), ft.attach_base_speed(val, runs_root)
        if not train or not val:
            raise SystemExit(f"[exp] not enough labeled windows: {len(train)} train / {len(val)} val")
        res = ft.finetune(base, train, val, out_dir, **cfg["finetune"])
        ckpt = res["checkpoint"]
        result.update(train_recordings=train_recs, n_train_windows=len(train), n_val_windows=len(val),
                      val_best=res["best"])

    test_recs = available(cfg.get("test", []), "windows.json")
    rows, per_rec = [], {}
    for r in test_recs:
        cars = ft.evaluate_recording(ckpt, runs_root / r).assign(recording=r)
        rows.append(cars)
        s = json.loads((runs_root / r / "summary.json").read_text())
        per_rec[r] = {"model": ft.error_summary(cars),
                      "geometry_median": ft.error_summary(cars, "median_kmh"),
                      "reference_median_mae": s.get("reference_system", {}).get("median", {}).get("mean")}
        print(f"[test] {r}: model MAE {per_rec[r]['model'].get('mae', float('nan')):.2f} km/h "
              f"(bias {per_rec[r]['model'].get('bias', float('nan')):+.2f}), "
              f"geometry {per_rec[r]['geometry_median'].get('mae', float('nan')):.2f}, "
              f"reference {per_rec[r]['reference_median_mae'] or float('nan'):.2f}  [{len(cars)} cars]")
    all_cars = pd.concat(rows) if rows else pd.DataFrame()
    if len(all_cars):  # model and geometry err for different reasons; their mean is a cheap combination to track
        all_cars["hybrid_kmh"] = all_cars[["model_kmh", "median_kmh"]].mean(axis=1, skipna=False)
    all_cars.to_csv(out_dir / "test_cars.csv", index=False)
    result.update(test_recordings=test_recs, per_recording=per_rec,
                  test_model=ft.error_summary(all_cars) if len(all_cars) else {"n": 0},
                  test_geometry_median=ft.error_summary(all_cars, "median_kmh") if len(all_cars) else {"n": 0},
                  test_hybrid=ft.error_summary(all_cars, "hybrid_kmh") if len(all_cars) else {"n": 0},
                  checkpoint=str(ckpt))
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=float))

    tm = result["test_model"]
    print(f"[exp] {cfg.get('name')}: test model MAE {tm.get('mae', float('nan')):.2f} km/h "
          f"(median {tm.get('median', float('nan')):.2f}, bias {tm.get('bias', float('nan')):+.2f}, "
          f"{tm.get('n', 0)} cars over {len(test_recs)} recordings)")

    board = project / "runs" / "experiments" / "leaderboard.csv"
    board.parent.mkdir(parents=True, exist_ok=True)
    fcfg = cfg.get("finetune") or {}
    row = pd.DataFrame([{
        "time": result["time"], "id": out_dir.name, "name": cfg.get("name"),
        "test_mae": tm.get("mae"), "test_median": tm.get("median"), "test_bias": tm.get("bias"),
        "test_cars": tm.get("n"), "test_recordings": len(test_recs),
        "geometry_mae": result["test_geometry_median"].get("mae"), "hybrid_mae": result["test_hybrid"].get("mae"),
        "val_mae": result.get("val_best", {}).get("mae"), "best_epoch": result.get("val_best", {}).get("epoch"),
        "train_recordings": len(result.get("train_recordings", [])), "train_windows": result.get("n_train_windows"),
        **{f"ft_{k}": fcfg.get(k) for k in ("epochs", "lr", "weight_decay", "freeze", "renorm", "loss", "scratch", "target", "seed")},
    }])
    if board.exists():  # merge so rows written before a new column was added stay aligned
        row = pd.concat([pd.read_csv(board), row], ignore_index=True)
    row.to_csv(board, index=False)
    cols = [c for c in ("id", "test_mae", "test_median", "test_bias", "hybrid_mae", "geometry_mae", "test_cars",
                        "test_recordings", "train_recordings", "train_windows", "val_mae", "best_epoch") if c in row]
    view = row[cols].round(2)
    (board.parent / "leaderboard_summary.txt").write_text(
        "TOP 12 by test_mae\n" + view.sort_values("test_mae").head(12).to_string(index=False)
        + "\n\nLATEST 12\n" + view.tail(12).to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
