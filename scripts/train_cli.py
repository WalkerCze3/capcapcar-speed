#!/usr/bin/env python3
"""
Train the speed regressor.

Usage:
    python scripts/train_cli.py --csv data/vs13_tracks.csv --config configs/default.yaml
    python scripts/train_cli.py --csv data/my_tracks.csv --config configs/default.yaml --out runs/exp1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speedmodel.train import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Trajectory CSV to train on.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default=None, help="Override train.out_dir from config.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result = train(args.csv, cfg, out_dir=args.out)
    print(f"\n[train_cli] best checkpoint: {result['checkpoint']}")
    print(f"[train_cli] best val MAE: {result['best_val_mae']:.3f}")
    print(f"[train_cli] test MAE: {result['test']['mae']:.3f}  RMSE: {result['test']['rmse']:.3f}")


if __name__ == "__main__":
    main()
