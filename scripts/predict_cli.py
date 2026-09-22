#!/usr/bin/env python3
"""
Predict speed for tracks in a new (unlabeled) trajectory CSV.

Usage:
    python scripts/predict_cli.py --csv data/new_tracks.csv \\
        --checkpoint runs/default/best_model.pt --out predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speedmodel.predict import predict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Trajectory CSV with no speed column needed.")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt from training.")
    p.add_argument("--out", default="predictions.csv")
    args = p.parse_args()

    df = predict(args.csv, args.checkpoint, out_csv=args.out)
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
