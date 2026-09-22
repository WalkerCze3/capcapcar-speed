#!/usr/bin/env python3
"""
JSON-window inference CLI.

Input JSON: a list of windows, each:
    {
      "timestamps": [t0, t1, ..., t15],                 // seconds, increasing
      "boxes2d": [[xmin,ymin,xmax,ymax], ...] (x16),     // required for 2d/combined checkpoints
      "boxes3d": [[cx,cy,cz,length,width,height], ...] (x16)  // meters, required for 3d/combined
    }

Each window must supply exactly the checkpoint's n_observations timestamps
(and matching boxes) for a SINGLE, already-tracked vehicle — this CLI has no
way to verify vehicle identity or frame continuity on its own.

Usage:
    python -m speed_lstm.predict --checkpoint runs/3d/best.pt --input windows.json --out predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speed_lstm.model import Predictor


def predict_file(checkpoint_path: str, input_path: str) -> list[dict]:
    predictor = Predictor(checkpoint_path)
    with open(input_path) as f:
        windows = json.load(f)

    results = []
    for i, w in enumerate(windows):
        try:
            speed_mps = predictor.predict(
                timestamps=w["timestamps"],
                boxes2d=w.get("boxes2d"),
                boxes3d=w.get("boxes3d"),
            )
            results.append({"index": i, "speed_mps": speed_mps, "speed_kmh": speed_mps * 3.6})
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="JSON file: list of windows (see module docstring).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    results = predict_file(args.checkpoint, args.input)
    Path(args.out).write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if "error" not in r)
    print(f"[predict] {n_ok}/{len(results)} windows predicted -> {args.out}")


if __name__ == "__main__":
    main()
