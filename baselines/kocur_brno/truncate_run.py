#!/usr/bin/env python3
"""
Cut a prepared Brno run (brno_eval_cli output) down to its first --seconds, as if
brno_eval_cli had been run with --max-seconds equal to that: windows that end
by then, ground-truth cars that finish crossing the lines by seconds - 1 (the
same t_max rule), and those cars' labeled windows. Lets a short Kocur run be
compared with a longer lift3d run without re-fitting lift3d.

Example:
    python baselines/kocur_brno/truncate_run.py --run-dir runs/brno/session4_center \
        --seconds 120 --out-dir runs/brno_lift3d_120s/session4_center
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src, dst, n = Path(args.run_dir), Path(args.out_dir), args.seconds
    dst.mkdir(parents=True, exist_ok=True)
    cars = pd.read_csv(src / "car_eval.csv")
    cars = cars[cars["t_last"] <= n - 1.0]
    cars.to_csv(dst / "car_eval.csv", index=False)

    windows = [w for w in json.loads((src / "windows.json").read_text()) if w["timestamps"][-1] < n]
    (dst / "windows.json").write_text(json.dumps(windows))
    if (src / "labeled_windows.json").exists():
        keep = {f"{src.name}:{int(g)}" for g in cars["gt_id"]}
        labeled = [w for w in json.loads((src / "labeled_windows.json").read_text())
                   if w["timestamps"][-1] < n and w["group"] in keep]
        (dst / "labeled_windows.json").write_text(json.dumps(labeled))
    else:
        labeled = []
    if (src / "boxes3d.csv").exists():
        boxes = pd.read_csv(src / "boxes3d.csv")
        boxes[boxes["timestamp"] < n].to_csv(dst / "boxes3d.csv", index=False)
    summary = json.loads((src / "summary.json").read_text())
    summary.update(seconds=n, truncated_from=str(src))
    (dst / "summary.json").write_text(json.dumps(summary, indent=2))
    shutil.copy(src / "run_config.json", dst / "run_config.json")
    valid = cars[cars["valid"]]
    print(f"[truncate] {src.name}: first {n:g} s -> {len(windows)} windows, {len(labeled)} labeled, "
          f"{int(valid['matched'].sum())}/{len(valid)} valid cars matched -> {dst}")


if __name__ == "__main__":
    main()
