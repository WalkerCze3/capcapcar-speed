#!/usr/bin/env python3
"""
Rebuild prepared Brno recordings with the 2D box's bottom-center as the ground point.

Reads what brno_eval_cli.py cached in <project>/runs/brno/<recording>/ (detections.csv,
boxes3d.csv, calib_P.json, summary.json) — no video or GPU needed — and writes the same
files experiments read (windows.json, labeled_windows.json, car_eval.csv, summary.json)
to <project>/<out-root>/<recording>/. Point an experiment at them with "runs_root".

Why: the cuboid fit's ground center drifts backwards along the vehicle as it approaches
the camera (measured against the dataset's reference tracks), so its speeds come out
0-6% low depending on the camera. The bottom-center of the detector's box is the
vehicle's front (or rear) bottom edge, which stays on the road.

Per detection: bottom-center pixel -> road plane (z = 0) through P, smoothed like the
lifted centers. boxes3d's cx/cy hold that point itself, not the vehicle's center:
shifting it back half a length would push it short of the last measurement line on
tracks that end right after it. Dimensions are the lifted track's median (class prior
for tracks the cuboid fit dropped).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm import brno  # noqa: E402
from speed_lstm.finetune import load_checkpoint  # noqa: E402
from speed_lstm.lift3d import DIM_PRIORS  # noqa: E402
from speed_lstm.video import BOX2D_COLS, BOX3D_COLS, _drop_truncated, _smooth, build_video_windows  # noqa: E402
from brno_eval_cli import mask_filter  # noqa: E402  (scripts/ is on sys.path when run as a script)


def ground_tracks(dets: pd.DataFrame, lifted: pd.DataFrame, P: np.ndarray, step: int, fps: float,
                  smooth_window: int = 5, border_margin: float = 3.0) -> pd.DataFrame:
    """boxes3d.csv-style rows from bottom-center ground points."""
    dets = _drop_truncated(dets, brno.WIDTH, brno.HEIGHT, border_margin)
    H_inv = np.linalg.inv(P[:, [0, 1, 3]])  # image -> road plane z = 0
    lifted_dims = lifted.groupby("track_id")[["length", "width", "height"]].median()
    out = []
    for tid, g in dets.groupby("track_id", sort=True):
        g = g.sort_values("frame").drop_duplicates("frame")
        if len(g) < 2:
            continue
        uv1 = np.column_stack([(g["xmin"] + g["xmax"]) / 2, g["ymax"], np.ones(len(g))])
        xyw = uv1 @ H_inv.T
        ok = xyw[:, 2] > 1e-9
        if ok.sum() < 2:
            continue
        g, xy = g[ok], xyw[ok, :2] / xyw[ok, 2:]
        frames = g["frame"].to_numpy()
        xy = _smooth(xy, frames, smooth_window)
        travel = xy[-1] - xy[0]
        if np.linalg.norm(travel) < 1e-6:
            continue
        cls = g["cls"].mode().iloc[0]
        dims = (lifted_dims.loc[tid].to_numpy() if tid in lifted_dims.index
                else np.asarray(DIM_PRIORS.get(cls, DIM_PRIORS["car"])[0]))
        direction = 1 if travel[0] >= 0 else -1
        for frame, c, box in zip(frames, xy, g[BOX2D_COLS].to_numpy(dtype=np.float64)):
            out.append({"track_id": tid, "frame": int(frame), "timestamp": frame * step / fps, "cls": cls,
                        "direction": direction, **dict(zip(BOX3D_COLS, [c[0], c[1], dims[2] / 2, *dims])),
                        **dict(zip(BOX2D_COLS, box)), "residual_px": 0.0})
    return pd.DataFrame(out, columns=["track_id", "frame", "timestamp", "cls", "direction",
                                      *BOX3D_COLS, *BOX2D_COLS, "residual_px"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--project", required=True, help="Final-Project-CHULA root (holds runs/)")
    p.add_argument("--brno", required=True, help="2016-ITS-BrnoCompSpeed root (dataset/, results/)")
    p.add_argument("--recordings", nargs="*", help="Default: every prepared recording")
    p.add_argument("--out-root", default="runs/brno_bottom")
    p.add_argument("--checkpoint", default="runs/v2/3d/best.pt", help="Only for its window length / gap rules")
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()

    project, brno_root = Path(args.project), Path(args.brno)
    src_root = project / "runs/brno"
    recs = args.recordings or sorted(d.name for d in src_root.iterdir() if (d / "detections.csv").exists())
    ckpt = load_checkpoint(project / args.checkpoint)
    for rec in recs:
        src, dst = src_root / rec, project / args.out_root / rec
        dst.mkdir(parents=True, exist_ok=True)
        session = brno_root / "dataset" / rec
        gt = brno.load_gt(session / "gt_data.pkl")
        calib, _ = brno.load_system(brno_root / "results" / rec / "system_dubska_optimal_calib.json")
        P = np.array(json.loads((src / "calib_P.json").read_text())["P"])
        old = json.loads((src / "summary.json").read_text())
        step, t_max = old["frame_step"], old["seconds"] - 1.0

        dets = pd.read_csv(src / "detections.csv")
        mask = session / "video_mask.png"
        if mask.exists():
            dets = mask_filter(dets, mask)
        tracks = ground_tracks(dets, pd.read_csv(src / "boxes3d.csv"), P, step, float(gt["fps"]))
        tracks.to_csv(dst / "boxes3d.csv", index=False)
        windows = build_video_windows(tracks, ckpt["n_observations"], args.stride, ckpt["max_timestamp_gap"])
        (dst / "windows.json").write_text(json.dumps(windows))

        cars = brno.prefilter(brno.tracks_to_cars(tracks, P, gt["fps"]), gt)
        brno.calculate_speeds(cars, gt, calib)
        matches = pd.DataFrame(brno.compute_matches(gt, cars, t_max))
        matches["model_kmh"] = np.nan  # experiments re-predict
        matches.to_csv(dst / "car_eval.csv", index=False)
        labeled = brno.label_windows(windows, matches, rec)
        (dst / "labeled_windows.json").write_text(json.dumps(labeled))

        valid = matches[matches["valid"]]
        scored = valid[valid["matched"]]
        summary = {**old, "ground_point": "bbox_bottom_center", "matched": int(len(scored)),
                   "recall": float(len(scored) / len(valid)) if len(valid) else float("nan"),
                   "tracks_measured": int(sum("speed" in c for c in cars)), "labeled_windows": len(labeled)}
        for col, name in (("full_kmh", "geometry_full"), ("median_kmh", "geometry_median")):
            err = scored[col] - scored["gt_kmh"]
            summary[name] = {**brno.error_stats(err), "bias": float(err.mean()) if len(err) else float("nan")}
        summary.pop("model", None)
        (dst / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[reground] {rec}: {len(tracks)} boxes, {len(windows)} windows, {summary['matched']}/{len(valid)} cars "
              f"matched, geometry MAE {summary['geometry_median'].get('mean', float('nan')):.2f} km/h "
              f"(cuboid: {old['geometry_median'].get('mean', float('nan')):.2f}), {len(labeled)} labeled windows")


if __name__ == "__main__":
    main()
