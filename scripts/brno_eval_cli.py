#!/usr/bin/env python3
"""
Run the video -> 3D box -> v2 model pipeline on one BrnoCompSpeed recording
and score it with the dataset's official rules (see speed_lstm/brno.py).

Usage:
    python scripts/brno_eval_cli.py \
        --session-dir /path/2016-ITS-BrnoCompSpeed/dataset/session4_center \
        --calib /path/2016-ITS-BrnoCompSpeed/results/session4_center/system_dubska_optimal_calib.json \
        --checkpoint runs/v2/3d/best.pt --max-seconds 600 --out-dir runs/brno/session4_center

Outputs in --out-dir: detections.csv, boxes3d.csv, window_predictions.csv,
car_eval.csv (one row per ground-truth car), summary.json.
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
from speed_lstm.video import (build_video_windows, detect_and_track, lift_tracks,  # noqa: E402
                              predict_windows, video_info)


def mask_filter(dets: pd.DataFrame, mask_path: Path) -> pd.DataFrame:
    """Keep detections whose bottom-center lies inside the recording's video_mask.png."""
    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or dets.empty:
        return dets
    h, w = mask.shape
    x = ((dets["xmin"] + dets["xmax"]) / 2).clip(0, w - 1).astype(int)
    y = (dets["ymax"] - 1).clip(0, h - 1).astype(int)
    return dets[mask[y.to_numpy(), x.to_numpy()] > 0]


def main() -> None:
    from speed_lstm.model import Predictor

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--session-dir", required=True, help="dataset/sessionN_<left|center|right>")
    p.add_argument("--calib", required=True, help="results/<same recording>/system_*.json (camera_calibration)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weights", default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--max-seconds", type=float, default=600.0, help="Process the first N seconds of the video")
    p.add_argument("--frame-step", type=int, default=2,
                   help="Read every n-th frame (2: 50 -> 25 fps, near the ~30 fps the model was trained on)")
    p.add_argument("--video-fps", type=float, default=None,
                   help="Time base: seconds = decoded frame index / this. Default: the ground truth's fps "
                        "(OpenCV reports 100 fps for 50 fps Brno AVIs)")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--stride", type=int, default=4, help="Processed frames between window starts")
    p.add_argument("--detections", default=None, help="Reuse detections.csv from an earlier run")
    p.add_argument("--no-mask", action="store_true", help="Don't drop detections outside video_mask.png")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    session = Path(args.session_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    gt = brno.load_gt(session / "gt_data.pkl")
    calib, reference_cars = brno.load_system(args.calib)
    P = brno.projection_from_calibration(calib["vp1"], calib["vp2"], calib["pp"], calib["scale"])
    (out_dir / "calib_P.json").write_text(json.dumps({"P": P.tolist(), "source": args.calib}))

    dist = brno.distance_check(gt, calib)
    along = dist[dist["toVP1"]]
    rel = (along["measured_m"] - along["true_m"]).abs() / along["true_m"] * 100
    print(f"[calib] {Path(args.calib).name}: {len(along)} along-road distances, "
          f"mean error {rel.mean():.2f}% (worst {rel.max():.2f}%)")

    video = session / "video.avi"
    fps, n_frames, img_w, img_h = video_info(video)
    print(f"[video] {video}: {n_frames} frames @ {fps:.2f} fps, {img_w}x{img_h}; ground truth fps {gt['fps']}")
    gt_first = min(c["intersections"][0]["videoTime"] for c in gt["cars"])
    gt_last = max(c["intersections"][-1]["videoTime"] for c in gt["cars"])
    # Brno AVIs declare 100 fps and pad with empty packets that OpenCV skips when decoding, so the
    # header's length (frames / fps) is right but decoded frames are 1/50 s apart.
    print(f"[time] header: {n_frames / fps:.0f} s of video; ground-truth cars cross the lines "
          f"from {gt_first:.0f} s to {gt_last:.0f} s")
    fps = args.video_fps or float(gt["fps"])
    print(f"[time] using {fps:g} fps (seconds = decoded frame / {fps:g}), as the official evaluation does")
    if (img_w, img_h) != (brno.WIDTH, brno.HEIGHT):
        print(f"[video] warning: Brno lines/calibration assume {brno.WIDTH}x{brno.HEIGHT}")
    step = max(1, args.frame_step)
    max_samples = int(args.max_seconds * fps / step)

    if args.detections:
        dets = pd.read_csv(args.detections)
        dets = dets[dets["frame"] < max_samples]
    else:
        dets = detect_and_track(video, args.weights, conf=args.conf, device=args.device,
                                max_frames=max_samples, frame_step=step)
    dets.to_csv(out_dir / "detections.csv", index=False)
    if not args.no_mask:
        dets = mask_filter(dets, session / "video_mask.png")
    print(f"[track] {len(dets)} detections in the mask, {dets['track_id'].nunique()} tracks "
          f"(every {step} frame(s), {step / fps * 1000:.0f} ms apart)")

    n_samples = int(dets["frame"].max()) + 1 if len(dets) else 0
    timestamps = np.arange(n_samples) * step / fps
    lifted = lift_tracks(dets, timestamps, {1: P, -1: P}, (img_w, img_h), smooth_window=args.smooth)
    lifted.to_csv(out_dir / "boxes3d.csv", index=False)
    print(f"[lift] {len(lifted)} 3D boxes, median reprojection error {lifted['residual_px'].median():.2f} px")

    predictor = Predictor(args.checkpoint)
    windows = build_video_windows(lifted, predictor.n_observations, args.stride, predictor.max_timestamp_gap)
    preds = predict_windows(predictor, windows)
    preds.to_csv(out_dir / "window_predictions.csv", index=False)
    print(f"[predict] {predictor.mode} model: {len(preds)} windows")

    cars = brno.prefilter(brno.tracks_to_cars(lifted, P, gt["fps"]), gt)
    failed = brno.calculate_speeds(cars, gt, calib)
    t_max = n_samples * step / fps - 1.0
    matches = pd.DataFrame(brno.compute_matches(gt, cars, t_max))
    matches["model_kmh"] = brno.model_speed_for_matches(matches, preds)
    matches.to_csv(out_dir / "car_eval.csv", index=False)

    valid = matches[matches["valid"]]
    scored = valid[valid["matched"]]
    summary = {
        "recording": session.name, "calibration": Path(args.calib).name, "seconds": round(t_max + 1.0, 1),
        "gt_cars_valid": int(len(valid)), "matched": int(len(scored)),
        "recall": float(len(scored) / len(valid)) if len(valid) else float("nan"),
        "tracks_measured": int(sum("speed" in c for c in cars)), "tracks_failed": int(failed),
        "gt_mean_kmh": float(scored["gt_kmh"].mean()) if len(scored) else float("nan"),
    }
    for col, name in (("model_kmh", "model"), ("full_kmh", "geometry_full"), ("median_kmh", "geometry_median")):
        err = scored[col] - scored["gt_kmh"]
        summary[name] = {**brno.error_stats(err), "bias": float(err.mean()) if len(err) else float("nan"),
                         "mean_rel_pct": float((err.abs() / scored["gt_kmh"]).mean() * 100) if len(err) else float("nan")}

    # The calibration file's own system (e.g. Dubska et al.), scored by the same code over the same
    # ground-truth cars: validates this evaluation and gives a reference for these numbers.
    if reference_cars:
        ref = brno.prefilter(reference_cars, gt)
        brno.calculate_speeds(ref, gt, calib)
        ref_m = pd.DataFrame(brno.compute_matches(gt, ref, t_max))
        ref_m = ref_m[ref_m["valid"]]
        ref_scored = ref_m[ref_m["matched"]]
        summary["reference_system"] = {"matched": int(len(ref_scored)), "recall": float(ref_m["matched"].mean())}
        for col, name in (("full_kmh", "full"), ("median_kmh", "median")):
            err = ref_scored[col] - ref_scored["gt_kmh"]
            summary["reference_system"][name] = {**brno.error_stats(err), "bias": float(err.mean())}
        r = summary["reference_system"]
        print(f"[reference] {Path(args.calib).name}'s own tracks, same scoring: {r['matched']} matched "
              f"(recall {r['recall']:.2f}); mean error full {r['full'].get('mean', float('nan')):.2f} km/h, "
              f"median-mode {r['median'].get('mean', float('nan')):.2f} km/h")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[eval] {summary['matched']}/{summary['gt_cars_valid']} valid ground-truth cars matched "
          f"(recall {summary['recall']:.2f}), mean true speed {summary['gt_mean_kmh']:.1f} km/h")
    table = pd.DataFrame({k: summary[k] for k in ("model", "geometry_full", "geometry_median")}).T
    print(table[["n", "mean", "median", "p95", "worst", "bias", "mean_rel_pct"]].round(2).to_string())
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
