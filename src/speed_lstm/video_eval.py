"""
Evaluate a video_speed run against I-24 annotations for the same scene/camera.

1. Calibration check: projects annotation cuboids through the camera's P
   under each (units, image scale) hypothesis and measures how well they
   overlap YOLO detections. The right hypothesis gives high IoU; the others
   land nowhere near the vehicles. This settles --calib-units and
   --calib-image-scale empirically instead of by assumption.
2. Track matching: each video track -> the annotated vehicle id its
   detections overlap most.
3. Per window: ground-truth speed (path length / time over the annotated
   centers, i.e. the exact training target) vs.
     - speed_mps:      model on video-lifted 3D boxes (the real pipeline)
     - geometric_mps:  path speed of the video-lifted centers (no model)
     - model_on_annotations_mps: model on the annotation boxes of the same
       frames, i.e. the training-distribution input — so
       (speed_mps error) - (this error) is what monocular lifting costs.

Usage:
    python -m speed_lstm.video_eval --run-dir runs/video/p1c1 --data-dir /path/to/i24 --scene scene1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from speed_lstm.data import (load_annotations, load_homography, load_timestamps, metric_center_and_dims,
                             project_to_bbox)
from speed_lstm.lift3d import DIRECTION_KEYS, scale_projection
from speed_lstm.video import BOX2D_COLS

CALIB_HYPOTHESES = [(u, s) for u in ("ft", "m") for s in (1.0, 0.5, 2.0)]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, 4), b: (M, 4) xyxy -> (N, M) IoU."""
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def camera_projections(hg: dict, camera: str, units: str, image_scale: float) -> dict[int, np.ndarray]:
    S = np.diag([image_scale, image_scale, 1.0])
    return {d: S @ scale_projection(hg[key][camera]["P"], units)
            for d, key in DIRECTION_KEYS.items() if camera in hg.get(key, {})}


def annotation_boxes(ann_cam: pd.DataFrame, projections: dict[int, np.ndarray]) -> pd.DataFrame:
    """Projected xyxy box per annotation row (rows whose cuboid crosses the horizon are dropped)."""
    rows = []
    for r in ann_cam.itertuples(index=False):
        P = projections.get(int(r.direction))
        if P is None:
            continue
        c, d = metric_center_and_dims(pd.Series(r._asdict()))
        box = project_to_bbox(c, d, P)
        if box is not None:
            rows.append((r.frame, r.id, *box))
    return pd.DataFrame(rows, columns=["frame", "id", *BOX2D_COLS])


def best_ious(dets: pd.DataFrame, ann_boxes: pd.DataFrame, frame_offset: int = 0) -> pd.DataFrame:
    """For each detection: the best-overlapping annotated id in the same frame and its IoU."""
    by_frame = {f: g for f, g in ann_boxes.groupby("frame")}
    out = []
    for frame, g in dets.groupby("frame"):
        a = by_frame.get(frame + frame_offset)
        if a is None:
            continue
        ious = iou_matrix(g[BOX2D_COLS].to_numpy(float), a[BOX2D_COLS].to_numpy(float))
        j = ious.argmax(axis=1)
        out.append(pd.DataFrame({"frame": frame, "track_id": g["track_id"].to_numpy(),
                                 "id": a["id"].to_numpy()[j], "iou": ious[np.arange(len(g)), j]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["frame", "track_id", "id", "iou"])


def calibration_check(dets: pd.DataFrame, ann_cam: pd.DataFrame, hg: dict, camera: str,
                      frame_offset: int = 0, n_frames: int = 150) -> pd.DataFrame:
    frames = np.intersect1d(dets["frame"].unique() + frame_offset, ann_cam["frame"].unique())
    frames = frames[np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)] if len(frames) else frames
    ann_s = ann_cam[ann_cam["frame"].isin(frames)]
    dets_s = dets[dets["frame"].isin(frames - frame_offset)]
    rows = []
    for units, scale in CALIB_HYPOTHESES:
        m = best_ious(dets_s, annotation_boxes(ann_s, camera_projections(hg, camera, units, scale)), frame_offset)
        rows.append({"units": units, "image_scale": scale, "n_dets": len(m),
                     "median_iou": float(m["iou"].median()) if len(m) else 0.0,
                     "frac_iou_gt_0.5": float((m["iou"] > 0.5).mean()) if len(m) else 0.0})
    return pd.DataFrame(rows).sort_values("median_iou", ascending=False, ignore_index=True)


def match_tracks(matches: pd.DataFrame, min_iou: float = 0.5, min_agree: float = 0.6) -> dict[int, int]:
    """track_id -> annotated id, when most of the track's well-overlapping detections agree on one id."""
    good = matches[matches["iou"] >= min_iou]
    out = {}
    for tid, g in good.groupby("track_id"):
        counts = g["id"].value_counts()
        if counts.iloc[0] / len(g) >= min_agree:
            out[int(tid)] = int(counts.index[0])
    return out


def annotation_window(ann_cam: pd.DataFrame, ts_cam: pd.Series, vid: int, frames: list[int]):
    """(timestamps, centers, dims, direction) for vehicle `vid` at exactly `frames`, or None if any is missing."""
    g = ann_cam[(ann_cam["id"] == vid) & ann_cam["frame"].isin(frames)].drop_duplicates("frame").set_index("frame")
    if len(g) != len(frames) or not set(frames) <= set(ts_cam.index):
        return None
    g = g.loc[frames]
    cd = [metric_center_and_dims(r) for _, r in g.reset_index().iterrows()]
    return (ts_cam.loc[frames].to_numpy(np.float64), np.stack([c for c, _ in cd]), np.stack([d for _, d in cd]),
            int(g["direction"].iloc[0]))


def path_speed(t: np.ndarray, centers: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(centers[:, :2], axis=0), axis=1).sum() / (t[-1] - t[0]))


def evaluate(run_dir: str | Path, data_dir: str | Path, scene: str, camera: str | None = None,
             frame_offset: int = 0) -> dict:
    from speed_lstm.model import Predictor

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    camera = camera or cfg["camera"]
    dets = pd.read_csv(run_dir / "detections.csv")
    preds = pd.read_csv(run_dir / "window_predictions.csv")
    windows = json.loads((run_dir / "windows.json").read_text())

    ann = load_annotations(Path(data_dir), scene)
    ann_cam = ann[ann["camera"] == camera]
    if ann_cam.empty:
        raise ValueError(f"No annotations for camera {camera!r} in {scene} (cameras: {sorted(ann['camera'].unique())})")
    ts_long = load_timestamps(Path(data_dir), scene)
    ts_cam = ts_long[ts_long["camera"] == camera].set_index("frame")["timestamp"]
    hg = load_homography(Path(data_dir), scene)

    calib = calibration_check(dets, ann_cam, hg, camera, frame_offset)
    calib.to_csv(run_dir / "calibration_check.csv", index=False)
    best = calib.iloc[0]
    used = (cfg.get("calib_units", "ft"), float(cfg.get("calib_image_scale", 1.0)))

    projections = camera_projections(hg, camera, best["units"], best["image_scale"])
    matches = best_ious(dets, annotation_boxes(ann_cam, projections), frame_offset)
    track_to_id = match_tracks(matches)

    predictor = Predictor(cfg["checkpoint"])
    # Raw hg P with metric input, exactly as speed_lstm.data built training boxes.
    raw_P = {d: np.asarray(hg[k][camera]["P"], dtype=np.float64) for d, k in DIRECTION_KEYS.items()
             if camera in hg.get(k, {})}

    rows = []
    for w, (_, p) in zip(windows, preds.iterrows()):
        vid = track_to_id.get(w["track_id"])
        if vid is None:
            continue
        aw = annotation_window(ann_cam, ts_cam, vid, [f + frame_offset for f in w["frames"]])
        if aw is None:
            continue
        t, centers, dims, d = aw
        boxes2d = None
        if predictor.mode in ("2d", "combined"):
            b = [project_to_bbox(c, dm, raw_P[d]) for c, dm in zip(centers, dims)]
            boxes2d = None if any(x is None for x in b) else np.stack(b)
        on_ann = (predictor.predict(t, boxes2d=boxes2d, boxes3d=np.hstack([centers, dims]))
                  if boxes2d is not None or predictor.mode == "3d" else np.nan)
        rows.append({"track_id": w["track_id"], "vehicle_id": vid, "start_frame": w["frames"][0],
                     "gt_mps": path_speed(t, centers), "speed_mps": p["speed_mps"],
                     "geometric_mps": p["geometric_mps"], "model_on_annotations_mps": on_ann})
    ev = pd.DataFrame(rows)
    ev.to_csv(run_dir / "window_eval.csv", index=False)

    def metrics(col):
        if ev.empty:
            return {"mae": None, "rmse": None, "bias": None}
        e = (ev[col] - ev["gt_mps"]).dropna()
        return {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}

    summary = {
        "calibration_best": {"units": best["units"], "image_scale": float(best["image_scale"]),
                             "median_iou": float(best["median_iou"])},
        "calibration_used_by_run": {"units": used[0], "image_scale": used[1]},
        "calibration_mismatch": (best["units"], float(best["image_scale"])) != used,
        "tracks": int(dets["track_id"].nunique()), "tracks_matched": len(track_to_id),
        "windows": len(windows), "windows_evaluated": len(ev),
        "gt_mean_mps": float(ev["gt_mps"].mean()) if len(ev) else None,
        "model_video_mps": metrics("speed_mps"),
        "geometric_video_mps": metrics("geometric_mps"),
        "model_on_annotations_mps": metrics("model_on_annotations_mps"),
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Score a video_speed run against I-24 annotations.")
    p.add_argument("--run-dir", required=True, help="Output dir of video_speed_cli.py")
    p.add_argument("--data-dir", required=True, help="I-24 dataset root with obj/, ts/, hg/")
    p.add_argument("--scene", required=True, help="e.g. scene1")
    p.add_argument("--camera", default=None, help="Defaults to the run's --camera")
    p.add_argument("--frame-offset", type=int, default=0, help="Annotation frame = video frame + offset")
    args = p.parse_args()

    summary = evaluate(args.run_dir, args.data_dir, args.scene, args.camera, args.frame_offset)
    print(pd.read_csv(Path(args.run_dir) / "calibration_check.csv").to_string(index=False))
    print(json.dumps(summary, indent=2))
    if summary["calibration_mismatch"]:
        b = summary["calibration_best"]
        print(f"\n[eval] WARNING: detections line up best with --calib-units {b['units']} "
              f"--calib-image-scale {b['image_scale']}, not what this run used. Re-run with those "
              f"(add --detections {Path(args.run_dir) / 'detections.csv'} to skip YOLO).")


if __name__ == "__main__":
    main()
