"""
Video -> per-vehicle speed with a trained v2 checkpoint.

    video --(YOLO + ByteTrack)--> tracked 2D boxes
          --(lift3d, camera P)--> road-frame 3D cuboids per frame
          --(16-obs windows)-----> Predictor(timestamps, boxes2d, boxes3d) -> m/s

The model inputs are built exactly as at training time: `boxes3d` rows are
[center_x, center_y, center_z, length, width, height] in metres in the road
frame, and `boxes2d` is the min/max box of the *projected fitted cuboid*
(training used projected annotation cuboids, not raw detector boxes).

Detection/tracking needs `ultralytics`, reading/writing video needs
`opencv-python` (requirements-video.txt); both are imported lazily so the
windowing/prediction half of this module works without them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from speed_lstm.lift3d import CUBOID_EDGES, DIM_PRIORS, cuboid_image_corners, fit_track, load_projections
from speed_lstm.data import project_to_bbox

BOX2D_COLS = ["xmin", "ymin", "xmax", "ymax"]
BOX3D_COLS = ["cx", "cy", "cz", "length", "width", "height"]


# ---------------------------------------------------------------- detection

def video_info(video_path: str | Path) -> tuple[float, int, int, int]:
    """(fps, n_frames, width, height)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    info = (cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def detect_and_track(video_path: str | Path, weights: str = "yolo11n.pt", tracker: str = "bytetrack.yaml",
                     conf: float = 0.3, device: str | None = None, max_frames: int | None = None) -> pd.DataFrame:
    """One row per (frame, track_id): frame, track_id, cls, conf, xmin, ymin, xmax, ymax (pixels)."""
    from ultralytics import YOLO

    model = YOLO(weights)
    class_ids = [i for i, name in model.names.items() if name in DIM_PRIORS]
    if not class_ids:
        raise ValueError(f"{weights} has none of the vehicle classes {sorted(DIM_PRIORS)}")

    rows = []
    stream = model.track(source=str(video_path), stream=True, persist=True, tracker=tracker,
                         classes=class_ids, conf=conf, device=device, verbose=False)
    for frame, result in enumerate(stream):
        if max_frames is not None and frame >= max_frames:
            break
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue  # nothing confirmed by the tracker this frame
        xyxy = boxes.xyxy.cpu().numpy()
        for tid, c, p, b in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), xyxy):
            rows.append({"frame": frame, "track_id": tid, "cls": model.names[c], "conf": p,
                         "xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]})
    return pd.DataFrame(rows, columns=["frame", "track_id", "cls", "conf", *BOX2D_COLS])


# --------------------------------------------------------------- timestamps

def frame_timestamps(frames: np.ndarray, fps: float, ts_csv: str | Path | None = None,
                     camera: str | None = None, frame_offset: int = 0) -> np.ndarray:
    """
    Seconds per video frame. Default is frame / fps. With an I-24-style
    timestamp csv (a `frame` column plus one column per camera), looks up
    `camera`'s corrected timestamp at (video frame + frame_offset).
    """
    frames = np.asarray(frames)
    if ts_csv is None:
        if not fps or fps <= 0:
            raise ValueError("Video reports no FPS; pass a timestamp csv instead")
        return frames.astype(np.float64) / fps
    ts = pd.read_csv(ts_csv)
    if camera not in ts.columns:
        raise ValueError(f"Camera {camera!r} is not a column of {ts_csv}")
    lookup = ts.set_index("frame")[camera]
    wanted = frames + frame_offset
    missing = np.setdiff1d(np.unique(wanted), lookup.index.to_numpy())
    if len(missing):
        raise ValueError(f"{len(missing)} frame(s) missing from {ts_csv}, e.g. {missing[:5].tolist()}")
    return lookup.loc[wanted].to_numpy(dtype=np.float64)


# ------------------------------------------------------------------ lifting

def _drop_truncated(dets: pd.DataFrame, img_w: int, img_h: int, margin: float) -> pd.DataFrame:
    """Boxes clipped by the image edge don't bound the whole vehicle and would bias the cuboid fit."""
    inside = ((dets["xmin"] > margin) & (dets["ymin"] > margin)
              & (dets["xmax"] < img_w - margin) & (dets["ymax"] < img_h - margin))
    return dets[inside]


def _smooth(values: np.ndarray, frames: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, applied separately within each run of consecutive frames."""
    if window <= 1:
        return values
    out = values.copy()
    run_starts = np.flatnonzero(np.diff(frames) != 1) + 1
    for run in np.split(np.arange(len(frames)), run_starts):
        seg = pd.DataFrame(values[run])
        out[run] = seg.rolling(window, center=True, min_periods=1).mean().to_numpy()
    return out


def _choose_direction(fits_by_dir: dict[int, list]) -> int:
    """Prefer the direction whose lifted motion agrees with it (EB: x increasing); tie-break on residual."""
    scored = []
    for d, fits in fits_by_dir.items():
        ok = [f for f in fits if f is not None]
        if len(ok) < 2:
            continue
        dx = ok[-1].center[0] - ok[0].center[0]
        agrees = np.sign(dx) == d
        scored.append((not agrees, np.mean([f.residual_px for f in ok]), d))
    return min(scored)[2] if scored else next(iter(fits_by_dir))


def lift_tracks(dets: pd.DataFrame, timestamps: np.ndarray, projections: dict[int, np.ndarray],
                img_size: tuple[int, int], direction: int | None = None,
                border_margin: float = 3.0, smooth_window: int = 5) -> pd.DataFrame:
    """
    dets: output of detect_and_track. timestamps: seconds indexed by frame.
    projections: {+1/-1: metre-space P}, from lift3d.load_projections.
    direction: force +1 (EB) / -1 (WB); None picks per track from its motion.

    Returns one row per kept (track_id, frame) with timestamp, direction,
    the fitted 3D box (BOX3D_COLS, metres), its reprojected 2D box
    (BOX2D_COLS, pixels) and residual_px against the detection.
    """
    dets = _drop_truncated(dets, img_size[0], img_size[1], border_margin)
    out = []
    for tid, g in dets.groupby("track_id", sort=True):
        g = g.sort_values("frame").drop_duplicates("frame")
        cls = g["cls"].mode().iloc[0]
        bboxes = g[BOX2D_COLS].to_numpy(dtype=np.float64)

        candidates = {direction: projections[direction]} if direction is not None else projections
        fits_by_dir = {d: fit_track(bboxes, P, cls) for d, P in candidates.items()}
        d = _choose_direction(fits_by_dir) if direction is None else direction
        P = projections[d]

        keep = [i for i, f in enumerate(fits_by_dir[d]) if f is not None]
        if not keep:
            continue
        fits = [fits_by_dir[d][i] for i in keep]
        frames = g["frame"].to_numpy()[keep]
        centers = _smooth(np.stack([f.center for f in fits]), frames, smooth_window)
        dims = np.stack([f.dims for f in fits])

        for frame, c, dm, f in zip(frames, centers, dims, fits):
            box = project_to_bbox(c, dm, P)
            if box is None:
                continue
            out.append({"track_id": tid, "frame": int(frame), "timestamp": timestamps[frame], "cls": cls,
                        "direction": d, **dict(zip(BOX3D_COLS, [*c, *dm])),
                        **dict(zip(BOX2D_COLS, box.astype(np.float64))), "residual_px": f.residual_px})
    return pd.DataFrame(out, columns=["track_id", "frame", "timestamp", "cls", "direction",
                                      *BOX3D_COLS, *BOX2D_COLS, "residual_px"])


# ---------------------------------------------------------------- windowing

def build_video_windows(lifted: pd.DataFrame, n_observations: int = 16, stride: int = 8,
                        max_timestamp_gap: float = 0.2) -> list[dict]:
    """
    Model-ready windows, same rules as speed_lstm.data: n_observations
    consecutive frames of one track, strictly increasing timestamps, no gap
    above max_timestamp_gap. Each dict is a predict.py-compatible window
    (timestamps, boxes2d, boxes3d) plus track_id / frames for bookkeeping.
    """
    windows = []
    for tid, g in lifted.groupby("track_id", sort=True):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy()
        ts = g["timestamp"].to_numpy(dtype=np.float64)
        b2 = g[BOX2D_COLS].to_numpy(dtype=np.float64)
        b3 = g[BOX3D_COLS].to_numpy(dtype=np.float64)
        # Stride within each run of consecutive frames, so a dropped detection
        # only costs the windows that span it, not the alignment of every later one.
        run_starts = np.flatnonzero(np.diff(frames) != 1) + 1
        for run in np.split(np.arange(len(frames)), run_starts):
            for start in range(0, len(run) - n_observations + 1, stride):
                sl = run[start:start + n_observations]
                dt = np.diff(ts[sl])
                if not (np.all(dt > 0) and np.all(dt <= max_timestamp_gap)):
                    continue
                windows.append({"track_id": int(tid), "frames": frames[sl].tolist(), "timestamps": ts[sl].tolist(),
                                "boxes2d": b2[sl].tolist(), "boxes3d": b3[sl].tolist()})
    return windows


def geometric_speed(window: dict) -> float:
    """Planar path length / elapsed time over the window's 3D centers — the training target's formula."""
    xy = np.asarray(window["boxes3d"])[:, :2]
    t = window["timestamps"]
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / (t[-1] - t[0]))


def predict_windows(predictor, windows: list[dict]) -> pd.DataFrame:
    rows = []
    for w in windows:
        speed = predictor.predict(timestamps=w["timestamps"], boxes2d=w["boxes2d"], boxes3d=w["boxes3d"])
        rows.append({"track_id": w["track_id"], "start_frame": w["frames"][0], "end_frame": w["frames"][-1],
                     "t_start": w["timestamps"][0], "t_end": w["timestamps"][-1],
                     "speed_mps": speed, "speed_kmh": speed * 3.6, "geometric_mps": geometric_speed(w)})
    return pd.DataFrame(rows, columns=["track_id", "start_frame", "end_frame", "t_start", "t_end",
                                       "speed_mps", "speed_kmh", "geometric_mps"])


def summarize_tracks(preds: pd.DataFrame, lifted: pd.DataFrame) -> pd.DataFrame:
    """Per vehicle: median window speed, plus class/direction/dims for context."""
    if preds.empty:
        return pd.DataFrame(columns=["track_id", "cls", "direction", "n_windows", "speed_mps", "speed_kmh",
                                     "geometric_mps", "length", "width", "height"])
    agg = preds.groupby("track_id").agg(n_windows=("speed_mps", "size"), speed_mps=("speed_mps", "median"),
                                        geometric_mps=("geometric_mps", "median"))
    agg["speed_kmh"] = agg["speed_mps"] * 3.6
    meta = lifted.groupby("track_id").agg(cls=("cls", "first"), direction=("direction", "first"),
                                          length=("length", "median"), width=("width", "median"),
                                          height=("height", "median"))
    return agg.join(meta).reset_index()[["track_id", "cls", "direction", "n_windows", "speed_mps", "speed_kmh",
                                         "geometric_mps", "length", "width", "height"]]


# ---------------------------------------------------------------- rendering

def render_video(video_path: str | Path, out_path: str | Path, lifted: pd.DataFrame, preds: pd.DataFrame,
                 projections: dict[int, np.ndarray], max_frames: int | None = None) -> None:
    """Draw each fitted cuboid, plus the latest window speed that has ended by that frame."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (w, h))

    by_frame = {f: g for f, g in lifted.groupby("frame")}
    speeds = {tid: g.sort_values("end_frame")[["end_frame", "speed_kmh"]].to_numpy()
              for tid, g in preds.groupby("track_id")}

    frame = 0
    while True:
        ok, img = cap.read()
        if not ok or (max_frames is not None and frame >= max_frames):
            break
        for _, r in by_frame.get(frame, pd.DataFrame()).iterrows():
            corners = cuboid_image_corners(r[BOX3D_COLS[:3]].to_numpy(float), r[BOX3D_COLS[3:]].to_numpy(float),
                                           projections[int(r["direction"])])
            if corners is None:
                continue
            color = (0, 200, 255) if r["direction"] == 1 else (255, 160, 0)
            for a, b in CUBOID_EDGES:
                cv2.line(img, tuple(map(int, corners[a])), tuple(map(int, corners[b])), color, 2, cv2.LINE_AA)
            label = f"#{int(r['track_id'])} {r['cls']}"
            hist = speeds.get(r["track_id"])
            if hist is not None:
                done = hist[hist[:, 0] <= frame]
                if len(done):
                    label += f" {done[-1, 1]:.0f} km/h"
            org = (int(r["xmin"]), max(int(r["ymin"]) - 6, 12))
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        writer.write(img)
        frame += 1
    cap.release()
    writer.release()


# ---------------------------------------------------------------------- CLI

def main() -> None:
    from speed_lstm.model import Predictor

    p = argparse.ArgumentParser(description="Video -> 3D boxes -> speed with a v2 checkpoint.")
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True, help="best.pt saved by speed_lstm.train")
    p.add_argument("--calib", required=True,
                   help='I-24 hg.json (with --camera) or a single-camera json {"P": 3x4}')
    p.add_argument("--camera", default=None, help="Camera name inside hg.json / timestamp csv, e.g. p1c1")
    p.add_argument("--calib-units", choices=["ft", "m"], default="ft",
                   help="World units P expects (I-24 hg.json uses feet)")
    p.add_argument("--calib-image-scale", type=float, default=1.0,
                   help="Video resolution / calibration resolution, e.g. 0.5 for 1080p video with a 4K calibration")
    p.add_argument("--direction", choices=["auto", "EB", "WB"], default="auto",
                   help="Force travel direction for every track; auto picks per track from its motion")
    p.add_argument("--ts-csv", default=None, help="Optional I-24 timestamp csv; default is frame / fps")
    p.add_argument("--frame-offset", type=int, default=0, help="Added to video frame index for --ts-csv lookup")
    p.add_argument("--weights", default="yolo11n.pt", help="Ultralytics detector weights")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--detections", default=None,
                   help="Reuse a detections.csv from an earlier run instead of running YOLO again")
    p.add_argument("--stride", type=int, default=8, help="Frames between window starts")
    p.add_argument("--smooth", type=int, default=5, help="Centered moving-average length for 3D centers (1 = off)")
    p.add_argument("--out-dir", default=None, help="Default: runs/video/<video stem>")
    p.add_argument("--render", action="store_true", help="Also write annotated.mp4 with cuboids + speeds")
    args = p.parse_args()

    out_dir = Path(args.out_dir or Path("runs/video") / Path(args.video).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = Predictor(args.checkpoint)
    projections = load_projections(args.calib, args.camera, args.calib_units, args.calib_image_scale)
    direction = None if args.direction == "auto" else {"EB": 1, "WB": -1}[args.direction]
    if direction is not None and direction not in projections:
        raise ValueError(f"No {args.direction} projection for this camera in {args.calib}")

    fps, n_frames, img_w, img_h = video_info(args.video)
    print(f"[video] {args.video}: {n_frames} frames @ {fps:.2f} fps, {img_w}x{img_h}")
    if fps and abs(fps - 30.0) > 1.0:
        print(f"[video] warning: model windows were trained at ~30 fps; {fps:.1f} fps changes the "
              f"time span of a {predictor.n_observations}-frame window")

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    if args.detections:
        dets = pd.read_csv(args.detections)
        if args.max_frames is not None:
            dets = dets[dets["frame"] < args.max_frames]
    else:
        dets = detect_and_track(args.video, args.weights, args.tracker, args.conf, args.device, args.max_frames)
    dets.to_csv(out_dir / "detections.csv", index=False)
    print(f"[track] {len(dets)} detections, {dets['track_id'].nunique()} tracks")

    n_ts = int(dets["frame"].max()) + 1 if len(dets) else 0
    timestamps = frame_timestamps(np.arange(n_ts), fps, args.ts_csv, args.camera, args.frame_offset)

    lifted = lift_tracks(dets, timestamps, projections, (img_w, img_h), direction, smooth_window=args.smooth)
    lifted.to_csv(out_dir / "boxes3d.csv", index=False)
    print(f"[lift] {len(lifted)} 3D boxes, median reprojection error "
          f"{lifted['residual_px'].median() if len(lifted) else float('nan'):.2f} px")

    windows = build_video_windows(lifted, predictor.n_observations, args.stride, predictor.max_timestamp_gap)
    (out_dir / "windows.json").write_text(json.dumps(windows))
    preds = predict_windows(predictor, windows)
    preds.to_csv(out_dir / "window_predictions.csv", index=False)
    summary = summarize_tracks(preds, lifted)
    summary.to_csv(out_dir / "track_speeds.csv", index=False)
    print(f"[predict] {predictor.mode} model: {len(windows)} windows over {len(summary)} vehicles")
    if len(summary):
        print(summary[["track_id", "cls", "direction", "n_windows", "speed_kmh"]].to_string(index=False))

    if args.render:
        render_video(args.video, out_dir / "annotated.mp4", lifted, preds, projections, args.max_frames)
        print(f"[render] {out_dir / 'annotated.mp4'}")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
