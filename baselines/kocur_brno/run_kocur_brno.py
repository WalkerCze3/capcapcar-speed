#!/usr/bin/env python3
"""
Run the Kocur 3D-box detector (YOLOv6-3D, authors' code and weights) on one
BrnoCompSpeed recording and save every 3D box, on the same frames our
YOLO + ByteTrack run used (scripts/brno_eval_cli.py: every `step`-th frame
of the first --max-seconds).

Inputs follow the authors' pipeline (speed_estimation.batch_process_video):
VPs from a calibration json and the recording's video_mask.png. The default
calibration is the one our lift3d run uses (system_dubska_optimal_calib.json),
so only the 3D-box step differs between the two.

Output <out>/detections_kocur.csv: frame (processed-frame index, as in our
detections.csv), the warped 2D box + c ("fub"), the 8 image corners of the
3D box and the authors' ground point (gx, gy). <out>/meta.json holds the warp M.

Example:
    python baselines/kocur_brno/run_kocur_brno.py --kocur-repo /content/kocur \
        --session-dir .../dataset/session4_center \
        --calib .../results/session4_center/system_dubska_optimal_calib.json \
        --out runs/kocur_brno/session4_center
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "kocur_i24"))
sys.path.insert(0, str(ROOT / "src"))
from kocur import import_kocur  # noqa: E402
from speed_lstm import brno  # noqa: E402

CORNER_COLS = [f"c{i}{a}" for i in range(8) for a in "xy"]
COLS = ["frame", "wx1", "wy1", "wx2", "wy2", "fub", "gx", "gy"] + CORNER_COLS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kocur-repo", required=True, help="clone of gajdosech2/Vehicle-Speed-Estimation-YOLOv6-3D")
    ap.add_argument("--weights", default="checkpoints/small/yolov6_small_3d_transform.pt",
                    help="path inside --kocur-repo (default: Small, the paper's best median error)")
    ap.add_argument("--yolo-size", type=int, nargs=2, default=[360, 640],
                    help="model input (h w); 360 640 is the paper's best Small setting")
    ap.add_argument("--warp-size", type=int, nargs=2, default=[960, 540], help="authors' default (w h)")
    ap.add_argument("--session-dir", required=True, help="dataset/sessionN_<left|center|right>")
    ap.add_argument("--calib", required=True, help="results/<same recording>/system_*.json")
    ap.add_argument("--video", default=None, help="default: <session-dir>/video.avi")
    ap.add_argument("--max-seconds", type=float, default=600.0, help="same as the brno_eval_cli run")
    ap.add_argument("--frame-step", type=int, default=None,
                    help="default: same rule as brno_eval_cli (closest to --target-fps)")
    ap.add_argument("--target-fps", type=float, default=25.0)
    ap.add_argument("--conf", type=float, default=0.65, help="authors' default 0.65")
    ap.add_argument("--iou", type=float, default=0.65, help="authors' default 0.65")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    KU, Radar, Inferer = import_kocur(args.kocur_repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = Path(args.session_dir)

    # Same time base and frame selection as brno_eval_cli: decoded frame / ground-truth fps.
    gt = brno.load_gt(session / "gt_data.pkl")
    fps = float(gt["fps"])
    step = args.frame_step or max(1, round(fps / args.target_fps))
    max_samples = int(args.max_seconds * fps / step)

    calib, _ = brno.load_system(args.calib)
    vp1, vp2, vp3, pp, _, _ = KU.compute_camera_calibration([calib["vp1"], calib["vp2"]], calib["pp"])
    vp1, vp2, vp3 = (np.asarray(v[:-1] / v[-1], dtype=np.float64) for v in (vp1, vp2, vp3))
    mask = cv2.imread(str(session / "video_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"no video_mask.png in {session}")

    im_w, im_h = args.warp_size
    M, IM = KU.get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h, constraint=0.8, vp_top=None)
    vp0_t = cv2.perspectiveTransform(np.array([[vp1]], np.float32), M)[0][0]
    radar = Radar(transform_matrix=M, inv_transform_matrix=IM, vanishing_points=[vp1, vp2, vp3], vp0_t=vp0_t,
                  image_size=(im_w, im_h), projector=None, video_fps=int(round(fps / step)), result_path=str(out),
                  result_name="kocur", camera_calib_structure=calib, save_often=False)
    inferer = Inferer(model=os.path.join(args.kocur_repo, args.weights), yolo_img_size=args.yolo_size, half=False)

    video = args.video or str(session / "video.avi")
    cap = cv2.VideoCapture(video)
    print(f"[run] {video}: every {step} frame(s), {max_samples} processed frames; calib {Path(args.calib).name}")

    rows, sample, n_failed, t0, done = [], 0, 0, time.time(), False
    while not done and sample < max_samples:
        frames = []
        while len(frames) < args.batch and sample + len(frames) < max_samples:
            # ultralytics vid_stride grabs `step` frames and keeps the last, so sample k = video frame k*step + step-1
            ok = all(cap.grab() for _ in range(step))
            if ok:
                ok, frame = cap.retrieve()
            if not ok:
                done = True
                break
            frames.append(frame)
        if not frames:
            break
        warped = [cv2.warpPerspective(cv2.bitwise_and(f, f, mask=mask), M, (im_w, im_h),
                                      borderMode=cv2.BORDER_CONSTANT) for f in frames]
        if sample == 0:
            cv2.imwrite(str(out / "warped_example.jpg"), warped[0])
        boxes, fubs = inferer.simple_inference(np.stack(warped), args.conf, args.iou)
        for k, (bb, fb) in enumerate(zip(boxes, fubs)):
            radar.frame += 1
            for b, f in zip(bb, fb):
                radar.get_tracker(b)  # get_3d_bbox reads the tracker state the authors keep per box
                try:
                    corners, center = radar.get_3d_bbox(b, float(f[0]))
                    corners = np.asarray(corners, dtype=np.float64).reshape(8, 2)
                except (TypeError, ValueError):  # their intersection() returns False for parallel lines
                    n_failed += 1
                    continue
                row = {"frame": sample + k, "wx1": b[0], "wy1": b[1], "wx2": b[2], "wy2": b[3], "fub": float(f[0]),
                       "gx": float(center[0]), "gy": float(center[1])}
                row.update(dict(zip(CORNER_COLS, corners.ravel())))
                rows.append(row)
            radar.remove_tracker()
        sample += len(frames)
        if sample % (args.batch * 50) < args.batch:
            print(f"[run] {sample}/{max_samples} frames, {len(rows)} boxes, {sample / (time.time() - t0):.1f} fps",
                  flush=True)
    cap.release()

    pd.DataFrame(rows, columns=COLS).to_csv(out / "detections_kocur.csv", index=False)
    meta = {"video": video, "calib": args.calib, "weights": args.weights, "yolo_size": args.yolo_size,
            "warp_size": [im_w, im_h], "conf": args.conf, "iou": args.iou, "fps": fps, "frame_step": step,
            "frames_processed": sample, "boxes": len(rows), "box_construction_failed": n_failed,
            "vps": [vp1.tolist(), vp2.tolist(), vp3.tolist()], "M": M.tolist(), "IM": IM.tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {sample} frames, {len(rows)} boxes ({n_failed} failed constructions) -> {out}")


if __name__ == "__main__":
    main()
