#!/usr/bin/env python3
"""
Run the Kocur 3D-box detector (YOLOv6-3D, authors' code and weights) on one
I-24 camera video and save every 3D box.

Inputs follow the authors' pipeline (speed_estimation.batch_process_video):
VP1/VP2/VP3 and a road mask. For I-24 the VPs come from the dataset's
hg.json (so this tests the 3D-box step with the dataset calibration, not
auto-calibration), and the road mask is the hull of the annotated vehicle
footprints seen by this camera (a stand-in for the hand-drawn road masks
BrnoCompSpeed ships).

Output <out>/detections.csv: one row per detection with the video frame,
the authors' IoU-tracker id, the warped 2D box + c ("fub"), the 8 image
corners of the 3D box (their order) and their ground point (`center`, the
middle of the bottom edge nearest the camera, which they use for speed).

Example:
    python baselines/kocur_i24/run_kocur_i24.py --kocur-repo /content/kocur \
        --video .../scene1/p2c1.mp4 --data-dir .../data --scene scene1 --camera p2c1 \
        --out runs/kocur_i24/scene1_p2c1
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i24  # noqa: E402
from kocur import import_kocur  # noqa: E402


def road_mask_from_gt(gt: pd.DataFrame, cam: dict, width: int, height: int, dilate_px: int) -> np.ndarray:
    pts = []
    for _, row in gt.iterrows():
        P = cam["P"].get(int(row["direction"]))
        if P is None:
            continue
        q = (P @ np.c_[i24.gt_corners(row)[:4], np.ones(4)].T).T
        if np.any(q[:, 2] <= 1e-9):
            continue
        uv = q[:, :2] / q[:, 2:3]
        if np.all(np.isfinite(uv)):
            pts.append(uv)
    if not pts:
        raise SystemExit("no annotated footprints project into this camera")
    pts = np.concatenate(pts)
    pts = pts[(pts[:, 0] > -width) & (pts[:, 0] < 2 * width) & (pts[:, 1] > -height) & (pts[:, 1] < 2 * height)]
    hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    if dilate_px:
        mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8))
    return mask


def clip_mask_to_vp_line(mask: np.ndarray, vp2: np.ndarray, vp3: np.ndarray, margin_px: float = 40) -> np.ndarray:
    """The warp maps the VP2-VP3 line to infinity; keep only the mask side away from it."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = vp3 - vp2
    side = (xx - vp2[0]) * d[1] - (yy - vp2[1]) * d[0]
    dist = side / np.hypot(*d)
    on = mask > 0
    keep_pos = (dist[on] > 0).sum() >= (dist[on] < 0).sum()
    good = dist > margin_px if keep_pos else dist < -margin_px
    out = mask.copy()
    out[~good] = 0
    removed = on.sum() - (out > 0).sum()
    if removed:
        print(f"[mask] removed {removed} px on the far side of the VP2-VP3 line")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kocur-repo", required=True, help="clone of gajdosech2/Vehicle-Speed-Estimation-YOLOv6-3D")
    ap.add_argument("--weights", default="checkpoints/nano/yolov6_nano_3d_transform.pt",
                    help="path inside --kocur-repo (authors' default: nano)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--data-dir", required=True, help="I-24 data dir with obj/, ts/, hg/")
    ap.add_argument("--scene", default="scene1")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--warp-size", type=int, nargs=2, default=[960, 540], help="authors' default (w h)")
    ap.add_argument("--yolo-size", type=int, nargs=2, default=[270, 480], help="authors' default (h w)")
    ap.add_argument("--conf", type=float, default=0.65, help="authors' default 0.65")
    ap.add_argument("--iou", type=float, default=0.65, help="authors' default 0.65")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mask-dilate", type=int, default=41)
    args = ap.parse_args()

    KU, Radar, Inferer = import_kocur(args.kocur_repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {width}x{height} @ {fps:.2f} fps, {n_video} frames")

    data = Path(args.data_dir)
    cam = i24.load_camera(data / "hg" / f"{args.scene}_hg.json", args.camera, width)
    gt = i24.load_gt(data / "obj" / f"{args.scene}_annotations.csv", data / "ts" / f"{args.scene}_ts.csv",
                     args.camera)
    vp1, vp2, vp3 = cam["vps"]
    print(f"[calib] scale {cam['scale']:.3f}; VP1 {vp1.round(0)}, VP2 {vp2.round(0)}, VP3 {vp3.round(0)}")

    mask = road_mask_from_gt(gt, cam, width, height, args.mask_dilate)
    mask = clip_mask_to_vp_line(mask, vp2, vp3)
    cv2.imwrite(str(out / "road_mask.png"), mask)

    im_w, im_h = args.warp_size
    # Same call and VP order as speed_estimation.batch_process_video (pair VP3-VP2).
    M, IM = KU.get_transform_matrix_with_criterion(vp3, vp2, mask, im_w, im_h, constraint=0.8, vp_top=None)
    vp0_t = cv2.perspectiveTransform(np.array([[vp1]], np.float32), M)[0][0]
    radar = Radar(transform_matrix=M, inv_transform_matrix=IM, vanishing_points=[vp1, vp2, vp3], vp0_t=vp0_t,
                  image_size=(im_w, im_h), projector=None, video_fps=int(round(fps)), result_path=str(out),
                  result_name="kocur", camera_calib_structure={"vp1": vp1.tolist(), "vp2": vp2.tolist()},
                  save_often=False)
    inferer = Inferer(model=os.path.join(args.kocur_repo, args.weights), yolo_img_size=args.yolo_size, half=False)

    meta = {"video": args.video, "camera": args.camera, "scene": args.scene, "width": width, "height": height,
            "fps": fps, "calib_scale": cam["scale"], "vps": cam["vps"].tolist(), "M": M.tolist(), "IM": IM.tolist(),
            "vp1_warped": vp0_t.tolist(), "weights": args.weights, "conf": args.conf, "iou": args.iou,
            "warp_size": [im_w, im_h], "yolo_size": args.yolo_size}

    rows, frame_idx, n_failed, t0 = [], 0, 0, time.time()
    limit = args.max_frames or n_video
    done = False
    while not done:
        frames = []
        while len(frames) < args.batch and frame_idx + len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                done = True
                break
            frames.append(frame)
        if not frames:
            break
        warped = [cv2.warpPerspective(cv2.bitwise_and(f, f, mask=mask), M, (im_w, im_h),
                                      borderMode=cv2.BORDER_CONSTANT) for f in frames]
        if frame_idx == 0:
            cv2.imwrite(str(out / "warped_example.jpg"), warped[0])
        boxes, fubs = inferer.simple_inference(np.stack(warped), args.conf, args.iou)
        for k, (bb, fb) in enumerate(zip(boxes, fubs)):
            radar.frame += 1  # Radar.process_frame bookkeeping, without drawing
            for b, f in zip(bb, fb):
                tracker = radar.get_tracker(b)
                try:
                    corners, center = radar.get_3d_bbox(b, float(f[0]))
                    corners = np.asarray(corners, dtype=np.float64).reshape(8, 2)
                except (TypeError, ValueError):  # their intersection() returns False for parallel lines
                    n_failed += 1
                    continue
                row = {"frame": frame_idx + k, "track_id": tracker.id,
                       "wx1": b[0], "wy1": b[1], "wx2": b[2], "wy2": b[3], "fub": float(f[0]),
                       "gx": float(center[0]), "gy": float(center[1])}
                row.update({f"c{i}{a}": corners[i, j] for i in range(8) for j, a in enumerate("xy")})
                rows.append(row)
            radar.remove_tracker()
        frame_idx += len(frames)
        if frame_idx % (args.batch * 20) < args.batch:
            print(f"[run] {frame_idx}/{limit} frames, {len(rows)} boxes, {frame_idx / (time.time() - t0):.1f} fps",
                  flush=True)
        if frame_idx >= limit:
            break
    cap.release()

    cols = ["frame", "track_id", "wx1", "wy1", "wx2", "wy2", "fub", "gx", "gy"] + \
        [f"c{i}{a}" for i in range(8) for a in "xy"]
    pd.DataFrame(rows, columns=cols).to_csv(out / "detections.csv", index=False)
    meta["frames_processed"] = frame_idx
    meta["boxes"] = len(rows)
    meta["box_construction_failed"] = n_failed
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {frame_idx} frames, {len(rows)} boxes -> {out / 'detections.csv'}")


if __name__ == "__main__":
    main()
