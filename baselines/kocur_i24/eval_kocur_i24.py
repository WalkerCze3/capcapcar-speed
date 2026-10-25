#!/usr/bin/env python3
"""
Score Kocur 3D boxes (run_kocur_i24.py output) against the I-24 annotations.

1. Frame offset: the video and annotation frame counters may not start
   together, so the offset maximising the image-space IoU between Kocur
   boxes and projected annotated cuboids is searched over +-R frames.
2. Per frame, Kocur boxes are matched to annotated vehicles on the road
   plane (footprint-center distance, greedy, threshold --match-m).
3. Per matched box: footprint-center error, error of the ground point the
   authors use for speed (bottom edge nearest the camera) vs the annotated
   face nearest the camera, length / width / height error, bird's-eye IoU.
4. Per tracked vehicle (authors' IoU tracker, >= --min-frames matched
   frames): speed from a linear fit of along-road position vs the camera's
   timestamps, Kocur ground point vs annotated face, same frames.

Writes summary.json, matches.csv, speeds.csv, results.md and, with --video,
overlay_*.jpg (green = annotation, orange = Kocur).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i24  # noqa: E402
from kocur import ground_corner_pairs  # noqa: E402

KOCUR_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]
GT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyxy boxes (N, 4) x (M, 4) -> (N, M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def bev_iou(p: np.ndarray, q: np.ndarray) -> float:
    try:
        from shapely.geometry import MultiPoint
    except ImportError:
        return float("nan")
    a, b = MultiPoint([tuple(v) for v in p]).convex_hull, MultiPoint([tuple(v) for v in q]).convex_hull
    u = a.union(b).area
    return float(a.intersection(b).area / u) if u > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--video", default=None, help="draw overlays from this video")
    ap.add_argument("--offset-range", type=int, default=90)
    ap.add_argument("--offset", type=int, default=None, help="skip the search and use this offset")
    ap.add_argument("--match-m", type=float, default=3.0)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--n-overlays", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run_dir)
    meta = json.load(open(run / "meta.json"))
    det = pd.read_csv(run / "detections.csv")
    scene, camera = meta["scene"], meta["camera"]
    data = Path(args.data_dir)
    cam = i24.load_camera(data / "hg" / f"{scene}_hg.json", camera, meta["width"])
    gt = i24.load_gt(data / "obj" / f"{scene}_annotations.csv", data / "ts" / f"{scene}_ts.csv", camera)
    M = np.array(meta["M"])
    Hinv = {d: i24.ground_homography_inv(P) for d, P in cam["P"].items()}
    cam_x = {d: i24.camera_center(P)[0] for d, P in cam["P"].items()}
    mask = cv2.imread(str(run / "road_mask.png"), cv2.IMREAD_GRAYSCALE)
    W, H = meta["width"], meta["height"]

    corners = det[[f"c{i}{a}" for i in range(8) for a in "xy"]].to_numpy().reshape(-1, 8, 2)
    det_box2d = np.c_[corners.min(1), corners.max(1)]

    # Projected annotated cuboids, per annotation frame.
    gt_img = []
    for _, row in gt.iterrows():
        P = cam["P"].get(int(row["direction"]))
        q = None if P is None else (P @ np.c_[i24.gt_corners(row, cam["up"][int(row["direction"])]), np.ones(8)].T).T
        if q is None or np.any(q[:, 2] <= 1e-9):
            gt_img.append(None)
            continue
        gt_img.append(q[:, :2] / q[:, 2:3])
    gt["ok"] = [g is not None for g in gt_img]
    gt_box2d = np.array([np.r_[g.min(0), g.max(0)] if g is not None else [np.nan] * 4 for g in gt_img])
    gt_by_frame = {f: idx.to_numpy() for f, idx in gt[gt["ok"]].groupby("frame").groups.items()}
    det_by_frame = {f: idx.to_numpy() for f, idx in det.groupby("frame").groups.items()}

    # 1. frame offset (annotation frame = video frame - offset)
    def mean_best_iou(offset: int, frames) -> float:
        vals = []
        for f in frames:
            g = gt_by_frame.get(f - offset)
            if g is None:
                continue
            vals.extend(iou_matrix(det_box2d[det_by_frame[f]], gt_box2d[g]).max(1))
        return float(np.mean(vals)) if vals else 0.0

    sample = sorted(det_by_frame)[::5]
    if args.offset is None:
        scores = {o: mean_best_iou(o, sample) for o in range(-args.offset_range, args.offset_range + 1)}
        offset = max(scores, key=scores.get)
        pd.Series(scores, name="mean_best_iou").rename_axis("offset").to_csv(run / "offset_search.csv")
    else:
        offset = args.offset
    iou_at = {"best": mean_best_iou(offset, sample), "zero": mean_best_iou(0, sample)}
    print(f"[offset] annotation frame = video frame - {offset}; mean best 2D IoU {iou_at['best']:.3f} "
          f"(offset 0: {iou_at['zero']:.3f})")

    # 2-3. per-frame matching on the road plane
    thr_ft = args.match_m / i24.FT_TO_M
    matches = []
    gt_eval = 0
    for f, didx in det_by_frame.items():
        g = gt_by_frame.get(f - offset)
        if g is None:
            continue
        # annotated vehicles whose footprint center lies inside the processed road region
        gin = []
        for j in g:
            c = i24.project(cam["P"][int(gt.at[j, "direction"])], np.r_[i24.gt_footprint(gt.loc[j]).mean(0), 0][None])[0]
            if 0 <= c[0] < W and 0 <= c[1] < H and mask[int(c[1]), int(c[0])] > 0:
                gin.append(j)
        gt_eval += len(gin)
        cand = []
        per_det = {}
        for i in didx:
            pairs = ground_corner_pairs(corners[i], M, cam["vps"][2])
            bottom = np.array([corners[i][a] for a, _ in pairs])
            top = np.array([corners[i][b] for _, b in pairs])
            per_det[i] = (bottom, top)
            for j in gin:
                d = int(gt.at[j, "direction"])
                fp = i24.to_road(Hinv[d], bottom)
                dist = np.linalg.norm(fp.mean(0) - i24.gt_footprint(gt.loc[j]).mean(0))
                if dist < thr_ft:
                    cand.append((dist, i, j))
        used_d, used_g = set(), set()
        for dist, i, j in sorted(cand):
            if i in used_d or j in used_g:
                continue
            used_d.add(i)
            used_g.add(j)
            row = gt.loc[j]
            d = int(row["direction"])
            bottom, top = per_det[i]
            fp = i24.to_road(Hinv[d], bottom)
            gfp = i24.gt_footprint(row)
            gpt = i24.to_road(Hinv[d], det.loc[i, ["gx", "gy"]].to_numpy(float))[0]
            near = i24.near_face_center(row, cam_x[d])
            hts = [i24.height_from_edge(cam["P"][d], fp[k], top[k]) for k in range(4)]
            matches.append({
                "frame": f, "gt_frame": f - offset, "det_row": i, "track_id": int(det.at[i, "track_id"]),
                "gt_id": int(row["id"]), "direction": d, "timestamp": row["timestamp"],
                "center_err_m": dist * i24.FT_TO_M,
                "ground_point_err_m": float(np.linalg.norm(gpt - near)) * i24.FT_TO_M,
                "ground_point_err_along_m": float(gpt[0] - near[0]) * i24.FT_TO_M * d,
                "length_m": float(np.ptp(fp[:, 0])) * i24.FT_TO_M, "gt_length_m": float(row["l"]) * i24.FT_TO_M,
                "width_m": float(np.ptp(fp[:, 1])) * i24.FT_TO_M, "gt_width_m": float(row["w"]) * i24.FT_TO_M,
                "height_m": float(np.median(hts)) * i24.FT_TO_M, "gt_height_m": float(row["h"]) * i24.FT_TO_M,
                "bev_iou": bev_iou(fp, gfp),
                "pred_x_ft": gpt[0], "gt_x_ft": near[0],
            })
    mt = pd.DataFrame(matches)
    mt.to_csv(run / "matches.csv", index=False)
    n_det_eval = sum(len(v) for f, v in det_by_frame.items() if (f - offset) in gt_by_frame)

    # 4. speed per tracked vehicle
    speeds = []
    if len(mt):
        for tid, g in mt.groupby("track_id"):
            gid = g["gt_id"].mode().iloc[0]
            g = g[g["gt_id"] == gid].dropna(subset=["timestamp"]).sort_values("timestamp")
            if len(g) < args.min_frames or np.ptp(g["timestamp"]) <= 0:
                continue
            t = g["timestamp"].to_numpy() - g["timestamp"].iloc[0]
            v_pred = abs(np.polyfit(t, g["pred_x_ft"], 1)[0]) * i24.FT_S_TO_KMH
            v_gt = abs(np.polyfit(t, g["gt_x_ft"], 1)[0]) * i24.FT_S_TO_KMH
            speeds.append({"track_id": tid, "gt_id": gid, "frames": len(g), "gt_kmh": v_gt, "pred_kmh": v_pred,
                           "err_kmh": v_pred - v_gt})
    sp = pd.DataFrame(speeds)
    sp.to_csv(run / "speeds.csv", index=False)

    def stats(x):
        x = pd.Series(x).dropna()
        return {"mean": float(x.mean()), "median": float(x.median()), "p90": float(x.quantile(0.9))} if len(x) else {}

    summary = {
        "scene": scene, "camera": camera, "offset": int(offset), "mean_best_2d_iou": iou_at,
        "boxes": int(len(det)), "boxes_in_annotated_frames": int(n_det_eval), "gt_in_road_region": int(gt_eval),
        "matched": int(len(mt)), "match_threshold_m": args.match_m,
        "recall": len(mt) / gt_eval if gt_eval else None,
        "precision": len(mt) / n_det_eval if n_det_eval else None,
    }
    if len(mt):
        summary.update({
            "center_err_m": stats(mt["center_err_m"]),
            "ground_point_err_m": stats(mt["ground_point_err_m"]),
            "ground_point_err_along_m": stats(mt["ground_point_err_along_m"]),
            "length_abs_err_m": stats((mt["length_m"] - mt["gt_length_m"]).abs()),
            "width_abs_err_m": stats((mt["width_m"] - mt["gt_width_m"]).abs()),
            "height_abs_err_m": stats((mt["height_m"] - mt["gt_height_m"]).abs()),
            "bev_iou": stats(mt["bev_iou"]),
        })
    if len(sp):
        summary["speed"] = {"vehicles": int(len(sp)), "mae_kmh": float(sp["err_kmh"].abs().mean()),
                            "median_ae_kmh": float(sp["err_kmh"].abs().median()),
                            "bias_kmh": float(sp["err_kmh"].mean()), "gt_mean_kmh": float(sp["gt_kmh"].mean())}
    with open(run / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [f"# Kocur (YOLOv6-3D) on I-24 {scene} {camera}", "",
             f"- frame offset {offset} (mean best 2D IoU {iou_at['best']:.3f}; at offset 0: {iou_at['zero']:.3f})",
             f"- boxes {len(det)}, annotated vehicles in road region {gt_eval}, matched {len(mt)} "
             f"(recall {summary['recall'] or 0:.2f}, precision {summary['precision'] or 0:.2f}, within {args.match_m} m)"]
    for k in ("center_err_m", "ground_point_err_m", "ground_point_err_along_m", "length_abs_err_m",
              "width_abs_err_m", "height_abs_err_m", "bev_iou"):
        if k in summary:
            s = summary[k]
            lines.append(f"- {k}: mean {s['mean']:.2f}, median {s['median']:.2f}, p90 {s['p90']:.2f}")
    if "speed" in summary:
        s = summary["speed"]
        lines.append(f"- speed over {s['vehicles']} vehicles: MAE {s['mae_kmh']:.2f} km/h, median {s['median_ae_kmh']:.2f}, "
                     f"bias {s['bias_kmh']:+.2f} (mean GT {s['gt_mean_kmh']:.1f} km/h)")
    (run / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if args.video and len(mt):
        draw_overlays(args.video, run, det, corners, gt, gt_img, mt, offset, args.n_overlays)


def draw_overlays(video, run, det, corners, gt, gt_img, mt, offset, n):
    counts = mt.groupby("frame").size().sort_values(ascending=False)
    frames = sorted(counts.index[: max(n * 5, n)].tolist())
    frames = [frames[k] for k in np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)]
    cap = cv2.VideoCapture(video)
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        lw = max(2, img.shape[1] // 700)
        for j in gt.index[gt["frame"] == f - offset]:
            g = gt_img[j]
            if g is None:
                continue
            for a, b in GT_EDGES:
                cv2.line(img, tuple(int(v) for v in g[a]), tuple(int(v) for v in g[b]), (60, 220, 60), lw)
        for i in det.index[det["frame"] == f]:
            c = corners[i]
            for a, b in KOCUR_EDGES:
                cv2.line(img, tuple(int(v) for v in c[a]), tuple(int(v) for v in c[b]), (0, 140, 255), lw)
            cv2.circle(img, (int(det.at[i, "gx"]), int(det.at[i, "gy"])), 3 * lw, (0, 255, 255), -1)
        cv2.imwrite(str(run / f"overlay_{f:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()


if __name__ == "__main__":
    main()
