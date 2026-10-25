#!/usr/bin/env python3
"""
Kocur 3D boxes -> boxes3d.csv in the format lift3d produces, on our own tracks.

To compare only the 3D-box step, the tracks stay those of our YOLO + ByteTrack
run (runs/brno/<recording>/detections.csv): in each frame, each tracked box
takes the Kocur box whose image extent overlaps it most (IoU >= --min-iou).
Frames without a match are dropped, as lift3d drops frames it can't fit.

Each Kocur box becomes the model's 3D input with the same calibration P as the
lift3d run: its 4 ground corners (the bottom ends of its vertical edges) are
mapped onto the road plane, length / width are their extent along / across the
road, height is the median height of the 4 vertical edges, and the center is
the footprint center at z = height / 2. Post-processing matches lift_tracks:
dims fixed to the track median (--dims frame keeps per-frame dims) and centers
smoothed with the same moving average. boxes2d is the projected cuboid's
min/max box, as at training time.

Example:
    python baselines/kocur_brno/kocur_boxes3d.py --run-dir runs/brno/session4_center \
        --kocur-dir runs/kocur_brno/session4_center --session-dir .../dataset/session4_center \
        --out runs/brno_kocur/session4_center/boxes3d.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "kocur_i24"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import i24  # noqa: E402
from kocur import ground_corner_pairs  # noqa: E402
from speed_lstm.data import project_to_bbox  # noqa: E402
from speed_lstm.video import BOX2D_COLS, BOX3D_COLS, _drop_truncated, _smooth  # noqa: E402

CORNER_COLS = [f"c{i}{a}" for i in range(8) for a in "xy"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every xyxy box in a against every xyxy box in b."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def box_to_3d(corners: np.ndarray, P: np.ndarray, Hinv: np.ndarray, M: np.ndarray, vp3: np.ndarray):
    """(center (3,), dims (3,)) in the road frame of P (x along the road, z up), from 8 image corners."""
    pairs = ground_corner_pairs(corners, M, vp3)
    bottom = np.array([corners[a] for a, _ in pairs])
    top = np.array([corners[b] for _, b in pairs])
    fp = i24.to_road(Hinv, bottom)
    if not np.all(np.isfinite(fp)):
        return None
    height = float(np.median([i24.height_from_edge(P, fp[k], top[k]) for k in range(4)]))
    dims = np.array([np.ptp(fp[:, 0]), np.ptp(fp[:, 1]), height])
    center = np.array([*fp.mean(0), height / 2.0])
    return center, dims


def match_frame(det_boxes: np.ndarray, k_boxes: np.ndarray, min_iou: float) -> list[tuple[int, int, float]]:
    """(det index, kocur index, IoU): each tracked box takes its best-overlapping Kocur box, if IoU >= min_iou.
    Not one-to-one, so the duplicate tracks YOLO sometimes makes for one car (car + truck) both keep a box,
    as both get a lift3d box."""
    if not len(det_boxes) or not len(k_boxes):
        return []
    iou = iou_matrix(det_boxes, k_boxes)
    best = iou.argmax(axis=1)
    return [(i, int(j), float(iou[i, j])) for i, j in enumerate(best) if iou[i, j] >= min_iou]


def kocur_lifted(dets: pd.DataFrame, kdet: pd.DataFrame, P: np.ndarray, M: np.ndarray, timestamps: np.ndarray,
                 min_iou: float = 0.5, smooth_window: int = 5, dims_mode: str = "median") -> pd.DataFrame:
    """Rows like video.lift_tracks: one per (track_id, frame) whose detection got a Kocur box."""
    Hinv = i24.ground_homography_inv(P)
    vp3 = P[:2, 2] / P[2, 2]  # image of the vertical direction
    k_by_frame = {f: g for f, g in kdet.groupby("frame")}
    rows = []
    for f, g in dets.groupby("frame"):
        kg = k_by_frame.get(f)
        if kg is None:
            continue
        kc = kg[CORNER_COLS].to_numpy(np.float64).reshape(-1, 8, 2)
        k_boxes = np.column_stack([kc[:, :, 0].min(1), kc[:, :, 1].min(1), kc[:, :, 0].max(1), kc[:, :, 1].max(1)])
        d_boxes = g[BOX2D_COLS].to_numpy(np.float64)
        for i, j, v in match_frame(d_boxes, k_boxes, min_iou):
            res = box_to_3d(kc[j], P, Hinv, M, vp3)
            if res is None:
                continue
            r = g.iloc[i]
            rows.append({"track_id": int(r["track_id"]), "frame": int(f), "cls": r["cls"], "iou": v,
                         "center": res[0], "dims": res[1], "det_box": d_boxes[i]})

    out = []
    by_track = {}
    for r in rows:
        by_track.setdefault(r["track_id"], []).append(r)
    for tid, rs in sorted(by_track.items()):
        rs.sort(key=lambda r: r["frame"])
        frames = np.array([r["frame"] for r in rs])
        centers = np.stack([r["center"] for r in rs])
        dims = np.stack([r["dims"] for r in rs])
        if dims_mode == "median":
            dims = np.repeat(np.median(dims, axis=0)[None], len(rs), axis=0)
            centers[:, 2] = dims[:, 2] / 2.0
        centers = _smooth(centers, frames, smooth_window)
        d = 1 if centers[-1, 0] >= centers[0, 0] else -1  # same P both ways on Brno: direction = motion
        cls = pd.Series([r["cls"] for r in rs]).mode().iloc[0]
        for r, f, c, dm in zip(rs, frames, centers, dims):
            box = project_to_bbox(c, dm, P)
            if box is None:
                continue
            rms = float(np.sqrt(np.mean((box.astype(np.float64) - r["det_box"]) ** 2)))
            out.append({"track_id": tid, "frame": int(f), "timestamp": timestamps[f], "cls": cls, "direction": d,
                        **dict(zip(BOX3D_COLS, [*c, *dm])), **dict(zip(BOX2D_COLS, box.astype(np.float64))),
                        "residual_px": rms, "match_iou": r["iou"]})
    return pd.DataFrame(out, columns=["track_id", "frame", "timestamp", "cls", "direction", *BOX3D_COLS,
                                      *BOX2D_COLS, "residual_px", "match_iou"])


def main():
    from brno_eval_cli import mask_filter
    from speed_lstm import brno

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="runs/brno/<recording> from brno_eval_cli (lift3d run)")
    ap.add_argument("--kocur-dir", required=True, help="output of run_kocur_brno.py for the same recording")
    ap.add_argument("--session-dir", required=True, help="dataset/<recording> (gt_data.pkl, video_mask.png)")
    ap.add_argument("--out", required=True, help="boxes3d.csv to write")
    ap.add_argument("--calib", default=None, help="default: the calib path in <run-dir>/run_config.json")
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--dims", choices=["median", "frame"], default="median",
                    help="median: one size per track, as lift3d does; frame: Kocur's per-frame size")
    args = ap.parse_args()

    run, kdir, session = Path(args.run_dir), Path(args.kocur_dir), Path(args.session_dir)
    cfg = json.loads((run / "run_config.json").read_text())
    calib_path = args.calib or cfg["calib"]
    calib, _ = brno.load_system(calib_path)
    P = brno.projection_from_calibration(calib["vp1"], calib["vp2"], calib["pp"], calib["scale"])
    meta = json.loads((kdir / "meta.json").read_text())
    if Path(meta["calib"]).name != Path(cfg["calib"]).name or Path(calib_path).name != Path(cfg["calib"]).name:
        print(f"[warn] calibrations differ: lift3d run {Path(cfg['calib']).name}, Kocur run "
              f"{Path(meta['calib']).name}, this conversion {Path(calib_path).name}")

    gt = brno.load_gt(session / "gt_data.pkl")
    fps = float(cfg.get("video_fps") or gt["fps"])
    step = meta["frame_step"]
    dets = pd.read_csv(run / "detections.csv")
    dets = dets[dets["frame"] < meta["frames_processed"]]
    if not cfg.get("no_mask"):
        dets = mask_filter(dets, session / "video_mask.png")
    dets = _drop_truncated(dets, brno.WIDTH, brno.HEIGHT, 3.0)  # same as lift_tracks
    n_samples = int(dets["frame"].max()) + 1 if len(dets) else 0
    timestamps = np.arange(n_samples) * step / fps

    kdet = pd.read_csv(kdir / "detections_kocur.csv")
    lifted = kocur_lifted(dets, kdet, P, np.asarray(meta["M"]), timestamps, args.min_iou, int(cfg.get("smooth", 5)),
                          args.dims)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lifted.to_csv(args.out, index=False)
    n_det = len(dets.drop_duplicates(["track_id", "frame"]))
    print(f"[kocur3d] {len(lifted)}/{n_det} tracked boxes got a Kocur box ({len(lifted) / max(n_det, 1):.0%}), "
          f"{lifted['track_id'].nunique()} tracks; median dims "
          f"{lifted[['length', 'width', 'height']].median().round(2).tolist()} m -> {args.out}")


if __name__ == "__main__":
    main()
