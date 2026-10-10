#!/usr/bin/env python3
"""
Calibrate a camera automatically from its traffic: video -> calibration json, no manual input.

    python scripts/autocalib_cli.py --video session4_center/video.avi --vp-model runs/vp_cnn/best.pt \
        --detections runs/brno/session4_center/detections.csv --frame-step 2 \
        --out runs/autocalib/session4_center.json \
        --compare runs/.../results/session4_center/system_dubska_optimal_calib.json

Detections come from --detections (a detect_and_track csv, with the --frame-step it was made with)
or are made here with YOLO + ByteTrack. The output is a BrnoCompSpeed-style system file
({"camera_calibration": {vp1, vp2, pp, scale}} + "P" in metres): pass it as --calib to
scripts/brno_eval_cli.py to score speeds with it, or to video_speed_cli.py with --calib-units m.
--compare prints how far it is from a reference calibration (VP angles, focal length, scale).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm import autocalib, brno  # noqa: E402
from speed_lstm.video import detect_and_track, video_info  # noqa: E402


def compare(cal: autocalib.AutoCalibration, ref_path: str) -> dict:
    """Errors of `cal` against a reference calibration file (BrnoCompSpeed results format)."""
    ref, _ = brno.load_system(ref_path)
    f0 = float(np.hypot(*cal.pp)) * 2.0
    out = {}
    for k in ("vp1", "vp2"):
        a = autocalib.to_direction(np.append(getattr(cal, k), 1.0), cal.pp, f0)[0]
        b = autocalib.to_direction(np.append(ref[k], 1.0), cal.pp, f0)[0]
        out[f"{k}_angle_deg"] = float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))
    _, ref_focal = brno.compute_camera_calibration(ref["vp1"], ref["vp2"], ref["pp"])
    out["focal_err_pct"] = float((cal.focal - ref_focal) / ref_focal * 100)
    # Compare metric scale through camera height, which doesn't depend on the calibrations' own units.
    h_ref = ref["scale"] * autocalib.camera_height_unit(ref["vp1"], ref["vp2"], ref["pp"])
    out["camera_height_err_pct"] = float((cal.quality["camera_height_m"] - h_ref) / h_ref * 100)
    return out


def main() -> None:
    from speed_lstm.vp_cnn import VPPredictor, extract_crops, sample_crop_boxes

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--video", required=True)
    p.add_argument("--vp-model", required=True, help="checkpoint from scripts/train_vp_cnn.py")
    p.add_argument("--detections", default=None, help="detect_and_track csv made with the same --frame-step")
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None, help="Processed frames to use (default: all)")
    p.add_argument("--weights", default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--mask", default=None, help="Road mask png: drop detections whose bottom-center is outside it")
    p.add_argument("--max-crops", type=int, default=2000)
    p.add_argument("--compare", default=None, help="Reference calibration json to report errors against")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _, _, img_w, img_h = video_info(args.video)

    if args.detections:
        dets = pd.read_csv(args.detections)
    else:
        dets = detect_and_track(args.video, args.weights, conf=args.conf, device=args.device,
                                max_frames=args.max_frames, frame_step=args.frame_step)
    if args.max_frames is not None:
        dets = dets[dets["frame"] < args.max_frames]
    if args.mask:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from brno_eval_cli import mask_filter
        dets = mask_filter(dets, args.mask)
    print(f"[track] {len(dets)} detections, {dets['track_id'].nunique()} tracks")

    boxes = sample_crop_boxes(dets, (img_w, img_h), max_crops=args.max_crops)
    crops, geoms, kept = extract_crops(args.video, boxes, args.frame_step)
    vp1s, vp2s = VPPredictor(args.vp_model, args.device).predict_vps(crops, geoms)
    print(f"[cnn] VPs predicted for {len(crops)} vehicle crops")

    cal = autocalib.calibrate(dets, (img_w, img_h), vp1_candidates=vp1s, vp2_candidates=vp2s)
    cal.save(out)
    q = cal.quality
    print(f"[calib] VP1 {np.round(cal.vp1, 1)} from {q['vp1_source']} ({q['vp1_lines']} track lines, "
          f"{q['vp1_inlier_frac']:.0%} inliers), VP2 {np.round(cal.vp2, 1)} (spread {q['vp2_spread_deg']:.2f} deg)")
    print(f"[calib] focal {cal.focal:.0f} px, camera height {q['camera_height_m']:.2f} m, "
          f"scale split diff {q.get('scale_split_diff', float('nan')):.1%}, residual {q['median_residual_px']:.2f} px")
    print(f"[calib] {'reliable' if cal.reliable else 'NOT reliable'}; saved {out}")

    if args.compare:
        errs = compare(cal, args.compare)
        print("[compare] " + ", ".join(f"{k} {v:.2f}" for k, v in errs.items()))
        data = json.loads(out.read_text())
        data["compare"] = {"reference": args.compare, **errs}
        out.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
