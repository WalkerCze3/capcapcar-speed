#!/usr/bin/env python3
"""
Train the vehicle vanishing-point CNN (speed_lstm.vp_cnn) on BrnoCompSpeed recordings.

Labels come for free from each recording's calibration: on its straight road every car's VP1 / VP2
are the road's. Detections are the ones brno_eval_cli.py cached (runs/brno/<recording>/detections.csv,
with that run's frame step), so no detector runs here; crops are cut from the video once and cached.

    python scripts/train_vp_cnn.py --dataset-root /data/2016-ITS-BrnoCompSpeed --prepared-root runs/brno \
        --train session0_center session1_center session2_center --val session3_center --out runs/vp_cnn

Never put the test sessions (4-6) in --train or --val: speed is scored on them later with the VPs
this model predicts. Validation reports the median per-crop angle error and the error of the
recording-level aggregate (what autocalib actually uses); the best epoch is picked on the latter.

With few training cameras the network could memorise their few VP configurations, so each training
crop is warped by a random homography with its labels mapped exactly (vp_cnn.random_homography),
on top of mirroring and brightness / contrast jitter. Crops are cached with extra context around
them so the warp has real pixels to show, and re-fitted to the warped vehicle box afterwards.

VP1 labels come from the recording's annotated along-road lines (brno.annotated_vp1) rather than its
calibration file, whose VP1 is 3-4 degrees off the road on session1_center / _right; VP2 labels come
from the calibration file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm import autocalib, brno  # noqa: E402
from speed_lstm.vp_cnn import (CONTEXT, CROP_PAD, CROP_SIZE, CropGeom, VPNet, angle_deg,  # noqa: E402
                               crop_dir_to_vp, direction_loss, extract_crops, flip_dir, random_homography,
                               refit_crop, render_crop, sample_crop_boxes, to_tensor, transform_dirs,
                               vp_to_crop_dir)

DEFAULT_LR = {"small": 1e-3, "resnet18": 3e-4}


def local_video(src: Path, cache_dir: str | None) -> tuple[Path, bool]:
    """
    (path to read, whether this call made a copy). Reading a multi-GB AVI straight off a Drive
    mount can time out in OpenCV, so with a cache dir the video is copied to local disk first.
    """
    if not cache_dir:
        return src, False
    dst = Path(cache_dir) / f"{src.parent.name}.avi"
    if dst.exists():
        return dst, False
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shutil.copy(src, dst)
    print(f"[video] copied {src} to {dst} ({dst.stat().st_size / 1e9:.1f} GB, {time.time() - t0:.0f} s)", flush=True)
    return dst, True


def encode_crops(crops: np.ndarray, quality: int = 95) -> dict:
    """JPEG-encode crops into one byte array + offsets (a tenth of the npz size, for a Drive cache)."""
    import cv2

    blobs = [cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].ravel() for c in crops]
    offsets = np.cumsum([0] + [len(b) for b in blobs])
    return {"jpeg": np.concatenate(blobs) if blobs else np.zeros(0, np.uint8), "jpeg_offsets": offsets,
            "crop_shape": np.array(crops.shape[1:])}


def decode_crops(data: dict) -> np.ndarray:
    import cv2

    buf, off = data["jpeg"], data["jpeg_offsets"]
    if len(off) <= 1:
        return np.zeros((0, *data["crop_shape"]), dtype=np.uint8)
    return np.stack([cv2.imdecode(buf[a:b], cv2.IMREAD_COLOR) for a, b in zip(off[:-1], off[1:])])


def load_recording(name: str, args) -> dict:
    """
    Context crops, plain-crop geometry, box sizes and crop-coordinate VP labels for one recording.
    Crops are cached as <cache-dir>/<name>_e<every>_n<max>_c<context>.npz; labels are recomputed on load.
    """
    session = Path(args.dataset_root) / "dataset" / name
    cache = Path(args.cache_dir or Path(args.out) / "crops") / f"{name}_e{args.every}_n{args.max_crops}_c{CONTEXT:g}.npz"
    if cache.exists():
        z = np.load(cache)
        data = {"crops": decode_crops(z), "geoms": z["geoms"], "box_hw": z["box_hw"]}
        print(f"[data] {name}: {len(data['crops'])} cached crops from {cache}", flush=True)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from brno_eval_cli import mask_filter

        prepared = Path(args.prepared_root) / name
        summary = json.loads((prepared / "summary.json").read_text()) if (prepared / "summary.json").exists() else {}
        step = int(summary.get("frame_step", args.frame_step))
        dets = mask_filter(pd.read_csv(prepared / "detections.csv"), session / "video_mask.png")
        boxes = sample_crop_boxes(dets, (brno.WIDTH, brno.HEIGHT), every=args.every, max_crops=args.max_crops)
        video, copied = local_video(session / "video.avi", args.video_cache)
        t0 = time.time()
        try:
            crops, geoms, kept = extract_crops(video, boxes, step, CROP_SIZE, CROP_PAD, context=CONTEXT)
        finally:
            if copied:
                video.unlink()
        geo = np.array([[g.cx, g.cy, g.side] for g in geoms]).reshape(-1, 3)
        box_hw = (np.stack([(kept["xmax"] - kept["xmin"]).to_numpy(), (kept["ymax"] - kept["ymin"]).to_numpy()], axis=1)
                  / geo[:, 2:3]) if len(kept) else np.zeros((0, 2))
        data = {"crops": crops, "geoms": geo, "box_hw": box_hw.astype(np.float64)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez(tmp, geoms=geo, box_hw=data["box_hw"], **encode_crops(crops))
        tmp.replace(cache)
        data["crops"] = decode_crops(np.load(cache))  # train on exactly what a cached rerun would see
        print(f"[data] {name}: {len(crops)} crops of {len(dets)} masked detections (frame step {step}, "
              f"{time.time() - t0:.0f} s) -> {cache}", flush=True)

    calib, _ = brno.load_system(Path(args.dataset_root) / "results" / name / args.calib_name)
    pp = np.asarray(calib["pp"], dtype=np.float64)
    vp1 = np.append(np.asarray(calib["vp1"], dtype=np.float64), 1.0)
    if args.vp1_label == "annotations":
        ann = brno.annotated_vp1(brno.load_gt(session / "gt_data.pkl"))
        f0 = 2.0 * float(np.hypot(*pp))
        a, b = autocalib.to_direction(np.stack([ann, vp1]), pp, f0)
        print(f"[data] {name}: VP1 label from annotated lines, {np.degrees(np.arccos(min(1.0, abs(a @ b)))):.2f} deg "
              f"from the calibration's", flush=True)
        vp1 = ann
    vp2 = np.append(np.asarray(calib["vp2"], dtype=np.float64), 1.0)
    geoms = [CropGeom(*g) for g in data["geoms"]]
    labels = np.array([[vp_to_crop_dir(vp1, g), vp_to_crop_dir(vp2, g)] for g in geoms]).reshape(-1, 2, 3)
    return {**data, "labels": labels.astype(np.float32), "vp1": vp1, "vp2": vp2, "pp": pp}


def augment(context: np.ndarray, labels: np.ndarray, box_hw: np.ndarray, rng: np.random.Generator,
            geometric: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Training crops from context crops: mirror, random homography with the crop re-fitted to the
    warped box (labels mapped exactly), brightness / contrast jitter.
    """
    x = np.empty((len(context), CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    y = labels.astype(np.float64)
    for i in range(len(context)):
        c = context[i]
        if rng.random() < 0.5:
            c = c[:, ::-1]
            y[i] = flip_dir(y[i])
        G = refit_crop(random_homography(rng), box_hw[i], rng) if geometric else np.eye(3)
        x[i] = render_crop(np.ascontiguousarray(c), G)
        y[i] = transform_dirs(y[i], G)
    t = to_tensor(x)
    gain = torch.from_numpy(rng.uniform(0.7, 1.3, (len(t), 1, 1, 1))).float()
    bias = torch.from_numpy(rng.uniform(-0.1, 0.1, (len(t), 1, 1, 1))).float()
    return (t * gain + bias).clamp(0.0, 1.0), torch.from_numpy(y).float()


@torch.no_grad()
def evaluate(model, recs: dict[str, dict], device) -> dict:
    model.eval()
    out = {}
    for name, r in recs.items():
        preds = []
        for i in range(0, len(r["plain"]), 256):
            preds.append(model(to_tensor(r["plain"][i:i + 256]).to(device)).cpu().numpy())
        d = np.concatenate(preds)
        per_crop = angle_deg(d, r["labels"])
        # Recording-level aggregate, as autocalib does it, on the Gaussian sphere around the image center.
        pp = np.array([brno.WIDTH / 2.0, brno.HEIGHT / 2.0])
        f0 = float(np.hypot(brno.WIDTH, brno.HEIGHT))
        geoms = [CropGeom(*g) for g in r["geoms"]]
        agg = {}
        for k, vp_true in ((0, r["vp1"]), (1, r["vp2"])):
            vps = np.array([crop_dir_to_vp(d[i, k], g) for i, g in enumerate(geoms)])
            est, _ = autocalib.aggregate_directions(autocalib.to_direction(vps, pp, f0))
            true = autocalib.to_direction(vp_true, pp, f0)[0]
            agg[k] = float(np.degrees(np.arccos(np.clip(abs(est @ true), 0.0, 1.0))))
        out[name] = {"crop_vp1_deg": float(np.median(per_crop[:, 0])), "crop_vp2_deg": float(np.median(per_crop[:, 1])),
                     "agg_vp1_deg": agg[0], "agg_vp2_deg": agg[1]}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-root", required=True, help="2016-ITS-BrnoCompSpeed (has dataset/ and results/)")
    p.add_argument("--prepared-root", default="runs/brno", help="brno_eval_cli.py outputs per recording")
    p.add_argument("--calib-name", default="system_dubska_optimal_calib.json", help="label source in results/<rec>/")
    p.add_argument("--vp1-label", choices=["annotations", "calib"], default="annotations",
                   help="VP1 labels from the annotated along-road lines (gt_data.pkl) or the calibration file")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--frame-step", type=int, default=2, help="Used when a recording has no summary.json")
    p.add_argument("--every", type=int, default=5, help="Crop every n-th detection of a track")
    p.add_argument("--max-crops", type=int, default=3000, help="Per recording")
    p.add_argument("--cache-dir", default=None, help="Crop cache (default <out>/crops); reused across runs")
    p.add_argument("--video-cache", default=None, help="Copy each video here before cropping (e.g. off a Drive mount)")
    p.add_argument("--arch", choices=["small", "resnet18"], default="resnet18")
    p.add_argument("--no-pretrained", action="store_true", help="resnet18 from random weights instead of ImageNet")
    p.add_argument("--no-geometric-aug", action="store_true", help="Only mirror + colour jitter")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help=f"Default per arch: {DEFAULT_LR}")
    p.add_argument("--width", type=int, default=32, help="Channels of the small arch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    test_sessions = {"session4", "session5", "session6"}
    leaked = [r for r in args.train + args.val if r.split("_")[0] in test_sessions]
    if leaked:
        raise SystemExit(f"Refusing to train/validate on test sessions: {leaked}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lr = args.lr or DEFAULT_LR[args.arch]

    train = [r for r in (load_recording(n, args) for n in args.train) if len(r["crops"])]
    val = {n: r for n, r in ((n, load_recording(n, args)) for n in args.val) if len(r["crops"])}
    if not train or not val:
        raise SystemExit("No crops to train or validate on")
    for r in val.values():
        r["plain"] = np.stack([render_crop(c) for c in r.pop("crops")])
    X = np.concatenate([r["crops"] for r in train])
    Y = np.concatenate([r["labels"] for r in train])
    B = np.concatenate([r["box_hw"] for r in train])
    print(f"[data] {len(X)} training crops from {len(train)} recordings; validating on {list(val)}", flush=True)

    model = VPNet(args.width, args.arch, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[train] {args.arch} ({'random init' if args.no_pretrained or args.arch == 'small' else 'ImageNet init'}), "
          f"lr {lr:g}, {args.epochs} epochs on {device}", flush=True)
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(X))
        losses = []
        for i in range(0, len(order), args.batch_size):
            idx = np.sort(order[i:i + args.batch_size])
            x, y = augment(X[idx], Y[idx], B[idx], rng, geometric=not args.no_geometric_aug)
            loss = direction_loss(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        metrics = evaluate(model, val, device)
        score = float(np.mean([m["agg_vp1_deg"] + m["agg_vp2_deg"] for m in metrics.values()]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics, "score": score})
        print(f"[epoch {epoch}] {time.time() - t0:.0f} s, loss {np.mean(losses):.4f}  val aggregate VP1+VP2 {score:.3f} deg  "
              + "  ".join(f"{n}: crop {m['crop_vp1_deg']:.1f}/{m['crop_vp2_deg']:.1f} "
                          f"agg {m['agg_vp1_deg']:.2f}/{m['agg_vp2_deg']:.2f}" for n, m in metrics.items()), flush=True)
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "arch": args.arch, "width": args.width, "crop_size": CROP_SIZE,
                        "crop_pad": CROP_PAD, "epoch": epoch, "val": metrics, "score": score, "train": args.train},
                       out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[done] best val aggregate {best:.3f} deg; checkpoint {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
