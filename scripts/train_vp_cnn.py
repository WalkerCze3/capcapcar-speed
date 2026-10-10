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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speed_lstm import autocalib, brno  # noqa: E402
from speed_lstm.vp_cnn import (CROP_PAD, CROP_SIZE, VPNet, angle_deg, direction_loss, extract_crops,  # noqa: E402
                               flip_dir, sample_crop_boxes, to_tensor, vp_to_crop_dir)


def load_recording(name: str, args) -> dict:
    """Crops + crop-coordinate VP labels for one recording, cached under <out>/crops/<name>.npz."""
    cache = Path(args.out) / "crops" / f"{name}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from brno_eval_cli import mask_filter

    session = Path(args.dataset_root) / "dataset" / name
    prepared = Path(args.prepared_root) / name
    calib, _ = brno.load_system(Path(args.dataset_root) / "results" / name / args.calib_name)
    summary = json.loads((prepared / "summary.json").read_text()) if (prepared / "summary.json").exists() else {}
    step = int(summary.get("frame_step", args.frame_step))
    dets = mask_filter(pd.read_csv(prepared / "detections.csv"), session / "video_mask.png")
    boxes = sample_crop_boxes(dets, (brno.WIDTH, brno.HEIGHT), every=args.every, max_crops=args.max_crops)
    crops, geoms, _ = extract_crops(session / "video.avi", boxes, step, CROP_SIZE, CROP_PAD)
    vp1, vp2 = (np.append(np.asarray(calib[k], dtype=np.float64), 1.0) for k in ("vp1", "vp2"))
    labels = np.array([[vp_to_crop_dir(vp1, g), vp_to_crop_dir(vp2, g)] for g in geoms]).reshape(-1, 2, 3)
    geo = np.array([[g.cx, g.cy, g.side] for g in geoms]).reshape(-1, 3)
    data = {"crops": crops, "labels": labels.astype(np.float32), "geoms": geo,
            "vp1": vp1, "vp2": vp2, "pp": np.asarray(calib["pp"], dtype=np.float64)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    print(f"[data] {name}: {len(crops)} crops (frame step {step})")
    return data


def augment(x: torch.Tensor, y: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random horizontal flip (labels mirrored) and brightness / contrast jitter."""
    flip = torch.from_numpy(rng.random(len(x)) < 0.5)
    x = x.clone()
    y = y.clone()
    x[flip] = torch.flip(x[flip], dims=[3])
    y[flip] = torch.from_numpy(flip_dir(y[flip].numpy())).float()
    gain = torch.from_numpy(rng.uniform(0.7, 1.3, (len(x), 1, 1, 1))).float()
    bias = torch.from_numpy(rng.uniform(-0.1, 0.1, (len(x), 1, 1, 1))).float()
    return (x * gain + bias).clamp(0.0, 1.0), y


@torch.no_grad()
def evaluate(model, recs: dict[str, dict], device) -> dict:
    from speed_lstm.vp_cnn import CropGeom, crop_dir_to_vp

    model.eval()
    out = {}
    for name, r in recs.items():
        preds = []
        for i in range(0, len(r["crops"]), 256):
            preds.append(model(to_tensor(r["crops"][i:i + 256]).to(device)).cpu().numpy())
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
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--frame-step", type=int, default=2, help="Used when a recording has no summary.json")
    p.add_argument("--every", type=int, default=5, help="Crop every n-th detection of a track")
    p.add_argument("--max-crops", type=int, default=3000, help="Per recording")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=32)
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

    train = [load_recording(n, args) for n in args.train]
    val = {n: load_recording(n, args) for n in args.val}
    X = np.concatenate([r["crops"] for r in train])
    Y = torch.from_numpy(np.concatenate([r["labels"] for r in train])).float()
    print(f"[data] {len(X)} training crops from {len(train)} recordings; validating on {list(val)}")

    model = VPNet(args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(X))
        losses = []
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            x, y = augment(to_tensor(X[idx]), Y[idx], rng)
            loss = direction_loss(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        metrics = evaluate(model, val, device)
        score = float(np.mean([m["agg_vp1_deg"] + m["agg_vp2_deg"] for m in metrics.values()]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics, "score": score})
        print(f"[epoch {epoch}] loss {np.mean(losses):.4f}  val aggregate VP1+VP2 {score:.3f} deg  "
              + "  ".join(f"{n}: crop {m['crop_vp1_deg']:.1f}/{m['crop_vp2_deg']:.1f} "
                          f"agg {m['agg_vp1_deg']:.2f}/{m['agg_vp2_deg']:.2f}" for n, m in metrics.items()))
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "width": args.width, "crop_size": CROP_SIZE,
                        "crop_pad": CROP_PAD, "epoch": epoch, "val": metrics, "train": args.train},
                       out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[done] best val aggregate {best:.3f} deg; checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    main()
