#!/usr/bin/env python3
"""
One Colab job for automatic calibration: train the VP CNN, keep the best model on validation, then
calibrate each evaluation recording from its own traffic and score its speeds with that calibration.

    python scripts/vp_cnn_job.py --project "$PROJECT" --brno "$BRNO" --name vp_cnn_v1 \
        --train session0_left session0_center ... session2_right \
        --val session3_left session3_center session3_right --archs resnet18 small

Outputs in <project>/runs/vp_cnn/<name>/:
  <arch>/best.pt, history.json     one training run per --archs entry (scripts/train_vp_cnn.py)
  best.pt, winner.json             the arch with the lowest validation aggregate VP error
  calib/<rec>.json                 automatic calibration (scripts/autocalib_cli.py --compare)
  eval/<rec>/, eval_ref/<rec>/     scripts/brno_eval_cli.py with that calibration / the dataset's, same detections
  results_<tag>.json / .md         per recording: calibration errors against the dataset's calibration,
                                   and speed errors with the automatic vs the dataset's calibration
Evaluation reuses each recording's cached detections (<project>/runs/brno/<rec>), so no YOLO runs.
--skip-eval only trains; --eval-only skips training and evaluates <name>/best.pt on --eval (e.g. the
test sessions with --tag test, once the setup is chosen). experiments/queue.json runs it as three
jobs: train, evaluate on validation, and (on hold) evaluate on the test sessions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from speed_lstm import autocalib, brno  # noqa: E402
from train_vp_cnn import local_video  # noqa: E402

PY = sys.executable


def run(cmd: list) -> None:
    """Run a script; on failure raise with its last stderr line (the exception), which goes in the results."""
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=REPO, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(p.stderr)
    sys.stderr.flush()
    if p.returncode:
        last = (p.stderr.strip().splitlines() or [""])[-1]
        raise RuntimeError(f"{Path(cmd[1]).name} exited {p.returncode}: {last}")


def distance_error_pct(gt: dict, calib: dict) -> float:
    """Mean |error| in % over the recording's measured along-road distances."""
    d = brno.distance_check(gt, calib)
    d = d[d["toVP1"]]
    return float(((d["measured_m"] - d["true_m"]).abs() / d["true_m"] * 100).mean())


def speed_errors(summary: dict) -> dict:
    return {k: summary.get(k, {}).get("mean") for k in ("geometry_full", "geometry_median", "model")}


def train(args, out: Path) -> None:
    scores, failed = {}, {}
    for arch in args.archs:
        try:
            run([PY, "scripts/train_vp_cnn.py", "--dataset-root", args.brno, "--prepared-root", args.prepared,
                 "--calib-name", args.calib_name, "--train", *args.train, "--val", *args.val, "--arch", arch,
                 "--epochs", args.epochs, "--cache-dir", args.crop_cache, "--video-cache", args.video_cache,
                 "--out", out / arch])
        except RuntimeError as e:  # train the other archs anyway
            print(f"[train] {arch} failed: {e}", flush=True)
            failed[arch] = str(e)
            continue
        history = json.loads((out / arch / "history.json").read_text())
        best = min(history, key=lambda h: h["score"])
        scores[arch] = {"score_deg": best["score"], "epoch": best["epoch"], "val": best["val"]}
    if not scores:
        raise SystemExit(f"Every arch failed: {failed}")
    winner = min(scores, key=lambda a: scores[a]["score_deg"])
    shutil.copy(out / winner / "best.pt", out / "best.pt")
    (out / "winner.json").write_text(json.dumps({"winner": winner, "runs": scores, "failed": failed}, indent=2))
    print(f"[train] winner {winner}: " + ", ".join(f"{a} {s['score_deg']:.3f} deg" for a, s in scores.items()), flush=True)


def evaluate(args, out: Path, rec: str) -> dict:
    brno_root, prepared = Path(args.brno), Path(args.prepared) / rec
    session = brno_root / "dataset" / rec
    ref_path = brno_root / "results" / rec / args.calib_name
    step = int(json.loads((prepared / "summary.json").read_text())["frame_step"])
    video, _ = local_video(session / "video.avi", args.video_cache)
    calib_path = out / "calib" / f"{rec}.json"
    score = ["--checkpoint", args.checkpoint, "--detections", prepared / "detections.csv", "--frame-step", step,
             "--max-seconds", 1e9, "--video", video]
    try:
        run([PY, "scripts/autocalib_cli.py", "--video", video, "--vp-model", out / "best.pt",
             "--detections", prepared / "detections.csv", "--frame-step", step,
             "--mask", session / "video_mask.png", "--compare", ref_path, "--out", calib_path])
        run([PY, "scripts/brno_eval_cli.py", "--session-dir", session, "--calib", calib_path, *score,
             "--out-dir", out / "eval" / rec])
        # The dataset's calibration through the same code and detections, so the comparison is like for like.
        run([PY, "scripts/brno_eval_cli.py", "--session-dir", session, "--calib", ref_path, *score,
             "--out-dir", out / "eval_ref" / rec])
    finally:
        if video.parent == Path(args.video_cache):
            video.unlink(missing_ok=True)
    auto = json.loads(calib_path.read_text())
    summaries = {k: json.loads((out / d / rec / "summary.json").read_text()) for k, d in (("auto", "eval"), ("manual", "eval_ref"))}
    gt = brno.load_gt(session / "gt_data.pkl")
    ref_calib, _ = brno.load_system(ref_path)
    auto_calib, _ = brno.load_system(calib_path)
    pp = np.asarray(auto_calib["pp"], dtype=np.float64)
    a, b = autocalib.to_direction(np.stack([np.append(auto_calib["vp1"], 1.0), brno.annotated_vp1(gt)]), pp,
                                  2.0 * float(np.hypot(*pp)))
    return {
        "recording": rec, "reliable": auto.get("reliable"), "compare": auto.get("compare", {}),
        "vp1_vs_lanes_deg": float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0)))),
        "quality": auto.get("quality", {}),
        "distance_err_pct": {"auto": distance_error_pct(gt, auto_calib), "manual": distance_error_pct(gt, ref_calib)},
        "mean_err_kmh": {k: speed_errors(v) for k, v in summaries.items()},
        "matched": {k: v.get("matched") for k, v in summaries.items()},
    }


def fmt(v, spec=".2f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def write_results(out: Path, tag: str, rows: list[dict], failed: dict) -> None:
    (out / f"results_{tag}.json").write_text(json.dumps({"recordings": rows, "failed": failed}, indent=2))
    lines = ["Errors are against the dataset's calibration (system file) unless noted; 'manual' rows score that "
             "calibration with the same code and detections.", "",
             "| recording | reliable | VP1 err vs lanes / vs calib (deg) | VP2 err (deg) | focal err (%) "
             "| camera height err (%) | distance err auto / manual (%) | geometric speed err auto / manual (km/h) "
             "| LSTM speed err auto / manual (km/h) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, d, e = r["compare"], r["distance_err_pct"], r["mean_err_kmh"]
        lines.append(f"| {r['recording']} | {r['reliable']} | {fmt(r['vp1_vs_lanes_deg'])} / {fmt(c.get('vp1_angle_deg'))} "
                     f"| {fmt(c.get('vp2_angle_deg'))} "
                     f"| {fmt(c.get('focal_err_pct'), '+.1f')} | {fmt(c.get('camera_height_err_pct'), '+.1f')} "
                     f"| {fmt(d['auto'])} / {fmt(d['manual'])} "
                     f"| {fmt(e['auto']['geometry_full'])} / {fmt(e['manual']['geometry_full'])} "
                     f"| {fmt(e['auto']['model'])} / {fmt(e['manual']['model'])} |")
    for rec, err in failed.items():
        lines.append(f"| {rec} | failed: {err} | | | | | | | |")
    (out / f"results_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--project", required=True, help="Drive project folder (has runs/brno, runs/v2/3d/best.pt)")
    p.add_argument("--brno", required=True, help="2016-ITS-BrnoCompSpeed (has dataset/ and results/)")
    p.add_argument("--name", required=True, help="Output folder name under <project>/runs/vp_cnn")
    p.add_argument("--train", nargs="+", default=[])
    p.add_argument("--val", nargs="+", default=[])
    p.add_argument("--eval", nargs="+", default=None, help="Recordings to calibrate and score (default: --val)")
    p.add_argument("--archs", nargs="+", default=["resnet18", "small"], choices=["resnet18", "small"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-only", action="store_true", help="Use the existing <name>/best.pt")
    p.add_argument("--skip-eval", action="store_true", help="Only train (evaluate in a later job)")
    p.add_argument("--tag", default="val", help="Results file suffix: results_<tag>.json / .md")
    p.add_argument("--calib-name", default="system_dubska_optimal_calib.json")
    p.add_argument("--prepared", default=None, help="Default <project>/runs/brno")
    p.add_argument("--checkpoint", default=None, help="Speed LSTM for brno_eval_cli (default <project>/runs/v2/3d/best.pt)")
    p.add_argument("--crop-cache", default=None, help="Default <project>/runs/vp_cnn/crops")
    p.add_argument("--video-cache", default="/content/video_cache", help="Local disk for video copies")
    args = p.parse_args()

    project = Path(args.project)
    args.prepared = args.prepared or str(project / "runs/brno")
    args.checkpoint = args.checkpoint or str(project / "runs/v2/3d/best.pt")
    args.crop_cache = args.crop_cache or str(project / "runs/vp_cnn/crops")
    out = project / "runs/vp_cnn" / args.name
    out.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        if not args.train or not args.val:
            raise SystemExit("--train and --val are needed unless --eval-only")
        train(args, out)
    elif not (out / "best.pt").exists():
        raise SystemExit(f"--eval-only but no model at {out / 'best.pt'}")
    if args.skip_eval:
        return

    rows, failed = [], {}
    for rec in args.eval or args.val:
        try:
            rows.append(evaluate(args, out, rec))
        except Exception as e:  # keep scoring the other recordings; report this one
            print(f"[eval] {rec} failed: {e!r}", flush=True)
            failed[rec] = str(e)[:300]
    write_results(out, args.tag, rows, failed)
    if not rows:
        raise SystemExit("Evaluation failed on every recording")


if __name__ == "__main__":
    main()
