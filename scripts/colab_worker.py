#!/usr/bin/env python3
"""
Colab worker: runs the jobs listed in experiments/queue.json on this branch,
one at a time, and keeps pulling the branch so new jobs (and code) are picked
up without touching the notebook.

Per job, <project>/runs/worker/<job id>/ gets log.txt and DONE or FAILED; a
job with either marker is never run again (give it a new id to rerun).
<project>/runs/worker/heartbeat.json says what the worker is doing.

Job types:
  {"id": ..., "type": "prepare", "recording": "session1_center", "max_seconds": 600,
   "reuse_detections": true}                     -> brno_eval_cli.py into <project>/runs/brno/<recording>
  {"id": ..., "type": "experiment", "config": {...}}  -> brno_experiment.py into <project>/runs/experiments/<id>
  {"id": ..., "type": "shell", "cmd": "..."}          -> anything else (run from the repo root)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


def run_logged(cmd: list[str], log_path: Path) -> int:
    with open(log_path, "a") as log:
        log.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return p.wait()


def job_cmds(job: dict, args) -> list[list[str]]:
    py = sys.executable
    if job["type"] == "prepare":
        rec = job["recording"]
        local = Path("/content/runs/brno") / rec
        drive_dir = Path(args.project) / "runs/brno" / rec
        cmd = [py, "scripts/brno_eval_cli.py", "--session-dir", f"{args.brno}/dataset/{rec}",
               "--calib", f"{args.brno}/results/{rec}/system_{job.get('calib', 'dubska_optimal_calib')}.json",
               "--checkpoint", str(Path(args.project) / job.get("checkpoint", "runs/v2/3d/best.pt")),
               "--weights", job.get("weights", "yolo11m.pt"), "--max-seconds", str(job.get("max_seconds", 600)),
               "--out-dir", str(local)]
        cached = drive_dir / "detections.csv"
        if job.get("reuse_detections", True) and cached.exists():
            shutil.copy(cached, "/content/detections_cache.csv")
            cmd += ["--detections", "/content/detections_cache.csv"]
        else:
            # Reading a multi-GB AVI straight off the Drive mount can time out in OpenCV; copy it first.
            cache = Path("/content/video_cache") / f"{rec}.avi"
            cmd += ["--video", str(cache)]
        cmd += job.get("extra_args", [])
        steps = [cmd, [py, "-c", f"import shutil; shutil.copytree({str(local)!r}, {str(drive_dir)!r}, dirs_exist_ok=True)"]]
        if "--video" in cmd:
            copy = [py, "-c", "import shutil, pathlib, sys; pathlib.Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True); "
                              "shutil.copy(sys.argv[1], sys.argv[2]); print('[worker] copied video to', sys.argv[2])",
                    f"{args.brno}/dataset/{rec}/video.avi", str(cache)]
            clean = [py, "-c", "import os, sys; os.remove(sys.argv[1])", str(cache)]
            steps = [copy, steps[0], clean, steps[1]]
        return steps
    if job["type"] == "experiment":
        out = Path(args.project) / "runs/experiments" / job["id"]
        return [[py, "scripts/brno_experiment.py", "--config", json.dumps(job["config"]),
                 "--project", args.project, "--out-dir", str(out)]]
    if job["type"] == "shell":
        return [["bash", "-c", job["cmd"]]]
    raise ValueError(f"unknown job type {job['type']!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--brno", required=True)
    p.add_argument("--branch", default="video-3d-bbox")
    p.add_argument("--queue", default="experiments/queue.json")
    p.add_argument("--poll", type=int, default=60)
    args = p.parse_args()

    state = Path(args.project) / "runs/worker"
    state.mkdir(parents=True, exist_ok=True)

    def beat(**kw):
        (state / "heartbeat.json").write_text(json.dumps({"time": now(), "commit": git("rev-parse", "--short", "HEAD"),
                                                          **kw}, indent=2))

    while True:
        before = git("rev-parse", "HEAD")
        try:
            git("fetch", "-q", "origin", args.branch)
            git("reset", "-q", "--hard", f"origin/{args.branch}")
        except subprocess.CalledProcessError as e:
            print(f"[worker] git update failed: {e}", flush=True)
        if git("rev-parse", "HEAD") != before:
            print(f"[worker] {now()} code updated to {git('rev-parse', '--short', 'HEAD')}; restarting", flush=True)
            os.execv(sys.executable, [sys.executable, *sys.argv])

        jobs = json.loads((REPO / args.queue).read_text())
        pending = [j for j in jobs if not any((state / j["id"] / m).exists() for m in ("DONE", "FAILED"))]
        if not pending:
            beat(status="idle", done=len(jobs))
            time.sleep(args.poll)
            continue

        job = pending[0]
        jdir = state / job["id"]
        jdir.mkdir(parents=True, exist_ok=True)
        beat(status="running", job=job["id"], started=now(), pending=len(pending))
        print(f"\n[worker] {now()} ===== {job['id']} ({job['type']}), {len(pending) - 1} more queued =====", flush=True)
        t0 = time.time()
        rc = 0
        for cmd in job_cmds(job, args):
            rc = run_logged(cmd, jdir / "log.txt")
            if rc:
                break
        marker = "DONE" if rc == 0 else "FAILED"
        (jdir / marker).write_text(json.dumps({"time": now(), "seconds": round(time.time() - t0), "returncode": rc}))
        print(f"[worker] {now()} {job['id']}: {marker} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
