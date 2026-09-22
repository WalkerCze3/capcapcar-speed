"""Fabricate a small synthetic I-24-shaped dataset directory for tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEET_TO_METERS = 0.3048

# w = 100 (constant, always positive) -> image_x = X, image_y = Y. Not
# physically meaningful, but numerically well-behaved (never crosses the
# horizon), which is what most tests want. test_data.py builds its own
# deliberately horizon-crossing P separately.
SAFE_P = [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 100]]


def make_scene(data_dir: Path, scene: str, vehicles: list[dict], cameras: list[str] = ("p1c1",),
                fps_dt: float = 0.033, P_by_camera: dict | None = None) -> None:
    """
    vehicles: list of dicts, each: {id, camera, direction, speed_mps, n_frames,
              start_frame, x0_ft, y_ft, length_ft, width_ft, height_ft}.
    Writes obj/<scene>_annotations.csv, ts/<scene>_ts.csv, hg/<scene>_hg.json.
    """
    (data_dir / "obj").mkdir(parents=True, exist_ok=True)
    (data_dir / "ts").mkdir(parents=True, exist_ok=True)
    (data_dir / "hg").mkdir(parents=True, exist_ok=True)

    rows = []
    max_frame = 0
    for v in vehicles:
        n = v["n_frames"]
        start = v.get("start_frame", 0)
        max_frame = max(max_frame, start + n - 1)
        speed_ft_s = v["speed_mps"] / FEET_TO_METERS
        for k in range(n):
            frame = start + k
            x = v["x0_ft"] + v["direction"] * speed_ft_s * fps_dt * k
            rows.append({
                "frame": frame, "camera": v["camera"], "id": v["id"],
                "x": x, "y": v["y_ft"],
                "l": v["length_ft"], "w": v["width_ft"], "h": v["height_ft"],
                "direction": v["direction"], "class": "sedan", "gen": "manual",
            })
    pd.DataFrame(rows).to_csv(data_dir / "obj" / f"{scene}_annotations.csv", index=False)

    ts_data = {"frame": list(range(max_frame + 1))}
    for cam in cameras:
        offset = hash(cam) % 7 * 0.0001  # small per-camera clock offset
        ts_data[cam] = [1_700_000_000.0 + offset + i * fps_dt for i in range(max_frame + 1)]
    pd.DataFrame(ts_data).to_csv(data_dir / "ts" / f"{scene}_ts.csv", index=False)

    hg = {}
    for cam in cameras:
        P = (P_by_camera or {}).get(cam, SAFE_P)
        hg[cam] = {"P": P}
    with open(data_dir / "hg" / f"{scene}_hg.json", "w") as f:
        json.dump(hg, f)


def default_vehicle(id_, camera="p1c1", direction=1, speed_mps=20.0, n_frames=32,
                     start_frame=0, x0_ft=0.0, y_ft=10.0, length_ft=15.0, width_ft=5.7, height_ft=4.5) -> dict:
    return dict(id=id_, camera=camera, direction=direction, speed_mps=speed_mps, n_frames=n_frames,
                start_frame=start_frame, x0_ft=x0_ft, y_ft=y_ft,
                length_ft=length_ft, width_ft=width_ft, height_ft=height_ft)
