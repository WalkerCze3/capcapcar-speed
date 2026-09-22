"""
Summarize speed-balanced training runs and check that separately trained
models (e.g. 2d/3d/combined) were evaluated on the SAME held-out vehicles —
a prerequisite for comparing their test metrics against each other.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_manifest(run_dir: str | Path) -> dict:
    with open(Path(run_dir) / "manifest.json") as f:
        return json.load(f)


def summarize_manifest(manifest: dict) -> str:
    lines = [
        f"mode: {manifest['mode']}  seed: {manifest['seed']}  scenes: {manifest['scenes']}",
        f"  train groups: {len(manifest['train_groups'])}",
        f"  val groups:   {len(manifest['val_groups'])}",
        f"  test groups:  {len(manifest['test_groups'])}",
        f"  test windows: {len(manifest['test_window_keys'])}",
    ]
    return "\n".join(lines)


def verify_consistent_test_sets(run_dirs: dict[str, str | Path]) -> dict:
    """
    run_dirs: {model_name: run_dir}. Returns a report of whether every
    model's test split covers the exact same (scene, vehicle) groups and
    the exact same windows — comparing their test metrics only makes sense
    if they do.
    """
    manifests = {name: load_manifest(d) for name, d in run_dirs.items()}
    group_sets = {name: set(m["test_groups"]) for name, m in manifests.items()}
    window_sets = {name: set(m["test_window_keys"]) for name, m in manifests.items()}

    names = list(manifests.keys())
    reference = names[0]
    groups_match = all(group_sets[n] == group_sets[reference] for n in names)
    windows_match = all(window_sets[n] == window_sets[reference] for n in names)

    return {
        "models": names,
        "test_groups_match": groups_match,
        "test_windows_match": windows_match,
        "test_group_counts": {n: len(group_sets[n]) for n in names},
        "test_window_counts": {n: len(window_sets[n]) for n in names},
    }
