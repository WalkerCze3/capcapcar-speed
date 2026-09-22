"""Metrics and error-by-speed-range reporting for a trained checkpoint's predictions."""

from __future__ import annotations

import numpy as np

DEFAULT_SPEED_RANGES = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, float("inf"))]


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def speed_coverage(targets: np.ndarray, ranges: list[tuple[float, float]] = DEFAULT_SPEED_RANGES) -> dict:
    """Fraction and count of targets falling in each speed range (m/s)."""
    n = len(targets)
    coverage = {}
    for lo, hi in ranges:
        mask = (targets >= lo) & (targets < hi)
        count = int(mask.sum())
        coverage[f"[{lo},{hi})"] = {"count": count, "fraction": count / n if n else 0.0}
    return coverage


def errors_by_speed_range(pred: np.ndarray, target: np.ndarray,
                           ranges: list[tuple[float, float]] = DEFAULT_SPEED_RANGES) -> dict:
    """MAE/RMSE/count broken down by target speed range (m/s)."""
    report = {}
    for lo, hi in ranges:
        mask = (target >= lo) & (target < hi)
        count = int(mask.sum())
        if count == 0:
            report[f"[{lo},{hi})"] = {"count": 0, "mae": None, "rmse": None}
            continue
        report[f"[{lo},{hi})"] = {
            "count": count,
            "mae": mae(pred[mask], target[mask]),
            "rmse": rmse(pred[mask], target[mask]),
        }
    return report


def full_report(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "n": len(target),
        "overall_mae": mae(pred, target),
        "overall_rmse": rmse(pred, target),
        "speed_coverage": speed_coverage(target),
        "errors_by_speed_range": errors_by_speed_range(pred, target),
    }
