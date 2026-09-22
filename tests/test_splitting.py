from dataclasses import dataclass

import numpy as np

from speed_lstm.splitting import speed_balanced_split


@dataclass
class FakeWindow:
    scene: str
    vehicle_id: int
    target_speed: float


def make_population(n_vehicles=300, windows_per_vehicle=20, seed=0):
    rng = np.random.default_rng(seed)
    windows = []
    for vid in range(n_vehicles):
        # a handful of vehicles get a rare high speed, most are mid-range
        speed = rng.uniform(45, 50) if vid % 25 == 0 else rng.uniform(15, 30)
        speed_mps = speed * 0.44704  # mph -> m/s, doesn't matter for the test, just a scale
        for _ in range(windows_per_vehicle):
            windows.append(FakeWindow("scene1", vid, speed_mps + rng.normal(0, 0.3)))
    return windows


def test_no_vehicle_group_split_across_partitions():
    windows = make_population()
    split = speed_balanced_split(windows, seed=42)
    seen = {}
    for part, ws in split.items():
        for w in ws:
            key = (w.scene, w.vehicle_id)
            assert seen.get(key, part) == part, f"vehicle {key} appears in multiple partitions"
            seen[key] = part


def test_all_windows_accounted_for():
    windows = make_population()
    split = speed_balanced_split(windows, seed=42)
    total = sum(len(ws) for ws in split.values())
    assert total == len(windows)


def test_reproducible_with_same_seed():
    windows = make_population()
    split_a = speed_balanced_split(windows, seed=42)
    split_b = speed_balanced_split(windows, seed=42)
    for part in split_a:
        keys_a = sorted((w.vehicle_id) for w in split_a[part])
        keys_b = sorted((w.vehicle_id) for w in split_b[part])
        assert keys_a == keys_b


def test_approximate_speed_balance():
    windows = make_population(n_vehicles=600, windows_per_vehicle=20)
    split = speed_balanced_split(windows, seed=42)
    fractions = {part: len(ws) / len(windows) for part, ws in split.items()}
    assert abs(fractions["train"] - 0.70) < 0.10
    assert abs(fractions["val"] - 0.15) < 0.08
    assert abs(fractions["test"] - 0.15) < 0.08


def test_every_partition_nonempty_even_with_few_groups():
    windows = make_population(n_vehicles=3, windows_per_vehicle=5)
    split = speed_balanced_split(windows, seed=42)
    assert all(len(ws) > 0 for ws in split.values())
