"""
Speed-balanced train/val/test split.

The indivisible allocation unit is (scene, vehicle id) — never an individual
window. All cameras and all overlapping windows of a vehicle stay together,
because splitting windows independently would let near-duplicate sequences
from the same vehicle appear in both training and testing.

Algorithm: build a per-(scene, vehicle) histogram of windows by 5 m/s speed
bin, process groups touching the rarest bins first (seeded shuffle breaks
ties within a rarity tier), and greedily assign each group to whichever
partition (train/val/test, target 70/15/15 per bin) yields the smallest
increase in normalized squared deviation from that partition's target
per-bin counts. A safeguard moves one group into any partition that ends up
empty. Exact target ratios aren't always reachable, especially for rare
speed bins, because whole vehicle groups can't be split.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

BIN_WIDTH = 5.0  # m/s
TARGET_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _speed_bin(speed: float) -> int:
    return int(speed // BIN_WIDTH)


def group_windows_by_vehicle(windows: list) -> dict[tuple[str, int], list]:
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for w in windows:
        groups[(w.scene, w.vehicle_id)].append(w)
    return dict(groups)


def _bin_histogram(group_windows: list) -> dict[int, int]:
    hist: dict[int, int] = defaultdict(int)
    for w in group_windows:
        hist[_speed_bin(w.target_speed)] += 1
    return dict(hist)


def speed_balanced_split(windows: list, seed: int = 42) -> dict[str, list]:
    """Returns {'train': [Window, ...], 'val': [...], 'test': [...]}."""
    groups = group_windows_by_vehicle(windows)
    group_keys = list(groups.keys())
    group_hist = {k: _bin_histogram(groups[k]) for k in group_keys}

    all_bins = sorted({b for h in group_hist.values() for b in h})
    total_per_bin: dict[int, int] = defaultdict(int)
    for h in group_hist.values():
        for b, c in h.items():
            total_per_bin[b] += c

    target_count = {
        part: {b: total_per_bin[b] * frac for b in all_bins}
        for part, frac in TARGET_FRACTIONS.items()
    }
    current_count = {part: defaultdict(int) for part in TARGET_FRACTIONS}

    bin_rarity = sorted(all_bins, key=lambda b: total_per_bin[b])
    rarity_rank = {b: i for i, b in enumerate(bin_rarity)}

    def group_priority(key) -> int:
        return min(rarity_rank[b] for b in group_hist[key])

    rng = np.random.default_rng(seed)
    order = list(group_keys)
    rng.shuffle(order)
    order.sort(key=group_priority)

    assignment: dict[tuple[str, int], str] = {}
    for key in order:
        h = group_hist[key]
        best_part, best_cost = None, None
        for part in TARGET_FRACTIONS:
            # Cost = how "full" (relative to this partition's own target) each
            # bin would be after adding this group, weighted by how many of
            # the group's windows land in that bin, summed across bins. This
            # ratio is scale-invariant across partitions with very different
            # absolute target sizes — unlike squared absolute deviation
            # normalized per-partition, which made train's huge target look
            # permanently "not urgent" and let val/test hoard rare-bin groups.
            cost = 0.0
            for b, c in h.items():
                new_count = current_count[part][b] + c
                denom = max(target_count[part][b], 1.0)
                cost += c * (new_count / denom)
            if best_cost is None or cost < best_cost:
                best_cost, best_part = cost, part
        assignment[key] = best_part
        for b, c in h.items():
            current_count[best_part][b] += c

    _ensure_every_partition_nonempty(assignment, group_hist)

    result: dict[str, list] = {part: [] for part in TARGET_FRACTIONS}
    for key, part in assignment.items():
        result[part].extend(groups[key])
    return result


def _ensure_every_partition_nonempty(assignment: dict, group_hist: dict) -> None:
    present = set(assignment.values())
    missing = set(TARGET_FRACTIONS) - present
    for part in missing:
        counts = defaultdict(int)
        for v in assignment.values():
            counts[v] += 1
        donor_part = max(counts, key=lambda p: counts[p])
        donor_keys = [k for k, v in assignment.items() if v == donor_part]
        if len(donor_keys) > 1:
            move_key = max(donor_keys, key=lambda k: sum(group_hist[k].values()))
            assignment[move_key] = part
