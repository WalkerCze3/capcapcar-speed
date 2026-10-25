"""
Thin adapter around the authors' code (gajdosech2/Vehicle-Speed-Estimation-YOLOv6-3D,
Macko, Gajdošech & Kocur 2025, the YOLOv6 version of Kocur & Ftáčnik 2020).
Nothing of the method is reimplemented here: the perspective warp, the
detector and the 2D box + c -> 3D box construction are the authors' functions.
Two compatibility shims only:
  - torch >= 2.6 loads checkpoints with weights_only=True; theirs are pickled models.
  - numpy >= 2 rejects np.cross on 2-vectors, which their is_right() uses.
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def import_kocur(repo: str):
    """Import the authors' modules from a clone of their repo."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import transform_3D_utils.utils as KU
    from transform_3D_utils.radar import Radar
    from yolov6.core.inferer import Inferer

    def is_right(line_1, line_2, point):  # same test as theirs, 2D cross written out
        a = np.asarray(point, float) - np.asarray(line_1, float)
        b = np.asarray(line_2, float) - np.asarray(line_1, float)
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0] < 0

    KU.is_right = is_right
    return KU, Radar, Inferer


def ground_corner_pairs(corners_img: np.ndarray, M: np.ndarray, vp3: np.ndarray) -> list[tuple[int, int]]:
    """
    The 4 vertical edges of a Kocur box as (bottom, top) corner indices.

    The warp M sends VP3 lines (verticals) to image columns, so vertical
    edges are the corner pairs with the same warped x. Which end is the
    bottom depends on how the warp is oriented for a camera, so it is taken
    from the original image instead: for a camera looking down, the ground
    end of a vertical edge is the one nearer VP3 (the nadir). This does not
    depend on the corner order of the six cases in Radar.get_3d_bbox.
    """
    import cv2

    w = cv2.perspectiveTransform(corners_img.reshape(-1, 1, 2).astype(np.float32), M)[:, 0, :]
    near = np.linalg.norm(corners_img - np.asarray(vp3)[None, :], axis=1)
    pairs = sorted((abs(w[i, 0] - w[j, 0]), i, j) for i in range(8) for j in range(i + 1, 8))
    used, edges = set(), []
    for _, i, j in pairs:
        if i in used or j in used:
            continue
        used |= {i, j}
        edges.append((i, j) if near[i] < near[j] else (j, i))
        if len(edges) == 4:
            break
    return edges
