"""
CNN that predicts a vehicle's two vanishing points from an image crop of it.

On a straight road every car's own vanishing points are the road's: VP1 along the direction of
travel, VP2 across it. Following Kocur and Ftacnik, "Traffic Camera Calibration via Vehicle
Vanishing Point Detection" (2021, https://github.com/kocurvik/deep_vp), the network sees one
vehicle at a time and many per-vehicle predictions are aggregated into the road's VPs
(speed_lstm.autocalib). Instead of their diamond-space heatmaps, each VP is regressed as a unit
direction on the Gaussian sphere in crop coordinates, which is bounded for VPs at infinity as well,
with a sign-invariant loss (a direction and its antipode are the same VP).

Crop coordinates: a square crop of side `s` centred on the box center (cx, cy); a pixel (u, v) is
((u - cx) / (s/2), (v - cy) / (s/2)), so the crop spans [-1, 1]. A homogeneous image VP (x, y, w)
in those coordinates is ((x - cx w) / (s/2), (y - cy w) / (s/2), w), normalized to unit length.

Training: scripts/train_vp_cnn.py, labels from a recording's calibration (Brno results json).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

CROP_SIZE = 128
CROP_PAD = 0.15  # crop side = (1 + 2 pad) * the box's longer side


# -------------------------------------------------------------- geometry

@dataclass
class CropGeom:
    cx: float
    cy: float
    side: float

    @classmethod
    def from_box(cls, box, pad: float = CROP_PAD) -> "CropGeom":
        x1, y1, x2, y2 = (float(v) for v in box)
        return cls((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1) * (1.0 + 2.0 * pad))


def vp_to_crop_dir(vp_h, g: CropGeom) -> np.ndarray:
    """Homogeneous image VP -> unit direction in crop coordinates."""
    x, y, w = (float(v) for v in vp_h)
    half = g.side / 2.0
    d = np.array([(x - g.cx * w) / half, (y - g.cy * w) / half, w])
    return d / np.linalg.norm(d)


def crop_dir_to_vp(d, g: CropGeom) -> np.ndarray:
    """Unit direction in crop coordinates -> homogeneous image VP."""
    dx, dy, dw = (float(v) for v in d)
    half = g.side / 2.0
    return np.array([dx * half + g.cx * dw, dy * half + g.cy * dw, dw])


def flip_dir(d: np.ndarray) -> np.ndarray:
    """Direction label for the horizontally mirrored crop."""
    d = np.array(d, dtype=np.float64, copy=True)
    d[..., 0] *= -1.0
    return d


def crop_image(frame: np.ndarray, g: CropGeom, size: int = CROP_SIZE) -> np.ndarray:
    """(size, size, 3) uint8 crop; regions outside the frame are black."""
    import cv2

    s = g.side / size
    # Affine map from crop pixel (i, j) centres to frame coordinates.
    M = np.array([[s, 0.0, g.cx - g.side / 2.0 + s / 2.0], [0.0, s, g.cy - g.side / 2.0 + s / 2.0]])
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ----------------------------------------------------------------- model

def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class VPNet(nn.Module):
    """[batch, 3, S, S] crops in [0, 1] -> [batch, 2, 3] unit directions (VP1 along, VP2 across)."""

    def __init__(self, width: int = 32):
        super().__init__()
        chans = [3, width, 2 * width, 4 * width, 8 * width, 8 * width]
        self.features = nn.Sequential(*[_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(chans[-1], 128),
                                  nn.ReLU(inplace=True), nn.Linear(128, 6))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.features((x - self.mean) / self.std)).view(-1, 2, 3)
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of 1 - cos^2 between predicted and target directions: zero for the VP or its antipode."""
    cos = (pred * target).sum(dim=-1)
    return (1.0 - cos ** 2).mean()


def angle_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Sign-invariant angle between directions, degrees."""
    cos = np.abs(np.sum(np.asarray(pred) * np.asarray(target), axis=-1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def to_tensor(crops: np.ndarray) -> torch.Tensor:
    """(N, S, S, 3) uint8 BGR (OpenCV) -> (N, 3, S, S) float RGB in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(crops[..., ::-1])).permute(0, 3, 1, 2).float() / 255.0


# ------------------------------------------------------------- inference

class VPPredictor:
    def __init__(self, checkpoint: str, device: str | None = None):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.crop_size = int(ckpt.get("crop_size", CROP_SIZE))
        self.crop_pad = float(ckpt.get("crop_pad", CROP_PAD))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = VPNet(ckpt.get("width", 32))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_dirs(self, crops: np.ndarray, batch_size: int = 64, flip_tta: bool = True) -> np.ndarray:
        """(N, S, S, 3) crops -> (N, 2, 3) crop-coordinate directions; averages with the mirrored crop."""
        out = []
        for i in range(0, len(crops), batch_size):
            x = to_tensor(crops[i:i + batch_size]).to(self.device)
            d = self.model(x).cpu().numpy()
            if flip_tta:
                df = flip_dir(self.model(torch.flip(x, dims=[3])).cpu().numpy())
                df *= np.sign(np.sum(d * df, axis=-1, keepdims=True) + 1e-12)  # align antipodes before averaging
                d = d + df
                d /= np.linalg.norm(d, axis=-1, keepdims=True)
            out.append(d)
        return np.concatenate(out) if out else np.zeros((0, 2, 3))

    def predict_vps(self, crops: np.ndarray, geoms: list[CropGeom]) -> tuple[np.ndarray, np.ndarray]:
        """Homogeneous image VP1s and VP2s, (N, 3) each."""
        d = self.predict_dirs(crops)
        vp1 = np.array([crop_dir_to_vp(d[i, 0], g) for i, g in enumerate(geoms)]).reshape(-1, 3)
        vp2 = np.array([crop_dir_to_vp(d[i, 1], g) for i, g in enumerate(geoms)]).reshape(-1, 3)
        return vp1, vp2


# ----------------------------------------------------------- crop sampling

def sample_crop_boxes(dets, img_size: tuple[int, int], every: int = 10, max_crops: int = 2000,
                      min_side: float = 24.0, border_margin: float = 3.0, seed: int = 0):
    """Unclipped vehicle boxes at least `min_side` px, every `every`-th frame of a track, at most `max_crops`."""
    from speed_lstm.video import _drop_truncated

    d = _drop_truncated(dets, img_size[0], img_size[1], border_margin)
    d = d[np.maximum(d["xmax"] - d["xmin"], d["ymax"] - d["ymin"]) >= min_side]
    d = d.sort_values(["track_id", "frame"])
    d = d[d.groupby("track_id").cumcount() % every == 0]
    if len(d) > max_crops:
        d = d.sample(max_crops, random_state=seed)
    return d.sort_values("frame")


def extract_crops(video_path, boxes, frame_step: int = 1, size: int = CROP_SIZE, pad: float = CROP_PAD):
    """
    Read the frames `boxes` (detect_and_track rows) sit on and crop each box.
    `frame` counts processed frames, so the video frame is frame * frame_step (as in detect_and_track).
    Returns (crops (N, size, size, 3) uint8, geoms, the rows kept, in that order).
    """
    import cv2

    by_frame = {f: g for f, g in boxes.groupby("frame")}
    wanted = sorted(by_frame)
    crops, geoms, kept = [], [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        idx, target_i = 0, 0
        while target_i < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_step == 0 and idx // frame_step == wanted[target_i]:
                for _, r in by_frame[wanted[target_i]].iterrows():
                    g = CropGeom.from_box((r["xmin"], r["ymin"], r["xmax"], r["ymax"]), pad)
                    crops.append(crop_image(frame, g, size))
                    geoms.append(g)
                    kept.append(r)
                target_i += 1
            idx += 1
    finally:
        cap.release()
    import pandas as pd

    crops_arr = np.stack(crops) if crops else np.zeros((0, size, size, 3), dtype=np.uint8)
    return crops_arr, geoms, pd.DataFrame(kept)
