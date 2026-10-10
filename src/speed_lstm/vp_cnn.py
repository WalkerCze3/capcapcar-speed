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

Training: scripts/train_vp_cnn.py, VP1 labels from the recording's annotated lane dividers and VP2
labels from its calibration (Brno results json). A few training cameras means few distinct VP
configurations, so training warps each crop with a random homography H (rotation, mild perspective)
and maps its labels with the same H: VPs are points, so the warped crop's VPs are exactly H applied
to the original ones. As in deep_vp, the warp is applied to a larger context crop and the crop is
then re-fitted to the warped vehicle box with jittered edges (refit_crop), so augmented crops are
framed like detector crops at test time; that refit is also what varies scale and position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

CROP_SIZE = 128
CROP_PAD = 0.15  # crop side = (1 + 2 pad) * the box's longer side
CONTEXT = 1.5    # training caches crops covering [-CONTEXT, CONTEXT] crop coordinates


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


def _pixel_to_crop(size: int, extent: float = 1.0) -> np.ndarray:
    """3x3 map from OpenCV pixel indices (pixel i's centre at i) of a `size` px image covering
    [-extent, extent] crop coordinates to those coordinates."""
    k = 2.0 * extent / size
    return np.array([[k, 0.0, k / 2.0 - extent], [0.0, k, k / 2.0 - extent], [0.0, 0.0, 1.0]])


def random_homography(rng: np.random.Generator, max_rot_deg: float = 15.0, max_persp: float = 0.15) -> np.ndarray:
    """
    Random 3x3 homography in crop coordinates: rotation, then mild perspective. (Scale and shift are
    left out: refit_crop re-frames the crop around the warped box, which would undo them.)
    """
    a = np.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    px, py = rng.uniform(-max_persp, max_persp, 2)
    A = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [px, py, 1.0]]) @ A


def refit_crop(H: np.ndarray, box_hw, rng: np.random.Generator | None = None, jitter: float = 0.04,
               pad: float = CROP_PAD) -> np.ndarray:
    """
    Crop-coordinate map from the original crop to a crop re-fitted around the vehicle box after
    warping with H: the box (half-sizes `box_hw` = (w, h) / crop side) is warped, its axis-aligned
    extent taken, each edge jittered by up to `jitter` of the box size, and padded like CropGeom.from_box.
    """
    hw, hh = (float(v) for v in box_hw)
    corners = np.array([[-hw, -hh, 1.0], [hw, -hh, 1.0], [hw, hh, 1.0], [-hw, hh, 1.0]]) @ H.T
    xy = corners[:, :2] / corners[:, 2:]
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    if rng is not None and jitter > 0:
        dx0, dx1 = rng.uniform(-jitter, jitter, 2) * (x1 - x0)
        dy0, dy1 = rng.uniform(-jitter, jitter, 2) * (y1 - y0)
        x0, x1, y0, y1 = x0 + dx0, x1 + dx1, y0 + dy0, y1 + dy1
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * pad)
    R = np.array([[2.0 / side, 0.0, -2.0 * cx / side], [0.0, 2.0 / side, -2.0 * cy / side], [0.0, 0.0, 1.0]])
    return R @ H


def within_context(G: np.ndarray, extent: float = CONTEXT) -> bool:
    """Whether the crop seen through G only uses pixels of a context crop covering [-extent, extent]."""
    corners = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]) @ np.linalg.inv(G).T
    if np.any(corners[:, 2] <= 0):
        return False
    return bool(np.all(np.abs(corners[:, :2] / corners[:, 2:]) <= extent))


def render_crop(context_crop: np.ndarray, G: np.ndarray | None = None, size: int = CROP_SIZE,
                extent: float = CONTEXT) -> np.ndarray:
    """
    The (size, size) crop that sees `context_crop` (covering [-extent, extent] crop coordinates) through
    the crop-coordinate map G (identity: the plain centre crop). Pixels from outside the context repeat its edge.
    """
    import cv2

    T_out = _pixel_to_crop(size)
    T_in = _pixel_to_crop(context_crop.shape[0], extent)
    G = np.eye(3) if G is None else G
    M = np.linalg.inv(T_in) @ np.linalg.inv(G) @ T_out  # output pixel -> context pixel
    return cv2.warpPerspective(context_crop, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def warp_crop(crop: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply crop-coordinate homography H to a (S, S, 3) crop; uncovered pixels are black."""
    import cv2

    size = crop.shape[0]
    T = _pixel_to_crop(size)
    M = np.linalg.inv(T) @ H @ T
    return cv2.warpPerspective(crop, M, (size, size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def transform_dirs(d: np.ndarray, H: np.ndarray) -> np.ndarray:
    """VP directions (..., 3) in crop coordinates -> the same VPs after warping the crop with H."""
    out = np.asarray(d, dtype=np.float64) @ H.T
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


# ----------------------------------------------------------------- model

def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


def _head(c_in: int) -> nn.Sequential:
    """Keeps a 4x4 spatial layout: where the edges converge says how far away the VP is."""
    return nn.Sequential(nn.Conv2d(c_in, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                         nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64 * 16, 256),
                         nn.ReLU(inplace=True), nn.Linear(256, 6))


class VPNet(nn.Module):
    """
    [batch, 3, S, S] crops in [0, 1] -> [batch, 2, 3] unit directions (VP1 along, VP2 across).
    arch "small": a from-scratch conv net `width` channels wide; "resnet18": torchvision's ResNet-18
    trunk (ImageNet weights when `pretrained`).
    """

    def __init__(self, width: int = 32, arch: str = "small", pretrained: bool = False):
        super().__init__()
        self.arch = arch
        if arch == "small":
            chans = [3, width, 2 * width, 4 * width, 8 * width, 8 * width]
            self.features = nn.Sequential(*[_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
            c_out = chans[-1]
        elif arch == "resnet18":
            import torchvision

            net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = nn.Sequential(*list(net.children())[:-2])
            c_out = 512
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.head = _head(c_out)
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
        self.model = VPNet(ckpt.get("width", 32), ckpt.get("arch", "small"))
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
                      min_side: float = 24.0, border_margin: float = 3.0, min_travel_px: float = 40.0,
                      per_track: int = 20, seed: int = 0):
    """
    Unclipped vehicle boxes at least `min_side` px, every `every`-th frame of a track and at most
    `per_track` per track, from tracks whose bottom-center moved at least `min_travel_px` (a parked
    vehicle need not point along the road and would otherwise fill the sample); at most `max_crops`.
    """
    import pandas as pd

    from speed_lstm.video import _drop_truncated

    d = _drop_truncated(dets, img_size[0], img_size[1], border_margin)
    d = d[np.maximum(d["xmax"] - d["xmin"], d["ymax"] - d["ymin"]) >= min_side]
    d = d.sort_values(["track_id", "frame"])
    g = pd.DataFrame({"track_id": d["track_id"], "u": (d["xmin"] + d["xmax"]) / 2.0, "v": d["ymax"]}).groupby("track_id")
    travel = np.hypot(g["u"].max() - g["u"].min(), g["v"].max() - g["v"].min())
    d = d[d["track_id"].isin(travel.index[travel >= min_travel_px])]
    d = d[d.groupby("track_id").cumcount() % every == 0]
    picks = [t.iloc[np.unique(np.linspace(0, len(t) - 1, min(per_track, len(t))).round().astype(int))]
             for _, t in d.groupby("track_id", sort=False)]
    d = pd.concat(picks) if picks else d.iloc[:0]
    if len(d) > max_crops:
        d = d.sample(max_crops, random_state=seed)
    return d.sort_values("frame")


def decoded_frame(frame, frame_step: int):
    """
    Decoded video frame of a detect_and_track `frame`: Ultralytics' vid_stride grabs `frame_step`
    frames and keeps the last, so processed frame k is decoded frame k * step + step - 1.
    """
    return frame * frame_step + frame_step - 1


def extract_crops(video_path, boxes, frame_step: int = 1, size: int = CROP_SIZE, pad: float = CROP_PAD,
                  context: float = 1.0):
    """
    Read the frames `boxes` (detect_and_track rows) sit on and crop each box.
    `frame` counts processed frames (see decoded_frame for the video frame it is).
    With context > 1 each crop covers [-context, context] crop coordinates at the same resolution
    (round(size * context) px; render_crop cuts the plain crop back out).
    Returns (crops (N, S, S, 3) uint8, geoms (of the plain crops), the rows kept, in that order).
    """
    import cv2
    import pandas as pd

    by_frame = {int(decoded_frame(f, frame_step)): g for f, g in boxes.groupby("frame")}
    wanted = sorted(by_frame)
    crops, geoms, kept = [], [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        idx, target_i = 0, 0
        while target_i < len(wanted):
            if not cap.grab():
                break
            if idx == wanted[target_i]:
                ok, frame = cap.retrieve()
                if ok:
                    for _, r in by_frame[idx].iterrows():
                        g = CropGeom.from_box((r["xmin"], r["ymin"], r["xmax"], r["ymax"]), pad)
                        out = int(round(size * context))
                        crops.append(crop_image(frame, CropGeom(g.cx, g.cy, g.side * out / size), out))
                        geoms.append(g)
                        kept.append(r)
                target_i += 1
            idx += 1
    finally:
        cap.release()
    if target_i < len(wanted):
        print(f"[crops] warning: video ended at frame {idx}; {len(wanted) - target_i} of {len(wanted)} wanted frames missing")
    n = int(round(size * context))
    crops_arr = np.stack(crops) if crops else np.zeros((0, n, n, 3), dtype=np.uint8)
    return crops_arr, geoms, pd.DataFrame(kept)
